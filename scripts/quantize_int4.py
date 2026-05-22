"""4-bit weight quantize the exported ONNX model (bitsandbytes-style).

Reads `models/onnx-toxic-bert/model.onnx` (the Opt 1 fp32 export) and
writes `models/onnx-toxic-bert-int4-<nf4|fp4>/model.onnx` using
`onnxruntime.quantization.matmul_bnb4_quantizer.MatMulBnb4Quantizer`.

This is the "thing that didn't work" deliverable per `plan.txt` —
deliberate half-day attempt at aggressive INT4 quantization (the
plan literally names "bitsandbytes INT4" as the example). The
quantizer packs MatMul constant weights into 4-bit blocks using
either the NF4 (NormalFloat, learned from a normal distribution
prior — generally better for transformer weights) or FP4 (4-bit
float with 3-bit bias) encoding, and rewrites the MatMul ops into
the contrib `com.microsoft::MatMulBnb4` op.

Measured outcome (see `docs/benchmarks.md` "Long-shot — INT4 NF4"
for the full writeup):

- The CPU `MatMulBnb4` kernel exists in ORT 1.26, so the model
  loads and produces correct logits — no early failure.
- Accuracy is *better* than INT8 dynamic quant against the FP32
  baseline (0.158% vs 0.505% flip rate over 383,868 binary
  decisions), because NF4's nonlinear 4-bit code matches
  transformer weight distributions better than INT8's uniform grid.
- Latency regresses 3-11x on CPU because there is no INT4 SIMD on
  x86 VNNI or M-series SDOT; the kernel falls back to
  dequant-then-FP32-matmul.
- Disk gets 33% *larger* than INT8 because `MatMulBnb4Quantizer`
  only packs `MatMul` weights; embeddings (~92 MB) stay FP32, but
  `quantize_dynamic(QInt8)` covers them too.

Block size trades accuracy against compression. The bitsandbytes
default is 64; smaller blocks recover more accuracy at the cost
of more scale storage. Default 64 here matches the upstream
reference.

Run from the repo root:
    uv run python scripts/quantize_int4.py                  # NF4, block=64
    uv run python scripts/quantize_int4.py --quant-type fp4
    uv run python scripts/quantize_int4.py --block-size 32
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "models" / "onnx-toxic-bert"

SIDECAR_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: models/onnx-toxic-bert-int4-<type>"
    )
    parser.add_argument(
        "--quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="NF4 (NormalFloat, transformer-friendly) or FP4 (4-bit float)",
    )
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    src_model = args.src / "model.onnx"
    if not src_model.exists():
        print(f"FATAL: {src_model.relative_to(ROOT)} not found — run scripts/export_onnx.py first")
        return 2

    out_dir = args.out or ROOT / "models" / f"onnx-toxic-bert-int4-{args.quant_type}"
    out_model = out_dir / "model.onnx"
    if out_model.exists() and not args.force:
        print(f"{out_model.relative_to(ROOT)} already exists; pass --force to re-quantize")
        return 0
    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import onnx
    from onnxruntime.quantization.matmul_bnb4_quantizer import MatMulBnb4Quantizer

    quant_type_const = (
        MatMulBnb4Quantizer.NF4 if args.quant_type == "nf4" else MatMulBnb4Quantizer.FP4
    )
    print(
        f"quantizing {src_model.relative_to(ROOT)} -> {out_model.relative_to(ROOT)}\n"
        f"  quant_type={args.quant_type.upper()}  block_size={args.block_size}"
    )

    fp32_model = onnx.load(str(src_model))
    quantizer = MatMulBnb4Quantizer(
        model=fp32_model,
        quant_type=quant_type_const,
        block_size=args.block_size,
    )
    quantizer.process()
    # MatMulBnb4Quantizer wraps the ModelProto in an internal ONNXModel
    # wrapper; the underlying proto is reachable as .model.model.
    quantized_proto = quantizer.model.model
    onnx.save_model(
        quantized_proto,
        str(out_model),
        save_as_external_data=False,
    )

    for name in SIDECAR_FILES:
        src = args.src / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
            print(f"  copied {name}")
        else:
            print(f"  skip   {name} (not in source)")

    src_bytes = sum(p.stat().st_size for p in args.src.rglob("*") if p.is_file())
    out_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(
        f"\ndone: {src_bytes / 1e6:.1f} MB (fp32) -> {out_bytes / 1e6:.1f} MB (int4-{args.quant_type})"
        f"  ({100 * out_bytes / src_bytes:.0f}% of original)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
