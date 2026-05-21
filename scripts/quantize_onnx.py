"""Dynamic-quantize the exported ONNX model to INT8.

Reads `models/onnx-toxic-bert/model.onnx` (the Opt 1 fp32 export) and
writes `models/onnx-toxic-bert-int8/model.onnx` (plus copies of the
tokenizer/config files so the runtime can load it standalone via
`ONNXToxicityClassifier`).

Dynamic quantization quantizes weights to INT8 ahead of time and
activations on the fly at inference — no calibration dataset needed,
unlike static quantization. The trade-off is some accuracy loss
(typically 1-3% F1 on per-label basis for BERT-class models).

Idempotent: skips if the int8 model already exists, unless `--force`.

Run from the repo root:
    uv run python scripts/quantize_onnx.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "models" / "onnx-toxic-bert"
DEFAULT_OUT = ROOT / "models" / "onnx-toxic-bert-int8"

# Tokenizer / config files that should travel with the quantized model so
# the runtime can load it standalone without falling back to the HF hub.
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--force", action="store_true", help="re-quantize even if model.onnx exists"
    )
    args = parser.parse_args()

    src_model = args.src / "model.onnx"
    if not src_model.exists():
        print(f"FATAL: {src_model.relative_to(ROOT)} not found — run scripts/export_onnx.py first")
        return 2

    out_model = args.out / "model.onnx"
    if out_model.exists() and not args.force:
        print(f"{out_model.relative_to(ROOT)} already exists; pass --force to re-quantize")
        return 0

    if args.out.exists() and args.force:
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    # Import here so the script can fail fast on missing files above without
    # paying the onnxruntime import cost.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"quantizing {src_model.relative_to(ROOT)} -> {out_model.relative_to(ROOT)}")
    quantize_dynamic(
        model_input=str(src_model),
        model_output=str(out_model),
        weight_type=QuantType.QInt8,
    )

    for name in SIDECAR_FILES:
        src = args.src / name
        if src.exists():
            shutil.copy2(src, args.out / name)
            print(f"  copied {name}")
        else:
            print(f"  skip   {name} (not in source)")

    src_bytes = sum(p.stat().st_size for p in args.src.rglob("*") if p.is_file())
    out_bytes = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(
        f"\ndone: {src_bytes / 1e6:.1f} MB (fp32) -> {out_bytes / 1e6:.1f} MB (int8)"
        f"  ({100 * out_bytes / src_bytes:.0f}% of original)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
