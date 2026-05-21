"""Export `unitary/toxic-bert` to ONNX via optimum.

Idempotent: if `models/onnx-toxic-bert/model.onnx` already exists, the script
exits 0 unless `--force` is passed.

Run from the repo root:
    uv run python scripts/export_onnx.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "unitary/toxic-bert"
DEFAULT_OUT = ROOT / "models" / "onnx-toxic-bert"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task", default="text-classification")
    parser.add_argument("--force", action="store_true", help="re-export even if model.onnx exists")
    args = parser.parse_args()

    onnx_file = args.out / "model.onnx"
    if onnx_file.exists() and not args.force:
        print(f"{onnx_file.relative_to(ROOT)} already exists; pass --force to re-export")
        return 0

    if args.out.exists() and args.force:
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        "--model",
        args.model,
        "--task",
        args.task,
        str(args.out),
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    bytes_total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"Export complete: {args.out.relative_to(ROOT)} ({bytes_total / 1e6:.1f} MB total)")
    for path in sorted(args.out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(args.out)}  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
