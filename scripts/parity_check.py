"""Compare ONNX-backed predictions against the PyTorch baseline.

Loads both backends, runs them on the same N rows of the Jigsaw scored test
split, and reports per-label max / mean absolute probability differences.
Exits non-zero if any difference exceeds `--tolerance` (default 1e-4) so the
script doubles as a CI gate before merging an optimization branch.

Run from the repo root:
    uv run python scripts/parity_check.py             # default 200 rows
    uv run python scripts/parity_check.py --n 1000    # tighter check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moderation_engine.backends.onnx import ONNXToxicityClassifier
from moderation_engine.backends.pytorch import PyTorchToxicityClassifier

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "jigsaw" / "test.csv"
LABELS_CSV = ROOT / "data" / "jigsaw" / "test_labels.csv"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_NAME = "unitary/toxic-bert"
ONNX_DIR = ROOT / "models" / "onnx-toxic-bert"


def load_sample(n: int, seed: int) -> list[str]:
    test = pd.read_csv(TEST_CSV)
    labels = pd.read_csv(LABELS_CSV)
    merged = test.merge(labels, on="id", how="inner")
    scored = merged[(merged[LABEL_COLS] != -1).all(axis=1)].reset_index(drop=True)
    sample = scored.sample(n=min(n, len(scored)), random_state=seed)
    return [t for t in sample["comment_text"].fillna("").tolist() if t.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    texts = load_sample(args.n, args.seed)
    print(f"comparing on {len(texts)} samples (seed={args.seed}, tol={args.tolerance:.0e})")

    print("loading PyTorch backend…")
    pt = PyTorchToxicityClassifier(MODEL_NAME)
    print("loading ONNX backend…")
    ox = ONNXToxicityClassifier(ONNX_DIR, MODEL_NAME)

    if pt.labels != ox.labels:
        raise RuntimeError(f"label order mismatch: pt={pt.labels} onnx={ox.labels}")

    pt_probs = np.array([list(pt.predict(t).values()) for t in texts])
    ox_probs = np.array([list(ox.predict(t).values()) for t in texts])
    diff = np.abs(pt_probs - ox_probs)

    print("\nper-label diff (probability units):")
    print(f"{'label':<16} {'max':>10} {'mean':>10} {'p99':>10} {'>tol':>8}")
    over_tol_total = 0
    for i, label in enumerate(pt.labels):
        col = diff[:, i]
        over = int((col > args.tolerance).sum())
        over_tol_total += over
        print(
            f"{label:<16} {col.max():>10.2e} {col.mean():>10.2e} "
            f"{np.percentile(col, 99):>10.2e} {over:>8}"
        )

    overall_max = float(diff.max())
    print(f"\noverall max abs diff = {overall_max:.2e} (tol = {args.tolerance:.0e})")
    if overall_max > args.tolerance:
        print(f"FAIL: {over_tol_total} (sample, label) pairs exceed tolerance")
        return 1
    print("PASS: ONNX matches PyTorch within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
