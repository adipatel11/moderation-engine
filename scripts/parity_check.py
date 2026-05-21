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
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=ONNX_DIR,
        help="ONNX export dir to compare against (default: %(default)s)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="print stats but always exit 0 (use for lossy backends like INT8)",
    )
    args = parser.parse_args()

    onnx_dir = args.onnx_dir.resolve()
    texts = load_sample(args.n, args.seed)
    label = "ONNX" if onnx_dir == ONNX_DIR.resolve() else onnx_dir.name
    print(
        f"comparing PyTorch vs {label} on {len(texts)} samples "
        f"(seed={args.seed}, tol={args.tolerance:.0e})"
    )

    print("loading PyTorch backend…")
    pt = PyTorchToxicityClassifier(MODEL_NAME)
    print(f"loading ONNX backend from {onnx_dir.relative_to(ROOT)}…")
    ox = ONNXToxicityClassifier(onnx_dir, MODEL_NAME)

    if pt.labels != ox.labels:
        raise RuntimeError(f"label order mismatch: pt={pt.labels} onnx={ox.labels}")

    pt_probs = np.array([list(pt.predict(t).values()) for t in texts])
    ox_probs = np.array([list(ox.predict(t).values()) for t in texts])
    diff = np.abs(pt_probs - ox_probs)

    # Label flip = the two backends would predict different binary labels
    # at the standard 0.5 threshold. More informative for lossy backends
    # than raw probability diff: probability shifts that don't cross 0.5
    # don't change the moderation decision.
    pt_pred = pt_probs >= 0.5
    ox_pred = ox_probs >= 0.5
    flips = pt_pred != ox_pred

    print("\nper-label diff (probability units) and label-flip rate at thr=0.5:")
    print(f"{'label':<16} {'max':>10} {'mean':>10} {'p99':>10} {'>tol':>8} {'flips':>7}")
    over_tol_total = 0
    flips_total = 0
    for i, lbl in enumerate(pt.labels):
        col = diff[:, i]
        over = int((col > args.tolerance).sum())
        flip = int(flips[:, i].sum())
        over_tol_total += over
        flips_total += flip
        print(
            f"{lbl:<16} {col.max():>10.2e} {col.mean():>10.2e} "
            f"{np.percentile(col, 99):>10.2e} {over:>8} {flip:>4}/{len(texts):<3}"
        )

    overall_max = float(diff.max())
    print(
        f"\noverall: max abs diff = {overall_max:.2e} (tol = {args.tolerance:.0e}); "
        f"label flips total = {flips_total} / {len(texts) * len(pt.labels)}"
    )
    if args.report_only:
        print("REPORT: lossy backend; tolerance gate disabled")
        return 0
    if overall_max > args.tolerance:
        print(f"FAIL: {over_tol_total} (sample, label) pairs exceed tolerance")
        return 1
    print("PASS: ONNX matches PyTorch within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
