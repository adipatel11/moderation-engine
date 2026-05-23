"""Download the Civil Comments bias-eval set from Kaggle.

The canonical Civil Comments mirror on HuggingFace (`google/civil_comments`,
`civil_comments`) does NOT include the demographic-identity annotations —
those live in the Jigsaw "Unintended Bias in Toxicity Classification"
Kaggle competition, which is the source Borkan et al. 2019 used to define
the subgroup-disaggregated bias metrics.

We download the two test files that carry both toxicity labels and
identity annotations (`test_public_expanded.csv`,
`test_private_expanded.csv`, ~88 MB combined) and concatenate them into a
single parquet. Combined size is ~97k rows — same as the competition
leaderboard's evaluation surface.

Prerequisites:
  1. Kaggle account + API token at `~/.kaggle/kaggle.json` (chmod 600).
  2. Accept the competition rules in your browser at
     https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/rules

Output: `data/civil_comments/test.parquet` (gitignored).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "civil_comments"
COMPETITION = "jigsaw-unintended-bias-in-toxicity-classification"
FILES = ["test_public_expanded.csv", "test_private_expanded.csv"]
OUT_PATH = DATA_DIR / "test.parquet"

# Standard 9 subgroups from Borkan et al. 2019 + Jigsaw competition leaderboard.
SUBGROUPS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]


def download_one(name: str) -> Path:
    out_csv = DATA_DIR / name
    if out_csv.exists():
        print(f"  {name} already present, skipping")
        return out_csv
    print(f"  fetching {name}…")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", str(DATA_DIR)],
        check=True,
    )
    # Kaggle wraps single-file downloads in a .zip
    zip_path = DATA_DIR / f"{name}.zip"
    if zip_path.exists():
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(DATA_DIR)], check=True)
        zip_path.unlink()
    return out_csv


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} test files into {DATA_DIR}")

    try:
        paths = [download_one(name) for name in FILES]
    except FileNotFoundError:
        print(
            "kaggle CLI not found. Try `uv run python scripts/download_civil_comments.py`.",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print(f"\nkaggle download failed: {e}", file=sys.stderr)
        print(
            "If you see a 403, accept the competition rules at "
            f"https://www.kaggle.com/c/{COMPETITION}/rules",
            file=sys.stderr,
        )
        return 1

    print("\nMerging public + private test splits…")
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  combined rows: {len(df):,}")
    print(f"  columns: {df.columns.tolist()}")

    missing = [c for c in ["toxicity", *SUBGROUPS] if c not in df.columns]
    if missing:
        print(f"ERROR: missing expected columns: {missing}", file=sys.stderr)
        return 2

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    annotated = (df[SUBGROUPS] >= 0.5).any(axis=1)
    print("\nSanity check — subgroup membership counts (score >= 0.5):")
    print(f"  any of the 9 subgroups:  {int(annotated.sum()):>6,} / {len(df):,}")
    for sg in SUBGROUPS:
        mentioned = int((df[sg] >= 0.5).sum())
        toxic_in_sg = int(((df[sg] >= 0.5) & (df["toxicity"] >= 0.5)).sum())
        print(f"  {sg:38s} {mentioned:>6,}  (toxic: {toxic_in_sg:>5,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
