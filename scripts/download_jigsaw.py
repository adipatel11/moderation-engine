"""Download the Jigsaw Toxic Comment Classification dataset from Kaggle.

Prerequisites:
  1. Kaggle account.
  2. Kaggle API token at `~/.kaggle/kaggle.json` (chmod 600).
     Get one at https://www.kaggle.com/settings -> "Create New Token".
  3. Accept the competition rules in your browser at
     https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules

The dataset is written to `data/jigsaw/` (gitignored).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "jigsaw"
COMPETITION = "jigsaw-toxic-comment-classification-challenge"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {COMPETITION} into {DATA_DIR}")
    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                COMPETITION,
                "-p",
                str(DATA_DIR),
            ],
            check=True,
        )
    except FileNotFoundError:
        print("kaggle CLI not found. Run `uv sync` and try again.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"kaggle download failed: {e}", file=sys.stderr)
        print(
            "Make sure ~/.kaggle/kaggle.json exists and you've accepted the "
            "competition rules at https://www.kaggle.com/c/"
            f"{COMPETITION}/rules",
            file=sys.stderr,
        )
        return 1

    # Unzip the bundle Kaggle gives us.
    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        print(f"Unzipping {zip_path}")
        subprocess.run(["unzip", "-o", str(zip_path), "-d", str(DATA_DIR)], check=True)
        zip_path.unlink()

    # Each CSV inside is itself zipped; unzip those too.
    for inner_zip in DATA_DIR.glob("*.zip"):
        print(f"Unzipping {inner_zip}")
        subprocess.run(["unzip", "-o", str(inner_zip), "-d", str(DATA_DIR)], check=True)
        inner_zip.unlink()

    print("\nFiles in data/jigsaw/:")
    for p in sorted(DATA_DIR.iterdir()):
        print(f"  {p.name:30s} {p.stat().st_size / 1024 / 1024:8.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
