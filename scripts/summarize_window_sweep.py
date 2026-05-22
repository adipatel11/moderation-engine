"""Aggregate per-window locust CSVs into a markdown table.

Reads `docs/locust/batching/w<MS>_stats.csv` files produced by
`scripts/run_window_sweep.sh` and emits a window-vs-latency-vs-throughput
table — the headline plot of Phase 2 Opt 3.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "docs" / "locust" / "batching"
OUT = ROOT / "docs" / "window_sweep_summary.md"


def parse_one(path: Path) -> dict[str, float] | None:
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("Name") == "POST /classify":
                return {
                    "requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                    "p50": float(row["50%"]),
                    "p95": float(row["95%"]),
                    "p99": float(row["99%"]),
                    "rps": float(row["Requests/s"]),
                }
    return None


def main() -> None:
    rows: list[tuple[int, dict[str, float]]] = []
    for path in sorted(DIR.glob("w*_stats.csv")):
        m = re.match(r"w(\d+)_stats\.csv$", path.name)
        if not m:
            continue
        stats = parse_one(path)
        if stats is None:
            print(f"warn: no POST /classify row in {path.name}")
            continue
        rows.append((int(m.group(1)), stats))
    rows.sort(key=lambda kv: kv[0])

    lines = [
        "# Dynamic-batching window sweep",
        "",
        "Closed-loop locust sweep against the EC2-deployed INT8 container at "
        "a fixed concurrency, varying `BATCHING_WINDOW_MS`. Each row is a "
        "separate 60 s run with the container restarted between windows. "
        "Same Jigsaw sample pool and seed as the standard concurrency sweep "
        "in `docs/benchmarks.md`.",
        "",
        "| Window (ms) | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) "
        "| Throughput (req/s) |",
        "|------------:|---------:|---------:|---------:|---------:|---------:"
        "|-------------------:|",
    ]
    for window, s in rows:
        lines.append(
            f"| {window} | {s['requests']:,} | {s['failures']:,} "
            f"| {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} "
            f"| {s['rps']:.1f} |"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
