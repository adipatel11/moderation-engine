"""Aggregate per-concurrency locust CSVs into a single markdown table.

Reads `docs/locust/u<N>_stats.csv` files produced by `run_locust_sweep.sh`,
extracts the aggregated POST /classify row, and emits a table to
`docs/locust_summary.md` matching the format used in `docs/benchmarks.md`.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCUST_DIR = ROOT / "docs" / "locust"
OUT = ROOT / "docs" / "locust_summary.md"


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
                    "fail_rps": float(row["Failures/s"]),
                }
    return None


def main() -> None:
    rows: list[tuple[int, dict[str, float]]] = []
    for path in sorted(LOCUST_DIR.glob("u*_stats.csv")):
        m = re.match(r"u(\d+)_stats\.csv$", path.name)
        if not m:
            continue
        stats = parse_one(path)
        if stats is None:
            print(f"warn: no POST /classify row in {path.name}")
            continue
        rows.append((int(m.group(1)), stats))
    rows.sort(key=lambda kv: kv[0])

    lines = [
        "# Baseline latency / throughput (locust sweep)",
        "",
        "Closed-loop concurrency sweep against the EC2-deployed container "
        "(see `docs/benchmarks.md` for the locked protocol). Each level runs "
        "for 60 s with zero think time; samples drawn from a 1000-row, "
        "seed-42 stratified mix of the Jigsaw scored test split.",
        "",
        "| Users | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) "
        "| Throughput (req/s) | Error rate |",
        "|------:|---------:|---------:|---------:|---------:|---------:"
        "|-------------------:|-----------:|",
    ]
    for users, s in rows:
        err = (s["failures"] / s["requests"] * 100.0) if s["requests"] else 0.0
        lines.append(
            f"| {users} | {s['requests']:,} | {s['failures']:,} "
            f"| {s['p50']:.1f} | {s['p95']:.1f} | {s['p99']:.1f} "
            f"| {s['rps']:.1f} | {err:.2f}% |"
        )
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
