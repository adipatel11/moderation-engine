"""Render the batching window sweep chart (Phase 2 Opt 3).

Reads the per-window stats CSVs under docs/locust/batching/ and produces
a two-panel chart: throughput vs window on the left, p99 latency vs
window on the right. This is the artifact plan.txt calls "gold for your
writeup" — the headline visual for the Phase 4 blog post.

Output: docs/charts/batching_window_sweep.png

Run from repo root:
    uv run python scripts/render_batching_chart.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SWEEP_DIR = ROOT / "docs" / "locust" / "batching"
OUT_PATH = ROOT / "docs" / "charts" / "batching_window_sweep.png"

# Bucketed-batcher window sweep at 10-user closed-loop on c6i.large, 60 s per
# window. w=10 and w=20 were cut short by host instability under sustained
# load — kept in the plot but flagged in-figure.
WINDOWS = [0, 2, 5, 10, 20]
SHORTENED = {10, 20}


def load_stats(window_ms: int) -> dict[str, float]:
    csv = SWEEP_DIR / f"w{window_ms}_stats.csv"
    df = pd.read_csv(csv)
    row = df.loc[df["Name"] == "Aggregated"].iloc[0]
    return {
        "window_ms": window_ms,
        "throughput": float(row["Requests/s"]),
        "p50": float(row["50%"]),
        "p99": float(row["99%"]),
        "requests": int(row["Request Count"]),
    }


def main() -> None:
    rows = [load_stats(w) for w in WINDOWS]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, (ax_thru, ax_p99) = plt.subplots(1, 2, figsize=(11, 4.2))

    accent = "#1f77b4"
    win_color = "#2ca02c"  # green for the winning bypass point
    short_color = "#aaaaaa"

    def split(df: pd.DataFrame, key: str) -> tuple[list[int], list[float], list[float]]:
        full_x, full_y, short_x, short_y = [], [], [], []
        for _, r in df.iterrows():
            (short_x if int(r["window_ms"]) in SHORTENED else full_x).append(int(r["window_ms"]))
            (short_y if int(r["window_ms"]) in SHORTENED else full_y).append(float(r[key]))
        return full_x, full_y, short_x, short_y

    # Throughput panel
    fx, fy, sx, sy = split(df, "throughput")
    ax_thru.plot(WINDOWS, df["throughput"], color=accent, linewidth=2, zorder=1, alpha=0.5)
    ax_thru.scatter(fx, fy, color=accent, s=70, zorder=3, label="60s sample (full)")
    if sx:
        ax_thru.scatter(
            sx,
            sy,
            color=short_color,
            s=70,
            zorder=3,
            edgecolors=accent,
            linewidths=1.5,
            label="sample cut short (host instability)",
        )
    # Highlight bypass (the chosen production default — wins on throughput,
    # which is what saturates first on this 2-vCPU host; the p99 trade is
    # marginal in either direction).
    win_row = df.iloc[0]
    ax_thru.scatter(
        [0],
        [win_row["throughput"]],
        color=win_color,
        s=140,
        marker="*",
        zorder=4,
        label="production default (window=0)",
    )
    ax_thru.set_xlabel("Batching window (ms)")
    ax_thru.set_ylabel("Throughput (req/s)")
    ax_thru.set_title("Throughput")
    ax_thru.grid(True, linestyle="--", alpha=0.3)
    ax_thru.set_xticks(WINDOWS)
    ax_thru.set_ylim(bottom=0)
    ax_thru.legend(loc="upper right", fontsize=9, frameon=False)

    # p99 panel
    fx, fy, sx, sy = split(df, "p99")
    ax_p99.plot(WINDOWS, df["p99"], color=accent, linewidth=2, zorder=1, alpha=0.5)
    ax_p99.scatter(fx, fy, color=accent, s=70, zorder=3)
    if sx:
        ax_p99.scatter(sx, sy, color=short_color, s=70, zorder=3, edgecolors=accent, linewidths=1.5)
    ax_p99.scatter([0], [win_row["p99"]], color=win_color, s=140, marker="*", zorder=4)
    ax_p99.set_xlabel("Batching window (ms)")
    ax_p99.set_ylabel("p99 latency (ms)")
    ax_p99.set_title("p99 latency")
    ax_p99.grid(True, linestyle="--", alpha=0.3)
    ax_p99.set_xticks(WINDOWS)
    ax_p99.set_ylim(bottom=0)

    fig.suptitle(
        "Dynamic batching window sweep — c6i.large (2 vCPU), 10-user closed loop, INT8 ONNX",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
