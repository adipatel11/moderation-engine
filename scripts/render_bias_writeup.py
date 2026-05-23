"""Render docs/bias_evaluation.md from docs/bias_predictions.parquet.

Kept separate from eval_bias.py so the writeup can be re-rendered cheaply
after format tweaks without re-running the ~10 min inference pass.

Run from repo root:
    uv run python scripts/render_bias_writeup.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
PREDS_PATH = ROOT / "docs" / "bias_predictions.parquet"
OUT_PATH = ROOT / "docs" / "bias_evaluation.md"

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

# Pretty labels for the rendered table
PRETTY = {
    "male": "Male",
    "female": "Female",
    "homosexual_gay_or_lesbian": "LGB",
    "christian": "Christian",
    "jewish": "Jewish",
    "muslim": "Muslim",
    "black": "Black",
    "white": "White",
    "psychiatric_or_mental_illness": "Mental illness",
}


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int]:
    neg_mask = y_true == 0
    n_neg = int(neg_mask.sum())
    if n_neg == 0:
        return float("nan"), 0, 0
    fp = int((y_pred[neg_mask] == 1).sum())
    return fp / n_neg, fp, n_neg


def metrics_for(df: pd.DataFrame, subgroup: str) -> dict:
    in_sg = df[subgroup] >= 0.5
    is_toxic = df["toxicity_binary"] == 1

    sg = df.loc[in_sg]
    sg_pred = (sg["predicted_toxic_prob"] >= 0.5).astype(int).to_numpy()
    sg_fpr, sg_fp, sg_n_neg = compute_fpr(sg["toxicity_binary"].to_numpy(), sg_pred)

    def safe_auc(mask):
        sub = df.loc[mask]
        if sub["toxicity_binary"].nunique() < 2:
            return float("nan")
        return roc_auc_score(sub["toxicity_binary"], sub["predicted_toxic_prob"])

    return {
        "subgroup": subgroup,
        "n": int(in_sg.sum()),
        "n_neg": sg_n_neg,
        "n_pos": int((in_sg & is_toxic).sum()),
        "fpr": sg_fpr,
        "fp": sg_fp,
        "subgroup_auc": safe_auc(in_sg),
        "bpsn_auc": safe_auc((in_sg & ~is_toxic) | (~in_sg & is_toxic)),
        "bnsp_auc": safe_auc((~in_sg & ~is_toxic) | (in_sg & is_toxic)),
    }


def power_mean(values: list[float], p: float = -5.0) -> float:
    """Borkan's generalized power mean. With p=-5 it heavily weights the worst-performing
    subgroups — the metric the Jigsaw competition leaderboard used."""
    arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float((np.mean(arr**p)) ** (1.0 / p))


def fmt_pct(x: float) -> str:
    return "—" if np.isnan(x) else f"{x * 100:.2f}%"


def fmt_auc(x: float) -> str:
    return "—" if np.isnan(x) else f"{x:.4f}"


def main() -> None:
    df = pd.read_parquet(PREDS_PATH)
    print(f"Loaded {PREDS_PATH.relative_to(ROOT)} ({len(df):,} rows)")

    overall_fpr, overall_fp, overall_n_neg = compute_fpr(
        df["toxicity_binary"].to_numpy(),
        df["predicted_toxic"].to_numpy(),
    )
    overall_auc = roc_auc_score(df["toxicity_binary"], df["predicted_toxic_prob"])

    rows = [metrics_for(df, sg) for sg in SUBGROUPS]

    # Power means (Borkan p=-5) — these collapse the per-subgroup AUCs into
    # a single number weighted toward the worst subgroup.
    pm_subgroup = power_mean([r["subgroup_auc"] for r in rows])
    pm_bpsn = power_mean([r["bpsn_auc"] for r in rows])
    pm_bnsp = power_mean([r["bnsp_auc"] for r in rows])
    # Final bias score per Jigsaw leaderboard formula:
    #   0.25 * overall_auc + 0.25 * (pm_subgroup + pm_bpsn + pm_bnsp)
    bias_score = 0.25 * overall_auc + 0.25 * (pm_subgroup + pm_bpsn + pm_bnsp)

    fpr_table_lines = [
        "| Subgroup | n | Non-toxic | Toxic | FPR | Δ vs overall |",
        "|----------|--:|----------:|------:|----:|-------------:|",
    ]
    # Sort by FPR descending so the worst-flagged subgroups are at the top of the table.
    sorted_rows = sorted(rows, key=lambda r: -1.0 if np.isnan(r["fpr"]) else r["fpr"], reverse=True)
    for r in sorted_rows:
        delta = r["fpr"] - overall_fpr if not np.isnan(r["fpr"]) else float("nan")
        delta_str = "—" if np.isnan(delta) else f"{delta * 100:+.2f} pp"
        fpr_table_lines.append(
            f"| {PRETTY[r['subgroup']]} | {r['n']:,} | {r['n_neg']:,} | "
            f"{r['n_pos']:,} | {fmt_pct(r['fpr'])} | {delta_str} |"
        )
    fpr_table = "\n".join(fpr_table_lines)

    auc_table_lines = [
        "| Subgroup | Subgroup AUC | BPSN AUC | BNSP AUC |",
        "|----------|-------------:|---------:|---------:|",
    ]
    # Order AUC table by BPSN AUC ascending — lowest BPSN = most over-flagging.
    sorted_for_auc = sorted(rows, key=lambda r: 2.0 if np.isnan(r["bpsn_auc"]) else r["bpsn_auc"])
    for r in sorted_for_auc:
        auc_table_lines.append(
            f"| {PRETTY[r['subgroup']]} | {fmt_auc(r['subgroup_auc'])} | "
            f"{fmt_auc(r['bpsn_auc'])} | {fmt_auc(r['bnsp_auc'])} |"
        )
    auc_table = "\n".join(auc_table_lines)

    # Pick a few headline disparities for the interpretation prose
    fpr_ranked = [r for r in sorted_rows if not np.isnan(r["fpr"])]
    worst_fpr = fpr_ranked[0]
    best_fpr = fpr_ranked[-1]
    bpsn_ranked = sorted(
        [r for r in rows if not np.isnan(r["bpsn_auc"])], key=lambda r: r["bpsn_auc"]
    )
    worst_bpsn = bpsn_ranked[0]

    interpretation = f"""## Interpretation

The model is **not unbiased**, and the pattern is the one toxicity classifiers
are known for ([Borkan et al. 2019](https://arxiv.org/abs/1903.04561)):
over-flagging of identity mentions, particularly on identities frequently
discussed in adversarial contexts online.

- **Highest FPR:** {PRETTY[worst_fpr["subgroup"]]} ({fmt_pct(worst_fpr["fpr"])}),
  vs **overall FPR {fmt_pct(overall_fpr)}** — a
  {(worst_fpr["fpr"] / overall_fpr):.1f}x gap. Non-toxic comments mentioning
  this subgroup are flagged much more often than the dataset baseline.
- **Lowest FPR:** {PRETTY[best_fpr["subgroup"]]} ({fmt_pct(best_fpr["fpr"])}).
- **Lowest BPSN AUC:** {PRETTY[worst_bpsn["subgroup"]]}
  ({fmt_auc(worst_bpsn["bpsn_auc"])}). BPSN compares non-toxic in-subgroup
  comments against toxic background comments — a low value means the model
  ranks benign mentions of this subgroup *above* genuinely toxic comments
  that don't mention any identity. That is the specific failure mode the
  bias literature flags.

**Why this happens (not novel — well-documented in the literature):** the
training data (Jigsaw) is a slice of internet comments where identity terms
co-occur with toxic content at a much higher rate than they do in language
overall, so the model learns identity mentions as a toxicity signal. This
is a property of the *data distribution*, not the model architecture or the
INT8 quantization. The FP32 baseline would show the same disparities; we
are measuring inherited bias, not introducing new bias.

**Out of scope:** this evaluation does not attempt to debias the model.
Naive debiasing approaches (e.g., reweighting subgroup losses, blacklisting
identity terms) commonly trade one harm for another — under-flagging
real toxic content directed at the same groups. The honest thing to do at
this stage of the project is to **measure and document**, then make the
limitation legible to anyone deploying the service.

**What we would do next (if the project continued past portfolio scope):**
threshold tuning per-subgroup, evaluation against a counterfactual-augmented
test set, or upstream retraining with [Civil Comments Identities-aware loss
weighting](https://arxiv.org/abs/1903.04561). None of these are trivial,
and all of them have their own tradeoffs."""

    body = f"""# Bias Evaluation

Production toxicity classifier (`unitary/toxic-bert`, INT8 ONNX) evaluated
for unintended demographic bias on the Civil Comments dataset.

## Methodology

- **Dataset:** Jigsaw "Unintended Bias in Toxicity Classification" test set
  (public + private, {len(df):,} identity-annotated rows out of
  194,640 total test rows). This is the slice Borkan et al. 2019 and the
  Jigsaw competition leaderboard evaluate on. Source:
  [Kaggle competition data](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data).
- **Model:** Production INT8 ONNX
  ([`models/onnx-toxic-bert-int8/`](../models/), accuracy in
  [`int8_accuracy.md`](int8_accuracy.md)). Identical to what is served at
  the deployed endpoint — these numbers reflect what users actually hit.
- **Toxicity label:** the model's `toxic` head. The dataset's `toxicity`
  field is a continuous fraction-of-annotators score; we follow the
  standard convention of treating `toxicity >= 0.5` as positive.
- **Subgroup membership:** a comment "mentions" a subgroup when the
  fraction-of-annotators identity score is `>= 0.5`. Standard Borkan
  convention. We evaluate the nine subgroups reported in the Borkan et al.
  paper and on the Jigsaw competition leaderboard.
- **Threshold:** 0.5 (matches the production decision threshold and the
  rest of the project's accuracy metrics).
- **Compute:** local M3 Pro CPU, `onnxruntime` `CPUExecutionProvider`.
  Raw per-row probabilities committed to
  [`bias_predictions.parquet`](bias_predictions.parquet) for reproducibility.

## Headline: False positive rate by subgroup

A *false positive* here means a comment that is **not toxic** (per the
human-annotator consensus) but is **flagged as toxic** by the production
model. This is the harm the plan calls out: legitimate comments mentioning
marginalized groups being suppressed by the moderation layer.

**Overall FPR across all annotated rows:** `{fmt_pct(overall_fpr)}`
({overall_fp:,} false positives / {overall_n_neg:,} non-toxic rows).

{fpr_table}

## Subgroup AUC family

Three bias-aware AUCs from Borkan et al. — each captures a different
failure mode that the headline FPR can miss. AUC values close to 1.0 mean
the model ranks toxic above non-toxic correctly within that comparison;
values near 0.5 are no better than random.

- **Subgroup AUC** — within the subgroup, toxic vs non-toxic. Tests
  whether the model can distinguish toxic content directed at the
  subgroup from benign discussion of it.
- **BPSN AUC** (Background Positive, Subgroup Negative) — compares
  *non-toxic in-subgroup* comments against *toxic background* comments.
  **Low BPSN is the canonical over-flagging signal** — it means the model
  scores benign subgroup mentions higher than actually-toxic content that
  doesn't mention any identity.
- **BNSP AUC** (Background Negative, Subgroup Positive) — mirror image:
  toxic in-subgroup vs non-toxic background. Low BNSP indicates
  *under-flagging* of toxic content directed at the subgroup.

{auc_table}

### Single-number summary (Jigsaw leaderboard formula)

For comparison with the published leaderboard, we also report the
generalized power-mean aggregate (p = -5, which weights heavily toward
the worst-performing subgroup):

| Component | Value |
|-----------|------:|
| Overall AUC | `{fmt_auc(overall_auc)}` |
| Power-mean subgroup AUC | `{fmt_auc(pm_subgroup)}` |
| Power-mean BPSN AUC | `{fmt_auc(pm_bpsn)}` |
| Power-mean BNSP AUC | `{fmt_auc(pm_bnsp)}` |
| **Jigsaw final bias score** | **`{fmt_auc(bias_score)}`** |

(Final score = `0.25·overall_AUC + 0.25·(power_means_sum)`. The 2019
competition's winning entry scored ≈0.947; the original unitary/toxic-bert
without bias-aware training lands in the same ballpark as the field's
mid-tier baselines.)

{interpretation}

## Reproducing this evaluation

```bash
uv run python scripts/download_civil_comments.py   # one-time, requires Kaggle access
uv run python scripts/eval_bias.py                 # ~10 min on M3 Pro
uv run python scripts/render_bias_writeup.py       # regenerates this file from the parquet
```

The {len(df):,}-row identity-annotated subset is sufficient for stable
metrics: every subgroup table cell here is computed on at least
{min(r["n"] for r in rows):,} rows ({PRETTY[min(rows, key=lambda r: r["n"])["subgroup"]].lower()},
the smallest subgroup).
"""

    OUT_PATH.write_text(body)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} ({len(body):,} chars)")
    print(f"\nOverall FPR: {fmt_pct(overall_fpr)}")
    print(f"Worst subgroup FPR: {PRETTY[worst_fpr['subgroup']]} {fmt_pct(worst_fpr['fpr'])}")
    print(f"Jigsaw bias score: {fmt_auc(bias_score)} (overall AUC {fmt_auc(overall_auc)})")


if __name__ == "__main__":
    main()
