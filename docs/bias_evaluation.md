# Bias Evaluation

Production toxicity classifier (`unitary/toxic-bert`, INT8 ONNX) evaluated
for unintended demographic bias on the Civil Comments dataset.

## Methodology

- **Dataset:** Jigsaw "Unintended Bias in Toxicity Classification" test set
  (public + private, 42,870 identity-annotated rows out of
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

**Overall FPR across all annotated rows:** `1.39%`
(530 false positives / 38,111 non-toxic rows).

| Subgroup | n | Non-toxic | Toxic | FPR | Δ vs overall |
|----------|--:|----------:|------:|----:|-------------:|
| LGB | 1,065 | 775 | 290 | 3.61% | +2.22 pp |
| Mental illness | 511 | 405 | 106 | 3.21% | +1.82 pp |
| Female | 5,155 | 4,463 | 692 | 1.99% | +0.60 pp |
| Black | 1,519 | 1,016 | 503 | 1.77% | +0.38 pp |
| Male | 4,386 | 3,716 | 670 | 1.75% | +0.36 pp |
| Jewish | 835 | 697 | 138 | 1.72% | +0.33 pp |
| White | 2,452 | 1,710 | 742 | 1.64% | +0.25 pp |
| Muslim | 2,040 | 1,557 | 483 | 1.22% | -0.17 pp |
| Christian | 4,226 | 3,809 | 417 | 0.81% | -0.58 pp |

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

| Subgroup | Subgroup AUC | BPSN AUC | BNSP AUC |
|----------|-------------:|---------:|---------:|
| LGB | 0.7947 | 0.7832 | 0.9123 |
| Black | 0.8071 | 0.8018 | 0.9125 |
| White | 0.8054 | 0.8213 | 0.9017 |
| Mental illness | 0.8695 | 0.8261 | 0.9303 |
| Jewish | 0.8493 | 0.8410 | 0.9103 |
| Female | 0.8738 | 0.8551 | 0.9193 |
| Male | 0.8732 | 0.8630 | 0.9129 |
| Muslim | 0.8238 | 0.8795 | 0.8683 |
| Christian | 0.8895 | 0.9146 | 0.8755 |

### Single-number summary (Jigsaw leaderboard formula)

For comparison with the published leaderboard, we also report the
generalized power-mean aggregate (p = -5, which weights heavily toward
the worst-performing subgroup):

| Component | Value |
|-----------|------:|
| Overall AUC | `0.9019` |
| Power-mean subgroup AUC | `0.8389` |
| Power-mean BPSN AUC | `0.8378` |
| Power-mean BNSP AUC | `0.9035` |
| **Jigsaw final bias score** | **`0.8705`** |

(Final score = `0.25·overall_AUC + 0.25·(power_means_sum)`. The 2019
competition's winning entry scored ≈0.947; the original unitary/toxic-bert
without bias-aware training lands in the same ballpark as the field's
mid-tier baselines.)

## Interpretation

The model is **not unbiased**, and the pattern is the one toxicity classifiers
are known for ([Borkan et al. 2019](https://arxiv.org/abs/1903.04561)):
over-flagging of identity mentions, particularly on identities frequently
discussed in adversarial contexts online.

- **Highest FPR:** LGB (3.61%),
  vs **overall FPR 1.39%** — a
  2.6x gap. Non-toxic comments mentioning
  this subgroup are flagged much more often than the dataset baseline.
- **Lowest FPR:** Christian (0.81%).
- **Lowest BPSN AUC:** LGB
  (0.7832). BPSN compares non-toxic in-subgroup
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
and all of them have their own tradeoffs.

## Reproducing this evaluation

```bash
uv run python scripts/download_civil_comments.py   # one-time, requires Kaggle access
uv run python scripts/eval_bias.py                 # ~10 min on M3 Pro
uv run python scripts/render_bias_writeup.py       # regenerates this file from the parquet
```

The 42,870-row identity-annotated subset is sufficient for stable
metrics: every subgroup table cell here is computed on at least
511 rows (mental illness,
the smallest subgroup).
