"""Phase 3 bias eval: identity-disaggregated FPR + subgroup-AUC family on Civil Comments.

Runs the production INT8 ONNX model over the Jigsaw "Unintended Bias"
identity-annotated test set, then computes:

  - Overall FPR (non-toxic comments flagged as toxic) at threshold 0.5
  - Per-subgroup FPR at threshold 0.5 (the headline plan.txt ask)
  - Subgroup AUC, BPSN AUC, BNSP AUC per subgroup (Borkan et al. 2019)

Filters to rows with non-NaN identity annotations (~43k of 194k) — the
population the bias literature evaluates on. The remaining 151k rows were
never sampled into the identity-annotation pool, so subgroup membership
is undefined there.

Outputs:
  - docs/bias_predictions.parquet — id, toxicity, identity scores,
    predicted_toxic_prob (raw probabilities, committed for reproducibility)
  - prints metrics tables; rendering into bias_evaluation.md is a separate step

Run from the repo root:
    uv run python scripts/eval_bias.py
    uv run python scripts/eval_bias.py --limit 1000   # dry-run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CIVIL_PARQUET = ROOT / "data" / "civil_comments" / "test.parquet"
DEFAULT_ONNX_DIR = ROOT / "models" / "onnx-toxic-bert-int8"
OUT_PREDS = ROOT / "docs" / "bias_predictions.parquet"

# Standard 9 subgroups from Borkan et al. 2019 / Jigsaw competition leaderboard.
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

# The model's `toxic` label is the head we evaluate against the dataset's
# continuous `toxicity` score. Both use the convention >= 0.5 means toxic.
TOXIC_LABEL = "toxic"
TOXIC_LABEL_IDX = 0  # see models/onnx-toxic-bert-int8/config.json id2label


def load_annotated_split() -> pd.DataFrame:
    df = pd.read_parquet(CIVIL_PARQUET)
    # Drop rows that were never sampled into the identity-annotation pool.
    # All identity columns are NaN-or-all-present together (one annotation pass),
    # so checking a single canonical column is sufficient and faster.
    annotated = df["male"].notna()
    df = df.loc[annotated].reset_index(drop=True)
    # Belt-and-suspenders: any remaining NaN identity treated as 0.0 (not mentioned).
    df[SUBGROUPS] = df[SUBGROUPS].fillna(0.0)
    return df


def predict_onnx(texts: list[str], batch_size: int, onnx_dir: Path) -> np.ndarray:
    """Returns per-row sigmoid prob for the `toxic` head only."""
    import onnxruntime as ort

    if not (onnx_dir / "model.onnx").exists():
        raise FileNotFoundError(
            f"missing {onnx_dir / 'model.onnx'}. Run scripts/export_onnx.py + "
            "scripts/quantize_onnx.py first."
        )

    config = json.loads((onnx_dir / "config.json").read_text())
    id2label = config["id2label"]
    if id2label[str(TOXIC_LABEL_IDX)] != TOXIC_LABEL:
        raise RuntimeError(
            f"label index {TOXIC_LABEL_IDX} is {id2label[str(TOXIC_LABEL_IDX)]!r}, "
            f"expected {TOXIC_LABEL!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_dir / "model.onnx"),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_names = {i.name for i in session.get_inputs()}

    # Length-bucket sort: most batches end up short and fast (same trick as eval_baseline.py).
    print("Pre-tokenizing to get sequence lengths…")
    lengths = np.fromiter(
        (
            len(ids)
            for ids in tokenizer(texts, truncation=True, max_length=512, add_special_tokens=True)[
                "input_ids"
            ]
        ),
        dtype=np.int32,
        count=len(texts),
    )
    sort_idx = np.argsort(lengths, kind="stable")
    unsort_idx = np.argsort(sort_idx, kind="stable")
    sorted_texts = [texts[i] for i in sort_idx]
    print(f"  seq_len: min={lengths.min()} median={int(np.median(lengths))} max={lengths.max()}")

    sorted_probs: list[float] = []
    for start in tqdm(range(0, len(sorted_texts), batch_size), desc="batches", unit="batch"):
        batch = sorted_texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=512)
        feed = {name: enc[name] for name in input_names if name in enc}
        logits = session.run(None, feed)[0]
        # Take the toxic head, apply sigmoid
        toxic_probs = 1.0 / (1.0 + np.exp(-logits[:, TOXIC_LABEL_IDX]))
        sorted_probs.extend(toxic_probs.tolist())

    out = np.empty(len(texts), dtype=np.float64)
    sorted_arr = np.asarray(sorted_probs)
    for i in range(len(texts)):
        out[i] = sorted_arr[unsort_idx[i]]
    return out


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, int, int]:
    """FPR = FP / (FP + TN) on a binary 0/1 label."""
    neg_mask = y_true == 0
    n_neg = int(neg_mask.sum())
    if n_neg == 0:
        return float("nan"), 0, 0
    fp = int((y_pred[neg_mask] == 1).sum())
    return fp / n_neg, fp, n_neg


def compute_bias_aucs(
    df: pd.DataFrame,
    subgroup: str,
    prob_col: str = "predicted_toxic_prob",
) -> dict[str, float | int]:
    """Borkan et al. 2019 'Power Mean' bias AUC family for one subgroup.

    - subgroup AUC: ranking within the subgroup
    - BPSN AUC (Background Positive, Subgroup Negative): tests whether the
      model assigns higher scores to non-toxic in-subgroup comments than
      toxic out-of-subgroup comments (low = over-flagging the subgroup)
    - BNSP AUC (Background Negative, Subgroup Positive): the mirror image
    """
    in_sg = df[subgroup] >= 0.5
    is_toxic = df["toxicity_binary"] == 1

    # Subgroup AUC: in-subgroup, toxic vs non-toxic
    sg = df.loc[in_sg]
    if sg["toxicity_binary"].nunique() < 2:
        sg_auc = float("nan")
    else:
        sg_auc = roc_auc_score(sg["toxicity_binary"], sg[prob_col])

    # BPSN: background non-toxic + subgroup toxic flipped is wrong; correct:
    #   positives = background NON-toxic (label them as the "should be higher" class)
    # Borkan defines BPSN as: AUC where the comparison is between
    #   non-toxic in subgroup (negatives we want ranked LOW)
    #   toxic outside subgroup (positives we want ranked HIGH)
    # A high BPSN AUC means the model correctly puts toxic-background above
    # non-toxic-subgroup. Low BPSN = over-flagging subgroup mentions.
    bpsn_mask = (in_sg & ~is_toxic) | (~in_sg & is_toxic)
    bpsn = df.loc[bpsn_mask]
    if bpsn["toxicity_binary"].nunique() < 2:
        bpsn_auc = float("nan")
    else:
        bpsn_auc = roc_auc_score(bpsn["toxicity_binary"], bpsn[prob_col])

    # BNSP: mirror. non-toxic outside subgroup vs toxic inside subgroup.
    # Low BNSP = under-flagging subgroup-toxic.
    bnsp_mask = (~in_sg & ~is_toxic) | (in_sg & is_toxic)
    bnsp = df.loc[bnsp_mask]
    if bnsp["toxicity_binary"].nunique() < 2:
        bnsp_auc = float("nan")
    else:
        bnsp_auc = roc_auc_score(bnsp["toxicity_binary"], bnsp[prob_col])

    sg_pred = (sg[prob_col] >= 0.5).astype(int).to_numpy()
    sg_fpr, sg_fp, sg_n_neg = compute_fpr(sg["toxicity_binary"].to_numpy(), sg_pred)

    return {
        "subgroup": subgroup,
        "n": int(in_sg.sum()),
        "n_toxic": int((in_sg & is_toxic).sum()),
        "fpr": sg_fpr,
        "fp": sg_fp,
        "negatives": sg_n_neg,
        "subgroup_auc": sg_auc,
        "bpsn_auc": bpsn_auc,
        "bnsp_auc": bnsp_auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=None, help="Eval only first N annotated rows (dry-run)"
    )
    parser.add_argument(
        "--onnx-dir",
        default=str(DEFAULT_ONNX_DIR),
        help="ONNX model dir (default: production INT8)",
    )
    parser.add_argument(
        "--out-preds", default=str(OUT_PREDS), help="Output parquet of raw probabilities"
    )
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.is_absolute():
        onnx_dir = ROOT / onnx_dir
    out_preds = Path(args.out_preds)
    if not out_preds.is_absolute():
        out_preds = ROOT / out_preds

    print(f"Loading Civil Comments identity-annotated test split from {CIVIL_PARQUET.name}…")
    df = load_annotated_split()
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"  identity-annotated rows: {len(df):,}")

    texts = df["comment_text"].fillna("").tolist()

    t0 = time.perf_counter()
    probs = predict_onnx(texts, args.batch_size, onnx_dir)
    elapsed = time.perf_counter() - t0
    print(f"\nInference time: {elapsed:.1f}s ({len(df) / elapsed:.1f} samples/s)")

    df = df.assign(
        predicted_toxic_prob=probs,
        toxicity_binary=(df["toxicity"] >= 0.5).astype(int),
        predicted_toxic=(probs >= 0.5).astype(int),
    )

    # Skip comment_text in the committed parquet — keeps it under the repo's
    # 10 MB pre-commit cap. Text is recoverable from data/civil_comments/test.parquet
    # via the `id` join key.
    keep_cols = [
        "id",
        "toxicity",
        "toxicity_binary",
        "predicted_toxic_prob",
        "predicted_toxic",
        *SUBGROUPS,
    ]
    preds_out = df[keep_cols].copy()
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    preds_out.to_parquet(out_preds, index=False)
    print(f"Wrote {out_preds.relative_to(ROOT)} ({out_preds.stat().st_size / 1e6:.1f} MB)")

    overall_fpr, overall_fp, overall_n_neg = compute_fpr(
        df["toxicity_binary"].to_numpy(), df["predicted_toxic"].to_numpy()
    )
    print(
        f"\nOverall FPR (annotated subset): {overall_fpr:.4f} "
        f"({overall_fp:,} / {overall_n_neg:,} non-toxic flagged)"
    )

    rows = [compute_bias_aucs(df, sg) for sg in SUBGROUPS]
    print("\nPer-subgroup metrics (n, n_toxic, FPR, subgroup-AUC, BPSN-AUC, BNSP-AUC):")
    print(
        f"{'subgroup':<35} {'n':>6} {'n_tox':>6} {'FPR':>7} {'SG-AUC':>7} {'BPSN':>7} {'BNSP':>7}"
    )
    for r in rows:
        print(
            f"{r['subgroup']:<35} {r['n']:>6,} {r['n_toxic']:>6,} "
            f"{r['fpr']:>7.4f} {r['subgroup_auc']:>7.4f} "
            f"{r['bpsn_auc']:>7.4f} {r['bnsp_auc']:>7.4f}"
        )

    # Stash for downstream rendering. We emit JSON to stdout the same way
    # eval_baseline.py does, so the bias_evaluation.md write step can scrape it.
    metrics_payload = {
        "overall_fpr": overall_fpr,
        "overall_fp": overall_fp,
        "overall_negatives": overall_n_neg,
        "n_rows": len(df),
        "n_toxic": int(df["toxicity_binary"].sum()),
        "inference_time_s": elapsed,
        "samples_per_s": len(df) / elapsed,
        "subgroups": rows,
    }
    print("\nmetrics_json=" + json.dumps(metrics_payload))


if __name__ == "__main__":
    main()
