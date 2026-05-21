"""Phase 1 accuracy floor: evaluate baseline toxic-bert on the Jigsaw test set.

Loads `data/jigsaw/test.csv` + `data/jigsaw/test_labels.csv`, filters out the
rows Kaggle masked with `-1`, runs `unitary/toxic-bert` in batches, and reports
per-label F1 / precision / recall / ROC-AUC at threshold 0.5.

Prints a markdown table to stdout and writes the same table plus a summary
header to `docs/baseline_accuracy.md`. Raw predicted probabilities are saved
to `docs/baseline_predictions.parquet` so future optimization runs (ONNX,
INT8) can be parity-checked against this snapshot without re-running PyTorch.

Run from the repo root:
    uv run python scripts/eval_baseline.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "jigsaw" / "test.csv"
LABELS_CSV = ROOT / "data" / "jigsaw" / "test_labels.csv"
OUT_TABLE = ROOT / "docs" / "baseline_accuracy.md"
OUT_PREDS = ROOT / "docs" / "baseline_predictions.parquet"
MODEL_NAME = "unitary/toxic-bert"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_scored_split() -> pd.DataFrame:
    """Return only the rows Kaggle actually scored (drops `-1` masked rows)."""
    test = pd.read_csv(TEST_CSV)
    labels = pd.read_csv(LABELS_CSV)
    merged = test.merge(labels, on="id", how="inner")
    scored = merged[(merged[LABEL_COLS] != -1).all(axis=1)].reset_index(drop=True)
    return scored


@torch.no_grad()
def predict_probs(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    out: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="batches", unit="batch"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().tolist()
        out.extend(probs)
    return out


def compute_metrics(y_true: pd.DataFrame, probs: pd.DataFrame) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    preds = (probs >= 0.5).astype(int)
    for label in LABEL_COLS:
        rows.append(
            {
                "label": label,
                "positives": int(y_true[label].sum()),
                "f1": f1_score(y_true[label], preds[label], zero_division=0),
                "precision": precision_score(y_true[label], preds[label], zero_division=0),
                "recall": recall_score(y_true[label], preds[label], zero_division=0),
                "roc_auc": roc_auc_score(y_true[label], probs[label]),
            }
        )
    return rows


def render_table(rows: list[dict[str, float]]) -> str:
    header = "| Label | Positives | F1 | Precision | Recall | ROC-AUC |\n"
    header += "|-------|----------:|---:|----------:|-------:|--------:|\n"
    body = "\n".join(
        f"| `{r['label']}` | {r['positives']:,} | {r['f1']:.4f} | "
        f"{r['precision']:.4f} | {r['recall']:.4f} | {r['roc_auc']:.4f} |"
        for r in rows
    )
    return header + body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=None, help="Eval only first N rows (dry-run)")
    args = parser.parse_args()

    device = pick_device(args.device)
    print(f"device={device} batch_size={args.batch_size}")

    print("Loading Jigsaw test split…")
    df = load_scored_split()
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"  scored rows: {len(df):,}")

    print(f"Loading model {MODEL_NAME}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    model_labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    if model_labels != LABEL_COLS:
        raise RuntimeError(f"Label order mismatch: model={model_labels} expected={LABEL_COLS}")

    t0 = time.perf_counter()
    raw_probs = predict_probs(
        df["comment_text"].fillna("").tolist(),
        tokenizer,
        model,
        device,
        args.batch_size,
    )
    elapsed = time.perf_counter() - t0
    print(f"Inference time: {elapsed:.1f}s ({len(df) / elapsed:.1f} samples/s)")

    probs = pd.DataFrame(raw_probs, columns=LABEL_COLS, index=df.index)
    metrics = compute_metrics(df[LABEL_COLS], probs)

    table = render_table(metrics)
    print("\n" + table + "\n")

    macro_f1 = sum(r["f1"] for r in metrics) / len(metrics)
    summary = (
        "# Baseline accuracy floor\n\n"
        f"- Model: `{MODEL_NAME}`\n"
        f"- Dataset: Jigsaw Toxic Comment Classification, scored test split "
        f"({len(df):,} rows)\n"
        f"- Threshold: 0.5\n"
        f"- Device: `{device}` · batch size {args.batch_size}\n"
        f"- Macro-average F1: **{macro_f1:.4f}**\n"
        f"- Inference time: {elapsed:.1f}s ({len(df) / elapsed:.1f} samples/s)\n\n"
        f"{table}\n"
    )
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.write_text(summary)
    print(f"Wrote {OUT_TABLE.relative_to(ROOT)}")

    preds_out = pd.concat([df[["id"]].reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    preds_out.to_parquet(OUT_PREDS, index=False)
    print(f"Wrote {OUT_PREDS.relative_to(ROOT)} ({OUT_PREDS.stat().st_size / 1e6:.1f} MB)")

    print("\nmetrics_json=" + json.dumps(metrics))


if __name__ == "__main__":
    main()
