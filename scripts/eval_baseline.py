"""Phase 1+ accuracy eval: per-label F1 / precision / recall / ROC-AUC on Jigsaw.

Backend-aware: select PyTorch (the floor) or ONNX (or any future backend) via
`--backend`. Output paths derive from the backend so multiple runs coexist
without overwriting each other:

    --backend pytorch -> docs/baseline_accuracy.md  + docs/baseline_predictions.parquet
    --backend onnx    -> docs/onnx_accuracy.md      + docs/onnx_predictions.parquet

Run from the repo root:
    uv run python scripts/eval_baseline.py --backend pytorch
    uv run python scripts/eval_baseline.py --backend onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "jigsaw" / "test.csv"
LABELS_CSV = ROOT / "data" / "jigsaw" / "test_labels.csv"
MODEL_NAME = "unitary/toxic-bert"
ONNX_DIR = ROOT / "models" / "onnx-toxic-bert"
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

Backend = Literal["pytorch", "onnx"]


def pick_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_scored_split() -> pd.DataFrame:
    test = pd.read_csv(TEST_CSV)
    labels = pd.read_csv(LABELS_CSV)
    merged = test.merge(labels, on="id", how="inner")
    return merged[(merged[LABEL_COLS] != -1).all(axis=1)].reset_index(drop=True)


@torch.no_grad()
def predict_pytorch(
    texts: list[str], device: torch.device, batch_size: int
) -> tuple[list[list[float]], list[str], str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    out: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="batches", unit="batch"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        logits = model(**enc).logits
        out.extend(torch.sigmoid(logits).cpu().tolist())
    return out, labels, f"{MODEL_NAME}@pytorch"


def predict_onnx(
    texts: list[str], batch_size: int, onnx_dir: Path, version_tag: str
) -> tuple[list[list[float]], list[str], str]:
    import onnxruntime as ort

    if not (onnx_dir / "model.onnx").exists():
        raise FileNotFoundError(
            f"missing {onnx_dir / 'model.onnx'}. Run scripts/export_onnx.py first."
        )

    config = json.loads((onnx_dir / "config.json").read_text())
    id2label = config["id2label"]
    labels = [id2label[str(i)] for i in range(len(id2label))]
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_dir / "model.onnx"),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    input_names = {i.name for i in session.get_inputs()}

    # Sort by tokenized length so batches share similar seq_len — most batches
    # become short and fast; only a few outlier batches near the end pad to 512.
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

    sorted_out: list[list[float]] = []
    for start in tqdm(range(0, len(sorted_texts), batch_size), desc="batches", unit="batch"):
        batch = sorted_texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="np", padding=True, truncation=True, max_length=512)
        feed = {name: enc[name] for name in input_names if name in enc}
        logits = session.run(None, feed)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        sorted_out.extend(probs.tolist())

    out: list[list[float]] = [sorted_out[unsort_idx[i]] for i in range(len(texts))]
    return out, labels, f"{MODEL_NAME}@{version_tag}"


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
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--limit", type=int, default=None, help="Eval only first N rows (dry-run)")
    parser.add_argument(
        "--onnx-dir",
        default=str(ONNX_DIR),
        help="ONNX model directory (use models/onnx-toxic-bert-int8 for INT8 eval)",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix (default: 'baseline' for pytorch, backend name otherwise)",
    )
    args = parser.parse_args()

    # PyTorch baseline keeps the legacy `baseline_*` filenames so existing
    # references in docs/benchmarks.md keep working.
    out_prefix = args.out_prefix or ("baseline" if args.backend == "pytorch" else args.backend)
    out_table = ROOT / "docs" / f"{out_prefix}_accuracy.md"
    out_preds = ROOT / "docs" / f"{out_prefix}_predictions.parquet"

    print(f"backend={args.backend} batch_size={args.batch_size}")

    print("Loading Jigsaw test split…")
    df = load_scored_split()
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"  scored rows: {len(df):,}")

    texts = df["comment_text"].fillna("").tolist()

    t0 = time.perf_counter()
    if args.backend == "pytorch":
        device = pick_device(args.device)
        print(f"PyTorch device={device}")
        raw_probs, model_labels, model_version = predict_pytorch(texts, device, args.batch_size)
        device_str = str(device)
    else:
        onnx_dir = Path(args.onnx_dir)
        if not onnx_dir.is_absolute():
            onnx_dir = ROOT / onnx_dir
        version_tag = "onnx-int8" if "int8" in onnx_dir.name else "onnx"
        raw_probs, model_labels, model_version = predict_onnx(
            texts, args.batch_size, onnx_dir, version_tag
        )
        device_str = "onnxruntime/CPUExecutionProvider"
    elapsed = time.perf_counter() - t0

    if model_labels != LABEL_COLS:
        raise RuntimeError(f"Label order mismatch: model={model_labels} expected={LABEL_COLS}")

    print(f"Inference time: {elapsed:.1f}s ({len(df) / elapsed:.1f} samples/s)")

    probs = pd.DataFrame(raw_probs, columns=LABEL_COLS, index=df.index)
    metrics = compute_metrics(df[LABEL_COLS], probs)

    table = render_table(metrics)
    print("\n" + table + "\n")

    macro_f1 = sum(r["f1"] for r in metrics) / len(metrics)
    summary = (
        f"# {args.backend.upper()} accuracy ({model_version})\n\n"
        f"- Model: `{MODEL_NAME}` · backend `{args.backend}`\n"
        f"- Dataset: Jigsaw Toxic Comment Classification, scored test split "
        f"({len(df):,} rows)\n"
        f"- Threshold: 0.5\n"
        f"- Compute: `{device_str}` · batch size {args.batch_size}\n"
        f"- Macro-average F1: **{macro_f1:.4f}**\n"
        f"- Inference time: {elapsed:.1f}s ({len(df) / elapsed:.1f} samples/s)\n\n"
        f"{table}\n"
    )
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_table.write_text(summary)
    print(f"Wrote {out_table.relative_to(ROOT)}")

    preds_out = pd.concat([df[["id"]].reset_index(drop=True), probs.reset_index(drop=True)], axis=1)
    preds_out.to_parquet(out_preds, index=False)
    print(f"Wrote {out_preds.relative_to(ROOT)} ({out_preds.stat().st_size / 1e6:.1f} MB)")

    print("\nmetrics_json=" + json.dumps(metrics))


if __name__ == "__main__":
    main()
