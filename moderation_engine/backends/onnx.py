"""ONNX Runtime backend.

Loads a toxic-bert export produced by `scripts/export_onnx.py` (optionally
INT8-quantized via `scripts/quantize_onnx.py`) and serves predictions through
`onnxruntime.InferenceSession` with the CPU execution provider — no PyTorch
on the hot path.

Exposes both `predict(text)` and `predict_batch(texts)`; the batched form is
the primary path (the dynamic-batcher in `moderation_engine.batcher` calls
it directly) and the single-item form delegates to it.

Expected layout under `model_dir`:
    model.onnx
    config.json   (used for the id2label mapping)
    tokenizer.json + tokenizer_config.json + vocab.txt + special_tokens_map.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


class ONNXToxicityClassifier:
    backend_name = "onnx"

    def __init__(
        self,
        model_dir: Path | str,
        model_name_fallback: str,
        intra_op_num_threads: int = 0,
    ) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"ONNX backend requested but {model_dir} does not exist. "
                "Run `uv run python scripts/export_onnx.py` first."
            )
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"missing {onnx_path}")

        config_path = model_dir / "config.json"
        config = json.loads(config_path.read_text())
        id2label = config["id2label"]
        self.labels: list[str] = [id2label[str(i)] for i in range(len(id2label))]

        # Prefer the tokenizer that was bundled with the export; fall back to
        # the hub model so a partial export still serves something.
        tokenizer_src = (
            model_dir if (model_dir / "tokenizer.json").exists() else model_name_fallback
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src))

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_num_threads > 0:
            # 0 keeps ORT's default (=num_logical_cores). Setting to 1 frees
            # the second vCPU so multiple batches can run truly in parallel
            # through the executor — see Opt 4 (`docs/benchmarks.md`).
            sess_opts.intra_op_num_threads = intra_op_num_threads
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {i.name for i in self.session.get_inputs()}
        self.model_version = f"{config.get('_name_or_path', model_name_fallback)}@onnx"

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        enc = self.tokenizer(
            texts, return_tensors="np", padding=True, truncation=True, max_length=512
        )
        feed = {name: enc[name] for name in self._input_names if name in enc}
        logits = self.session.run(None, feed)[0]
        probs = 1.0 / (1.0 + np.exp(-logits))
        return [dict(zip(self.labels, row.tolist(), strict=True)) for row in probs]
