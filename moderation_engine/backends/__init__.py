"""Backend-agnostic toxicity-classifier interface.

The service code depends only on the `ToxicityClassifier` protocol and the
`build_classifier` factory; the concrete backend (PyTorch baseline, ONNX
Runtime, INT8-quantized ONNX, …) is selected from settings at startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from moderation_engine.config import Settings


class ToxicityClassifier(Protocol):
    labels: list[str]
    backend_name: str
    model_version: str
    # Public tokenizer so the batcher can do length-based bucketing without
    # re-implementing tokenization. Typed as Any to avoid hardcoding the
    # transformers AutoTokenizer surface in the Protocol.
    tokenizer: Any

    def predict(self, text: str) -> dict[str, float]: ...
    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]: ...


def build_classifier(settings: Settings) -> ToxicityClassifier:
    if settings.backend == "pytorch":
        from .pytorch import PyTorchToxicityClassifier

        return PyTorchToxicityClassifier(settings.model_name)
    if settings.backend == "onnx":
        from .onnx import ONNXToxicityClassifier

        return ONNXToxicityClassifier(
            settings.onnx_model_dir,
            settings.model_name,
            intra_op_num_threads=settings.onnx_intra_op_threads,
        )
    raise ValueError(f"unknown backend: {settings.backend!r}")
