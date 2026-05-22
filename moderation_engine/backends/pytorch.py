from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PyTorchToxicityClassifier:
    backend_name = "pytorch"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels: list[str] = [
            self.model.config.id2label[i] for i in range(self.model.config.num_labels)
        ]
        self.model_version = f"{model_name}@pytorch"

    def predict(self, text: str) -> dict[str, float]:
        return self.predict_batch([text])[0]

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).tolist()
        return [dict(zip(self.labels, row, strict=True)) for row in probs]
