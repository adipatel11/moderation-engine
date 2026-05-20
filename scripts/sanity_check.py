"""Phase 0 model sanity check.

Loads `unitary/toxic-bert`, runs inference on a fixed set of example inputs,
and writes per-label probabilities to `docs/sanity_check_output.txt` as a
reference snapshot for future comparison after ONNX export / quantization.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "unitary/toxic-bert"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "sanity_check_output.txt"

EXAMPLES = [
    "I love this, what a wonderful project!",
    "Thanks so much for your help.",
    "She is a brilliant engineer.",
    "I disagree with your opinion, but I respect it.",
    "Have a wonderful day!",
    "You are an idiot.",
    "This is the worst garbage I have ever seen.",
    "Shut up, you moron.",
    "Go back to where you came from.",
    "I am going to make you regret this.",
]


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    print(f"Labels: {labels}\n")

    lines: list[str] = [f"Model: {MODEL_NAME}", f"Labels: {labels}", ""]

    with torch.no_grad():
        for text in EXAMPLES:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).tolist()
            scores = {label: round(p, 4) for label, p in zip(labels, probs, strict=True)}
            lines.append(f"INPUT: {text}")
            lines.append(f"  {json.dumps(scores)}")
            lines.append("")
            print(f"{text}\n  {scores}\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"Wrote reference output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
