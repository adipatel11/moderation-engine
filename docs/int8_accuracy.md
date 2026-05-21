# ONNX accuracy (unitary/toxic-bert@onnx-int8)

- Model: `unitary/toxic-bert` · backend `onnx`
- Dataset: Jigsaw Toxic Comment Classification, scored test split (63,978 rows)
- Threshold: 0.5
- Compute: `onnxruntime/CPUExecutionProvider` · batch size 64
- Macro-average F1: **0.6146**
- Inference time: 836.4s (76.5 samples/s)

| Label | Positives | F1 | Precision | Recall | ROC-AUC |
|-------|----------:|---:|----------:|-------:|--------:|
| `toxic` | 6,090 | 0.6924 | 0.5676 | 0.8874 | 0.9736 |
| `severe_toxic` | 367 | 0.4464 | 0.3528 | 0.6076 | 0.9913 |
| `obscene` | 3,691 | 0.6893 | 0.5743 | 0.8621 | 0.9827 |
| `threat` | 211 | 0.5333 | 0.4113 | 0.7583 | 0.9956 |
| `insult` | 3,427 | 0.6938 | 0.6207 | 0.7864 | 0.9806 |
| `identity_hate` | 712 | 0.6325 | 0.5355 | 0.7725 | 0.9929 |
