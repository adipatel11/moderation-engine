# INT4-NF4 accuracy (unitary/toxic-bert via MatMulBnb4Quantizer)

- Model: `unitary/toxic-bert` · backend `onnx` (Bnb4 NF4 weight quant, block_size=64)
- Dataset: Jigsaw Toxic Comment Classification, scored test split (63,978 rows)
- Threshold: 0.5
- Compute: `onnxruntime/CPUExecutionProvider` (M3 Pro, intra_op default) · batch size 32
- Macro-average F1: **0.6127** (FP32 baseline 0.6101, INT8 dynamic 0.6146)
- Decision flips vs FP32 baseline: 608 / 383,868 (0.158%) — lowest of any backend
- Inference time: 2,527.5 s (25.3 samples/s — 3× slower than INT8's 76.5 samples/s)

| Label | Positives | F1 | Precision | Recall | ROC-AUC |
|-------|----------:|---:|----------:|-------:|--------:|
| `toxic` | 6,090 | 0.6766 | 0.5377 | 0.9123 | 0.9739 |
| `severe_toxic` | 367 | 0.4560 | 0.3892 | 0.5504 | 0.9915 |
| `obscene` | 3,691 | 0.6835 | 0.5662 | 0.8621 | 0.9825 |
| `threat` | 211 | 0.5338 | 0.4039 | 0.7867 | 0.9966 |
| `insult` | 3,427 | 0.6946 | 0.6294 | 0.7747 | 0.9806 |
| `identity_hate` | 712 | 0.6317 | 0.5741 | 0.7022 | 0.9928 |
