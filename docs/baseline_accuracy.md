# Baseline accuracy floor

- Model: `unitary/toxic-bert`
- Dataset: Jigsaw Toxic Comment Classification, scored test split (63,978 rows)
- Threshold: 0.5
- Device: `mps` · batch size 32
- Macro-average F1: **0.6101**
- Inference time: 2010.9s (31.8 samples/s)

| Label | Positives | F1 | Precision | Recall | ROC-AUC |
|-------|----------:|---:|----------:|-------:|--------:|
| `toxic` | 6,090 | 0.6775 | 0.5394 | 0.9107 | 0.9739 |
| `severe_toxic` | 367 | 0.4491 | 0.3755 | 0.5586 | 0.9915 |
| `obscene` | 3,691 | 0.6822 | 0.5623 | 0.8672 | 0.9824 |
| `threat` | 211 | 0.5307 | 0.3923 | 0.8199 | 0.9965 |
| `insult` | 3,427 | 0.6920 | 0.6197 | 0.7835 | 0.9805 |
| `identity_hate` | 712 | 0.6293 | 0.5603 | 0.7177 | 0.9928 |
