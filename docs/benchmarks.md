# Benchmarks

This file is the single source of truth for the project's measured numbers. Every optimization in Phase 2 is appended here against a locked protocol established in Phase 1.

## Protocol (locked in Phase 1 — do not change once set)

- **Hardware**: AWS EC2 `c6i.large` (2 vCPU compute-optimized, x86_64), `us-east-1`
- **Test inputs**: Jigsaw Toxic Comment Classification test set (realistic mix of toxic and non-toxic, never a single repeated string)
- **Load generator**: `locustfile.py`, concurrency sweep at 1, 5, 10, 25, 50, 100 users
- **Metrics recorded**: p50 / p95 / p99 latency (ms), throughput (req/s), error rate (%)
- **Accuracy metrics**: per-label F1, precision, recall, ROC-AUC on Jigsaw test set

## Accuracy floor (Phase 1)

_Pending — populated by `scripts/eval_baseline.py`._

| Label | F1 | Precision | Recall | ROC-AUC |
|-------|----|-----------|--------|---------|
| _TBD_ |    |           |        |         |

## Latency / throughput

| Run | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Errors | Notes |
|-----|---------:|---------:|---------:|-------------------:|-------:|-------|
| _Baseline — pending_ |   |   |   |   |   | PyTorch on CPU, no optimization |

## Optimization journey

_Populated through Phase 2. Each row gets a one-paragraph "what changed / what surprised" beneath the table._

| Stage | p99 (ms) | Throughput (req/s) | F1 (avg) | Notes |
|-------|---------:|-------------------:|---------:|-------|
| Baseline (PyTorch CPU)      | _TBD_ | _TBD_ | _TBD_ | — |
| + ONNX Runtime              | _TBD_ | _TBD_ | _TBD_ | — |
| + INT8 dynamic quantization | _TBD_ | _TBD_ | _TBD_ | — |
| + Dynamic batching (5 ms)   | _TBD_ | _TBD_ | _TBD_ | — |
| + Opt 4 (TBD)               | _TBD_ | _TBD_ | _TBD_ | — |
