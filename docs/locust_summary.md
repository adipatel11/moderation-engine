# Latest locust sweep

Closed-loop concurrency sweep against the EC2-deployed container (see `docs/benchmarks.md` for the locked protocol and the historical per-stage tables). Each level runs for 60 s with zero think time; samples drawn from a 1000-row, seed-42 stratified mix of the Jigsaw scored test split. This file reflects the most recent run only — diff against `docs/benchmarks.md` for the stage-by-stage record.

| Users | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error rate |
|------:|---------:|---------:|---------:|---------:|---------:|-------------------:|-----------:|
| 1 | 320 | 0 | 120.0 | 510.0 | 930.0 | 5.6 | 0.00% |
| 5 | 514 | 0 | 370.0 | 1500.0 | 2500.0 | 8.8 | 0.00% |
| 10 | 470 | 0 | 750.0 | 3900.0 | 7300.0 | 8.1 | 0.00% |
| 25 | 499 | 0 | 1900.0 | 8000.0 | 19000.0 | 8.4 | 0.00% |
| 50 | 512 | 0 | 4200.0 | 12000.0 | 23000.0 | 8.7 | 0.00% |
| 100 | 454 | 0 | 11000.0 | 20000.0 | 30000.0 | 7.7 | 0.00% |
