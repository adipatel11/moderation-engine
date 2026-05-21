# Latest locust sweep

Closed-loop concurrency sweep against the EC2-deployed container (see `docs/benchmarks.md` for the locked protocol and the historical per-stage tables). Each level runs for 60 s with zero think time; samples drawn from a 1000-row, seed-42 stratified mix of the Jigsaw scored test split. This file reflects the most recent run only — diff against `docs/benchmarks.md` for the stage-by-stage record.

| Users | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error rate |
|------:|---------:|---------:|---------:|---------:|---------:|-------------------:|-----------:|
| 1 | 714 | 0 | 60.0 | 170.0 | 430.0 | 12.3 | 0.00% |
| 5 | 1,322 | 0 | 140.0 | 610.0 | 1300.0 | 22.7 | 0.00% |
| 10 | 1,344 | 0 | 260.0 | 1300.0 | 3100.0 | 22.7 | 0.00% |
| 25 | 1,395 | 0 | 640.0 | 3100.0 | 7000.0 | 23.6 | 0.00% |
| 50 | 1,334 | 0 | 1500.0 | 5400.0 | 13000.0 | 23.1 | 0.00% |
| 100 | 1,319 | 0 | 3700.0 | 7600.0 | 13000.0 | 22.4 | 0.00% |
