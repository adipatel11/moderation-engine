# Baseline latency / throughput (locust sweep)

Closed-loop concurrency sweep against the EC2-deployed container (see `docs/benchmarks.md` for the locked protocol). Each level runs for 60 s with zero think time; samples drawn from a 1000-row, seed-42 stratified mix of the Jigsaw scored test split.

| Users | Requests | Failures | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (req/s) | Error rate |
|------:|---------:|---------:|---------:|---------:|---------:|-------------------:|-----------:|
| 1 | 252 | 0 | 160.0 | 600.0 | 860.0 | 4.3 | 0.00% |
| 5 | 421 | 0 | 510.0 | 1700.0 | 3100.0 | 7.1 | 0.00% |
| 10 | 409 | 0 | 1100.0 | 3000.0 | 4800.0 | 7.0 | 0.00% |
| 25 | 372 | 0 | 3300.0 | 7400.0 | 10000.0 | 6.3 | 0.00% |
| 50 | 368 | 0 | 6800.0 | 13000.0 | 18000.0 | 6.2 | 0.00% |
| 100 | 334 | 0 | 15000.0 | 22000.0 | 27000.0 | 5.7 | 0.00% |
