# moderation-engine

> Low-latency content moderation service for real-time chat.

![status](https://img.shields.io/badge/status-%F0%9F%9A%A7%20in%20active%20development-yellow)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A toxicity-classification HTTP service built around `unitary/toxic-bert`, optimized for CPU inference latency. Headline numbers and architecture details will land here once Phase 1 (deployed baseline) is in.

## Quick start

_Coming after Phase 1._ Once the service is containerized, `docker compose up` will boot a local instance on port 8000.

## Roadmap

The project is structured as four phases:

1. **Pre-flight** — repo + tooling + AWS + model sanity-check
2. **Naive baseline** — FastAPI service deployed to EC2 with locked benchmark protocol
3. **Optimization sprint** — ONNX export → INT8 quantization → dynamic batching → wildcard
4. **Production polish + writeup** — observability, CI, bias evaluation, blog post

Benchmark log: [`docs/benchmarks.md`](docs/benchmarks.md).
Bias evaluation: [`docs/bias_evaluation.md`](docs/bias_evaluation.md).
