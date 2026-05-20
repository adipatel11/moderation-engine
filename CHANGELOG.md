# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0 scaffold: README with tagline + status badge, expanded `.gitignore` covering Python/macOS/VSCode/secrets, `CHANGELOG.md`, `docs/benchmarks.md` placeholder.
- Phase 1 (deploy-first): `scripts/ec2_userdata.sh` cloud-init bootstrap (installs Docker on AL2023, enables daemon, adds `ec2-user` to docker group). Verified end-to-end with a `c6i.large` instance + `hello-world` smoke test; instance stopped after verification to avoid spend.
- Phase 1 service skeleton: `moderation_engine.model.ToxicityClassifier` wrapping `unitary/toxic-bert`; `moderation_engine.api` FastAPI app with `POST /classify` and `GET /health`; pydantic-settings-driven `config.py`; structlog JSON logging via `_logging.py`. Smoke tests in `tests/test_smoke.py` (toxic-input flagged, clean-input not flagged, all six labels returned). All checks pass; local `curl` against `/classify` returns expected scores.
- Phase 1 containerization: multi-stage `Dockerfile` (uv builder + slim runtime, non-root user, `HEALTHCHECK` on `/health`, toxic-bert weights pre-baked into `/opt/hf-cache`, `TRANSFORMERS_OFFLINE=1` + `HF_HUB_OFFLINE=1`). `.dockerignore` strips data, tests, scripts, docs. `docker-compose.yml` for local dev with healthcheck and port mapping. `pyproject.toml` pins torch to the PyTorch CPU wheel index on Linux only via `[[tool.uv.index]]` + `marker = "sys_platform == 'linux'"`, removing 14 CUDA/nvidia-* packages and triton from the Linux lockfile. Final single-platform `linux/arm64` image is 822 MB (target: < 1.5 GB). Classification output is byte-identical to the host service.
