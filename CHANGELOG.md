# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 0 scaffold: README with tagline + status badge, expanded `.gitignore` covering Python/macOS/VSCode/secrets, `CHANGELOG.md`, `docs/benchmarks.md` placeholder.
- Phase 1 (deploy-first): `scripts/ec2_userdata.sh` cloud-init bootstrap (installs Docker on AL2023, enables daemon, adds `ec2-user` to docker group). Verified end-to-end with a `c6i.large` instance + `hello-world` smoke test; instance stopped after verification to avoid spend.
- Phase 1 service skeleton: `moderation_engine.model.ToxicityClassifier` wrapping `unitary/toxic-bert`; `moderation_engine.api` FastAPI app with `POST /classify` and `GET /health`; pydantic-settings-driven `config.py`; structlog JSON logging via `_logging.py`. Smoke tests in `tests/test_smoke.py` (toxic-input flagged, clean-input not flagged, all six labels returned). All checks pass; local `curl` against `/classify` returns expected scores.
