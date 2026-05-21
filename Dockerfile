# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11.15
ARG UV_VERSION=0.11.15

# --- builder ---------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/opt/hf-cache

# Deps first (caches when only app code changes). README.md is referenced
# by pyproject.toml's `readme` field so hatchling needs it at build time.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Then install the project itself.
COPY moderation_engine ./moderation_engine
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Bake the INT8-quantized ONNX export into the image (used by BACKEND=onnx,
# the only backend supported in this image). Built locally via
# `uv run python scripts/export_onnx.py && uv run python scripts/quantize_onnx.py`.
# The export dir bundles the tokenizer files alongside model.onnx so the
# runtime needs neither the HF cache nor network access.
COPY models/onnx-toxic-bert-int8 /opt/onnx-toxic-bert-int8

# --- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    BACKEND=onnx \
    ONNX_MODEL_DIR=/opt/onnx-toxic-bert-int8

RUN useradd --create-home --shell /bin/bash --uid 1000 app
USER app
WORKDIR /home/app

COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/onnx-toxic-bert-int8 /opt/onnx-toxic-bert-int8
COPY --from=builder --chown=app:app /app/moderation_engine ./moderation_engine

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "moderation_engine.api:app", "--host", "0.0.0.0", "--port", "8000"]
