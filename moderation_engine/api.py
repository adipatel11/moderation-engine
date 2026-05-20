from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from ._logging import configure_logging
from .config import settings
from .model import ToxicityClassifier

configure_logging(settings.log_level)
log = structlog.get_logger()


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)


class ClassifyResponse(BaseModel):
    labels: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("model_load_start", model_name=settings.model_name)
    app.state.classifier = ToxicityClassifier(settings.model_name)
    log.info("model_load_done", labels=app.state.classifier.labels)
    yield
    log.info("shutdown")


app = FastAPI(title="moderation-engine", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = hasattr(request.app.state, "classifier")
    return HealthResponse(status="ok" if loaded else "loading", model_loaded=loaded)


@app.post("/classify", response_model=ClassifyResponse)
def classify(payload: ClassifyRequest, request: Request) -> ClassifyResponse:
    request_id = uuid.uuid4().hex
    bound = log.bind(request_id=request_id, text_length=len(payload.text))
    started = time.perf_counter()
    try:
        labels = request.app.state.classifier.predict(payload.text)
    except Exception as exc:
        bound.error("classify_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="classification failed") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    top_label = max(labels.items(), key=lambda kv: kv[1])[0]
    bound.info("classify", latency_ms=round(latency_ms, 2), top_label=top_label)
    return ClassifyResponse(labels=labels, model_version=settings.model_name)
