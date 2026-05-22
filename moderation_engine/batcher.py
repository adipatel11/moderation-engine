"""In-process dynamic request batcher.

Coalesces concurrent `/classify` requests into a single batched inference so
throughput rises far faster than per-request latency. Implemented from
scratch with `asyncio.Queue` + a single background task — no mosec / litserve
on the hot path. The implementation is the Phase 2 Opt 3 interview story.

Algorithm
---------
1. Caller awaits `batcher.predict(text)`. The batcher creates a future,
   enqueues `(text, future)`, and the caller blocks on the future.
2. A single worker task pops items off the queue:

   - Block until at least one item arrives (`queue.get()`).
   - Drain up to `max_batch_size - 1` additional items, waiting at most
     `window_ms` total for siblings to join (`asyncio.wait_for`).
   - Run `backend.predict_batch(texts)` in the default thread executor so the
     CPU-bound inference call doesn't block the event loop.
   - Resolve every batched future with its corresponding prediction.

3. `window_ms == 0` disables the queue entirely: each call runs as a
   single-item batch in the executor. This keeps the same image deployable
   as both the "baseline" and "batched" arm of the window sweep.

Order is preserved: results come back in the same order as the input texts,
so future N gets prediction N.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Protocol

import structlog


class BatchPredictor(Protocol):
    """Minimum surface the batcher needs from a classifier backend."""

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]: ...


log = structlog.get_logger("batcher")


class Batcher:
    def __init__(
        self,
        backend: BatchPredictor,
        window_ms: float,
        max_batch_size: int = 32,
    ) -> None:
        if window_ms < 0:
            raise ValueError(f"window_ms must be >= 0, got {window_ms}")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        self.backend = backend
        self.window_ms = float(window_ms)
        self.max_batch_size = int(max_batch_size)
        self.enabled = self.window_ms > 0
        self._queue: asyncio.Queue[tuple[str, asyncio.Future[dict[str, float]]]] | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self.enabled:
            log.info("batcher_disabled", window_ms=self.window_ms)
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run(), name="batcher-worker")
        log.info("batcher_started", window_ms=self.window_ms, max_batch_size=self.max_batch_size)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._queue = None
        log.info("batcher_stopped")

    async def predict(self, text: str) -> dict[str, float]:
        if not self.enabled:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self.backend.predict_batch, [text])
            return results[0]

        assert self._queue is not None, "Batcher.predict called before start()"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, float]] = loop.create_future()
        await self._queue.put((text, future))
        return await future

    async def _run(self) -> None:
        assert self._queue is not None
        window_s = self.window_ms / 1000.0
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch: list[tuple[str, asyncio.Future[dict[str, float]]]] = [first]
            deadline = loop.time() + window_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                batch.append(item)

            texts = [t for t, _ in batch]
            futures = [f for _, f in batch]
            t0 = loop.time()
            try:
                results = await loop.run_in_executor(None, self.backend.predict_batch, texts)
            except BaseException as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
                continue
            inference_ms = (loop.time() - t0) * 1000.0
            log.info("batch_run", size=len(batch), inference_ms=round(inference_ms, 2))
            for fut, res in zip(futures, results, strict=True):
                if not fut.done():
                    fut.set_result(res)
