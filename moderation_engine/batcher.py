"""Length-aware dynamic request batcher with per-bucket workers.

Coalesces concurrent `/classify` requests into batched inferences while
avoiding the padding-waste regression that naive max-padding batching hits
on heterogeneous-length input. Implemented from scratch with
`asyncio.Queue` + per-bucket background workers — no mosec / litserve.

History
-------
- v1 (Opt 3, naive): single queue, max-padding. 4x throughput regression
  vs bypass on Jigsaw because random batches hit a long-tail outlier ~40%
  of the time and forced all callers to pay 512-token compute.
- v2 (Opt 3, bucketed): single worker, length-bucketed queues. Padding
  waste eliminated, but bypass still won on a 2-vCPU host because ORT's
  intra-op parallelism already saturated both cores at batch=1.
- v3 (Opt 4, this file): **one worker per bucket** so multiple batches
  can run concurrently through the executor. Pair with
  `ONNX_INTRA_OP_THREADS=1` so each in-flight batch uses one vCPU and
  two batches can truly parallelize on the c6i.large.

Algorithm
---------
1. Caller awaits `batcher.predict(text)`. The batcher tokenizes the text
   to measure its length (cheap with the Rust-backed HF tokenizer),
   picks the smallest bucket whose cap fits, creates a future, and
   enqueues into that bucket's queue.
2. Each bucket has its own background worker (`_run_bucket`). The worker
   pops the head item, drains up to `max_batch_size - 1` more siblings
   within `window_ms`, runs `backend.predict_batch(texts)` through the
   default thread executor, and resolves every future in the batch.
3. While one bucket's worker is awaiting its inference, the other
   buckets' workers can advance and submit their own inferences — the
   executor's thread pool runs them concurrently. With
   `intra_op_num_threads=1`, this is real parallelism on multi-vCPU.
4. `window_ms == 0` disables the queue entirely: each call runs as a
   single-item batch in the executor, identical to pre-batching
   behaviour.

Order is FIFO within a bucket; order across buckets is not preserved —
a short request enqueued after a long one may finish first if the short
bucket's worker outpaces the long one. This is the intended trade-off.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Protocol

import structlog


class BatchPredictor(Protocol):
    """Minimum surface the batcher needs from a classifier backend."""

    tokenizer: object  # HF tokenizer (used for length detection only)

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]: ...


log = structlog.get_logger("batcher")


class Batcher:
    def __init__(
        self,
        backend: BatchPredictor,
        window_ms: float,
        max_batch_size: int = 32,
        buckets: list[int] | None = None,
    ) -> None:
        if window_ms < 0:
            raise ValueError(f"window_ms must be >= 0, got {window_ms}")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        if buckets is not None and len(buckets) == 0:
            raise ValueError("buckets must contain at least one length cap")

        self.backend = backend
        self.window_ms = float(window_ms)
        self.max_batch_size = int(max_batch_size)
        self.buckets: list[int] = sorted(buckets) if buckets else [512]
        self.enabled = self.window_ms > 0
        self._queues: dict[int, asyncio.Queue[tuple[str, asyncio.Future[dict[str, float]]]]] = {
            b: asyncio.Queue() for b in self.buckets
        }
        self._tasks: list[asyncio.Task[None]] = []

    def _bucket_for(self, length: int) -> int:
        for cap in self.buckets:
            if length <= cap:
                return cap
        return self.buckets[-1]

    def _length_of(self, text: str) -> int:
        enc = self.backend.tokenizer(
            text,
            truncation=True,
            max_length=self.buckets[-1],
            add_special_tokens=True,
        )
        return len(enc["input_ids"])

    async def start(self) -> None:
        if not self.enabled:
            log.info("batcher_disabled", window_ms=self.window_ms)
            return
        for bucket in self.buckets:
            task = asyncio.create_task(self._run_bucket(bucket), name=f"batcher-{bucket}")
            self._tasks.append(task)
        log.info(
            "batcher_started",
            window_ms=self.window_ms,
            max_batch_size=self.max_batch_size,
            buckets=self.buckets,
        )

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        log.info("batcher_stopped")

    async def predict(self, text: str) -> dict[str, float]:
        if not self.enabled:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self.backend.predict_batch, [text])
            return results[0]

        bucket = self._bucket_for(self._length_of(text))
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, float]] = loop.create_future()
        await self._queues[bucket].put((text, future))
        return await future

    async def _run_bucket(self, bucket: int) -> None:
        queue = self._queues[bucket]
        loop = asyncio.get_running_loop()
        window_s = self.window_ms / 1000.0
        while True:
            first = await queue.get()
            batch: list[tuple[str, asyncio.Future[dict[str, float]]]] = [first]
            deadline = loop.time() + window_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
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
            log.info(
                "batch_run",
                bucket=bucket,
                size=len(batch),
                inference_ms=round(inference_ms, 2),
            )
            for fut, res in zip(futures, results, strict=True):
                if not fut.done():
                    fut.set_result(res)
