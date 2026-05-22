"""Length-aware dynamic request batcher.

Coalesces concurrent `/classify` requests into batched inferences while
avoiding the padding-waste regression that naive max-padding batching hits
on heterogeneous-length input. Implemented from scratch with `asyncio.Queue`
+ a single background worker — no mosec / litserve.

Why length-bucketing
--------------------
The first iteration of this batcher used a single queue. On Jigsaw the
seq_len distribution is heavy-tailed (median 51 tokens, p99 = 512 tokens),
so a random batch of 10 hits a long outlier ~40% of the time. With
`padding=True` the entire batch then pays the 512-token compute, and
throughput at saturation collapsed from ~22 req/s (single-sample bypass)
to ~5 req/s.

The fix: route each request into a length bucket *before* enqueueing.
Each bucket has its own FIFO queue, and a batch is built from one bucket
only — so the worst-case padding inside a batch is bounded by the bucket's
seq_len cap, not by the global maximum. Bucket boundaries default to
`[64, 256, 512]`, sized against the locust sample distribution (58%/37%/5%
traffic split). Tune via the `BATCHING_BUCKETS` env var.

Algorithm
---------
1. Caller awaits `batcher.predict(text)`. The batcher tokenizes the text to
   measure its length (cheap with a Rust-backed HF tokenizer), picks the
   smallest bucket that fits, creates a future, and enqueues into that
   bucket's queue.
2. A single worker task races `queue.get()` across all buckets
   (`asyncio.wait(..., return_when=FIRST_COMPLETED)`). Whichever bucket
   fires first becomes the active bucket for this tick. Pending getters on
   other buckets are cancelled (their queues retain the items).
3. The worker drains the active bucket for up to `window_ms`, or until
   `max_batch_size` items are gathered, then runs
   `backend.predict_batch(texts)` in the default thread executor so the
   CPU-bound inference call doesn't block the event loop.
4. Resolves every batched future with its prediction.
5. `window_ms == 0` disables the queue entirely — each call runs as a
   single-item batch in the executor, identical to pre-batching behavior.

Order is preserved within a bucket (FIFO). Order across buckets is not
guaranteed: a short request enqueued after a long one may finish first if
the short bucket fires before the long one. This is the intended trade-off
— it's what gives short requests their latency.
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
        self._task: asyncio.Task[None] | None = None

    def _bucket_for(self, length: int) -> int:
        for cap in self.buckets:
            if length <= cap:
                return cap
        return self.buckets[-1]

    def _length_of(self, text: str) -> int:
        # Tokenize with the same truncation cap as the largest bucket so the
        # returned length never exceeds the bucket boundary.
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
        self._task = asyncio.create_task(self._run(), name="batcher-worker")
        log.info(
            "batcher_started",
            window_ms=self.window_ms,
            max_batch_size=self.max_batch_size,
            buckets=self.buckets,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None
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

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        window_s = self.window_ms / 1000.0
        while True:
            # Race a get() across all bucket queues. Whichever bucket has the
            # oldest pending item fires first; pending getters on the others
            # are cancelled (no items lost — Queue.get() cancellation just
            # removes the waiter, items remain in queue).
            get_tasks: dict[asyncio.Task[tuple[str, asyncio.Future[dict[str, float]]]], int] = {
                asyncio.create_task(q.get()): bucket for bucket, q in self._queues.items()
            }
            try:
                done, pending = await asyncio.wait(
                    get_tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
            except asyncio.CancelledError:
                for task in get_tasks:
                    task.cancel()
                raise

            for task in pending:
                task.cancel()

            # Process every bucket that completed in this tick — usually just
            # one, but multiple buckets may have had items waiting when the
            # worker re-entered the loop.
            for task in done:
                bucket = get_tasks[task]
                first = task.result()
                queue = self._queues[bucket]

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
