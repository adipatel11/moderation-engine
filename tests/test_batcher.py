"""Batcher unit tests.

The batcher is exercised against an in-memory fake backend so the tests
stay fast and don't depend on ONNX / model weights. Tests use `asyncio.run`
rather than pytest-asyncio to avoid the extra dev dependency.

`FakeTokenizer` reports one "token" per input character so we can drive
bucket routing deterministically from text length.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from moderation_engine.batcher import Batcher


class FakeTokenizer:
    def __call__(
        self,
        text: str,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = False,
        **kwargs: object,
    ) -> dict[str, list[int]]:
        n = len(text)
        if add_special_tokens:
            n += 2
        if max_length is not None and truncation:
            n = min(n, max_length)
        return {"input_ids": list(range(n))}


class FakeBackend:
    """Records every batch it sees; returns a dict mirroring the input text."""

    def __init__(self, sleep_s: float = 0.0, raise_exc: BaseException | None = None) -> None:
        self.tokenizer = FakeTokenizer()
        self.batches: list[list[str]] = []
        self.sleep_s = sleep_s
        self.raise_exc = raise_exc

    def predict_batch(self, texts: list[str]) -> list[dict[str, float]]:
        self.batches.append(list(texts))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.raise_exc is not None:
            raise self.raise_exc
        return [{"echo": float(len(t))} for t in texts]


def test_single_request_runs_as_size_one_batch() -> None:
    async def main() -> None:
        backend = FakeBackend()
        batcher = Batcher(backend, window_ms=5)
        await batcher.start()
        try:
            result = await batcher.predict("hello")
            assert result == {"echo": 5.0}
            assert backend.batches == [["hello"]]
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_concurrent_same_bucket_requests_get_coalesced() -> None:
    async def main() -> None:
        backend = FakeBackend(sleep_s=0.05)
        # Single bucket so all five texts queue together.
        batcher = Batcher(backend, window_ms=20, max_batch_size=8, buckets=[512])
        await batcher.start()
        try:
            texts = ["a", "bb", "ccc", "dddd", "eeeee"]
            results = await asyncio.gather(*(batcher.predict(t) for t in texts))
            assert [r["echo"] for r in results] == [1, 2, 3, 4, 5]
            assert max(len(b) for b in backend.batches) >= 2
            flat = [t for b in backend.batches for t in b]
            assert sorted(flat) == sorted(texts)
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_short_and_long_dont_mix() -> None:
    """A short text and a long text must never appear in the same batch."""

    async def main() -> None:
        backend = FakeBackend(sleep_s=0.03)
        # Three-bucket split: 32 / 128 / 512 (matches the default proposal).
        batcher = Batcher(backend, window_ms=15, max_batch_size=16, buckets=[32, 128, 512])
        await batcher.start()
        try:
            # 5 short (≤32 chars+2 specials → bucket 32) + 5 long (≥256 → bucket 512)
            shorts = ["s" * 10] * 5
            longs = ["l" * 300] * 5
            texts = shorts + longs
            await asyncio.gather(*(batcher.predict(t) for t in texts))

            for batch in backend.batches:
                kinds = {t[0] for t in batch}
                assert kinds == {"s"} or kinds == {"l"}, f"mixed batch: {batch[:3]}…"

            kinds_seen = {next(iter({t[0] for t in b})) for b in backend.batches}
            assert kinds_seen == {"s", "l"}
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_max_batch_size_caps_inference() -> None:
    async def main() -> None:
        backend = FakeBackend(sleep_s=0.05)
        batcher = Batcher(backend, window_ms=50, max_batch_size=3, buckets=[512])
        await batcher.start()
        try:
            texts = [f"t{i}" for i in range(10)]
            await asyncio.gather(*(batcher.predict(t) for t in texts))
            assert all(len(b) <= 3 for b in backend.batches)
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_window_zero_bypasses_queue() -> None:
    async def main() -> None:
        backend = FakeBackend()
        batcher = Batcher(backend, window_ms=0)
        await batcher.start()
        assert batcher.enabled is False
        try:
            await asyncio.gather(*(batcher.predict(t) for t in ["x", "y", "z"]))
            assert sorted(b[0] for b in backend.batches) == ["x", "y", "z"]
            assert all(len(b) == 1 for b in backend.batches)
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_exception_propagates_to_all_callers_in_batch() -> None:
    class BoomError(RuntimeError):
        pass

    async def main() -> None:
        backend = FakeBackend(sleep_s=0.02, raise_exc=BoomError("kaboom"))
        batcher = Batcher(backend, window_ms=20, max_batch_size=8, buckets=[512])
        await batcher.start()
        try:
            with pytest.raises(BoomError):
                await asyncio.gather(
                    batcher.predict("a"),
                    batcher.predict("b"),
                    batcher.predict("c"),
                )
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_invalid_args_rejected() -> None:
    backend = FakeBackend()
    with pytest.raises(ValueError):
        Batcher(backend, window_ms=-1)
    with pytest.raises(ValueError):
        Batcher(backend, window_ms=5, max_batch_size=0)
    with pytest.raises(ValueError):
        Batcher(backend, window_ms=5, buckets=[])


def test_bucket_for_routes_lengths_correctly() -> None:
    batcher = Batcher(FakeBackend(), window_ms=5, buckets=[64, 256, 512])
    assert batcher._bucket_for(1) == 64
    assert batcher._bucket_for(64) == 64
    assert batcher._bucket_for(65) == 256
    assert batcher._bucket_for(256) == 256
    assert batcher._bucket_for(257) == 512
    assert batcher._bucket_for(9999) == 512  # clamped to largest bucket


def test_buckets_run_concurrently() -> None:
    """A short-bucket batch and a long-bucket batch should run in parallel
    through the executor — total wall time stays close to one batch, not
    two batches. This is the Opt 4 behavioural promise.
    """

    async def main() -> None:
        backend = FakeBackend(sleep_s=0.15)
        batcher = Batcher(backend, window_ms=20, max_batch_size=4, buckets=[32, 512])
        await batcher.start()
        try:
            short = "s" * 10  # bucket 32
            long_text = "l" * 200  # bucket 512
            t0 = time.perf_counter()
            await asyncio.gather(batcher.predict(short), batcher.predict(long_text))
            wall = time.perf_counter() - t0

            # Two serial batches would take ~0.30 s (2 x sleep). Concurrent
            # batches should take ~0.15 s + scheduling overhead. Pick a
            # generous threshold to avoid flakiness on a loaded CI box.
            assert wall < 0.25, f"buckets didn't parallelize: wall={wall:.3f}s"
            # And both batches were actually invoked.
            assert len(backend.batches) == 2
        finally:
            await batcher.stop()

    asyncio.run(main())
