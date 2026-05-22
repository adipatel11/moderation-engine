"""Batcher unit tests.

The batcher is exercised against an in-memory fake backend so the tests stay
fast and don't depend on ONNX / model weights. Tests use `asyncio.run` rather
than pytest-asyncio to avoid the extra dev dependency.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from moderation_engine.batcher import Batcher


class FakeBackend:
    """Records every batch it sees; returns a dict mirroring the input text."""

    def __init__(self, sleep_s: float = 0.0, raise_exc: BaseException | None = None) -> None:
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


def test_concurrent_requests_get_coalesced() -> None:
    async def main() -> None:
        # Long-ish inference so all concurrent callers queue before the worker
        # has a chance to start the first batch.
        backend = FakeBackend(sleep_s=0.05)
        batcher = Batcher(backend, window_ms=20, max_batch_size=8)
        await batcher.start()
        try:
            texts = ["a", "bb", "ccc", "dddd", "eeeee"]
            results = await asyncio.gather(*(batcher.predict(t) for t in texts))
            assert [r["echo"] for r in results] == [1, 2, 3, 4, 5]
            # At least one batch must have contained > 1 item — otherwise
            # batching didn't actually happen.
            assert max(len(b) for b in backend.batches) >= 2
            # Every input must show up exactly once across all batches.
            flattened = [t for b in backend.batches for t in b]
            assert sorted(flattened) == sorted(texts)
        finally:
            await batcher.stop()

    asyncio.run(main())


def test_max_batch_size_caps_inference() -> None:
    async def main() -> None:
        backend = FakeBackend(sleep_s=0.05)
        batcher = Batcher(backend, window_ms=50, max_batch_size=3)
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
        # No worker task should have been created.
        try:
            await asyncio.gather(*(batcher.predict(t) for t in ["x", "y", "z"]))
            # Three independent size-1 batches, in some order.
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
        batcher = Batcher(backend, window_ms=20, max_batch_size=8)
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
