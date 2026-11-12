"""Tests for RequestQueue: the async pool where requests wait, each holding a Future."""

import asyncio
import time

import pytest

from flux.queue import RequestQueue
from flux.schemas import GenerateResponse


class TestRequestQueue:
    async def test_submit_blocks_until_a_future_is_resolved(self):
        queue = RequestQueue()

        task = asyncio.create_task(
            queue.submit(prompt="hi", max_new_tokens=4, priority=5, deadline_ms=5000)
        )
        await asyncio.sleep(0.01)  # let it enqueue

        assert queue.depth == 1
        pending = queue.snapshot()
        assert len(pending) == 1

        pending[0].future.set_result(
            GenerateResponse(text="hello", queue_time_ms=1.0, compute_time_ms=2.0, batch_size=1)
        )
        queue.remove(pending)

        result = await task
        assert result.text == "hello"
        assert queue.depth == 0

    async def test_backpressure_rejects_when_full(self):
        queue = RequestQueue(max_depth=1)
        first = asyncio.create_task(
            queue.submit(prompt="a", max_new_tokens=4, priority=5, deadline_ms=5000)
        )
        await asyncio.sleep(0.01)

        with pytest.raises(RuntimeError):
            await queue.submit(prompt="b", max_new_tokens=4, priority=5, deadline_ms=5000)

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    async def test_wait_for_arrival_returns_once_something_is_pending(self):
        queue = RequestQueue()
        submitted = asyncio.create_task(
            queue.submit(prompt="x", max_new_tokens=4, priority=5, deadline_ms=5000)
        )

        await queue.wait_for_arrival(timeout=1.0)
        assert queue.depth == 1

        submitted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submitted

    async def test_wait_for_arrival_times_out_when_empty(self):
        queue = RequestQueue()

        start = time.perf_counter()
        await queue.wait_for_arrival(timeout=0.05)
        elapsed = time.perf_counter() - start

        assert elapsed >= 0.04
