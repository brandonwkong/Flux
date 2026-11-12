"""Tests for scheduling policies: who runs next when multiple requests are pending."""

import asyncio

from flux.queue import QueuedRequest
from flux.scheduler import SchedulingPolicy, select_batch


async def _request(
    priority: int = 5,
    deadline_ms: int = 5000,
    arrived_at: float | None = None,
) -> QueuedRequest:
    request = QueuedRequest(
        prompt="hello",
        max_new_tokens=8,
        priority=priority,
        deadline_ms=deadline_ms,
        future=asyncio.get_running_loop().create_future(),
    )
    if arrived_at is not None:
        request.arrived_at = arrived_at
    return request


class TestFIFO:
    async def test_oldest_first(self):
        first = await _request(arrived_at=1.0)
        second = await _request(arrived_at=2.0)
        third = await _request(arrived_at=3.0)

        ordered = select_batch([third, first, second], max_batch_size=3, policy=SchedulingPolicy.FIFO)

        assert ordered == [first, second, third]


class TestPriority:
    async def test_highest_priority_first(self):
        low = await _request(priority=1, arrived_at=1.0)
        high = await _request(priority=10, arrived_at=2.0)

        ordered = select_batch([low, high], max_batch_size=2, policy=SchedulingPolicy.PRIORITY)

        assert ordered == [high, low]

    async def test_ties_broken_by_arrival_order(self):
        """Same priority shouldn't mean random order - earlier arrival wins, so nothing starves."""
        first = await _request(priority=5, arrived_at=1.0)
        second = await _request(priority=5, arrived_at=2.0)

        ordered = select_batch([second, first], max_batch_size=2, policy=SchedulingPolicy.PRIORITY)

        assert ordered == [first, second]


class TestEDF:
    async def test_earliest_absolute_deadline_first(self):
        """Arrived later but with a tight deadline should still run before a relaxed early arrival."""
        urgent = await _request(arrived_at=5.0, deadline_ms=100)
        relaxed = await _request(arrived_at=1.0, deadline_ms=10000)

        ordered = select_batch([relaxed, urgent], max_batch_size=2, policy=SchedulingPolicy.EDF)

        assert ordered == [urgent, relaxed]


class TestSelectBatch:
    async def test_truncates_to_max_batch_size(self):
        requests = [await _request(arrived_at=float(i)) for i in range(5)]

        batch = select_batch(requests, max_batch_size=2, policy=SchedulingPolicy.FIFO)

        assert batch == requests[:2]
