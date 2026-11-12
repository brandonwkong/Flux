"""Async request queue: each request carries a Future that resolves when the engine finishes it.

This decouples "a request arrived" from "a request was computed." The HTTP
handler that calls submit() just awaits the Future - it has no idea whether
the engine ran this request alone, batched with others, first, or last.
That decision belongs to the scheduler and the engine's batch loop, not here.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from uuid import uuid4

from flux.schemas import GenerateResponse


@dataclass
class QueuedRequest:
    """A generation request waiting for the engine, paired with the Future its caller is awaiting."""

    prompt: str
    max_new_tokens: int
    priority: int
    deadline_ms: int
    future: "asyncio.Future[GenerateResponse]"
    id: str = field(default_factory=lambda: str(uuid4()))
    arrived_at: float = field(default_factory=time.perf_counter)

    @property
    def age_ms(self) -> float:
        """How long this request has been waiting, in milliseconds."""
        return (time.perf_counter() - self.arrived_at) * 1000


class RequestQueue:
    """Holds requests the engine hasn't picked up yet.

    This is deliberately not asyncio.Queue: asyncio.Queue is strict FIFO,
    but FIFO is only one of the scheduling policies the engine can use.
    Instead this is an unordered pool - the scheduler picks which pending
    requests to pull next each time the engine forms a batch.
    """

    def __init__(self, max_depth: int = 1000) -> None:
        self._pending: list[QueuedRequest] = []
        self._max_depth = max_depth
        self._arrived = asyncio.Event()

    @property
    def depth(self) -> int:
        """Number of requests currently waiting."""
        return len(self._pending)

    async def submit(
        self,
        prompt: str,
        max_new_tokens: int,
        priority: int,
        deadline_ms: int,
    ) -> GenerateResponse:
        """Enqueue a request and wait for the engine to produce its result.

        This is the call a request makes to join the queue and block until
        the engine has run it - the Future is what makes that blocking wait
        possible without the caller polling anything.
        """
        if self.depth >= self._max_depth:
            raise RuntimeError("Queue full, server overloaded")

        future: asyncio.Future[GenerateResponse] = asyncio.get_event_loop().create_future()
        request = QueuedRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            priority=priority,
            deadline_ms=deadline_ms,
            future=future,
        )

        self._pending.append(request)
        self._arrived.set()

        return await future

    async def wait_for_arrival(self, timeout: float) -> None:
        """Block until at least one request is pending, or timeout elapses."""
        try:
            await asyncio.wait_for(self._arrived.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    def snapshot(self) -> list[QueuedRequest]:
        """All currently pending requests, in arrival order."""
        return list(self._pending)

    def remove(self, requests: list[QueuedRequest]) -> None:
        """Remove requests the engine has pulled out to run as a batch."""
        ids = {r.id for r in requests}
        self._pending = [r for r in self._pending if r.id not in ids]
        if not self._pending:
            self._arrived.clear()
