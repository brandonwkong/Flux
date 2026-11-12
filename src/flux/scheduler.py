"""Scheduling policies: which pending requests does the engine run next?

Batching - how many requests fit in one forward pass - is the engine's
concern, in its own loop. Scheduling - which of the pending requests get
selected into that batch when more are waiting than fit - is this
module's concern, and it's deliberately separate: the same batch loop
can run under any of these policies unchanged.
"""

from __future__ import annotations

from enum import Enum

from flux.queue import QueuedRequest


class SchedulingPolicy(str, Enum):
    """Swappable orderings for picking the next batch from pending requests."""

    FIFO = "fifo"
    PRIORITY = "priority"
    EDF = "edf"  # earliest deadline first


def select_batch(
    pending: list[QueuedRequest],
    max_batch_size: int,
    policy: SchedulingPolicy,
) -> list[QueuedRequest]:
    """Pick up to max_batch_size requests from pending, ordered by policy."""
    if policy == SchedulingPolicy.FIFO:
        ordered = _fifo_order(pending)
    elif policy == SchedulingPolicy.PRIORITY:
        ordered = _priority_order(pending)
    elif policy == SchedulingPolicy.EDF:
        ordered = _edf_order(pending)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")

    return ordered[:max_batch_size]


def _fifo_order(pending: list[QueuedRequest]) -> list[QueuedRequest]:
    """Oldest request first - first come, first served."""
    return sorted(pending, key=lambda r: r.arrived_at)


def _priority_order(pending: list[QueuedRequest]) -> list[QueuedRequest]:
    """Highest priority first; ties broken by arrival order so same-priority requests don't starve."""
    return sorted(pending, key=lambda r: (-r.priority, r.arrived_at))


def _edf_order(pending: list[QueuedRequest]) -> list[QueuedRequest]:
    """Earliest absolute deadline first - the request closest to timing out runs next."""
    return sorted(pending, key=lambda r: r.arrived_at + r.deadline_ms / 1000)
