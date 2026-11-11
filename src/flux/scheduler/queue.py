"""Priority queue for job scheduling."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from typing import Optional

from flux.models.job import Job, JobStatus


@dataclass(order=True)
class PriorityItem:
    """Wrapper for heap queue ordering.

    Uses negative priority so higher priority = earlier in heap.
    """

    priority: float
    submission_order: int
    job: Job = field(compare=False)


class JobQueue:
    """Thread-safe priority queue for jobs.

    Features:
    - Priority-based ordering (higher priority first)
    - FIFO within same priority (submission order)
    - Aging: jobs gain priority over time
    - O(log n) push/pop operations
    """

    def __init__(self) -> None:
        self._heap: list[PriorityItem] = []
        self._job_map: dict[str, Job] = {}  # For O(1) lookup by ID
        self._counter = 0  # For FIFO within same priority
        self._lock = threading.Lock()

    def push(self, job: Job) -> None:
        """Add a job to the queue."""
        with self._lock:
            # Negative priority so higher priority comes first
            item = PriorityItem(
                priority=-job.effective_priority(),
                submission_order=self._counter,
                job=job,
            )
            self._counter += 1
            heapq.heappush(self._heap, item)
            self._job_map[job.id] = job

    def pop(self) -> Optional[Job]:
        """Remove and return highest priority job."""
        with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                job = item.job

                # Skip if job was cancelled or already processed
                if job.id not in self._job_map:
                    continue
                if job.status != JobStatus.PENDING:
                    continue

                del self._job_map[job.id]
                return job

            return None

    def peek(self) -> Optional[Job]:
        """Return highest priority job without removing."""
        with self._lock:
            for item in self._heap:
                if item.job.id in self._job_map and item.job.status == JobStatus.PENDING:
                    return item.job
            return None

    def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._lock:
            return self._job_map.get(job_id)

    def remove(self, job_id: str) -> Optional[Job]:
        """Remove a job from the queue by ID."""
        with self._lock:
            job = self._job_map.pop(job_id, None)
            if job:
                job.status = JobStatus.CANCELLED
            return job

    def requeue(self, job: Job) -> None:
        """Re-add a preempted/failed job to the queue."""
        job.status = JobStatus.PENDING
        job.worker_id = None
        self.push(job)

    def refresh_priorities(self) -> None:
        """Rebuild heap with updated priorities (for aging).

        Call periodically to apply aging boosts.
        """
        with self._lock:
            pending_jobs = [
                job for job in self._job_map.values() if job.status == JobStatus.PENDING
            ]

            self._heap.clear()
            self._counter = 0

            for job in pending_jobs:
                item = PriorityItem(
                    priority=-job.effective_priority(),
                    submission_order=self._counter,
                    job=job,
                )
                self._counter += 1
                heapq.heappush(self._heap, item)

    def __len__(self) -> int:
        """Return number of pending jobs."""
        with self._lock:
            return sum(1 for j in self._job_map.values() if j.status == JobStatus.PENDING)

    @property
    def pending_count(self) -> int:
        """Number of pending jobs."""
        return len(self)

    def list_pending(self, limit: int = 100) -> list[Job]:
        """List pending jobs in priority order."""
        with self._lock:
            pending = [
                j for j in self._job_map.values() if j.status == JobStatus.PENDING
            ]
            pending.sort(key=lambda j: -j.effective_priority())
            return pending[:limit]
