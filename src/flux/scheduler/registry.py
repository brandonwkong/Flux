"""Worker registry for tracking compute nodes."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime
from typing import Callable, Optional

import structlog

from flux.models.worker import Worker, WorkerRegistration, WorkerStatus

logger = structlog.get_logger()


class WorkerRegistry:
    """Registry for tracking and managing workers.

    Features:
    - Worker registration and deregistration
    - Heartbeat monitoring
    - Automatic unhealthy detection
    - Draining support for graceful shutdown
    """

    def __init__(
        self,
        heartbeat_timeout_seconds: float = 30.0,
        on_worker_unhealthy: Optional[Callable[[Worker], None]] = None,
    ) -> None:
        self._workers: dict[str, Worker] = {}
        self._lock = threading.Lock()
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._on_worker_unhealthy = on_worker_unhealthy
        self._monitor_task: Optional[asyncio.Task] = None

    def register(self, registration: WorkerRegistration) -> Worker:
        """Register a new worker."""
        worker = Worker(
            name=registration.name,
            host=registration.host,
            port=registration.port,
            capacity=registration.capacity,
            available=registration.capacity.model_copy(),
        )

        with self._lock:
            self._workers[worker.id] = worker

        logger.info(
            "worker_registered",
            worker_id=worker.id,
            name=worker.name,
            capacity=worker.capacity.model_dump(),
        )

        return worker

    def deregister(self, worker_id: str) -> Optional[Worker]:
        """Remove a worker from the registry."""
        with self._lock:
            worker = self._workers.pop(worker_id, None)

        if worker:
            logger.info("worker_deregistered", worker_id=worker_id, name=worker.name)

        return worker

    def get(self, worker_id: str) -> Optional[Worker]:
        """Get a worker by ID."""
        with self._lock:
            return self._workers.get(worker_id)

    def get_healthy(self) -> list[Worker]:
        """Get all healthy workers."""
        with self._lock:
            return [w for w in self._workers.values() if w.status == WorkerStatus.HEALTHY]

    def get_all(self) -> list[Worker]:
        """Get all workers."""
        with self._lock:
            return list(self._workers.values())

    def heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat timestamp."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False

            worker.update_heartbeat()

            # Restore health if was unhealthy
            if worker.status == WorkerStatus.UNHEALTHY:
                worker.mark_healthy()
                logger.info("worker_restored", worker_id=worker_id)

            return True

    def mark_draining(self, worker_id: str) -> bool:
        """Mark a worker as draining (no new jobs)."""
        with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False

            worker.mark_draining()
            logger.info("worker_draining", worker_id=worker_id)
            return True

    def check_heartbeats(self) -> list[Worker]:
        """Check for workers that have missed heartbeats.

        Returns list of workers that became unhealthy.
        """
        now = datetime.utcnow()
        newly_unhealthy = []

        with self._lock:
            for worker in self._workers.values():
                if worker.status not in (WorkerStatus.HEALTHY, WorkerStatus.DRAINING):
                    continue

                if worker.seconds_since_heartbeat > self._heartbeat_timeout:
                    worker.mark_unhealthy()
                    newly_unhealthy.append(worker)
                    logger.warning(
                        "worker_unhealthy",
                        worker_id=worker.id,
                        last_heartbeat=worker.last_heartbeat.isoformat(),
                        timeout=self._heartbeat_timeout,
                    )

        # Notify callback for each unhealthy worker
        if self._on_worker_unhealthy:
            for worker in newly_unhealthy:
                self._on_worker_unhealthy(worker)

        return newly_unhealthy

    async def start_monitor(self, check_interval: float = 5.0) -> None:
        """Start background heartbeat monitoring."""
        async def monitor_loop():
            while True:
                self.check_heartbeats()
                await asyncio.sleep(check_interval)

        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("heartbeat_monitor_started", interval=check_interval)

    async def stop_monitor(self) -> None:
        """Stop background heartbeat monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("heartbeat_monitor_stopped")

    @property
    def worker_count(self) -> int:
        """Total number of registered workers."""
        with self._lock:
            return len(self._workers)

    @property
    def healthy_count(self) -> int:
        """Number of healthy workers."""
        with self._lock:
            return sum(1 for w in self._workers.values() if w.status == WorkerStatus.HEALTHY)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        with self._lock:
            workers = list(self._workers.values())

        return {
            "total_workers": len(workers),
            "healthy_workers": sum(1 for w in workers if w.status == WorkerStatus.HEALTHY),
            "unhealthy_workers": sum(1 for w in workers if w.status == WorkerStatus.UNHEALTHY),
            "draining_workers": sum(1 for w in workers if w.status == WorkerStatus.DRAINING),
            "total_capacity": {
                "cpu": sum(w.capacity.cpu for w in workers),
                "memory": sum(w.capacity.memory for w in workers),
                "gpu": sum(w.capacity.gpu for w in workers),
            },
            "total_available": {
                "cpu": sum(w.available.cpu for w in workers),
                "memory": sum(w.available.memory for w in workers),
                "gpu": sum(w.available.gpu for w in workers),
            },
        }
