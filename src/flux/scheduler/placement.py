"""Placement engine for assigning inference batches to workers."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import structlog

from flux.models.job import Job
from flux.models.request import InferenceBatch
from flux.models.worker import Worker, WorkerStatus

logger = structlog.get_logger()


class PlacementStrategy(str, Enum):
    """Placement strategies."""

    FIRST_FIT = "first_fit"        # First worker that fits
    BEST_FIT = "best_fit"          # Minimize wasted resources
    WORST_FIT = "worst_fit"        # Maximize remaining resources (spread load)
    ROUND_ROBIN = "round_robin"    # Rotate through workers
    MODEL_AWARE = "model_aware"    # Prefer workers with model loaded (default for inference)


class PlacementEngine:
    """Engine for placing inference batches on workers.

    Implements model-aware routing and bin-packing for efficient GPU utilization.
    """

    def __init__(self, strategy: PlacementStrategy = PlacementStrategy.MODEL_AWARE) -> None:
        self.strategy = strategy
        self._round_robin_index = 0

    def find_worker_for_batch(
        self,
        batch: InferenceBatch,
        workers: list[Worker],
    ) -> Optional[Worker]:
        """Find best worker for an inference batch.

        Priority:
        1. Workers with model already loaded (warm) - lowest latency
        2. Workers with available GPU memory - can load model
        3. None if no capacity

        Args:
            batch: The inference batch to place
            workers: Available workers

        Returns:
            Best worker or None if no capacity
        """
        healthy = [w for w in workers if w.status == WorkerStatus.HEALTHY]
        if not healthy:
            return None

        if self.strategy == PlacementStrategy.MODEL_AWARE:
            return self._model_aware_placement(batch, healthy)
        elif self.strategy == PlacementStrategy.FIRST_FIT:
            return self._first_fit_inference(batch, healthy)
        elif self.strategy == PlacementStrategy.BEST_FIT:
            return self._best_fit_inference(batch, healthy)
        elif self.strategy == PlacementStrategy.WORST_FIT:
            return self._worst_fit_inference(batch, healthy)
        elif self.strategy == PlacementStrategy.ROUND_ROBIN:
            return self._round_robin_inference(healthy)
        else:
            return self._model_aware_placement(batch, healthy)

    def _model_aware_placement(
        self,
        batch: InferenceBatch,
        workers: list[Worker],
    ) -> Optional[Worker]:
        """Model-aware placement: prefer workers with model loaded.

        Strategy:
        1. Find workers with model already loaded (warm)
        2. Among warm workers, pick one with lowest queue depth
        3. If no warm workers, pick any with lowest queue depth (cold start)
        """
        model = batch.model

        # Warm workers (model already loaded)
        warm = [w for w in workers if w.has_model_loaded(model)]

        if warm:
            # Pick warm worker with lowest pending batches
            best = min(warm, key=lambda w: w.pending_batches)
            logger.debug(
                "warm_placement",
                model=model,
                worker=best.name,
                pending=best.pending_batches,
            )
            return best

        # Cold workers (need to load model)
        # Pick worker with most available GPU memory and lowest queue
        cold = sorted(
            workers,
            key=lambda w: (-w.gpu_memory_available_mb, w.pending_batches),
        )

        if cold:
            best = cold[0]
            logger.debug(
                "cold_placement",
                model=model,
                worker=best.name,
                gpu_available_mb=best.gpu_memory_available_mb,
            )
            return best

        return None

    def _first_fit_inference(
        self,
        batch: InferenceBatch,
        workers: list[Worker],
    ) -> Optional[Worker]:
        """Return first available worker."""
        return workers[0] if workers else None

    def _best_fit_inference(
        self,
        batch: InferenceBatch,
        workers: list[Worker],
    ) -> Optional[Worker]:
        """Return worker with lowest queue depth."""
        return min(workers, key=lambda w: w.pending_batches)

    def _worst_fit_inference(
        self,
        batch: InferenceBatch,
        workers: list[Worker],
    ) -> Optional[Worker]:
        """Return worker with most available resources (spread load)."""
        return max(
            workers,
            key=lambda w: w.gpu_memory_available_mb,
        )

    def _round_robin_inference(self, workers: list[Worker]) -> Optional[Worker]:
        """Rotate through workers."""
        if not workers:
            return None
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker

    # === Legacy job-based methods (kept for compatibility) ===

    def find_worker(
        self,
        job: Job,
        workers: list[Worker],
    ) -> Optional[Worker]:
        """Find best worker for a job based on placement strategy."""
        # Filter to healthy workers that can fit the job
        candidates = [w for w in workers if w.can_fit(job.resources)]

        if not candidates:
            return None

        if self.strategy == PlacementStrategy.FIRST_FIT:
            return self._first_fit(job, candidates)
        elif self.strategy == PlacementStrategy.BEST_FIT:
            return self._best_fit(job, candidates)
        elif self.strategy == PlacementStrategy.WORST_FIT:
            return self._worst_fit(job, candidates)
        elif self.strategy == PlacementStrategy.ROUND_ROBIN:
            return self._round_robin(job, candidates)
        else:
            return self._best_fit(job, candidates)

    def _first_fit(self, job: Job, candidates: list[Worker]) -> Worker:
        """Return first worker that fits."""
        return candidates[0]

    def _best_fit(self, job: Job, candidates: list[Worker]) -> Worker:
        """Return worker that minimizes wasted resources."""
        return min(
            candidates,
            key=lambda w: w.available.waste_if_allocated(job.resources, w.capacity),
        )

    def _worst_fit(self, job: Job, candidates: list[Worker]) -> Worker:
        """Return worker with most available resources."""
        return max(
            candidates,
            key=lambda w: (w.available.cpu + w.available.memory + w.available.gpu),
        )

    def _round_robin(self, job: Job, candidates: list[Worker]) -> Worker:
        """Rotate through workers."""
        worker = candidates[self._round_robin_index % len(candidates)]
        self._round_robin_index += 1
        return worker

    def find_preemption_candidate(
        self,
        high_priority_job: Job,
        workers: list[Worker],
        running_jobs: dict[str, Job],
    ) -> Optional[tuple[Worker, Job]]:
        """Find a low-priority job to preempt for a high-priority job."""
        for worker in workers:
            if worker.status != WorkerStatus.HEALTHY:
                continue

            for job_id in worker.current_jobs:
                running_job = running_jobs.get(job_id)
                if not running_job:
                    continue

                if running_job.priority >= high_priority_job.priority:
                    continue

                potential_available = worker.available + running_job.resources
                if high_priority_job.resources.fits_in(potential_available):
                    return (worker, running_job)

        return None


def calculate_cluster_utilization(workers: list[Worker]) -> float:
    """Calculate overall cluster resource utilization."""
    if not workers:
        return 0.0

    total_utilization = sum(w.utilization for w in workers)
    return total_utilization / len(workers)


def calculate_gpu_utilization(workers: list[Worker]) -> float:
    """Calculate overall GPU memory utilization."""
    if not workers:
        return 0.0

    total_gpu = sum(w.gpu_memory_total_mb for w in workers)
    if total_gpu == 0:
        return 0.0

    used_gpu = sum(w.gpu_memory_used_mb for w in workers)
    return used_gpu / total_gpu
