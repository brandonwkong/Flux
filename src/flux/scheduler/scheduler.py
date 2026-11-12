"""Main scheduler service that orchestrates job scheduling."""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import httpx
import structlog
from prometheus_client import Counter, Gauge, Histogram

from flux.models.job import Job, JobStatus, JobSubmission
from flux.models.request import InferenceBatch
from flux.models.worker import Worker, WorkerRegistration, WorkerStatus
from flux.scheduler.placement import PlacementEngine, PlacementStrategy, calculate_cluster_utilization
from flux.scheduler.queue import JobQueue
from flux.scheduler.registry import WorkerRegistry

logger = structlog.get_logger()

# Prometheus metrics
JOBS_SUBMITTED = Counter("flux_jobs_submitted_total", "Total jobs submitted")
JOBS_COMPLETED = Counter("flux_jobs_completed_total", "Total jobs completed")
JOBS_FAILED = Counter("flux_jobs_failed_total", "Total jobs failed")
JOBS_PREEMPTED = Counter("flux_jobs_preempted_total", "Total jobs preempted")

QUEUE_DEPTH = Gauge("flux_queue_depth", "Current queue depth")
CLUSTER_UTILIZATION = Gauge("flux_cluster_utilization", "Cluster utilization ratio")
WORKER_COUNT = Gauge("flux_worker_count", "Number of workers", ["status"])

SCHEDULING_LATENCY = Histogram(
    "flux_scheduling_latency_seconds",
    "Time to schedule a job",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
JOB_WAIT_TIME = Histogram(
    "flux_job_wait_time_seconds",
    "Time jobs spend waiting in queue",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)

# Inference metrics
BATCHES_DISPATCHED = Counter(
    "flux_batches_dispatched_total",
    "Total batches dispatched to workers",
    ["model"],
)
BATCH_LATENCY = Histogram(
    "flux_batch_latency_seconds",
    "Batch dispatch and inference latency",
    ["model"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)


class Scheduler:
    """Central scheduler coordinating job placement and execution.

    Responsibilities:
    - Accept job submissions
    - Maintain job queue with priorities
    - Place jobs on workers using bin-packing
    - Handle preemption for high-priority jobs
    - Recover from worker failures
    - Track metrics
    """

    def __init__(
        self,
        placement_strategy: PlacementStrategy = PlacementStrategy.MODEL_AWARE,
        heartbeat_timeout: float = 30.0,
        scheduling_interval: float = 0.1,
        enable_preemption: bool = True,
    ) -> None:
        self.queue = JobQueue()
        self.registry = WorkerRegistry(
            heartbeat_timeout_seconds=heartbeat_timeout,
            on_worker_unhealthy=self._handle_worker_failure,
        )
        self.placement = PlacementEngine(strategy=placement_strategy)

        self._running_jobs: dict[str, Job] = {}  # job_id -> Job
        self._all_jobs: dict[str, Job] = {}      # All jobs ever submitted

        self._scheduling_interval = scheduling_interval
        self._enable_preemption = enable_preemption
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

    # === Job Operations ===

    def submit_job(self, submission: JobSubmission) -> Job:
        """Submit a new job for scheduling."""
        job = Job(
            name=submission.name,
            command=submission.command,
            resources=submission.resources,
            priority=submission.priority,
            max_retries=submission.max_retries,
        )

        self._all_jobs[job.id] = job
        self.queue.push(job)

        JOBS_SUBMITTED.inc()
        QUEUE_DEPTH.set(self.queue.pending_count)

        logger.info(
            "job_submitted",
            job_id=job.id,
            name=job.name,
            priority=job.priority,
            resources=job.resources.model_dump(),
        )

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._all_jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        job = self._all_jobs.get(job_id)
        if not job:
            return False

        if job.status == JobStatus.PENDING:
            self.queue.remove(job_id)
            job.status = JobStatus.CANCELLED
            QUEUE_DEPTH.set(self.queue.pending_count)
            logger.info("job_cancelled", job_id=job_id)
            return True

        return False

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._all_jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by submission time (newest first)
        jobs.sort(key=lambda j: j.submitted_at, reverse=True)

        return jobs[:limit]

    # === Worker Operations ===

    def register_worker(self, registration: WorkerRegistration) -> Worker:
        """Register a new worker."""
        worker = self.registry.register(registration)
        self._update_worker_metrics()
        return worker

    def deregister_worker(self, worker_id: str) -> Optional[Worker]:
        """Deregister a worker."""
        worker = self.registry.deregister(worker_id)
        if worker:
            self._handle_worker_failure(worker)
        self._update_worker_metrics()
        return worker

    def worker_heartbeat(self, worker_id: str) -> bool:
        """Process worker heartbeat."""
        return self.registry.heartbeat(worker_id)

    def drain_worker(self, worker_id: str) -> bool:
        """Mark worker as draining."""
        return self.registry.mark_draining(worker_id)

    # === Scheduling Loop ===

    async def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        await self.registry.start_monitor()
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        logger.info("scheduler_started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        await self.registry.stop_monitor()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("scheduler_stopped")

    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            try:
                await self._schedule_pending_jobs()
                self._update_metrics()
            except Exception as e:
                logger.error("scheduling_error", error=str(e))

            await asyncio.sleep(self._scheduling_interval)

    async def _schedule_pending_jobs(self) -> None:
        """Try to schedule all pending jobs."""
        workers = self.registry.get_healthy()

        while True:
            job = self.queue.peek()
            if not job:
                break

            start_time = time.perf_counter()

            # Try normal placement
            worker = self.placement.find_worker(job, workers)

            if worker:
                # Pop from queue and schedule
                self.queue.pop()
                await self._assign_job_to_worker(job, worker)

                elapsed = time.perf_counter() - start_time
                SCHEDULING_LATENCY.observe(elapsed)
                JOB_WAIT_TIME.observe(job.wait_time_seconds)

            elif self._enable_preemption and job.priority >= 10:
                # Try preemption for high priority jobs
                result = self.placement.find_preemption_candidate(
                    job, workers, self._running_jobs
                )

                if result:
                    self.queue.pop()
                    target_worker, preempt_job = result
                    await self._preempt_and_schedule(job, target_worker, preempt_job)

                    elapsed = time.perf_counter() - start_time
                    SCHEDULING_LATENCY.observe(elapsed)
                else:
                    # Cannot schedule, stop trying
                    break
            else:
                # No capacity, stop trying
                break

    async def _assign_job_to_worker(self, job: Job, worker: Worker) -> None:
        """Assign a job to a worker."""
        job.mark_scheduled(worker.id)
        worker.allocate(job.id, job.resources)
        self._running_jobs[job.id] = job

        logger.info(
            "job_scheduled",
            job_id=job.id,
            worker_id=worker.id,
            worker_name=worker.name,
        )

        # Dispatch to worker (fire and forget for now)
        asyncio.create_task(self._dispatch_to_worker(job, worker))

    async def _dispatch_to_worker(self, job: Job, worker: Worker) -> None:
        """Send job to worker for execution."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{worker.host}:{worker.port}/execute",
                    json={
                        "job_id": job.id,
                        "command": job.command,
                        "resources": job.resources.model_dump(),
                    },
                    timeout=10.0,
                )

                if response.status_code == 200:
                    job.mark_running()
                    logger.info("job_dispatched", job_id=job.id, worker_id=worker.id)
                else:
                    logger.error(
                        "job_dispatch_failed",
                        job_id=job.id,
                        status=response.status_code,
                    )
                    await self._handle_job_failure(job, "Dispatch failed")

        except Exception as e:
            logger.error("job_dispatch_error", job_id=job.id, error=str(e))
            await self._handle_job_failure(job, str(e))

    # === Inference Batch Dispatch (Sync) ===

    async def dispatch_batch(
        self,
        batch: InferenceBatch,
        worker: Worker,
    ) -> list:
        """Dispatch an inference batch to a worker and wait for results.

        This is a synchronous operation - we wait for the worker to return
        the inference results before returning.

        Args:
            batch: The inference batch to process
            worker: The worker to process it on

        Returns:
            List of predictions from the worker
        """
        start = time.perf_counter()

        worker.pending_batches += 1

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://{worker.host}:{worker.port}/infer",
                    json={
                        "batch_id": batch.id,
                        "model": batch.model,
                        "inputs": batch.combined_inputs,
                    },
                    timeout=30.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    predictions = result.get("predictions", [])

                    elapsed = time.perf_counter() - start
                    BATCH_LATENCY.labels(model=batch.model).observe(elapsed)
                    BATCHES_DISPATCHED.labels(model=batch.model).inc()

                    # Update worker stats
                    worker.total_inferences += batch.total_inputs

                    # Mark model as loaded on worker if not already
                    if not worker.has_model_loaded(batch.model):
                        worker.load_model(batch.model)

                    logger.debug(
                        "batch_dispatched",
                        batch_id=batch.id,
                        model=batch.model,
                        worker=worker.name,
                        latency_ms=elapsed * 1000,
                    )

                    return predictions
                else:
                    error = f"Worker returned {response.status_code}"
                    logger.error("batch_dispatch_failed", batch_id=batch.id, error=error)
                    raise RuntimeError(error)

        except httpx.TimeoutException:
            logger.error("batch_dispatch_timeout", batch_id=batch.id)
            raise RuntimeError("Worker timeout")

        except Exception as e:
            logger.error("batch_dispatch_error", batch_id=batch.id, error=str(e))
            raise

        finally:
            worker.pending_batches -= 1

    async def _preempt_and_schedule(
        self,
        high_priority_job: Job,
        worker: Worker,
        preempt_job: Job,
    ) -> None:
        """Preempt a job and schedule a higher priority one."""
        logger.info(
            "preempting_job",
            preempt_job_id=preempt_job.id,
            high_priority_job_id=high_priority_job.id,
            worker_id=worker.id,
        )

        # Kill the preempted job
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"http://{worker.host}:{worker.port}/kill",
                    json={"job_id": preempt_job.id},
                    timeout=5.0,
                )
        except Exception as e:
            logger.warning("preempt_kill_failed", job_id=preempt_job.id, error=str(e))

        # Update state
        preempt_job.mark_preempted()
        worker.release(preempt_job.id, preempt_job.resources)
        self._running_jobs.pop(preempt_job.id, None)

        # Requeue preempted job
        self.queue.requeue(preempt_job)
        JOBS_PREEMPTED.inc()

        # Schedule high priority job
        await self._assign_job_to_worker(high_priority_job, worker)

    # === Job Completion Callbacks ===

    async def job_completed(
        self,
        job_id: str,
        exit_code: int = 0,
        result: Optional[dict] = None,
    ) -> None:
        """Handle job completion from worker."""
        job = self._running_jobs.pop(job_id, None)
        if not job:
            logger.warning("unknown_job_completed", job_id=job_id)
            return

        job.mark_completed(exit_code, result)

        # Release resources
        worker = self.registry.get(job.worker_id) if job.worker_id else None
        if worker:
            worker.release(job_id, job.resources)
            worker.jobs_completed += 1

        JOBS_COMPLETED.inc()
        logger.info("job_completed", job_id=job_id, exit_code=exit_code)

    async def job_failed(self, job_id: str, error: str) -> None:
        """Handle job failure from worker."""
        await self._handle_job_failure(
            self._running_jobs.get(job_id) or self._all_jobs.get(job_id),
            error,
        )

    async def _handle_job_failure(self, job: Optional[Job], error: str) -> None:
        """Handle a failed job - retry or mark failed."""
        if not job:
            return

        self._running_jobs.pop(job.id, None)

        # Release resources
        worker = self.registry.get(job.worker_id) if job.worker_id else None
        if worker:
            worker.release(job.id, job.resources)
            worker.jobs_failed += 1

        if job.can_retry():
            job.retry_count += 1
            self.queue.requeue(job)
            logger.info("job_retrying", job_id=job.id, retry=job.retry_count)
        else:
            job.mark_failed(error)
            JOBS_FAILED.inc()
            logger.error("job_failed", job_id=job.id, error=error)

    def _handle_worker_failure(self, worker: Worker) -> None:
        """Handle a failed worker - reschedule its jobs."""
        logger.warning(
            "handling_worker_failure",
            worker_id=worker.id,
            jobs=worker.current_jobs,
        )

        for job_id in list(worker.current_jobs):
            job = self._running_jobs.pop(job_id, None)
            if job:
                job.retry_count += 1
                self.queue.requeue(job)
                logger.info("job_rescheduled", job_id=job_id)

        worker.current_jobs.clear()

    # === Metrics ===

    def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        QUEUE_DEPTH.set(self.queue.pending_count)
        CLUSTER_UTILIZATION.set(
            calculate_cluster_utilization(self.registry.get_all())
        )

    def _update_worker_metrics(self) -> None:
        """Update worker-related metrics."""
        stats = self.registry.get_stats()
        WORKER_COUNT.labels(status="healthy").set(stats["healthy_workers"])
        WORKER_COUNT.labels(status="unhealthy").set(stats["unhealthy_workers"])
        WORKER_COUNT.labels(status="draining").set(stats["draining_workers"])

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            "queue": {
                "pending": self.queue.pending_count,
            },
            "jobs": {
                "total": len(self._all_jobs),
                "running": len(self._running_jobs),
                "completed": sum(
                    1 for j in self._all_jobs.values() if j.status == JobStatus.COMPLETED
                ),
                "failed": sum(
                    1 for j in self._all_jobs.values() if j.status == JobStatus.FAILED
                ),
            },
            "workers": self.registry.get_stats(),
            "cluster_utilization": calculate_cluster_utilization(self.registry.get_all()),
        }
