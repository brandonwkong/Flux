"""FastAPI application for TaskForge ML inference platform."""

from contextlib import asynccontextmanager
from typing import Any, Optional
import time
import os

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest, Counter, Histogram, Gauge

from flux.models.job import Job, JobStatus, JobSubmission
from flux.models.request import (
    PredictRequest,
    PredictResponse,
    InferenceRequest,
    InferenceBatch,
    ModelInfo,
)
from flux.models.worker import Worker, WorkerRegistration
from flux.batcher.batcher import Batcher, BatcherConfig
from flux.scheduler.scheduler import Scheduler

logger = structlog.get_logger()

# Metrics
REQUEST_LATENCY = Histogram(
    "flux_request_latency_seconds",
    "End-to-end request latency",
    ["model"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
REQUESTS_TOTAL = Counter(
    "flux_requests_total",
    "Total requests",
    ["model", "status"],
)
QUEUE_DEPTH = Gauge(
    "flux_queue_depth",
    "Pending requests in queue",
)

# Global instances
scheduler: Optional[Scheduler] = None
batcher: Optional[Batcher] = None

# Configuration from environment
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))
BATCH_TIMEOUT_MS = float(os.environ.get("BATCH_TIMEOUT_MS", "10"))
MAX_QUEUE_DEPTH = int(os.environ.get("MAX_QUEUE_DEPTH", "1000"))


async def process_batch(batch: InferenceBatch) -> list[Any]:
    """Process a batch through the scheduler to a worker.

    This function is called by the batcher when a batch is ready.
    """
    if not scheduler:
        raise RuntimeError("Scheduler not initialized")

    # Find a worker for this batch
    worker = scheduler.placement.find_worker_for_batch(
        batch, scheduler.registry.get_healthy()
    )

    if not worker:
        raise RuntimeError("No workers available")

    # Dispatch to worker and get results
    results = await scheduler.dispatch_batch(batch, worker)
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global scheduler, batcher

    # Startup
    scheduler = Scheduler()
    await scheduler.start()

    batcher_config = BatcherConfig(
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_ms=BATCH_TIMEOUT_MS,
        max_queue_depth=MAX_QUEUE_DEPTH,
    )
    batcher = Batcher(config=batcher_config, process_batch=process_batch)
    await batcher.start()

    logger.info(
        "flux_started",
        max_batch_size=MAX_BATCH_SIZE,
        batch_timeout_ms=BATCH_TIMEOUT_MS,
    )

    yield

    # Shutdown
    if batcher:
        await batcher.stop()
    if scheduler:
        await scheduler.stop()
    logger.info("flux_stopped")


app = FastAPI(
    title="TaskForge",
    description="Distributed ML Inference Platform",
    version="0.1.0",
    lifespan=lifespan,
)


# === Inference API (Primary) ===


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Submit an inference request.

    This is the primary endpoint. Requests are batched for efficiency,
    then routed to workers for inference.

    Args:
        request: Prediction request with model name and inputs

    Returns:
        Predictions for the inputs
    """
    if not batcher:
        raise HTTPException(status_code=503, detail="Service not ready")

    start = time.perf_counter()
    model = request.model

    try:
        # Check queue depth for backpressure
        if batcher.queue_depth >= MAX_QUEUE_DEPTH:
            REQUESTS_TOTAL.labels(model=model, status="rejected").inc()
            raise HTTPException(
                status_code=503,
                detail="Server overloaded, please retry later",
            )

        # Create internal request
        inference_req = InferenceRequest(
            model=model,
            inputs=request.inputs,
            priority=request.priority,
            timeout_ms=request.timeout_ms or 5000,
        )

        # Submit to batcher and wait for result
        predictions = await batcher.submit(inference_req)

        elapsed_ms = (time.perf_counter() - start) * 1000
        REQUEST_LATENCY.labels(model=model).observe(elapsed_ms / 1000)
        REQUESTS_TOTAL.labels(model=model, status="success").inc()

        return PredictResponse(
            predictions=predictions,
            latency_ms=elapsed_ms,
            batch_size=len(request.inputs),
            model=model,
        )

    except HTTPException:
        raise
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        REQUESTS_TOTAL.labels(model=model, status="timeout").inc()
        raise HTTPException(
            status_code=504,
            detail=f"Request timed out after {elapsed_ms:.0f}ms",
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        REQUESTS_TOTAL.labels(model=model, status="error").inc()
        logger.error("predict_error", model=model, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# === Health & Monitoring ===


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/ready")
async def readiness_check() -> dict:
    """Readiness check - are we ready to serve traffic?"""
    if not scheduler or not batcher:
        raise HTTPException(status_code=503, detail="Service not ready")

    healthy_workers = len(scheduler.registry.get_healthy())
    if healthy_workers == 0:
        raise HTTPException(status_code=503, detail="No healthy workers")

    return {
        "status": "ready",
        "healthy_workers": healthy_workers,
    }


@app.get("/stats")
async def get_stats() -> dict:
    """Get system statistics."""
    if not scheduler or not batcher:
        raise HTTPException(status_code=503, detail="Service not ready")

    stats = scheduler.get_stats()
    stats["batcher"] = {
        "queue_depth": batcher.queue_depth,
        "max_batch_size": MAX_BATCH_SIZE,
        "batch_timeout_ms": BATCH_TIMEOUT_MS,
    }
    return stats


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# === Models ===


@app.get("/models")
async def list_models() -> list[str]:
    """List available models across all workers."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Collect models from all workers
    models = set()
    for worker in scheduler.registry.get_all():
        models.update(worker.loaded_models)

    return sorted(models)


@app.get("/models/{model_name}")
async def get_model_info(model_name: str) -> dict:
    """Get information about a specific model."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Find workers with this model loaded
    workers_with_model = [
        w for w in scheduler.registry.get_healthy()
        if w.has_model_loaded(model_name)
    ]

    return {
        "name": model_name,
        "loaded_on_workers": len(workers_with_model),
        "workers": [w.name for w in workers_with_model],
    }


# === Workers ===


@app.post("/workers", status_code=201)
async def register_worker(registration: WorkerRegistration) -> Worker:
    """Register a new worker."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")
    return scheduler.register_worker(registration)


@app.get("/workers")
async def list_workers() -> list[Worker]:
    """List all registered workers."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")
    return scheduler.registry.get_all()


@app.get("/workers/{worker_id}")
async def get_worker(worker_id: str) -> Worker:
    """Get a specific worker by ID."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    worker = scheduler.registry.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")
    return worker


@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str) -> dict:
    """Receive heartbeat from worker."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    if scheduler.worker_heartbeat(worker_id):
        return {"status": "ok"}
    else:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")


@app.post("/workers/{worker_id}/drain")
async def drain_worker(worker_id: str) -> dict:
    """Mark worker as draining (no new requests)."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    if scheduler.drain_worker(worker_id):
        return {"status": "draining", "worker_id": worker_id}
    else:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")


@app.delete("/workers/{worker_id}")
async def deregister_worker(worker_id: str) -> dict:
    """Deregister a worker."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    worker = scheduler.deregister_worker(worker_id)
    if worker:
        return {"status": "deregistered", "worker_id": worker_id}
    else:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")


# === Legacy Job API (kept for compatibility) ===


@app.post("/jobs", status_code=201)
async def submit_job(submission: JobSubmission) -> Job:
    """Submit a new job for scheduling (legacy)."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")
    return scheduler.submit_job(submission)


@app.get("/jobs")
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[Job]:
    """List jobs, optionally filtered by status (legacy)."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")
    return scheduler.list_jobs(status=status, limit=limit)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> Job:
    """Get a specific job by ID (legacy)."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    job = scheduler.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    """Cancel a pending job (legacy)."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    if scheduler.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job {job_id} - not found or not pending",
        )


# === Callbacks (from workers) ===


@app.post("/callbacks/completed")
async def job_completed_callback(
    job_id: str,
    exit_code: int = 0,
    result: Optional[dict] = None,
) -> dict:
    """Callback from worker when job completes."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    await scheduler.job_completed(job_id, exit_code, result)
    return {"status": "ok"}


@app.post("/callbacks/failed")
async def job_failed_callback(job_id: str, error: str) -> dict:
    """Callback from worker when job fails."""
    if not scheduler:
        raise HTTPException(status_code=503, detail="Service not ready")

    await scheduler.job_failed(job_id, error)
    return {"status": "ok"}


# Need asyncio for TimeoutError
import asyncio


def run() -> None:
    """Run the API server."""
    uvicorn.run(
        "flux.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    run()
