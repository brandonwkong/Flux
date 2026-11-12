"""Worker service that executes jobs."""

from __future__ import annotations

import asyncio
import os
import socket
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from flux.models.resources import Resources
from flux.worker.executor import JobExecutor, InferenceExecutor

logger = structlog.get_logger()


# === Request Models ===


class InferRequest(BaseModel):
    """Request to run inference on a batch."""

    batch_id: str
    model: str
    inputs: list[list[float]]


class ExecuteRequest(BaseModel):
    """Request to execute a job (legacy)."""

    job_id: str
    command: str
    resources: dict


class KillRequest(BaseModel):
    """Request to kill a job."""

    job_id: str


class WorkerConfig(BaseModel):
    """Worker configuration."""

    name: str = socket.gethostname()
    host: str = "localhost"
    port: int = 8001
    scheduler_url: str = "http://localhost:8000"
    cpu: float = float(os.cpu_count() or 4)
    memory: int = 4096  # MB
    gpu: int = 0
    heartbeat_interval: float = 10.0


# Global state
config: WorkerConfig = WorkerConfig()
worker_id: Optional[str] = None
executor: JobExecutor = JobExecutor()  # Legacy job executor
inference_executor: InferenceExecutor = InferenceExecutor()  # ML inference executor
heartbeat_task: Optional[asyncio.Task] = None


async def register_with_scheduler() -> str:
    """Register this worker with the scheduler."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{config.scheduler_url}/workers",
            json={
                "name": config.name,
                "host": config.host,
                "port": config.port,
                "capacity": {
                    "cpu": config.cpu,
                    "memory": config.memory,
                    "gpu": config.gpu,
                },
            },
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["id"]


async def send_heartbeat() -> None:
    """Send heartbeat to scheduler."""
    if not worker_id:
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{config.scheduler_url}/workers/{worker_id}/heartbeat",
                timeout=5.0,
            )
    except Exception as e:
        logger.warning("heartbeat_failed", error=str(e))


async def heartbeat_loop() -> None:
    """Background heartbeat loop."""
    while True:
        await send_heartbeat()
        await asyncio.sleep(config.heartbeat_interval)


async def notify_job_completed(
    job_id: str,
    exit_code: int,
    result: Optional[dict] = None,
) -> None:
    """Notify scheduler that job completed."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{config.scheduler_url}/callbacks/completed",
                params={
                    "job_id": job_id,
                    "exit_code": exit_code,
                },
                json=result,
                timeout=10.0,
            )
    except Exception as e:
        logger.error("notify_completed_failed", job_id=job_id, error=str(e))


async def notify_job_failed(job_id: str, error: str) -> None:
    """Notify scheduler that job failed."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{config.scheduler_url}/callbacks/failed",
                params={
                    "job_id": job_id,
                    "error": error,
                },
                timeout=10.0,
            )
    except Exception as e:
        logger.error("notify_failed_failed", job_id=job_id, error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global worker_id, heartbeat_task

    # Register with scheduler
    try:
        worker_id = await register_with_scheduler()
        logger.info("worker_registered", worker_id=worker_id)
    except Exception as e:
        logger.error("registration_failed", error=str(e))
        raise

    # Start heartbeat
    heartbeat_task = asyncio.create_task(heartbeat_loop())

    yield

    # Stop heartbeat
    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    logger.info("worker_stopped")


app = FastAPI(
    title="Flux Worker",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {
        "status": "healthy",
        "worker_id": worker_id,
        "running_jobs": executor.running_count,
        "loaded_models": inference_executor.loaded_models,
    }


# === Inference API (Primary) ===


@app.post("/infer")
async def infer(request: InferRequest) -> dict:
    """Run inference on a batch of inputs.

    This is the primary endpoint for ML inference. The scheduler
    sends batches here after collecting requests from multiple clients.

    Args:
        request: Batch inference request with model name and inputs

    Returns:
        Predictions for all inputs in the batch
    """
    logger.info(
        "inference_request",
        batch_id=request.batch_id,
        model=request.model,
        batch_size=len(request.inputs),
    )

    result = await inference_executor.infer(
        batch_id=request.batch_id,
        model_name=request.model,
        inputs=request.inputs,
    )

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    return {
        "batch_id": result.batch_id,
        "predictions": result.predictions,
        "latency_ms": result.latency_ms,
        "model": result.model,
        "batch_size": result.batch_size,
    }


@app.get("/models")
async def list_models() -> dict:
    """List models loaded on this worker."""
    return {
        "models": inference_executor.loaded_models,
        "max_loaded": inference_executor.max_loaded_models,
    }


@app.post("/models/{model_name}/load")
async def load_model(model_name: str) -> dict:
    """Pre-load a model into memory (warm up)."""
    success = await inference_executor.load_model(model_name)
    if success:
        return {"status": "loaded", "model": model_name}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load {model_name}")


@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str) -> dict:
    """Unload a model from memory."""
    success = await inference_executor.unload_model(model_name)
    if success:
        return {"status": "unloaded", "model": model_name}
    else:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")


# === Legacy Job API ===


@app.post("/execute")
async def execute_job(request: ExecuteRequest) -> dict:
    """Execute a job."""
    logger.info("received_job", job_id=request.job_id)

    # Run in background so we can respond immediately
    asyncio.create_task(_execute_and_report(request))

    return {"status": "accepted", "job_id": request.job_id}


async def _execute_and_report(request: ExecuteRequest) -> None:
    """Execute job and report result to scheduler."""
    result = await executor.execute(
        job_id=request.job_id,
        command=request.command,
    )

    if result.error:
        await notify_job_failed(request.job_id, result.error)
    else:
        await notify_job_completed(
            request.job_id,
            result.exit_code,
            {
                "stdout": result.stdout[:10000],  # Truncate
                "stderr": result.stderr[:10000],
            },
        )


@app.post("/kill")
async def kill_job(request: KillRequest) -> dict:
    """Kill a running job."""
    if executor.kill(request.job_id):
        return {"status": "killed", "job_id": request.job_id}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Job {request.job_id} not found or not running",
        )


def run() -> None:
    """Run the worker server."""
    uvicorn.run(
        "flux.worker.main:app",
        host="0.0.0.0",
        port=config.port,
        reload=True,
    )


if __name__ == "__main__":
    run()
