"""FastAPI server: the thin HTTP layer over the queue and the engine.

This file does almost nothing on purpose - it creates a queue and an
engine at startup, and /generate just calls queue.submit() and awaits
the result. All the actual engineering (batching, scheduling, timing)
lives in queue.py, scheduler.py, and engine.py.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException

from flux.engine import DEFAULT_MODEL, Engine
from flux.queue import RequestQueue
from flux.scheduler import SchedulingPolicy
from flux.schemas import GenerateRequest, GenerateResponse

logger = structlog.get_logger()

MODEL_NAME = os.environ.get("FLUX_MODEL", DEFAULT_MODEL)
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "20"))
MAX_QUEUE_DEPTH = int(os.environ.get("MAX_QUEUE_DEPTH", "1000"))
SCHEDULING_POLICY = SchedulingPolicy(os.environ.get("SCHEDULING_POLICY", "fifo"))

queue: Optional[RequestQueue] = None
engine: Optional[Engine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup loads the model and starts the batch loop; shutdown stops it."""
    global queue, engine

    queue = RequestQueue(max_depth=MAX_QUEUE_DEPTH)
    engine = Engine(
        queue=queue,
        model_name=MODEL_NAME,
        max_batch_size=MAX_BATCH_SIZE,
        max_wait_ms=MAX_WAIT_MS,
        policy=SCHEDULING_POLICY,
    )
    await engine.start()

    yield

    await engine.stop()


app = FastAPI(title="Flux", version="0.1.0", lifespan=lifespan)


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """Submit a prompt and wait for the engine's result."""
    if not queue:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        return await queue.submit(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            priority=request.priority,
            deadline_ms=request.deadline_ms,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {"status": "healthy"}


@app.get("/stats")
async def stats() -> dict:
    """Current queue depth and engine configuration."""
    if not queue or not engine:
        raise HTTPException(status_code=503, detail="Service not ready")

    return {
        "queue_depth": queue.depth,
        "policy": engine.policy,
        "max_batch_size": engine.max_batch_size,
        "model": engine.model_name,
        "device": engine.device,
    }


def run() -> None:
    """Run the server. SCHEDULING_POLICY is read once at process start -
    changing policy means restarting the process, same as changing any
    other engine configuration."""
    uvicorn.run("flux.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
