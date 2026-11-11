"""Dynamic batcher that collects requests into batches."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import structlog
from prometheus_client import Counter, Gauge, Histogram

from flux.models.request import InferenceBatch, InferenceRequest

logger = structlog.get_logger()

# Metrics
BATCH_SIZE = Histogram(
    "flux_batch_size",
    "Batch sizes",
    ["model"],
    buckets=[1, 2, 4, 8, 16, 32, 64],
)
BATCH_WAIT_TIME = Histogram(
    "flux_batch_wait_ms",
    "Time requests wait for batching",
    ["model"],
    buckets=[1, 2, 5, 10, 20, 50, 100],
)
QUEUE_DEPTH = Gauge(
    "flux_batcher_queue_depth",
    "Pending requests in batcher",
    ["model"],
)
REQUESTS_BATCHED = Counter(
    "flux_requests_batched_total",
    "Total requests batched",
    ["model"],
)


@dataclass
class BatcherConfig:
    """Configuration for the batcher."""

    max_batch_size: int = 32
    max_wait_ms: float = 10.0
    max_queue_depth: int = 1000


@dataclass
class PendingRequest:
    """A request waiting to be batched."""

    request: InferenceRequest
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.perf_counter)


class Batcher:
    """Dynamic batcher that collects requests into batches.

    Batching improves GPU utilization by processing multiple requests together.
    Requests are batched until:
    - max_batch_size is reached, OR
    - max_wait_ms has elapsed since first request

    Usage:
        batcher = Batcher(config, process_batch_fn)
        await batcher.start()

        # Submit request and wait for result
        result = await batcher.submit(request)
    """

    def __init__(
        self,
        config: BatcherConfig,
        process_batch: Callable[[InferenceBatch], Any],
    ) -> None:
        self.config = config
        self.process_batch = process_batch

        # Per-model queues
        self._queues: dict[str, asyncio.Queue[PendingRequest]] = {}
        self._batch_tasks: dict[str, asyncio.Task] = {}
        self._running = False

    async def start(self) -> None:
        """Start the batcher."""
        self._running = True
        logger.info("batcher_started", config=self.config)

    async def stop(self) -> None:
        """Stop the batcher."""
        self._running = False

        # Cancel all batch tasks
        for task in self._batch_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("batcher_stopped")

    def _get_queue(self, model: str) -> asyncio.Queue[PendingRequest]:
        """Get or create queue for a model."""
        if model not in self._queues:
            self._queues[model] = asyncio.Queue()
            # Start batch loop for this model
            self._batch_tasks[model] = asyncio.create_task(
                self._batch_loop(model)
            )
        return self._queues[model]

    @property
    def queue_depth(self) -> int:
        """Total pending requests across all models."""
        return sum(q.qsize() for q in self._queues.values())

    def queue_depth_for_model(self, model: str) -> int:
        """Pending requests for a specific model."""
        if model in self._queues:
            return self._queues[model].qsize()
        return 0

    async def submit(self, request: InferenceRequest) -> list[Any]:
        """Submit a request and wait for result.

        Args:
            request: The inference request

        Returns:
            List of predictions for the request's inputs

        Raises:
            asyncio.TimeoutError: If request times out
            Exception: If inference fails
        """
        # Backpressure check
        if self.queue_depth >= self.config.max_queue_depth:
            raise RuntimeError("Batcher queue full, server overloaded")

        # Create future for result
        future: asyncio.Future[list[Any]] = asyncio.get_event_loop().create_future()
        pending = PendingRequest(request=request, future=future)

        # Enqueue
        queue = self._get_queue(request.model)
        await queue.put(pending)

        QUEUE_DEPTH.labels(model=request.model).set(queue.qsize())

        # Wait for result with timeout
        try:
            timeout_s = request.timeout_ms / 1000
            result = await asyncio.wait_for(future, timeout=timeout_s)
            return result
        except asyncio.TimeoutError:
            logger.warning("request_timeout", request_id=request.id)
            raise

    async def _batch_loop(self, model: str) -> None:
        """Batch loop for a specific model."""
        queue = self._queues[model]

        while self._running:
            try:
                batch = await self._collect_batch(model, queue)
                if batch.size > 0:
                    await self._process_and_distribute(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("batch_loop_error", model=model, error=str(e))
                await asyncio.sleep(0.1)  # Backoff on error

    async def _collect_batch(
        self,
        model: str,
        queue: asyncio.Queue[PendingRequest],
    ) -> InferenceBatch:
        """Collect requests into a batch."""
        batch = InferenceBatch(model=model)
        pending_requests: list[PendingRequest] = []

        # Wait for first request
        try:
            first = await asyncio.wait_for(queue.get(), timeout=1.0)
            pending_requests.append(first)
            batch.add_request(first.request)
        except asyncio.TimeoutError:
            return batch  # Empty batch, will retry

        # Collect more until batch full or timeout
        deadline = time.perf_counter() + (self.config.max_wait_ms / 1000)

        while batch.size < self.config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break

            try:
                pending = await asyncio.wait_for(queue.get(), timeout=remaining)
                pending_requests.append(pending)
                batch.add_request(pending.request)
            except asyncio.TimeoutError:
                break

        # Record metrics
        if batch.size > 0:
            BATCH_SIZE.labels(model=model).observe(batch.size)
            REQUESTS_BATCHED.labels(model=model).inc(batch.size)

            # Record wait times
            now = time.perf_counter()
            for pending in pending_requests:
                wait_ms = (now - pending.enqueued_at) * 1000
                BATCH_WAIT_TIME.labels(model=model).observe(wait_ms)

            QUEUE_DEPTH.labels(model=model).set(queue.qsize())

        # Store pending requests for result distribution
        batch._pending_requests = pending_requests  # type: ignore

        return batch

    async def _process_and_distribute(self, batch: InferenceBatch) -> None:
        """Process batch and distribute results to waiting requests."""
        pending_requests: list[PendingRequest] = batch._pending_requests  # type: ignore

        try:
            # Process the batch (calls scheduler -> worker)
            results = await self.process_batch(batch)

            # Distribute results back to individual requests
            result_idx = 0
            for pending in pending_requests:
                num_inputs = len(pending.request.inputs)
                request_results = results[result_idx:result_idx + num_inputs]
                result_idx += num_inputs

                if not pending.future.done():
                    pending.future.set_result(request_results)

            logger.debug(
                "batch_processed",
                batch_id=batch.id,
                size=batch.size,
                model=batch.model,
            )

        except Exception as e:
            # Propagate error to all waiting requests
            for pending in pending_requests:
                if not pending.future.done():
                    pending.future.set_exception(e)

            logger.error(
                "batch_failed",
                batch_id=batch.id,
                error=str(e),
            )
