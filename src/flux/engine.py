"""The engine: owns the model, the continuous batch loop, and the queue/compute timing split.

Continuous (iteration-level) batching: between every single decode step,
the loop admits pending requests (up to free capacity, in the scheduler's
chosen order) and evicts anyone who just finished, resolving their Future
immediately - not at the end of some larger, shared batch everyone had to
wait for. See docs/continuous_batching_notes.md for how this replaced the
old static-batching loop, step by step.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog
import torch

from flux.decode import ContinuousBatch
from flux.queue import QueuedRequest, RequestQueue
from flux.scheduler import SchedulingPolicy, select_batch
from flux.schemas import GenerateResponse

logger = structlog.get_logger()

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class Engine:
    """Loads one model onto one GPU and runs the continuous batch loop against one queue."""

    def __init__(
        self,
        queue: RequestQueue,
        model_name: str = DEFAULT_MODEL,
        max_batch_size: int = 8,
        policy: SchedulingPolicy = SchedulingPolicy.FIFO,
        device: Optional[str] = None,
    ) -> None:
        self.queue = queue
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.policy = policy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._model = None
        self._batch: Optional[ContinuousBatch] = None
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Per-active-sequence bookkeeping, keyed by QueuedRequest.id (used
        # directly as ContinuousBatch's seq_id too - one id, one meaning).
        self._requests_by_id: dict[str, QueuedRequest] = {}
        self._admitted_at: dict[str, float] = {}

    async def start(self) -> None:
        """Load the model (blocking, so it runs off the event loop thread) and start the loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self._batch = ContinuousBatch(self._model, self._tokenizer, self.device)
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info(
            "engine_started",
            model=self.model_name,
            device=self.device,
            max_batch_size=self.max_batch_size,
            policy=self.policy,
        )

    async def stop(self) -> None:
        """Stop the loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("engine_stopped")

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model.to(self.device)
        self._model.eval()

    async def _run_loop(self) -> None:
        """Every iteration: admit as many pending requests as fit (the
        scheduler decides which ones, same policies as before - just
        deciding admission order now instead of whole-batch composition
        once per generation), run one decode step, resolve anyone who
        just finished."""
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                await self._admit_pending(loop)

                if self._batch.active_count > 0:
                    finished = await loop.run_in_executor(None, self._batch.step)
                    self._resolve_finished(finished)
                else:
                    await self.queue.wait_for_arrival(timeout=1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("engine_loop_error", error=str(e))
                await asyncio.sleep(0.1)

    async def _admit_pending(self, loop: asyncio.AbstractEventLoop) -> None:
        """Pull in pending requests up to free capacity, in the scheduler's order."""
        free_slots = self.max_batch_size - self._batch.active_count
        if free_slots <= 0:
            return

        pending = self.queue.snapshot()
        if not pending:
            return

        to_admit = select_batch(pending, free_slots, self.policy)
        self.queue.remove(to_admit)

        for request in to_admit:
            # Timestamp BEFORE the (blocking, possibly slow-on-cold-cache)
            # prefill call, not after - admission means "entered the active
            # set," and the prefill compute itself belongs to compute_time,
            # not queue_time. Recording it after would silently move real
            # compute cost into the wait-time number.
            admitted_at = time.perf_counter()
            try:
                await loop.run_in_executor(
                    None, self._batch.admit, request.id, request.prompt, request.max_new_tokens
                )
            except Exception as e:
                logger.error("admit_error", error=str(e), request_id=request.id)
                if not request.future.done():
                    request.future.set_exception(e)
                continue

            self._requests_by_id[request.id] = request
            self._admitted_at[request.id] = admitted_at

    def _resolve_finished(self, finished: list[tuple[str, str, int, float]]) -> None:
        """Resolve each just-finished sequence's Future the moment it finishes.

        queue_time_ms: arrival into the queue -> admission into the active
        set. compute_time_ms: admission -> this finish. Both now genuinely
        per-sequence - no longer a value shared across a whole batch,
        because there's no longer a "whole batch" that starts and ends
        together.

        batch_size: average number of active sequences across every step
        this request's generation shared the batch with (Step 4's choice,
        over a snapshot-at-admission or peak alternative) - this is the
        number that answers "how batched was this request's compute,
        overall," which is what the field always meant back when the
        batch was fixed for the whole request.
        """
        now = time.perf_counter()
        for seq_id, text, output_tokens, avg_batch_size in finished:
            request = self._requests_by_id.pop(seq_id, None)
            admitted_at = self._admitted_at.pop(seq_id, None)
            if request is None or admitted_at is None:
                continue

            if not request.future.done():
                request.future.set_result(
                    GenerateResponse(
                        text=text,
                        queue_time_ms=(admitted_at - request.arrived_at) * 1000,
                        compute_time_ms=(now - admitted_at) * 1000,
                        batch_size=round(avg_batch_size),
                        output_tokens=output_tokens,
                    )
                )
