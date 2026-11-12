"""The engine: owns the model, the batch loop, and the queue/compute timing split.

Batching lives here, not in a separate pre-worker stage. Each pass through
the loop decides how many pending requests to pull (the scheduler's job),
pads and runs them through the model together in one forward pass (the
batching), and times the two phases separately so the split in
GenerateResponse is honest: queue_time_ms is how long a request sat waiting,
compute_time_ms is how long the shared forward pass actually took.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog
import torch

from flux.queue import QueuedRequest, RequestQueue
from flux.scheduler import SchedulingPolicy, select_batch
from flux.schemas import GenerateResponse

logger = structlog.get_logger()

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
POLL_INTERVAL_S = 0.001


class Engine:
    """Loads one model onto one GPU and runs the batch loop against one queue."""

    def __init__(
        self,
        queue: RequestQueue,
        model_name: str = DEFAULT_MODEL,
        max_batch_size: int = 8,
        max_wait_ms: float = 20.0,
        policy: SchedulingPolicy = SchedulingPolicy.FIFO,
        device: Optional[str] = None,
    ) -> None:
        self.queue = queue
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000
        self.policy = policy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._model = None
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Load the model (blocking, so it runs off the event loop thread) and start the batch loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)

        self._running = True
        self._loop_task = asyncio.create_task(self._batch_loop())
        logger.info(
            "engine_started",
            model=self.model_name,
            device=self.device,
            max_batch_size=self.max_batch_size,
            policy=self.policy,
        )

    async def stop(self) -> None:
        """Stop the batch loop."""
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
        # Left padding so every sequence's last real token lands in the same
        # column - generate() continues from that column for all of them.
        self._tokenizer.padding_side = "left"

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
        self._model.to(self.device)
        self._model.eval()

    async def _batch_loop(self) -> None:
        """Wait for requests, give stragglers a short window to join, then run a batch."""
        while self._running:
            try:
                await self.queue.wait_for_arrival(timeout=1.0)
                if self.queue.depth == 0:
                    continue

                deadline = time.perf_counter() + self.max_wait_s
                while self.queue.depth < self.max_batch_size and time.perf_counter() < deadline:
                    await asyncio.sleep(POLL_INTERVAL_S)

                pending = self.queue.snapshot()
                if not pending:
                    continue

                batch = select_batch(pending, self.max_batch_size, self.policy)
                self.queue.remove(batch)
                await self._run_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("batch_loop_error", error=str(e))
                await asyncio.sleep(0.1)

    async def _run_batch(self, batch: list[QueuedRequest]) -> None:
        """Run one batch through the model and resolve every request's Future."""
        compute_start = time.perf_counter()
        queue_times_ms = [(compute_start - r.arrived_at) * 1000 for r in batch]

        loop = asyncio.get_event_loop()
        try:
            texts = await loop.run_in_executor(None, self._generate, batch)
        except Exception as e:
            logger.error("generate_error", error=str(e), batch_size=len(batch))
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
            return

        compute_time_ms = (time.perf_counter() - compute_start) * 1000

        for request, text, queue_time_ms in zip(batch, texts, queue_times_ms):
            if not request.future.done():
                request.future.set_result(
                    GenerateResponse(
                        text=text,
                        queue_time_ms=queue_time_ms,
                        compute_time_ms=compute_time_ms,
                        batch_size=len(batch),
                    )
                )

    def _generate(self, batch: list[QueuedRequest]) -> list[str]:
        """Blocking: tokenize, pad, run one shared forward pass, decode each request's share."""
        inputs = self._tokenizer(
            [r.prompt for r in batch],
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]

        # One generate() call needs one max_new_tokens - use the largest ask in
        # the batch, then truncate each sequence back down to what it asked for.
        max_new_tokens = max(r.max_new_tokens for r in batch)

        with torch.no_grad():
            generated = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
                do_sample=False,
            )

        texts = []
        for i, request in enumerate(batch):
            new_tokens = generated[i, input_len : input_len + request.max_new_tokens]
            texts.append(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        return texts
