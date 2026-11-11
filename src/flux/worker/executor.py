"""Executors for running inference and jobs on workers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog
from prometheus_client import Counter, Histogram

logger = structlog.get_logger()

# Metrics
INFERENCE_LATENCY = Histogram(
    "flux_worker_inference_latency_seconds",
    "Inference execution latency",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
INFERENCE_TOTAL = Counter(
    "flux_worker_inference_total",
    "Total inference executions",
    ["model", "status"],
)
MODEL_LOAD_TIME = Histogram(
    "flux_worker_model_load_seconds",
    "Model loading time",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


@dataclass
class InferenceResult:
    """Result of inference execution."""

    batch_id: str
    predictions: list[Any]
    latency_ms: float
    model: str
    batch_size: int
    error: Optional[str] = None


class InferenceExecutor:
    """Executes ML inference using ONNX Runtime.

    Features:
    - Model caching (warm pool)
    - Batch inference
    - GPU/CPU provider selection
    - Async execution
    """

    def __init__(
        self,
        model_dir: str = "models",
        use_gpu: bool = True,
        max_loaded_models: int = 5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.use_gpu = use_gpu
        self.max_loaded_models = max_loaded_models

        # Model cache: model_name -> (session, last_used_time)
        self._sessions: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

        # Track loaded models
        self._loaded_models: list[str] = []

        # Lazy import onnxruntime
        self._ort = None

    def _get_ort(self):
        """Lazy import ONNX Runtime."""
        if self._ort is None:
            try:
                import onnxruntime as ort
                self._ort = ort
            except ImportError:
                logger.warning("onnxruntime not installed, using mock inference")
                self._ort = None
        return self._ort

    def _get_providers(self) -> list[str]:
        """Get ONNX Runtime execution providers."""
        if self.use_gpu:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    async def load_model(self, model_name: str) -> bool:
        """Load a model into memory.

        Args:
            model_name: Name of the model to load

        Returns:
            True if model loaded successfully
        """
        async with self._lock:
            if model_name in self._sessions:
                return True

            model_path = self.model_dir / f"{model_name}.onnx"

            # Check if model file exists
            if not model_path.exists():
                logger.warning("model_not_found", model=model_name, path=str(model_path))
                # For development: create mock session
                self._sessions[model_name] = (None, time.time())
                self._loaded_models.append(model_name)
                return True

            ort = self._get_ort()
            if ort is None:
                # Mock mode
                self._sessions[model_name] = (None, time.time())
                self._loaded_models.append(model_name)
                return True

            try:
                start = time.perf_counter()

                # Run model loading in thread pool to not block
                loop = asyncio.get_event_loop()
                session = await loop.run_in_executor(
                    None,
                    lambda: ort.InferenceSession(
                        str(model_path),
                        providers=self._get_providers(),
                    ),
                )

                elapsed = time.perf_counter() - start
                MODEL_LOAD_TIME.labels(model=model_name).observe(elapsed)

                # Evict old models if at capacity
                if len(self._sessions) >= self.max_loaded_models:
                    await self._evict_oldest()

                self._sessions[model_name] = (session, time.time())
                self._loaded_models.append(model_name)

                logger.info(
                    "model_loaded",
                    model=model_name,
                    load_time_ms=elapsed * 1000,
                )
                return True

            except Exception as e:
                logger.error("model_load_error", model=model_name, error=str(e))
                return False

    async def _evict_oldest(self) -> None:
        """Evict the least recently used model."""
        if not self._sessions:
            return

        oldest_name = min(
            self._sessions.keys(),
            key=lambda k: self._sessions[k][1],
        )

        del self._sessions[oldest_name]
        if oldest_name in self._loaded_models:
            self._loaded_models.remove(oldest_name)

        logger.info("model_evicted", model=oldest_name)

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        async with self._lock:
            if model_name not in self._sessions:
                return False

            del self._sessions[model_name]
            if model_name in self._loaded_models:
                self._loaded_models.remove(model_name)

            logger.info("model_unloaded", model=model_name)
            return True

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._sessions

    @property
    def loaded_models(self) -> list[str]:
        """List of currently loaded models."""
        return list(self._loaded_models)

    async def infer(
        self,
        batch_id: str,
        model_name: str,
        inputs: list[list[float]],
    ) -> InferenceResult:
        """Run inference on a batch of inputs.

        Args:
            batch_id: Unique batch identifier
            model_name: Name of model to use
            inputs: List of input tensors

        Returns:
            InferenceResult with predictions
        """
        start = time.perf_counter()

        try:
            # Ensure model is loaded
            if not self.is_model_loaded(model_name):
                loaded = await self.load_model(model_name)
                if not loaded:
                    return InferenceResult(
                        batch_id=batch_id,
                        predictions=[],
                        latency_ms=0,
                        model=model_name,
                        batch_size=len(inputs),
                        error=f"Failed to load model: {model_name}",
                    )

            session, _ = self._sessions[model_name]

            # Update last used time
            self._sessions[model_name] = (session, time.time())

            # Run inference
            if session is None:
                # Mock inference for development
                predictions = await self._mock_inference(model_name, inputs)
            else:
                predictions = await self._run_inference(session, inputs)

            elapsed_ms = (time.perf_counter() - start) * 1000

            INFERENCE_LATENCY.labels(model=model_name).observe(elapsed_ms / 1000)
            INFERENCE_TOTAL.labels(model=model_name, status="success").inc()

            logger.debug(
                "inference_complete",
                batch_id=batch_id,
                model=model_name,
                batch_size=len(inputs),
                latency_ms=elapsed_ms,
            )

            return InferenceResult(
                batch_id=batch_id,
                predictions=predictions,
                latency_ms=elapsed_ms,
                model=model_name,
                batch_size=len(inputs),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            INFERENCE_TOTAL.labels(model=model_name, status="error").inc()

            logger.error(
                "inference_error",
                batch_id=batch_id,
                model=model_name,
                error=str(e),
            )

            return InferenceResult(
                batch_id=batch_id,
                predictions=[],
                latency_ms=elapsed_ms,
                model=model_name,
                batch_size=len(inputs),
                error=str(e),
            )

    async def _run_inference(
        self,
        session: Any,
        inputs: list[list[float]],
    ) -> list[Any]:
        """Run actual ONNX inference."""
        import numpy as np

        # Convert to numpy array
        input_array = np.array(inputs, dtype=np.float32)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run in thread pool
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: session.run(None, {input_name: input_array}),
        )

        # Return predictions
        return outputs[0].tolist()

    async def _mock_inference(
        self,
        model_name: str,
        inputs: list[list[float]],
    ) -> list[Any]:
        """Mock inference for development/testing."""
        # Simulate inference latency
        await asyncio.sleep(0.005)  # 5ms

        # Return mock predictions based on model type
        batch_size = len(inputs)

        if "resnet" in model_name.lower() or "image" in model_name.lower():
            # Image classification: return class indices
            import random
            return [random.randint(0, 999) for _ in range(batch_size)]
        elif "bert" in model_name.lower() or "text" in model_name.lower():
            # Text model: return embeddings
            return [[0.1] * 768 for _ in range(batch_size)]
        else:
            # Generic: return input length
            return [len(inp) for inp in inputs]


# === Legacy JobExecutor (kept for compatibility) ===


@dataclass
class ExecutionResult:
    """Result of job execution."""

    job_id: str
    exit_code: int
    stdout: str
    stderr: str
    error: Optional[str] = None


class JobExecutor:
    """Executes jobs in isolated subprocesses (legacy)."""

    def __init__(self, default_timeout: float = 3600.0) -> None:
        self._running: dict[str, asyncio.subprocess.Process] = {}
        self._default_timeout = default_timeout

    async def execute(
        self,
        job_id: str,
        command: str,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a job command."""
        timeout = timeout or self._default_timeout

        logger.info("executing_job", job_id=job_id, command=command[:100])

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._running[job_id] = process

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                return ExecutionResult(
                    job_id=job_id,
                    exit_code=process.returncode or 0,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                )

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    job_id=job_id,
                    exit_code=-1,
                    stdout="",
                    stderr="",
                    error=f"Job timed out after {timeout}s",
                )

        except Exception as e:
            logger.error("execution_error", job_id=job_id, error=str(e))
            return ExecutionResult(
                job_id=job_id,
                exit_code=-1,
                stdout="",
                stderr="",
                error=str(e),
            )

        finally:
            self._running.pop(job_id, None)

    async def kill(self, job_id: str) -> bool:
        """Kill a running job."""
        process = self._running.get(job_id)
        if not process:
            return False

        try:
            process.kill()
            await process.wait()
            logger.info("job_killed", job_id=job_id)
            return True
        except Exception as e:
            logger.error("kill_error", job_id=job_id, error=str(e))
            return False

    def is_running(self, job_id: str) -> bool:
        """Check if a job is currently running."""
        return job_id in self._running

    @property
    def running_count(self) -> int:
        """Number of currently running jobs."""
        return len(self._running)
