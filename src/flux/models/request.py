"""Request and response models for inference API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Inference request from client."""

    model: str = Field(description="Model name to use for inference")
    inputs: list[list[float]] = Field(description="Batch of input tensors")
    priority: int = Field(default=5, ge=0, le=20, description="Request priority (0-20)")
    timeout_ms: Optional[int] = Field(default=5000, description="Request timeout in ms")


class PredictResponse(BaseModel):
    """Inference response to client."""

    predictions: list[Any] = Field(description="Model predictions")
    latency_ms: float = Field(description="End-to-end latency in milliseconds")
    batch_size: int = Field(description="Number of inputs in batch")
    model: str = Field(description="Model used for inference")


class InferenceRequest(BaseModel):
    """Internal inference request with tracking info."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    model: str
    inputs: list[list[float]]
    priority: int = 5

    # Tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    batch_id: Optional[str] = None
    worker_id: Optional[str] = None

    # Timeout
    timeout_ms: int = 5000

    @property
    def age_ms(self) -> float:
        """Time since request was created."""
        return (datetime.utcnow() - self.created_at).total_seconds() * 1000

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded timeout."""
        return self.age_ms > self.timeout_ms


class InferenceBatch(BaseModel):
    """A batch of requests to be processed together."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    model: str
    requests: list[InferenceRequest] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def size(self) -> int:
        """Number of requests in batch."""
        return len(self.requests)

    @property
    def total_inputs(self) -> int:
        """Total number of input tensors across all requests."""
        return sum(len(r.inputs) for r in self.requests)

    @property
    def combined_inputs(self) -> list[list[float]]:
        """Flatten all inputs into single batch."""
        inputs = []
        for req in self.requests:
            inputs.extend(req.inputs)
        return inputs

    def add_request(self, request: InferenceRequest) -> None:
        """Add a request to the batch."""
        request.batch_id = self.id
        self.requests.append(request)


class ModelInfo(BaseModel):
    """Information about a loaded model."""

    name: str
    version: str = "1.0"
    input_shape: list[int] = Field(default_factory=lambda: [-1, 3, 224, 224])
    output_shape: list[int] = Field(default_factory=lambda: [-1, 1000])
    loaded_at: datetime = Field(default_factory=datetime.utcnow)

    # Performance hints
    max_batch_size: int = 32
    avg_latency_ms: float = 10.0
