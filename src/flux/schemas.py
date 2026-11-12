"""Request and response schemas for the generation API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """A client's request to generate text from a prompt."""

    prompt: str
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    priority: int = Field(default=5, ge=0, le=20, description="Higher runs sooner under PRIORITY policy")
    deadline_ms: int = Field(default=5000, description="Soft deadline from arrival, used by the EDF policy")


class GenerateResponse(BaseModel):
    """What the client gets back: the completion plus an honest timing breakdown."""

    text: str
    queue_time_ms: float = Field(description="Time spent waiting before the engine started this request")
    compute_time_ms: float = Field(description="Time spent in the model's forward pass(es) for this request")
    batch_size: int = Field(description="How many requests were batched together for this request's compute")
