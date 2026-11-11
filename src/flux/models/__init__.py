"""Core data models for TaskForge."""

from flux.models.resources import Resources
from flux.models.job import Job, JobStatus, JobPriority
from flux.models.worker import Worker, WorkerStatus
from flux.models.request import (
    PredictRequest,
    PredictResponse,
    InferenceRequest,
    InferenceBatch,
    ModelInfo,
)

__all__ = [
    # Resources
    "Resources",
    # Jobs (legacy)
    "Job",
    "JobStatus",
    "JobPriority",
    # Workers
    "Worker",
    "WorkerStatus",
    # Inference
    "PredictRequest",
    "PredictResponse",
    "InferenceRequest",
    "InferenceBatch",
    "ModelInfo",
]
