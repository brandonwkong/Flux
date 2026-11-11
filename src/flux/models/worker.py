"""Worker model representing a compute node."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from flux.models.resources import Resources


class WorkerStatus(str, Enum):
    """Worker health states."""

    HEALTHY = "healthy"          # Accepting jobs
    UNHEALTHY = "unhealthy"      # Not responding, jobs will be rescheduled
    DRAINING = "draining"        # No new jobs, waiting for current to finish
    OFFLINE = "offline"          # Explicitly disconnected


class Worker(BaseModel):
    """A compute worker node for ML inference."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description="Human-readable worker name")
    host: str = Field(description="Worker hostname or IP")
    port: int = Field(default=8001, description="Worker API port")

    # Resources
    capacity: Resources = Field(description="Total resource capacity")
    available: Resources = Field(description="Currently available resources")

    # GPU-specific
    gpu_memory_total_mb: int = Field(default=0, description="Total GPU memory in MB")
    gpu_memory_used_mb: int = Field(default=0, description="Used GPU memory in MB")

    # Loaded models (warm pool)
    loaded_models: list[str] = Field(default_factory=list, description="Models loaded in memory")

    # State
    status: WorkerStatus = Field(default=WorkerStatus.HEALTHY)
    current_jobs: list[str] = Field(default_factory=list)
    pending_batches: int = Field(default=0, description="Batches queued on worker")

    # Heartbeat
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)

    # Stats
    jobs_completed: int = Field(default=0)
    jobs_failed: int = Field(default=0)
    total_inferences: int = Field(default=0, description="Total inference requests processed")
    avg_latency_ms: float = Field(default=0.0, description="Average inference latency")

    def can_fit(self, resources: Resources) -> bool:
        """Check if worker can fit a job with given resources."""
        if self.status != WorkerStatus.HEALTHY:
            return False
        return resources.fits_in(self.available)

    def has_model_loaded(self, model: str) -> bool:
        """Check if worker has model loaded (warm)."""
        return model in self.loaded_models

    def load_model(self, model: str) -> None:
        """Mark a model as loaded on this worker."""
        if model not in self.loaded_models:
            self.loaded_models.append(model)

    def unload_model(self, model: str) -> None:
        """Unload a model from this worker."""
        if model in self.loaded_models:
            self.loaded_models.remove(model)

    @property
    def gpu_memory_available_mb(self) -> int:
        """Available GPU memory in MB."""
        return self.gpu_memory_total_mb - self.gpu_memory_used_mb

    @property
    def gpu_utilization(self) -> float:
        """GPU memory utilization ratio."""
        if self.gpu_memory_total_mb == 0:
            return 0.0
        return self.gpu_memory_used_mb / self.gpu_memory_total_mb

    def allocate(self, job_id: str, resources: Resources) -> None:
        """Allocate resources for a job."""
        if not self.can_fit(resources):
            raise ValueError(f"Cannot fit job {job_id} - insufficient resources")

        self.available = self.available - resources
        self.current_jobs.append(job_id)

    def release(self, job_id: str, resources: Resources) -> None:
        """Release resources from a completed/failed job."""
        if job_id in self.current_jobs:
            self.current_jobs.remove(job_id)
        self.available = self.available + resources

        # Clamp to capacity (safety)
        self.available = Resources(
            cpu=min(self.available.cpu, self.capacity.cpu),
            memory=min(self.available.memory, self.capacity.memory),
            gpu=min(self.available.gpu, self.capacity.gpu),
        )

    def update_heartbeat(self) -> None:
        """Update last heartbeat time."""
        self.last_heartbeat = datetime.utcnow()

    def mark_draining(self) -> None:
        """Mark worker as draining (no new jobs)."""
        self.status = WorkerStatus.DRAINING

    def mark_unhealthy(self) -> None:
        """Mark worker as unhealthy."""
        self.status = WorkerStatus.UNHEALTHY

    def mark_healthy(self) -> None:
        """Mark worker as healthy."""
        self.status = WorkerStatus.HEALTHY

    @property
    def utilization(self) -> float:
        """Calculate current resource utilization."""
        allocated = self.capacity - self.available
        return allocated.utilization(self.capacity)

    @property
    def is_idle(self) -> bool:
        """Check if worker has no running jobs."""
        return len(self.current_jobs) == 0

    @property
    def seconds_since_heartbeat(self) -> float:
        """Seconds since last heartbeat."""
        return (datetime.utcnow() - self.last_heartbeat).total_seconds()


class WorkerRegistration(BaseModel):
    """Request model for worker registration."""

    name: str = Field(description="Human-readable worker name")
    host: str = Field(description="Worker hostname or IP")
    port: int = Field(default=8001, description="Worker API port")
    capacity: Resources = Field(description="Total resource capacity")

    # GPU info
    gpu_memory_total_mb: int = Field(default=0, description="Total GPU memory in MB")
    loaded_models: list[str] = Field(default_factory=list, description="Pre-loaded models")


class WorkerHeartbeat(BaseModel):
    """Heartbeat message from worker."""

    worker_id: str
    current_jobs: list[str] = Field(default_factory=list)
    available: Optional[Resources] = None
    pending_batches: int = Field(default=0)
    loaded_models: list[str] = Field(default_factory=list)
    gpu_memory_used_mb: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
