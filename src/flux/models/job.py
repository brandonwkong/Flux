"""Job model representing a schedulable task."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from flux.models.resources import Resources


class JobStatus(str, Enum):
    """Job lifecycle states."""

    PENDING = "pending"          # Waiting in queue
    SCHEDULED = "scheduled"      # Assigned to worker, not yet started
    RUNNING = "running"          # Currently executing
    COMPLETED = "completed"      # Finished successfully
    FAILED = "failed"            # Finished with error
    CANCELLED = "cancelled"      # Cancelled by user
    PREEMPTED = "preempted"      # Preempted by higher priority job


class JobPriority(int, Enum):
    """Job priority levels.

    Higher value = higher priority.
    """

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class Job(BaseModel):
    """A schedulable job/task."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(description="Human-readable job name")
    command: str = Field(description="Command to execute")
    resources: Resources = Field(default_factory=Resources)
    priority: int = Field(default=JobPriority.NORMAL, ge=0)

    # Lifecycle timestamps
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # State
    status: JobStatus = Field(default=JobStatus.PENDING)
    worker_id: Optional[str] = None

    # Retry configuration
    max_retries: int = Field(default=3, ge=0)
    retry_count: int = Field(default=0, ge=0)

    # Result
    exit_code: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def effective_priority(self) -> float:
        """Calculate priority with aging boost.

        Jobs waiting longer get priority boost to prevent starvation.
        +1 priority per hour of waiting.
        """
        if self.status != JobStatus.PENDING:
            return float(self.priority)

        wait_seconds = (datetime.utcnow() - self.submitted_at).total_seconds()
        aging_boost = wait_seconds / 3600  # +1 per hour
        return self.priority + aging_boost

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries

    def mark_scheduled(self, worker_id: str) -> None:
        """Mark job as scheduled to a worker."""
        self.status = JobStatus.SCHEDULED
        self.worker_id = worker_id
        self.scheduled_at = datetime.utcnow()

    def mark_running(self) -> None:
        """Mark job as running."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()

    def mark_completed(self, exit_code: int = 0, result: Any = None) -> None:
        """Mark job as completed successfully."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.exit_code = exit_code
        self.result = result

    def mark_failed(self, error: str, exit_code: int = 1) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.exit_code = exit_code
        self.error = error

    def mark_preempted(self) -> None:
        """Mark job as preempted (will be rescheduled)."""
        self.status = JobStatus.PREEMPTED
        self.worker_id = None
        self.scheduled_at = None
        self.started_at = None

    @property
    def wait_time_seconds(self) -> float:
        """Time spent waiting in queue."""
        if self.started_at:
            return (self.started_at - self.submitted_at).total_seconds()
        return (datetime.utcnow() - self.submitted_at).total_seconds()

    @property
    def run_time_seconds(self) -> Optional[float]:
        """Time spent running."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


class JobSubmission(BaseModel):
    """Request model for submitting a new job."""

    name: str = Field(description="Human-readable job name")
    command: str = Field(description="Command to execute")
    resources: Resources = Field(default_factory=Resources)
    priority: int = Field(default=JobPriority.NORMAL, ge=0)
    max_retries: int = Field(default=3, ge=0)
