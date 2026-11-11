"""Resource representation for jobs and workers."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Resources(BaseModel):
    """Resource requirements or capacity.

    Used for both job requirements and worker capacity.
    Supports CPU cores, memory (MB), and GPU count.
    """

    cpu: float = Field(default=1.0, ge=0, description="CPU cores")
    memory: int = Field(default=512, ge=0, description="Memory in MB")
    gpu: int = Field(default=0, ge=0, description="GPU count")

    def fits_in(self, other: Resources) -> bool:
        """Check if this resource requirement fits within another's capacity."""
        return (
            self.cpu <= other.cpu
            and self.memory <= other.memory
            and self.gpu <= other.gpu
        )

    def __sub__(self, other: Resources) -> Resources:
        """Subtract resources (e.g., when allocating a job)."""
        return Resources(
            cpu=self.cpu - other.cpu,
            memory=self.memory - other.memory,
            gpu=self.gpu - other.gpu,
        )

    def __add__(self, other: Resources) -> Resources:
        """Add resources (e.g., when freeing a job)."""
        return Resources(
            cpu=self.cpu + other.cpu,
            memory=self.memory + other.memory,
            gpu=self.gpu + other.gpu,
        )

    def utilization(self, capacity: Resources) -> float:
        """Calculate utilization ratio against a capacity."""
        if capacity.cpu == 0 and capacity.memory == 0 and capacity.gpu == 0:
            return 0.0

        ratios = []
        if capacity.cpu > 0:
            ratios.append(self.cpu / capacity.cpu)
        if capacity.memory > 0:
            ratios.append(self.memory / capacity.memory)
        if capacity.gpu > 0:
            ratios.append(self.gpu / capacity.gpu)

        return sum(ratios) / len(ratios) if ratios else 0.0

    def waste_if_allocated(self, job: Resources, capacity: Resources) -> float:
        """Calculate wasted resources if a job is allocated.

        Used for best-fit bin packing - lower waste is better.
        """
        remaining = self - job
        return (
            (remaining.cpu / capacity.cpu if capacity.cpu > 0 else 0)
            + (remaining.memory / capacity.memory if capacity.memory > 0 else 0)
            + (remaining.gpu / capacity.gpu if capacity.gpu > 0 else 0)
        )
