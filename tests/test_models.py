"""Tests for core models."""

import pytest
from flux.models.resources import Resources
from flux.models.job import Job, JobStatus, JobPriority
from flux.models.worker import Worker, WorkerStatus


class TestResources:
    def test_fits_in(self):
        small = Resources(cpu=2, memory=1024, gpu=0)
        large = Resources(cpu=4, memory=2048, gpu=1)

        assert small.fits_in(large)
        assert not large.fits_in(small)

    def test_subtraction(self):
        a = Resources(cpu=4, memory=2048, gpu=2)
        b = Resources(cpu=2, memory=1024, gpu=1)
        result = a - b

        assert result.cpu == 2
        assert result.memory == 1024
        assert result.gpu == 1

    def test_addition(self):
        a = Resources(cpu=2, memory=1024, gpu=1)
        b = Resources(cpu=2, memory=1024, gpu=1)
        result = a + b

        assert result.cpu == 4
        assert result.memory == 2048
        assert result.gpu == 2


class TestJob:
    def test_job_creation(self):
        job = Job(name="test", command="echo hello")

        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.worker_id is None

    def test_effective_priority_increases_over_time(self):
        from datetime import datetime, timedelta

        job = Job(name="test", command="echo hello", priority=5)
        initial_priority = job.effective_priority()

        # Simulate waiting 1 hour
        job.submitted_at = datetime.utcnow() - timedelta(hours=1)
        aged_priority = job.effective_priority()

        assert aged_priority > initial_priority
        assert aged_priority == pytest.approx(6.0, abs=0.1)

    def test_mark_completed(self):
        job = Job(name="test", command="echo hello")
        job.mark_scheduled("worker-1")
        job.mark_running()
        job.mark_completed(exit_code=0)

        assert job.status == JobStatus.COMPLETED
        assert job.exit_code == 0
        assert job.completed_at is not None

    def test_can_retry(self):
        job = Job(name="test", command="echo hello", max_retries=3)

        assert job.can_retry()

        job.retry_count = 3
        assert not job.can_retry()


class TestWorker:
    def test_worker_creation(self):
        capacity = Resources(cpu=4, memory=4096, gpu=1)
        worker = Worker(
            name="test-worker",
            host="localhost",
            capacity=capacity,
            available=capacity.model_copy(),
        )

        assert worker.status == WorkerStatus.HEALTHY
        assert worker.is_idle
        assert worker.utilization == 0.0

    def test_allocate_and_release(self):
        capacity = Resources(cpu=4, memory=4096, gpu=1)
        worker = Worker(
            name="test-worker",
            host="localhost",
            capacity=capacity,
            available=capacity.model_copy(),
        )

        job_resources = Resources(cpu=2, memory=2048, gpu=0)
        worker.allocate("job-1", job_resources)

        assert worker.available.cpu == 2
        assert worker.available.memory == 2048
        assert "job-1" in worker.current_jobs
        assert worker.utilization > 0

        worker.release("job-1", job_resources)

        assert worker.available.cpu == 4
        assert worker.available.memory == 4096
        assert "job-1" not in worker.current_jobs

    def test_can_fit(self):
        capacity = Resources(cpu=4, memory=4096, gpu=1)
        worker = Worker(
            name="test-worker",
            host="localhost",
            capacity=capacity,
            available=capacity.model_copy(),
        )

        small_job = Resources(cpu=2, memory=2048, gpu=0)
        large_job = Resources(cpu=8, memory=8192, gpu=2)

        assert worker.can_fit(small_job)
        assert not worker.can_fit(large_job)

        # Unhealthy worker can't fit anything
        worker.mark_unhealthy()
        assert not worker.can_fit(small_job)
