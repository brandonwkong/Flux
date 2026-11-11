"""Tests for scheduler components."""

import pytest
from flux.models.resources import Resources
from flux.models.job import Job, JobStatus, JobPriority
from flux.models.worker import Worker
from flux.scheduler.queue import JobQueue
from flux.scheduler.placement import PlacementEngine, PlacementStrategy


class TestJobQueue:
    def test_push_and_pop(self):
        queue = JobQueue()

        job1 = Job(name="job1", command="echo 1", priority=5)
        job2 = Job(name="job2", command="echo 2", priority=10)

        queue.push(job1)
        queue.push(job2)

        assert len(queue) == 2

        # Higher priority should come first
        popped = queue.pop()
        assert popped.name == "job2"

        popped = queue.pop()
        assert popped.name == "job1"

        assert len(queue) == 0

    def test_fifo_within_same_priority(self):
        queue = JobQueue()

        job1 = Job(name="first", command="echo 1", priority=5)
        job2 = Job(name="second", command="echo 2", priority=5)
        job3 = Job(name="third", command="echo 3", priority=5)

        queue.push(job1)
        queue.push(job2)
        queue.push(job3)

        assert queue.pop().name == "first"
        assert queue.pop().name == "second"
        assert queue.pop().name == "third"

    def test_remove(self):
        queue = JobQueue()

        job = Job(name="test", command="echo")
        queue.push(job)

        removed = queue.remove(job.id)
        assert removed is not None
        assert removed.status == JobStatus.CANCELLED
        assert len(queue) == 0

    def test_get(self):
        queue = JobQueue()

        job = Job(name="test", command="echo")
        queue.push(job)

        found = queue.get(job.id)
        assert found is not None
        assert found.id == job.id


class TestPlacementEngine:
    def _create_workers(self):
        """Create test workers with different capacities."""
        return [
            Worker(
                name="small",
                host="localhost",
                capacity=Resources(cpu=2, memory=2048, gpu=0),
                available=Resources(cpu=2, memory=2048, gpu=0),
            ),
            Worker(
                name="medium",
                host="localhost",
                capacity=Resources(cpu=4, memory=4096, gpu=1),
                available=Resources(cpu=4, memory=4096, gpu=1),
            ),
            Worker(
                name="large",
                host="localhost",
                capacity=Resources(cpu=8, memory=8192, gpu=2),
                available=Resources(cpu=8, memory=8192, gpu=2),
            ),
        ]

    def test_first_fit(self):
        engine = PlacementEngine(strategy=PlacementStrategy.FIRST_FIT)
        workers = self._create_workers()

        job = Job(
            name="test",
            command="echo",
            resources=Resources(cpu=1, memory=1024, gpu=0),
        )

        worker = engine.find_worker(job, workers)
        assert worker is not None
        assert worker.name == "small"  # First one that fits

    def test_best_fit(self):
        engine = PlacementEngine(strategy=PlacementStrategy.BEST_FIT)
        workers = self._create_workers()

        job = Job(
            name="test",
            command="echo",
            resources=Resources(cpu=2, memory=2048, gpu=0),
        )

        worker = engine.find_worker(job, workers)
        assert worker is not None
        assert worker.name == "small"  # Minimizes waste

    def test_no_fit(self):
        engine = PlacementEngine(strategy=PlacementStrategy.BEST_FIT)
        workers = self._create_workers()

        # Job that's too big for any worker
        job = Job(
            name="test",
            command="echo",
            resources=Resources(cpu=16, memory=16384, gpu=4),
        )

        worker = engine.find_worker(job, workers)
        assert worker is None

    def test_gpu_requirement(self):
        engine = PlacementEngine(strategy=PlacementStrategy.BEST_FIT)
        workers = self._create_workers()

        # Job requiring GPU
        job = Job(
            name="test",
            command="echo",
            resources=Resources(cpu=2, memory=2048, gpu=1),
        )

        worker = engine.find_worker(job, workers)
        assert worker is not None
        assert worker.name == "medium"  # Smallest with GPU
