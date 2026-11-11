#!/usr/bin/env python
"""Script to test the scheduler locally."""

import asyncio
import httpx
import time


SCHEDULER_URL = "http://localhost:8000"


async def main():
    async with httpx.AsyncClient() as client:
        # Check health
        print("Checking scheduler health...")
        response = await client.get(f"{SCHEDULER_URL}/health")
        print(f"Health: {response.json()}")

        # Get initial stats
        response = await client.get(f"{SCHEDULER_URL}/stats")
        print(f"Initial stats: {response.json()}")

        # Submit some jobs
        print("\nSubmitting jobs...")
        jobs = []

        for i in range(5):
            response = await client.post(
                f"{SCHEDULER_URL}/jobs",
                json={
                    "name": f"test-job-{i}",
                    "command": f"echo 'Hello from job {i}' && sleep 2",
                    "resources": {"cpu": 1, "memory": 512, "gpu": 0},
                    "priority": 5 + i,  # Varying priorities
                },
            )
            job = response.json()
            jobs.append(job)
            print(f"  Submitted: {job['name']} (id={job['id'][:8]}..., priority={job['priority']})")

        # List jobs
        print("\nListing jobs...")
        response = await client.get(f"{SCHEDULER_URL}/jobs")
        for job in response.json():
            print(f"  {job['name']}: {job['status']}")

        # Get stats
        print("\nScheduler stats:")
        response = await client.get(f"{SCHEDULER_URL}/stats")
        stats = response.json()
        print(f"  Queue depth: {stats['queue']['pending']}")
        print(f"  Running jobs: {stats['jobs']['running']}")
        print(f"  Workers: {stats['workers']['total_workers']}")

        # Poll for completion
        print("\nWaiting for jobs to complete...")
        for _ in range(30):
            await asyncio.sleep(1)

            response = await client.get(f"{SCHEDULER_URL}/stats")
            stats = response.json()

            pending = stats["queue"]["pending"]
            running = stats["jobs"]["running"]
            completed = stats["jobs"]["completed"]

            print(f"  Pending: {pending}, Running: {running}, Completed: {completed}")

            if pending == 0 and running == 0:
                print("\nAll jobs completed!")
                break

        # Final stats
        print("\nFinal stats:")
        response = await client.get(f"{SCHEDULER_URL}/stats")
        print(response.json())


if __name__ == "__main__":
    asyncio.run(main())
