#!/usr/bin/env python
"""
Benchmark script for Flux inference platform.

Run this BEFORE and AFTER optimizations to measure improvement.

Usage:
    # Start server first
    python -m flux.api.main

    # Run benchmark
    python scripts/benchmark.py

    # Save results
    python scripts/benchmark.py --save baseline.json
    python scripts/benchmark.py --save optimized.json

    # Compare
    python scripts/benchmark.py --compare baseline.json optimized.json
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    name: str = "benchmark"

    # Config
    num_requests: int = 0
    concurrency: int = 0
    batch_size: int = 0

    # Latency (milliseconds)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_avg: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0

    # Throughput
    total_time_s: float = 0.0
    requests_per_sec: float = 0.0

    # Errors
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print("\n" + "=" * 60)
        print(f"  BENCHMARK RESULTS: {self.name}")
        print("=" * 60)
        print(f"\n  Configuration:")
        print(f"    Requests:    {self.num_requests}")
        print(f"    Concurrency: {self.concurrency}")
        print(f"    Batch size:  {self.batch_size}")

        print(f"\n  Latency (ms):")
        print(f"    p50:  {self.latency_p50:>8.2f} ms")
        print(f"    p95:  {self.latency_p95:>8.2f} ms")
        print(f"    p99:  {self.latency_p99:>8.2f} ms")
        print(f"    avg:  {self.latency_avg:>8.2f} ms")
        print(f"    min:  {self.latency_min:>8.2f} ms")
        print(f"    max:  {self.latency_max:>8.2f} ms")

        print(f"\n  Throughput:")
        print(f"    Total time:  {self.total_time_s:.2f} s")
        print(f"    Requests/s:  {self.requests_per_sec:.1f}")

        print(f"\n  Results:")
        print(f"    Success: {self.success_count}")
        print(f"    Errors:  {self.error_count}")
        print(f"    Timeout: {self.timeout_count}")

        success_rate = (self.success_count / self.num_requests * 100) if self.num_requests > 0 else 0
        print(f"    Rate:    {success_rate:.1f}%")
        print("=" * 60)


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    timeout: float = 30.0,
) -> tuple[float, bool, Optional[str]]:
    """Make a single request and return (latency_ms, success, error)."""
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.status_code == 200:
            return elapsed_ms, True, None
        else:
            return elapsed_ms, False, f"HTTP {response.status_code}"

    except httpx.TimeoutException:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, False, "timeout"
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, False, str(e)


async def run_benchmark(
    url: str = "http://localhost:8000/predict",
    num_requests: int = 1000,
    concurrency: int = 10,
    batch_size: int = 1,
    model: str = "resnet50",
    name: str = "benchmark",
) -> BenchmarkResult:
    """Run the benchmark."""

    print(f"\nRunning benchmark: {name}")
    print(f"  URL: {url}")
    print(f"  Requests: {num_requests}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: {model}")

    # Create payload
    # Simulate image input (224x224x3 = 150528 floats, but we'll use smaller for testing)
    input_data = [[0.5] * 100 for _ in range(batch_size)]
    payload = {
        "model": model,
        "inputs": input_data,
    }

    latencies: list[float] = []
    success_count = 0
    error_count = 0
    timeout_count = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client: httpx.AsyncClient) -> tuple[float, bool, Optional[str]]:
        async with semaphore:
            return await make_request(client, url, payload)

    print("\n  Running...")
    start_time = time.perf_counter()

    async with httpx.AsyncClient() as client:
        # Create all tasks
        tasks = [bounded_request(client) for _ in range(num_requests)]

        # Run with progress
        completed = 0
        for coro in asyncio.as_completed(tasks):
            latency_ms, success, error = await coro
            latencies.append(latency_ms)

            if success:
                success_count += 1
            elif error == "timeout":
                timeout_count += 1
            else:
                error_count += 1

            completed += 1
            if completed % 100 == 0:
                print(f"    Progress: {completed}/{num_requests}")

    total_time = time.perf_counter() - start_time

    # Calculate statistics
    latencies.sort()

    def percentile(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    result = BenchmarkResult(
        name=name,
        num_requests=num_requests,
        concurrency=concurrency,
        batch_size=batch_size,
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        latency_p99=percentile(latencies, 99),
        latency_avg=statistics.mean(latencies),
        latency_min=min(latencies),
        latency_max=max(latencies),
        total_time_s=total_time,
        requests_per_sec=num_requests / total_time,
        success_count=success_count,
        error_count=error_count,
        timeout_count=timeout_count,
    )

    return result


def compare_results(baseline_path: str, optimized_path: str) -> None:
    """Compare two benchmark results."""

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(optimized_path) as f:
        optimized = json.load(f)

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPARISON")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Change':>12}")
    print("  " + "-" * 67)

    metrics = [
        ("Latency p50 (ms)", "latency_p50", True),
        ("Latency p95 (ms)", "latency_p95", True),
        ("Latency p99 (ms)", "latency_p99", True),
        ("Latency avg (ms)", "latency_avg", True),
        ("Throughput (req/s)", "requests_per_sec", False),
    ]

    for label, key, lower_is_better in metrics:
        base_val = baseline[key]
        opt_val = optimized[key]

        if base_val > 0:
            if lower_is_better:
                change_pct = (base_val - opt_val) / base_val * 100
                change_str = f"{change_pct:+.1f}%"
                if change_pct > 0:
                    change_str += " better"
            else:
                change_pct = (opt_val - base_val) / base_val * 100
                change_str = f"{change_pct:+.1f}%"
                if change_pct > 0:
                    change_str += " better"
        else:
            change_str = "N/A"

        print(f"  {label:<25} {base_val:>15.2f} {opt_val:>15.2f} {change_str:>12}")

    print("=" * 70)

    # Resume bullet suggestion
    throughput_improvement = optimized["requests_per_sec"] / baseline["requests_per_sec"]
    latency_improvement = baseline["latency_p99"] / optimized["latency_p99"]

    print("\n  SUGGESTED RESUME BULLET:")
    print("  " + "-" * 50)
    print(f"  Improved inference throughput {throughput_improvement:.1f}x")
    print(f"  ({baseline['requests_per_sec']:.0f} -> {optimized['requests_per_sec']:.0f} req/s)")
    print(f"  and reduced p99 latency {latency_improvement:.1f}x")
    print(f"  ({baseline['latency_p99']:.0f}ms -> {optimized['latency_p99']:.0f}ms)")
    print("=" * 70)


async def check_server() -> bool:
    """Check if the server is running."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health", timeout=5.0)
            return response.status_code == 200
    except:
        return False


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Flux inference")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="API URL")
    parser.add_argument("--requests", "-n", type=int, default=1000, help="Number of requests")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Inputs per request")
    parser.add_argument("--model", "-m", default="resnet50", help="Model name")
    parser.add_argument("--name", default="benchmark", help="Benchmark name")
    parser.add_argument("--save", type=str, help="Save results to JSON file")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "OPTIMIZED"),
                        help="Compare two result files")

    args = parser.parse_args()

    # Handle comparison mode
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    # Check server
    print("Checking server...")
    if not await check_server():
        print("\nERROR: Server not running at http://localhost:8000")
        print("Start it with: python -m flux.api.main")
        return
    print("Server is up!")

    # Also register a mock worker if none exist
    try:
        async with httpx.AsyncClient() as client:
            # Check workers
            resp = await client.get("http://localhost:8000/workers")
            workers = resp.json()

            if len(workers) == 0:
                print("No workers registered. Registering mock worker...")
                await client.post(
                    "http://localhost:8000/workers",
                    json={
                        "name": "benchmark-worker",
                        "host": "localhost",
                        "port": 8001,
                        "capacity": {"cpu": 8, "memory": 16384, "gpu": 1},
                        "gpu_memory_total_mb": 16000,
                        "loaded_models": [args.model],
                    }
                )
                print("Mock worker registered.")
    except Exception as e:
        print(f"Warning: Could not check/register workers: {e}")

    # Run benchmark
    result = await run_benchmark(
        url=args.url,
        num_requests=args.requests,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        model=args.model,
        name=args.name,
    )

    # Print results
    result.print_summary()

    # Save if requested
    if args.save:
        path = Path(args.save)
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"\nResults saved to: {path}")
        print(f"Compare later with: python scripts/benchmark.py --compare {path} optimized.json")


if __name__ == "__main__":
    asyncio.run(main())
