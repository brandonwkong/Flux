#!/usr/bin/env python
"""
Quick standalone benchmark - no server needed.

This simulates different scenarios to show what numbers you'd get:
1. No batching (naive)
2. With batching (optimized)
3. With batching + warm models (fully optimized)

Use these as reference targets for your real implementation.

Usage:
    python scripts/quick_benchmark.py
"""

import asyncio
import random
import statistics
import time
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    name: str
    inference_time_ms: float      # Time to run inference
    batch_overhead_ms: float      # Overhead per batch
    per_item_time_ms: float       # Additional time per item in batch
    model_load_time_ms: float     # Cold start penalty
    batching_enabled: bool
    max_batch_size: int
    batch_timeout_ms: float


# Simulation scenarios
SCENARIOS = [
    SimulationConfig(
        name="No Batching (Baseline)",
        inference_time_ms=10.0,
        batch_overhead_ms=0,
        per_item_time_ms=10.0,
        model_load_time_ms=500.0,
        batching_enabled=False,
        max_batch_size=1,
        batch_timeout_ms=0,
    ),
    SimulationConfig(
        name="With Batching",
        inference_time_ms=10.0,
        batch_overhead_ms=5.0,
        per_item_time_ms=0.5,    # GPU parallelism!
        model_load_time_ms=500.0,
        batching_enabled=True,
        max_batch_size=32,
        batch_timeout_ms=10.0,
    ),
    SimulationConfig(
        name="Batching + Warm Models",
        inference_time_ms=8.0,    # Optimized
        batch_overhead_ms=3.0,
        per_item_time_ms=0.3,
        model_load_time_ms=0.0,  # Already loaded
        batching_enabled=True,
        max_batch_size=32,
        batch_timeout_ms=5.0,
    ),
]


async def simulate_inference(config: SimulationConfig, batch_size: int, is_cold: bool) -> float:
    """Simulate inference and return latency in ms."""

    # Cold start penalty
    if is_cold:
        await asyncio.sleep(config.model_load_time_ms / 1000)
        cold_penalty = config.model_load_time_ms
    else:
        cold_penalty = 0

    # Inference time
    if config.batching_enabled:
        # Batching: fixed overhead + small per-item cost (GPU parallelism)
        inference_time = config.batch_overhead_ms + (config.per_item_time_ms * batch_size)
    else:
        # No batching: each item takes full inference time
        inference_time = config.per_item_time_ms * batch_size

    # Add some jitter
    jitter = random.uniform(0.8, 1.2)
    total_time = (cold_penalty + inference_time) * jitter

    await asyncio.sleep(total_time / 1000)
    return total_time


async def run_scenario(config: SimulationConfig, num_requests: int = 500) -> dict:
    """Run a benchmark scenario."""

    print(f"\n  Running: {config.name}")
    print(f"  Requests: {num_requests}")

    latencies = []
    start_time = time.perf_counter()

    # First request is cold
    is_cold = True

    if config.batching_enabled:
        # Simulate batching: group requests
        batch_size = config.max_batch_size
        num_batches = (num_requests + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_requests - i * batch_size)

            # Wait for batch to fill (simulated)
            await asyncio.sleep(config.batch_timeout_ms / 1000)

            # Run inference
            latency = await simulate_inference(config, current_batch_size, is_cold)
            is_cold = False

            # Each item in batch gets the same latency
            for _ in range(current_batch_size):
                latencies.append(latency + config.batch_timeout_ms)

            if (i + 1) % 10 == 0:
                print(f"    Batch {i + 1}/{num_batches}")
    else:
        # No batching: each request is individual
        for i in range(num_requests):
            latency = await simulate_inference(config, 1, is_cold)
            is_cold = False
            latencies.append(latency)

            if (i + 1) % 100 == 0:
                print(f"    Request {i + 1}/{num_requests}")

    total_time = time.perf_counter() - start_time

    # Calculate stats
    latencies.sort()

    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return {
        "name": config.name,
        "num_requests": num_requests,
        "total_time_s": total_time,
        "requests_per_sec": num_requests / total_time,
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
        "latency_p99": percentile(latencies, 99),
        "latency_avg": statistics.mean(latencies),
    }


def print_comparison(results: list[dict]) -> None:
    """Print comparison table."""

    print("\n" + "=" * 80)
    print("  BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n  {'Scenario':<30} {'Throughput':>12} {'p50':>10} {'p99':>10}")
    print(f"  {'':30} {'(req/s)':>12} {'(ms)':>10} {'(ms)':>10}")
    print("  " + "-" * 76)

    baseline = results[0]

    for r in results:
        speedup = r["requests_per_sec"] / baseline["requests_per_sec"]
        latency_reduction = baseline["latency_p99"] / r["latency_p99"]

        print(f"  {r['name']:<30} {r['requests_per_sec']:>12.1f} {r['latency_p50']:>10.1f} {r['latency_p99']:>10.1f}")

        if r != baseline:
            print(f"  {'  -> Improvement:':<30} {speedup:>11.1f}x {'':>10} {latency_reduction:>9.1f}x")

    print("=" * 80)

    # Resume suggestions
    best = results[-1]
    print("\n  RESUME BULLET SUGGESTIONS:")
    print("  " + "-" * 50)

    throughput_improvement = best["requests_per_sec"] / baseline["requests_per_sec"]
    latency_improvement = baseline["latency_p99"] / best["latency_p99"]

    print(f"""
  Option 1 (throughput focus):
  "Improved inference throughput {throughput_improvement:.0f}x through dynamic
   batching and model warm pools ({baseline['requests_per_sec']:.0f} -> {best['requests_per_sec']:.0f} req/s)"

  Option 2 (latency focus):
  "Reduced p99 latency {latency_improvement:.0f}x with GPU-optimized batching
   ({baseline['latency_p99']:.0f}ms -> {best['latency_p99']:.0f}ms)"

  Option 3 (combined):
  "Built ML inference platform achieving {best['requests_per_sec']:.0f} req/s
   at <{best['latency_p99']:.0f}ms p99 through dynamic batching"
""")
    print("=" * 80)


async def main():
    print("=" * 80)
    print("  FLUX BENCHMARK SIMULATION")
    print("  (Simulated numbers - run against real server for actual results)")
    print("=" * 80)

    results = []
    for scenario in SCENARIOS:
        result = await run_scenario(scenario)
        results.append(result)

    print_comparison(results)


if __name__ == "__main__":
    asyncio.run(main())
