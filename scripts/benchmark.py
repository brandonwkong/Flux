#!/usr/bin/env python
"""Benchmark harness: real concurrent load against a running Flux server.

This is a first-class part of the project, not an afterthought - its job
is to answer "how does each scheduling policy actually behave under
concurrent load" honestly, with real HTTP requests hitting a real model.

Because SCHEDULING_POLICY is fixed when the server process starts (see
server.py), sweeping policies means restarting the server - this script
does that itself, once per policy, and reuses that one running server
across all of that policy's concurrency levels. One command runs the
whole sweep and writes one row per (policy, concurrency) combination
to CSV: latency percentiles, throughput, and the queue-time vs
compute-time split the server reports for every request.

Usage:
    # Default sweep: 3 policies x 5 concurrency levels, one command.
    python scripts/benchmark.py --output results.csv

    # Customize:
    python scripts/benchmark.py \
        --policies fifo,priority,edf \
        --concurrencies 1,2,4,8,16 \
        --requests 200 \
        --output results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

SERVER_URL = "http://127.0.0.1:8000"  # not "localhost": Windows httpx tries ::1 first and stalls ~2s before IPv4 fallback
STARTUP_TIMEOUT_S = 180.0  # TinyLlama load + first-run HF download can take a while


@dataclass
class RunResult:
    """Everything measured for one (policy, concurrency) combination."""

    policy: str
    concurrency: int
    num_requests: int
    success_count: int = 0
    error_count: int = 0
    total_time_s: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    queue_times_ms: list[float] = field(default_factory=list)
    compute_times_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        ordered = sorted(data)
        idx = int(len(ordered) * p / 100)
        return ordered[min(idx, len(ordered) - 1)]

    def to_row(self) -> dict:
        return {
            "policy": self.policy,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "requests_per_sec": (
                self.success_count / self.total_time_s if self.total_time_s > 0 else 0.0
            ),
            "latency_p50_ms": self._percentile(self.latencies_ms, 50),
            "latency_p95_ms": self._percentile(self.latencies_ms, 95),
            "latency_p99_ms": self._percentile(self.latencies_ms, 99),
            "latency_avg_ms": statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0,
            "queue_time_avg_ms": statistics.mean(self.queue_times_ms) if self.queue_times_ms else 0.0,
            "compute_time_avg_ms": statistics.mean(self.compute_times_ms) if self.compute_times_ms else 0.0,
            "avg_batch_size": statistics.mean(self.batch_sizes) if self.batch_sizes else 0.0,
        }


async def _one_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_new_tokens: int,
) -> tuple[float, Optional[dict]]:
    """Fire one request, return (client-measured latency_ms, response JSON or None on failure)."""
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{SERVER_URL}/generate",
            json={"prompt": prompt, "max_new_tokens": max_new_tokens},
            timeout=60.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code == 200:
            return elapsed_ms, response.json()
        return elapsed_ms, None
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, None


async def run_one(
    policy: str,
    concurrency: int,
    num_requests: int,
    max_new_tokens: int,
) -> RunResult:
    """Fire num_requests requests at the (already running) server, bounded by a semaphore."""
    result = RunResult(policy=policy, concurrency=concurrency, num_requests=num_requests)
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(client: httpx.AsyncClient, i: int) -> tuple[float, Optional[dict]]:
        async with semaphore:
            return await _one_request(
                client, f"Request {i}: tell me something interesting about space.", max_new_tokens
            )

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, i) for i in range(num_requests)]
        for coro in asyncio.as_completed(tasks):
            elapsed_ms, payload = await coro
            if payload is not None:
                result.success_count += 1
                result.latencies_ms.append(elapsed_ms)
                result.queue_times_ms.append(payload["queue_time_ms"])
                result.compute_times_ms.append(payload["compute_time_ms"])
                result.batch_sizes.append(payload["batch_size"])
            else:
                result.error_count += 1

    result.total_time_s = time.perf_counter() - start
    return result


def start_server(policy: str) -> subprocess.Popen:
    """Launch the server as a subprocess with SCHEDULING_POLICY set for this run."""
    env = os.environ.copy()
    env["SCHEDULING_POLICY"] = policy
    return subprocess.Popen([sys.executable, "-m", "flux.server"], env=env)


def wait_for_server(timeout: float = STARTUP_TIMEOUT_S) -> None:
    """Poll /health until the server (and its model) is ready, or raise on timeout."""
    deadline = time.perf_counter() + timeout
    with httpx.Client() as client:
        while time.perf_counter() < deadline:
            try:
                response = client.get(f"{SERVER_URL}/health", timeout=2.0)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
    raise RuntimeError(f"Server did not become healthy within {timeout}s")


def stop_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print("No results to write.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


async def sweep(
    policies: list[str],
    concurrencies: list[int],
    num_requests: int,
    max_new_tokens: int,
) -> list[dict]:
    """Run every (policy, concurrency) combination, restarting the server once per policy."""
    rows = []
    for policy in policies:
        print(f"\n=== Starting server with SCHEDULING_POLICY={policy} ===")
        process = start_server(policy)
        try:
            wait_for_server()
            print("Server ready.")

            for concurrency in concurrencies:
                print(f"  policy={policy} concurrency={concurrency} requests={num_requests}")
                result = await run_one(policy, concurrency, num_requests, max_new_tokens)
                row = result.to_row()
                rows.append(row)
                print(
                    f"    p50={row['latency_p50_ms']:.1f}ms  p99={row['latency_p99_ms']:.1f}ms  "
                    f"req/s={row['requests_per_sec']:.2f}  avg_batch={row['avg_batch_size']:.1f}  "
                    f"errors={result.error_count}"
                )
        finally:
            stop_server(process)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Flux under concurrent load, swept across scheduling policies"
    )
    parser.add_argument("--policies", default="fifo,priority,edf", help="Comma-separated policies")
    parser.add_argument("--concurrencies", default="1,2,4,8,16", help="Comma-separated concurrency levels")
    parser.add_argument(
        "--requests", "-n", type=int, default=100, help="Requests per (policy, concurrency) combination"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="max_new_tokens per request")
    parser.add_argument("--output", "-o", default="results.csv", help="CSV output path")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",")]
    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",")]

    print(
        f"Sweeping {len(policies)} policies x {len(concurrencies)} concurrency levels "
        f"({len(policies) * len(concurrencies)} runs, {args.requests} requests each)"
    )

    rows = asyncio.run(sweep(policies, concurrencies, args.requests, args.max_new_tokens))

    output_path = Path(args.output)
    write_csv(rows, output_path)
    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
