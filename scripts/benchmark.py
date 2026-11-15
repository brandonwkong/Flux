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
to CSV: latency percentiles, throughput, deadline-hit rate, and the
queue-time vs compute-time split the server reports for every request.

Every request is tagged as one of two synthetic traffic classes -
"interactive" (high priority, tight deadline) or "batch" (low priority,
relaxed deadline) - using a fixed random seed, so PRIORITY and EDF have
something real to schedule on. Without this, every request would look
identical and all three policies would just tie-break to arrival order
(i.e. behave exactly like FIFO). The same seed means every (policy,
concurrency) run sees the identical request stream, so differences in
the results come from the policy, not from random variation in load.

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
import random
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

# Two synthetic traffic classes so PRIORITY and EDF have something real to
# differentiate on. Fixed seed -> every (policy, concurrency) run generates
# the identical sequence of request types, so the comparison across policies
# isn't confounded by different random loads.
REQUEST_MIX_SEED = 1234
INTERACTIVE_SHARE = 0.3
INTERACTIVE_PRIORITY, INTERACTIVE_DEADLINE_MS = 15, 800
BATCH_PRIORITY, BATCH_DEADLINE_MS = 3, 8000


def _make_request(rng: random.Random, i: int) -> tuple[str, int, int, str]:
    """Return (prompt, priority, deadline_ms, traffic_class) for request i from the fixed two-class mix."""
    if rng.random() < INTERACTIVE_SHARE:
        return (
            f"Interactive {i}: answer in one short sentence - what color is the sky?",
            INTERACTIVE_PRIORITY,
            INTERACTIVE_DEADLINE_MS,
            "interactive",
        )
    return (
        f"Batch {i}: tell me something interesting about space.",
        BATCH_PRIORITY,
        BATCH_DEADLINE_MS,
        "batch",
    )


@dataclass
class RunResult:
    """Everything measured for one (policy, concurrency) combination."""

    policy: str
    concurrency: int
    num_requests: int
    success_count: int = 0
    error_count: int = 0
    deadline_hits: int = 0
    interactive_total: int = 0
    interactive_hits: int = 0
    batch_total: int = 0
    batch_hits: int = 0
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
            "deadline_met_pct": (
                100.0 * self.deadline_hits / self.num_requests if self.num_requests else 0.0
            ),
            "deadline_met_pct_interactive": (
                100.0 * self.interactive_hits / self.interactive_total if self.interactive_total else 0.0
            ),
            "deadline_met_pct_batch": (
                100.0 * self.batch_hits / self.batch_total if self.batch_total else 0.0
            ),
            "latency_p50_ms": self._percentile(self.latencies_ms, 50),
            "latency_p95_ms": self._percentile(self.latencies_ms, 95),
            "latency_p99_ms": self._percentile(self.latencies_ms, 99),
            "latency_avg_ms": statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0,
            "queue_time_avg_ms": statistics.mean(self.queue_times_ms) if self.queue_times_ms else 0.0,
            "queue_time_p50_ms": self._percentile(self.queue_times_ms, 50),
            "queue_time_p95_ms": self._percentile(self.queue_times_ms, 95),
            "queue_time_p99_ms": self._percentile(self.queue_times_ms, 99),
            "compute_time_avg_ms": statistics.mean(self.compute_times_ms) if self.compute_times_ms else 0.0,
            "avg_batch_size": statistics.mean(self.batch_sizes) if self.batch_sizes else 0.0,
        }


async def _one_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_new_tokens: int,
    priority: int,
    deadline_ms: int,
) -> Optional[dict]:
    """Fire one request, return response JSON or None on failure."""
    try:
        response = await client.post(
            f"{SERVER_URL}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "priority": priority,
                "deadline_ms": deadline_ms,
            },
            timeout=60.0,
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


async def run_one(
    policy: str,
    concurrency: int,
    num_requests: int,
    max_new_tokens: int,
) -> RunResult:
    """Fire num_requests requests at the (already running) server, bounded by a semaphore."""
    result = RunResult(policy=policy, concurrency=concurrency, num_requests=num_requests)
    semaphore = asyncio.Semaphore(concurrency)

    # Precomputed synchronously, before any concurrency starts, so the sequence
    # of request types is identical across every (policy, concurrency) run.
    rng = random.Random(REQUEST_MIX_SEED)
    request_params = [_make_request(rng, i) for i in range(num_requests)]

    async def bounded(client: httpx.AsyncClient, i: int) -> tuple[float, Optional[dict], int, str]:
        prompt, priority, deadline_ms, traffic_class = request_params[i]
        # Client latency means submit -> response, full stop - timer starts
        # before the semaphore, so time spent queued in THIS client (waiting
        # for a concurrency slot) counts too, not just time on the wire.
        start = time.perf_counter()
        async with semaphore:
            payload = await _one_request(client, prompt, max_new_tokens, priority, deadline_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms, payload, deadline_ms, traffic_class

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, i) for i in range(num_requests)]
        for coro in asyncio.as_completed(tasks):
            elapsed_ms, payload, deadline_ms, traffic_class = await coro
            if traffic_class == "interactive":
                result.interactive_total += 1
            else:
                result.batch_total += 1

            if payload is not None:
                result.success_count += 1
                result.latencies_ms.append(elapsed_ms)
                result.queue_times_ms.append(payload["queue_time_ms"])
                result.compute_times_ms.append(payload["compute_time_ms"])
                result.batch_sizes.append(payload["batch_size"])
                # Deadline success is judged on client-observed latency, not
                # the server's internal queue+compute view - that's the
                # whole point of measuring elapsed_ms at all.
                if elapsed_ms <= deadline_ms:
                    result.deadline_hits += 1
                    if traffic_class == "interactive":
                        result.interactive_hits += 1
                    else:
                        result.batch_hits += 1
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
                    f"deadline_met interactive={row['deadline_met_pct_interactive']:.1f}% "
                    f"batch={row['deadline_met_pct_batch']:.1f}%  "
                    f"queue_avg={row['queue_time_avg_ms']:.1f}ms  queue_p99={row['queue_time_p99_ms']:.1f}ms  "
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
