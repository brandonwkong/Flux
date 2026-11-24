#!/usr/bin/env python
"""Benchmark harness: real concurrent load against a running Flux server.

This is a first-class part of the project, not an afterthought - its job
is to answer "how does each scheduling policy actually behave under
concurrent load" honestly, with real HTTP requests hitting a real model.

Because SCHEDULING_POLICY is fixed when the server process starts (see
server.py), sweeping policies means restarting the server - this script
does that itself, once per policy, and reuses that one running server
across all of that policy's concurrency levels. One command runs the
whole sweep and writes one row per (policy, concurrency) combination to
CSV: latency percentiles, throughput, tokens/sec, deadline-hit rate
(blended and per traffic class), goodput, and the queue-time vs
compute-time split the server reports for every request.

Every request is tagged as one of two synthetic traffic classes -
"interactive" (high priority, tight deadline) or "batch" (low priority,
relaxed deadline) - using a fixed random seed, so PRIORITY and EDF have
something real to schedule on. The same seed means every (policy,
concurrency) run sees the identical request stream, so differences in
the results come from the policy, not from random variation in load.

Load model (--load-model):
    virtual (default) - `concurrency` virtual users, each looping
        send -> await response -> send next, pulling the next request
        off the same fixed-seed stream until num_requests total have
        completed. Client latency is measured per request, send to
        response, inside that loop - there is no artificial queue
        beyond genuine concurrency.
    burst - the old behavior: all num_requests created at once,
        gated through a semaphore of size `concurrency`. At low
        concurrency this means most requests sit in a long client-side
        line before their turn even starts, which that line's wait
        now counts as part of their latency (see git history - this
        used to be silently excluded, which was its own bug). Kept
        available for comparison, not the default.

Usage:
    # Default sweep: 3 policies x 5 concurrency levels, one command.
    python scripts/benchmark.py --output results.csv

    # Customize:
    python scripts/benchmark.py \
        --policies fifo,priority,edf \
        --concurrencies 1,2,4,8,16 \
        --requests 200 --repeats 3 \
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

WARMUP_REQUESTS = 3  # fired once per server start, discarded, so CUDA kernel
WARMUP_MAX_NEW_TOKENS = 8  # warmup (cudnn autotune, first-call JIT) doesn't pollute real measurements

# Two synthetic traffic classes so PRIORITY and EDF have something real to
# differentiate on. Fixed seed -> every (policy, concurrency, repeat) run
# generates the identical sequence of request types, so the comparison
# across policies isn't confounded by different random loads.
REQUEST_MIX_SEED = 1234
INTERACTIVE_SHARE = 0.3
INTERACTIVE_PRIORITY, INTERACTIVE_DEADLINE_MS = 15, 1500  # measured floor at concurrency=1 is ~900-1100ms compute alone
BATCH_PRIORITY, BATCH_DEADLINE_MS = 3, 8000

# Mixed workload (--workload mixed): interactive keeps the uniform workload's
# token count; batch gets a much longer generation, so there's actually a
# long request for a short one to get trapped behind under static batching -
# the uniform workload (every request exactly the same length) gives
# continuous batching nothing to rescue. The uniform workload's behavior is
# completely unaffected by any of this - see max_new_tokens_override below.
MIXED_INTERACTIVE_MAX_NEW_TOKENS = 32
MIXED_BATCH_MAX_NEW_TOKENS = 256  # trim target if ETA is too long: 256 -> 192


def _make_request(rng: random.Random, i: int, workload: str, mixed_batch_max_new_tokens: int) -> tuple:
    """Return (prompt, priority, deadline_ms, traffic_class, max_new_tokens_override).

    max_new_tokens_override is None for the uniform workload - callers fall
    back to their own shared max_new_tokens argument, so that path's
    behavior is byte-for-byte what it always was. For the mixed workload
    it's the per-class token count (interactive stays short, batch goes
    long), which is the entire point of this workload variant.
    """
    if rng.random() < INTERACTIVE_SHARE:
        override = MIXED_INTERACTIVE_MAX_NEW_TOKENS if workload == "mixed" else None
        return (
            f"Interactive {i}: answer in one short sentence - what color is the sky?",
            INTERACTIVE_PRIORITY,
            INTERACTIVE_DEADLINE_MS,
            "interactive",
            override,
        )
    override = mixed_batch_max_new_tokens if workload == "mixed" else None
    return (
        f"Batch {i}: tell me something interesting about space.",
        BATCH_PRIORITY,
        BATCH_DEADLINE_MS,
        "batch",
        override,
    )


def _request_stream(num_requests: int, workload: str = "uniform", mixed_batch_max_new_tokens: int = MIXED_BATCH_MAX_NEW_TOKENS) -> list[tuple]:
    """The fixed-seed request sequence for one run - identical across every
    policy, concurrency level, and repeat, so comparisons aren't confounded
    by different random loads."""
    rng = random.Random(REQUEST_MIX_SEED)
    return [_make_request(rng, i, workload, mixed_batch_max_new_tokens) for i in range(num_requests)]


@dataclass
class RunResult:
    """Everything measured for one (policy, concurrency) run."""

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
    interactive_latencies_ms: list[float] = field(default_factory=list)
    batch_latencies_ms: list[float] = field(default_factory=list)
    queue_times_ms: list[float] = field(default_factory=list)
    compute_times_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    output_tokens: list[int] = field(default_factory=list)

    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        ordered = sorted(data)
        idx = int(len(ordered) * p / 100)
        return ordered[min(idx, len(ordered) - 1)]

    def record(self, elapsed_ms: float, payload: Optional[dict], deadline_ms: int, traffic_class: str) -> None:
        """Tally one completed request (success or failure) into this run."""
        if traffic_class == "interactive":
            self.interactive_total += 1
        else:
            self.batch_total += 1

        if payload is None:
            self.error_count += 1
            return

        self.success_count += 1
        self.latencies_ms.append(elapsed_ms)
        if traffic_class == "interactive":
            self.interactive_latencies_ms.append(elapsed_ms)
        else:
            self.batch_latencies_ms.append(elapsed_ms)
        self.queue_times_ms.append(payload["queue_time_ms"])
        self.compute_times_ms.append(payload["compute_time_ms"])
        self.batch_sizes.append(payload["batch_size"])
        self.output_tokens.append(payload["output_tokens"])

        # Deadline success is judged on client-observed latency, not the
        # server's internal queue+compute view - that's the whole point
        # of measuring elapsed_ms at all.
        if elapsed_ms <= deadline_ms:
            self.deadline_hits += 1
            if traffic_class == "interactive":
                self.interactive_hits += 1
            else:
                self.batch_hits += 1

    def to_row(self) -> dict:
        requests_per_sec = self.success_count / self.total_time_s if self.total_time_s > 0 else 0.0
        deadline_met_fraction = self.deadline_hits / self.num_requests if self.num_requests else 0.0
        return {
            "policy": self.policy,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "requests_per_sec": requests_per_sec,
            "tokens_per_sec": (
                sum(self.output_tokens) / self.total_time_s if self.total_time_s > 0 else 0.0
            ),
            "avg_output_tokens": statistics.mean(self.output_tokens) if self.output_tokens else 0.0,
            "goodput_req_per_sec": requests_per_sec * deadline_met_fraction,
            "deadline_met_pct": 100.0 * deadline_met_fraction,
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
            "interactive_latency_p50_ms": self._percentile(self.interactive_latencies_ms, 50),
            "interactive_latency_avg_ms": (
                statistics.mean(self.interactive_latencies_ms) if self.interactive_latencies_ms else 0.0
            ),
            "batch_latency_p50_ms": self._percentile(self.batch_latencies_ms, 50),
            "batch_latency_avg_ms": (
                statistics.mean(self.batch_latencies_ms) if self.batch_latencies_ms else 0.0
            ),
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


async def warmup(num_requests: int = WARMUP_REQUESTS, max_new_tokens: int = WARMUP_MAX_NEW_TOKENS) -> None:
    """Fire a few throwaway requests right after server start, discarded, so
    CUDA kernel warmup (cudnn autotune, first-call JIT) doesn't land inside
    the first real measurement - a cold first request can run several times
    slower than a warm one."""
    async with httpx.AsyncClient() as client:
        for i in range(num_requests):
            await _one_request(client, f"Warmup {i}: hello.", max_new_tokens, priority=5, deadline_ms=5000)


async def run_virtual(
    policy: str,
    concurrency: int,
    num_requests: int,
    max_new_tokens: int,
    workload: str = "uniform",
    mixed_batch_max_new_tokens: int = MIXED_BATCH_MAX_NEW_TOKENS,
) -> RunResult:
    """Virtual-user load: `concurrency` workers, each looping
    send -> await response -> send next, pulling from the same fixed-seed
    stream until num_requests total have completed. No artificial queue
    beyond genuine concurrency - a worker's next send only happens once
    its previous response actually arrived.

    workload="uniform" (default) is exactly the original behavior: every
    request gets the same max_new_tokens. workload="mixed" gives each
    request its own max_new_tokens per traffic class (see _make_request)."""
    result = RunResult(policy=policy, concurrency=concurrency, num_requests=num_requests)
    request_stream = _request_stream(num_requests, workload, mixed_batch_max_new_tokens)
    next_index = 0

    def _next_params() -> Optional[tuple]:
        nonlocal next_index
        if next_index >= len(request_stream):
            return None
        params = request_stream[next_index]
        next_index += 1
        return params

    async def worker(client: httpx.AsyncClient) -> None:
        while True:
            params = _next_params()
            if params is None:
                return
            prompt, priority, deadline_ms, traffic_class, max_new_tokens_override = params
            effective_max_new_tokens = (
                max_new_tokens_override if max_new_tokens_override is not None else max_new_tokens
            )
            start = time.perf_counter()
            payload = await _one_request(client, prompt, effective_max_new_tokens, priority, deadline_ms)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.record(elapsed_ms, payload, deadline_ms, traffic_class)

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        workers = [asyncio.create_task(worker(client)) for _ in range(concurrency)]
        await asyncio.gather(*workers)
    result.total_time_s = time.perf_counter() - start

    return result


async def run_burst(
    policy: str,
    concurrency: int,
    num_requests: int,
    max_new_tokens: int,
    workload: str = "uniform",
    mixed_batch_max_new_tokens: int = MIXED_BATCH_MAX_NEW_TOKENS,
) -> RunResult:
    """Burst load (legacy default): all num_requests created at once, gated
    through a semaphore of size `concurrency`. At low concurrency, most
    requests sit in a long client-side line before their turn even starts -
    that wait is included in their latency (submit -> response, full stop),
    which is why burst mode and virtual-user mode give very different
    numbers at the same "concurrency" value. Kept for comparison."""
    result = RunResult(policy=policy, concurrency=concurrency, num_requests=num_requests)
    semaphore = asyncio.Semaphore(concurrency)
    request_stream = _request_stream(num_requests, workload, mixed_batch_max_new_tokens)

    async def bounded(client: httpx.AsyncClient, i: int) -> None:
        prompt, priority, deadline_ms, traffic_class, max_new_tokens_override = request_stream[i]
        effective_max_new_tokens = (
            max_new_tokens_override if max_new_tokens_override is not None else max_new_tokens
        )
        start = time.perf_counter()
        async with semaphore:
            payload = await _one_request(client, prompt, effective_max_new_tokens, priority, deadline_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000
        result.record(elapsed_ms, payload, deadline_ms, traffic_class)

    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, i) for i in range(num_requests)]
        await asyncio.gather(*tasks)
    result.total_time_s = time.perf_counter() - start

    return result


LOAD_MODELS = {"virtual": run_virtual, "burst": run_burst}


def _median_row(rows: list[dict], key: str = "requests_per_sec") -> dict:
    """Pick the whole row whose `key` is the median among repeats.

    Keeps every column internally consistent - all from one real run -
    rather than averaging each column independently across repeats, which
    could produce a row that never actually happened (e.g. one repeat's
    queue time paired with a different repeat's throughput).
    """
    ordered = sorted(rows, key=lambda r: r[key])
    return ordered[len(ordered) // 2]


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


def init_csv(path: Path) -> None:
    """Start this run's output file fresh - a stale file from a previous run
    at the same path must not get silently appended to."""
    if path.exists():
        path.unlink()


def append_csv_row(row: dict, path: Path) -> None:
    """Append one completed cell's row immediately, writing the header first
    if this is the first row. So that stopping or crashing the sweep midway
    never loses the cells that already finished - each is durable as soon
    as it completes, not held in memory until the very end."""
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


async def sweep(
    policies: list[str],
    concurrencies: list[int],
    num_requests: int,
    max_new_tokens: int,
    load_model: str,
    repeats: int,
    concurrency_1_requests: int,
    concurrency_1_repeats: int,
    output_path: Path,
    workload: str = "uniform",
    mixed_batch_max_new_tokens: int = MIXED_BATCH_MAX_NEW_TOKENS,
) -> list[dict]:
    """Run every (policy, concurrency) combination, restarting the server once
    per policy, repeating each combination `repeats` times and keeping the
    median run. concurrency=1 is a control point, not a load test - with
    only one request ever in flight, the three scheduling policies are
    behaviorally identical (nothing to ever schedule between), so it gets
    its own much smaller request/repeat count rather than the full size."""
    run_fn = LOAD_MODELS[load_model]
    rows = []
    for policy in policies:
        print(f"\n=== Starting server with SCHEDULING_POLICY={policy} ===")
        process = start_server(policy)
        try:
            wait_for_server()
            print("Server ready, warming up...")
            await warmup()
            print("Warmup complete.")

            for concurrency in concurrencies:
                if concurrency == 1:
                    cell_requests, cell_repeats = concurrency_1_requests, concurrency_1_repeats
                else:
                    cell_requests, cell_repeats = num_requests, repeats

                print(f"  policy={policy} concurrency={concurrency} requests={cell_requests} repeats={cell_repeats}")
                repeat_rows = []
                for _ in range(cell_repeats):
                    result = await run_fn(
                        policy, concurrency, cell_requests, max_new_tokens, workload, mixed_batch_max_new_tokens
                    )
                    repeat_rows.append(result.to_row())

                row = _median_row(repeat_rows)
                rows.append(row)
                append_csv_row(row, output_path)
                print(
                    f"    [median of {cell_repeats}] p50={row['latency_p50_ms']:.1f}ms  p99={row['latency_p99_ms']:.1f}ms  "
                    f"req/s={row['requests_per_sec']:.2f}  tok/s={row['tokens_per_sec']:.1f}  "
                    f"goodput={row['goodput_req_per_sec']:.2f}  avg_batch={row['avg_batch_size']:.1f}  "
                    f"deadline_met interactive={row['deadline_met_pct_interactive']:.1f}% "
                    f"batch={row['deadline_met_pct_batch']:.1f}%  "
                    f"queue_avg={row['queue_time_avg_ms']:.1f}ms  queue_p99={row['queue_time_p99_ms']:.1f}ms  "
                    f"errors={row['error_count']}  [saved]"
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
        "--requests", "-n", type=int, default=200, help="Requests per (policy, concurrency) combination"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Repeats per combination (except concurrency=1); the median run is kept"
    )
    parser.add_argument(
        "--concurrency-1-requests",
        type=int,
        default=50,
        help="Requests for the concurrency=1 cell specifically - a control point, not a load test",
    )
    parser.add_argument(
        "--concurrency-1-repeats",
        type=int,
        default=1,
        help="Repeats for the concurrency=1 cell specifically",
    )
    parser.add_argument(
        "--load-model",
        choices=["virtual", "burst"],
        default="virtual",
        help="virtual = N workers looping send/await/send-next (default); "
        "burst = all requests created at once, gated by a semaphore",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="max_new_tokens per request (uniform workload only)")
    parser.add_argument(
        "--workload",
        choices=["uniform", "mixed"],
        default="uniform",
        help="uniform (default) = every request gets --max-new-tokens, unchanged original behavior; "
        "mixed = interactive requests stay short (32 tokens), batch requests go long "
        "(--mixed-batch-max-new-tokens), so there's a long generation for a short one to get trapped behind",
    )
    parser.add_argument(
        "--mixed-batch-max-new-tokens",
        type=int,
        default=MIXED_BATCH_MAX_NEW_TOKENS,
        help="Batch-class max_new_tokens under --workload mixed (trim target if runtime is too long)",
    )
    parser.add_argument("--output", "-o", default="results.csv", help="CSV output path")
    args = parser.parse_args()

    policies = [p.strip() for p in args.policies.split(",")]
    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",")]
    output_path = Path(args.output)

    print(
        f"Sweeping {len(policies)} policies x {len(concurrencies)} concurrency levels "
        f"({args.repeats} repeats each at {args.requests} requests, except concurrency=1: "
        f"{args.concurrency_1_repeats} repeat(s) at {args.concurrency_1_requests} requests, "
        f"load_model={args.load_model}, workload={args.workload}) -> {output_path} "
        f"(written incrementally, one row per cell)"
    )

    init_csv(output_path)
    rows = asyncio.run(
        sweep(
            policies,
            concurrencies,
            args.requests,
            args.max_new_tokens,
            args.load_model,
            args.repeats,
            args.concurrency_1_requests,
            args.concurrency_1_repeats,
            output_path,
            args.workload,
            args.mixed_batch_max_new_tokens,
        )
    )

    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
