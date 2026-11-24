#!/usr/bin/env python
"""Controlled static-vs-continuous comparison for Flux.

Measurement script ONLY. Does not modify engine.py, scheduler.py, or
benchmark.py - it imports and calls benchmark.py's run_virtual(),
warmup(), wait_for_server(), stop_server(), _median_row() exactly as
written. The only new logic here is: which server binary runs for a
given cell (the current checkout's continuous engine, or a git worktree
pinned at the last pre-continuous-batching commit, running the static
engine, via PYTHONPATH - no code in either is touched), interleaving
the two back-to-back per cell, alternating start order, rotating policy
order across repeats, and logging GPU state around every run.

Usage:
    python scripts/controlled_comparison.py --pilot
        Runs exactly one cell (continuous, FIFO, concurrency=8, 200
        requests) and reports timing - no CSV written, no full plan run.

    python scripts/controlled_comparison.py --full --output controlled_sweep.csv
        Runs the full protocol.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark  # noqa: E402  - reused unmodified: run_virtual, warmup, wait_for_server, stop_server, _median_row

REPO_ROOT = SCRIPT_DIR.parent
CONTINUOUS_SRC = str(REPO_ROOT / "src")
STATIC_SRC = str(REPO_ROOT.parent / "flux-static-worktree" / "src")

POLICIES = ["fifo", "priority", "edf"]
CONCURRENCY_REQUESTS = {8: 100, 16: 200}  # user's trimmed cut: drop 1/2/4, c=8 halved, c=16 full
CONCURRENCY_REPEATS = {8: 3, 16: 3}
MAX_NEW_TOKENS = 32


def gpu_snapshot() -> dict:
    """Same fields as the standalone GPU diagnostic: utilization%, memory,
    SM clock, power draw, temperature, throttle reasons, process count/names."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        util, mem, sm_clock, power, temp, throttle = [x.strip() for x in out.split(",")]

        procs_out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).strip()
        proc_lines = [line for line in procs_out.splitlines() if line.strip()]
        proc_names = ";".join(Path(line.split(",", 1)[1].strip()).name for line in proc_lines)

        return {
            "gpu_util_pct": util,
            "gpu_mem_used_mib": mem,
            "gpu_sm_clock_mhz": sm_clock,
            "gpu_power_w": power,
            "gpu_temp_c": temp,
            "gpu_throttle_reasons": throttle,
            "gpu_proc_count": len(proc_lines),
            "gpu_proc_names": proc_names,
        }
    except Exception as e:
        return {
            "gpu_util_pct": "", "gpu_mem_used_mib": "", "gpu_sm_clock_mhz": "", "gpu_power_w": "",
            "gpu_temp_c": "", "gpu_throttle_reasons": "", "gpu_proc_count": "", "gpu_proc_names": "",
            "gpu_error": str(e),
        }


def start_server(engine: str, policy: str) -> subprocess.Popen:
    """engine is "static" (worktree, pre-continuous-batching code) or
    "continuous" (current checkout). Neither codebase is modified - this
    just points PYTHONPATH at the one to run."""
    env = os.environ.copy()
    env["SCHEDULING_POLICY"] = policy
    env["PYTHONPATH"] = STATIC_SRC if engine == "static" else CONTINUOUS_SRC
    return subprocess.Popen([sys.executable, "-m", "flux.server"], env=env, cwd=str(REPO_ROOT))


def port_8000_occupant_pid() -> str | None:
    """Returns the PID holding a LISTENING socket on port 8000, or None if
    free. Exists because of a real bug caught while building the mixed-
    workload comparison: a crashed earlier run left a server process
    alive, and wait_for_server() kept "succeeding" against that stale
    process for every subsequent cell - silently mislabeling every row
    with the wrong engine/policy. This is the pre-flight half of the fix;
    verify_server_policy below is the post-start half."""
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, timeout=5)
    except Exception:
        return None
    for line in out.splitlines():
        if "LISTENING" in line and ":8000 " in line.replace("\t", " "):
            parts = line.split()
            if parts:
                return parts[-1]
    return None


def ensure_port_free(engine: str, policy: str, retries: int = 5, retry_delay_s: float = 1.0) -> None:
    """Raise loudly, before starting anything, if port 8000 is still
    occupied after a short grace period - rather than silently proceeding
    and talking to whatever's already there.

    The grace period exists because of a real false alarm caught while
    building this: the previous cell's stop_server() had already returned
    (process.wait() confirmed the process exited), but Windows hadn't
    fully released the LISTENING socket for a brief moment afterward -
    this isn't a stuck process, just normal OS teardown lag. A genuinely
    leaked process (the original bug this function exists for) stays
    occupied across the whole retry window; a normal one clears within
    the first second or two."""
    occupant = None
    for attempt in range(retries):
        occupant = port_8000_occupant_pid()
        if occupant is None:
            return
        time.sleep(retry_delay_s)

    raise RuntimeError(
        f"Port 8000 is still held by PID {occupant} after {retries * retry_delay_s:.0f}s of waiting, "
        f"before we even tried to start engine={engine} policy={policy} - a leaked server from an "
        f"earlier run. Refusing to proceed (would silently talk to the wrong server). Kill PID "
        f"{occupant} and re-run."
    )


def verify_server_policy(expected_policy: str) -> None:
    """After wait_for_server() succeeds, confirm /stats reports the policy
    we actually asked for - catches a stale server from a prior cell
    answering in place of the one we think we just started."""
    with httpx.Client() as client:
        response = client.get(f"{benchmark.SERVER_URL}/stats", timeout=5.0)
    actual_policy = response.json().get("policy")
    if actual_policy != expected_policy:
        raise RuntimeError(
            f"Server integrity check failed: expected policy={expected_policy!r} but "
            f"/stats reports policy={actual_policy!r}. This almost certainly means a "
            f"stale server from an earlier run is still listening on port 8000 - "
            f"stopping rather than silently recording mismatched data."
        )


def rotate(items: list[str], n: int) -> list[str]:
    n = n % len(items)
    return items[n:] + items[:n]


def init_csv(path: Path) -> None:
    if path.exists():
        path.unlink()


def append_csv_row(row: dict, path: Path) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


async def run_one_cell(engine: str, policy: str, concurrency: int, num_requests: int) -> dict:
    """Boot one server, warm it up, run the load test, tear it down.
    Returns the result row with engine name and GPU state before/after folded in."""
    ensure_port_free(engine, policy)
    gpu_before = gpu_snapshot()
    process = start_server(engine, policy)
    try:
        benchmark.wait_for_server()
        verify_server_policy(policy)
        await benchmark.warmup()
        result = await benchmark.run_virtual(policy, concurrency, num_requests, MAX_NEW_TOKENS)
    finally:
        benchmark.stop_server(process)
    gpu_after = gpu_snapshot()

    row = result.to_row()
    row["engine"] = engine
    for k, v in gpu_before.items():
        row[f"gpu_before_{k}"] = v
    for k, v in gpu_after.items():
        row[f"gpu_after_{k}"] = v
    return row


async def pilot(num_requests: int = 100) -> dict:
    print(f"=== Pilot cell: continuous, FIFO, concurrency=8, {num_requests} requests ===")
    start = time.perf_counter()
    row = await run_one_cell("continuous", "fifo", 8, num_requests)
    elapsed = time.perf_counter() - start

    per_request_s = elapsed / num_requests
    print(f"\nPilot cell wall time (boot + warmup + {num_requests} requests + teardown): {elapsed:.1f}s")
    print(f"  -> {per_request_s * 1000:.1f} ms/request measured pace")
    print(f"  req/s reported: {row['requests_per_sec']:.2f}")
    return row


def estimate_trimmed_eta(req_per_sec_c8: float) -> float:
    """Trimmed plan: concurrency 8 (100 req) and 16 (200 req), 3 policies x
    3 repeats x 2 engines each. Uses the measured c=8 req/s as a stand-in
    for c=16 too (historically similar magnitude for this workload) - a
    rough estimate, not a precise prediction."""
    boot_overhead_s = 15.0  # observed typical model-load + warmup cost
    total_s = 0.0
    for concurrency, num_requests in CONCURRENCY_REQUESTS.items():
        repeats = CONCURRENCY_REPEATS[concurrency]
        load_s = num_requests / req_per_sec_c8
        per_run_s = boot_overhead_s + load_s
        n_runs = 3 * repeats * 2  # 3 policies x repeats x (static, continuous)
        total_s += n_runs * per_run_s
    return total_s / 60


async def full_run(output_path: Path) -> None:
    init_csv(output_path)
    first_engine = "continuous"

    for concurrency in [8, 16]:
        repeats = CONCURRENCY_REPEATS[concurrency]
        num_requests = CONCURRENCY_REQUESTS[concurrency]

        for rep_idx in range(repeats):
            policy_order = rotate(POLICIES, rep_idx)
            for policy in policy_order:
                engines_order = [first_engine, "static" if first_engine == "continuous" else "continuous"]
                print(
                    f"\n--- concurrency={concurrency} policy={policy} repeat={rep_idx} "
                    f"order={engines_order} requests={num_requests} ---"
                )
                for engine in engines_order:
                    print(f"  running engine={engine} ...")
                    row = await run_one_cell(engine, policy, concurrency, num_requests)
                    row["repeat"] = rep_idx
                    append_csv_row(row, output_path)
                    print(
                        f"    req/s={row['requests_per_sec']:.2f} "
                        f"deadline_interactive={row['deadline_met_pct_interactive']:.1f}% "
                        f"compute_avg={row['compute_time_avg_ms']:.1f}ms "
                        f"gpu_before={row['gpu_before_gpu_util_pct']}% "
                        f"gpu_after={row['gpu_after_gpu_util_pct']}% [saved]"
                    )
                first_engine = engines_order[1]  # alternate who starts next

    print(f"\nDone. Results in {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled static-vs-continuous comparison")
    parser.add_argument("--pilot", action="store_true", help="Run only the pilot cell and report ETA")
    parser.add_argument("--full", action="store_true", help="Run the full protocol")
    parser.add_argument("--output", default="controlled_sweep.csv", help="CSV output path")
    args = parser.parse_args()

    if args.pilot:
        row = asyncio.run(pilot(num_requests=CONCURRENCY_REQUESTS[8]))
        eta_min = estimate_trimmed_eta(row["requests_per_sec"])
        print(f"\nEstimated total runs in trimmed plan: {3 * sum(CONCURRENCY_REPEATS.values()) * 2}")
        print(f"Rough ETA for trimmed plan (c=8, c=16 only): {eta_min:.1f} minutes")
    elif args.full:
        asyncio.run(full_run(Path(args.output)))
    else:
        print("Specify --pilot or --full")


if __name__ == "__main__":
    main()
