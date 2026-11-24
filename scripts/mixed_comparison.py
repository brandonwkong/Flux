#!/usr/bin/env python
"""Mixed-length-workload static-vs-continuous comparison for Flux.

Measurement script ONLY - no changes to engine.py, scheduler.py, or
benchmark.py's existing behavior. Reuses benchmark.py's run_virtual()
(now workload-aware), warmup(), wait_for_server(), stop_server(),
_median_row() exactly as written. Shares controlled_comparison.py's
server-switching trick (PYTHONPATH pointed at either the current
checkout's continuous engine or a git worktree pinned at the last
pre-continuous-batching commit for static) - neither codebase is
touched.

New in this script versus controlled_comparison.py: continuous 5-second
GPU logging for the full duration of each run (boot through teardown),
not just before/after snapshots - closing the gap where the controlled
run's snapshots never actually sampled mid-run.

Usage:
    python scripts/mixed_comparison.py --pilot
        One cell (continuous, FIFO, concurrency=8, mixed workload,
        100 requests) - reports pace and ETA, no CSV written.

    python scripts/mixed_comparison.py --full --output mixed_sweep.csv
        Runs the full protocol: concurrency 8 and 16, 3 policies x
        3 repeats x both engines, GPU-logged throughout.
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
import benchmark  # noqa: E402  - reused unmodified
import controlled_comparison as cc  # noqa: E402  - reused: gpu_snapshot, start_server, rotate

POLICIES = ["fifo", "priority", "edf"]
CONCURRENCIES = [8, 16]
MAX_NEW_TOKENS_BASE = 32  # fallback only - mixed workload overrides every request's max_new_tokens per class
GPU_LOG_INTERVAL_S = 5.0

# Throttle bits that indicate a REAL problem (thermal/power limiting the
# clock), as opposed to the benign idle/active flags that are always
# present. See docs/continuous_batching_notes.md-adjacent investigation.
BAD_THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80  # hw_slowdown | sw_thermal | hw_thermal | hw_power_brake


def _safe_float(s) -> float:
    try:
        return float(s)
    except Exception:
        return 0.0


def _is_bad_throttle(hex_str: str) -> bool:
    try:
        return bool(int(hex_str, 16) & BAD_THROTTLE_MASK)
    except Exception:
        return False


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


async def gpu_logger(context: dict, rows: list[dict], stop_event: asyncio.Event) -> None:
    """Poll GPU state every GPU_LOG_INTERVAL_S seconds for the duration of
    one run, tagging each sample with engine/policy/concurrency/repeat so
    a contaminated moment can be traced to the exact run afterward."""
    loop = asyncio.get_event_loop()
    while not stop_event.is_set():
        snap = await loop.run_in_executor(None, cc.gpu_snapshot)
        snap.update(context)
        snap["sample_unix_time"] = time.time()
        rows.append(snap)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=GPU_LOG_INTERVAL_S)
        except asyncio.TimeoutError:
            pass


async def run_one_cell(
    engine: str,
    policy: str,
    concurrency: int,
    num_requests: int,
    repeat: int,
    mixed_batch_max_new_tokens: int,
    gpu_log_path: Path,
) -> dict:
    """Boot one server, GPU-log continuously throughout, run the mixed-
    workload load test, tear down. Returns the result row with engine
    name and a GPU-behavior summary folded in; full per-sample GPU log
    goes to gpu_log_path."""
    process = cc.start_server(engine, policy)
    gpu_rows: list[dict] = []
    stop_event = asyncio.Event()
    context = {"engine": engine, "policy": policy, "concurrency": concurrency, "repeat": repeat}
    logger_task = asyncio.create_task(gpu_logger(context, gpu_rows, stop_event))

    try:
        benchmark.wait_for_server()
        await benchmark.warmup()
        result = await benchmark.run_virtual(
            policy, concurrency, num_requests, MAX_NEW_TOKENS_BASE, "mixed", mixed_batch_max_new_tokens
        )
    finally:
        stop_event.set()
        await logger_task
        benchmark.stop_server(process)

    for row in gpu_rows:
        append_csv_row(row, gpu_log_path)

    out_row = result.to_row()
    out_row["engine"] = engine
    out_row["repeat"] = repeat
    if gpu_rows:
        out_row["gpu_sample_count"] = len(gpu_rows)
        out_row["gpu_max_util_pct"] = max(_safe_float(r["gpu_util_pct"]) for r in gpu_rows)
        out_row["gpu_max_temp_c"] = max(_safe_float(r["gpu_temp_c"]) for r in gpu_rows)
        out_row["gpu_min_sm_clock_mhz"] = min(_safe_float(r["gpu_sm_clock_mhz"]) for r in gpu_rows)
        out_row["gpu_max_proc_count"] = max(r["gpu_proc_count"] for r in gpu_rows)
        out_row["gpu_bad_throttle"] = any(_is_bad_throttle(r["gpu_throttle_reasons"]) for r in gpu_rows)
    else:
        out_row["gpu_sample_count"] = 0
        out_row["gpu_max_util_pct"] = ""
        out_row["gpu_max_temp_c"] = ""
        out_row["gpu_min_sm_clock_mhz"] = ""
        out_row["gpu_max_proc_count"] = ""
        out_row["gpu_bad_throttle"] = ""

    return out_row


async def pilot(num_requests: int, mixed_batch_max_new_tokens: int) -> dict:
    print(
        f"=== Pilot cell: continuous, FIFO, concurrency=8, mixed workload "
        f"(interactive=32tok, batch={mixed_batch_max_new_tokens}tok), {num_requests} requests ==="
    )
    start = time.perf_counter()
    row = await run_one_cell(
        "continuous", "fifo", 8, num_requests, repeat=0,
        mixed_batch_max_new_tokens=mixed_batch_max_new_tokens,
        gpu_log_path=Path("mixed_pilot_gpu_log.csv"),
    )
    elapsed = time.perf_counter() - start

    print(f"\nPilot cell wall time (boot + warmup + {num_requests} requests + teardown): {elapsed:.1f}s")
    print(f"  req/s reported: {row['requests_per_sec']:.2f}")
    print(f"  tokens/s reported: {row['tokens_per_sec']:.1f}")
    print(f"  interactive_latency_p50_ms: {row['interactive_latency_p50_ms']:.1f}")
    print(f"  batch_latency_p50_ms: {row['batch_latency_p50_ms']:.1f}")
    print(f"  gpu_max_util_pct: {row['gpu_max_util_pct']}  gpu_bad_throttle: {row['gpu_bad_throttle']}")
    return row, elapsed


def estimate_eta(pilot_elapsed_s: float, num_requests: int, repeats: dict[int, int]) -> float:
    """Uses the pilot's OWN measured wall time (boot+warmup+load+teardown)
    directly as the per-run cost, for both concurrency levels - a rough
    estimate given mixed-workload token volume is ~6x the uniform
    workload's, not a precise prediction."""
    total_s = 0.0
    for concurrency in CONCURRENCIES:
        n_runs = 3 * repeats[concurrency] * 2  # 3 policies x repeats x (static, continuous)
        total_s += n_runs * pilot_elapsed_s
    return total_s / 60


async def full_run(
    output_path: Path,
    gpu_log_path: Path,
    num_requests: int,
    repeats: dict[int, int],
    mixed_batch_max_new_tokens: int,
    eta_gate_minutes: float = 75.0,
    eta_check_after_runs: int = 4,
) -> None:
    """No separate pilot this time - projects ETA from the first
    eta_check_after_runs actual runs (results already written incrementally
    by then) and stops if the projection exceeds eta_gate_minutes. Everyone
    completed before that point stays in the CSV either way."""
    init_csv(output_path)
    init_csv(gpu_log_path)

    cell_visits: list[tuple[int, int, str]] = []
    for concurrency in CONCURRENCIES:
        for rep_idx in range(repeats[concurrency]):
            policy_order = cc.rotate(POLICIES, rep_idx)
            for policy in policy_order:
                cell_visits.append((concurrency, rep_idx, policy))
    total_runs = len(cell_visits) * 2

    sweep_start = time.perf_counter()
    runs_done = 0
    first_engine = "continuous"
    stopped_early = False

    for concurrency, rep_idx, policy in cell_visits:
        engines_order = [first_engine, "static" if first_engine == "continuous" else "continuous"]
        print(
            f"\n--- concurrency={concurrency} policy={policy} repeat={rep_idx} "
            f"order={engines_order} requests={num_requests} ---"
        )
        for engine in engines_order:
            print(f"  running engine={engine} ...")
            row = await run_one_cell(
                engine, policy, concurrency, num_requests, rep_idx, mixed_batch_max_new_tokens, gpu_log_path
            )
            append_csv_row(row, output_path)
            runs_done += 1
            print(
                f"    req/s={row['requests_per_sec']:.2f} "
                f"interactive_p50={row['interactive_latency_p50_ms']:.1f}ms "
                f"deadline_interactive={row['deadline_met_pct_interactive']:.1f}% "
                f"gpu_max_util={row['gpu_max_util_pct']}% "
                f"gpu_bad_throttle={row['gpu_bad_throttle']} [saved]"
            )

            if runs_done == eta_check_after_runs:
                elapsed_s = time.perf_counter() - sweep_start
                avg_s_per_run = elapsed_s / runs_done
                projected_min = avg_s_per_run * total_runs / 60
                print(
                    f"\n=== ETA projection after {runs_done} runs: avg {avg_s_per_run:.1f}s/run, "
                    f"projected total {projected_min:.1f} min (gate: {eta_gate_minutes} min) ==="
                )
                if projected_min > eta_gate_minutes:
                    print(
                        f"STOPPING: projected {projected_min:.1f}min exceeds the {eta_gate_minutes}min gate. "
                        f"{runs_done}/{total_runs} runs completed and saved in {output_path}."
                    )
                    stopped_early = True
                    break
        if stopped_early:
            break
        first_engine = engines_order[1]

    status = "Stopped early (ETA gate)" if stopped_early else "Done"
    print(f"\n{status}. {runs_done}/{total_runs} runs in {output_path}, GPU log in {gpu_log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mixed-length-workload static-vs-continuous comparison")
    parser.add_argument("--pilot", action="store_true", help="Run only the pilot cell and report ETA")
    parser.add_argument("--full", action="store_true", help="Run the full protocol")
    parser.add_argument("--requests", type=int, default=100, help="Requests per cell")
    parser.add_argument(
        "--mixed-batch-max-new-tokens", type=int, default=256, help="Batch-class max_new_tokens (trim target: 192)"
    )
    parser.add_argument("--output", default="mixed_sweep.csv", help="CSV output path")
    parser.add_argument("--gpu-log", default="mixed_sweep_gpu_log.csv", help="Continuous GPU log CSV path")
    args = parser.parse_args()

    repeats = {8: 3, 16: 3}

    if args.pilot:
        (row, elapsed) = asyncio.run(pilot(args.requests, args.mixed_batch_max_new_tokens))
        eta_min = estimate_eta(elapsed, args.requests, repeats)
        n_runs = sum(3 * r * 2 for r in repeats.values())
        print(f"\nEstimated total runs in plan: {n_runs}")
        print(f"Rough ETA for full plan: {eta_min:.1f} minutes")
    elif args.full:
        asyncio.run(
            full_run(Path(args.output), Path(args.gpu_log), args.requests, repeats, args.mixed_batch_max_new_tokens)
        )
    else:
        print("Specify --pilot or --full")


if __name__ == "__main__":
    main()
