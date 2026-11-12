# Flux Architecture

> A single-process LLM inference runtime for learning inference-runtime engineering:
> batching, scheduling, KV cache, and eventually kernel-level optimization. See
> [Known Limitations](#known-limitations) for what's deliberately not built yet.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Request Flow](#request-flow)
3. [Components](#components)
4. [Key Concepts](#key-concepts)
5. [File Structure](#file-structure)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Known Limitations](#known-limitations)
9. [Benchmarking](#benchmarking)

---

## System Overview

Flux is one process, serving one model, on one GPU. There is no worker registry, no
placement engine, no network hop between "scheduler" and "worker" - the scheduler and
the model live in the same Python process, in the same event loop.

```
                        Many concurrent clients
                                   |
                                   v
    +---------------------------------------------------------------+
    |                              FLUX                              |
    |                                                                |
    |   +---------+      +--------------+      +------------------+  |
    |   | server  |----->| RequestQueue |<-----| Engine batch loop |  |
    |   | /generate|     | (Future per  |      | - select_batch() |  |
    |   +---------+      |  request)    |      | - pad + forward  |  |
    |                    +--------------+      |   pass (GPU)     |  |
    |                                            +--------+---------+  |
    |                                                     |            |
    |                                           resolves each Future   |
    |                                           with text + timing     |
    +---------------------------------------------------------------+
```

### Why this shape

The platform version of this project (worker registry, heartbeats, bin-packing
placement across machines) solved a problem this project doesn't have: running
inference across multiple machines. That's a distributed-systems problem, orthogonal to
the actual learning goal here, which is what happens *inside* a single inference
process - how requests get batched, how batches get scheduled, and eventually how the
GPU kernels underneath them get optimized. Cutting the distributed layer entirely was
the first and largest decision in this codebase's history; see git log for the
commit-by-commit cut.

---

## Request Flow

### Step-by-Step Walkthrough

```
1. CLIENT REQUEST
   POST /generate {"prompt": "...", "max_new_tokens": 16}
                                |
                                v
2. SERVER (server.py)
   - Validates request (GenerateRequest)
   - Calls queue.submit(...) and awaits the result
                                |
                                v
3. QUEUE (queue.py)
   - Creates an asyncio.Future for this request
   - Wraps it in a QueuedRequest (prompt, params, arrived_at, the Future)
   - Appends to the pending pool, signals the engine something arrived
   - The server's await now blocks here until the Future resolves
                                |
                                v
4. ENGINE BATCH LOOP (engine.py)
   - Wakes on the arrival signal
   - Waits up to MAX_WAIT_MS for stragglers to join the forming batch
   - Calls scheduler.select_batch(pending, MAX_BATCH_SIZE, policy) -
     this is where FIFO / PRIORITY / EDF actually differ
   - Removes the selected requests from the queue's pending pool
                                |
                                v
5. MODEL FORWARD PASS (engine.py._generate, off the event loop thread)
   - Tokenizes all selected prompts together, left-padded
   - One generate() call, max_new_tokens = the largest ask in the batch
   - Decodes each sequence, truncated back to its own requested length
                                |
                                v
6. RESOLUTION
   - queue_time_ms computed per request: (batch start - request arrival)
   - compute_time_ms computed once for the whole batch (shared forward pass)
   - Each request's Future resolves with its own GenerateResponse
   - The server's await from step 2 returns; HTTP response goes out
```

### Timing Example

```
t=0ms      Request A arrives, enters the queue, Future created
t=3ms      Request B arrives
t=20ms     MAX_WAIT_MS elapsed - engine forms a batch of [A, B]
t=20ms     scheduler.select_batch() orders/selects them (FIFO here: both fit)
t=20ms     Forward pass starts (compute_start)
t=520ms    Forward pass finishes (500ms shared compute)
           A's queue_time_ms = 20 - 0  = 20ms,  compute_time_ms = 500ms
           B's queue_time_ms = 20 - 3  = 17ms,  compute_time_ms = 500ms
t=520ms    Both Futures resolve; both HTTP responses go out
```

Note what's *not* simulated here: `compute_time_ms` is the actual wall-clock duration of
one real `model.generate()` call, measured with `time.perf_counter()` around the
blocking call. It is the same number for every request in that batch because that's
genuinely how much shared GPU time they consumed - not a per-request slice invented to
look fair.

---

## Components

### 1. Server (`server.py`)

The entry point. Deliberately does almost nothing: builds one `RequestQueue` and one
`Engine` at startup (`lifespan`), and `/generate` is a single `await queue.submit(...)`.
All the engineering lives downstream of this file, not in it.

### 2. Queue (`queue.py`)

`RequestQueue` is an unordered pool of `QueuedRequest`, not `asyncio.Queue` - `asyncio.
Queue` is strict FIFO, and FIFO is only one of the policies the engine can run under.
Each `QueuedRequest` carries its own `asyncio.Future`; `submit()` creates that Future,
appends the request, and awaits the Future. Nothing resolves it except the engine.

This is the mechanism that decouples "a request arrived" from "a request was computed."
The HTTP handler blocking on `await future` has no idea whether its request ran alone,
first, last, or batched with five others - that decision happens entirely downstream,
and the Future is what lets the handler not care.

### 3. Scheduler (`scheduler.py`)

One function, `select_batch(pending, max_batch_size, policy)`, and three sort keys
behind it:

| Policy | Ordering | Use case |
|---|---|---|
| `FIFO` | Oldest arrival first | Fairness, simplicity |
| `PRIORITY` | Highest `priority` first, FIFO tie-break | Some requests matter more |
| `EDF` | Earliest absolute deadline (`arrived_at + deadline_ms`) first | Latency-sensitive requests under load |

Swapping policy is passing a different enum value into `select_batch` - the engine's
batch loop doesn't change. This is the classic OS-scheduling-policy pattern applied to
request ordering, deliberately kept separate from *how many* requests run together
(that's batching, the engine's concern).

### 4. Engine (`engine.py`)

Owns the model and the batch loop. On `start()`, it loads the tokenizer and model onto
the configured device (blocking, run off the event loop via `run_in_executor` so model
load doesn't freeze the server). The batch loop:

1. Waits for at least one pending request (or a 1s timeout, to notice shutdown).
2. Gives stragglers up to `MAX_WAIT_MS` to join - a short busy-poll on queue depth
   against a deadline, the same collect-until-full-or-timeout idea the project's earlier
   platform version used in a separate `Batcher` class, now folded into this loop
   directly.
3. Asks the scheduler which pending requests to pull.
4. Removes them from the queue and runs them as one padded batch through the model, off
   the event loop thread.
5. Resolves every request's Future with its own `GenerateResponse`.

---

## Key Concepts

### 1. Scheduling vs. Batching

These are two different questions, kept in two different files on purpose:

- **Scheduling** (`scheduler.py`): given more requests are waiting than fit in one
  batch, *which ones* go in this round? This is what FIFO/PRIORITY/EDF answer.
- **Batching** (`engine.py`): *how many* requests run together in one GPU forward pass,
  and how long do we wait for more to arrive before giving up and running what we have?

A continuous-batching engine (vLLM, TensorRT-LLM) answers "how many" at every decode
step instead of once per whole-sequence `generate()` call - see
[Known Limitations](#known-limitations). The scheduling question stays the same shape
either way, which is why it's useful to have factored it out now.

### 2. The Future-Carrying Queue

Every request gets an `asyncio.Future` the moment it's enqueued. This is what lets one
coroutine (`/generate`'s handler) block on *its own* result while a completely different
coroutine (the engine's batch loop) decides, independently, when and with whom that
request actually runs. Without the Future, the handler would have to poll, or the queue
would have to be strict FIFO so "my turn" was predictable - neither works once
scheduling policy is swappable.

### 3. Honest Timing

`queue_time_ms` and `compute_time_ms` are measured with `time.perf_counter()` around the
actual wait and the actual forward pass - not estimated, not simulated. The project's
earlier platform version had a benchmark script (`quick_benchmark.py`, since deleted)
that generated plausible-looking numbers from `asyncio.sleep()` calls and printed
"resume bullet suggestions." That script never touched a real server. This project's
numbers come from a real model on real hardware instead, every time.

### 4. Backpressure

`RequestQueue.submit()` raises `RuntimeError` once `depth >= max_depth`, which `server.
py` turns into a 503. Same shape as the old platform's backpressure check, just against
the queue directly instead of a separate batcher's queue depth.

---

## File Structure

```
flux/
├── src/flux/
│   ├── schemas.py     # GenerateRequest / GenerateResponse - the wire format
│   ├── queue.py         # RequestQueue, QueuedRequest - the Future-carrying pool
│   ├── scheduler.py     # SchedulingPolicy, select_batch() - FIFO / PRIORITY / EDF
│   ├── engine.py         # Engine - model load, batch loop, queue/compute timing
│   └── server.py         # FastAPI app - POST /generate, GET /health, GET /stats
├── tests/
│   ├── test_queue.py       # RequestQueue behavior, no model needed
│   └── test_scheduler.py   # Policy ordering guarantees, no model needed
├── scripts/
│   └── benchmark.py         # Policy x concurrency sweep -> CSV
├── pyproject.toml
├── README.md
└── ARCHITECTURE.md           # This file
```

Flat on purpose - a handful of files, not nested packages, matching the project's actual
size. The earlier platform version had `api/`, `batcher/`, `scheduler/` (as a package),
`worker/`, and `models/` as separate directories; all of that collapsed into five files
once the distributed layer was cut.

---

## API Reference

### POST /generate

**Request:**
```json
{
  "prompt": "The capital of France is",
  "max_new_tokens": 16,
  "priority": 5,
  "deadline_ms": 5000
}
```

**Response:**
```json
{
  "text": "Paris.\n\n2. B. The capital of Germany is Berlin.\n",
  "queue_time_ms": 26.5,
  "compute_time_ms": 495.5,
  "batch_size": 1
}
```

`priority` is only read by the `PRIORITY` policy; `deadline_ms` only by `EDF`. Sending
them under `FIFO` is harmless - they're just ignored.

### GET /stats

```json
{
  "queue_depth": 2,
  "policy": "fifo",
  "max_batch_size": 8,
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "device": "cuda"
}
```

---

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `FLUX_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hugging Face model id |
| `MAX_BATCH_SIZE` | 8 | Max requests per forward pass |
| `MAX_WAIT_MS` | 20 | Straggler-collection window per batch |
| `MAX_QUEUE_DEPTH` | 1000 | Backpressure threshold |
| `SCHEDULING_POLICY` | `fifo` | `fifo` \| `priority` \| `edf` - read once at process start |

---

## Known Limitations

Real, current gaps - not oversights to be embarrassed about, but the actual edges of
what's built so far:

- **Request-level batching, not continuous batching.** One `generate()` call runs an
  entire batch to completion (every sequence finishes) before the engine can form the
  next one. A request arriving mid-batch waits for the whole batch to finish even if its
  own generation would have been short. Real continuous-batching engines let requests
  join and leave every decode step - that requires managing KV cache per request across
  steps, which this project hasn't built yet.
- **Shared `max_new_tokens` per batch.** `_generate()` uses `max(r.max_new_tokens for r
  in batch)` for the one shared `generate()` call, then truncates each sequence back
  down. A request asking for 4 tokens batched with one asking for 64 waits for the full
  64 before it gets its (correctly truncated) 4 back.
- **No KV cache reuse across requests or batches.** Each `generate()` call manages its
  own ephemeral cache internally and discards it afterward. No prefix sharing, no
  paging, no cache-aware admission or routing.
- **No streaming.** `/generate` returns the full completion in one response; no
  token-by-token streaming, so there's no TTFT (time to first token) metric yet -
  `compute_time_ms` covers the whole generation, not just the first token.
- **Single GPU, single model, single process.** By design - see
  [System Overview](#system-overview).

---

## Benchmarking

`scripts/benchmark.py` runs a full sweep - every scheduling policy against every
concurrency level - from one command, and writes one row per combination to CSV.

```bash
python scripts/benchmark.py --output results.csv
```

Because `SCHEDULING_POLICY` is fixed at server startup, the script restarts the server
once per policy (reusing that one process across all of that policy's concurrency
levels, so the model loads once per policy, not once per run) and fires
`--requests` HTTP requests at `/generate` through a concurrency-bounded semaphore for
each level. Every row carries: success/error counts, requests/sec, latency p50/p95/p99,
and the average `queue_time_ms`/`compute_time_ms`/`batch_size` pulled straight from the
response bodies - not estimated separately.

Verified on an RTX 4070 Laptop GPU: a single request's forward pass (16 new tokens,
TinyLlama-1.1B, fp16, post-warmup) takes ~500ms; six requests fired concurrently land in
one batch and share a ~765ms forward pass - about 4x the throughput of running them one
at a time. `client-measured latency` on a *fresh* connection can look ~2s inflated on
Windows if the client dials `localhost` instead of `127.0.0.1` (IPv6-first connection
attempt stalls before falling back) - `benchmark.py` already targets `127.0.0.1`
directly for this reason.

Closed-loop caveat: like the platform version's original benchmark script, this one
bounds concurrency with a semaphore rather than firing at a fixed arrival rate. Fine for
comparing scheduling policies against each other under load; not sufficient for an
open-loop "sustained N req/s" claim, which would need a separate fixed-rate load
generator.
