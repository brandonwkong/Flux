# Flux

A single-process LLM inference runtime, built to learn inference-runtime engineering —
batching, scheduling, KV cache — and eventually kernel-level optimization. Not a
platform: one process, one model (TinyLlama-1.1B), one GPU.

## Architecture

```
POST /generate
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                            FLUX                              │
│                                                               │
│   RequestQueue                Engine                         │
│   ┌──────────────┐    pick    ┌──────────────────────────┐   │
│   │ req ─ Future  │◄──next────│ scheduler.select_batch()  │   │
│   │ req ─ Future  │   batch   │  FIFO / PRIORITY / EDF    │   │
│   │ req ─ Future  │           │           │               │   │
│   └──────────────┘           │           ▼               │   │
│                               │   one padded forward pass │   │
│                               │   through TinyLlama (GPU) │   │
│                               └──────────────┬─────────────┘   │
│                                              │                 │
│                               resolves each Future with        │
│                               text + queue_time_ms +            │
│                               compute_time_ms + batch_size      │
└─────────────────────────────────────────────────────────────┘
```

Two concerns kept deliberately separate:

- **Scheduling** (`scheduler.py`) decides *which* pending requests get pulled into the
  next batch, given more are waiting than fit. Swappable at process start: FIFO, PRIORITY
  (highest priority first, FIFO tie-break), EDF (earliest absolute deadline first).
- **Batching** (`engine.py`) decides *how many* requests run together in one GPU forward
  pass, and lives inside the engine's own loop — not a separate pre-worker stage.

Each request gets an `asyncio.Future` the instant it's enqueued (`queue.py`); the HTTP
handler just awaits it. Whether the engine ran that request alone, first, last, or
batched with five others is invisible to the caller.

## Quick Start

```bash
cd flux
pip install -e ".[dev]"

python -m flux.server
```

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_new_tokens": 16}'
# {"text": "Paris.\n...", "queue_time_ms": 26.5, "compute_time_ms": 495.5, "batch_size": 1}
```

Use `127.0.0.1`, not `localhost` — on Windows, httpx tries IPv6 (`::1`) first and stalls
~2s before falling back to IPv4.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Submit a prompt, block until the engine returns text + timing |
| GET | `/health` | Health check |
| GET | `/stats` | Queue depth, active policy, model, device |

`GenerateRequest`: `prompt`, `max_new_tokens`, `priority` (read by the PRIORITY policy),
`deadline_ms` (read by the EDF policy).

`GenerateResponse`: `text`, `queue_time_ms` (time spent waiting before this request's
batch started), `compute_time_ms` (wall-clock time of the shared forward pass — the same
value for every request in that batch, since that's literally how much GPU time they
shared), `batch_size`.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `FLUX_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HF model id |
| `MAX_BATCH_SIZE` | 8 | Max requests per forward pass |
| `MAX_WAIT_MS` | 20 | How long the engine waits for stragglers to join a forming batch |
| `MAX_QUEUE_DEPTH` | 1000 | Backpressure threshold (503 past this) |
| `SCHEDULING_POLICY` | `fifo` | `fifo` \| `priority` \| `edf` — **read once at startup** |

Changing `SCHEDULING_POLICY` means restarting the process, same as changing the model.

## Benchmarking

`scripts/benchmark.py` is a first-class part of this project, not an afterthought: it
answers "how does each scheduling policy actually behave under concurrent load,"
honestly, against the real model.

```bash
# Default sweep: 3 policies x 5 concurrency levels, one command.
python scripts/benchmark.py --output results.csv
```

Since the policy is fixed at startup, the script manages the server itself — restarting
it once per policy (reusing that one process across all of that policy's concurrency
levels) and writing one row per (policy, concurrency) combination: latency percentiles,
throughput, and the average queue-time/compute-time split pulled straight from every
response body.

Measured on an RTX 4070 Laptop GPU, TinyLlama-1.1B, 16 new tokens, after warmup: a single
request's shared forward pass takes ~500ms; six concurrent requests land in one batch and
share a ~765ms forward pass — ~4x the throughput of running them one at a time. That's
the whole point of batching, with real numbers instead of simulated ones.

## Project Structure

```
flux/
├── src/flux/
│   ├── schemas.py     # GenerateRequest/GenerateResponse - the wire format
│   ├── queue.py        # RequestQueue: async pool, each request carries a Future
│   ├── scheduler.py    # select_batch(): FIFO / PRIORITY / EDF, swappable
│   ├── engine.py        # Loads TinyLlama, owns the batch loop, times queue vs compute
│   └── server.py        # FastAPI app: POST /generate, GET /health, GET /stats
├── scripts/
│   └── benchmark.py     # Policy x concurrency sweep -> CSV
├── tests/
│   ├── test_queue.py
│   └── test_scheduler.py
└── pyproject.toml
```

## Known Limitations

Real gaps in the current design, worth knowing before quoting a number or planning the
next piece of work:

- **Request-level batching, not continuous batching.** A batch runs to completion (every
  sequence finishes its `generate()` call) before the engine can form the next one — a
  request that arrives mid-batch waits for the whole batch to finish, even if its own
  generation would be short. Real systems (vLLM, TensorRT-LLM) let new requests join and
  finished ones leave every decode step. That's the natural next target once KV cache
  management is in place.
- **Shared `max_new_tokens` per batch.** One `generate()` call uses the largest
  `max_new_tokens` asked for in the batch; a request asking for fewer tokens still waits
  for the longest one in its batch, then gets truncated down to what it asked for.
- **No KV cache reuse across requests or batches.** Each `generate()` call manages its
  own ephemeral KV cache internally; there's no prefix sharing, paging, or cache-aware
  scheduling yet. That's the KV-cache half of this project's learning goal, still ahead.
- **Single GPU, single model, single process — by design.** This is deliberately not a
  platform. Docker/Prometheus/Grafana were cut from the core path; they can come back as
  optional extras once the runtime itself has real numbers to show.

## Roadmap

The two end artifacts this project is building toward:

1. An LLM inference runtime benchmarked honestly under concurrent load (the sweep above,
   with real percentiles and a real queue/compute split - not simulated numbers).
2. One custom profiled CUDA kernel with before-and-after numbers and a roofline
   explanation for *why* it helped.

Roughly in order from here:

- **Continuous batching** - let requests join/leave a forming batch every decode step,
  not just between whole-batch `generate()` calls.
- **KV-cache-aware scheduling** - track per-request KV cache memory as the real
  batching constraint (token budget, not request count), and route/prioritize
  accordingly.
- **Token-level metrics** - TTFT (time to first token), TPOT (time per output token),
  tok/s, and goodput-vs-SLO as the headline numbers instead of end-to-end latency alone.
- **Kernel-level optimization** - profile one decode step with `torch.profiler` /
  Nsight Systems (needs WSL2 - Triton's compiler is Linux-first), find what's actually
  memory-bound, write one Triton kernel (fused RMSNorm+RoPE is a reasonable first
  target), and report before/after TPOT with a roofline explanation.

Deprioritized on purpose: auto-retry, autoscaling, worker pools, scheduler HA - none of
that teaches batching, scheduling, KV cache, or kernels, which is the actual point.
