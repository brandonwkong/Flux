# Flux

A single-process LLM inference runtime, built to learn inference-runtime engineering —
batching, scheduling, KV cache — and eventually kernel-level optimization. Not a
platform: one process, one model (TinyLlama-1.1B), one GPU.

The engine runs **continuous (iteration-level) batching**: between every single decode
step it admits newly-arrived requests and evicts anyone who just finished, resolving
each one's Future immediately — not at the end of some larger, shared `generate()` call
everyone had to wait for. An earlier static-batching version (whole batches run to
completion together) was replaced after being benchmarked against it head-to-head; see
[Benchmarking](#benchmarking) for the real numbers and the (workload-dependent) verdict.

## Architecture

```
POST /generate
      │
      ▼
┌──────────────────────────────────────────────────────────────────┐
│                              FLUX                                 │
│                                                                    │
│   RequestQueue                      Engine (_run_loop, every step) │
│   ┌──────────────┐    admit up to  ┌───────────────────────────┐   │
│   │ req ─ Future  │◄──free slots───│ scheduler.select_batch()  │   │
│   │ req ─ Future  │   (scheduler   │  FIFO / PRIORITY / EDF     │   │
│   │ req ─ Future  │    order)      │           │                │   │
│   └──────────────┘                 │           ▼                │   │
│                                     │   ContinuousBatch.step():  │   │
│                                     │   one decode step across   │   │
│                                     │   all active sequences,    │   │
│                                     │   finished ones evicted    │   │
│                                     └──────────────┬─────────────┘   │
│                                                     │                 │
│                                      resolves each finished Future    │
│                                      with text + per-sequence          │
│                                      queue_time_ms + compute_time_ms + │
│                                      avg batch_size it ran alongside   │
└──────────────────────────────────────────────────────────────────┘
```

Two concerns kept deliberately separate:

- **Scheduling** (`scheduler.py`) decides *which* pending requests get pulled into the
  active set next, given more are waiting than fit. Swappable at process start: FIFO,
  PRIORITY (highest priority first, FIFO tie-break), EDF (earliest absolute deadline
  first). Unchanged by the continuous-batching migration — the same `select_batch()` now
  just gets called every decode step instead of once per whole batch.
- **Batching** (`decode.py`'s `ContinuousBatch`, driven by `engine.py`'s loop) decides
  *how many and which* sequences share one GPU forward pass, and that membership can
  change every step — a request admitted mid-step joins the very next decode step; a
  request that finishes is evicted and its Future resolved immediately, without waiting
  for anyone else still generating.

Each request gets an `asyncio.Future` the instant it's enqueued (`queue.py`); the HTTP
handler just awaits it. Whether the engine ran that request alone, admitted it mid-step
next to five others already decoding, or evicted it early because it hit its own
`max_new_tokens` while its neighbors kept going, is invisible to the caller.

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

`GenerateResponse`: `text`, `queue_time_ms` (time from arrival to admission into the
active batch), `compute_time_ms` (admission to finish — genuinely per-request now, since
continuous batching has no single "batch" that starts and ends together), `batch_size`
(the average number of active sequences this request's generation ran alongside, across
every decode step it was part of).

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `FLUX_MODEL` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HF model id |
| `MAX_BATCH_SIZE` | 8 | Max concurrently-active sequences in the continuous batch |
| `MAX_QUEUE_DEPTH` | 1000 | Backpressure threshold (503 past this) |
| `SCHEDULING_POLICY` | `fifo` | `fifo` \| `priority` \| `edf` — **read once at startup** |

Changing `SCHEDULING_POLICY` means restarting the process, same as changing the model.

## Benchmarking

`scripts/benchmark.py` is a first-class part of this project, not an afterthought: it
answers "how does each scheduling policy actually behave under concurrent load,"
honestly, against the real model. It models virtual users issuing requests under a
traffic mix of two classes — **interactive** (short, tight deadline, high priority) and
**batch** (long, loose deadline, low priority) — and tracks per-class latency, deadline
compliance, throughput, and the queue-time/compute-time split pulled straight from every
response body.

```bash
# Default sweep: 3 policies x 5 concurrency levels, uniform-length workload.
python scripts/benchmark.py --output results.csv

# Mixed-length workload: interactive requests (32 tokens) and batch requests
# (256 tokens, same traffic mix otherwise) — see below for why this exists.
python scripts/benchmark.py --workload mixed --output results.csv
```

### Static vs. continuous batching: a controlled, workload-dependent result

Flux originally used static (request-level) batching — a whole `generate()` call ran to
completion for every sequence in a batch before the engine could form the next one. It
was replaced with continuous (iteration-level) batching (see Architecture above). Rather
than assume continuous batching is strictly better — it's the harder design to get right,
and real systems adopt it for a reason — both engines were benchmarked head-to-head under
identical conditions: same GPU, same model, alternating start order, rotated policy order
across repeats, GPU state logged throughout each run to rule out thermal/power
contamination (`scripts/controlled_comparison.py`, `scripts/mixed_comparison.py`).

**Uniform-length workload** (`controlled_sweep.csv` — every request asks for the same 32
tokens, so there's nothing for fine-grained admission to rescue): static batching won on
raw throughput and latency at both concurrency levels tested.

| Concurrency | Engine | req/s (avg) | latency p50 (avg, ms) |
|---|---|---|---|
| 8 | static | 6.56 | ~1,090 |
| 8 | continuous | 5.89 | ~1,155 |
| 16 | static | 7.59 | ~2,093 |
| 16 | continuous | 5.82 | ~2,711 |

At concurrency 16, continuous batching had ~23% lower throughput and ~30% higher median
latency than static — the per-step overhead of continuous batching (admitting/evicting
every decode step, concatenating and slicing the KV cache to change membership) has
nothing to buy back when every request is the same length. This result held up under a
GPU diagnostic (clocks, power, temperature, throttle flags) run before launching, and
across 3 repeats with rotated ordering — it isn't noise.

**Mixed-length workload** (`mixed_sweep.csv` — interactive requests ask for 32 tokens,
batch requests ask for 192-256; the traffic mix otherwise matches): continuous batching
won decisively on the metric that's the actual point of having two traffic classes —
interactive requests meeting their deadline instead of queuing behind a long batch
request.

| Concurrency | Engine | interactive deadline-met % (avg across policies/repeats) |
|---|---|---|
| 8 | static | **0%** (every single run) |
| 8 | continuous | **~62%** |
| 16 | static | **0%** (every single run) |
| 16 | continuous | **~15%** |

Under static batching, an interactive request arriving mid-batch has no way to avoid
waiting for every long batch request already running in that batch to finish — it missed
its deadline in all 18 runs at both concurrency levels, with no exceptions. Continuous
batching lets a short request get admitted (and finish) between decode steps, without
waiting for anyone else's generation to complete — the exact mechanism static batching
lacks. Compliance drops as concurrency rises (62% → 15%) because admission itself becomes
the bottleneck once `MAX_BATCH_SIZE` active slots are saturated — a request still has to
wait for a slot to free up, continuous batching just frees slots far more often than
static does.

**The honest overall conclusion is workload-dependent, not a blanket "continuous is
better":** continuous batching loses on a uniform-length workload (pure per-step overhead,
nothing to rescue) and wins decisively on a mixed-length workload (admission-time rescue
is the whole point of the design) — which is also the realistic case for an interactive
API serving a mix of request shapes. `scripts/profile_decode_step.py` investigated the
uniform-workload overhead directly (per-token GPU syncs, KV cache concat/slice copies,
the executor round-trip) and found none of the three suspects individually explained the
gap under isolated profiling — it's left as an open question, not a code-level fix, since
the evidence didn't point at one specific line to change.

Measured on an RTX 4070 Laptop GPU, TinyLlama-1.1B throughout.

## Project Structure

```
flux/
├── src/flux/
│   ├── schemas.py     # GenerateRequest/GenerateResponse - the wire format
│   ├── queue.py        # RequestQueue: async pool, each request carries a Future
│   ├── scheduler.py    # select_batch(): FIFO / PRIORITY / EDF, swappable
│   ├── decode.py        # ContinuousBatch: admit()/step(), per-sequence KV cache + position tracking
│   ├── engine.py        # Loads TinyLlama, owns the continuous batch loop, times queue vs compute
│   └── server.py        # FastAPI app: POST /generate, GET /health, GET /stats
├── scripts/
│   ├── benchmark.py             # Policy x concurrency sweep -> CSV; --workload uniform|mixed
│   ├── controlled_comparison.py # Static-vs-continuous, uniform workload (git worktree + PYTHONPATH switch)
│   ├── mixed_comparison.py      # Static-vs-continuous, mixed workload, continuous GPU logging
│   └── profile_decode_step.py   # torch.profiler breakdown of one continuous decode step
├── tests/
│   ├── test_queue.py
│   ├── test_scheduler.py
│   └── test_decode.py
├── baseline_sweep.csv, continuous_sweep.csv   # Single-engine sweeps
├── controlled_sweep.csv, mixed_sweep.csv       # Static-vs-continuous results (see Benchmarking)
├── mixed_sweep_gpu_log.csv                     # Continuous 5s GPU samples, tagged per cell
└── pyproject.toml
```

## Known Limitations

Real gaps in the current design, worth knowing before quoting a number or planning the
next piece of work:

- **Continuous batching's per-step overhead loses on uniform-length workloads.** See
  Benchmarking above — at concurrency 16, continuous batching is ~23% lower throughput
  and ~30% higher median latency than the static engine it replaced, when every request
  is the same length. The cause wasn't pinned to one specific line (profiling ruled out
  three named suspects without a conclusive winner); it's an open question, not yet a
  fixed one.
- **Admission becomes the bottleneck at high concurrency, even for continuous batching.**
  Interactive deadline compliance under the mixed workload drops from ~62% at
  concurrency 8 to ~15% at concurrency 16 — once `MAX_BATCH_SIZE` active slots are
  saturated, a short request still waits for a slot, regardless of how fast slots turn
  over once it's in.
- **No KV cache reuse across requests.** Each sequence's cache is independent; there's no
  prefix sharing or paging across different requests, just the single concatenated cache
  across the active batch. That's the KV-cache-as-constraint half of this project's
  learning goal, still ahead.
- **Single GPU, single model, single process — by design.** This is deliberately not a
  platform. Docker/Prometheus/Grafana were cut from the core path; they can come back as
  optional extras once the runtime itself has real numbers to show.

## Roadmap

The two end artifacts this project is building toward:

1. An LLM inference runtime benchmarked honestly under concurrent load — done: a
   continuous-batching engine, a scheduler-policy sweep, and a controlled static-vs-
   continuous comparison across two workload shapes, all with real percentiles and a
   real queue/compute split (see Benchmarking above).
2. One custom profiled CUDA kernel with before-and-after numbers and a roofline
   explanation for *why* it helped — not yet started.

Roughly in order from here:

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
