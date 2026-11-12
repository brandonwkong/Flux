# Flux

An ML inference serving platform: dynamic batching, model-aware worker routing, and a
central scheduler orchestrating horizontally-scalable worker nodes over HTTP.

## Architecture

```
POST /predict
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                             FLUX                             │
│                                                              │
│         ┌─────────┐    ┌─────────┐    ┌───────────┐          │
│         │   API   │───►│ Batcher │───►│ Scheduler │          │
│         └─────────┘    └─────────┘    └─────┬─────┘          │
│                                                              │
│      Collects requests into batches                          │
│      (max 32, or 10ms wait) then routes to best worker       │
│                                                              │
│      ┌───────────────────┼───────────────────┐               │
│      ▼                   ▼                   ▼               │
│      ┌──────────┐       ┌──────────┐      ┌──────────┐       │
│      │ Worker 1 │       │ Worker 2 │      │ Worker 3 │       │
│      │ [resnet] │       │ [bert]   │      │ [resnet] │       │
│      │   ONNX   │       │   ONNX   │      │   ONNX   │       │
│      └──────────┘       └──────────┘      └──────────┘       │
│           │                   │                  │           │
│           └───────────────────┴──────────────────┘           │
│                                                              │
│                   ┌───────▼───────┐                          │
│                   │    Metrics    │                          │
│                   │ latency, tput │──► Grafana               │
│                   └───────────────┘                          │
└──────────────────────────────────────────────────────────────┘
```

This is a centralized scheduler / distributed worker-pool architecture — one coordinator
dispatching to independently-scaling worker processes over HTTP (same shape as Celery or
a Kubernetes control plane + kubelets), not a leaderless/replicated system: the scheduler
itself is a single in-memory process with no HA today. See
[Known Limitations](#known-limitations).

## Key Features

| Feature | Description |
|---------|-------------|
| **Dynamic Batching** | Collects requests into batches for GPU efficiency (max 32 or 10ms) |
| **Model-Aware Routing** | Prefers workers with the model already loaded (warm); falls back to the worker with the most free GPU memory on a cold start |
| **Backpressure** | Rejects requests with 503 when the batcher queue is full |
| **Heartbeat-Based Failure Detection** | Workers that miss heartbeats are marked unhealthy and removed from routing |
| **Observability** | Prometheus metrics; Grafana datasource auto-provisioned via Docker Compose |

> A separate, legacy generic job-scheduling API (`/jobs`) also lives in this codebase —
> priority queue with aging, preemption, and real best-fit bin-packing placement. It
> predates the inference pipeline and is unrelated to `/predict`. See
> [Legacy Job API](#legacy-job-api) below.

## Quick Start

### Local Development

```bash
cd flux
pip install -e ".[dev]"

# Terminal 1: Start API server
python -m flux.api.main

# Terminal 2: Start a worker
python -m flux.worker.main

# Terminal 3: Send a prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "resnet50", "inputs": [[1.0, 2.0, 3.0]]}'
```

No ONNX model files are required to run this — if `models/<name>.onnx` doesn't exist (or
`onnxruntime` isn't installed), the worker transparently falls back to mock inference so
the full pipeline (batching → routing → dispatch) is exercisable without real weights.

### Docker Compose

```bash
docker-compose up --build

# Access:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (Prometheus datasource pre-provisioned; build your own panels)
```

## API

### Inference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Submit inference request (sync) |
| GET | `/models` | List available models |
| GET | `/models/{name}` | Get model info |

### Workers

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/workers` | Register a worker |
| GET | `/workers` | List workers |
| POST | `/workers/{id}/heartbeat` | Worker heartbeat |
| POST | `/workers/{id}/drain` | Drain worker |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/stats` | System statistics |
| GET | `/metrics` | Prometheus metrics |

### Legacy Job API

`/jobs`, `/workers/{id}/execute` (via worker's `/execute`), `/callbacks/completed|failed` —
a generic "submit a shell command, schedule it with priority + preemption + bin-packing"
API, independent of the inference path above. Kept for compatibility; not exercised by
`/predict`.

## Example Usage

```python
import httpx

# Single prediction
response = httpx.post("http://localhost:8000/predict", json={
    "model": "resnet50",
    "inputs": [[0.1, 0.2, 0.3, ...]]  # Your input tensor
})
print(response.json())
# {"predictions": [293], "latency_ms": 12.5, "batch_size": 1, "model": "resnet50"}

# Batch prediction
response = httpx.post("http://localhost:8000/predict", json={
    "model": "resnet50",
    "inputs": [
        [0.1, 0.2, ...],
        [0.3, 0.4, ...],
        [0.5, 0.6, ...]
    ]
})
print(response.json())
# {"predictions": [293, 151, 7], "latency_ms": 15.2, "batch_size": 3, "model": "resnet50"}
```

(Predictions are class indices for image-ish model names, mock embeddings for
text/bert-ish names, or input length otherwise — see `worker/executor.py`'s mock mode.)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MAX_BATCH_SIZE` | 32 | Maximum requests per batch |
| `BATCH_TIMEOUT_MS` | 10 | Max wait time to form batch |
| `MAX_QUEUE_DEPTH` | 1000 | Backpressure threshold |
| `WORKER_TIMEOUT_S` | 30 | Worker heartbeat timeout before marked unhealthy |

## Metrics

Primary (inference path):

| Metric | Type | Description |
|--------|------|-------------|
| `flux_request_latency_seconds` | Histogram | End-to-end `/predict` latency |
| `flux_requests_total` | Counter | Total requests by model/status |
| `flux_batch_size` | Histogram | Batch size distribution |
| `flux_batch_wait_ms` | Histogram | Time requests wait for batching |
| `flux_batcher_queue_depth` | Gauge | Pending requests per model, in the batcher |
| `flux_worker_inference_latency_seconds` | Histogram | Worker-side inference latency |
| `flux_worker_model_load_seconds` | Histogram | Model load time on worker |

Legacy job scheduler (not used by `/predict`):

| Metric | Type | Description |
|--------|------|-------------|
| `flux_queue_depth` | Gauge | Pending legacy jobs |
| `flux_cluster_utilization` | Gauge | Declared resource utilization across workers (allocated ÷ capacity, not live GPU telemetry) |

## Project Structure

```
flux/
├── src/flux/
│   ├── api/
│   │   └── main.py           # FastAPI app: /predict, /workers, /jobs (legacy), /metrics
│   ├── batcher/
│   │   └── batcher.py        # Dynamic batching logic
│   ├── scheduler/
│   │   ├── scheduler.py      # Main scheduler + inference dispatch
│   │   ├── placement.py      # Model-aware routing (+ legacy bin-packing strategies)
│   │   ├── registry.py       # Worker registry, heartbeats
│   │   └── queue.py          # Priority queue w/ aging (legacy job path)
│   ├── worker/
│   │   ├── executor.py       # ONNX Runtime executor (mock-mode fallback)
│   │   └── main.py           # Worker service
│   └── models/
│       ├── request.py        # PredictRequest/Response, InferenceBatch
│       ├── worker.py         # Worker with GPU/model info
│       ├── job.py            # Job (legacy)
│       └── resources.py      # CPU/memory/GPU resource math
├── models/                   # ONNX model files (empty — mock mode used until populated)
├── tests/
├── docker/
└── pyproject.toml
```

## Known Limitations

Found while cleaning this up — real gaps, not yet fixed, worth knowing before scaling:

- **No connection pooling on dispatch.** `Scheduler.dispatch_batch` opens a new
  `httpx.AsyncClient()` per batch dispatch instead of reusing a pooled client. Measurable
  overhead per request; first thing to fix before any real throughput benchmark.
- **Heartbeats carry no payload.** `WorkerHeartbeat` (GPU memory used, loaded models,
  available resources) is defined but never sent or read — `POST /workers/{id}/heartbeat`
  only refreshes the timestamp. `gpu_memory_used_mb`/`gpu_memory_total_mb` are therefore
  fixed at whatever was passed at registration (0 unless set explicitly), so the
  cold-start "most free GPU memory" tie-break has no real signal in practice.
- **Worker-failure recovery doesn't cover in-flight inference batches.** Heartbeat-based
  failure detection and rescheduling (`_handle_worker_failure`) only re-queues legacy
  `Job`s via `current_jobs`; an inference batch in flight to a worker that dies just
  errors out to the waiting client (no reroute/retry).
- **No retry on `/predict`.** A failed batch propagates the exception to every request in
  it. Retry logic exists only in the legacy job scheduler.
- **Scheduler has no HA.** Single in-memory process; see the architecture note above.
- **`flux_cluster_utilization` is dead weight for inference-only workers.** It's computed
  from `Worker.available` vs. `Worker.capacity`, but `dispatch_batch` (the inference
  path) never calls `worker.allocate()`/`.release()` — only the legacy job-dispatch path
  does. A worker serving only inference traffic always reports whatever utilization it
  had at registration (static, usually 0), regardless of real load.

## Roadmap

### Phase 1: Core MVP (COMPLETE)
- [x] Request/Response models
- [x] Worker registry
- [x] Dynamic batcher (32 req or 10ms timeout)
- [x] Model-aware placement (prefer warm workers)
- [x] ONNX executor (with mock mode for dev)
- [x] `/predict` endpoint (sync response)
- [x] `/infer` worker endpoint
- [x] Backpressure handling (503 when overloaded)
- [x] Request timeout handling (`timeout_ms` → 504)
- [x] Heartbeat-based worker health detection
- [x] Prometheus metrics
- [x] Legacy job scheduler: priority queue + aging, preemption, best-fit bin-packing

Phase 1 proves the plumbing — queueing, routing, failure detection — on a toy
static-batching model server. It does not demonstrate what inference-infra roles
actually hire for in 2026: continuous batching, token-level scheduling, and
KV-cache-aware routing for LLMs. Phases 2-4 below replace the old "more Grafana,
autoscaling, TensorRT-as-a-checkbox" plan with that instead. Each phase is gated on the
previous one actually working — this is a sequence, not a parallel checklist.

### Phase 2: Real LLM Serving (continuous batching)
- [ ] Swap the primary path to a small LLM (Qwen2.5-1.5B or Llama-3.2-1B); keep the
      current ONNX/mock path as a secondary demo, not the headline
- [ ] Replace fixed-window dynamic batching (32 req / 10ms) with continuous batching —
      new requests join every decode iteration, finished ones leave mid-batch
- [ ] Batch by token budget (e.g. 2048 tokens/step), not request count; track per-request
      KV cache memory — this is the real GPU bin-packing problem, not "worker has model
      loaded"
- [ ] Streaming responses + token-level metrics: TTFT, TPOT/inter-token latency, tok/s —
      replace end-to-end `latency_ms` as the headline metric
- [ ] Stand up vLLM as a separate comparison baseline (benchmark against it — don't wrap
      it and call the wrapper Flux)

### Phase 3: Scheduling That Actually Matters
- [ ] Two traffic classes (interactive vs. batch); prove priority/EDF admission protects
      interactive p99 TPOT when batch traffic floods the queue (this is the EDF design
      from the old Phase 2, re-applied to real token-level load instead of toy requests)
- [ ] Prefix/KV-cache-aware routing — route to the worker already holding the matching
      prefix's KV cache, not just "has the model loaded"
- [ ] Open-loop load generator (fixed QPS / Poisson arrivals) — the current
      `scripts/benchmark.py` is closed-loop/uniform-traffic and cannot produce or
      differentiate this
- [ ] Report **goodput** (throughput that still meets an SLO, e.g. TPOT < 50ms) as the
      headline number, as a latency/throughput-vs-SLO graph — not a benchmark table

### Phase 4: Kernel-Level Optimization
- [ ] Requires WSL2 (Ubuntu) — Triton's compiler and Nsight Systems CLI profiling are
      Linux-first; don't fight MSVC/native-Windows CUDA tooling instead of learning this
- [ ] Profile one decode step with Nsight Systems / `torch.profiler` before writing
      anything — find out whether attention/norms are actually memory-bound
- [ ] Write one Triton kernel (fused RMSNorm+RoPE first, fused softmax second) and plug
      it into the decode loop
- [ ] Report before/after TPOT with a roofline explanation for *why* it helped

### Deprioritized — don't move the needle on the above
Auto-retry (actively bad for LLM serving — doubles compute on a timeout that was just
queueing, not failed), autoscaling, additional Prometheus metrics, warm pools, scheduler
HA. Real production concerns, just not what gets this project from L2 to L3.

### v2 Finish Line
Flux serves an LLM with continuous batching, beats naive HF `generate()`, lands within a
clearly-stated % of vLLM on goodput, and one custom kernel measurably drops TPOT with an
explanation of why — not "Phase 5 done."

## Benchmarking

`scripts/benchmark.py` drives `/predict` at a configurable concurrency and batch size and
reports p50/p95/p99 latency + throughput; `--save`/`--compare` let you diff a baseline
against a change. No results are checked into this repo yet — run it yourself and record
real numbers before quoting any latency/throughput figure for this project. This harness
is closed-loop and request-level — fine for Phase 1, not sufficient for Phase 3's
open-loop/goodput work (new harness needed there; see Roadmap).

```bash
python -m flux.api.main &
python -m flux.worker.main &
python scripts/benchmark.py --requests 1000 --concurrency 10 --save baseline.json
```

## Tech Stack

**Phase 1 (current):**
- **API**: FastAPI + Uvicorn
- **Batching**: asyncio Queue + Futures, fixed window (32 req / 10ms)
- **Inference**: ONNX Runtime (CPU/CUDA providers), mock-mode fallback
- **Metrics**: Prometheus + Grafana
- **Container**: Docker Compose (scheduler, N workers, Prometheus, Grafana)

**Phase 2+ (target):**
- **Inference**: Qwen2.5-1.5B / Llama-3.2-1B via a hand-rolled continuous-batching decode
  loop, benchmarked against vLLM
- **Kernels**: Triton (fused RMSNorm+RoPE, fused softmax), profiled with Nsight
  Systems / `torch.profiler` — requires WSL2
- **Metrics**: TTFT, TPOT, tok/s, goodput-vs-SLO
