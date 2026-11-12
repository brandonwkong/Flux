# Flux Architecture

> An ML inference serving platform: dynamic batching, model-aware worker routing, and a
> central scheduler orchestrating horizontally-scalable worker nodes over HTTP. See
> [Known Limitations](#known-limitations) for what's not true yet.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Request Flow](#request-flow)
3. [Components](#components)
4. [Key Concepts](#key-concepts)
5. [File Structure](#file-structure)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Metrics](#metrics)
9. [Known Limitations](#known-limitations)
10. [Benchmarking](#benchmarking)

---

## System Overview

Flux is an ML inference serving system that handles the complete lifecycle from API request to model prediction.

```
                         Thousands of concurrent users
                                    |
                                    v
    +------------------------------------------------------------------+
    |                               FLUX                               |
    |                                                                  |
    |   +----------+     +-----------+     +-------------+             |
    |   |   API    |---->|  Batcher  |---->|  Scheduler  |             |
    |   | /predict |     |           |     |             |             |
    |   +----------+     +-----------+     +------+------+             |
    |                                             |                    |
    |        "Collect 32 requests           "Route to worker           |
    |         or wait 10ms"                  with model loaded"        |
    |                                             |                    |
    |                    +------------------------+----------+         |
    |                    |                        |          |         |
    |                    v                        v          v         |
    |              +----------+            +----------+  +----------+  |
    |              | Worker 1 |            | Worker 2 |  | Worker 3 |  |
    |              | [resnet] |            | [bert]   |  | [resnet] |  |
    |              | GPU 0    |            | GPU 1    |  | CPU      |  |
    |              +----------+            +----------+  +----------+  |
    |                    |                        |          |         |
    |                    +------------------------+----------+         |
    |                                             |                    |
    |                                      +------v------+             |
    |                                      |   Metrics   |---> Grafana |
    |                                      | Prometheus  |             |
    |                                      +-------------+             |
    +------------------------------------------------------------------+
```

### What Problem Does This Solve?

**Without Flux:**
```
User A request -> GPU processes 1 image  -> 10ms -> respond
User B request -> GPU processes 1 image  -> 10ms -> respond
User C request -> GPU processes 1 image  -> 10ms -> respond
User D request -> GPU processes 1 image  -> 10ms -> respond

Total: 40ms, GPU sitting idle between requests (5% utilization)
```

**With Flux:**
```
User A -+
User B -+-> Batcher collects -> Batch[A,B,C,D] -> GPU processes 4 -> 12ms -> respond all
User C -+     (waits 10ms)
User D -+

Total: 12ms for ALL users, GPU fully utilized (95% utilization)
```

---

## Request Flow

### Step-by-Step Walkthrough

```
1. CLIENT REQUEST
   POST /predict {"model": "resnet50", "inputs": [[0.1, 0.2, ...]]}
                                |
                                v
2. API SERVER (api/main.py)
   - Validates request
   - Checks backpressure (queue not full?)
   - Creates InferenceRequest
   - Submits to Batcher
   - WAITS for result (sync)
                                |
                                v
3. BATCHER (batcher/batcher.py)
   - Adds request to per-model queue
   - Waits until:
     * 32 requests collected, OR
     * 10ms elapsed since first request
   - Creates InferenceBatch with all collected requests
   - Calls process_batch() callback
                                |
                                v
4. SCHEDULER (scheduler/scheduler.py)
   - Receives batch from batcher
   - Calls placement.find_worker_for_batch()
   - Prefers workers with model already loaded (warm)
   - Falls back to any available worker (cold start)
   - Dispatches batch via HTTP POST to worker
                                |
                                v
5. WORKER (worker/main.py)
   - Receives POST /infer request
   - Loads model if not cached (cold start ~2-5s)
   - Runs ONNX inference on batch
   - Returns predictions
                                |
                                v
6. RESPONSE DISTRIBUTION
   - Scheduler receives predictions
   - Batcher splits results back to individual requests
   - Each waiting client receives their prediction
   - Metrics recorded (latency, throughput)
```

### Timing Example

```
t=0ms     User A sends request
t=2ms     User B sends request
t=5ms     User C sends request
t=10ms    Batcher timeout reached, batch formed [A, B, C]
t=10ms    Scheduler finds Worker 1 (has resnet50 loaded)
t=11ms    Batch dispatched to Worker 1
t=11ms    Worker runs ONNX inference on 3 inputs
t=18ms    Worker returns predictions [cat, dog, bird]
t=18ms    Batcher distributes: A->cat, B->dog, C->bird
t=18ms    All 3 users receive response

Total latency: 18ms for all 3 users
Without batching: 30ms (10ms each sequential)
```

---

## Components

### 1. API Server (`api/main.py`)

The entry point for all client requests.

**Responsibilities:**
- REST API endpoints (`/predict`, `/health`, `/stats`)
- Request validation (Pydantic models)
- Backpressure (reject when queue > 1000)
- Timeout handling
- Prometheus metrics exposure

**Key Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Submit inference request (sync) |
| `/health` | GET | Health check |
| `/ready` | GET | Readiness check (has workers?) |
| `/stats` | GET | System statistics |
| `/metrics` | GET | Prometheus metrics |
| `/models` | GET | List available models |
| `/workers` | GET | List registered workers |

---

### 2. Batcher (`batcher/batcher.py`)

Collects individual requests into efficient batches.

**Why Batching?**
- GPUs are parallel processors with thousands of cores
- Processing 1 input uses ~5% of GPU capacity
- Processing 32 inputs uses ~95% of GPU capacity
- Batching = higher throughput, similar latency

**Configuration:**
```python
BatcherConfig(
    max_batch_size=32,    # Collect up to 32 requests
    max_wait_ms=10.0,     # Wait max 10ms to form batch
    max_queue_depth=1000, # Backpressure threshold
)
```

**How It Works:**
```
Per-Model Queues:
  resnet50: [req1, req2, req3, ...]
  bert:     [req4, req5, ...]

Batch Loop (per model):
  1. Wait for first request
  2. Start 10ms timer
  3. Collect more requests until:
     - 32 requests reached, OR
     - 10ms timer expires
  4. Form batch, send to scheduler
  5. Repeat
```

---

### 3. Scheduler (`scheduler/scheduler.py`)

The brain that decides which worker processes each batch.

**Responsibilities:**
- Worker registry management
- Placement decisions (which worker?)
- Batch dispatch
- Failure handling
- Metrics tracking

**Placement Strategies:**
```
FIRST_FIT:   First worker that can handle it (fast)
BEST_FIT:    Worker with least wasted capacity (efficient)
WORST_FIT:   Worker with most free capacity (spread load)
ROUND_ROBIN: Rotate through workers (fair)
```

**Model-Aware Routing:**
```python
def find_worker_for_batch(batch, workers):
    # Prefer warm workers (model already loaded)
    warm = [w for w in workers if w.has_model_loaded(batch.model)]
    if warm:
        return best_fit(warm)  # ~10ms latency

    # Fall back to cold start
    return best_fit(workers)  # ~2-5s latency (loads model)
```

---

### 4. Worker (`worker/main.py`)

The muscle that runs actual ML inference.

**Responsibilities:**
- Register with scheduler on startup
- Send heartbeats (health check)
- Load/unload models (warm pool)
- Run ONNX inference
- Report results

**Key Endpoints:**
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/infer` | POST | Run batch inference |
| `/health` | GET | Health check |
| `/models` | GET | List loaded models |
| `/models/{name}/load` | POST | Pre-load a model |
| `/models/{name}/unload` | POST | Unload a model |

---

### 5. Inference Executor (`worker/executor.py`)

Runs actual ML models using ONNX Runtime.

**Features:**
- Model caching (warm pool)
- LRU eviction when at capacity
- GPU/CPU provider selection
- Async execution (non-blocking)
- Mock mode for development

**Model Loading:**
```python
# Load model (cold start: 2-5 seconds)
session = ort.InferenceSession("resnet50.onnx", providers=["CUDAExecutionProvider"])

# Run inference (warm: ~10ms)
outputs = session.run(None, {"input": batch_tensor})
```

**Warm Pool:**
```python
InferenceExecutor(
    model_dir="models",      # Where .onnx files live
    use_gpu=True,            # Use CUDA if available
    max_loaded_models=5,     # Keep 5 models in memory
)

# LRU eviction: least recently used model unloaded when at capacity
```

---

## Key Concepts

### 1. Dynamic Batching

**Problem:** Individual requests underutilize GPU.

**Solution:** Collect requests into batches before inference.

```
Single request:   GPU 5% utilized,  10ms latency
Batch of 32:      GPU 95% utilized, 15ms latency

Trade: +5ms latency for 32x throughput
```

**Configurable Tradeoffs:**
```
High batch size + long timeout = Higher throughput, higher latency
Low batch size + short timeout = Lower latency, lower throughput

Defaults: 32 requests OR 10ms = Good balance
```

---

### 2. Model-Aware Placement

**Problem:** Loading a model takes 2-5 seconds.

**Solution:** Route to workers that already have the model loaded.

```
Worker 1: [resnet50, bert]     <- warm for these models
Worker 2: [resnet50]           <- warm for resnet50
Worker 3: []                   <- cold, will need to load

Request for resnet50 -> Route to Worker 1 or 2 (warm, ~10ms)
Request for yolo     -> Route to any, cold start (~3s first time)
```

---

### 3. Backpressure

**Problem:** More requests than system can handle = OOM, cascading failures.

**Solution:** Reject requests when overloaded.

```python
if batcher.queue_depth >= MAX_QUEUE_DEPTH:  # 1000
    raise HTTPException(503, "Server overloaded, please retry later")
```

**Client sees:**
```
HTTP 503 Service Unavailable
```

Client can retry with exponential backoff.

---

### 4. Warm Pool

**Problem:** Can't keep all models in GPU memory.

**Solution:** LRU cache of most recently used models.

```
GPU Memory: 8GB
ResNet50:   100MB
BERT:       400MB
YOLO:       250MB

Can fit: ~5-10 models

When loading model #6:
  1. Find least recently used model
  2. Unload it (free memory)
  3. Load new model
```

---

## File Structure

```
flux/
├── src/flux/
│   ├── __init__.py
│   │
│   ├── api/                      # REST API
│   │   ├── __init__.py
│   │   └── main.py               # FastAPI app, /predict endpoint
│   │
│   ├── batcher/                  # Dynamic batching
│   │   ├── __init__.py
│   │   └── batcher.py            # Collects requests into batches
│   │
│   ├── scheduler/                # Scheduling & placement
│   │   ├── __init__.py
│   │   ├── scheduler.py          # Main scheduler, dispatch logic
│   │   ├── placement.py          # Model-aware routing (inference) + bin-packing (legacy jobs)
│   │   ├── registry.py           # Worker registry, heartbeats
│   │   └── queue.py              # Priority queue (legacy jobs)
│   │
│   ├── worker/                   # Inference workers
│   │   ├── __init__.py
│   │   ├── main.py               # Worker FastAPI app, /infer endpoint
│   │   └── executor.py           # ONNX Runtime executor
│   │
│   ├── models/                   # Pydantic data models
│   │   ├── __init__.py
│   │   ├── request.py            # PredictRequest, InferenceBatch
│   │   ├── worker.py             # Worker, WorkerRegistration
│   │   ├── job.py                # Job (legacy)
│   │   └── resources.py          # CPU/Memory/GPU resources
│   │
│   └── utils/                    # Utilities
│       └── __init__.py
│
├── models/                       # ONNX model files (.onnx)
│   └── (place models here)
│
├── tests/
│   ├── test_models.py
│   └── test_scheduler.py
│
├── docker/
│   ├── Dockerfile.scheduler
│   ├── Dockerfile.worker
│   └── prometheus.yml
│
├── scripts/
│   ├── test_local.py             # Local testing script
│   └── benchmark.py              # Load testing
│
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── ARCHITECTURE.md               # This file
```

---

## API Reference

### POST /predict

Submit an inference request.

**Request:**
```json
{
  "model": "resnet50",
  "inputs": [[0.1, 0.2, 0.3, ...]],
  "priority": 5,
  "timeout_ms": 5000
}
```

**Response:**
```json
{
  "predictions": [293],
  "latency_ms": 12.5,
  "batch_size": 1,
  "model": "resnet50"
}
```

**Errors:**
| Status | Meaning |
|--------|---------|
| 503 | Server overloaded (backpressure) |
| 504 | Request timeout |
| 500 | Inference error |

---

### GET /stats

System statistics.

**Response:**
```json
{
  "queue": {"pending": 5},
  "jobs": {"total": 100, "running": 3, "completed": 90, "failed": 7},
  "workers": {
    "total_workers": 3,
    "healthy_workers": 3,
    "unhealthy_workers": 0
  },
  "batcher": {
    "queue_depth": 12,
    "max_batch_size": 32,
    "batch_timeout_ms": 10
  },
  "cluster_utilization": 0.75
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_BATCH_SIZE` | 32 | Maximum requests per batch |
| `BATCH_TIMEOUT_MS` | 10 | Max wait time to form batch (ms) |
| `MAX_QUEUE_DEPTH` | 1000 | Backpressure threshold |
| `WORKER_TIMEOUT_S` | 30 | Worker heartbeat timeout (s) |

### Tuning Guide

**High Throughput (batch processing):**
```
MAX_BATCH_SIZE=64
BATCH_TIMEOUT_MS=50
```

**Low Latency (real-time):**
```
MAX_BATCH_SIZE=8
BATCH_TIMEOUT_MS=5
```

**Balanced (default):**
```
MAX_BATCH_SIZE=32
BATCH_TIMEOUT_MS=10
```

---

## Metrics

### Prometheus Metrics

Inference path (`/predict`):

| Metric | Type | Description |
|--------|------|-------------|
| `flux_request_latency_seconds` | Histogram | End-to-end latency |
| `flux_requests_total` | Counter | Total requests by model/status |
| `flux_batch_size` | Histogram | Batch size distribution |
| `flux_batch_wait_ms` | Histogram | Time requests wait for batching |
| `flux_batcher_queue_depth` | Gauge | Pending requests per model, in the batcher |
| `flux_worker_inference_latency_seconds` | Histogram | Worker inference time |
| `flux_worker_model_load_seconds` | Histogram | Model load time on worker |

Legacy job scheduler (not touched by `/predict`):

| Metric | Type | Description |
|--------|------|-------------|
| `flux_queue_depth` | Gauge | Pending legacy jobs |
| `flux_cluster_utilization` | Gauge | Declared resource utilization (allocated ÷ capacity) across workers — not live GPU telemetry; see Known Limitations |

### Key SLOs (targets — not yet measured; see Benchmarking)

| Metric | Target |
|--------|--------|
| p50 latency | < 15ms |
| p99 latency | < 50ms |
| Throughput | > 1000 req/sec |
| Error rate | < 0.1% |

---

## Running the System

### Local Development

```bash
cd flux
pip install -e ".[dev]"

# Terminal 1: Start API/Scheduler
python -m flux.api.main

# Terminal 2: Start Worker
python -m flux.worker.main

# Terminal 3: Test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "resnet50", "inputs": [[1.0, 2.0, 3.0]]}'
```

### Docker Compose

```bash
docker-compose up --build

# Services:
# - API:        http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000
```

---

## Known Limitations

Real, verified gaps — not fixed here, worth knowing before scaling further:

- **No connection pooling on dispatch.** `Scheduler.dispatch_batch` opens a new
  `httpx.AsyncClient()` per batch dispatch rather than reusing a pooled client. First
  thing to fix before trusting any throughput benchmark.
- **Heartbeats carry no payload.** `WorkerHeartbeat` (GPU memory, loaded models,
  available resources) is defined in `models/worker.py` but never sent by the worker or
  read by the scheduler — `POST /workers/{id}/heartbeat` only refreshes the timestamp.
  `gpu_memory_used_mb`/`gpu_memory_total_mb` stay fixed at whatever was passed at
  registration (0 by default), so the cold-start "most free GPU memory" tie-break in
  `_model_aware_placement` has no real signal in a live deployment.
- **`flux_cluster_utilization` is dead weight for inference-only workers.** It's derived
  from `Worker.available` vs. `Worker.capacity`, but the inference dispatch path
  (`dispatch_batch`) never calls `worker.allocate()`/`.release()` — only the legacy
  job-dispatch path does. An inference-only worker reports whatever utilization it had
  at registration, forever.
- **Worker-failure recovery doesn't cover in-flight inference batches.**
  `_handle_worker_failure` re-queues legacy `Job`s via `worker.current_jobs`; an
  inference batch dispatched to a worker that dies mid-flight just errors out to the
  waiting client — no reroute, no retry.
- **No retry on `/predict`.** A failed batch propagates the exception to every request
  in it. Retry (`can_retry`/`retry_count`) exists only in the legacy job scheduler.
- **Scheduler has no HA.** Single in-memory process, no replication, no leader election.
  This is a centralized-coordinator architecture, not a decentralized one — see the note
  at the top of this document.

## Benchmarking

`scripts/benchmark.py` drives `/predict` and reports p50/p95/p99 + throughput, with
`--save`/`--compare` for diffing runs. No results are checked into this repo — the SLO
targets above are targets, not measurements. Run it yourself before quoting a number:

```bash
python -m flux.api.main &
python -m flux.worker.main &
python scripts/benchmark.py --requests 1000 --concurrency 10 --save baseline.json
```

Note the closed-loop caveat: `benchmark.py` bounds concurrency with a semaphore rather
than firing at a fixed arrival rate, and sends uniform traffic (no priority/deadline
mix) — fine for a smoke test, not sufficient to make a "sustained N req/s" claim or to
differentiate scheduling policies once those exist (see README.md's Roadmap, Phase 3 —
"Scheduling That Actually Matters").
