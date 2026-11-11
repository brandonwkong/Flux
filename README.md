# Flux

A ML inference platform with dynamic batching and GPU-aware scheduling, working on kernel level optimizations next

## Architecture

```
POST /predict
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                      FLUX                          │
│                                                         │
│  ┌─────────┐    ┌─────────┐    ┌───────────┐           │
│  │   API   │───►│ Batcher │───►│ Scheduler │           │
│  └─────────┘    └─────────┘    └─────┬─────┘           │
│                                      │                  │
│       Collect requests          Route to best           │
│       into batches              worker (model           │
│       (max 32 or 10ms)          loaded, GPU free)       │
│                                      │                  │
│                 ┌────────────────────┼────────────────┐ │
│                 ▼                    ▼                ▼ │
│           ┌──────────┐         ┌──────────┐    ┌──────────┐
│           │ Worker 1 │         │ Worker 2 │    │ Worker 3 │
│           │ [resnet] │         │ [bert]   │    │ [resnet] │
│           │ ONNX/TRT │         │ ONNX/TRT │    │ ONNX/TRT │
│           └──────────┘         └──────────┘    └──────────┘
│                 │                    │                │ │
│                 └────────────────────┴────────────────┘ │
│                                      │                  │
│                              ┌───────▼───────┐          │
│                              │    Metrics    │          │
│                              │ latency, tput │──► Grafana
│                              │ GPU util      │          │
│                              └───────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Dynamic Batching** | Collects requests into batches for GPU efficiency |
| **Model-Aware Routing** | Routes to workers with model already loaded (warm) |
| **GPU Bin-Packing** | Packs inference onto GPUs efficiently |
| **Backpressure** | Rejects requests when overloaded (503) |
| **Auto-Retry** | Failed requests retry automatically |
| **Observability** | Prometheus metrics + Grafana dashboards |

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

### Docker Compose

```bash
docker-compose up --build

# Access:
# - API: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
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

## Example Usage

```python
import httpx

# Single prediction
response = httpx.post("http://localhost:8000/predict", json={
    "model": "resnet50",
    "inputs": [[0.1, 0.2, 0.3, ...]]  # Your input tensor
})
print(response.json())
# {"predictions": ["cat"], "latency_ms": 12.5, "batch_size": 1}

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
# {"predictions": ["cat", "dog", "bird"], "latency_ms": 15.2, "batch_size": 3}
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MAX_BATCH_SIZE` | 32 | Maximum requests per batch |
| `BATCH_TIMEOUT_MS` | 10 | Max wait time to form batch |
| `MAX_QUEUE_DEPTH` | 1000 | Backpressure threshold |
| `WORKER_TIMEOUT_S` | 30 | Worker heartbeat timeout |

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `flux_request_latency_seconds` | Histogram | End-to-end request latency |
| `flux_requests_total` | Counter | Total requests by model/status |
| `flux_batch_size` | Histogram | Batch size distribution |
| `flux_queue_depth` | Gauge | Pending requests in batcher |
| `flux_gpu_utilization` | Gauge | GPU utilization per worker |
| `flux_model_load_seconds` | Histogram | Model loading time |

## Project Structure

```
flux/
├── src/flux/
│   ├── api/
│   │   └── main.py           # FastAPI /predict endpoint
│   ├── batcher/
│   │   └── batcher.py        # Dynamic batching logic
│   ├── scheduler/
│   │   ├── scheduler.py      # Main scheduler
│   │   ├── placement.py      # Model-aware bin-packing
│   │   └── registry.py       # Worker + model registry
│   ├── worker/
│   │   ├── executor.py       # ONNX Runtime executor
│   │   └── main.py           # Worker service
│   └── models/
│       ├── request.py        # PredictRequest/Response
│       └── worker.py         # Worker with GPU/model info
├── models/                    # ONNX model files
├── tests/
├── docker/
└── pyproject.toml
```

## Roadmap

### Phase 1: Core MVP (COMPLETE)
- [x] Request/Response models
- [x] Worker registry
- [x] Basic placement (bin-packing)
- [x] Dynamic batcher (32 req or 10ms timeout)
- [x] ONNX executor (with mock mode for dev)
- [x] `/predict` endpoint (sync response)
- [x] `/infer` worker endpoint
- [x] Backpressure handling (503 when overloaded)
- [x] Prometheus metrics

### Phase 2: Smart Scheduling
- [ ] Model-aware placement (prefer warm workers)
- [ ] GPU memory tracking
- [ ] Batch size optimization

### Phase 3: Reliability
- [ ] Backpressure (reject when overloaded)
- [ ] Request timeout handling
- [ ] Worker failure recovery
- [ ] Model loading retry

### Phase 4: Observability
- [ ] Latency histograms (p50/p95/p99)
- [ ] Throughput metrics
- [ ] Grafana dashboards
- [ ] Request tracing

### Phase 5: Performance
- [ ] TensorRT backend
- [ ] Model warm pools
- [ ] Autoscaling
- [ ] Load testing + benchmarks


## Tech Stack

- **API**: FastAPI + Uvicorn
- **Batching**: asyncio Queue + Futures
- **Inference**: ONNX Runtime (TensorRT optional)
- **Metrics**: Prometheus + Grafana
- **Container**: Docker + NVIDIA runtime
