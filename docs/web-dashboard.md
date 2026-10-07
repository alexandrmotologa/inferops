# Web Dashboard & Chat Playground

InferOps includes a browser-accessible management interface built with FastAPI, WebSockets, and vanilla modern JavaScript (zero heavy npm build steps required).

## Starting the Dashboard

To launch the web interface:

```bash
inferops web --port 8900 --host 127.0.0.1
```

Once started, navigate to `http://127.0.0.1:8900` in any modern browser.

## Features

### 1. Hardware & GPU Fleet Topology
- Live VRAM utilization gauges per detected GPU card.
- Core utilization percentage, GPU temperatures, and memory headroom.
- Automated fallback notice when operating in CPU or emulation mode.

### 2. Model Fleet Management
- Cards for every model defined in `configs/models/*.yaml`.
- Real-time status indicators (`HEALTHY`, `STARTING`, `STOPPED`, `CRASHED`).
- One-click Start and Stop buttons that trigger asynchronous background supervisor actions.

### 3. VRAM Sizer Pre-flight Calculator
- Interactive form to test any model ID or Hugging Face repository against target context lengths, quantization types, and tensor-parallelism degrees.
- Instant breakdown of weight footprint, KV cache consumption, and feasibility verdict.

### 4. Interactive Chat Playground
- Select any active, healthy model instance directly from a dropdown.
- Send test prompts with adjustable sampling temperature and max tokens.
- **Live Token Streaming**: Receives Server-Sent Events (SSE) from the inference engine and computes Time to First Token (TTFT) and token generation throughput (tokens/second) in real time.

## WebSocket Telemetry Protocol

The dashboard maintains an active WebSocket connection at `/ws/telemetry` which pushes hardware snapshots every 1.5 seconds:

```json
{
  "models": [
    {
      "name": "qwen2.5-coder-7b",
      "status": "HEALTHY",
      "port": 8001,
      "throughput_tok_s": 42.5,
      "kv_cache_pct": 18.2,
      "requests_running": 2
    }
  ],
  "gpus": [
    {
      "index": 0,
      "name": "NVIDIA RTX 4090",
      "used_gb": 16.8,
      "total_gb": 24.0,
      "util_gpu": 78.0,
      "temp_c": 64
    }
  ]
}
```
