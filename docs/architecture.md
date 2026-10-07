# Architecture Overview

InferOps provides a declarative control plane and multi-engine inference runtime for bare-metal servers. It abstracts the operational overhead of running vLLM and SGLang instances across single or multi-GPU hosts.

```
                           +----------------------+
                           |   configs/*.yaml     |
                           +----------+-----------+
                                      |
                           +----------v-----------+
                           |   InferOps Engine    |
                           |   - Supervisor       |
                           |   - VRAM Pre-check   |
                           |   - Dynamic Router   |
                           +-----+----------+-----+
                                 |          |
                   +-------------v-+      +-v-------------+
                   |  vLLM Engine  |      | SGLang Engine |
                   |  Port 8001    |      | Port 8002     |
                   +---------------+      +---------------+
                                 \          /
                           +------v--------v------+
                           |   OpenAI Gateway     |
                           |   http://:8000/v1    |
                           +----------------------+
```

## System Components

### 1. Configuration & Domain Model (`inferops.core.config`)
- **Declarative YAMLs**: Every model instance is declared in `configs/models/<name>.yaml`.
- **Environment Interpolation**: Fields support `${VAR}` or `${VAR:-default}` syntax, pulling values from `.env` or system variables without exposing tokens in git.
- **Profiles**: Models are grouped into execution profiles in `configs/profiles.yaml` (e.g. `dev: [qwen2.5-coder-7b]`, `prod: [llama3.1-70b]`).

### 2. Process Supervisor (`inferops.core.supervisor`)
- **Cross-Platform Management**: Manages subprocess lifetimes using OS-native primitives without hard platform dependencies. Works on Linux, macOS, WSL2, and Windows.
- **Process State Records**: Stored in `runtime/pids/<name>.json` with timestamps, PIDs, allocated GPUs, and startup parameters.
- **Log Management**: Subprocess stdout and stderr are combined and written to `runtime/logs/<name>.log`. Upon restart, the previous log is rotated to `<name>.log.prev`.
- **Readiness Loops**: When starting a model, the supervisor polls the HTTP `/health` probe until the server signals readiness or a timeout occurs.

### 3. Multi-Engine Runtime (`inferops.engines`)
- **Engine Adapters**: Concrete implementations of `EngineAdapter` translate common parameters (`model`, `port`, `gpus`, `tensor_parallel_size`, `max_model_len`, `dtype`, `quantization`) into engine-specific CLI flags.
- **vLLM Adapter**: Spawns `vllm.entrypoints.openai.api_server` with `--served-model-name`, `--gpu-memory-utilization`, and tensor-parallel flags.
- **SGLang Adapter**: Spawns `sglang.launch_server` with `--mem-fraction-static`, `--tp`, and `--context-length`.

### 4. Dynamic OpenAI Gateway (`inferops.gateway.router`)
- **Multiplexing**: An ASGI FastAPI reverse proxy listening on port 8000 (configurable) intercepts `/v1/chat/completions` and `/v1/models`.
- **Dynamic Routing**: Inspects the `"model"` field in requests and forwards traffic directly to the internal port of the corresponding instance.
- **SSE Streaming**: Supports Server-Sent Events (`stream: true`) transparently without buffering.

### 5. Telemetry & Hardware Scanning (`inferops.hardware`)
- **GPU Discovery**: Scans NVIDIA devices using PyNVML or `nvidia-smi` to monitor VRAM capacity, memory utilization, temperature, and power draw.
- **Metric Scraping**: Collects metrics from `/metrics` endpoints and computes real-time request counts, cache hit rates, and tokens-per-second throughput via an in-memory ring buffer.

### 6. Dual Interfaces (TUI & Web Dashboard)
- **Terminal UI**: Textual-based live dashboard (`inferops tui`) with keyboard shortcuts for starting, stopping, and monitoring instances.
- **Web Dashboard**: Browser-based interface (`inferops web`) featuring real-time hardware gauges, instance controls, an analytical VRAM sizing tool, and a streaming chat test client.
