# Architecture Overview

InferOps provides a declarative control plane, multi-engine inference supervisor, and intelligent gateway for bare-metal servers and heterogeneous hardware. It abstracts the operational overhead of running vLLM, SGLang, and llama.cpp instances across NVIDIA, AMD ROCm, Apple Silicon, and Host CPU environments.

```
                           +----------------------+
                           |   configs/*.yaml     |
                           +----------+-----------+
                                      |
                           +----------v-----------+
                           |   InferOps Engine    |
                           |   * Supervisor       |
                           |   * VRAM Pre-check   |
                           |   * Dynamic Router   |
                           |   * Watchdog Daemon  |
                           |   * Workload Scaler  |
                           +-----+----+-----+-----+
                                 |    |     |
               +-----------------+    |     +-----------------+
               |                      |                       |
        +------v--------+      +------v--------+       +------v--------+
        |  vLLM Engine  |      | SGLang Engine |       | llama.cpp     |
        |  Port 8001    |      | Port 8002     |       | Port 8003     |
        |  (Datacenter) |      | (Throughput)  |       | (GGUF/Metal)  |
        +------+--------+      +------+--------+       +------+--------+
               |                      |                       |
               +-----------------+    |     +-----------------+
                                 |    |     |
                           +-----v----v-----v-----+
                           |    OpenAI Gateway    |
                           |    http://:8000/v1   |
                           |    * Token Analytics |
                           |    * API Key Auth    |
                           |    * Scale-to-Zero   |
                           +----------+-----------+
                                      |
                           +----------v-----------+
                           |    Observability     |
                           |    * Prometheus Scrape
                           |    * Grafana Dashboard
                           +----------------------+
```

---

## Core System Layers

### 1. Configuration & Domain Model (`inferops.core.config`)
* **Declarative YAMLs**: Every model instance is declared in `configs/models/<name>.yaml`.
* **Environment Interpolation**: Fields support `${VAR}` or `${VAR:-default}` syntax, pulling secrets from `.env` or system variables.
* **Profiles**: Models are grouped into execution profiles in `configs/profiles.yaml` (e.g. `dev`, `prod`).
* **Scale-to-Zero Parameters**: Support for `idle_timeout_seconds` and `auto_scale` triggers.
* **Speculative Decoding**: Support for draft models and speculative verification tokens.
* **LoRA Modules**: Native declaration of dynamic LoRA adapters (`lora_modules: [{name: ..., path: ...}]`).

### 2. HuggingFace Hub Integration (`inferops.core.hf_hub`)
* **Remote Inspection**: Queries HuggingFace Hub directly to parse architecture specs (`num_hidden_layers`, `hidden_size`, `num_key_value_heads`, `vocab_size`, `quantization_config`).
* **Manifest Synthesis**: Converts remote configurations into complete InferOps YAML manifests.
* **Pre-flight VRAM Validation**: Automatically estimates required memory before initiating downloads.

### 3. Process Supervisor & Crash Watchdog (`inferops.core.supervisor`)
* **Cross-Platform Management**: Manages subprocess lifetimes using OS-native primitives without external daemon dependencies. Fully tested on Linux, macOS, WSL2, and Windows.
* **Process State Records**: Stored in `runtime/pids/<name>.json` with timestamps, PIDs, allocated GPUs, and startup parameters.
* **Log Management**: Subprocess stdout and stderr are combined into `runtime/logs/<name>.log`. Upon restart, the previous log is rotated to `<name>.log.prev`.
* **Readiness Loops**: When starting a model, the supervisor polls the HTTP `/health` probe until the server signals readiness.
* **Crash Watchdog**: Background daemon (`inferops watchdog`) that monitors running processes, detects crashes, and auto-revives them with exponential backoff.

### 4. Multi-Engine Runtime Adapters (`inferops.engines`)
* **vLLM Adapter**: Spawns `vllm.entrypoints.openai.api_server` with `--served-model-name`, `--gpu-memory-utilization`, `--enable-lora`, tensor parallelism, speculative decoding, and CPU fallback.
* **SGLang Adapter**: Spawns `sglang.launch_server` with `--mem-fraction-static`, `--tp`, and `--context-length`.
* **llama.cpp Adapter**: Spawns `llama-server` or `llama_cpp.server` with `-m` GGUF path, `--alias`, `-ngl` (GPU layer offloading), and `-t` (thread count).

### 5. Intelligent Gateway & Reverse Proxy (`inferops.gateway`)
* **Dynamic Routing**: An ASGI FastAPI reverse proxy on port 8000 intercepts `/v1/chat/completions` and `/v1/models`.
* **SSE Streaming**: Forwards Server-Sent Events (`stream: true`) chunk by chunk with minimal latency.
* **Workload Autoscaler**: Tracks active in-flight requests and latency to emit scale-up or scale-down recommendations.
* **Scale-to-Zero & Cold Starts**: Shuts down idle instances after configured inactivity windows, and transparently initiates background cold starts when incoming requests arrive.
* **Token Accounting & Cost Estimation**: Records token throughput in `runtime/usage.db` (SQLite) and computes real-time commercial cost savings compared to closed cloud APIs.
* **API Key Auth**: Cryptographically hashes and validates bearer tokens (`sk-inferops-...`) with per-key rate limits.

### 6. Heterogeneous Hardware & Interconnect Diagnostics (`inferops.hardware`)
* **Hardware Detection**: Scans NVIDIA CUDA (NVML), AMD ROCm (`rocm-smi`), Apple Silicon unified memory (`sysctl hw.memsize`), and Host CPU RAM.
* **Interconnect Analysis**: Parses `nvidia-smi topo -m` to classify interconnect links (NVLink vs PCIe bridges vs System Bus) and evaluate Tensor Parallelism feasibility.

### 7. DevOps & Observability Exporters (`inferops.export`)
* **Docker Compose**: Generates valid `docker-compose.yml` with NVIDIA GPU container passthrough reservations, healthchecks, and cache volume mounts.
* **Kubernetes Manifests**: Generates production `Deployment` and `Service` YAML specifications with GPU resource limits, probes, and shared memory volumes.
* **Grafana 10+ Dashboards**: Synthesizes dashboard JSON with QPS, TTFT percentiles, KV cache capacity gauges, commercial savings counters, and GPU thermals.
* **Prometheus Config**: Generates `prometheus.yml` scrape jobs targeting the InferOps gateway and model engines.

### 8. Dual Interfaces (TUI & Web Dashboard)
* **Terminal UI**: Textual-based live dashboard (`inferops tui`) for headless server monitoring.
* **Web Dashboard**: Browser interface (`inferops web`) featuring real-time GPU cards, model controls, predictive VRAM Sizer with GGUF/MLA, interactive Chat Playground, Live Terminal Console via WebSockets, and Cost Analytics.
