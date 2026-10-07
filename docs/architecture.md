# Architecture Overview

InferOps provides a declarative control plane, multi-engine inference supervisor, and intelligent gateway for bare-metal servers. It abstracts the operational overhead of running vLLM and SGLang instances across single or multi-GPU environments.

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
                           |   * Scale-to-Zero    |
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
                           |   * Token Analytics  |
                           |   * API Key Auth     |
                           +----------------------+
```

---

## Core System Layers

### 1. Configuration & Domain Model (`inferops.core.config`)
* **Declarative YAMLs**: Every model instance is declared in `configs/models/<name>.yaml`.
* **Environment Interpolation**: Fields support `${VAR}` or `${VAR:-default}` syntax, pulling secrets from `.env` or system variables.
* **Profiles**: Models are grouped into execution profiles in `configs/profiles.yaml` (e.g. `dev`, `prod`).
* **Scale-to-Zero Parameters**: Support for `idle_timeout_seconds` and `auto_scale` triggers.
* **LoRA Modules**: Native declaration of dynamic LoRA adapters (`lora_modules: [{name: ..., path: ...}]`).

### 2. HuggingFace Hub Integration (`inferops.core.hf_hub`)
* **Remote Inspection**: Queries HuggingFace Hub directly to parse architecture specs (`num_hidden_layers`, `hidden_size`, `num_key_value_heads`, `vocab_size`, `quantization_config`).
* **Manifest Synthesis**: Converts remote configurations into complete InferOps YAML manifests.
* **Pre-flight VRAM Validation**: Automatically estimates required memory before initiating downloads.

### 3. Process Supervisor (`inferops.core.supervisor`)
* **Cross-Platform Management**: Manages subprocess lifetimes using OS-native primitives without external daemon dependencies. Fully tested on Linux, macOS, WSL2, and Windows.
* **Process State Records**: Stored in `runtime/pids/<name>.json` with timestamps, PIDs, allocated GPUs, and startup parameters.
* **Log Management**: Subprocess stdout and stderr are combined into `runtime/logs/<name>.log`. Upon restart, the previous log is rotated to `<name>.log.prev`.
* **Readiness Loops**: When starting a model, the supervisor polls the HTTP `/health` probe until the server signals readiness.

### 4. Multi-Engine Runtime Adapters (`inferops.engines`)
* **vLLM Adapter**: Spawns `vllm.entrypoints.openai.api_server` with `--served-model-name`, `--gpu-memory-utilization`, `--enable-lora`, and tensor-parallel flags.
* **SGLang Adapter**: Spawns `sglang.launch_server` with `--mem-fraction-static`, `--tp`, and `--context-length`.

### 5. Intelligent Gateway & Reverse Proxy (`inferops.gateway`)
* **Dynamic Routing**: An ASGI FastAPI reverse proxy on port 8000 intercepts `/v1/chat/completions` and `/v1/models`.
* **SSE Streaming**: Forwards Server-Sent Events (`stream: true`) chunk by chunk with minimal latency.
* **Scale-to-Zero & Cold Starts**: Shuts down idle instances after configured inactivity windows, and transparently initiates background cold starts when incoming requests arrive.
* **Token Accounting & Cost Estimation**: Records token throughput in `runtime/usage.db` (SQLite) and computes real-time commercial cost savings compared to closed cloud APIs.
* **API Key Auth**: Cryptographically hashes and validates bearer tokens (`sk-inferops-...`) with per-key rate limits.

### 6. Hardware Discovery & Interconnect Topology (`inferops.hardware`)
* **GPU Telemetry**: Scans NVIDIA devices using PyNVML or `nvidia-smi` to monitor VRAM, temperature, load, and wattage.
* **Interconnect Analysis**: Parses `nvidia-smi topo -m` to classify interconnect links (NVLink vs PCIe bridges vs System Bus) and evaluate Tensor Parallelism feasibility.

### 7. Production DevOps Exporters (`inferops.export`)
* **Docker Compose**: Generates valid `docker-compose.yml` with NVIDIA GPU container passthrough reservations, healthchecks, and cache volume mounts.
* **Kubernetes Manifests**: Generates production `Deployment` and `Service` YAML specifications with GPU resource limits (`nvidia.com/gpu: <count>`), probes, and shared memory volumes.

### 8. Dual Interfaces (TUI & Web Dashboard)
* **Terminal UI**: Textual-based live dashboard (`inferops tui`) for headless server monitoring.
* **Web Dashboard**: Modern browser interface (`inferops web`) featuring real-time GPU cards, model controls, predictive VRAM Sizer, interactive Chat Playground, Live Terminal Console via WebSockets, and Cost Analytics.
