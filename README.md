# InferOps

Bare-metal control plane and multi-engine supervisor for local LLM inference.

InferOps manages `vLLM` and `SGLang` processes from declarative YAML manifests, calculates VRAM requirements before launching models to prevent out-of-memory errors, exposes an OpenAI-compatible routing gateway, and provides both a terminal interface and a web dashboard with a live chat playground.

![InferOps Demo Walkthrough](docs/images/inferops_demo.gif)

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

## Key Capabilities

* **Multi-Engine Support**: Run `vLLM` and `SGLang` side by side with consistent lifecycle commands (`start`, `stop`, `restart`, `status`, `health`, `logs`).
* **Declarative Configuration**: Define model runtimes in `configs/models/<name>.yaml` and group models into execution profiles (`dev`, `prod`).
* **Predictive VRAM Sizing**: Calculates weights, GQA KV cache, and runtime overhead before starting a model to verify that GPU memory is sufficient.
* **Unified Gateway**: Built-in OpenAI-compatible reverse proxy at `/v1/chat/completions` and `/v1/models` that dynamically routes requests to running instances with server-sent events (SSE) streaming.
* **Automated Tuning**: Evaluates available GPU topology to suggest optimal tensor parallelism, max model length, and KV cache allocation (`inferops tune`).
* **Synthetic Benchmarks**: Measures time-to-first-token (TTFT) and token generation throughput with synthetic concurrency tests (`inferops benchmark`).
* **Dual Dashboards**: Real-time terminal UI with GPU gauges (`inferops tui`) and browser interface (`inferops web`) with an interactive chat playground.
* **Cross-Platform Supervisor**: Clean process isolation, health checks, log rotation, and graceful shutdown handling across Linux, macOS, WSL2, and Windows.

---

## Interface Preview

### Fleet and Model Management
Live monitoring of active models, runtime ports, health readiness probes, and GPU resource utilization.
![InferOps Fleet Overview](docs/images/inferops-1.png)

### Predictive VRAM Calculator
Calculates parameter weights, KV cache overhead based on context length, and CUDA runtime buffers to prevent crashes before launching inference processes.
![InferOps VRAM Sizer](docs/images/inferops-2.png)

### Interactive Chat Playground
Directly test streaming generation and latency on any active model through the integrated web dashboard.
![InferOps Chat Playground](docs/images/inferops-3.png)

---

## Installation

InferOps requires Python 3.10+.

```bash
git clone https://github.com/alexandrmotologa/inferops.git
cd inferops
python -m venv .venv

# On Linux or macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

pip install -e ".[dev]"
```

---

## Quick Start

### 1. Initialize a Project Workspace

```bash
inferops init
```

This generates the standard directory structure:
```
.
|-- configs/
|   |-- models/
|   |   `-- qwen2.5-coder-7b.yaml
|   `-- profiles.yaml
|-- runtime/
|   |-- logs/
|   `-- pids/
`-- .env
```

### 2. Verify GPU Memory Capacity

Check whether a target model fits within physical GPU memory before starting:

```bash
inferops vram configs/models/qwen2.5-coder-7b.yaml
```

Output:
```
VRAM Sizing Report: Qwen/Qwen2.5-Coder-7B-Instruct
Weights (FP16):           14.00 GB
KV Cache (32768 ctx):      4.00 GB
Overhead & CUDA Runtime:   1.50 GB
Total Required VRAM:      19.50 GB
GPU Memory Available:     24.00 GB
Status:                   [PASS] Fits with 4.50 GB margin
```

### 3. Generate Tuned Recommendations

Let InferOps analyze GPU memory and recommend optimized parameters:

```bash
inferops tune configs/models/qwen2.5-coder-7b.yaml
```

### 4. Start Models

Start an individual model or an entire profile:

```bash
# Start a single model:
inferops start qwen2.5-coder-7b

# Or start all models in a profile:
inferops start --profile dev
```

### 5. Check Fleet Status

```bash
inferops status
```

### 6. Query the Unified OpenAI Gateway

Start the routing proxy and send chat completion requests:

```bash
# Start proxy in background or standalone:
inferops proxy --port 8000

# Send request:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [{"role": "user", "content": "Write a quicksort in Python"}]
  }'
```

### 7. Run Synthetic Benchmarks

```bash
inferops benchmark qwen2.5-coder-7b --concurrency 4 --requests 20
```

### 8. Launch the Web Dashboard or Terminal UI

```bash
# Web dashboard:
inferops web --port 8900

# Terminal UI:
inferops tui
```

---

## CLI Reference Summary

| Command | Description |
|---|---|
| `inferops init` | Scaffolds configs, profiles, and runtime directories |
| `inferops doctor` | Inspects NVIDIA/ROCm GPU environment and engine binaries |
| `inferops vram <manifest>` | Computes mathematical VRAM requirements |
| `inferops tune <manifest>` | Generates hardware-optimized engine parameters |
| `inferops model list` | Lists all discovered model manifests |
| `inferops model create` | Interactively generates a new model configuration |
| `inferops start <model>` | Launches a model with readiness verification |
| `inferops stop <model>` | Gracefully terminates an inference process |
| `inferops restart <model>` | Restarts a model process |
| `inferops status` | Displays process table, PIDs, ports, and health |
| `inferops logs <model>` | Streams runtime engine logs |
| `inferops benchmark <model>`| Profiles throughput and TTFT under synthetic load |
| `inferops proxy` | Starts the OpenAI-compatible reverse proxy |
| `inferops web` | Launches the browser dashboard and playground |
| `inferops tui` | Opens the Textual terminal monitor |

For full CLI documentation, see [docs/cli-reference.md](docs/cli-reference.md).

---

## Architecture

InferOps separates orchestration into decoupled components:

* `src/inferops/core/`: Pydantic schema validation, environment variable interpolation, VRAM estimator, benchmark runner, and the cross-platform process supervisor.
* `src/inferops/engines/`: Engine adapters that translate declarative manifests into exact CLI flags for `vLLM` or `SGLang`.
* `src/inferops/hardware/`: Hardware detection via NVML and metrics aggregation.
* `src/inferops/gateway/`: Dynamic ASGI reverse proxy with SSE streaming and LiteLLM configuration generator.
* `src/inferops/web/`: Embedded FastAPI server with WebSocket telemetry and a dark-mode browser dashboard.
* `src/inferops/tui/`: Textual terminal application for low-overhead server monitoring.

---

## License

InferOps is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
