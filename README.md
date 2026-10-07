# InferOps

Bare-metal control plane and multi-engine orchestrator for local LLM inference servers.

InferOps manages `vLLM` and `SGLang` processes from declarative YAML files, checks available GPU VRAM before starting models to prevent CUDA OOM, provides an OpenAI-compatible unified proxy, and includes both a terminal UI and a browser dashboard with a live chat playground.

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

## Features

- **Multi-Engine Support**: Run `vLLM` and `SGLang` side by side with consistent lifecycle management (`start`, `stop`, `restart`, `status`, `health`, `logs`).
- **Declarative YAML Config**: Define each model in `configs/models/<name>.yaml` and group models into profiles (`dev`, `prod`).
- **Smart VRAM Pre-flight Check**: Calculates parameter weights, KV-cache consumption, and activation overhead before starting a model to ensure it fits in GPU memory.
- **Unified Gateway**: Built-in OpenAI-compatible reverse proxy at `/v1/chat/completions` and `/v1/models` that dynamically routes requests to running instances.
- **Dual Interfaces**: Live terminal dashboard via Textual (`inferops tui`) and browser interface (`inferops web`) with an interactive chat playground and GPU charts.
- **Cross-Platform Process Supervisor**: Clean process isolation, health checks, log rotation, and graceful shutdowns across Linux, macOS, WSL2, and Windows.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/alexandrmotologa/inferops.git
cd inferops
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -e ".[dev]"
```

## Quick Start

1. Initialize a project workspace:
```bash
inferops init
```

This creates:
```
.
├── configs/
│   ├── models/
│   │   └── qwen2.5-coder-7b.yaml
│   └── profiles.yaml
├── runtime/
│   ├── logs/
│   └── pids/
└── .env
```

2. Validate GPU capacity before launching:
```bash
inferops vram configs/models/qwen2.5-coder-7b.yaml
```

3. Start a model or an entire profile:
```bash
inferops start qwen2.5-coder-7b
# or run a full profile:
inferops start --profile dev
```

4. Launch the web dashboard and chat playground:
```bash
inferops web --port 8900
```

5. Query the unified OpenAI endpoint:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [{"role": "user", "content": "Write a quicksort in Python"}]
  }'
```

## Architecture

InferOps decouples configuration, process management, and telemetry into modular layers:

- `src/inferops/core/`: Pydantic schema validation, environment variable interpolation, and the async process supervisor.
- `src/inferops/engines/`: Adapters translating unified model definitions into CLI arguments for `vLLM` or `SGLang`.
- `src/inferops/hardware/`: GPU discovery via NVML, ROCm SMI, and metric scrapers.
- `src/inferops/gateway/`: Dynamic ASGI reverse proxy that multiplexes requests to model ports.
- `src/inferops/web/`: Embedded FastAPI server with WebSocket metrics and an interactive test client.
- `src/inferops/tui/`: Textual-based terminal dashboard with real-time token throughput and memory telemetry.

## License

Apache-2.0
