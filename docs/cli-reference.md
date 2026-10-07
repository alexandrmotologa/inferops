# CLI Reference

The `inferops` command-line utility provides comprehensive controls for process supervision, hardware diagnostics, predictive VRAM sizing, automated tuning, benchmarking, HuggingFace integration, and production export.

## Global Options

* `--help`: Display CLI usage and command options.
* `--version`: Print currently installed InferOps version.

---

## Commands

### `inferops init`
Initialize an InferOps workspace in the current working directory.
* Creates `configs/models/` with sample vLLM (`qwen2.5-coder-7b.yaml`) and SGLang (`llama3.1-8b.yaml`) manifests.
* Creates `configs/profiles.yaml` with `dev` and `prod` groups.
* Creates `runtime/logs/` and `runtime/pids/`.

```bash
inferops init
```

---

### `inferops pull <REPO_ID>`
Fetch model architecture from HuggingFace, synthesize a validated YAML manifest, and run an analytical VRAM sizing calculation against available GPU memory.

Options:
* `-p, --port <INT>`: Port to assign to the new model instance (default: 8001).
* `-e, --engine <STR>`: Target engine: `vllm` or `sglang` (default: vllm).
* `-t, --token <STR>`: Optional HuggingFace access token for gated models.
* `-d, --download`: Automatically download model weights via `huggingface-cli`.

```bash
inferops pull meta-llama/Llama-3.1-8B-Instruct --port 8005
```

---

### `inferops status`
Display a table of configured models, runtime engines, ports, assigned GPUs, active PIDs, and lifecycle states (`HEALTHY`, `STARTING`, `STOPPED`, `CRASHED`).

```bash
inferops status
```

---

### `inferops start [MODEL_NAME]`
Start an individual inference process or an entire profile group.

Options:
* `-p, --profile <NAME>`: Start all models grouped under the profile.
* `--no-wait`: Launch process in background without blocking on `/health` probe.
* `-t, --timeout <SECONDS>`: Seconds to wait for health readiness (default: 180s).

```bash
# Start a single model
inferops start qwen2.5-coder-7b

# Start all models in the 'dev' profile
inferops start --profile dev --no-wait
```

---

### `inferops stop [MODEL_NAME]`
Stop a running model or an entire profile group gracefully.

Options:
* `-p, --profile <NAME>`: Stop all models belonging to the specified profile.

```bash
# Stop a single model
inferops stop qwen2.5-coder-7b

# Stop all running models
inferops stop
```

---

### `inferops vram <MODEL_NAME_OR_PATH>`
Run an analytical VRAM sizing pre-flight check.

Options:
* `-c, --context <INT>`: Target context length in tokens (default: 8192).
* `--dtype <STR>`: Data precision (auto, fp16, bfloat16, fp8) (default: auto).
* `-q, --quant <STR>`: Quantization format (awq, gptq, fp8, bitsandbytes).
* `--tp <INT>`: Tensor parallelism degree (default: 1).

```bash
# Sizing check for a 7B model
inferops vram qwen2.5-coder-7b

# Sizing check for a 70B model with AWQ quantization and TP=2
inferops vram meta-llama/Llama-3.1-70B-Instruct --quant awq --tp 2
```

---

### `inferops tune <MANIFEST_PATH>`
Inspect available GPU memory and topology to calculate optimal tensor parallelism, context length, and KV cache allocation parameters.

```bash
inferops tune configs/models/qwen2.5-coder-7b.yaml
```

---

### `inferops benchmark <MODEL_NAME>`
Run synthetic load profiling against an active model to measure Time-to-First-Token (TTFT) and token generation throughput.

Options:
* `-c, --concurrency <INT>`: Number of concurrent client streams (default: 2).
* `-r, --requests <INT>`: Total requests to execute (default: 10).
* `-t, --tokens <INT>`: Target output tokens per request (default: 64).

```bash
inferops benchmark qwen2.5-coder-7b --concurrency 4 --requests 20
```

---

### `inferops doctor`
Run environment diagnostics, verifying Python version, engine binaries, and GPU devices.

Options:
* `--deep`: Probe multi-GPU interconnect topology (NVLink vs PCIe) and evaluate Tensor Parallelism readiness.

```bash
inferops doctor --deep
```

---

### `inferops proxy`
Start the dynamic reverse proxy routing OpenAI-compatible `/v1/chat/completions` and `/v1/models` requests to running instances with streaming SSE, scale-to-zero, and token accounting.

Options:
* `-p, --port <INT>`: Port to bind (default: 8000).
* `-h, --host <STR>`: Host address (default: 0.0.0.0).
* `--auth`: Enforce API key authentication via `Authorization: Bearer sk-inferops-...`.

```bash
inferops proxy --port 8000 --auth
```

---

### `inferops usage`
Display aggregated token consumption statistics and commercial cost savings compared to closed cloud LLM APIs.

```bash
inferops usage
```

---

### `inferops key` (API Key Management)

* `inferops key create <NAME> [--rpm <LIMIT>]`: Generate a new API key with a rate limit.
* `inferops key list`: List all active and revoked API keys.
* `inferops key revoke <KEY_ID>`: Revoke an existing API key immediately.

```bash
inferops key create internal-agent --rpm 120
inferops key list
inferops key revoke key_ab4c28f68257
```

---

### `inferops export` (DevOps Manifests)

* `inferops export docker-compose [--output <FILE>] [--profile <NAME>]`: Generate a production-ready `docker-compose.yml` with NVIDIA GPU container passthrough.
* `inferops export k8s <MODEL_NAME> [--output <FILE>] [--namespace <NAME>]`: Generate Kubernetes Deployment and Service YAML manifests with GPU resource limits and probes.

```bash
inferops export docker-compose --output docker-compose.yml
inferops export k8s qwen2.5-coder-7b --output k8s-deployment.yaml
```

---

### `inferops web`
Start the browser dashboard featuring GPU monitoring, model lifecycle actions, predictive VRAM calculator, interactive Chat Playground, Live Terminal Console, and Token Analytics.

Options:
* `-p, --port <INT>`: Web dashboard port (default: 8900).
* `-h, --host <STR>`: Host to bind (default: 127.0.0.1).

```bash
inferops web --port 8900
```

---

### `inferops tui`
Launch the Textual interactive terminal dashboard for headless server monitoring.

```bash
inferops tui
```
