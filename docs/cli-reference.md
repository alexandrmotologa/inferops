# CLI Reference

The `inferops` CLI provides lifecycle controls, hardware diagnostics, and memory sizing tools.

## Global Options

- `--help`: Show CLI usage and available commands.
- `--version`: Display currently installed InferOps version.

---

## Commands

### `inferops init`
Initialize an InferOps workspace in the current directory.
- Creates `configs/models/` with sample vLLM (`qwen2.5-coder-7b.yaml`) and SGLang (`llama3.1-8b.yaml`) models.
- Creates `configs/profiles.yaml` with `dev` and `prod` groups.
- Sets up `runtime/logs/` and `runtime/pids/`.

```bash
inferops init
```

---

### `inferops status`
Display a table showing all configured models, ports, assigned GPUs, PIDs, and active lifecycle states (`HEALTHY`, `STARTING`, `STOPPED`, `CRASHED`).

```bash
inferops status
```

---

### `inferops start [MODEL_NAME]`
Start an inference model or an entire profile group.

Options:
- `-p, --profile <NAME>`: Start all models grouped under the specified profile in parallel.
- `--no-wait`: Launch the process in background without blocking on the `/health` check.
- `-t, --timeout <SECONDS>`: Seconds to wait for readiness (default: 180s).

```bash
# Start a single model and wait for health
inferops start qwen2.5-coder-7b

# Start all models in the 'dev' profile
inferops start --profile dev --no-wait
```

---

### `inferops stop [MODEL_NAME]`
Stop a running model or an entire profile group.

Options:
- `-p, --profile <NAME>`: Stop all models in the specified profile.

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
- `-c, --context <INT>`: Target sequence context length in tokens (default: 8192).
- `--dtype <STR>`: Data precision (auto, fp16, bfloat16, fp8) (default: auto).
- `-q, --quant <STR>`: Quantization format (awq, gptq, fp8, bitsandbytes).
- `--tp <INT>`: Tensor parallelism degree (default: 1).

```bash
# Check memory requirements for a 7B model
inferops vram qwen2.5-coder-7b

# Check a 70B model with 4-bit quantization and TP=2
inferops vram meta-llama/Llama-3.1-70B-Instruct --quant awq --tp 2
```

---

### `inferops doctor`
Run environment diagnostics, checking:
- Python version and operating system platform.
- Availability of `vllm` and `sglang` executables or modules.
- Detected NVIDIA GPU cards, available VRAM, and driver status.

```bash
inferops doctor
```

---

### `inferops web`
Start the embedded browser dashboard and chat playground.

Options:
- `-p, --port <INT>`: Port to bind the web server (default: 8900).
- `-h, --host <STR>`: Host to bind (default: 127.0.0.1).

```bash
inferops web --port 8900
```

---

### `inferops tui`
Launch the Textual interactive terminal dashboard.

```bash
inferops tui
```

---

### `inferops proxy`
Start the dynamic reverse proxy routing OpenAI-compatible `/v1/chat/completions` requests to running models.

Options:
- `-p, --port <INT>`: Port for the gateway (default: 8000).
- `-h, --host <STR>`: Host address (default: 0.0.0.0).

```bash
inferops proxy --port 8000
```
