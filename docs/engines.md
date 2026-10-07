# Multi-Engine Guide: vLLM, SGLang, and llama.cpp

InferOps treats **vLLM**, **SGLang**, and **llama.cpp** as first-class engines. You configure individual models to use any of these engines by setting the `engine` parameter in your model manifest.

## Engine Comparison

| Feature | vLLM | SGLang | llama.cpp |
| :--- | :--- | :--- | :--- |
| **Primary Strength** | Broad model ecosystem, multi-modal, vLLM V1 engine | RadixAttention (prefix caching), structured outputs, high concurrency | GGUF quantizations, Apple Silicon (Metal), CPU servers, consumer GPUs |
| **Target Hardware** | NVIDIA CUDA, AMD ROCm, Host CPU | NVIDIA CUDA, AMD ROCm | Apple Silicon Metal, NVIDIA/AMD GPUs, Host CPU |
| **Model Formats** | SafeTensors, FP16, AWQ, GPTQ, FP8 | SafeTensors, FP16, AWQ, FP8 | GGUF (`Q4_K_M`, `Q5_K_M`, `Q8_0`, `Q2_K`) |
| **Radix / Prefix Caching** | Automatic Prefix Caching (APC) | Native Radix Tree with LRU cache eviction | Prompt cache slots |
| **Structured Output** | Outlines / Guided Decoding | Fast regex and grammar-constrained decoding | GBNF Grammars |
| **Invocation** | `vllm serve <model>` | `python -m sglang.launch_server --model-path <model>` | `llama-server -m <model>` |

## Configuration Mapping

InferOps maps unified declarative YAML properties to the respective engine arguments automatically:

| ModelConfig Field | vLLM Argument | SGLang Argument | llama.cpp Argument |
| :--- | :--- | :--- | :--- |
| `port` | `--port <port>` | `--port <port>` | `--port <port>` |
| `host` | `--host <host>` | `--host <host>` | `--host <host>` |
| `served_model_name` | `--served-model-name <name>` | `--served-model-name <name>` | `--alias <name>` |
| `tensor_parallel_size` | `--tensor-parallel-size <N>` | `--tp <N>` | N/A (single process) |
| `gpu_memory_utilization` | `--gpu-memory-utilization <ratio>` | `--mem-fraction-static <ratio>` | N/A |
| `max_model_len` | `--max-model-len <tokens>` | `--context-length <tokens>` | `-c <tokens>` |
| `gpu_layers` | N/A | N/A | `-ngl <N>` (99 for full GPU offload) |
| `threads` | N/A | N/A | `-t <N>` (CPU worker threads) |
| `dtype` | `--dtype <type>` | `--dtype <type>` | N/A (encoded in GGUF) |
| `quantization` | `--quantization <quant>` | `--quantization <quant>` | N/A (encoded in GGUF) |
| `kv_cache_dtype` | `--kv-cache-dtype <type>` | `--kv-cache-dtype <type>` | N/A |

## Sample Configurations

### vLLM Configuration (`configs/models/qwen2.5-coder-7b.yaml`)
```yaml
name: qwen2.5-coder-7b
model: Qwen/Qwen2.5-Coder-7B-Instruct
engine: vllm
served_model_name: qwen2.5-coder-7b
port: 8001
gpus: [0]
tensor_parallel_size: 1
max_model_len: 8192
gpu_memory_utilization: 0.90
dtype: auto
extra_args:
  - --disable-log-requests
```

### SGLang Configuration (`configs/models/llama3.1-8b.yaml`)
```yaml
name: llama3.1-8b
model: meta-llama/Llama-3.1-8B-Instruct
engine: sglang
served_model_name: llama3.1-8b
port: 8002
gpus: [0]
tensor_parallel_size: 1
max_model_len: 8192
gpu_memory_utilization: 0.90
dtype: auto
extra_args:
  - --disable-cuda-graph
```

### llama.cpp GGUF Configuration (`configs/models/llama3.2-3b-gguf.yaml`)
```yaml
name: llama3.2-3b-gguf
model: ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
engine: llamacpp
served_model_name: llama3.2-3b-gguf
port: 8003
gpu_layers: 99
threads: 8
max_model_len: 4096
extra_args:
  - --cont-batching
```
