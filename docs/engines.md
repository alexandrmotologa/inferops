# Multi-Engine Guide: vLLM vs SGLang

InferOps treats both **vLLM** and **SGLang** as first-class engines. You can configure individual models to use either engine simply by switching the `engine` parameter in your model YAML file.

## Engine Comparison

| Feature | vLLM | SGLang |
| :--- | :--- | :--- |
| **Primary Strength** | Broad model ecosystem, multi-modal, vLLM V1 engine | RadixAttention (prefix caching), structured outputs, high-concurrency throughput |
| **Radix / Prefix Caching** | Automatic Prefix Caching (APC) | Native Radix Tree with LRU cache eviction |
| **Structured Output** | Outlines / Guided Decoding | Fast regex and grammar-constrained decoding |
| **Command Line Invocation** | `vllm serve <model>` | `python -m sglang.launch_server --model-path <model>` |

## Configuration Mapping

InferOps maps unified declarative YAML properties to the respective engine arguments automatically:

| ModelConfig Field | vLLM Argument | SGLang Argument |
| :--- | :--- | :--- |
| `port` | `--port <port>` | `--port <port>` |
| `host` | `--host <host>` | `--host <host>` |
| `served_model_name` | `--served-model-name <name>` | `--served-model-name <name>` |
| `tensor_parallel_size` | `--tensor-parallel-size <N>` | `--tp <N>` |
| `gpu_memory_utilization` | `--gpu-memory-utilization <ratio>` | `--mem-fraction-static <ratio>` |
| `max_model_len` | `--max-model-len <tokens>` | `--context-length <tokens>` |
| `dtype` | `--dtype <type>` | `--dtype <type>` |
| `quantization` | `--quantization <quant>` | `--quantization <quant>` |
| `kv_cache_dtype` | `--kv-cache-dtype <type>` | `--kv-cache-dtype <type>` |

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
