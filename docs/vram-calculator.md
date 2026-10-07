# Predictive VRAM Sizing Engine

InferOps calculates estimated GPU memory requirements prior to launching models. This pre-flight validation prevents out-of-memory errors caused by allocating too much context or running an unquantized model on insufficient hardware.

## Mathematical Formulation

Total GPU memory required by an inference process is composed of three components:

$$\text{VRAM}_{\text{total}} = \text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{KV-cache}} + \text{VRAM}_{\text{overhead}}$$

Under Tensor Parallelism degree $TP$, weights and KV cache are split across GPUs:

$$\text{VRAM}_{\text{per\_gpu}} = \frac{\text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{KV-cache}}}{TP} + \text{VRAM}_{\text{overhead}}$$

---

### 1. Weights Memory ($\text{VRAM}_{\text{weights}}$)

Given parameter count $N$ (in billions) and precision bytes per parameter $B_p$:

$$\text{VRAM}_{\text{weights}} = \frac{N \times 10^9 \times B_p}{1024^3} \text{ GB}$$

#### Precision Scaling Factors

| Format / Quantization | Bytes per Parameter ($B_p$) | Description |
| :--- | :--- | :--- |
| **FP32** | 4.0 | Full precision floating point |
| **FP16 / BF16** | 2.0 | Standard 16-bit half precision |
| **FP8 (e4m3 / e5m2)** | 1.05 | 8-bit float with scale factors |
| **INT8** | 1.10 | 8-bit integer with scales |
| **AWQ / GPTQ (4-bit)** | 0.55 | 4-bit weights + zero-point scales |

---

### 2. Key-Value Cache Memory ($\text{VRAM}_{\text{KV-cache}}$)

Modern transformer architectures employ Grouped-Query Attention (GQA) or Multi-Query Attention (MQA) to reduce memory consumption.

For an architecture with:
- Number of layers: $L$
- Number of key-value heads: $H_{kv}$
- Head dimension: $D_h$
- Precision bytes per element: $B_{kv}$ (typically 2 for FP16, 1 for FP8)
- Target sequence context: $C$ tokens
- Concurrent active requests: $K$

The memory consumption per token is:

$$\text{Bytes per token} = 2 \times L \times H_{kv} \times D_h \times B_{kv}$$

The total KV cache requirement across $K$ concurrent requests is:

$$\text{VRAM}_{\text{KV-cache}} = \frac{\text{Bytes per token} \times C \times K}{1024^3} \text{ GB}$$

#### Example: Qwen 2.5 7B ($L=28, H_{kv}=4, D_h=128$)
- Bytes per token in FP16 ($B_{kv}=2$): $2 \times 28 \times 4 \times 128 \times 2 = 57,344 \text{ bytes} \approx 56 \text{ KB}$.
- Context $C = 8,192$ tokens: $56 \text{ KB} \times 8,192 \approx 458 \text{ MB}$ per sequence.
- At 16 concurrent requests: $458 \text{ MB} \times 16 \approx 7.16 \text{ GB}$.

---

### 3. Runtime & CUDA Overhead ($\text{VRAM}_{\text{overhead}}$)

A baseline memory footprint is reserved per GPU process for:
- CUDA context and kernel binaries (~600 MB - 1 GB).
- PyTorch CUDA caching allocator fragmentation.
- CUDA Graph capture workspace (~200 - 400 MB).

InferOps models this baseline overhead as $1.20 \text{ GB}$ per GPU process.

---

## Pre-flight Feasibility Check

When a target GPU is detected, InferOps compares:

$$\text{VRAM}_{\text{per\_gpu}} \le \text{GPU}_{\text{available}} \times 0.95$$

If required memory exceeds 95% of available capacity, InferOps outputs specific recommendations:
1. Apply 4-bit AWQ or 8-bit FP8 weight quantization.
2. Enable 8-bit FP8 KV cache (`kv_cache_dtype: fp8`).
3. Increase tensor parallelism degree ($TP=2$ or $TP=4$).
4. Reduce `max_model_len` (e.g. from 32,768 to 8,192).
