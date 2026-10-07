# Predictive VRAM Sizing Engine

InferOps calculates estimated GPU memory requirements prior to launching models. This pre-flight validation prevents out-of-memory errors caused by allocating too much context, high concurrency pools, or running an unquantized model on insufficient hardware.

## Mathematical Formulation

Total memory required by an inference process is composed of three primary components:

$$\text{Memory}_{\text{total}} = \text{Memory}_{\text{weights}} + \text{Memory}_{\text{KV-cache}} + \text{Memory}_{\text{overhead}}$$

Under Tensor Parallelism degree $TP$, weights and KV cache are partitioned across GPUs:

$$\text{VRAM}_{\text{per\_gpu}} = \frac{\text{VRAM}_{\text{weights}} + \text{VRAM}_{\text{KV-cache}}}{TP} + \text{VRAM}_{\text{overhead}}$$

---

### 1. Weights Memory ($\text{Memory}_{\text{weights}}$)

Given parameter count $N$ (in billions) and precision bytes per parameter $B_p$:

$$\text{Memory}_{\text{weights}} = \frac{N \times 10^9 \times B_p}{1024^3} \text{ GB}$$

#### Precision Scaling Factors

| Format / Quantization | Bytes per Parameter ($B_p$) | Description |
| :--- | :--- | :--- |
| **FP32** | 4.00 | Full precision floating point |
| **FP16 / BF16** | 2.00 | Standard 16-bit half precision |
| **FP8 (e4m3 / e5m2)** | 1.05 | 8-bit float with block scaling |
| **INT8** | 1.10 | 8-bit integer with quantization scales |
| **AWQ / GPTQ (4-bit)** | 0.55 | 4-bit weights + zero-point overhead |
| **GGUF Q8_0** | 1.08 | 8-bit quantized GGUF format |
| **GGUF Q5_K_M** | 0.72 | 5-bit medium K-quants |
| **GGUF Q4_K_M** | 0.58 | 4-bit medium K-quants |
| **GGUF Q2_K / Q3_K** | 0.35 - 0.45 | Extreme low-bit K-quants |

#### GGUF Partial Layer Offloading (`gpu_layers`)

When using `llama.cpp` with partial GPU offloading (`gpu_layers = N` of total $L$ layers):

$$\text{VRAM}_{\text{weights}} = \text{Memory}_{\text{weights}} \times \min\left(1.0, \frac{\text{gpu\_layers}}{L}\right)$$

The remaining weights remain mapped in Host RAM without consuming VRAM.

#### Mixture-of-Experts (MoE)

For MoE models (e.g. Mixtral 8x7B, Mixtral 8x22B, DeepSeek-V3 671B), all expert weights must reside in memory (GPU or Host RAM), while per-token computation only activates a subset of experts ($K_{\text{active}}$):

$$\text{Total Parameters} = N_{\text{shared}} + E \times N_{\text{expert}}$$

---

### 2. Key-Value Cache Memory ($\text{Memory}_{\text{KV-cache}}$)

#### Standard GQA / MHA Attention
For standard Multi-Head Attention (MHA) or Grouped-Query Attention (GQA):

$$\text{Bytes per token} = 2 \times L \times H_{kv} \times D_h \times B_{kv}$$

Where:
- $L$: Number of layers
- $H_{kv}$: Number of key-value heads
- $D_h$: Dimension per attention head
- $B_{kv}$: Bytes per element (2 for FP16, 1 for FP8 KV cache)
- $C$: Target context length
- $K$: Concurrent request capacity

$$\text{Memory}_{\text{KV-cache}} = \frac{\text{Bytes per token} \times C \times K}{1024^3} \text{ GB}$$

#### DeepSeek MLA (Multi-Head Latent Attention)
DeepSeek-V3 and DeepSeek-R1 use Multi-Head Latent Attention (MLA), which compresses keys and values into a low-rank latent vector:

$$\text{Latent Dimension} = D_{\text{kv\_lora\_rank}} + D_{\text{qk\_rope\_head\_dim}} = 512 + 64 = 576$$

The KV cache memory per token drops to:

$$\text{Bytes per token}_{\text{MLA}} = L \times 576 \times B_{kv}$$

For DeepSeek-V3 ($L=61$ layers):
$$\text{Bytes per token} = 61 \times 576 \times 2 = 70,272 \text{ bytes} \approx 68.6 \text{ KB}$$
Compared to standard MHA which would require $>3.5 \text{ MB}$ per token, MLA reduces KV memory consumption by over **93%**, allowing large concurrent batch sizes on modest VRAM budgets.

---

### 3. Runtime & Overhead

* **CUDA / ROCm Runtime Overhead**: $1.20 \text{ GB}$ baseline per GPU for CUDA context, PyTorch caching allocator, and CUDA Graph capture.
* **Apple Silicon / CPU**: $0.20 \text{ GB}$ baseline overhead for unified memory buffers.

---

### 4. Pre-flight Feasibility Check

InferOps compares calculated requirements against available device memory:

$$\text{VRAM}_{\text{required}} \le \text{Hardware}_{\text{available}} \times 0.95$$

If required memory exceeds 95% of available capacity, InferOps outputs recommended mitigation steps:
1. Apply 4-bit AWQ/GGUF quantization.
2. Enable FP8 KV cache (`kv_cache_dtype: fp8`).
3. Scale tensor parallelism ($TP=2$ or $TP=4$) across available GPUs.
4. Adjust context length or GPU layer offload ratio (`gpu_layers`).
