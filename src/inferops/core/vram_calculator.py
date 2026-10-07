"""Intelligent VRAM sizing and pre-flight feasibility calculator for LLM serving."""

import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ModelArchitectureSpec:
    """Architectural specs needed to compute exact model weights and KV cache."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    approx_params_b: float
    is_moe: bool = False
    active_params_b: Optional[float] = None
    is_mla: bool = False  # Multi-Head Latent Attention (DeepSeek)
    mla_kv_dim: int = 576  # kv_lora_rank (512) + qk_rope_head_dim (64)
    is_vision: bool = False
    vision_encoder_gb: float = 0.0


# Catalog of prominent model architectures
KNOWN_MODEL_ARCHITECTURES: dict[str, ModelArchitectureSpec] = {
    # Llama 3 / 3.1 / 3.2 family
    "llama-3-8b": ModelArchitectureSpec(32, 4096, 32, 8, 128, 128256, 8.03),
    "llama-3.1-8b": ModelArchitectureSpec(32, 4096, 32, 8, 128, 128256, 8.03),
    "llama-3.2-1b": ModelArchitectureSpec(16, 2048, 32, 8, 64, 128256, 1.23),
    "llama-3.2-3b": ModelArchitectureSpec(28, 3072, 24, 8, 128, 128256, 3.21),
    "llama-3-70b": ModelArchitectureSpec(80, 8192, 64, 8, 128, 128256, 70.6),
    "llama-3.1-70b": ModelArchitectureSpec(80, 8192, 64, 8, 128, 128256, 70.6),
    "llama-3.2-11b-vision": ModelArchitectureSpec(40, 4096, 32, 8, 128, 128256, 10.6, is_vision=True, vision_encoder_gb=1.6),

    # Qwen 2 / 2.5 family
    "qwen-2.5-0.5b": ModelArchitectureSpec(24, 896, 14, 2, 64, 151936, 0.49),
    "qwen-2.5-1.5b": ModelArchitectureSpec(28, 1536, 12, 2, 128, 151936, 1.54),
    "qwen-2.5-3b": ModelArchitectureSpec(36, 2048, 16, 2, 128, 151936, 3.09),
    "qwen-2.5-7b": ModelArchitectureSpec(28, 3584, 28, 4, 128, 152064, 7.61),
    "qwen-2.5-14b": ModelArchitectureSpec(48, 5120, 40, 8, 128, 152064, 14.77),
    "qwen-2.5-32b": ModelArchitectureSpec(64, 5120, 40, 8, 128, 152064, 32.76),
    "qwen-2.5-72b": ModelArchitectureSpec(80, 8192, 64, 8, 128, 152064, 72.71),
    "qwen2-57b-a14b": ModelArchitectureSpec(28, 3584, 28, 4, 128, 152064, 57.0, is_moe=True, active_params_b=14.0),
    "qwen2-vl-7b": ModelArchitectureSpec(28, 3584, 28, 4, 128, 152064, 7.61, is_vision=True, vision_encoder_gb=1.2),
    "qwen2-vl-2b": ModelArchitectureSpec(24, 1536, 12, 2, 128, 152064, 2.21, is_vision=True, vision_encoder_gb=0.6),

    # Mistral & Mixtral MoE
    "mistral-7b": ModelArchitectureSpec(32, 4096, 32, 8, 128, 32000, 7.24),
    "mistral-nemo-12b": ModelArchitectureSpec(40, 5120, 32, 8, 128, 131072, 12.2),
    "mixtral-8x7b": ModelArchitectureSpec(32, 4096, 32, 8, 128, 32000, 46.7, is_moe=True, active_params_b=12.9),
    "mixtral-8x22b": ModelArchitectureSpec(56, 6144, 48, 8, 128, 32000, 141.0, is_moe=True, active_params_b=39.0),
    "pixtral-12b": ModelArchitectureSpec(40, 5120, 32, 8, 128, 131072, 12.4, is_vision=True, vision_encoder_gb=1.5),
    "mistral-large": ModelArchitectureSpec(88, 12288, 96, 8, 128, 32768, 123.0),

    # DeepSeek Dense & MoE (MLA KV-Cache)
    "deepseek-v3": ModelArchitectureSpec(61, 7168, 128, 128, 128, 129280, 671.0, is_moe=True, active_params_b=37.0, is_mla=True, mla_kv_dim=576),
    "deepseek-r1": ModelArchitectureSpec(61, 7168, 128, 128, 128, 129280, 671.0, is_moe=True, active_params_b=37.0, is_mla=True, mla_kv_dim=576),
    "deepseek-v2": ModelArchitectureSpec(60, 5120, 128, 128, 128, 102400, 236.0, is_moe=True, active_params_b=21.0, is_mla=True, mla_kv_dim=576),
    "deepseek-v2-lite": ModelArchitectureSpec(27, 2048, 16, 16, 128, 102400, 15.7, is_moe=True, active_params_b=2.4, is_mla=True, mla_kv_dim=576),
    "deepseek-coder-6.7b": ModelArchitectureSpec(32, 4096, 32, 32, 128, 32256, 6.7),

    # Microsoft Phi family
    "phi-3-mini": ModelArchitectureSpec(32, 3072, 32, 32, 96, 32064, 3.82),
    "phi-3-medium": ModelArchitectureSpec(40, 5120, 40, 40, 128, 32064, 14.0),
    "phi-3.5-mini": ModelArchitectureSpec(32, 3072, 32, 32, 96, 32064, 3.82),
    "phi-4": ModelArchitectureSpec(40, 5120, 40, 10, 128, 100352, 14.7),

    # Gemma 2 family
    "gemma-2-2b": ModelArchitectureSpec(26, 2304, 8, 4, 256, 256000, 2.61),
    "gemma-2-9b": ModelArchitectureSpec(42, 3584, 16, 8, 256, 256000, 9.24),
    "gemma-2-27b": ModelArchitectureSpec(46, 4608, 32, 16, 128, 256000, 27.2),

    # Cohere Command
    "command-r-plus": ModelArchitectureSpec(64, 12288, 96, 8, 128, 256000, 104.0),
}


@dataclass
class VRAMEstimate:
    """Comprehensive memory sizing assessment result."""
    model_name: str
    params_billions: float
    precision_bytes_per_param: float
    weights_vram_gb: float
    kv_cache_vram_gb: float
    cuda_overhead_gb: float
    total_required_vram_gb: float
    vram_per_gpu_gb: float
    tensor_parallel_size: int
    context_length: int
    fits: bool
    is_moe: bool = False
    active_params_b: Optional[float] = None
    architecture_type: str = "dense"
    available_vram_per_gpu_gb: Optional[float] = None
    suggestion: str = ""


def infer_params_from_name(model_id: str) -> Tuple[float, Optional[ModelArchitectureSpec]]:
    """Infer parameter count and matching architecture from model name string."""
    normalized = model_id.lower().replace("_", "-").replace("/", "-")
    stripped = re.sub(r"-(coder|instruct|chat|it|base|v\d+)", "", normalized)

    # 1. Direct key match in catalog
    for key, spec in KNOWN_MODEL_ARCHITECTURES.items():
        if key in normalized or key in stripped:
            return spec.approx_params_b, spec

    # 2. DeepSeek specific patterns
    if "deepseek" in normalized:
        if "v3" in normalized:
            return 671.0, KNOWN_MODEL_ARCHITECTURES["deepseek-v3"]
        if "r1" in normalized:
            return 671.0, KNOWN_MODEL_ARCHITECTURES["deepseek-r1"]
        if "v2-lite" in normalized:
            return 15.7, KNOWN_MODEL_ARCHITECTURES["deepseek-v2-lite"]
        if "v2" in normalized:
            return 236.0, KNOWN_MODEL_ARCHITECTURES["deepseek-v2"]
        if "6.7b" in normalized or "7b" in normalized:
            return 6.7, KNOWN_MODEL_ARCHITECTURES["deepseek-coder-6.7b"]

    # 3. Mixtral MoE patterns
    if "mixtral" in normalized:
        if "8x22b" in normalized:
            return 141.0, KNOWN_MODEL_ARCHITECTURES["mixtral-8x22b"]
        if "8x7b" in normalized:
            return 46.7, KNOWN_MODEL_ARCHITECTURES["mixtral-8x7b"]

    # 4. Phi patterns
    if "phi-4" in normalized:
        return 14.7, KNOWN_MODEL_ARCHITECTURES["phi-4"]
    if "phi-3.5" in normalized:
        return 3.82, KNOWN_MODEL_ARCHITECTURES["phi-3.5-mini"]
    if "phi-3" in normalized:
        if "medium" in normalized or "14b" in normalized:
            return 14.0, KNOWN_MODEL_ARCHITECTURES["phi-3-medium"]
        return 3.82, KNOWN_MODEL_ARCHITECTURES["phi-3-mini"]

    # 5. Qwen family
    if "qwen" in normalized:
        if "vl" in normalized:
            if "72b" in normalized:
                return 72.71, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-72b"]
            if "2b" in normalized:
                return 2.21, KNOWN_MODEL_ARCHITECTURES["qwen2-vl-2b"]
            return 7.61, KNOWN_MODEL_ARCHITECTURES["qwen2-vl-7b"]
        if "72b" in normalized or "70b" in normalized:
            return 72.71, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-72b"]
        if "32b" in normalized:
            return 32.76, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-32b"]
        if "14b" in normalized:
            return 14.77, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-14b"]
        if "7b" in normalized:
            return 7.61, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-7b"]
        if "3b" in normalized:
            return 3.09, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-3b"]
        if "1.5b" in normalized:
            return 1.54, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-1.5b"]
        if "0.5b" in normalized:
            return 0.49, KNOWN_MODEL_ARCHITECTURES["qwen-2.5-0.5b"]

    # 6. Llama 3 family
    if "llama-3" in normalized or "llama3" in normalized:
        if "70b" in normalized:
            return 70.6, KNOWN_MODEL_ARCHITECTURES["llama-3.1-70b"]
        if "8b" in normalized:
            return 8.03, KNOWN_MODEL_ARCHITECTURES["llama-3.1-8b"]
        if "3b" in normalized:
            return 3.21, KNOWN_MODEL_ARCHITECTURES["llama-3.2-3b"]
        if "1b" in normalized:
            return 1.23, KNOWN_MODEL_ARCHITECTURES["llama-3.2-1b"]

    # 7. Regex patterns for parameter count e.g. 7b, 14b, 70b, 1.5b, 0.5b
    m = re.search(r"(\d+(?:\.\d+)?)\s*b(?:\b|[^a-z])", normalized)
    if m:
        try:
            val = float(m.group(1))
            return val, None
        except ValueError:
            pass

    return 7.0, None  # Default heuristic fallback to 7B


def get_bytes_per_param(dtype: str, quantization: Optional[str] = None) -> float:
    """Calculate average bytes per parameter for given dtype and quantization."""
    q = (quantization or "").lower()
    d = (dtype or "auto").lower()

    if q in ("awq", "gptq", "int4", "q4_k_m", "q4_0", "q4_1"):
        return 0.55  # 4-bit with metadata/scales
    if q in ("q5_k_m", "q5_0", "q5_1", "int5"):
        return 0.68  # 5-bit
    if q in ("q6_k", "int6"):
        return 0.82  # 6-bit
    if q in ("int3", "q3_k_m", "q3_k_s"):
        return 0.42
    if q in ("q2_k", "int2"):
        return 0.32
    if q in ("fp8", "fp8_e4m3", "fp8_e5m2"):
        return 1.05
    if q in ("int8", "q8_0"):
        return 1.10
    if q in ("bitsandbytes", "bnb"):
        return 0.60
    if q == "gguf":
        return 0.55  # Default GGUF assumption to 4-bit

    if d in ("fp8", "float8"):
        return 1.05
    if d in ("float16", "fp16", "bfloat16", "bf16", "auto"):
        return 2.0
    if d in ("float32", "fp32"):
        return 4.0

    return 2.0


def calculate_vram_requirements(
    model_name_or_path: str,
    context_length: int = 8192,
    dtype: str = "auto",
    quantization: Optional[str] = None,
    tensor_parallel_size: int = 1,
    concurrent_requests: int = 16,
    kv_cache_dtype: str = "auto",
    available_vram_per_gpu_gb: Optional[float] = None,
    gpu_layers: Optional[int] = None,
) -> VRAMEstimate:
    """Compute exact required GPU memory breakdown and feasibility."""
    tp = max(1, tensor_parallel_size)
    params_b, spec = infer_params_from_name(model_name_or_path)

    bytes_per_param = get_bytes_per_param(dtype, quantization)

    # 1. Weights memory
    weights_gb = (params_b * 1e9 * bytes_per_param) / (1024**3)

    # Add vision encoder overhead if multimodal model
    if spec and spec.is_vision:
        weights_gb += spec.vision_encoder_gb

    # Calculate partial layer offload (e.g. for llama.cpp -ngl)
    if gpu_layers is not None:
        total_layers = spec.num_layers if spec else 32
        offload_ratio = min(1.0, max(0.0, gpu_layers / total_layers))
        weights_gb = weights_gb * offload_ratio

    # 2. KV Cache memory
    kv_bytes = 1.0 if kv_cache_dtype.lower() in ("fp8", "int8") else 2.0
    total_tokens = context_length * concurrent_requests

    if spec:
        if spec.is_mla:
            # DeepSeek Multi-Head Latent Attention (MLA) compressed KV-cache:
            # Key & Value are compressed into low-rank latent vector:
            # bytes_per_token = layers * mla_kv_dim * precision
            bytes_per_token = spec.num_layers * spec.mla_kv_dim * kv_bytes
            kv_cache_gb = (bytes_per_token * total_tokens) / (1024**3)
        else:
            # Standard GQA / MHA: 2 * layers * kv_heads * head_dim * precision * tokens
            bytes_per_token = 2 * spec.num_layers * spec.num_kv_heads * spec.head_dim * kv_bytes
            kv_cache_gb = (bytes_per_token * total_tokens) / (1024**3)
    else:
        # Realistic GQA KV cache approximation: ~56 KB per token for a 7B model
        scale = max(0.2, params_b / 7.0)
        bytes_per_token = 56_000 * scale * (kv_bytes / 2.0)
        kv_cache_gb = (bytes_per_token * total_tokens) / (1024**3)

    # 3. CUDA context runtime overhead (kernels, graph capture, buffers)
    cuda_overhead_gb = 1.2
    if params_b >= 70.0:
        cuda_overhead_gb = 2.5  # Higher buffer allocation for large graphs
    elif params_b >= 200.0:
        cuda_overhead_gb = 4.0

    # 4. Total and per-GPU requirements under Tensor Parallelism
    weights_per_gpu = weights_gb / tp
    kv_per_gpu = kv_cache_gb / tp
    vram_per_gpu_gb = weights_per_gpu + kv_per_gpu + cuda_overhead_gb
    total_required_gb = (vram_per_gpu_gb * tp)

    # Architectural classifications
    arch_type = "dense"
    is_moe = False
    active_params = None
    if spec:
        if spec.is_mla:
            arch_type = "deepseek-mla"
        elif spec.is_moe:
            arch_type = "moe"
        elif spec.is_vision:
            arch_type = "vlm"
        is_moe = spec.is_moe
        active_params = spec.active_params_b

    fits = True
    suggestion = ""
    if available_vram_per_gpu_gb is not None:
        fits = vram_per_gpu_gb <= (available_vram_per_gpu_gb * 0.95)
        if not fits:
            shortage = vram_per_gpu_gb - available_vram_per_gpu_gb
            recs: list[str] = []
            if bytes_per_param > 1.1:
                recs.append("apply FP8 or AWQ quantization")
            if tp == 1 and available_vram_per_gpu_gb < vram_per_gpu_gb:
                recs.append("increase tensor_parallel_size to 2 or 4")
            if context_length > 4096:
                recs.append(f"reduce max_model_len from {context_length} to 4096")
            recs.append("enable fp8 kv_cache_dtype")
            suggestion = (
                f"Exceeds GPU memory by {shortage:.1f} GB. Recommendations: {'; '.join(recs)}."
            )
        else:
            free_margin = available_vram_per_gpu_gb - vram_per_gpu_gb
            suggestion = f"Fits comfortably. Estimated free headroom: {free_margin:.1f} GB per GPU."

    return VRAMEstimate(
        model_name=model_name_or_path,
        params_billions=round(params_b, 2),
        precision_bytes_per_param=bytes_per_param,
        weights_vram_gb=round(weights_gb, 2),
        kv_cache_vram_gb=round(kv_cache_gb, 2),
        cuda_overhead_gb=round(cuda_overhead_gb, 2),
        total_required_vram_gb=round(total_required_gb, 2),
        vram_per_gpu_gb=round(vram_per_gpu_gb, 2),
        tensor_parallel_size=tp,
        context_length=context_length,
        fits=fits,
        is_moe=is_moe,
        active_params_b=active_params,
        architecture_type=arch_type,
        available_vram_per_gpu_gb=available_vram_per_gpu_gb,
        suggestion=suggestion,
    )


VRAMEstimationResult = VRAMEstimate


class VRAMCalculator:
    """Helper class providing calculation methods on ModelConfig instances."""

    @staticmethod
    def calculate(config: Any, available_vram_gb: Optional[float] = None) -> VRAMEstimate:
        return calculate_vram_requirements(
            model_name_or_path=getattr(config, "model", str(config)),
            context_length=getattr(config, "max_model_len", 8192) or 8192,
            dtype=getattr(config, "dtype", "auto"),
            quantization=getattr(config, "quantization", None),
            tensor_parallel_size=getattr(config, "tensor_parallel_size", 1),
            available_vram_per_gpu_gb=available_vram_gb,
            gpu_layers=getattr(config, "gpu_layers", None),
        )
