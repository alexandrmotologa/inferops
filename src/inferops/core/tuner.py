"""Intelligent auto-tuning configuration optimizer for LLM serving."""

import re
from dataclasses import dataclass
from typing import List, Optional

from inferops.core.config import EngineType, ModelConfig
from inferops.core.vram_calculator import calculate_vram_requirements
from inferops.hardware.gpu import GPUDeviceInfo, get_gpu_devices


@dataclass
class TuningRecommendation:
    model_identifier: str
    suggested_name: str
    suggested_engine: EngineType
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    dtype: str
    quantization: Optional[str]
    kv_cache_dtype: str
    gpus: List[int]
    estimated_vram_per_gpu_gb: float
    hardware_headroom_gb: float
    rationale: List[str]

    def to_model_config(self, port: int = 8001) -> ModelConfig:
        return ModelConfig(
            name=self.suggested_name,
            model=self.model_identifier,
            engine=self.suggested_engine,
            port=port,
            gpus=self.gpus,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            quantization=self.quantization,
            kv_cache_dtype=self.kv_cache_dtype,
        )


def tune_model_for_hardware(
    model_identifier: str,
    target_engine: EngineType = EngineType.VLLM,
    devices: Optional[List[GPUDeviceInfo]] = None,
) -> TuningRecommendation:
    """Analyze detected GPUs and compute an optimal, failure-resistant ModelConfig."""
    gpus = devices if devices is not None else get_gpu_devices()
    num_gpus = len(gpus) if gpus else 1
    vram_per_gpu = gpus[0].total_memory_gb if gpus else 24.0

    # Clean slug name from model repo
    clean_name = model_identifier.split("/")[-1].lower()
    clean_name = re.sub(r"[^a-z0-9._-]", "-", clean_name).strip("-")

    rationale: List[str] = []

    # Priority 1: Unquantized BF16 / FP16 on 1 GPU
    est_bf16_tp1 = calculate_vram_requirements(
        model_name_or_path=model_identifier,
        context_length=8192,
        dtype="bfloat16",
        tensor_parallel_size=1,
        available_vram_per_gpu_gb=vram_per_gpu,
    )

    if est_bf16_tp1.fits:
        rationale.append("Model weights and 8K KV cache fit comfortably on a single GPU in 16-bit.")
        return TuningRecommendation(
            model_identifier=model_identifier,
            suggested_name=clean_name,
            suggested_engine=target_engine,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=16384,
            dtype="bfloat16",
            quantization=None,
            kv_cache_dtype="auto",
            gpus=[0],
            estimated_vram_per_gpu_gb=est_bf16_tp1.vram_per_gpu_gb,
            hardware_headroom_gb=round(vram_per_gpu - est_bf16_tp1.vram_per_gpu_gb, 1),
            rationale=rationale,
        )

    # Priority 2: Multi-GPU Tensor Parallelism in 16-bit
    if num_gpus > 1:
        for tp_candidate in [2, 4, 8]:
            if tp_candidate <= num_gpus:
                est_tp = calculate_vram_requirements(
                    model_name_or_path=model_identifier,
                    context_length=8192,
                    dtype="bfloat16",
                    tensor_parallel_size=tp_candidate,
                    available_vram_per_gpu_gb=vram_per_gpu,
                )
                if est_tp.fits:
                    rationale.append(f"Model scaled across {tp_candidate} GPUs via Tensor Parallelism ($TP={tp_candidate}$).")
                    return TuningRecommendation(
                        model_identifier=model_identifier,
                        suggested_name=clean_name,
                        suggested_engine=target_engine,
                        tensor_parallel_size=tp_candidate,
                        gpu_memory_utilization=0.90,
                        max_model_len=8192,
                        dtype="bfloat16",
                        quantization=None,
                        kv_cache_dtype="auto",
                        gpus=list(range(tp_candidate)),
                        estimated_vram_per_gpu_gb=est_tp.vram_per_gpu_gb,
                        hardware_headroom_gb=round(vram_per_gpu - est_tp.vram_per_gpu_gb, 1),
                        rationale=rationale,
                    )

    # Priority 3: FP8 Quantization
    est_fp8 = calculate_vram_requirements(
        model_name_or_path=model_identifier,
        context_length=8192,
        dtype="fp8",
        quantization="fp8",
        tensor_parallel_size=1,
        available_vram_per_gpu_gb=vram_per_gpu,
    )
    if est_fp8.fits:
        rationale.append("Applying FP8 quantization to weights and KV-cache keeps model on single GPU with minimal quality degradation.")
        return TuningRecommendation(
            model_identifier=model_identifier,
            suggested_name=clean_name,
            suggested_engine=target_engine,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.92,
            max_model_len=8192,
            dtype="fp8",
            quantization="fp8",
            kv_cache_dtype="fp8",
            gpus=[0],
            estimated_vram_per_gpu_gb=est_fp8.vram_per_gpu_gb,
            hardware_headroom_gb=round(vram_per_gpu - est_fp8.vram_per_gpu_gb, 1),
            rationale=rationale,
        )

    # Priority 4: AWQ 4-bit Quantization
    est_awq = calculate_vram_requirements(
        model_name_or_path=model_identifier,
        context_length=4096,
        dtype="auto",
        quantization="awq",
        tensor_parallel_size=1,
        available_vram_per_gpu_gb=vram_per_gpu,
    )
    rationale.append("Applying 4-bit AWQ weight quantization and limiting context to 4K tokens to fit available memory.")
    return TuningRecommendation(
        model_identifier=model_identifier,
        suggested_name=clean_name,
        suggested_engine=target_engine,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=4096,
        dtype="auto",
        quantization="awq",
        kv_cache_dtype="fp8",
        gpus=[0],
        estimated_vram_per_gpu_gb=est_awq.vram_per_gpu_gb,
        hardware_headroom_gb=round(max(0.0, vram_per_gpu - est_awq.vram_per_gpu_gb), 1),
        rationale=rationale,
    )
