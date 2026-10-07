"""Unit tests for the predictive VRAM sizing calculator."""

from inferops.core.vram_calculator import (
    calculate_vram_requirements,
    get_bytes_per_param,
    infer_params_from_name,
)


def test_infer_params_from_name() -> None:
    # Known exact architectures
    params, spec = infer_params_from_name("Qwen/Qwen2.5-Coder-7B-Instruct")
    assert round(params, 1) == 7.6
    assert spec is not None
    assert spec.num_layers == 28

    params_llama, spec_llama = infer_params_from_name("meta-llama/Llama-3.1-8B-Instruct")
    assert round(params_llama, 1) == 8.0
    assert spec_llama is not None

    # Regex heuristic fallback
    params_unknown, spec_unknown = infer_params_from_name("custom-org/my-finetuned-14b-model")
    assert params_unknown == 14.0
    assert spec_unknown is None


def test_get_bytes_per_param() -> None:
    assert get_bytes_per_param("bfloat16") == 2.0
    assert get_bytes_per_param("auto", quantization="awq") == 0.55
    assert get_bytes_per_param("auto", quantization="fp8") == 1.05


def test_calculate_vram_requirements() -> None:
    # 7B model in FP16 with 8K context
    est = calculate_vram_requirements(
        model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
        context_length=8192,
        dtype="bfloat16",
        tensor_parallel_size=1,
        available_vram_per_gpu_gb=24.0,
    )
    assert est.weights_vram_gb > 13.0
    assert est.weights_vram_gb < 16.0
    assert est.vram_per_gpu_gb < 24.0
    assert est.fits is True

    # 70B model with TP=1 on 24GB GPU should NOT fit
    est_70b = calculate_vram_requirements(
        model_name_or_path="meta-llama/Llama-3.1-70B-Instruct",
        context_length=8192,
        dtype="bfloat16",
        tensor_parallel_size=1,
        available_vram_per_gpu_gb=24.0,
    )
    assert est_70b.fits is False
    assert "Exceeds GPU memory" in est_70b.suggestion

    # 70B model with TP=4 on 4x80GB GPUs should fit
    est_70b_tp4 = calculate_vram_requirements(
        model_name_or_path="meta-llama/Llama-3.1-70B-Instruct",
        context_length=8192,
        dtype="bfloat16",
        tensor_parallel_size=4,
        available_vram_per_gpu_gb=80.0,
    )
    assert est_70b_tp4.fits is True
