"""Tests for advanced model architectures (DeepSeek MLA, MoE, Phi-4, VLMs)."""


from inferops.core.vram_calculator import (
    calculate_vram_requirements,
    infer_params_from_name,
)


def test_deepseek_v3_mla_and_moe():
    """Verify DeepSeek-V3 MoE sizing and MLA compressed KV-cache calculations."""
    params_b, spec = infer_params_from_name("deepseek-ai/DeepSeek-V3")
    assert params_b == 671.0
    assert spec is not None
    assert spec.is_moe is True
    assert spec.active_params_b == 37.0
    assert spec.is_mla is True
    assert spec.mla_kv_dim == 576

    # Test with FP8 quantization and TP=8
    est = calculate_vram_requirements(
        "deepseek-ai/DeepSeek-V3",
        context_length=8192,
        quantization="fp8",
        tensor_parallel_size=8,
        available_vram_per_gpu_gb=96.0,
    )

    assert est.is_moe is True
    assert est.active_params_b == 37.0
    assert est.architecture_type == "deepseek-mla"
    assert est.tensor_parallel_size == 8
    # MLA KV cache must be under 15 GB for 16 requests of 8K context (vs >400 GB for standard MHA)
    assert est.kv_cache_vram_gb < 15.0
    assert est.vram_per_gpu_gb < 90.0
    assert est.fits is True


def test_deepseek_r1_reasoning():
    """Verify DeepSeek-R1 reasoning model detection."""
    params_b, spec = infer_params_from_name("deepseek-ai/DeepSeek-R1")
    assert params_b == 671.0
    assert spec is not None
    assert spec.is_moe is True


def test_mixtral_moe_sizing():
    """Verify Mixtral 8x7B and 8x22B MoE weight calculations."""
    est_8x7b = calculate_vram_requirements("mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=2)
    assert est_8x7b.is_moe is True
    assert est_8x7b.params_billions == 46.7
    assert est_8x7b.active_params_b == 12.9

    est_8x22b = calculate_vram_requirements("mistralai/Mixtral-8x22B-Instruct-v0.1", tensor_parallel_size=4, quantization="awq")
    assert est_8x22b.is_moe is True
    assert est_8x22b.params_billions == 141.0
    assert est_8x22b.weights_vram_gb > 70.0


def test_phi4_and_gemma2():
    """Verify Microsoft Phi-4 and Gemma-2 detection."""
    params_phi4, spec_phi4 = infer_params_from_name("microsoft/phi-4")
    assert params_phi4 == 14.7
    assert spec_phi4 is not None

    params_gemma, spec_gemma = infer_params_from_name("google/gemma-2-9b-it")
    assert params_gemma == 9.24
    assert spec_gemma is not None


def test_vision_language_model():
    """Verify multimodal vision model detection and overhead."""
    est_vlm = calculate_vram_requirements("Qwen/Qwen2-VL-7B-Instruct")
    assert est_vlm.architecture_type == "vlm"
    # Vision encoder adds overhead beyond base weights
    assert est_vlm.weights_vram_gb > 14.0
