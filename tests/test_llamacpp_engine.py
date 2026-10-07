"""Tests for llama.cpp engine adapter and GGUF quantization calculations."""

from inferops.core.config import EngineType, ModelConfig
from inferops.core.vram_calculator import calculate_vram_requirements
from inferops.engines import get_engine_adapter
from inferops.engines.llamacpp import LlamaCppEngineAdapter


def test_llamacpp_adapter_registration():
    """Verify llama.cpp adapter is registered and retrieved properly."""
    adapter = get_engine_adapter(EngineType.LLAMACPP)
    assert isinstance(adapter, LlamaCppEngineAdapter)
    assert adapter.engine_name == "llamacpp"
    assert adapter.get_executable() == "llama-server"


def test_llamacpp_command_builder():
    """Verify command building for llama.cpp with gpu_layers and thread settings."""
    cfg = ModelConfig(
        name="llama3-gguf",
        model="/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        engine=EngineType.LLAMACPP,
        port=8003,
        gpu_layers=33,
        threads=8,
        max_model_len=4096,
    )
    adapter = LlamaCppEngineAdapter()
    cmd = adapter.build_command(cfg)

    assert "-m" in cmd
    assert "/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" in cmd
    assert "--port" in cmd
    assert "8003" in cmd
    assert "-c" in cmd
    assert "4096" in cmd
    assert "-ngl" in cmd
    assert "33" in cmd
    assert "-t" in cmd
    assert "8" in cmd


def test_gguf_quantization_and_layer_offloading():
    """Verify memory calculation for GGUF Q4_K_M and partial GPU offload."""
    # Full offload of 8B model with Q4_K_M (~0.55 bytes/param -> ~4.1 GB weights)
    est_full = calculate_vram_requirements("llama-3-8b", quantization="q4_k_m")
    assert est_full.weights_vram_gb < 4.5

    # Partial offload (16 out of 32 layers = 50% offload)
    est_half = calculate_vram_requirements("llama-3-8b", quantization="q4_k_m", gpu_layers=16)
    assert est_half.weights_vram_gb < (est_full.weights_vram_gb * 0.6)
    assert est_half.weights_vram_gb > (est_full.weights_vram_gb * 0.4)
