"""Tests for multi-hardware detection, ROCm environment variables, and speculative decoding flags."""


from inferops.core.config import EngineType, ModelConfig
from inferops.engines.sglang import SglangEngineAdapter
from inferops.engines.vllm import VllmEngineAdapter
from inferops.hardware.gpu import get_hardware_summary, get_host_memory_gb


def test_host_memory_detection():
    """Verify standard-library host RAM inspection returns positive numbers."""
    total_gb, free_gb = get_host_memory_gb()
    assert total_gb > 0.0
    assert free_gb >= 0.0
    assert free_gb <= total_gb


def test_hardware_summary():
    """Verify hardware summary dictionary structure."""
    summary = get_hardware_summary()
    assert "backend" in summary
    assert "host_ram_total_gb" in summary
    assert "cpu_cores" in summary
    assert summary["cpu_cores"] >= 1
    assert summary["host_ram_total_gb"] > 0.0


def test_vllm_engine_trust_remote_code_and_speculative():
    """Verify vLLM command building with trust_remote_code, device, and speculative decoding."""
    cfg = ModelConfig(
        name="llama-speculative",
        model="meta-llama/Llama-3.1-70B-Instruct",
        engine=EngineType.VLLM,
        trust_remote_code=True,
        speculative_model="meta-llama/Llama-3.2-1B-Instruct",
        num_speculative_tokens=4,
        device="cuda",
    )
    adapter = VllmEngineAdapter()
    cmd = adapter.build_command(cfg)

    assert "--trust-remote-code" in cmd
    assert "--speculative-model" in cmd
    assert "meta-llama/Llama-3.2-1B-Instruct" in cmd
    assert "--num-speculative-tokens" in cmd
    assert "4" in cmd


def test_sglang_engine_trust_remote_code_and_speculative():
    """Verify SGLang command building with trust_remote_code and speculative draft model."""
    cfg = ModelConfig(
        name="qwen-speculative",
        model="Qwen/Qwen2.5-72B-Instruct",
        engine=EngineType.SGLANG,
        trust_remote_code=True,
        speculative_model="Qwen/Qwen2.5-0.5B-Instruct",
        num_speculative_tokens=3,
    )
    adapter = SglangEngineAdapter()
    cmd = adapter.build_command(cfg)

    assert "--trust-remote-code" in cmd
    assert "--speculative-draft-model" in cmd
    assert "Qwen/Qwen2.5-0.5B-Instruct" in cmd
    assert "--speculative-num-steps" in cmd
    assert "3" in cmd


def test_rocm_and_cpu_environment_generation():
    """Verify ROCm HIP variables and CPU target device environment injection."""
    # Test 1: GPU configuration with ROCm propagation
    gpu_cfg = ModelConfig(
        name="gpu-test",
        model="Qwen/Qwen2.5-7B-Instruct",
        gpus=[0, 1],
    )
    adapter = VllmEngineAdapter()
    env = adapter.build_environment(gpu_cfg)

    assert env.get("CUDA_VISIBLE_DEVICES") == "0,1"
    assert env.get("HIP_VISIBLE_DEVICES") == "0,1"
    assert env.get("ROCR_VISIBLE_DEVICES") == "0,1"
    assert env.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"

    # Test 2: CPU device configuration
    cpu_cfg = ModelConfig(
        name="cpu-test",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    cpu_env = adapter.build_environment(cpu_cfg)
    assert cpu_env.get("VLLM_TARGET_DEVICE") == "cpu"
    assert cpu_env.get("CUDA_VISIBLE_DEVICES") == ""
    assert cpu_env.get("HIP_VISIBLE_DEVICES") == ""
