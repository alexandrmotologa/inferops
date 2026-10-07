"""Unit tests for inference engine adapters."""

from inferops.core.config import EngineType, ModelConfig
from inferops.engines import get_engine_adapter
from inferops.engines.sglang import SglangEngineAdapter
from inferops.engines.vllm import VllmEngineAdapter


def test_get_engine_adapter() -> None:
    vllm_ad = get_engine_adapter(EngineType.VLLM)
    assert isinstance(vllm_ad, VllmEngineAdapter)

    sglang_ad = get_engine_adapter(EngineType.SGLANG)
    assert isinstance(sglang_ad, SglangEngineAdapter)


def test_vllm_command_generation() -> None:
    cfg = ModelConfig(
        name="test-qwen",
        model="Qwen/Qwen2.5-7B",
        port=8001,
        tensor_parallel_size=2,
        max_model_len=16384,
        quantization="fp8",
        extra_args=["--enforce-eager"],
    )
    adapter = VllmEngineAdapter()
    cmd = adapter.build_command(cfg)

    assert "Qwen/Qwen2.5-7B" in cmd
    assert "--port" in cmd
    assert "8001" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "2" in cmd
    assert "--max-model-len" in cmd
    assert "16384" in cmd
    assert "--quantization" in cmd
    assert "fp8" in cmd
    assert "--enforce-eager" in cmd


def test_sglang_command_generation() -> None:
    cfg = ModelConfig(
        name="test-sglang",
        model="meta-llama/Llama-3.1-8B",
        engine=EngineType.SGLANG,
        port=8002,
        tensor_parallel_size=1,
        max_model_len=8192,
    )
    adapter = SglangEngineAdapter()
    cmd = adapter.build_command(cfg)

    assert "sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert "meta-llama/Llama-3.1-8B" in cmd
    assert "--port" in cmd
    assert "8002" in cmd
    assert "--context-length" in cmd
    assert "8192" in cmd
