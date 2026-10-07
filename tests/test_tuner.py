"""Unit tests for the hardware tuner module."""

from inferops.core.config import EngineType
from inferops.core.tuner import tune_model_for_hardware
from inferops.hardware.gpu import GPUDeviceInfo


def test_tune_model_small_model_fits_unquantized() -> None:
    # 3B model on 24GB GPU
    rec = tune_model_for_hardware(
        model_identifier="Qwen/Qwen2.5-3B",
        target_engine=EngineType.VLLM,
        devices=[GPUDeviceInfo(0, "RTX 4090", 24.0, 24.0, 0.0, 0.0, 0.0)],
    )
    assert rec.suggested_name == "qwen2.5-3b"
    assert rec.tensor_parallel_size == 1
    assert rec.dtype == "bfloat16"
    assert rec.quantization is None
    assert rec.hardware_headroom_gb > 10.0


def test_tune_model_scales_tp_on_multigpu() -> None:
    # 70B model on 4x80GB A100 GPUs
    gpus_4x80 = [GPUDeviceInfo(i, f"A100-{i}", 80.0, 80.0, 0.0, 0.0, 0.0) for i in range(4)]
    rec = tune_model_for_hardware(
        model_identifier="meta-llama/Llama-3.1-70B-Instruct",
        target_engine=EngineType.VLLM,
        devices=gpus_4x80,
    )
    # Needs TP=2 or TP=4
    assert rec.tensor_parallel_size in (2, 4)
    assert len(rec.gpus) == rec.tensor_parallel_size
