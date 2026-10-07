"""Unit tests for GPU hardware queries and Prometheus metrics parsing."""

import time

from inferops.hardware.gpu import GPUDeviceInfo, get_gpu_devices
from inferops.hardware.metrics import (
    MetricsRingBuffer,
    ModelLiveMetrics,
    parse_prometheus_text,
)


def test_parse_prometheus_text() -> None:
    sample = """
    # HELP vllm:num_requests_running Number of requests currently running on GPU.
    # TYPE vllm:num_requests_running gauge
    vllm:num_requests_running{model_name="qwen"} 4.0
    vllm:gpu_cache_usage_factor 0.425
    vllm:generation_tokens_total 12500
    """
    metrics = parse_prometheus_text(sample)
    assert metrics["vllm:num_requests_running"] == 4.0
    assert metrics["vllm:gpu_cache_usage_factor"] == 0.425
    assert metrics["vllm:generation_tokens_total"] == 12500.0


def test_metrics_ring_buffer_throughput() -> None:
    buf = MetricsRingBuffer(max_points=5)

    now = time.time()
    p1 = ModelLiveMetrics(timestamp=now, generation_tokens_total=1000)
    buf.add(p1)
    assert buf.latest() is not None
    assert buf.latest().throughput_tokens_per_sec == 0.0

    p2 = ModelLiveMetrics(timestamp=now + 2.0, generation_tokens_total=1100)
    buf.add(p2)
    assert buf.latest() is not None
    # 100 tokens generated in 2 seconds = 50.0 tok/sec
    assert buf.latest().throughput_tokens_per_sec == 50.0
    assert len(buf.series()) == 2


def test_get_gpu_devices_returns_list() -> None:
    # Safe on any machine (whether NVIDIA GPU is present or not)
    devices = get_gpu_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert isinstance(d, GPUDeviceInfo)
        assert d.total_memory_gb >= 0
