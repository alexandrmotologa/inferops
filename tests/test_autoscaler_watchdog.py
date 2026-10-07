"""Tests for WorkloadAutoscaler and ProcessWatchdog crash recovery."""

import pytest

from inferops.core.config import EngineType, ModelConfig
from inferops.core.supervisor import (
    ModelLifecycleStatus,
    ProcessRecord,
    ProcessSupervisor,
    ProcessWatchdog,
)
from inferops.gateway.autoscaler import AutoscalingPolicy, ScalingAction, WorkloadAutoscaler


def test_workload_autoscaler_signals():
    """Verify autoscaler metrics collection and scaling actions."""
    policy = AutoscalingPolicy(
        target_concurrency_per_replica=10,
        max_concurrency_per_replica=20,
        latency_threshold_ms=200.0,
        scale_down_idle_seconds=60.0,
    )
    scaler = WorkloadAutoscaler(policy=policy)

    # 1. Start requests
    scaler.record_request_start("model-a")
    scaler.record_request_start("model-a")
    metrics = scaler.get_or_create_metrics("model-a")
    assert metrics.active_requests == 2

    # Normal load -> OPTIMAL
    assert scaler.evaluate_model("model-a", current_replicas=1) == ScalingAction.OPTIMAL

    # 2. Overload concurrency -> SCALE_UP
    for _ in range(25):
        scaler.record_request_start("model-a")
    assert metrics.active_requests == 27
    assert scaler.evaluate_model("model-a", current_replicas=1) == ScalingAction.SCALE_UP

    # 3. Complete requests with low latency
    for _ in range(27):
        scaler.record_request_end("model-a", latency_seconds=0.05)
    assert metrics.active_requests == 0
    assert metrics.total_completed_requests == 27
    assert metrics.avg_latency_ms == 50.0


@pytest.mark.asyncio
async def test_process_watchdog_auto_heal(tmp_path):
    """Verify watchdog detects dead PID and attempts recovery with backoff."""
    runtime_dir = tmp_path / "runtime"
    supervisor = ProcessSupervisor(runtime_dir)

    cfg = ModelConfig(
        name="test-healer",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        engine=EngineType.VLLM,
        port=8099,
    )
    catalog = {"test-healer": cfg}

    # Simulate a crashed process record with non-existent PID (e.g. 99999999)
    fake_record = ProcessRecord(
        name="test-healer",
        pid=99999999,
        status=ModelLifecycleStatus.HEALTHY,
        engine="vllm",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        host="127.0.0.1",
        port=8099,
        gpus="0",
        started_at=100.0,
        command=["python"],
        log_file=str(runtime_dir / "logs" / "test-healer.log"),
    )
    supervisor.save_record(fake_record)

    watchdog = ProcessWatchdog(
        supervisor=supervisor,
        models_catalog=catalog,
        max_restart_attempts=2,
        backoff_base_sec=1.1,
    )

    # Run check and heal
    events = await watchdog.check_and_heal()
    assert len(events) == 1
    assert events[0].model_name == "test-healer"
    assert watchdog.restart_counts["test-healer"] == 1

    # Kill newly spawned process for test cleanup
    rec = supervisor.get_record("test-healer")
    if rec and rec.pid:
        await supervisor.stop_model("test-healer")
