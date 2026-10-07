"""Unit tests for the process supervisor and record management."""

import time
from pathlib import Path

import pytest

from inferops.core.config import ModelConfig
from inferops.core.supervisor import (
    ModelLifecycleStatus,
    ProcessRecord,
    ProcessSupervisor,
    is_process_running,
)


def test_is_process_running_invalid_pid() -> None:
    assert is_process_running(-1) is False
    assert is_process_running(0) is False
    # Large nonexistent PID
    assert is_process_running(99999999) is False


def test_supervisor_record_crud(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(tmp_path)
    rec = ProcessRecord(
        name="test-m",
        pid=1234,
        status=ModelLifecycleStatus.STARTING,
        engine="vllm",
        model="org/test",
        host="0.0.0.0",
        port=8001,
        gpus="0",
        started_at=time.time(),
        command=["vllm", "serve"],
        log_file=str(tmp_path / "test-m.log"),
    )

    supervisor.save_record(rec)
    loaded = supervisor.get_record("test-m")
    assert loaded is not None
    assert loaded.name == "test-m"
    assert loaded.pid == 1234
    assert loaded.status == ModelLifecycleStatus.STARTING

    supervisor.remove_record("test-m")
    assert supervisor.get_record("test-m") is None


@pytest.mark.asyncio
async def test_supervisor_status_stopped_when_no_record(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(tmp_path)
    cfg = ModelConfig(name="non-existent", model="foo/bar", port=8001)
    status = await supervisor.get_status(cfg)
    assert status == ModelLifecycleStatus.STOPPED
