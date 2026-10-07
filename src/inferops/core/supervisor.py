"""Cross-platform asynchronous process supervisor for inference engines."""

import asyncio
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from inferops.core.config import ModelConfig
from inferops.core.exceptions import PortConflictError, ProcessCrashedError, ProcessStartupTimeoutError
from inferops.core.health import check_http_health, wait_for_health
from inferops.engines import get_engine_adapter


class ModelLifecycleStatus(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"
    CRASHED = "CRASHED"


@dataclass
class ProcessRecord:
    name: str
    pid: int
    status: ModelLifecycleStatus
    engine: str
    model: str
    host: str
    port: int
    gpus: str
    started_at: float
    command: List[str]
    log_file: str


def is_process_running(pid: int) -> bool:
    """Safely check if a process with given PID exists across all platforms."""
    if pid <= 0:
        return False

    if sys.platform == "win32":
        # Windows API check using kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32  # type: ignore
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        still_active = 259
        try:
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return exit_code.value == still_active
            return False
        finally:
            kernel32.CloseHandle(handle)
    else:
        # POSIX check
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def terminate_pid(pid: int, timeout_sec: float = 8.0) -> bool:
    """Gracefully terminate a process, escalating to force-kill if it doesn't exit."""
    if not is_process_running(pid):
        return True

    if sys.platform == "win32":
        # On Windows, try taskkill with tree (/T) to catch child python processes
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    else:
        # POSIX graceful SIGTERM -> SIGKILL
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return True

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if not is_process_running(pid):
                return True
            time.sleep(0.2)

        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass

    return not is_process_running(pid)


class ProcessSupervisor:
    """Manages spawning, monitoring, and stopping engine worker processes."""

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = runtime_dir
        self.pids_dir = runtime_dir / "pids"
        self.logs_dir = runtime_dir / "logs"
        self.pids_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _pid_file_for(self, model_name: str) -> Path:
        return self.pids_dir / f"{model_name}.json"

    def _log_file_for(self, model_name: str) -> Path:
        return self.logs_dir / f"{model_name}.log"

    def get_record(self, model_name: str) -> Optional[ProcessRecord]:
        """Read process metadata record from disk."""
        pid_file = self._pid_file_for(model_name)
        if not pid_file.is_file():
            return None
        try:
            data = json.loads(pid_file.read_text(encoding="utf-8"))
            data["status"] = ModelLifecycleStatus(data["status"])
            return ProcessRecord(**data)
        except Exception:
            return None

    def save_record(self, record: ProcessRecord) -> None:
        """Persist process record to disk."""
        pid_file = self._pid_file_for(record.name)
        dump = asdict(record)
        dump["status"] = record.status.value
        pid_file.write_text(json.dumps(dump, indent=2), encoding="utf-8")

    def remove_record(self, model_name: str) -> None:
        """Remove state record when process stops."""
        pid_file = self._pid_file_for(model_name)
        if pid_file.is_file():
            pid_file.unlink(missing_ok=True)

    async def get_status(self, config: ModelConfig) -> ModelLifecycleStatus:
        """Inspect true live status of a model (process alive + HTTP probe)."""
        record = self.get_record(config.name)
        if not record:
            return ModelLifecycleStatus.STOPPED

        if not is_process_running(record.pid):
            self.remove_record(config.name)
            return ModelLifecycleStatus.CRASHED

        # Probe health endpoint
        is_healthy = await check_http_health(config.host, config.port, config.health_endpoint)
        if is_healthy:
            if record.status != ModelLifecycleStatus.HEALTHY:
                record.status = ModelLifecycleStatus.HEALTHY
                self.save_record(record)
            return ModelLifecycleStatus.HEALTHY

        return ModelLifecycleStatus.STARTING

    async def start_model(
        self,
        config: ModelConfig,
        wait: bool = True,
        timeout_seconds: float = 180.0,
    ) -> ProcessRecord:
        """Launch an inference engine process and optionally wait for /health."""
        current_status = await self.get_status(config)
        if current_status in (ModelLifecycleStatus.HEALTHY, ModelLifecycleStatus.STARTING):
            rec = self.get_record(config.name)
            if rec:
                return rec

        # Check port availability
        if await check_http_health(config.host, config.port, timeout=0.5):
            raise PortConflictError(config.port, config.name)

        # Prepare log files and rotation
        log_path = self._log_file_for(config.name)
        prev_log = self.logs_dir / f"{config.name}.log.prev"
        if log_path.is_file():
            if prev_log.is_file():
                prev_log.unlink(missing_ok=True)
            log_path.rename(prev_log)

        # Build command and env via adapter
        adapter = get_engine_adapter(config.engine)
        cmd = adapter.build_command(config)
        env = adapter.build_environment(config)

        # Open log file for output redirection
        log_file_handle = open(log_path, "w", encoding="utf-8")

        # Spawn background process
        try:
            # Platform-specific process detachment
            creationflags = 0
            if sys.platform == "win32":
                # CREATE_NEW_PROCESS_GROUP
                creationflags = 0x00000200

            process = subprocess.Popen(
                cmd,
                stdout=log_file_handle,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=creationflags,
                close_fds=(sys.platform != "win32"),
            )
        except Exception as e:
            log_file_handle.close()
            raise RuntimeError(f"Failed to spawn {config.engine} for '{config.name}': {e}") from e

        record = ProcessRecord(
            name=config.name,
            pid=process.pid,
            status=ModelLifecycleStatus.STARTING,
            engine=config.engine.value,
            model=config.model,
            host=config.host,
            port=config.port,
            gpus=config.cuda_visible_devices,
            started_at=time.time(),
            command=cmd,
            log_file=str(log_path),
        )
        self.save_record(record)

        if not wait:
            return record

        # Wait for health probe
        healthy = await wait_for_health(
            host=config.host,
            port=config.port,
            endpoint=config.health_endpoint,
            timeout_seconds=timeout_seconds,
            poll_interval=1.5,
            is_alive_check=lambda: is_process_running(process.pid),
        )

        if not healthy:
            if not is_process_running(process.pid):
                self.remove_record(config.name)
                raise ProcessCrashedError(config.name, process.poll(), str(log_path))
            terminate_pid(process.pid)
            self.remove_record(config.name)
            raise ProcessStartupTimeoutError(config.name, timeout_seconds)

        record.status = ModelLifecycleStatus.HEALTHY
        self.save_record(record)
        return record

    async def stop_model(self, model_name: str, timeout_sec: float = 8.0) -> bool:
        """Stop a running model process gracefully."""
        record = self.get_record(model_name)
        if not record:
            return True

        stopped = terminate_pid(record.pid, timeout_sec=timeout_sec)
        self.remove_record(model_name)
        return stopped
