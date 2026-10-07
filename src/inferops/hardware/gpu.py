"""Hardware detection and GPU telemetry (NVIDIA NVML & ROCm)."""

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GPUDeviceInfo:
    """Telemetry and identity data for a detected GPU."""
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_gpu_pct: float
    utilization_mem_pct: float
    temperature_c: Optional[int] = None
    power_watts: Optional[float] = None


def get_gpu_devices() -> List[GPUDeviceInfo]:
    """Scan and retrieve all available physical GPUs, falling back gracefully if none."""
    # Attempt 1: nvidia-smi query
    devices = _query_nvidia_smi()
    if devices:
        return devices

    # Attempt 2: PyNVML if installed
    devices = _query_pynvml()
    if devices:
        return devices

    # If no GPU detected, return empty list (or mock in development)
    return []


def _query_nvidia_smi() -> List[GPUDeviceInfo]:
    """Query nvidia-smi with standard CSV format."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []

    query_fields = [
        "index",
        "name",
        "memory.total",
        "memory.free",
        "memory.used",
        "utilization.gpu",
        "utilization.memory",
        "temperature.gpu",
        "power.draw",
    ]
    cmd = [
        smi,
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=True)
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        devices: List[GPUDeviceInfo] = []

        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                idx = int(parts[0])
                name = parts[1]
                total_mb = float(parts[2])
                free_mb = float(parts[3])
                used_mb = float(parts[4])
                util_gpu = float(parts[5])
                util_mem = float(parts[6])
                temp = int(parts[7]) if len(parts) > 7 and parts[7].isdigit() else None
                power = None
                if len(parts) > 8:
                    try:
                        power = float(parts[8])
                    except ValueError:
                        pass

                devices.append(
                    GPUDeviceInfo(
                        index=idx,
                        name=name,
                        total_memory_gb=round(total_mb / 1024, 2),
                        free_memory_gb=round(free_mb / 1024, 2),
                        used_memory_gb=round(used_mb / 1024, 2),
                        utilization_gpu_pct=util_gpu,
                        utilization_mem_pct=util_mem,
                        temperature_c=temp,
                        power_watts=power,
                    )
                )
        return devices
    except Exception:
        return []


def _query_pynvml() -> List[GPUDeviceInfo]:
    """Attempt reading via pynvml library if available."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        devices: List[GPUDeviceInfo] = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

            devices.append(
                GPUDeviceInfo(
                    index=i,
                    name=name,
                    total_memory_gb=round(mem.total / (1024**3), 2),
                    free_memory_gb=round(mem.free / (1024**3), 2),
                    used_memory_gb=round(mem.used / (1024**3), 2),
                    utilization_gpu_pct=float(util.gpu),
                    utilization_mem_pct=float(util.memory),
                    temperature_c=temp,
                )
            )
        pynvml.nvmlShutdown()
        return devices
    except Exception:
        return []
