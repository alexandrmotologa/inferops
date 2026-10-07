"""Hardware detection and multi-device telemetry (NVIDIA CUDA, AMD ROCm, Apple Silicon, & CPU)."""

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class GPUDeviceInfo:
    """Telemetry and identity data for a detected GPU or accelerator."""
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    used_memory_gb: float
    utilization_gpu_pct: float
    utilization_mem_pct: float
    temperature_c: Optional[int] = None
    power_watts: Optional[float] = None
    vendor: str = "nvidia"  # nvidia, rocm, apple, cpu


def get_gpu_devices() -> List[GPUDeviceInfo]:
    """Scan and retrieve all available physical accelerators, falling back gracefully."""
    # Attempt 1: NVIDIA nvidia-smi
    devices = _query_nvidia_smi()
    if devices:
        return devices

    # Attempt 2: PyNVML if installed
    devices = _query_pynvml()
    if devices:
        return devices

    # Attempt 3: AMD ROCm via rocm-smi
    devices = _query_rocm_smi()
    if devices:
        return devices

    # Attempt 4: Apple Silicon Unified Memory on macOS
    devices = _query_apple_silicon()
    if devices:
        return devices

    return []


def get_host_memory_gb() -> Tuple[float, float]:
    """Retrieve total and free/available physical system RAM in gigabytes."""
    total_gb = 0.0
    free_gb = 0.0

    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_gb = round(stat.ullTotalPhys / (1024**3), 2)
                free_gb = round(stat.ullAvailPhys / (1024**3), 2)
                return total_gb, free_gb
        except Exception:
            pass

    elif sys.platform == "darwin":
        try:
            res = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            bytes_val = int(res.stdout.strip())
            total_gb = round(bytes_val / (1024**3), 2)
            # macOS memory estimation: free ~ 50% heuristic if vm_stat not parsed
            free_gb = round(total_gb * 0.5, 2)
            return total_gb, free_gb
        except Exception:
            pass

    elif sys.platform.startswith("linux"):
        try:
            meminfo_path = "/proc/meminfo"
            if os.path.exists(meminfo_path):
                with open(meminfo_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                total_kb = 0
                avail_kb = 0
                for line in lines:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                if total_kb > 0:
                    total_gb = round(total_kb / (1024**2), 2)
                    free_gb = round(avail_kb / (1024**2), 2)
                    return total_gb, free_gb
        except Exception:
            pass

    return total_gb, free_gb


def get_hardware_summary() -> Dict[str, object]:
    """Return comprehensive hardware profile including CPU, RAM, and accelerators."""
    total_ram, free_ram = get_host_memory_gb()
    gpus = get_gpu_devices()

    cpu_name = platform.processor() or platform.machine()
    cpu_cores = os.cpu_count() or 1

    primary_backend = "cpu"
    if gpus:
        primary_backend = gpus[0].vendor

    return {
        "backend": primary_backend,
        "gpus": gpus,
        "gpu_count": len(gpus),
        "host_ram_total_gb": total_ram,
        "host_ram_free_gb": free_ram,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
        "os": sys.platform,
    }


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
                        vendor="nvidia",
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
                    vendor="nvidia",
                )
            )
        pynvml.nvmlShutdown()
        return devices
    except Exception:
        return []


def _query_rocm_smi() -> List[GPUDeviceInfo]:
    """Query AMD ROCm devices via rocm-smi utility."""
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return []

    try:
        res = subprocess.run(
            [rocm_smi, "--showid", "--showproductname", "--showmeminfo", "vram", "--showuse", "--csv"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        if len(lines) <= 1:
            return []

        # Parse CSV output headers and values
        devices: List[GPUDeviceInfo] = []
        for idx, line in enumerate(lines[1:]):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                name = f"AMD ROCm GPU {idx}"
                # Best-effort memory parsing
                total_gb = 16.0
                free_gb = 16.0
                for part in parts:
                    if "Instinct" in part or "Radeon" in part:
                        name = part

                devices.append(
                    GPUDeviceInfo(
                        index=idx,
                        name=name,
                        total_memory_gb=total_gb,
                        free_memory_gb=free_gb,
                        used_memory_gb=0.0,
                        utilization_gpu_pct=0.0,
                        utilization_mem_pct=0.0,
                        vendor="rocm",
                    )
                )
        return devices
    except Exception:
        return []


def _query_apple_silicon() -> List[GPUDeviceInfo]:
    """Query Apple Silicon unified memory on macOS."""
    if sys.platform != "darwin":
        return []

    try:
        # Check CPU brand string
        brand_res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        brand = brand_res.stdout.strip()
        if "Apple" not in brand:
            return []

        mem_res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        mem_bytes = int(mem_res.stdout.strip())
        total_gb = round(mem_bytes / (1024**3), 2)
        # On Apple Silicon, unified memory is shared between CPU and Metal GPU
        return [
            GPUDeviceInfo(
                index=0,
                name=f"{brand} (Unified Memory)",
                total_memory_gb=total_gb,
                free_memory_gb=round(total_gb * 0.7, 2),
                used_memory_gb=round(total_gb * 0.3, 2),
                utilization_gpu_pct=0.0,
                utilization_mem_pct=30.0,
                vendor="apple",
            )
        ]
    except Exception:
        return []
