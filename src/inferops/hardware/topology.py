"""NVIDIA GPU Interconnect Topology discovery and Tensor Parallelism readiness assessment."""

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GpuInterconnectLink:
    gpu_a: int
    gpu_b: int
    link_type: str
    is_nvlink: bool
    is_high_speed: bool
    description: str


@dataclass
class TopologyReport:
    available: bool
    num_gpus: int
    raw_matrix: Optional[str]
    links: List[GpuInterconnectLink]
    tensor_parallel_recommended: bool
    recommended_tp_limit: int
    recommendations: List[str]


def parse_topology_matrix(output: str) -> List[GpuInterconnectLink]:
    """Parse output of `nvidia-smi topo -m` into structured links."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return []

    # Find matrix header line
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("GPU0") or "\tGPU0" in line or " GPU0" in line:
            header_idx = i
            break

    if header_idx == -1:
        return []

    header_tokens = lines[header_idx].split()
    gpu_cols = [t for t in header_tokens if t.startswith("GPU") and t[3:].isdigit()]
    num_gpus = len(gpu_cols)

    links: List[GpuInterconnectLink] = []

    for line in lines[header_idx + 1 :]:
        if not line.startswith("GPU"):
            continue
        tokens = line.split()
        if not tokens:
            continue
        row_gpu_str = tokens[0]
        try:
            row_gpu = int(row_gpu_str.replace("GPU", ""))
        except ValueError:
            continue

        for col_idx in range(min(num_gpus, len(tokens) - 1)):
            col_gpu = col_idx
            if row_gpu >= col_gpu:
                continue  # Avoid duplicate or self pairs

            link_code = tokens[col_idx + 1]
            is_nv = link_code.startswith("NV")
            is_high_speed = is_nv or link_code in ["PIX", "PXB"]

            if is_nv:
                desc = f"NVLink interconnect ({link_code})"
            elif link_code == "PIX":
                desc = "Single PCIe bridge (Direct switch)"
            elif link_code == "PXB":
                desc = "Multiple PCIe bridges"
            elif link_code == "PHB":
                desc = "PCIe Host Bridge traversal"
            elif link_code == "NODE":
                desc = "NUMA node boundary traversal"
            elif link_code == "SYS":
                desc = "System Bus / CPU socket traversal (Slow)"
            else:
                desc = f"Link connection: {link_code}"

            links.append(
                GpuInterconnectLink(
                    gpu_a=row_gpu,
                    gpu_b=col_gpu,
                    link_type=link_code,
                    is_nvlink=is_nv,
                    is_high_speed=is_high_speed,
                    description=desc,
                )
            )

    return links


def inspect_gpu_topology() -> TopologyReport:
    """Run `nvidia-smi topo -m` and produce a diagnostic report for Tensor Parallelism."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return TopologyReport(
            available=False,
            num_gpus=0,
            raw_matrix=None,
            links=[],
            tensor_parallel_recommended=False,
            recommended_tp_limit=1,
            recommendations=["nvidia-smi not found. Running in single-GPU or emulation mode."],
        )

    try:
        res = subprocess.run(
            [nvidia_smi, "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if res.returncode != 0:
            return TopologyReport(
                available=False,
                num_gpus=0,
                raw_matrix=None,
                links=[],
                tensor_parallel_recommended=False,
                recommended_tp_limit=1,
                recommendations=["Failed to retrieve GPU topology from nvidia-smi."],
            )

        raw_output = res.stdout
        links = parse_topology_matrix(raw_output)

        gpus_seen = set()
        for link in links:
            gpus_seen.add(link.gpu_a)
            gpus_seen.add(link.gpu_b)
        num_gpus = len(gpus_seen) if gpus_seen else (1 if "GPU0" in raw_output else 0)

        recommendations = []
        has_nvlink = any(link_item.is_nvlink for link_item in links)
        has_sys_bottleneck = any(link_item.link_type in ["SYS", "NODE"] for link_item in links)

        if num_gpus <= 1:
            rec_tp = 1
            tp_ok = True
            recommendations.append("Single GPU system detected. Tensor Parallelism is limited to TP=1.")
        elif has_nvlink:
            rec_tp = num_gpus
            tp_ok = True
            recommendations.append(f"High-speed NVLink detected across GPUs. Optimal for Tensor Parallelism up to TP={num_gpus}.")
        elif has_sys_bottleneck:
            rec_tp = 1
            tp_ok = False
            recommendations.append(
                "Inter-GPU communication traverses CPU/System Bus (SYS/NODE). High all-reduce latency will degrade Tensor Parallelism. Prefer Pipeline Parallelism (PP) or independent replicas."
            )
        else:
            rec_tp = min(2, num_gpus)
            tp_ok = True
            recommendations.append(
                f"PCIe interconnect without NVLink. Tensor Parallelism acceptable up to TP={rec_tp}, but may exhibit slight latency scaling limits."
            )

        return TopologyReport(
            available=True,
            num_gpus=num_gpus,
            raw_matrix=raw_output,
            links=links,
            tensor_parallel_recommended=tp_ok,
            recommended_tp_limit=rec_tp,
            recommendations=recommendations,
        )

    except Exception as e:
        return TopologyReport(
            available=False,
            num_gpus=0,
            raw_matrix=None,
            links=[],
            tensor_parallel_recommended=False,
            recommended_tp_limit=1,
            recommendations=[f"Topology error: {e}"],
        )
