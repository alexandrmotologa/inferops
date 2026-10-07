"""Docker Compose generator for multi-engine production inference deployments."""

from typing import List

import yaml

from inferops.core.config import EngineType, ModelConfig


def generate_docker_compose(
    models: List[ModelConfig],
    gateway_port: int = 8000,
    hf_cache_dir: str = "~/.cache/huggingface",
) -> str:
    """Generate a production-grade docker-compose.yml configuration with NVIDIA GPU passthrough."""
    services = {}

    for cfg in models:
        service_name = f"model-{cfg.name}"

        if cfg.engine == EngineType.VLLM:
            image = "vllm/vllm-openai:latest"
            entrypoint_cmd = [
                "--model",
                cfg.model,
                "--host",
                "0.0.0.0",
                "--port",
                str(cfg.port),
                "--tensor-parallel-size",
                str(cfg.tensor_parallel_size),
                "--gpu-memory-utilization",
                str(cfg.gpu_memory_utilization),
            ]
            if cfg.max_model_len:
                entrypoint_cmd.extend(["--max-model-len", str(cfg.max_model_len)])
            if cfg.quantization:
                entrypoint_cmd.extend(["--quantization", cfg.quantization])
            if cfg.dtype and cfg.dtype != "auto":
                entrypoint_cmd.extend(["--dtype", cfg.dtype])
        else:
            image = "lmsysorg/sglang:latest"
            entrypoint_cmd = [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                cfg.model,
                "--host",
                "0.0.0.0",
                "--port",
                str(cfg.port),
                "--tp",
                str(cfg.tensor_parallel_size),
                "--mem-fraction-static",
                str(cfg.gpu_memory_utilization),
            ]

        # Determine GPU device reservation
        gpu_reservation = {
            "driver": "nvidia",
            "capabilities": ["gpu"],
        }
        if isinstance(cfg.gpus, list) and cfg.gpus:
            gpu_reservation["device_ids"] = [str(g) for g in cfg.gpus]
        else:
            gpu_reservation["count"] = "all"

        service_def = {
            "image": image,
            "container_name": f"inferops-{cfg.name}",
            "restart": "unless-stopped",
            "ipc": "host",
            "ports": [f"{cfg.port}:{cfg.port}"],
            "volumes": [
                f"{hf_cache_dir}:/root/.cache/huggingface",
            ],
            "environment": {
                "HF_TOKEN": "${HF_TOKEN:-}",
                "NCCL_DEBUG": "WARN",
            },
            "command": entrypoint_cmd,
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [gpu_reservation],
                    }
                }
            },
            "healthcheck": {
                "test": ["CMD-SHELL", f"curl -f http://localhost:{cfg.port}/health || exit 1"],
                "interval": "15s",
                "timeout": "5s",
                "retries": 5,
                "start_period": "60s",
            },
        }

        if cfg.env:
            service_def["environment"].update(cfg.env)

        services[service_name] = service_def

    compose_dict = {
        "version": "3.8",
        "services": services,
    }

    return yaml.dump(compose_dict, sort_keys=False, default_flow_style=False)
