"""vLLM engine adapter."""

import sys
from typing import List

from inferops.core.config import ModelConfig
from inferops.engines.base import EngineAdapter


class VllmEngineAdapter(EngineAdapter):
    """Adapter for running vLLM OpenAI-compatible API server."""

    @property
    def engine_name(self) -> str:
        return "vllm"

    def get_executable(self) -> str:
        return "vllm"

    def is_available(self) -> bool:
        # Check standard PATH or python module
        if super().is_available():
            return True
        try:
            import vllm  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False

    def build_command(self, config: ModelConfig) -> List[str]:
        # Prefer invoking via python module if vllm CLI isn't in PATH
        base_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        if super().is_available():
            base_cmd = ["vllm", "serve"]

        cmd = list(base_cmd)
        cmd.append(config.model)

        cmd.extend(["--host", config.host])
        cmd.extend(["--port", str(config.port)])
        cmd.extend(["--served-model-name", config.public_alias])
        cmd.extend(["--tensor-parallel-size", str(config.tensor_parallel_size)])
        cmd.extend(["--pipeline-parallel-size", str(config.pipeline_parallel_size)])
        cmd.extend(["--gpu-memory-utilization", str(config.gpu_memory_utilization)])
        cmd.extend(["--dtype", config.dtype])

        if config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(config.max_model_len)])

        if config.quantization:
            cmd.extend(["--quantization", config.quantization])

        if config.kv_cache_dtype and config.kv_cache_dtype != "auto":
            cmd.extend(["--kv-cache-dtype", config.kv_cache_dtype])

        if config.lora_modules:
            cmd.append("--enable-lora")
            for mod in config.lora_modules:
                name = mod.get("name")
                path = mod.get("path")
                if name and path:
                    cmd.extend(["--lora-modules", f"{name}={path}"])

        # Extra freeform flags
        cmd.extend(config.extra_args)

        return cmd
