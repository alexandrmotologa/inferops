"""SGLang engine adapter."""

import sys
from typing import List

from inferops.core.config import ModelConfig
from inferops.engines.base import EngineAdapter


class SglangEngineAdapter(EngineAdapter):
    """Adapter for running SGLang high-throughput LLM server."""

    @property
    def engine_name(self) -> str:
        return "sglang"

    def get_executable(self) -> str:
        return "sglang"

    def is_available(self) -> bool:
        if super().is_available():
            return True
        try:
            import sglang  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False

    def build_command(self, config: ModelConfig) -> List[str]:
        # SGLang can be launched via python -m sglang.launch_server or sglang launch_server
        cmd = [sys.executable, "-m", "sglang.launch_server"]
        cmd.extend(["--model-path", config.model])
        cmd.extend(["--host", config.host])
        cmd.extend(["--port", str(config.port)])
        cmd.extend(["--served-model-name", config.public_alias])
        cmd.extend(["--tp", str(config.tensor_parallel_size)])
        cmd.extend(["--mem-fraction-static", str(config.gpu_memory_utilization)])

        if config.dtype and config.dtype != "auto":
            cmd.extend(["--dtype", config.dtype])

        if config.max_model_len is not None:
            cmd.extend(["--context-length", str(config.max_model_len)])

        if config.quantization:
            cmd.extend(["--quantization", config.quantization])

        if config.kv_cache_dtype and config.kv_cache_dtype != "auto":
            cmd.extend(["--kv-cache-dtype", config.kv_cache_dtype])

        if config.trust_remote_code:
            cmd.append("--trust-remote-code")

        if config.speculative_model:
            cmd.extend(["--speculative-draft-model", config.speculative_model])
            if config.num_speculative_tokens:
                cmd.extend(["--speculative-num-steps", str(config.num_speculative_tokens)])

        if config.lora_modules:
            cmd.append("--enable-lora")
            for mod in config.lora_modules:
                path = mod.get("path")
                if path:
                    cmd.extend(["--lora-paths", path])

        # Extra freeform flags
        cmd.extend(config.extra_args)

        return cmd
