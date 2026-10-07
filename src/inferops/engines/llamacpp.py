"""llama.cpp engine adapter for GGUF model serving across CPU, Apple Metal, and GPUs."""

import shutil
import sys
from typing import List

from inferops.core.config import ModelConfig
from inferops.engines.base import EngineAdapter


class LlamaCppEngineAdapter(EngineAdapter):
    """Adapter for running llama.cpp OpenAI-compatible server (llama-server)."""

    @property
    def engine_name(self) -> str:
        return "llamacpp"

    def get_executable(self) -> str:
        return "llama-server"

    def is_available(self) -> bool:
        if shutil.which("llama-server") or shutil.which("llama-cli"):
            return True
        try:
            import llama_cpp  # type: ignore # noqa: F401
            return True
        except ImportError:
            return False

    def build_command(self, config: ModelConfig) -> List[str]:
        # Check if llama-server binary exists, otherwise invoke python module
        if shutil.which("llama-server"):
            cmd = ["llama-server"]
        elif shutil.which("llama_cpp.server"):
            cmd = ["llama_cpp.server"]
        else:
            cmd = [sys.executable, "-m", "llama_cpp.server"]

        cmd.extend(["-m", config.model])
        cmd.extend(["--host", config.host])
        cmd.extend(["--port", str(config.port)])
        cmd.extend(["--alias", config.public_alias])

        if config.max_model_len is not None:
            cmd.extend(["-c", str(config.max_model_len)])

        # GPU offload layers (e.g. 99 for all layers to GPU/Metal, 0 for CPU)
        if config.gpu_layers is not None:
            cmd.extend(["-ngl", str(config.gpu_layers)])
        elif config.device.lower() != "cpu":
            # Default to full GPU offload if not explicitly cpu
            cmd.extend(["-ngl", "99"])

        if config.threads:
            cmd.extend(["-t", str(config.threads)])

        # Append freeform arguments
        cmd.extend(config.extra_args)

        return cmd
