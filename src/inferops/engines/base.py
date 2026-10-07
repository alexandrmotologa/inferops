"""Abstract interface for LLM serving engine adapters."""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List

from inferops.core.config import ModelConfig


class EngineAdapter(ABC):
    """Base adapter translating declarative ModelConfig into engine CLI commands and env."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Name of the engine (e.g. 'vllm' or 'sglang')."""
        pass

    @abstractmethod
    def get_executable(self) -> str:
        """Name of binary or executable command to run."""
        pass

    def is_available(self) -> bool:
        """Check if the required engine binary or package exists in current environment."""
        exe = self.get_executable()
        return shutil.which(exe) is not None

    @abstractmethod
    def build_command(self, config: ModelConfig) -> List[str]:
        """Convert a declarative ModelConfig into the full CLI invocation arguments."""
        pass

    def build_environment(self, config: ModelConfig) -> Dict[str, str]:
        """Prepare environment variables for subprocess, injecting multi-device settings."""
        env = dict(os.environ)

        # Multi-vendor accelerator isolation
        cuda_devices = config.cuda_visible_devices
        if config.device.lower() == "cpu":
            env["VLLM_TARGET_DEVICE"] = "cpu"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["HIP_VISIBLE_DEVICES"] = ""
        elif cuda_devices:
            env["CUDA_VISIBLE_DEVICES"] = cuda_devices
            # AMD ROCm visibility
            env["HIP_VISIBLE_DEVICES"] = cuda_devices
            env["ROCR_VISIBLE_DEVICES"] = cuda_devices

        # Avoid PyTorch allocator memory fragmentation
        if "PYTORCH_CUDA_ALLOC_CONF" not in env:
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        # Merge model-specific custom environment variables
        env.update(config.env)
        return env
