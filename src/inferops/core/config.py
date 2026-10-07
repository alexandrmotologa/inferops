"""Declarative configuration models and YAML parser for InferOps."""

import os
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator

from inferops.core.exceptions import ConfigurationError


class EngineType(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    LLAMACPP = "llamacpp"


def interpolate_env_vars(text: str) -> str:
    """Interpolate environment variables with syntax ${VAR} or ${VAR:-default}."""
    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

    def replacer(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default_val = match.group(2) if match.group(2) is not None else ""
        return os.environ.get(var_name, default_val)

    return pattern.sub(replacer, text)


class ModelConfig(BaseModel):
    """Specification of an individual LLM model serving instance."""

    name: str = Field(..., description="Unique model identifier slug, e.g. 'qwen2.5-coder-7b'")
    model: str = Field(..., description="Hugging Face repo id or local path to model weights")
    engine: EngineType = Field(default=EngineType.VLLM, description="Inference engine: vllm or sglang")
    served_model_name: Optional[str] = Field(
        default=None,
        description="Public model alias exposed via OpenAI-compatible API. Defaults to name.",
    )
    host: str = Field(default="0.0.0.0", description="Host address to bind to")
    port: int = Field(default=8001, ge=1024, le=65535, description="Port for this model instance")
    gpus: Union[List[int], str] = Field(
        default=[0],
        description="GPU indices to allocate, e.g. [0] or [0, 1] or 'all'",
    )
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallelism degree")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallelism degree")
    max_model_len: Optional[int] = Field(default=None, description="Max sequence/context length")
    gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.1,
        le=0.99,
        description="Fraction of GPU VRAM reserved for weights and KV cache",
    )
    dtype: str = Field(default="auto", description="Weights data type: auto, float16, bfloat16, float8")
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization format, e.g. awq, gptq, fp8, bitsandbytes",
    )
    kv_cache_dtype: Optional[str] = Field(
        default="auto",
        description="KV cache data type, e.g. auto, fp8",
    )
    env: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables passed directly to the engine process",
    )
    extra_args: List[str] = Field(
        default_factory=list,
        description="Additional raw CLI arguments passed directly to the engine",
    )
    health_endpoint: str = Field(default="/health", description="HTTP healthcheck endpoint")
    metrics_endpoint: str = Field(default="/metrics", description="Prometheus metrics endpoint")
    idle_timeout_seconds: Optional[int] = Field(
        default=None,
        description="Inactivity window in seconds before model is automatically scaled to zero",
    )
    auto_scale: bool = Field(
        default=False,
        description="Whether to wake up model on demand when incoming requests arrive",
    )
    lora_modules: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of LoRA modules as [{'name': 'alias', 'path': '/path/or/repo'}]",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote executable code from Hugging Face repository",
    )
    device: str = Field(
        default="auto",
        description="Target execution device/backend: auto, cuda, rocm, cpu, mps",
    )
    speculative_model: Optional[str] = Field(
        default=None,
        description="Draft model repository or path used for speculative decoding acceleration",
    )
    num_speculative_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=16,
        description="Number of speculative tokens drafted per iteration",
    )
    gpu_layers: Optional[int] = Field(
        default=None,
        description="Number of layers to offload to GPU/Metal in llama.cpp (-ngl / --n-gpu-layers). 99=full GPU, 0=CPU",
    )
    threads: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of CPU compute threads to allocate in llama.cpp (-t / --threads)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip().lower()
        if not re.match(r"^[a-z0-9][a-z0-9._-]*$", v):
            raise ValueError(
                f"Model name '{v}' is invalid. Use lowercase letters, digits, '.', '-', and '_'."
            )
        return v

    @property
    def public_alias(self) -> str:
        return self.served_model_name or self.name

    @property
    def cuda_visible_devices(self) -> str:
        if isinstance(self.gpus, str):
            if self.gpus.lower() == "all":
                return ""
            return self.gpus
        return ",".join(str(g) for g in self.gpus)


class ProfileConfig(BaseModel):
    """Configuration mapping profile names to lists of model names."""

    profiles: Dict[str, List[str]] = Field(
        default_factory=lambda: {"dev": [], "prod": []},
        description="Mapping of profile name to model names",
    )


class InferOpsProjectConfig(BaseModel):
    """Global configuration for an InferOps workspace."""

    project_name: str = Field(default="inferops-workspace")
    gateway_port: int = Field(default=8000, description="Port for the unified OpenAI API gateway")
    web_port: int = Field(default=8900, description="Port for the embedded Web Dashboard")
    default_engine: EngineType = Field(default=EngineType.VLLM)
    profiles: Dict[str, List[str]] = Field(default_factory=lambda: {"default": []})


def load_model_config(path: Union[str, Path]) -> ModelConfig:
    """Load and validate a ModelConfig from a YAML file with env var substitution."""
    path = Path(path)
    if not path.is_file():
        raise ConfigurationError(f"Model config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
        interpolated = interpolate_env_vars(raw_text)
        data = yaml.safe_load(interpolated)
        if not isinstance(data, dict):
            raise ConfigurationError(f"Model YAML at {path} must contain a mapping object.")
        return ModelConfig.model_validate(data)
    except Exception as e:
        if isinstance(e, ConfigurationError):
            raise
        raise ConfigurationError(f"Failed to parse model config {path}: {e}") from e


def save_model_config(config: ModelConfig, path: Union[str, Path]) -> None:
    """Serialize a ModelConfig to a formatted YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = config.model_dump(mode="json", exclude_none=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(dumped, f, sort_keys=False, default_flow_style=False)


def load_profiles(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Load profile groupings from a YAML file."""
    path = Path(path)
    if not path.is_file():
        return {}

    try:
        raw_text = path.read_text(encoding="utf-8")
        interpolated = interpolate_env_vars(raw_text)
        data = yaml.safe_load(interpolated) or {}
        if "profiles" in data and isinstance(data["profiles"], dict):
            return data["profiles"]
        if isinstance(data, dict):
            return data
        return {}
    except Exception as e:
        raise ConfigurationError(f"Failed to parse profiles from {path}: {e}") from e


def discover_models(models_dir: Union[str, Path]) -> Dict[str, ModelConfig]:
    """Scan directory for *.yaml model configs and return a map of name -> ModelConfig."""
    models_dir = Path(models_dir)
    catalog: Dict[str, ModelConfig] = {}
    if not models_dir.is_dir():
        return catalog

    for file_path in sorted(models_dir.glob("*.yaml")):
        try:
            cfg = load_model_config(file_path)
            catalog[cfg.name] = cfg
        except Exception:
            continue
    for file_path in sorted(models_dir.glob("*.yml")):
        if file_path.stem in catalog:
            continue
        try:
            cfg = load_model_config(file_path)
            catalog[cfg.name] = cfg
        except Exception:
            continue

    return catalog
