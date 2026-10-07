"""HuggingFace Hub integration for automatic model metadata fetching and configuration synthesis."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import httpx

from inferops.core.config import EngineType, ModelConfig
from inferops.core.exceptions import ConfigurationError
from inferops.core.vram_calculator import VRAMCalculator, VRAMEstimationResult


def slugify_repo_id(repo_id: str) -> str:
    """Convert a HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-Coder-7B-Instruct') into a valid slug name."""
    parts = repo_id.strip().split("/")
    model_name = parts[-1].replace("_", "-")
    slug = re.sub(r"[^a-zA-Z0-9.-]+", "-", model_name).strip("-").lower()
    if not slug or not slug[0].isalnum():
        slug = f"model-{slug}".strip("-")
    return slug


class HuggingFaceClient:
    """Client for querying HuggingFace Hub metadata and configuration files."""

    BASE_URL = "https://huggingface.co"

    def __init__(self, token: Optional[str] = None, timeout: float = 15.0):
        self.token = token
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": "InferOps-Hub-Client/0.1"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def fetch_model_info(self, repo_id: str) -> Dict[str, Any]:
        """Fetch model repository details from HuggingFace API."""
        url = f"{self.BASE_URL}/api/models/{repo_id}"
        with httpx.Client(timeout=self.timeout, headers=self._headers(), follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                raise ConfigurationError(f"HuggingFace model '{repo_id}' not found.")
            if resp.status_code == 401 or resp.status_code == 403:
                raise ConfigurationError(
                    f"Access to '{repo_id}' restricted. Please set HUGGING_FACE_HUB_TOKEN or provide a token."
                )
            resp.raise_for_status()
            return resp.json()

    def fetch_model_config(self, repo_id: str, revision: str = "main") -> Dict[str, Any]:
        """Fetch raw config.json for a model."""
        url = f"{self.BASE_URL}/{repo_id}/raw/{revision}/config.json"
        with httpx.Client(timeout=self.timeout, headers=self._headers(), follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code == 404:
                raise ConfigurationError(f"config.json not found in repository '{repo_id}'.")
            resp.raise_for_status()
            return resp.json()


def calculate_parameters_from_config(cfg: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> float:
    """
    Extract exact or estimated parameter count in billions from model config and metadata.
    """
    if metadata and "safetensors" in metadata and isinstance(metadata["safetensors"], dict):
        total_params = metadata["safetensors"].get("total")
        if total_params and isinstance(total_params, (int, float)) and total_params > 0:
            return round(float(total_params) / 1e9, 2)

    hidden_size = cfg.get("hidden_size") or cfg.get("d_model")
    num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    intermediate_size = cfg.get("intermediate_size")
    vocab_size = cfg.get("vocab_size", 32000)

    if hidden_size and num_layers:
        h = int(hidden_size)
        L = int(num_layers)
        inter = int(intermediate_size) if intermediate_size else int(h * 3.5)

        # Embedding parameters: V * h
        embed_params = vocab_size * h

        # Attention parameters per layer: Q, K, V, O projections
        num_heads = cfg.get("num_attention_heads", 32)
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        head_dim = cfg.get("head_dim") or (h // num_heads)

        q_params = h * (num_heads * head_dim)
        k_params = h * (num_kv_heads * head_dim)
        v_params = h * (num_kv_heads * head_dim)
        o_params = (num_heads * head_dim) * h
        attn_params = q_params + k_params + v_params + o_params

        # MLP parameters per layer (typically Gate, Up, Down for SwiGLU)
        mlp_params = 3 * h * inter

        layer_params = attn_params + mlp_params
        total = embed_params + (L * layer_params)
        return round(total / 1e9, 2)

    return 7.0


def extract_quantization_and_dtype(cfg: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Detect quantization method and weight data type from config."""
    quant = None
    if "quantization_config" in cfg and isinstance(cfg["quantization_config"], dict):
        q_cfg = cfg["quantization_config"]
        quant_method = q_cfg.get("quant_method", "").lower()
        if "awq" in quant_method:
            quant = "awq"
        elif "gptq" in quant_method:
            quant = "gptq"
        elif "bitsandbytes" in quant_method or "bnb" in quant_method:
            quant = "bitsandbytes"
        elif "fp8" in quant_method:
            quant = "fp8"

    dtype_str = cfg.get("torch_dtype", "auto")
    if dtype_str in ["bfloat16", "float16", "float8", "auto"]:
        dtype = dtype_str
    else:
        dtype = "auto"

    return quant, dtype


def synthesize_model_config(
    repo_id: str,
    hf_config: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    engine: EngineType = EngineType.VLLM,
    port: int = 8001,
) -> ModelConfig:
    """Generate a validated ModelConfig instance from HuggingFace config data."""
    slug = slugify_repo_id(repo_id)
    quant, dtype = extract_quantization_and_dtype(hf_config)

    max_pos = (
        hf_config.get("max_position_embeddings")
        or hf_config.get("max_sequence_length")
        or hf_config.get("seq_length")
        or 32768
    )
    # Cap default context to a reasonable operational boundary (e.g. 32k)
    max_model_len = min(int(max_pos), 32768)

    return ModelConfig(
        name=slug,
        model=repo_id,
        engine=engine,
        served_model_name=slug,
        port=port,
        gpus=[0],
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
        dtype=dtype,
        quantization=quant,
        health_endpoint="/health",
        metrics_endpoint="/metrics",
    )


def pull_and_synthesize(
    repo_id: str,
    output_dir: Path,
    token: Optional[str] = None,
    engine: EngineType = EngineType.VLLM,
    port: int = 8001,
) -> Tuple[ModelConfig, Path, VRAMEstimationResult]:
    """
    Fetch repository info, generate model manifest YAML, and run VRAM sizing.
    """
    client = HuggingFaceClient(token=token)
    hf_config = client.fetch_model_config(repo_id)
    try:
        model_info = client.fetch_model_info(repo_id)
    except Exception:
        model_info = None

    model_config = synthesize_model_config(
        repo_id=repo_id,
        hf_config=hf_config,
        metadata=model_info,
        engine=engine,
        port=port,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{model_config.name}.yaml"

    from inferops.core.config import save_model_config

    save_model_config(model_config, manifest_path)

    vram_calc = VRAMCalculator()
    vram_result = vram_calc.calculate(model_config)

    return model_config, manifest_path, vram_result
