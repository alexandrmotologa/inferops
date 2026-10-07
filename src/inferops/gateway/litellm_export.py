"""Generator for LiteLLM proxy configuration file."""

from pathlib import Path
from typing import Dict, Union
import yaml

from inferops.core.config import ModelConfig


def generate_litellm_config(
    catalog: Dict[str, ModelConfig],
    active_models: Dict[str, int],
) -> Dict:
    """Generate LiteLLM compatible configuration dictionary."""
    model_list = []

    for name, port in active_models.items():
        cfg = catalog.get(name)
        alias = cfg.public_alias if cfg else name
        model_list.append(
            {
                "model_name": alias,
                "litellm_params": {
                    "model": f"openai/{alias}",
                    "api_base": f"http://127.0.0.1:{port}/v1",
                    "api_key": "dummy",
                },
            }
        )

    return {
        "model_list": model_list,
        "litellm_settings": {
            "drop_params": True,
            "set_verbose": False,
        },
    }


def write_litellm_config(
    catalog: Dict[str, ModelConfig],
    active_models: Dict[str, int],
    target_path: Union[str, Path],
) -> None:
    """Write generated LiteLLM config to disk."""
    cfg = generate_litellm_config(catalog, active_models)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
