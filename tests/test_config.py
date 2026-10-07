"""Unit tests for configuration models and YAML parser."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from inferops.core.config import (
    EngineType,
    ModelConfig,
    discover_models,
    interpolate_env_vars,
    load_model_config,
    load_profiles,
    save_model_config,
)


def test_interpolate_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_HF_TOKEN", "hf_secret_123")
    text = "token: ${TEST_HF_TOKEN} and port: ${CUSTOM_PORT:-8080}"
    result = interpolate_env_vars(text)
    assert result == "token: hf_secret_123 and port: 8080"


def test_model_config_validation() -> None:
    cfg = ModelConfig(
        name="qwen-7b",
        model="Qwen/Qwen2.5-7B-Instruct",
        port=8001,
        gpus=[0, 1],
    )
    assert cfg.name == "qwen-7b"
    assert cfg.engine == EngineType.VLLM
    assert cfg.cuda_visible_devices == "0,1"
    assert cfg.public_alias == "qwen-7b"


def test_model_config_invalid_name() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(name="Invalid Name With Spaces!", model="foo/bar")


def test_save_and_load_model_config(tmp_path: Path) -> None:
    cfg = ModelConfig(
        name="test-model",
        model="meta-llama/Llama-3.1-8B",
        engine=EngineType.SGLANG,
        port=8005,
        gpus=[0],
        max_model_len=16384,
    )
    target_file = tmp_path / "test-model.yaml"
    save_model_config(cfg, target_file)

    loaded = load_model_config(target_file)
    assert loaded.name == "test-model"
    assert loaded.engine == EngineType.SGLANG
    assert loaded.max_model_len == 16384


def test_discover_models(tmp_path: Path) -> None:
    cfg1 = ModelConfig(name="m1", model="org/m1", port=8001)
    cfg2 = ModelConfig(name="m2", model="org/m2", port=8002)

    save_model_config(cfg1, tmp_path / "m1.yaml")
    save_model_config(cfg2, tmp_path / "m2.yaml")

    catalog = discover_models(tmp_path)
    assert "m1" in catalog
    assert "m2" in catalog
    assert catalog["m1"].port == 8001
    assert catalog["m2"].port == 8002


def test_load_profiles(tmp_path: Path) -> None:
    prof_file = tmp_path / "profiles.yaml"
    prof_file.write_text("profiles:\n  dev:\n    - m1\n  prod:\n    - m2\n", encoding="utf-8")

    profiles = load_profiles(prof_file)
    assert profiles == {"dev": ["m1"], "prod": ["m2"]}
