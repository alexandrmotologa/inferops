"""Unit tests for HuggingFace Hub client and configuration synthesizer."""

from unittest.mock import patch

from inferops.core.config import EngineType
from inferops.core.hf_hub import (
    calculate_parameters_from_config,
    extract_quantization_and_dtype,
    pull_and_synthesize,
    slugify_repo_id,
    synthesize_model_config,
)


def test_slugify_repo_id():
    assert slugify_repo_id("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b-instruct"
    assert slugify_repo_id("Qwen/Qwen2.5-Coder-7B-Instruct") == "qwen2.5-coder-7b-instruct"
    assert slugify_repo_id("mistralai/Mistral-7B-v0.1") == "mistral-7b-v0.1"
    assert slugify_repo_id("user/my_custom_model_v1") == "my-custom-model-v1"


def test_calculate_parameters_from_config():
    qwen_cfg = {
        "hidden_size": 3584,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "intermediate_size": 18944,
        "vocab_size": 152064,
    }
    params_b = calculate_parameters_from_config(qwen_cfg)
    # Expected around 7.6B params
    assert 7.0 <= params_b <= 8.0

    # Test with safetensors total in metadata
    meta = {"safetensors": {"total": 8030000000}}
    assert calculate_parameters_from_config({}, meta) == 8.03


def test_extract_quantization_and_dtype():
    awq_cfg = {"quantization_config": {"quant_method": "awq"}, "torch_dtype": "float16"}
    quant, dtype = extract_quantization_and_dtype(awq_cfg)
    assert quant == "awq"
    assert dtype == "float16"

    fp8_cfg = {"quantization_config": {"quant_method": "fp8"}, "torch_dtype": "float8"}
    quant, dtype = extract_quantization_and_dtype(fp8_cfg)
    assert quant == "fp8"
    assert dtype == "float8"

    standard_cfg = {"torch_dtype": "bfloat16"}
    quant, dtype = extract_quantization_and_dtype(standard_cfg)
    assert quant is None
    assert dtype == "bfloat16"


def test_synthesize_model_config():
    hf_cfg = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "max_position_embeddings": 131072,
        "torch_dtype": "bfloat16",
    }
    model_cfg = synthesize_model_config(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        hf_config=hf_cfg,
        engine=EngineType.VLLM,
        port=8005,
    )
    assert model_cfg.name == "llama-3.1-8b-instruct"
    assert model_cfg.port == 8005
    assert model_cfg.engine == EngineType.VLLM
    # Max model len capped at 32768 by default
    assert model_cfg.max_model_len == 32768
    assert model_cfg.dtype == "bfloat16"


def test_pull_and_synthesize(tmp_path):
    mock_hf_config = {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "max_position_embeddings": 8192,
        "torch_dtype": "float16",
    }
    with patch("inferops.core.hf_hub.HuggingFaceClient.fetch_model_config", return_value=mock_hf_config), \
         patch("inferops.core.hf_hub.HuggingFaceClient.fetch_model_info", return_value=None):

        cfg, manifest_path, vram = pull_and_synthesize(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            output_dir=tmp_path,
            engine=EngineType.VLLM,
            port=8009,
        )

        assert manifest_path.is_file()
        assert cfg.name == "mistral-7b-instruct-v0.3"
        assert cfg.port == 8009
        assert vram.params_billions > 5.0
