"""Unit tests for production DevOps export generators (Docker Compose & Kubernetes)."""

import yaml

from inferops.core.config import EngineType, ModelConfig
from inferops.export.docker import generate_docker_compose
from inferops.export.k8s import generate_kubernetes_manifest


def test_generate_docker_compose():
    m1 = ModelConfig(
        name="qwen-7b",
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        engine=EngineType.VLLM,
        port=8001,
        gpus=[0],
        tensor_parallel_size=1,
        max_model_len=16384,
    )
    m2 = ModelConfig(
        name="llama-8b",
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine=EngineType.SGLANG,
        port=8002,
        gpus=[1, 2],
        tensor_parallel_size=2,
    )

    yaml_str = generate_docker_compose([m1, m2])
    data = yaml.safe_load(yaml_str)

    assert "services" in data
    assert "model-qwen-7b" in data["services"]
    assert "model-llama-8b" in data["services"]

    qwen_svc = data["services"]["model-qwen-7b"]
    assert qwen_svc["image"] == "vllm/vllm-openai:latest"
    assert "8001:8001" in qwen_svc["ports"]
    assert "--max-model-len" in qwen_svc["command"]

    llama_svc = data["services"]["model-llama-8b"]
    assert llama_svc["image"] == "lmsysorg/sglang:latest"
    assert "8002:8002" in llama_svc["ports"]


def test_generate_kubernetes_manifest():
    m = ModelConfig(
        name="mistral-7b",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        engine=EngineType.VLLM,
        port=8001,
        tensor_parallel_size=2,
    )

    k8s_manifest = generate_kubernetes_manifest(m, namespace="inference-prod")
    docs = list(yaml.safe_load_all(k8s_manifest))

    assert len(docs) == 2
    dep = docs[0]
    svc = docs[1]

    assert dep["kind"] == "Deployment"
    assert dep["metadata"]["name"] == "inferops-mistral-7b"
    assert dep["metadata"]["namespace"] == "inference-prod"

    limits = dep["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == 2

    assert svc["kind"] == "Service"
    assert svc["spec"]["ports"][0]["port"] == 8001
