"""Kubernetes manifest generator for production LLM serving workloads."""

import yaml

from inferops.core.config import EngineType, ModelConfig


def generate_kubernetes_manifest(
    model: ModelConfig,
    namespace: str = "inferops",
    pvc_claim_name: str = "huggingface-cache-pvc",
) -> str:
    """Generate production Kubernetes Deployment and Service YAML manifests with GPU limits and health probes."""
    app_label = f"inferops-{model.name}"

    if model.engine == EngineType.VLLM:
        image = "vllm/vllm-openai:latest"
        args = [
            "--model",
            model.model,
            "--port",
            str(model.port),
            "--tensor-parallel-size",
            str(model.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(model.gpu_memory_utilization),
        ]
        if model.max_model_len:
            args.extend(["--max-model-len", str(model.max_model_len)])
        if model.quantization:
            args.extend(["--quantization", model.quantization])
    else:
        image = "lmsysorg/sglang:latest"
        args = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model.model,
            "--port",
            str(model.port),
            "--tp",
            str(model.tensor_parallel_size),
            "--mem-fraction-static",
            str(model.gpu_memory_utilization),
        ]

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": app_label,
            "namespace": namespace,
            "labels": {"app": app_label, "role": "inference-engine"},
        },
        "spec": {
            "replicas": 1,
            "selector": {"matchLabels": {"app": app_label}},
            "template": {
                "metadata": {"labels": {"app": app_label}},
                "spec": {
                    "containers": [
                        {
                            "name": "engine",
                            "image": image,
                            "args": args,
                            "ports": [{"containerPort": model.port, "name": "http"}],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": model.tensor_parallel_size,
                                    "memory": "32Gi",
                                    "cpu": "8",
                                },
                                "requests": {
                                    "nvidia.com/gpu": model.tensor_parallel_size,
                                    "memory": "16Gi",
                                    "cpu": "4",
                                },
                            },
                            "volumeMounts": [
                                {
                                    "name": "hf-cache",
                                    "mountPath": "/root/.cache/huggingface",
                                },
                                {
                                    "name": "dshm",
                                    "mountPath": "/dev/shm",
                                },
                            ],
                            "readinessProbe": {
                                "httpGet": {"path": model.health_endpoint, "port": model.port},
                                "initialDelaySeconds": 45,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                            },
                            "livenessProbe": {
                                "httpGet": {"path": model.health_endpoint, "port": model.port},
                                "initialDelaySeconds": 90,
                                "periodSeconds": 20,
                                "timeoutSeconds": 5,
                            },
                        }
                    ],
                    "volumes": [
                        {
                            "name": "hf-cache",
                            "persistentVolumeClaim": {"claimName": pvc_claim_name},
                        },
                        {
                            "name": "dshm",
                            "emptyDir": {"medium": "Memory"},
                        },
                    ],
                },
            },
        },
    }

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": app_label,
            "namespace": namespace,
            "labels": {"app": app_label},
        },
        "spec": {
            "selector": {"app": app_label},
            "ports": [{"port": model.port, "targetPort": model.port, "name": "http"}],
            "type": "ClusterIP",
        },
    }

    dep_yaml = yaml.dump(deployment, sort_keys=False, default_flow_style=False)
    svc_yaml = yaml.dump(service, sort_keys=False, default_flow_style=False)

    return f"{dep_yaml}\n---\n{svc_yaml}"
