"""Prometheus scrape configuration generator for InferOps LLM inference engines."""

from typing import Dict, List, Optional

import yaml

from inferops.core.config import ModelConfig


def generate_prometheus_config(
    models: Optional[List[ModelConfig]] = None,
    gateway_host: str = "localhost",
    gateway_port: int = 8000,
    scrape_interval: str = "5s",
) -> str:
    """Generate production-ready prometheus.yml scrape configuration."""
    scrape_configs: List[Dict[str, object]] = [
        {
            "job_name": "inferops-gateway",
            "scrape_interval": scrape_interval,
            "metrics_path": "/metrics",
            "static_configs": [
                {
                    "targets": [f"{gateway_host}:{gateway_port}"],
                    "labels": {"app": "inferops", "role": "gateway"},
                }
            ],
        }
    ]

    # Add individual engine targets if present
    if models:
        for m in models:
            scrape_configs.append(
                {
                    "job_name": f"inferops-engine-{m.name}",
                    "scrape_interval": scrape_interval,
                    "metrics_path": m.metrics_endpoint or "/metrics",
                    "static_configs": [
                        {
                            "targets": [f"{m.host if m.host != '0.0.0.0' else 'localhost'}:{m.port}"],
                            "labels": {
                                "app": "inferops",
                                "role": "engine",
                                "model": m.name,
                                "engine": m.engine.value,
                            },
                        }
                    ],
                }
            )

    config = {
        "global": {
            "scrape_interval": scrape_interval,
            "evaluation_interval": scrape_interval,
        },
        "scrape_configs": scrape_configs,
    }

    return yaml.dump(config, sort_keys=False, default_flow_style=False)


def export_prometheus_config_file(
    filepath: str,
    models: Optional[List[ModelConfig]] = None,
    gateway_host: str = "localhost",
    gateway_port: int = 8000,
) -> str:
    """Generate and write prometheus.yml to disk."""
    content = generate_prometheus_config(models=models, gateway_host=gateway_host, gateway_port=gateway_port)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return content
