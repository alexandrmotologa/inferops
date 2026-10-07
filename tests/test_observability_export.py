"""Tests for Grafana dashboard and Prometheus scrape config generators."""

import json

import yaml

from inferops.core.config import EngineType, ModelConfig
from inferops.export.grafana import generate_grafana_dashboard
from inferops.export.prometheus import generate_prometheus_config


def test_grafana_dashboard_structure():
    """Verify generated Grafana 10+ dashboard structure and KPI panels."""
    dashboard = generate_grafana_dashboard(title="Test Inference Observability")
    assert dashboard["title"] == "Test Inference Observability"
    assert dashboard["uid"] == "inferops-overview"
    assert len(dashboard["panels"]) >= 8

    # Verify panel titles exist
    titles = [p["title"] for p in dashboard["panels"]]
    assert "Total API Invocations" in titles
    assert "Total Tokens Served" in titles
    assert "Commercial Cost Saved" in titles
    assert "GPU VRAM Allocation (GB)" in titles
    assert "KV Cache Block Utilization (%)" in titles

    # Must serialize cleanly to valid JSON
    serialized = json.dumps(dashboard)
    assert len(serialized) > 1000


def test_prometheus_config_generation():
    """Verify generated prometheus.yml has scrape configs for gateway and models."""
    models = [
        ModelConfig(name="qwen", model="Qwen/Qwen2.5-7B", port=8001),
        ModelConfig(name="llama", model="meta-llama/Llama-3.1-8B", engine=EngineType.LLAMACPP, port=8003),
    ]
    prom_yaml = generate_prometheus_config(models=models, gateway_port=8000)
    data = yaml.safe_load(prom_yaml)

    assert "global" in data
    assert "scrape_configs" in data
    job_names = [sc["job_name"] for sc in data["scrape_configs"]]

    assert "inferops-gateway" in job_names
    assert "inferops-engine-qwen" in job_names
    assert "inferops-engine-llama" in job_names
