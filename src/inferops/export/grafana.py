"""Production-grade Grafana dashboard generator for InferOps LLM inference monitoring."""

import json
from typing import Any, Dict


def generate_grafana_dashboard(title: str = "InferOps LLM Inference Control Plane") -> Dict[str, Any]:
    """Generate a comprehensive Grafana 10+ dashboard JSON model."""
    dashboard = {
        "annotations": {"list": []},
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 1,
        "id": None,
        "links": [],
        "liveNow": True,
        "panels": [
            # Row 1: Key Performance Indicators (Stats)
            {
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 0},
                "id": 1,
                "title": "Total API Invocations",
                "type": "stat",
                "targets": [
                    {
                        "expr": "sum(inferops_requests_total)",
                        "legendFormat": "Requests",
                        "refId": "A",
                    }
                ],
                "options": {
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                },
            },
            {
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 0},
                "id": 2,
                "title": "Total Tokens Served",
                "type": "stat",
                "targets": [
                    {
                        "expr": "sum(inferops_tokens_total)",
                        "legendFormat": "Tokens",
                        "refId": "A",
                    }
                ],
                "options": {
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                },
            },
            {
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 0},
                "id": 3,
                "title": "Commercial Cost Saved",
                "type": "stat",
                "targets": [
                    {
                        "expr": "inferops_cost_saved_usd_total",
                        "legendFormat": "Savings ($)",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [{"color": "green", "value": None}],
                        },
                        "unit": "currencyUSD",
                    }
                },
                "options": {
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                },
            },
            {
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 0},
                "id": 4,
                "title": "P95 TTFT (Time-To-First-Token)",
                "type": "stat",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[5m])) by (le))",
                        "legendFormat": "P95 TTFT",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "s",
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 0.15},
                                {"color": "red", "value": 0.5},
                            ],
                        },
                    }
                },
                "options": {
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "auto",
                    "reduceOptions": {"calcs": ["lastNotNull"], "values": False},
                },
            },
            # Row 2: Throughput and Velocity
            {
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 4},
                "id": 5,
                "title": "Request Throughput by Model (QPS)",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "sum(rate(inferops_requests_total[1m])) by (model)",
                        "legendFormat": "{{model}}",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineInterpolation": "smooth", "fillOpacity": 15},
                        "unit": "reqps",
                    }
                },
            },
            {
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 4},
                "id": 6,
                "title": "Token Generation Velocity (tok/s)",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "sum(rate(vllm:avg_generation_throughput_tok_per_s[1m])) by (model)",
                        "legendFormat": "{{model}} Tok/s",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "custom": {"drawStyle": "line", "lineInterpolation": "smooth", "fillOpacity": 15},
                        "unit": "short",
                    }
                },
            },
            # Row 3: Memory and Hardware Telemetry
            {
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 12},
                "id": 7,
                "title": "GPU VRAM Allocation (GB)",
                "type": "timeseries",
                "targets": [
                    {
                        "expr": "inferops_gpu_memory_used_bytes / (1024 * 1024 * 1024)",
                        "legendFormat": "GPU {{gpu_index}} Used VRAM",
                        "refId": "A",
                    },
                    {
                        "expr": "inferops_gpu_memory_total_bytes / (1024 * 1024 * 1024)",
                        "legendFormat": "GPU {{gpu_index}} Total VRAM",
                        "refId": "B",
                    },
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "decgbytes",
                        "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                    }
                },
            },
            {
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 12},
                "id": 8,
                "title": "KV Cache Block Utilization (%)",
                "type": "gauge",
                "targets": [
                    {
                        "expr": "vllm:gpu_cache_usage_factor * 100",
                        "legendFormat": "{{model}} KV Cache",
                        "refId": "A",
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "min": 0,
                        "max": 100,
                        "unit": "percent",
                        "color": {"mode": "thresholds"},
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "green", "value": None},
                                {"color": "yellow", "value": 75},
                                {"color": "red", "value": 90},
                            ],
                        },
                    }
                },
            },
        ],
        "refresh": "5s",
        "schemaVersion": 39,
        "tags": ["inferops", "vllm", "sglang", "llamacpp", "llm-serving"],
        "templating": {
            "list": [
                {
                    "name": "model",
                    "type": "query",
                    "label": "Model",
                    "query": "label_values(inferops_requests_total, model)",
                    "refresh": 1,
                    "includeAll": True,
                    "multi": True,
                }
            ]
        },
        "time": {"from": "now-30m", "to": "now"},
        "timepicker": {"refresh_intervals": ["1s", "5s", "10s", "30s"]},
        "timezone": "browser",
        "title": title,
        "uid": "inferops-overview",
        "version": 1,
    }
    return dashboard


def export_grafana_dashboard_json(filepath: str, title: str = "InferOps LLM Inference Control Plane") -> str:
    """Generate and write Grafana dashboard JSON to specified path."""
    dashboard = generate_grafana_dashboard(title=title)
    content = json.dumps(dashboard, indent=2)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return content
