"""Prometheus metrics parser and live ring buffer for vLLM & SGLang."""

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import httpx


@dataclass
class ModelLiveMetrics:
    """Snapshot of active throughput, concurrency, and KV cache usage."""
    timestamp: float
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    gpu_cache_usage_pct: float = 0.0
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    throughput_tokens_per_sec: float = 0.0
    time_to_first_token_avg_ms: Optional[float] = None


class MetricsRingBuffer:
    """Thread-safe ring buffer storing time-series metric snapshots."""

    def __init__(self, max_points: int = 60) -> None:
        self.max_points = max_points
        self._history: Deque[ModelLiveMetrics] = deque(maxlen=max_points)

    def add(self, snapshot: ModelLiveMetrics) -> None:
        # Calculate instant token throughput based on difference with last point
        if self._history:
            prev = self._history[-1]
            dt = max(0.001, snapshot.timestamp - prev.timestamp)
            delta_tokens = max(0, snapshot.generation_tokens_total - prev.generation_tokens_total)
            snapshot.throughput_tokens_per_sec = round(delta_tokens / dt, 1)

        self._history.append(snapshot)

    def latest(self) -> Optional[ModelLiveMetrics]:
        return self._history[-1] if self._history else None

    def series(self) -> List[ModelLiveMetrics]:
        return list(self._history)


def parse_prometheus_text(text: str) -> Dict[str, float]:
    """Extract float values from Prometheus text format lines."""
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: metric_name{labels} value or metric_name value
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(\{[^}]*\})?\s+([0-9eE.+-]+)", line)
        if m:
            name = m.group(1)
            val_str = m.group(3)
            try:
                metrics[name] = float(val_str)
            except ValueError:
                pass
    return metrics


async def fetch_model_metrics(
    host: str,
    port: int,
    metrics_path: str = "/metrics",
    timeout: float = 2.0,
) -> Optional[ModelLiveMetrics]:
    """Fetch and parse live metrics from a model's Prometheus endpoint."""
    url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}{metrics_path}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            metrics = parse_prometheus_text(resp.text)

            running = int(metrics.get("vllm:num_requests_running", metrics.get("sglang:num_requests_running", 0)))
            waiting = int(metrics.get("vllm:num_requests_waiting", metrics.get("sglang:num_requests_waiting", 0)))
            cache_usage = metrics.get("vllm:gpu_cache_usage_factor", metrics.get("sglang:gpu_cache_usage", 0.0)) * 100.0
            prompt_tokens = int(metrics.get("vllm:prompt_tokens_total", metrics.get("sglang:prompt_tokens_total", 0)))
            gen_tokens = int(metrics.get("vllm:generation_tokens_total", metrics.get("sglang:generation_tokens_total", 0)))

            return ModelLiveMetrics(
                timestamp=time.time(),
                num_requests_running=running,
                num_requests_waiting=waiting,
                gpu_cache_usage_pct=round(cache_usage, 1),
                prompt_tokens_total=prompt_tokens,
                generation_tokens_total=gen_tokens,
            )
    except Exception:
        return None
