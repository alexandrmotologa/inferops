"""Dynamic workload tracking and concurrency autoscaling recommendations for InferOps."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class ScalingAction(str, Enum):
    OPTIMAL = "OPTIMAL"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"


@dataclass
class ModelWorkloadMetrics:
    """Live traffic and concurrency telemetry for a model."""
    model_name: str
    active_requests: int = 0
    total_completed_requests: int = 0
    total_latency_seconds: float = 0.0
    recent_latencies: List[float] = field(default_factory=list)
    last_request_timestamp: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        if not self.recent_latencies:
            return 0.0
        return round((sum(self.recent_latencies) / len(self.recent_latencies)) * 1000.0, 2)


@dataclass
class AutoscalingPolicy:
    """Rules and thresholds for triggering replica adjustments."""
    target_concurrency_per_replica: int = 16
    max_concurrency_per_replica: int = 32
    latency_threshold_ms: float = 500.0
    scale_down_idle_seconds: float = 300.0


class WorkloadAutoscaler:
    """Monitors live request volume and computes autoscaling signals."""

    def __init__(self, policy: Optional[AutoscalingPolicy] = None) -> None:
        self.policy = policy or AutoscalingPolicy()
        self._metrics: Dict[str, ModelWorkloadMetrics] = {}

    def get_or_create_metrics(self, model_name: str) -> ModelWorkloadMetrics:
        if model_name not in self._metrics:
            self._metrics[model_name] = ModelWorkloadMetrics(model_name=model_name)
        return self._metrics[model_name]

    def record_request_start(self, model_name: str) -> None:
        metrics = self.get_or_create_metrics(model_name)
        metrics.active_requests += 1
        metrics.last_request_timestamp = time.time()

    def record_request_end(self, model_name: str, latency_seconds: float) -> None:
        metrics = self.get_or_create_metrics(model_name)
        metrics.active_requests = max(0, metrics.active_requests - 1)
        metrics.total_completed_requests += 1
        metrics.total_latency_seconds += latency_seconds
        metrics.recent_latencies.append(latency_seconds)
        if len(metrics.recent_latencies) > 50:
            metrics.recent_latencies.pop(0)

    def evaluate_model(self, model_name: str, current_replicas: int = 1) -> ScalingAction:
        """Evaluate whether a model needs more replicas or scale-down."""
        metrics = self.get_or_create_metrics(model_name)
        now = time.time()

        # If zero traffic for idle window
        if metrics.active_requests == 0 and (now - metrics.last_request_timestamp) > self.policy.scale_down_idle_seconds:
            if current_replicas > 0:
                return ScalingAction.SCALE_DOWN

        # Concurrency load per current replica
        effective_replicas = max(1, current_replicas)
        concurrency_per_replica = metrics.active_requests / effective_replicas

        # Scale UP conditions: high concurrency or elevated latency
        if concurrency_per_replica > self.policy.max_concurrency_per_replica:
            return ScalingAction.SCALE_UP
        if metrics.avg_latency_ms > self.policy.latency_threshold_ms and metrics.active_requests > 4:
            return ScalingAction.SCALE_UP

        # Scale DOWN condition if over-provisioned
        if current_replicas > 1 and concurrency_per_replica < (self.policy.target_concurrency_per_replica / 4):
            return ScalingAction.SCALE_DOWN

        return ScalingAction.OPTIMAL
