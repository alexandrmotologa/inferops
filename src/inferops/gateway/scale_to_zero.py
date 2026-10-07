"""Scale-to-Zero and on-demand cold start coordinator for InferOps inference models."""

import asyncio
import logging
import time
from typing import Dict, Optional

from inferops.core.config import ModelConfig
from inferops.core.health import HealthChecker
from inferops.core.supervisor import ProcessSupervisor

logger = logging.getLogger("inferops.scale_to_zero")


class ScaleToZeroCoordinator:
    """Monitors model inactivity and orchestrates automatic scale-to-zero and cold starts."""

    def __init__(self) -> None:
        self._last_accessed: Dict[str, float] = {}
        self._wake_locks: Dict[str, asyncio.Lock] = {}

    def touch(self, model_name: str) -> None:
        """Record activity for a model."""
        self._last_accessed[model_name] = time.time()

    def get_last_accessed(self, model_name: str) -> float:
        """Get timestamp of last recorded activity."""
        return self._last_accessed.get(model_name, time.time())

    async def check_and_reap_idle(
        self,
        supervisor: ProcessSupervisor,
        catalog: Dict[str, ModelConfig],
        active_ports: Dict[str, int],
    ) -> Dict[str, str]:
        """
        Check all running models; if inactivity exceeds idle_timeout_seconds, gracefully shut down.
        Returns a dict of model_name -> reason for reaped models.
        """
        now = time.time()
        reaped: Dict[str, str] = {}

        for name, port in list(active_ports.items()):
            cfg = catalog.get(name)
            if not cfg or not cfg.idle_timeout_seconds:
                continue

            last_seen = self._last_accessed.get(name, now)
            idle_duration = now - last_seen

            if idle_duration >= cfg.idle_timeout_seconds:
                logger.info(
                    "Model '%s' idle for %.1fs (threshold %ds). Scaling to zero.",
                    name,
                    idle_duration,
                    cfg.idle_timeout_seconds,
                )
                try:
                    supervisor.stop_model(name)
                    del active_ports[name]
                    reaped[name] = f"Idle for {int(idle_duration)}s, exceeding limit of {cfg.idle_timeout_seconds}s"
                except Exception as e:
                    logger.error("Failed to scale model '%s' to zero: %s", name, e)

        return reaped

    async def wake_on_demand(
        self,
        model_name: str,
        supervisor: ProcessSupervisor,
        catalog: Dict[str, ModelConfig],
        active_ports: Dict[str, int],
        timeout_seconds: float = 90.0,
    ) -> Optional[int]:
        """
        Wake up an idle/stopped model on demand if auto_scale is enabled.
        Blocks until the model passes its readiness probe, then returns its active port.
        """
        cfg = catalog.get(model_name)
        if not cfg or not cfg.auto_scale:
            return None

        # Concurrency guard: avoid multiple simultaneous cold starts for the same model
        if model_name not in self._wake_locks:
            self._wake_locks[model_name] = asyncio.Lock()

        async with self._wake_locks[model_name]:
            # Double check if model was already started while waiting for lock
            if model_name in active_ports:
                self.touch(model_name)
                return active_ports[model_name]

            logger.info("Cold-starting model '%s' on demand on port %d...", model_name, cfg.port)
            record = supervisor.start_model(cfg)
            if not record:
                logger.error("Supervisor failed to initiate process for '%s'", model_name)
                return None

            checker = HealthChecker(host="127.0.0.1", port=cfg.port, endpoint=cfg.health_endpoint)
            # Asynchronously poll readiness until healthy
            start_wait = time.time()
            is_ready = False
            while (time.time() - start_wait) < timeout_seconds:
                if checker.check():
                    is_ready = True
                    break
                await asyncio.sleep(1.0)

            if not is_ready:
                logger.error("Model '%s' failed to become ready within %ds", model_name, timeout_seconds)
                return None

            active_ports[model_name] = cfg.port
            self.touch(model_name)
            logger.info("Model '%s' successfully cold-started and ready on port %d", model_name, cfg.port)
            return cfg.port
