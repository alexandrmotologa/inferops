"""Unit tests for scale-to-zero coordinator and on-demand cold starts."""

from unittest.mock import MagicMock

import pytest

from inferops.core.config import EngineType, ModelConfig
from inferops.gateway.scale_to_zero import ScaleToZeroCoordinator


@pytest.mark.asyncio
async def test_scale_to_zero_idle_reaping():
    coord = ScaleToZeroCoordinator()

    cfg = ModelConfig(
        name="idle-model",
        model="fake/model",
        engine=EngineType.VLLM,
        port=8001,
        idle_timeout_seconds=5,
    )

    catalog = {"idle-model": cfg}
    active_ports = {"idle-model": 8001}

    # Simulate activity in the past (10 seconds ago)
    coord._last_accessed["idle-model"] = 1000.0

    mock_supervisor = MagicMock()
    mock_supervisor.stop_model.return_value = True

    # Reaping should identify the model and stop it
    reaped = await coord.check_and_reap_idle(
        supervisor=mock_supervisor,
        catalog=catalog,
        active_ports=active_ports,
    )

    assert "idle-model" in reaped
    assert "idle-model" not in active_ports
    mock_supervisor.stop_model.assert_called_once_with("idle-model")


@pytest.mark.asyncio
async def test_wake_on_demand():
    coord = ScaleToZeroCoordinator()

    cfg = ModelConfig(
        name="autoscale-model",
        model="fake/model",
        engine=EngineType.VLLM,
        port=8003,
        auto_scale=True,
    )

    catalog = {"autoscale-model": cfg}
    active_ports = {}

    mock_supervisor = MagicMock()
    mock_supervisor.start_model.return_value = MagicMock(pid=12345)

    with pytest.MonkeyPatch.context() as mp:
        # Mock health check to immediately return True
        mp.setattr("inferops.core.health.HealthChecker.check", lambda self: True)

        port = await coord.wake_on_demand(
            model_name="autoscale-model",
            supervisor=mock_supervisor,
            catalog=catalog,
            active_ports=active_ports,
            timeout_seconds=5.0,
        )

        assert port == 8003
        assert "autoscale-model" in active_ports
        assert active_ports["autoscale-model"] == 8003
        mock_supervisor.start_model.assert_called_once()
