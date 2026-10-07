"""Unit tests for the Web Dashboard API server."""

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from inferops.core.config import ModelConfig, save_model_config
from inferops.web.server import create_web_app


@pytest.mark.asyncio
async def test_web_api_endpoints(tmp_path: Path) -> None:
    models_dir = tmp_path / "configs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        name="web-test-model",
        model="Qwen/Qwen2.5-7B",
        port=8001,
    )
    save_model_config(cfg, models_dir / "web-test.yaml")

    app = create_web_app(tmp_path)
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. GPUs endpoint
        resp_gpus = await client.get("/api/gpus")
        assert resp_gpus.status_code == 200
        assert isinstance(resp_gpus.json(), list)

        # 2. Models endpoint
        resp_models = await client.get("/api/models")
        assert resp_models.status_code == 200
        models_data = resp_models.json()
        assert len(models_data) == 1
        assert models_data[0]["name"] == "web-test-model"
        assert models_data[0]["status"] == "STOPPED"

        # 3. VRAM estimate endpoint
        resp_vram = await client.post(
            "/api/vram/estimate",
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "context_length": 8192,
                "dtype": "auto",
                "tensor_parallel_size": 1,
            },
        )
        assert resp_vram.status_code == 200
        vram_data = resp_vram.json()
        assert vram_data["params_b"] > 7.0
        assert "weights_gb" in vram_data
        assert "fits" in vram_data
