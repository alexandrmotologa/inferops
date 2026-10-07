"""Unit tests for the dynamic OpenAI gateway router."""

import pytest
from httpx import ASGITransport, AsyncClient

from inferops.core.config import ModelConfig
from inferops.gateway.litellm_export import generate_litellm_config
from inferops.gateway.router import ModelGatewayRouter


@pytest.mark.asyncio
async def test_gateway_models_endpoint() -> None:
    catalog = {
        "qwen": ModelConfig(name="qwen", model="Qwen/Qwen2.5-7B", port=8001),
        "llama": ModelConfig(name="llama", model="meta-llama/Llama-3-8B", port=8002),
    }
    active_ports = {"qwen": 8001}

    router = ModelGatewayRouter(catalog, active_ports)
    app = router.create_fastapi_app()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "qwen"

        health_resp = await client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "ok"


def test_litellm_export_generation() -> None:
    catalog = {
        "qwen": ModelConfig(name="qwen", model="Qwen/Qwen2.5-7B", port=8001),
    }
    active_ports = {"qwen": 8001}
    conf = generate_litellm_config(catalog, active_ports)

    assert "model_list" in conf
    assert len(conf["model_list"]) == 1
    assert conf["model_list"][0]["model_name"] == "qwen"
    assert "8001" in conf["model_list"][0]["litellm_params"]["api_base"]
