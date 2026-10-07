"""Dynamic OpenAI-compatible reverse proxy and model router."""

from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from inferops.core.config import ModelConfig


class ModelGatewayRouter:
    """Dynamic reverse proxy directing /v1/chat/completions to model ports."""

    def __init__(self, catalog: Dict[str, ModelConfig], active_ports: Dict[str, int]) -> None:
        self.catalog = catalog
        self.active_ports = active_ports  # model_name or alias -> port
        self.client = httpx.AsyncClient(timeout=300.0)

    def update_routes(self, catalog: Dict[str, ModelConfig], active_ports: Dict[str, int]) -> None:
        self.catalog = catalog
        self.active_ports = active_ports

    def resolve_target_port(self, requested_model: str) -> Optional[int]:
        """Match requested model id with active port, handling aliases."""
        if requested_model in self.active_ports:
            return self.active_ports[requested_model]

        # Check public aliases
        for name, config in self.catalog.items():
            if config.public_alias == requested_model:
                if name in self.active_ports:
                    return self.active_ports[name]

        return None

    def create_fastapi_app(self) -> FastAPI:
        """Create the FastAPI application implementing the OpenAI-compatible proxy."""
        app = FastAPI(title="InferOps Unified Gateway", version="0.1.0")

        @app.get("/health")
        async def health() -> Dict[str, Any]:
            return {"status": "ok", "active_models": list(self.active_ports.keys())}

        @app.get("/v1/models")
        async def list_models() -> Dict[str, Any]:
            models_list = []
            for name, port in self.active_ports.items():
                cfg = self.catalog.get(name)
                alias = cfg.public_alias if cfg else name
                models_list.append(
                    {
                        "id": alias,
                        "object": "model",
                        "created": 1728000000,
                        "owned_by": "inferops",
                        "permission": [],
                        "root": cfg.model if cfg else name,
                        "parent": None,
                    }
                )
            return {"object": "list", "data": models_list}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request) -> Response:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON payload")

            requested_model = body.get("model")
            if not requested_model:
                raise HTTPException(status_code=400, detail="'model' field is required")

            port = self.resolve_target_port(requested_model)
            if not port:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{requested_model}' is not currently running. Active: {list(self.active_ports.keys())}",
                )

            target_url = f"http://127.0.0.1:{port}/v1/chat/completions"
            stream = body.get("stream", False)

            if stream:
                async def stream_generator():
                    async with httpx.AsyncClient(timeout=300.0) as client:
                        async with client.stream("POST", target_url, json=body, headers={"Content-Type": "application/json"}) as resp:
                            async for chunk in resp.aiter_bytes():
                                yield chunk

                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            # Non-streaming forward
            try:
                resp = await self.client.post(
                    target_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Inference engine proxy error: {e}")

        return app
