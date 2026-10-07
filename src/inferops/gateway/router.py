"""Dynamic OpenAI-compatible reverse proxy, model router, and cold-start gateway."""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from inferops.core.config import ModelConfig
from inferops.core.supervisor import ProcessSupervisor
from inferops.gateway.accounting import TokenAccountingManager
from inferops.gateway.scale_to_zero import ScaleToZeroCoordinator


class ModelGatewayRouter:
    """Dynamic reverse proxy directing /v1/chat/completions to model ports with scale-to-zero and token accounting."""

    def __init__(
        self,
        catalog: Dict[str, ModelConfig],
        active_ports: Dict[str, int],
        db_path: Optional[Path] = None,
        supervisor: Optional[ProcessSupervisor] = None,
        require_api_keys: bool = False,
    ) -> None:
        self.catalog = catalog
        self.active_ports = active_ports  # model_name -> port
        self.supervisor = supervisor
        self.require_api_keys = require_api_keys
        self.accounting = TokenAccountingManager(db_path or Path("runtime/usage.db"))
        self.scale_coordinator = ScaleToZeroCoordinator()
        self.client = httpx.AsyncClient(timeout=300.0)

        # Initialize last accessed timestamps for already active ports
        for name in self.active_ports:
            self.scale_coordinator.touch(name)

    def update_routes(self, catalog: Dict[str, ModelConfig], active_ports: Dict[str, int]) -> None:
        self.catalog = catalog
        self.active_ports = active_ports
        for name in self.active_ports:
            self.scale_coordinator.touch(name)

    def resolve_target_model_name(self, requested_model: str) -> Optional[str]:
        """Find the canonical model name matching request or public alias."""
        if requested_model in self.catalog:
            return requested_model
        for name, config in self.catalog.items():
            if config.public_alias == requested_model:
                return name
        return None

    def resolve_target_port(self, requested_model: str) -> Optional[int]:
        """Match requested model id with active port, handling aliases."""
        canonical_name = self.resolve_target_model_name(requested_model)
        if canonical_name and canonical_name in self.active_ports:
            return self.active_ports[canonical_name]

        if requested_model in self.active_ports:
            return self.active_ports[requested_model]

        return None

    def create_fastapi_app(self) -> FastAPI:
        """Create the FastAPI application implementing the OpenAI-compatible proxy."""
        app = FastAPI(title="InferOps Unified Gateway", version="0.2.0")

        @app.get("/health")
        async def health() -> Dict[str, Any]:
            return {
                "status": "ok",
                "active_models": list(self.active_ports.keys()),
                "total_catalog_models": len(self.catalog),
            }

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

        @app.get("/v1/analytics/usage")
        async def usage_statistics() -> Dict[str, Any]:
            return self.accounting.get_summary_statistics()

        @app.post("/v1/chat/completions")
        async def chat_completions(
            request: Request,
            authorization: Optional[str] = Header(None),
        ) -> Response:
            start_time = time.time()
            api_key_id: Optional[str] = None

            # Verify API key if required
            if self.require_api_keys:
                if not authorization or not authorization.startswith("Bearer "):
                    raise HTTPException(
                        status_code=401,
                        detail="API key required. Supply Authorization: Bearer sk-inferops-...",
                    )
                raw_token = authorization.split("Bearer ", 1)[1].strip()
                key_info = self.accounting.verify_api_key(raw_token)
                if not key_info:
                    raise HTTPException(status_code=403, detail="Invalid or revoked API key.")
                api_key_id = key_info.key_id

            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON payload")

            requested_model = body.get("model")
            if not requested_model:
                raise HTTPException(status_code=400, detail="'model' field is required")

            canonical_name = self.resolve_target_model_name(requested_model) or requested_model
            port = self.resolve_target_port(requested_model)

            # Cold-start check: If model is not running, can we scale from zero?
            if not port and self.supervisor and canonical_name in self.catalog:
                cfg = self.catalog[canonical_name]
                if cfg.auto_scale:
                    port = await self.scale_coordinator.wake_on_demand(
                        model_name=canonical_name,
                        supervisor=self.supervisor,
                        catalog=self.catalog,
                        active_ports=self.active_ports,
                        timeout_seconds=90.0,
                    )

            if not port:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{requested_model}' is not currently running. Active: {list(self.active_ports.keys())}",
                )

            # Mark activity for scale-to-zero reaper
            self.scale_coordinator.touch(canonical_name)

            target_url = f"http://127.0.0.1:{port}/v1/chat/completions"
            stream = body.get("stream", False)

            if stream:
                async def stream_generator():
                    prompt_len = sum(len(m.get("content", "")) for m in body.get("messages", []))
                    approx_prompt_tokens = max(1, prompt_len // 4)
                    completion_tokens = 0

                    async with httpx.AsyncClient(timeout=300.0) as client:
                        async with client.stream(
                            "POST",
                            target_url,
                            json=body,
                            headers={"Content-Type": "application/json"},
                        ) as resp:
                            async for chunk in resp.aiter_bytes():
                                text_chunk = chunk.decode("utf-8", errors="ignore")
                                if "content" in text_chunk:
                                    completion_tokens += 1
                                yield chunk

                    # Record approximate usage on stream end
                    latency_ms = (time.time() - start_time) * 1000.0
                    self.accounting.record_usage(
                        model=requested_model,
                        prompt_tokens=approx_prompt_tokens,
                        completion_tokens=max(1, completion_tokens),
                        latency_ms=latency_ms,
                        api_key_id=api_key_id,
                    )

                return StreamingResponse(stream_generator(), media_type="text/event-stream")

            # Non-streaming forward
            try:
                resp = await self.client.post(
                    target_url,
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
                latency_ms = (time.time() - start_time) * 1000.0

                # Extract usage from engine response
                prompt_tokens = 0
                completion_tokens = 0
                try:
                    data = resp.json()
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                except Exception:
                    pass

                if prompt_tokens == 0:
                    prompt_len = sum(len(m.get("content", "")) for m in body.get("messages", []))
                    prompt_tokens = max(1, prompt_len // 4)
                if completion_tokens == 0:
                    completion_tokens = max(1, len(resp.content) // 16)

                self.accounting.record_usage(
                    model=requested_model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    api_key_id=api_key_id,
                )

                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    media_type="application/json",
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Inference engine proxy error: {e}")

        return app
