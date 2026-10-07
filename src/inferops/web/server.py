"""Web Dashboard backend API server with WebSockets and Chat Playground."""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inferops.core.config import ModelConfig, discover_models, load_profiles
from inferops.core.supervisor import ModelLifecycleStatus, ProcessSupervisor
from inferops.core.vram_calculator import calculate_vram_requirements
from inferops.gateway.router import ModelGatewayRouter
from inferops.hardware.gpu import get_gpu_devices
from inferops.hardware.metrics import fetch_model_metrics


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = True


class VRAMCheckRequest(BaseModel):
    model: str
    context_length: int = 8192
    dtype: str = "auto"
    quantization: Optional[str] = None
    tensor_parallel_size: int = 1


def create_web_app(workspace_dir: Path) -> FastAPI:
    """Create the FastAPI app hosting both REST API, WebSockets, and Web UI static files."""
    app = FastAPI(title="InferOps Web Dashboard", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    models_dir = workspace_dir / "configs" / "models"
    runtime_dir = workspace_dir / "runtime"
    supervisor = ProcessSupervisor(runtime_dir)
    static_dir = Path(__file__).parent / "static"

    @app.get("/api/gpus")
    async def api_get_gpus() -> List[Dict[str, Any]]:
        gpus = get_gpu_devices()
        return [
            {
                "index": g.index,
                "name": g.name,
                "total_memory_gb": g.total_memory_gb,
                "free_memory_gb": g.free_memory_gb,
                "used_memory_gb": g.used_memory_gb,
                "utilization_gpu_pct": g.utilization_gpu_pct,
                "utilization_mem_pct": g.utilization_mem_pct,
                "temperature_c": g.temperature_c,
                "power_watts": g.power_watts,
            }
            for g in gpus
        ]

    @app.get("/api/models")
    async def api_list_models() -> List[Dict[str, Any]]:
        catalog = discover_models(models_dir)
        results = []
        for name, config in catalog.items():
            status = await supervisor.get_status(config)
            rec = supervisor.get_record(name)
            results.append(
                {
                    "name": config.name,
                    "model": config.model,
                    "engine": config.engine.value,
                    "port": config.port,
                    "gpus": config.cuda_visible_devices,
                    "status": status.value,
                    "pid": rec.pid if rec else None,
                    "uptime_seconds": round(time.time() - rec.started_at, 1) if rec else None,
                }
            )
        return results

    @app.post("/api/models/{name}/start")
    async def api_start_model(name: str) -> Dict[str, Any]:
        catalog = discover_models(models_dir)
        config = catalog.get(name)
        if not config:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")

        try:
            record = await supervisor.start_model(config, wait=False)
            return {"status": "starting", "pid": record.pid}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/models/{name}/stop")
    async def api_stop_model(name: str) -> Dict[str, Any]:
        stopped = await supervisor.stop_model(name)
        return {"status": "stopped" if stopped else "failed"}

    @app.post("/api/vram/estimate")
    async def api_vram_estimate(req: VRAMCheckRequest) -> Dict[str, Any]:
        gpus = get_gpu_devices()
        avail_gb = gpus[0].total_memory_gb if gpus else 24.0
        est = calculate_vram_requirements(
            model_name_or_path=req.model,
            context_length=req.context_length,
            dtype=req.dtype,
            quantization=req.quantization,
            tensor_parallel_size=req.tensor_parallel_size,
            available_vram_per_gpu_gb=avail_gb,
        )
        return {
            "model": est.model_name,
            "params_b": est.params_billions,
            "weights_gb": est.weights_vram_gb,
            "kv_cache_gb": est.kv_cache_vram_gb,
            "cuda_overhead_gb": est.cuda_overhead_gb,
            "total_gb": est.total_required_vram_gb,
            "per_gpu_gb": est.vram_per_gpu_gb,
            "fits": est.fits,
            "suggestion": est.suggestion,
        }

    @app.post("/api/chat")
    async def api_chat_playground(req: ChatRequest) -> Response:
        catalog = discover_models(models_dir)
        config = catalog.get(req.model)
        if not config:
            raise HTTPException(status_code=404, detail=f"Model '{req.model}' is not registered.")

        target_url = f"http://127.0.0.1:{config.port}/v1/chat/completions"
        payload = {
            "model": config.public_alias,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": req.stream,
        }

        if req.stream:
            async def event_generator():
                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream("POST", target_url, json=payload) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(target_url, json=payload)
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                catalog = discover_models(models_dir)
                models_data = []
                for name, cfg in catalog.items():
                    status = await supervisor.get_status(cfg)
                    metrics = None
                    if status == ModelLifecycleStatus.HEALTHY:
                        metrics = await fetch_model_metrics(cfg.host, cfg.port, cfg.metrics_endpoint)

                    models_data.append(
                        {
                            "name": name,
                            "status": status.value,
                            "port": cfg.port,
                            "throughput_tok_s": metrics.throughput_tokens_per_sec if metrics else 0.0,
                            "kv_cache_pct": metrics.gpu_cache_usage_pct if metrics else 0.0,
                            "requests_running": metrics.num_requests_running if metrics else 0,
                        }
                    )

                gpus_data = [
                    {
                        "index": g.index,
                        "name": g.name,
                        "used_gb": g.used_memory_gb,
                        "total_gb": g.total_memory_gb,
                        "util_gpu": g.utilization_gpu_pct,
                        "temp_c": g.temperature_c,
                    }
                    for g in get_gpu_devices()
                ]

                await websocket.send_json({"models": models_data, "gpus": gpus_data})
                await asyncio.sleep(1.5)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    # Mount static assets
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        index_file = static_dir / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Web dashboard index.html not found.")
        return FileResponse(str(index_file))

    return app
