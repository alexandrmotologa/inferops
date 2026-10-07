"""InferOps main CLI dispatcher with rich formatted terminal outputs."""

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from inferops import __version__
from inferops.core.config import (
    EngineType,
    ModelConfig,
    discover_models,
    load_model_config,
    load_profiles,
    save_model_config,
)
from inferops.core.supervisor import ModelLifecycleStatus, ProcessSupervisor
from inferops.core.vram_calculator import calculate_vram_requirements
from inferops.gateway.router import ModelGatewayRouter
from inferops.hardware.gpu import get_gpu_devices
from inferops.web.server import create_web_app

# Ensure proper utf-8 output on Windows consoles
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = typer.Typer(
    name="inferops",
    help="InferOps: Modern Bare-Metal LLM Control Plane & Multi-Engine Orchestrator (vLLM & SGLang)",
    no_args_is_help=True,
)
model_app = typer.Typer(name="model", help="Manage declarative model specifications")
app.add_typer(model_app, name="model")

console = Console(safe_box=True)


def get_workspace() -> Path:
    return Path.cwd()


@app.command()
def version() -> None:
    """Print the installed InferOps version."""
    console.print(f"[bold cyan]InferOps[/bold cyan] version [green]{__version__}[/green]")


@app.command()
def init() -> None:
    """Initialize an InferOps workspace with directory structure and sample configs."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    profiles_file = ws / "configs" / "profiles.yaml"
    runtime_logs = ws / "runtime" / "logs"
    runtime_pids = ws / "runtime" / "pids"
    env_file = ws / ".env"

    models_dir.mkdir(parents=True, exist_ok=True)
    runtime_logs.mkdir(parents=True, exist_ok=True)
    runtime_pids.mkdir(parents=True, exist_ok=True)

    # Sample model 1: Qwen 2.5 Coder 7B (vLLM)
    qwen_cfg = ModelConfig(
        name="qwen2.5-coder-7b",
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        engine=EngineType.VLLM,
        served_model_name="qwen2.5-coder-7b",
        port=8001,
        gpus=[0],
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        dtype="auto",
    )
    qwen_path = models_dir / "qwen2.5-coder-7b.yaml"
    if not qwen_path.exists():
        save_model_config(qwen_cfg, qwen_path)

    # Sample model 2: Llama 3.1 8B (SGLang)
    llama_cfg = ModelConfig(
        name="llama3.1-8b",
        model="meta-llama/Llama-3.1-8B-Instruct",
        engine=EngineType.SGLANG,
        served_model_name="llama3.1-8b",
        port=8002,
        gpus=[0],
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
        dtype="auto",
    )
    llama_path = models_dir / "llama3.1-8b.yaml"
    if not llama_path.exists():
        save_model_config(llama_cfg, llama_path)

    # Profiles file
    if not profiles_file.exists():
        profiles_file.write_text(
            "profiles:\n  dev:\n    - qwen2.5-coder-7b\n  prod:\n    - llama3.1-8b\n",
            encoding="utf-8",
        )

    # Sample .env
    if not env_file.exists():
        env_file.write_text(
            "# InferOps Environment Configuration\nHF_TOKEN=\nVLLM_LOGGING_LEVEL=INFO\n",
            encoding="utf-8",
        )

    console.print(
        Panel.fit(
            f"[bold green]InferOps initialized successfully![/bold green]\n\n"
            f"Configs: [cyan]{models_dir}[/cyan]\n"
            f"Profiles: [cyan]{profiles_file}[/cyan]\n"
            f"Runtime: [cyan]{ws / 'runtime'}[/cyan]\n\n"
            f"Next: Check GPU memory with [bold yellow]inferops vram qwen2.5-coder-7b[/bold yellow]\n"
            f"Or start serving with [bold yellow]inferops start qwen2.5-coder-7b[/bold yellow]",
            title="Workspace Ready",
        )
    )


@app.command()
def status() -> None:
    """Display real-time lifecycle status of all configured models."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    catalog = discover_models(models_dir)
    supervisor = ProcessSupervisor(ws / "runtime")

    if not catalog:
        console.print("[yellow]No models found. Run [bold]inferops init[/bold] first.[/yellow]")
        return

    table = Table(title="InferOps Model Fleet Status", border_style="cyan")
    table.add_column("Model Name", style="bold")
    table.add_column("Engine", style="magenta")
    table.add_column("Port", justify="right")
    table.add_column("GPUs", justify="center")
    table.add_column("PID", justify="right")
    table.add_column("Status", justify="center")

    async def gather_statuses():
        for name, cfg in catalog.items():
            status = await supervisor.get_status(cfg)
            rec = supervisor.get_record(name)
            pid_str = str(rec.pid) if rec else "-"

            status_style = {
                ModelLifecycleStatus.HEALTHY: "[bold green]HEALTHY[/bold green]",
                ModelLifecycleStatus.STARTING: "[bold yellow]STARTING[/bold yellow]",
                ModelLifecycleStatus.STOPPED: "[dim]STOPPED[/dim]",
                ModelLifecycleStatus.CRASHED: "[bold red]CRASHED[/bold red]",
                ModelLifecycleStatus.UNHEALTHY: "[red]UNHEALTHY[/red]",
            }.get(status, str(status))

            table.add_row(
                cfg.name,
                cfg.engine.value.upper(),
                str(cfg.port),
                cfg.cuda_visible_devices or "all",
                pid_str,
                status_style,
            )

    asyncio.run(gather_statuses())
    console.print(table)


@app.command()
def start(
    model_name: Optional[str] = typer.Argument(None, help="Name of model to start"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Profile group to start"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Do not wait for health check"),
    timeout: float = typer.Option(180.0, "--timeout", "-t", help="Readiness timeout in seconds"),
) -> None:
    """Start one model or an entire profile group."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    catalog = discover_models(models_dir)
    supervisor = ProcessSupervisor(ws / "runtime")

    targets: list[ModelConfig] = []

    if profile:
        profiles = load_profiles(ws / "configs" / "profiles.yaml")
        if profile not in profiles:
            console.print(f"[bold red]Profile '{profile}' not found in profiles.yaml[/bold red]")
            raise typer.Exit(1)
        for m_name in profiles[profile]:
            if m_name in catalog:
                targets.append(catalog[m_name])
            else:
                console.print(f"[yellow]Warning: Model '{m_name}' in profile not found in catalog[/yellow]")
    elif model_name:
        if model_name not in catalog:
            console.print(f"[bold red]Model '{model_name}' not found in configs/models/[/bold red]")
            raise typer.Exit(1)
        targets.append(catalog[model_name])
    else:
        console.print("[red]Please specify either a model name or --profile.[/red]")
        raise typer.Exit(1)

    async def run_starts():
        for cfg in targets:
            console.print(
                f"Starting [bold cyan]{cfg.name}[/bold cyan] ({cfg.engine.value.upper()}) on port {cfg.port}..."
            )
            try:
                rec = await supervisor.start_model(cfg, wait=(not no_wait), timeout_seconds=timeout)
                console.print(
                    f"[bold green][OK][/bold green] [bold cyan]{cfg.name}[/bold cyan] is [green]{rec.status.value}[/green] (PID: {rec.pid})"
                )
            except Exception as e:
                console.print(f"[bold red][FAIL] Failed to start {cfg.name}: {e}[/bold red]")

    asyncio.run(run_starts())


@app.command()
def stop(
    model_name: Optional[str] = typer.Argument(None, help="Name of model to stop"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Profile group to stop"),
) -> None:
    """Stop a running model or an entire profile group."""
    ws = get_workspace()
    supervisor = ProcessSupervisor(ws / "runtime")

    names: list[str] = []
    if profile:
        profiles = load_profiles(ws / "configs" / "profiles.yaml")
        if profile in profiles:
            names.extend(profiles[profile])
    elif model_name:
        names.append(model_name)
    else:
        # Stop all running models
        catalog = discover_models(ws / "configs" / "models")
        names.extend(list(catalog.keys()))

    async def run_stops():
        for n in names:
            console.print(f"Stopping [bold cyan]{n}[/bold cyan]...")
            stopped = await supervisor.stop_model(n)
            if stopped:
                console.print(f"[bold green][OK][/bold green] {n} stopped.")
            else:
                console.print(f"[yellow]Could not stop {n} (might already be stopped)[/yellow]")

    asyncio.run(run_stops())


@app.command()
def vram(
    model_or_config: str = typer.Argument(..., help="Model identifier, HF repo, or YAML path"),
    context_length: int = typer.Option(8192, "--context", "-c", help="Max sequence length"),
    dtype: str = typer.Option("auto", "--dtype", help="Data precision (auto, fp16, bf16, fp8)"),
    quantization: Optional[str] = typer.Option(None, "--quant", "-q", help="Quantization (awq, gptq, fp8)"),
    tp: int = typer.Option(1, "--tp", help="Tensor parallelism degree"),
) -> None:
    """Run an analytical VRAM pre-flight check to verify if a model fits on your GPU."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"

    model_id = model_or_config
    # Check if user passed an existing model name in catalog
    catalog = discover_models(models_dir)
    if model_or_config in catalog:
        cfg = catalog[model_or_config]
        model_id = cfg.model
        if context_length == 8192 and cfg.max_model_len:
            context_length = cfg.max_model_len
        if dtype == "auto" and cfg.dtype:
            dtype = cfg.dtype
        if quantization is None and cfg.quantization:
            quantization = cfg.quantization
        if tp == 1 and cfg.tensor_parallel_size:
            tp = cfg.tensor_parallel_size
    elif Path(model_or_config).is_file():
        try:
            cfg = load_model_config(model_or_config)
            model_id = cfg.model
            if context_length == 8192 and cfg.max_model_len:
                context_length = cfg.max_model_len
            if dtype == "auto" and cfg.dtype:
                dtype = cfg.dtype
            if quantization is None and cfg.quantization:
                quantization = cfg.quantization
            if tp == 1 and cfg.tensor_parallel_size:
                tp = cfg.tensor_parallel_size
        except Exception:
            pass

    gpus = get_gpu_devices()
    avail_gb = gpus[0].total_memory_gb if gpus else None

    est = calculate_vram_requirements(
        model_name_or_path=model_id,
        context_length=context_length,
        dtype=dtype,
        quantization=quantization,
        tensor_parallel_size=tp,
        available_vram_per_gpu_gb=avail_gb,
    )

    table = Table(title=f"VRAM Pre-flight Analysis: {est.model_name}", border_style="blue")
    table.add_column("Component", style="cyan")
    table.add_column("Memory (GB)", justify="right")
    table.add_column("Details", style="dim")

    table.add_row("Model Weights", f"{est.weights_vram_gb:.2f} GB", f"{est.params_billions}B params @ {est.precision_bytes_per_param}B/param")
    table.add_row("KV Cache (16 req)", f"{est.kv_cache_vram_gb:.2f} GB", f"Context: {est.context_length} tokens")
    table.add_row("CUDA Runtime", f"{est.cuda_overhead_gb:.2f} GB", "Driver, kernels, graph memory")
    table.add_section()
    table.add_row("[bold]Required Per GPU[/bold]", f"[bold]{est.vram_per_gpu_gb:.2f} GB[/bold]", f"TP={est.tensor_parallel_size}")
    if est.available_vram_per_gpu_gb is not None:
        table.add_row("Detected GPU VRAM", f"{est.available_vram_per_gpu_gb:.2f} GB", f"GPU 0: {gpus[0].name}")

    console.print(table)

    verdict_style = "bold green" if est.fits else "bold red"
    verdict_text = "[FITS COMFORTABLY]" if est.fits else "[WILL EXCEED VRAM - OOM RISK]"
    console.print(Panel(f"[{verdict_style}]{verdict_text}[/{verdict_style}]\n{est.suggestion}", title="Verdict"))


@app.command()
def doctor() -> None:
    """Run comprehensive diagnostics for system, CUDA, GPUs, and engine runtimes."""
    console.print("[bold cyan]Running InferOps Environment Diagnostics...[/bold cyan]\n")

    # 1. OS & Python
    console.print(f"[bold]Python:[/bold] {sys.version.split()[0]} ({sys.platform})")

    # 2. Executable checks
    vllm_avail = shutil.which("vllm") is not None
    sglang_avail = shutil.which("sglang") is not None
    console.print(f"[bold]vLLM binary:[/bold] {'[green]Installed[/green]' if vllm_avail else '[yellow]Not on PATH (will use python module)[/yellow]'}")
    console.print(f"[bold]SGLang binary:[/bold] {'[green]Installed[/green]' if sglang_avail else '[yellow]Not on PATH (will use python module)[/yellow]'}")

    # 3. GPUs
    gpus = get_gpu_devices()
    if gpus:
        console.print(f"[bold green]Detected {len(gpus)} GPU(s):[/bold green]")
        for g in gpus:
            console.print(f"  * GPU {g.index}: [cyan]{g.name}[/cyan] - {g.free_memory_gb}/{g.total_memory_gb} GB Free ({g.utilization_gpu_pct}% load)")
    else:
        console.print("[yellow]No discrete NVIDIA GPU detected via nvidia-smi / NVML.[/yellow]")

    console.print("\n[bold green][OK] Diagnostic check completed.[/bold green]")


@app.command()
def web(
    port: int = typer.Option(8900, "--port", "-p", help="Port for web dashboard"),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host address"),
) -> None:
    """Start the InferOps Web Dashboard with interactive chat playground and GPU charts."""
    ws = get_workspace()
    app_instance = create_web_app(ws)
    console.print(
        Panel.fit(
            f"[bold green]InferOps Web Dashboard is running![/bold green]\n\n"
            f"Open: [bold cyan]http://{host}:{port}[/bold cyan]\n"
            f"Features: Live GPU Gauges, Model Fleet Controls, VRAM Sizer, and Chat Playground.\n\n"
            f"Press Ctrl+C to terminate.",
            title="Dashboard Active",
        )
    )
    uvicorn.run(app_instance, host=host, port=port, log_level="warning")


@app.command()
def tui() -> None:
    """Launch the interactive Textual terminal dashboard."""
    from inferops.tui.app import InferOpsTUI
    ws = get_workspace()
    tui_app = InferOpsTUI(ws)
    tui_app.run()


@app.command()
def proxy(
    port: int = typer.Option(8000, "--port", "-p", help="Port for the unified OpenAI API gateway"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host address"),
) -> None:
    """Start the dynamic OpenAI-compatible gateway reverse proxy."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    catalog = discover_models(models_dir)
    supervisor = ProcessSupervisor(ws / "runtime")

    async def get_active():
        active = {}
        for name, cfg in catalog.items():
            status = await supervisor.get_status(cfg)
            if status == ModelLifecycleStatus.HEALTHY:
                active[name] = cfg.port
        return active

    active_ports = asyncio.run(get_active())
    router = ModelGatewayRouter(catalog, active_ports)
    fastapi_proxy = router.create_fastapi_app()

    console.print(
        Panel.fit(
            f"[bold green]InferOps Unified Gateway active![/bold green]\n\n"
            f"Endpoint: [bold cyan]http://{host}:{port}/v1/chat/completions[/bold cyan]\n"
            f"Active Models Routed: [magenta]{list(active_ports.keys()) or 'None (start models first)'}[/magenta]\n\n"
            f"Connect standard OpenAI SDK clients or LangChain directly to this port.",
            title="Unified Proxy",
        )
    )
    uvicorn.run(fastapi_proxy, host=host, port=port, log_level="warning")


# Subcommands for model
@model_app.command("list")
def model_list() -> None:
    """List all declared models in the catalog."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    catalog = discover_models(models_dir)

    if not catalog:
        console.print("[yellow]Catalog is empty. Run [bold]inferops init[/bold] to scaffold models.[/yellow]")
        return

    table = Table(title="Model Definitions in Catalog", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Model Identifier", style="cyan")
    table.add_column("Engine", style="magenta")
    table.add_column("Port", justify="right")
    table.add_column("TP", justify="center")
    table.add_column("DType", justify="center")

    for name, cfg in catalog.items():
        table.add_row(
            cfg.name,
            cfg.model,
            cfg.engine.value.upper(),
            str(cfg.port),
            str(cfg.tensor_parallel_size),
            cfg.dtype,
        )
    console.print(table)


if __name__ == "__main__":
    app()
