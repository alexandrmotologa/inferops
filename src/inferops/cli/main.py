"""InferOps main CLI dispatcher with rich formatted terminal outputs."""

import asyncio
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
from inferops.core.benchmark import run_model_benchmark
from inferops.core.config import (
    EngineType,
    ModelConfig,
    discover_models,
    load_model_config,
    load_profiles,
    save_model_config,
)
from inferops.core.hf_hub import pull_and_synthesize
from inferops.core.supervisor import ModelLifecycleStatus, ProcessSupervisor
from inferops.core.tuner import tune_model_for_hardware
from inferops.core.vram_calculator import calculate_vram_requirements
from inferops.export.docker import generate_docker_compose
from inferops.export.k8s import generate_kubernetes_manifest
from inferops.gateway.accounting import TokenAccountingManager
from inferops.gateway.router import ModelGatewayRouter
from inferops.hardware.gpu import get_gpu_devices
from inferops.hardware.topology import inspect_gpu_topology
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

export_app = typer.Typer(name="export", help="Export production deployment manifests")
app.add_typer(export_app, name="export")

key_app = typer.Typer(name="key", help="Manage API access keys and rate limits")
app.add_typer(key_app, name="key")

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
def doctor(
    deep: bool = typer.Option(False, "--deep", help="Run multi-GPU interconnect topology diagnostics"),
) -> None:
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

    # 4. Deep Interconnect Topology
    if deep:
        console.print("\n[bold cyan]Probing Multi-GPU Interconnect Matrix...[/bold cyan]")
        topo_report = inspect_gpu_topology()
        if topo_report.available and topo_report.links:
            table = Table(title="GPU Interconnect Links", border_style="cyan")
            table.add_column("Pair", style="bold")
            table.add_column("Link Code", style="magenta")
            table.add_column("Description", style="cyan")
            table.add_column("High-Speed", justify="center")

            for link in topo_report.links:
                hs_style = "[green]YES[/green]" if link.is_high_speed else "[yellow]NO (Bottleneck)[/yellow]"
                table.add_row(
                    f"GPU {link.gpu_a} <-> GPU {link.gpu_b}",
                    link.link_type,
                    link.description,
                    hs_style,
                )
            console.print(table)
        for rec in topo_report.recommendations:
            console.print(f"[cyan]Recommendation:[/cyan] {rec}")

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
    auth: bool = typer.Option(False, "--auth", help="Enforce API key authentication"),
) -> None:
    """Start the dynamic OpenAI-compatible gateway reverse proxy with scale-to-zero and token accounting."""
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
    router = ModelGatewayRouter(
        catalog=catalog,
        active_ports=active_ports,
        db_path=ws / "runtime" / "usage.db",
        supervisor=supervisor,
        require_api_keys=auth,
    )
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


@app.command()
def logs(
    model_name: str = typer.Argument(..., help="Model name to display logs for"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of trailing lines to view"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output in real-time"),
) -> None:
    """Display or follow the stdout/stderr log output of a model."""
    ws = get_workspace()
    log_file = ws / "runtime" / "logs" / f"{model_name}.log"
    if not log_file.is_file():
        console.print(f"[yellow]No log file found for '{model_name}' at {log_file}[/yellow]")
        return

    import time as pytime
    try:
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                console.print(line, end="")

            if follow:
                console.print("[dim]-- Following log stream (Ctrl+C to exit) --[/dim]")
                while True:
                    line = f.readline()
                    if line:
                        console.print(line, end="")
                    else:
                        pytime.sleep(0.5)
    except KeyboardInterrupt:
        pass


@app.command()
def tune(
    model_or_hf_id: str = typer.Argument(..., help="Hugging Face repo id or model name"),
    engine: str = typer.Option("vllm", "--engine", "-e", help="Target engine: vllm or sglang"),
    save: bool = typer.Option(False, "--save", "-s", help="Save tuned model configuration to configs/models/"),
) -> None:
    """Auto-tune model configuration based on detected hardware and VRAM headroom."""
    eng_enum = EngineType.VLLM if engine.lower() == "vllm" else EngineType.SGLANG
    rec = tune_model_for_hardware(model_or_hf_id, target_engine=eng_enum)

    table = Table(title=f"Hardware Tuning Recommendation: {model_or_hf_id}", border_style="green")
    table.add_column("Parameter", style="cyan")
    table.add_column("Recommended Value", style="bold green")

    table.add_row("Model Name", rec.suggested_name)
    table.add_row("Engine", rec.suggested_engine.value.upper())
    table.add_row("Tensor Parallel Size", str(rec.tensor_parallel_size))
    table.add_row("Assigned GPUs", str(rec.gpus))
    table.add_row("GPU Memory Utilization", f"{rec.gpu_memory_utilization:.2f}")
    table.add_row("Max Model Len (Context)", f"{rec.max_model_len:,} tokens")
    table.add_row("Data Precision (DType)", rec.dtype)
    table.add_row("Quantization", rec.quantization or "None (Full 16-bit)")
    table.add_row("KV Cache DType", rec.kv_cache_dtype)
    table.add_section()
    table.add_row("Estimated VRAM / GPU", f"{rec.estimated_vram_per_gpu_gb:.2f} GB")
    table.add_row("Free VRAM Margin", f"{rec.hardware_headroom_gb:.2f} GB")

    console.print(table)
    for r in rec.rationale:
        console.print(f"  * {r}")

    if save:
        ws = get_workspace()
        target_path = ws / "configs" / "models" / f"{rec.suggested_name}.yaml"
        save_model_config(rec.to_model_config(), target_path)
        console.print(f"\n[bold green][OK] Saved configuration to {target_path}[/bold green]")


@app.command()
def benchmark(
    model_name: str = typer.Argument(..., help="Name of running model to benchmark"),
    requests: int = typer.Option(10, "--requests", "-r", help="Total requests to fire"),
    concurrency: int = typer.Option(2, "--concurrency", "-c", help="Concurrent workers"),
    max_tokens: int = typer.Option(64, "--tokens", "-t", help="Max tokens per response"),
) -> None:
    """Benchmark latency, TTFT, and generation throughput of an active model."""
    ws = get_workspace()
    catalog = discover_models(ws / "configs" / "models")
    if model_name not in catalog:
        console.print(f"[bold red]Model '{model_name}' not found in configs/models/[/bold red]")
        raise typer.Exit(1)

    cfg = catalog[model_name]
    console.print(
        f"Running benchmark on [bold cyan]{cfg.name}[/bold cyan] (Port: {cfg.port}) with {requests} requests (concurrency={concurrency})..."
    )

    result = asyncio.run(
        run_model_benchmark(
            host=cfg.host,
            port=cfg.port,
            model_alias=cfg.public_alias,
            num_requests=requests,
            concurrency=concurrency,
            max_tokens=max_tokens,
        )
    )

    table = Table(title=f"Benchmark Results: {cfg.name}", border_style="magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bold")

    table.add_row("Requests Completed", f"{result.successful_requests}/{result.num_requests}")
    table.add_row("Failed Requests", str(result.failed_requests))
    table.add_row("Total Generated Tokens", str(result.total_tokens_generated))
    table.add_row("Elapsed Time", f"{result.elapsed_time_sec:.2f}s")
    table.add_row("Generation Throughput", f"{result.tokens_per_second:.1f} tok/s")
    table.add_section()
    table.add_row("Average TTFT", f"{result.avg_ttft_ms:.1f} ms")
    table.add_row("p95 TTFT", f"{result.p95_ttft_ms:.1f} ms")
    table.add_row("Min TTFT", f"{result.min_ttft_ms:.1f} ms")
    table.add_row("Max TTFT", f"{result.max_ttft_ms:.1f} ms")

    console.print(table)


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


@model_app.command("create")
def model_create(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Model slug name"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Hugging Face repo or local path"),
    engine: str = typer.Option("vllm", "--engine", "-e", help="Inference engine: vllm or sglang"),
    port: int = typer.Option(8001, "--port", "-p", help="Port number"),
    gpus: str = typer.Option("0", "--gpus", "-g", help="GPU indices, e.g. '0' or '0,1'"),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size"),
    context: int = typer.Option(8192, "--context", "-c", help="Max sequence length"),
    quant: Optional[str] = typer.Option(None, "--quant", "-q", help="Quantization format"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run interactive guided setup"),
) -> None:
    """Create a new model configuration YAML file."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if interactive or not name or not model:
        console.print("[bold cyan]Guided Model Setup Wizard[/bold cyan]\n")
        if not name:
            name = typer.prompt("Model name slug (e.g. mistral-7b)")
        if not model:
            model = typer.prompt("Hugging Face model ID or path (e.g. mistralai/Mistral-7B-Instruct-v0.3)")
        engine_choice = typer.prompt("Engine (vllm or sglang)", default=engine)
        engine = engine_choice
        port = int(typer.prompt("Port", default=str(port)))
        gpus = typer.prompt("GPU indices (e.g. 0 or 0,1)", default=gpus)

    assert name is not None
    assert model is not None

    eng_enum = EngineType.VLLM if engine.lower() == "vllm" else EngineType.SGLANG
    gpu_list = [int(x.strip()) for x in gpus.split(",") if x.strip().isdigit()] or [0]

    cfg = ModelConfig(
        name=name,
        model=model,
        engine=eng_enum,
        served_model_name=name,
        port=port,
        gpus=gpu_list,
        tensor_parallel_size=tp,
        max_model_len=context,
        quantization=quant,
    )

    target_path = models_dir / f"{cfg.name}.yaml"
    save_model_config(cfg, target_path)
    console.print(f"[bold green][OK] Model configuration saved to {target_path}[/bold green]")


@app.command("pull")
def pull_model(
    repo_id: str = typer.Argument(..., help="Hugging Face model ID (e.g. meta-llama/Llama-3.1-8B-Instruct)"),
    port: int = typer.Option(8001, "--port", "-p", help="Port number for this model instance"),
    engine: str = typer.Option("vllm", "--engine", "-e", help="Inference engine: vllm or sglang"),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Optional Hugging Face access token"),
    download: bool = typer.Option(False, "--download", "-d", help="Download model weights via huggingface-cli"),
) -> None:
    """Fetch model architecture from HuggingFace, synthesize YAML manifest, and run VRAM check."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    eng = EngineType.VLLM if engine.lower() == "vllm" else EngineType.SGLANG

    console.print(f"[bold cyan]Fetching HuggingFace metadata for '{repo_id}'...[/bold cyan]")
    try:
        cfg, manifest_path, vram_result = pull_and_synthesize(
            repo_id=repo_id,
            output_dir=models_dir,
            token=token,
            engine=eng,
            port=port,
        )
        console.print(f"[bold green][OK] Configuration synthesized and saved to {manifest_path}[/bold green]")

        table = Table(title=f"Predictive VRAM Analysis: {cfg.name}", border_style="cyan")
        table.add_column("Component", style="bold")
        table.add_column("Memory (GB)", justify="right")
        table.add_row("Model Weights", f"{vram_result.weights_vram_gb:.2f} GB")
        table.add_row("KV Cache", f"{vram_result.kv_cache_vram_gb:.2f} GB")
        table.add_row("CUDA Overhead", f"{vram_result.cuda_overhead_gb:.2f} GB")
        table.add_row("[bold]Required Per GPU[/bold]", f"[bold]{vram_result.vram_per_gpu_gb:.2f} GB[/bold]")
        console.print(table)

        verdict_style = "bold green" if vram_result.fits else "bold red"
        console.print(Panel(f"[{verdict_style}]{vram_result.suggestion}[/{verdict_style}]", title="VRAM Sizer Verdict"))

        if download:
            hf_cli = shutil.which("huggingface-cli")
            if not hf_cli:
                console.print("[yellow]huggingface-cli not found on PATH. Install huggingface_hub to download weights.[/yellow]")
            else:
                import subprocess
                console.print("[bold cyan]Downloading model weights using huggingface-cli...[/bold cyan]")
                cmd = [hf_cli, "download", repo_id]
                if token:
                    cmd.extend(["--token", token])
                subprocess.run(cmd, check=True)
                console.print("[bold green][OK] Weights download complete.[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Failed to pull model: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("usage")
def usage_stats() -> None:
    """Display token consumption accounting and commercial cost savings."""
    ws = get_workspace()
    db_path = ws / "runtime" / "usage.db"
    accounting = TokenAccountingManager(db_path)
    summary = accounting.get_summary_statistics()

    table = Table(title="InferOps Token Accounting & Cost Savings", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Total Invocations", f"{summary['total_requests']:,}")
    table.add_row("Prompt Tokens", f"{summary['total_prompt_tokens']:,}")
    table.add_row("Completion Tokens", f"{summary['total_completion_tokens']:,}")
    table.add_row("Grand Total Tokens", f"{summary['grand_total_tokens']:,}")
    table.add_row("Average Latency", f"{summary['avg_latency_ms']:.1f} ms")
    table.add_row("[bold green]Commercial Cost Saved[/bold green]", f"[bold green]${summary['total_savings_usd']:.4f}[/bold green]")
    console.print(table)

    if summary["by_model"]:
        m_table = Table(title="Usage Breakdown by Model", border_style="magenta")
        m_table.add_column("Model Identifier", style="bold")
        m_table.add_column("Requests", justify="right")
        m_table.add_column("Tokens", justify="right")
        m_table.add_column("Commercial Savings", justify="right", style="green")

        for m in summary["by_model"]:
            m_table.add_row(m["model"], f"{m['requests']:,}", f"{m['tokens']:,}", f"${m['savings']:.4f}")
        console.print(m_table)


@key_app.command("create")
def key_create(
    name: str = typer.Argument(..., help="Name or service identity for this API key"),
    rpm: int = typer.Option(60, "--rpm", help="Rate limit requests per minute"),
) -> None:
    """Generate a new secure API key with rate limits."""
    ws = get_workspace()
    accounting = TokenAccountingManager(ws / "runtime" / "usage.db")
    raw_key, info = accounting.create_api_key(name, rate_limit_rpm=rpm)

    console.print(
        Panel.fit(
            f"[bold green]API Key Generated Successfully![/bold green]\n\n"
            f"Key Token: [bold cyan]{raw_key}[/bold cyan]\n"
            f"Key ID:    {info.key_id}\n"
            f"Identity:  {info.name}\n"
            f"Rate Limit:{info.rate_limit_rpm} req/min\n\n"
            f"[yellow]Store this token securely; it will not be displayed again.[/yellow]",
            title="Access Key Created",
        )
    )


@key_app.command("list")
def key_list() -> None:
    """List all registered API keys."""
    ws = get_workspace()
    accounting = TokenAccountingManager(ws / "runtime" / "usage.db")
    keys = accounting.list_api_keys()

    if not keys:
        console.print("[yellow]No API keys generated yet. Run [bold]inferops key create <name>[/bold].[/yellow]")
        return

    table = Table(title="Registered API Keys", border_style="cyan")
    table.add_column("Key ID", style="bold")
    table.add_column("Name / Identity", style="cyan")
    table.add_column("RPM Limit", justify="right")
    table.add_column("Status", justify="center")

    for k in keys:
        status_str = "[red]REVOKED[/red]" if k.revoked else "[green]ACTIVE[/green]"
        table.add_row(k.key_id, k.name, str(k.rate_limit_rpm), status_str)
    console.print(table)


@key_app.command("revoke")
def key_revoke(
    key_id: str = typer.Argument(..., help="Key ID to revoke (e.g. key_...)"),
) -> None:
    """Revoke an active API key."""
    ws = get_workspace()
    accounting = TokenAccountingManager(ws / "runtime" / "usage.db")
    revoked = accounting.revoke_api_key(key_id)
    if revoked:
        console.print(f"[bold green][OK] API key '{key_id}' has been revoked.[/bold green]")
    else:
        console.print(f"[bold red]Key ID '{key_id}' not found.[/bold red]")


@export_app.command("docker-compose")
def export_docker(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File to write docker-compose.yml to"),
    profile: Optional[str] = typer.Option(None, "--profile", "-p", help="Filter models by profile"),
) -> None:
    """Generate production-ready docker-compose.yml with NVIDIA GPU container passthrough."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    catalog = discover_models(models_dir)

    models_to_export = []
    if profile:
        profiles = load_profiles(ws / "configs" / "profiles.yaml")
        if profile not in profiles:
            console.print(f"[bold red]Profile '{profile}' not found.[/bold red]")
            raise typer.Exit(1)
        for name in profiles[profile]:
            if name in catalog:
                models_to_export.append(catalog[name])
    else:
        models_to_export = list(catalog.values())

    if not models_to_export:
        console.print("[yellow]No models found to export.[/yellow]")
        return

    compose_yaml = generate_docker_compose(models_to_export)
    if output:
        output.write_text(compose_yaml, encoding="utf-8")
        console.print(f"[bold green][OK] Docker compose written to {output}[/bold green]")
    else:
        console.print(compose_yaml)


@export_app.command("k8s")
def export_k8s(
    model_name: str = typer.Argument(..., help="Name of model from catalog to export manifests for"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="File to write manifests to"),
    namespace: str = typer.Option("inferops", "--namespace", "-n", help="Target Kubernetes namespace"),
) -> None:
    """Generate production Kubernetes Deployment & Service YAML manifests with GPU limits."""
    ws = get_workspace()
    models_dir = ws / "configs" / "models"
    catalog = discover_models(models_dir)

    if model_name not in catalog:
        console.print(f"[bold red]Model '{model_name}' not found in catalog.[/bold red]")
        raise typer.Exit(1)

    k8s_yaml = generate_kubernetes_manifest(catalog[model_name], namespace=namespace)
    if output:
        output.write_text(k8s_yaml, encoding="utf-8")
        console.print(f"[bold green][OK] Kubernetes manifests written to {output}[/bold green]")
    else:
        console.print(k8s_yaml)


if __name__ == "__main__":
    app()
