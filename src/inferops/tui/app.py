"""Interactive Terminal User Interface (TUI) for InferOps using Textual."""

import asyncio
from pathlib import Path
from typing import Dict, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Log, ProgressBar, Static

from inferops.core.config import ModelConfig, discover_models
from inferops.core.supervisor import ModelLifecycleStatus, ProcessSupervisor
from inferops.hardware.gpu import GPUDeviceInfo, get_gpu_devices
from inferops.hardware.metrics import fetch_model_metrics


class GpuWidget(Static):
    """Widget displaying GPU utilization and memory bars."""

    def update_gpus(self, gpus: list[GPUDeviceInfo]) -> None:
        if not gpus:
            self.update("[dim]No physical NVIDIA GPU detected (Host/Emulation mode)[/dim]")
            return

        lines = ["[bold cyan]GPU Devices & VRAM Topology[/bold cyan]\n"]
        for g in gpus:
            pct = round((g.used_memory_gb / g.total_memory_gb * 100), 1) if g.total_memory_gb > 0 else 0
            temp_str = f"{g.temperature_c}°C" if g.temperature_c is not None else "N/A"
            lines.append(
                f"[bold]GPU {g.index}:[/bold] {g.name} | [yellow]{temp_str}[/yellow] | Load: [green]{g.utilization_gpu_pct}%[/green]"
            )
            lines.append(
                f"VRAM: [cyan]{g.used_memory_gb:.1f}[/cyan] / [bold]{g.total_memory_gb:.1f} GB[/bold] ({pct}% utilized)\n"
            )
        self.update("\n".join(lines))


class MetricsWidget(Static):
    """Widget displaying active model serving telemetry."""

    def update_telemetry(self, stats: dict[str, str]) -> None:
        lines = ["[bold cyan]Real-time Model Telemetry[/bold cyan]\n"]
        for k, v in stats.items():
            lines.append(f"[bold]{k}:[/bold] {v}")
        self.update("\n".join(lines))


class InferOpsTUI(App[None]):
    """Textual Terminal Dashboard for InferOps."""

    CSS = """
    Screen {
        background: #090d16;
        color: #f1f5f9;
    }
    #main-container {
        layout: horizontal;
        height: 1fr;
    }
    #left-pane {
        width: 45%;
        border-right: solid #23314d;
        padding: 1;
    }
    #right-pane {
        width: 55%;
        padding: 1;
    }
    #gpu-box {
        background: #111726;
        border: solid #23314d;
        padding: 1;
        margin-bottom: 1;
        height: 35%;
    }
    #metrics-box {
        background: #111726;
        border: solid #23314d;
        padding: 1;
        margin-bottom: 1;
        height: 25%;
    }
    #log-viewer {
        background: #111726;
        border: solid #23314d;
        height: 40%;
    }
    DataTable {
        height: 1fr;
        background: #111726;
        border: solid #23314d;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_data", "Refresh"),
        ("s", "start_selected", "Start Model"),
        ("x", "stop_selected", "Stop Model"),
    ]

    selected_model: reactive[Optional[str]] = reactive(None)

    def __init__(self, workspace_dir: Path) -> None:
        super().__init__()
        self.workspace_dir = workspace_dir
        self.models_dir = workspace_dir / "configs" / "models"
        self.supervisor = ProcessSupervisor(workspace_dir / "runtime")
        self.catalog: Dict[str, ModelConfig] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main-container"):
            with Vertical(id="left-pane"):
                yield Static("[bold cyan]Configured Inference Models[/bold cyan]\n")
                yield DataTable(id="models-table")
            with Vertical(id="right-pane"):
                yield GpuWidget(id="gpu-box")
                yield MetricsWidget(id="metrics-box")
                yield Log(id="log-viewer")
        yield Footer()

    async def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("Name", "Engine", "Port", "GPUs", "Status")
        await self.refresh_fleet()
        self.set_interval(2.0, self.poll_telemetry)

    async def refresh_fleet(self) -> None:
        self.catalog = discover_models(self.models_dir)
        table = self.query_one(DataTable)
        table.clear()

        for name, cfg in self.catalog.items():
            status = await self.supervisor.get_status(cfg)
            table.add_row(
                cfg.name,
                cfg.engine.value.upper(),
                str(cfg.port),
                cfg.cuda_visible_devices or "all",
                status.value,
                key=cfg.name,
            )

        gpu_box = self.query_one(GpuWidget)
        gpu_box.update_gpus(get_gpu_devices())

    async def poll_telemetry(self) -> None:
        table = self.query_one(DataTable)
        for name, cfg in self.catalog.items():
            status = await self.supervisor.get_status(cfg)
            try:
                table.update_cell(name, "Status", status.value)
            except Exception:
                pass

        gpu_box = self.query_one(GpuWidget)
        gpu_box.update_gpus(get_gpu_devices())

        metrics_box = self.query_one(MetricsWidget)
        metrics_box.update_telemetry(
            {
                "Managed Models": str(len(self.catalog)),
                "Engines": "vLLM, SGLang",
                "Unified Gateway": "http://127.0.0.1:8000/v1",
                "Web UI": "http://127.0.0.1:8900",
            }
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.row_key and event.row_key.value:
            self.selected_model = str(event.row_key.value)
            log = self.query_one(Log)
            log.write_line(f"Selected model: {self.selected_model}")

    async def action_start_selected(self) -> None:
        if not self.selected_model or self.selected_model not in self.catalog:
            return
        cfg = self.catalog[self.selected_model]
        log = self.query_one(Log)
        log.write_line(f"Spawning {cfg.name} ({cfg.engine.value})...")
        try:
            rec = await self.supervisor.start_model(cfg, wait=False)
            log.write_line(f"Spawned {cfg.name} (PID: {rec.pid})")
            await self.refresh_fleet()
        except Exception as e:
            log.write_line(f"Start error: {e}")

    async def action_stop_selected(self) -> None:
        if not self.selected_model:
            return
        log = self.query_one(Log)
        log.write_line(f"Stopping {self.selected_model}...")
        stopped = await self.supervisor.stop_model(self.selected_model)
        log.write_line(f"Stopped {self.selected_model}: {stopped}")
        await self.refresh_fleet()

    async def action_refresh_data(self) -> None:
        await self.refresh_fleet()
