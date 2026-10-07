"""Unit tests for the Typer CLI commands."""

from typer.testing import CliRunner
from inferops.cli.main import app

runner = CliRunner()


def test_cli_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "InferOps version" in res.stdout


def test_cli_doctor() -> None:
    res = runner.invoke(app, ["doctor"])
    assert res.exit_code == 0
    assert "Running InferOps Environment Diagnostics" in res.stdout


def test_cli_vram_calculation() -> None:
    res = runner.invoke(app, ["vram", "Qwen/Qwen2.5-7B", "--dtype", "fp8"])
    assert res.exit_code == 0
    assert "VRAM Pre-flight Analysis" in res.stdout
    assert "Model Weights" in res.stdout
    assert "Required Per GPU" in res.stdout
