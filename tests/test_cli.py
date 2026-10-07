"""Unit tests for the Typer CLI commands."""

from pathlib import Path

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


def test_cli_tune() -> None:
    res = runner.invoke(app, ["tune", "Qwen/Qwen2.5-7B"])
    assert res.exit_code == 0
    assert "Hardware Tuning Recommendation" in res.stdout
    assert "Tensor Parallel Size" in res.stdout


def test_cli_model_create(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Init first
    runner.invoke(app, ["init"])
    res = runner.invoke(
        app,
        [
            "model",
            "create",
            "--name",
            "custom-m",
            "--model",
            "org/custom-model",
            "--port",
            "8099",
            "--engine",
            "vllm",
        ],
    )
    assert res.exit_code == 0
    assert "[OK] Model configuration saved" in res.stdout
    assert (tmp_path / "configs" / "models" / "custom-m.yaml").is_file()


def test_cli_logs_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(app, ["logs", "non-existent-model"])
    assert res.exit_code == 0
    assert "No log file found" in res.stdout
