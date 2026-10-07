# Contributing to InferOps

Thank you for your interest in improving InferOps. This document outlines how to set up your development environment, run test suites, and submit pull requests.

## Development Setup

InferOps requires Python 3.10 or newer.

1. Clone the repository:
```bash
git clone https://github.com/alexandrmotologa/inferops.git
cd inferops
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install editable dependencies including development tools:
```bash
pip install -e ".[dev]"
```

## Running Tests and Linting

Before pushing changes or submitting a PR, verify that tests pass and the code matches style checks:

```bash
# Run the test suite
pytest

# Check code coverage
pytest --cov=inferops

# Lint with Ruff
ruff check .

# Type checking with mypy
mypy src/inferops
```

## Architecture Guidelines

- **Decoupled Engine Adapters**: All inference engine logic (vLLM, SGLang, or future backends) must inherit from `inferops.engines.base.EngineAdapter` and stay isolated from CLI and supervisor logic.
- **Cross-Platform Safety**: Do not make assumptions about POSIX-only APIs. Process handling must support Linux, macOS, WSL2, and Windows via `inferops.core.supervisor`.
- **Pure Functions in VRAM Sizing**: Memory estimation formulas in `inferops.core.vram_calculator` should remain pure and deterministic so they can be unit-tested without physical GPUs.

## Submitting Pull Requests

1. Create a feature branch from `main`:
```bash
git checkout -b feature/my-feature-name
```
2. Write clean commit messages following Conventional Commits (e.g. `feat(supervisor): add auto-restart on OOM`).
3. Include unit tests covering any new functions or bug fixes.
4. Open a pull request against `main`.
