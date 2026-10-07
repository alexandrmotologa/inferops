"""Custom typed exception hierarchy for InferOps."""

from typing import Optional


class InferOpsError(Exception):
    """Base exception for all InferOps errors."""
    pass


class ConfigurationError(InferOpsError):
    """Raised when configuration validation or file parsing fails."""
    pass


class ModelNotFoundError(InferOpsError):
    """Raised when a referenced model definition does not exist."""
    def __init__(self, model_name: str, searched_paths: Optional[list[str]] = None) -> None:
        msg = f"Model '{model_name}' was not found in catalog."
        if searched_paths:
            msg += f" Searched: {', '.join(searched_paths)}"
        super().__init__(msg)
        self.model_name = model_name


class ProfileNotFoundError(InferOpsError):
    """Raised when a profile name is not declared in profiles config."""
    def __init__(self, profile_name: str, available_profiles: Optional[list[str]] = None) -> None:
        msg = f"Profile '{profile_name}' is not defined."
        if available_profiles:
            msg += f" Available profiles: {', '.join(available_profiles)}"
        super().__init__(msg)
        self.profile_name = profile_name


class EngineNotInstalledError(InferOpsError):
    """Raised when the required engine binary (vllm or sglang) is missing."""
    def __init__(self, engine_name: str, executable: str) -> None:
        super().__init__(
            f"Engine '{engine_name}' executable '{executable}' was not found in environment."
        )
        self.engine_name = engine_name
        self.executable = executable


class PortConflictError(InferOpsError):
    """Raised when a designated port is already occupied."""
    def __init__(self, port: int, model_name: str) -> None:
        super().__init__(f"Port {port} for model '{model_name}' is already in use.")
        self.port = port
        self.model_name = model_name


class InsufficientVRAMError(InferOpsError):
    """Raised when a model requires more VRAM than available on target GPUs."""
    def __init__(
        self,
        model_name: str,
        required_gb: float,
        available_gb: float,
        recommendation: str = "",
    ) -> None:
        msg = (
            f"Model '{model_name}' requires approx {required_gb:.2f} GB VRAM, "
            f"but only {available_gb:.2f} GB is available across target GPUs."
        )
        if recommendation:
            msg += f" Suggestion: {recommendation}"
        super().__init__(msg)
        self.model_name = model_name
        self.required_gb = required_gb
        self.available_gb = available_gb


class ProcessStartupTimeoutError(InferOpsError):
    """Raised when a model process fails to become healthy within the timeout period."""
    def __init__(self, model_name: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Model '{model_name}' did not report healthy within {timeout_seconds}s timeout."
        )
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds


class ProcessCrashedError(InferOpsError):
    """Raised when a model process terminates unexpectedly during startup or execution."""
    def __init__(self, model_name: str, exit_code: Optional[int], log_path: str) -> None:
        super().__init__(
            f"Model '{model_name}' process crashed with exit code {exit_code}. Logs: {log_path}"
        )
        self.model_name = model_name
        self.exit_code = exit_code
        self.log_path = log_path
