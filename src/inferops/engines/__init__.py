"""Engine adapter registry and lookup."""

from inferops.core.config import EngineType
from inferops.engines.base import EngineAdapter
from inferops.engines.sglang import SglangEngineAdapter
from inferops.engines.vllm import VllmEngineAdapter

_ENGINES: dict[EngineType, type[EngineAdapter]] = {
    EngineType.VLLM: VllmEngineAdapter,
    EngineType.SGLANG: SglangEngineAdapter,
}


def get_engine_adapter(engine: EngineType) -> EngineAdapter:
    """Retrieve an instantiated engine adapter for the specified engine type."""
    adapter_cls = _ENGINES.get(engine)
    if not adapter_cls:
        raise ValueError(f"Unsupported engine type: {engine}. Available: {list(_ENGINES.keys())}")
    return adapter_cls()


__all__ = ["EngineAdapter", "VllmEngineAdapter", "SglangEngineAdapter", "get_engine_adapter"]
