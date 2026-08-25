"""Model adapter registry."""

from typing import Any, List, Type

from .base import ModelAdapter
from .deepseek_v3_adapter import DeepSeekV3Adapter
from .qwen3_5_moe_adapter import Qwen35MoeAdapter


# More specific adapters must appear before less specific adapters.
_ADAPTER_CLASSES: List[Type[ModelAdapter]] = [
    Qwen35MoeAdapter,
    DeepSeekV3Adapter,
]


def get_model_adapter(config: Any) -> ModelAdapter:
    """Select an adapter from a Transformers config."""
    for adapter_class in _ADAPTER_CLASSES:
        if adapter_class.matches(config):
            return adapter_class(config)

    raise ValueError(
        "Unsupported model configuration: "
        f"model_type={getattr(config, 'model_type', None)!r}, "
        f"architectures={getattr(config, 'architectures', None)!r}. "
        "MoE-Quant currently supports DeepSeek-V3-style models "
        "and Qwen3.5/Qwen3.8 MoE text models."
    )


__all__ = [
    "ModelAdapter",
    "DeepSeekV3Adapter",
    "Qwen35MoeAdapter",
    "get_model_adapter",
]
