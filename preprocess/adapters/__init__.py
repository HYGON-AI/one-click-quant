"""Model adapter registry for preprocess."""

from __future__ import annotations

from typing import Dict, Optional, Type

from preprocess.adapters.base import ModelAdapter
from preprocess.adapters.deepseek_v3 import DeepSeekV3Adapter
from preprocess.adapters.kimi_k3 import KimiK3Adapter
from preprocess.adapters.qwen3_8_flash_next import Qwen38FlashNextAdapter

ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
    KimiK3Adapter.name: KimiK3Adapter,
    Qwen38FlashNextAdapter.name: Qwen38FlashNextAdapter,
    DeepSeekV3Adapter.name: DeepSeekV3Adapter,
    # Aliases
    "kimi-k3": KimiK3Adapter,
    "moonshot_kimi_k3": KimiK3Adapter,
    "qwen3.8-flash-next": Qwen38FlashNextAdapter,
    "qwen3_8_flash": Qwen38FlashNextAdapter,
    "qwen4_exp": Qwen38FlashNextAdapter,
    "qwen4_exp_text": Qwen38FlashNextAdapter,
    "deepseek-v3": DeepSeekV3Adapter,
    "DeepseekV3ForCausalLM": DeepSeekV3Adapter,
}

def register_adapter(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
    """Decorator / helper to register a custom adapter class."""
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls} must define a non-empty .name")
    ADAPTER_REGISTRY[cls.name] = cls
    return cls


def get_adapter(name: str) -> ModelAdapter:
    key = (name or "").strip()
    if not key:
        raise ValueError(
            "Recipe must set `adapter: <name>` (e.g. kimi_k3). "
            f"Available: {sorted(set(ADAPTER_REGISTRY))}"
        )
    cls = ADAPTER_REGISTRY.get(key) or ADAPTER_REGISTRY.get(key.replace("-", "_"))
    if cls is None:
        raise KeyError(
            f"Unknown preprocess adapter '{name}'. "
            f"Available: {sorted(set(ADAPTER_REGISTRY))}. "
            "Add preprocess/adapters/<model>.py and register it."
        )
    return cls()


def resolve_adapter_name(recipe: dict, config: Optional[dict] = None) -> str:
    """Pick adapter id from recipe, then optional config hint."""
    if recipe.get("adapter"):
        return str(recipe["adapter"])
    if recipe.get("model"):
        return str(recipe["model"])
    cfg = config or {}
    # Weak auto-detect (optional); prefer explicit recipe.adapter.
    mt = str(cfg.get("model_type") or cfg.get("architectures") or "").lower()
    if "kimi" in mt or "moonshot" in mt:
        return "kimi_k3"
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else {}
    mt2 = str(tc.get("model_type") or "").lower()
    if "qwen4_exp" in mt or "qwen4_exp" in mt2 or "qwen4exp" in mt.replace("_", ""):
        return "qwen3_8_flash_next"
    if "deepseek_v3" in mt or "deepseekv3" in mt.replace("_", ""):
        return "deepseek_v3"
    raise ValueError(
        "Cannot resolve preprocess adapter: set `adapter: kimi_k3` / "
        "`qwen3_8_flash_next` / `deepseek_v3` (or another registered name) "
        "in the recipe YAML."
    )


__all__ = [
    "ADAPTER_REGISTRY",
    "ModelAdapter",
    "KimiK3Adapter",
    "Qwen38FlashNextAdapter",
    "DeepSeekV3Adapter",
    "get_adapter",
    "register_adapter",
    "resolve_adapter_name",
]
