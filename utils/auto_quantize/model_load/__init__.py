"""Select architecture-specific source loading and EP runtimes."""

from importlib import import_module


_MODELS = {"mimo_v2": ("mimo", "MiMoRuntime"), "kimi_k3": ("kimi", "KimiRuntime")}


def get_model_loader(model_type):
    entry = _MODELS.get(model_type)
    return import_module(f"{__name__}.{entry[0]}") if entry else None


def get_ep_runtime(model_type):
    loader = get_model_loader(model_type)
    if loader is None:
        raise ValueError(f"EP streaming supports mimo_v2 and kimi_k3, got {model_type!r}")
    return getattr(loader, _MODELS[model_type][1])
