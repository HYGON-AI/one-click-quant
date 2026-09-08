"""DeepSeek-V3 model adapter."""

from typing import Any

from .base import ModelAdapter


class DeepSeekV3Adapter(ModelAdapter):
    """Adapter for DeepSeek-V3-style Hugging Face causal language models."""

    name = "deepseek_v3"

    @classmethod
    def matches(cls, config: Any) -> bool:
        model_type = str(getattr(config, "model_type", "")).lower()
        architectures = {
            str(value).lower()
            for value in getattr(config, "architectures", []) or []
        }
        return (
            model_type == "deepseek_v3"
            or "deepseekv3forcausallm" in architectures
        )

    def prepare_config(self, config: Any, world_size: int) -> None:
        # DeepSeek expert parallelism uses the distributed world size.
        config.ep_size = world_size

    def validate_quantization_args(self, args: Any) -> None:
        if args.bits != 4:
            raise ValueError(
                "DeepSeek-V3 currently supports only --bits 4 (W4A16); 8-bit is not validated "
                "for this adapter."
            )
