# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Model-specific utilities and adapter selection for MoE-Quant."""

from .base import ModelAdapter
from .deepseek_v3_adapter import DeepSeekV3Adapter
from .glm5_next_adapter import Glm5NextAdapter
from .glm_moe_dsa_adapter import GlmMoeDsaAdapter
from .kimi_k3_adapter import KimiK3Adapter
from .qwen3_5_moe_adapter import Qwen35MoeAdapter
from .registry import get_model_adapter

__all__ = [
    "ModelAdapter",
    "DeepSeekV3Adapter",
    "Glm5NextAdapter",
    "GlmMoeDsaAdapter",
    "KimiK3Adapter",
    "Qwen35MoeAdapter",
    "get_model_adapter",
]
