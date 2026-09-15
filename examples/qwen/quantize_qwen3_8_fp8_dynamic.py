# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.8 FP8 block → FP8_DYNAMIC 量化脚本

流程：
  1. FP8BlockDequantizer: 将源 checkpoint 中的 FP8 block expert 权重
     (weight + weight_scale_inv) 还原为 BF16
  2. model_free_ptq: 重新量化为 FP8_DYNAMIC

前置条件：
  - 源模型是 HuggingFace FP8 block checkpoint，不是 compressed-tensors MXFP4
  - llmcompressor / compressed-tensors 已安装，且支持 FP8BlockDequantizer

用法：
  python3 quantize_qwen3_8_fp8_dynamic.py
"""

from __future__ import annotations

import json
import os

from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from llmcompressor import model_free_ptq

MODEL_ID = "/models/Qwen3.8-2.4T-A95B-FP8"
SAVE_DIR = "/models/Qwen3.8-2.4T-A95B-FP8-CHANNEL"

# 源 checkpoint 里真正带 weight_scale_inv 的只有 routed expert FFN。
# 这里先把它们从 FP8 block 还原为 BF16，再交给 model_free_ptq 做 FP8_DYNAMIC。
FP8_BLOCK_TARGETS = [
    r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",
]

# Qwen3.8 的 ignore 不能照搬 Kimi；这里用 regex 表达源模型
# config.json:quantization_config.modules_to_not_convert 的同等规则。
IGNORE = [
    "lm_head",
    "model.embed_tokens",
    r"re:.*linear_attn\.conv1d$",
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
    r"re:.*linear_attn\.in_proj_qkv$",
    r"re:.*linear_attn\.in_proj_z$",
    r"re:.*linear_attn\.out_proj$",
    r"re:.*mlp\.gate$",
    r"re:.*mlp\.shared_expert\.(gate|up|down)_proj$",
    r"re:.*mlp\.shared_expert_gate$",
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    "mtp.fc",
    r"re:^mtp\.layers\.0\.mlp\.gate$",
    r"re:^mtp\.layers\.0\.mlp\.shared_expert\.(gate|up|down)_proj$",
    r"re:^mtp\.layers\.0\.mlp\.shared_expert_gate$",
    r"re:^mtp\.layers\.0\.self_attn\.(q|k|v|o)_proj$",

    # model_free_ptq 在不加载模型定义时会把所有未 ignore 的 *.weight 当作候选，
    # 这两个 MTP 1D 权重不在源 config 的 modules_to_not_convert 里，
    # 但不是 2D Linear 权重，必须额外跳过，否则 validate_weight_for_quantization 会报错。
    "mtp.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_DYNAMIC",
    ignore=IGNORE,
    converter=FP8BlockDequantizer(
        targets=FP8_BLOCK_TARGETS,
        weight_block_size=(128, 128),
    ),
    max_workers=8,
    device=None,
)

# ── 清理嵌套的旧 quantization_config ──
# model_free_ptq 会写入新的顶层 config["quantization_config"]，
# 但不会主动清掉 config 中其它嵌套字段遗留的旧配置。
# config_path = os.path.join(SAVE_DIR, "config.json")
# if os.path.exists(config_path):
#     with open(config_path, "r", encoding="utf-8") as f:
#         config = json.load(f)
#
#     cleaned = False
#     for key in list(config.keys()):
#         if key == "quantization_config":
#             continue
#         value = config[key]
#         if isinstance(value, dict) and "quantization_config" in value:
#             old = value.pop("quantization_config")
#             print(
#                 f"已清理嵌套的旧 quantization_config: {key}.quantization_config "
#                 f"(format={old.get('format', 'unknown')})"
#             )
#             cleaned = True
#
#     if cleaned:
#         with open(config_path, "w", encoding="utf-8") as f:
#             json.dump(config, f, indent=2, ensure_ascii=False)
#         print("config.json 已更新。")
