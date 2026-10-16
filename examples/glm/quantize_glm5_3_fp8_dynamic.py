# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3 FP8 block -> FP8_DYNAMIC 量化脚本

流程：
  1. FP8BlockDequantizer: 将源 checkpoint 中的 FP8 block 权重还原为 BF16
  2. model_free_ptq: 重新量化为 FP8_DYNAMIC

前置条件：
  - 源模型是 HuggingFace FP8 block checkpoint
  - llmcompressor / compressed-tensors 已安装，且支持 FP8BlockDequantizer

用法：
  python3 quantize_glm5_3_fp8_dynamic.py --model-path <输入模型> --save-dir <输出目录> [选项]

必填：
  --model-path               输入模型路径
  --save-dir                 输出模型保存目录

选项（默认全部关闭，对应这三类模块保持 BF16）：
  --quantize-shared-experts  量化 shared experts（第 3 层起的 gate/up/down_proj）
  --quantize-first-3-layers  量化前 3 层（dense MLP 与 attention）
  --quantize-indexer         量化 indexer 的 wk 与 wq_b
"""

from __future__ import annotations

import argparse
import json
import os

from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from llmcompressor import model_free_ptq

# 源 checkpoint 中所有带 weight_scale_inv 的层都需要先还原为 BF16。
# model_free_ptq 再根据 IGNORE 决定：routed experts 和 attention 重新量化，
# IGNORE 中的层保持 BF16。
# FP8BlockDequantizer 的 targets 必须覆盖所有实际含 weight_scale_inv 的模块，
# 但不能使用 r"re:.*"，否则会为 lm_head 等没有 scale 的权重创建错误依赖。
FP8_BLOCK_TARGETS = [
    # routed experts：反量化后重新量化为 FP8_DYNAMIC
    r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",

    # 普通 attention：反量化后重新量化为 FP8_DYNAMIC
    r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",

    # 原始权重含 weight_scale_inv，反量化后由 IGNORE 保持 BF16
    r"re:^model\.layers\.[012]\.mlp\.(gate_proj|up_proj|down_proj)$",
    r"re:.*mlp\.shared_experts\.(gate|up|down)_proj$",
    r"re:.*self_attn\.indexer\.(wk|wq_b)$",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-5.3 FP8 block -> FP8_DYNAMIC 量化。"
        "默认量化 routed experts 与第 3 层起的 attention，其余保持 BF16。"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="输入模型路径（HuggingFace FP8 block checkpoint 目录）",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="输出模型保存目录",
    )
    parser.add_argument(
        "--quantize-shared-experts",
        action="store_true",
        help="量化 shared experts（每层 token 必经，量化误差不会被 top-k 稀释，精度风险相对较高）",
    )
    parser.add_argument(
        "--quantize-first-3-layers",
        action="store_true",
        help="量化前 3 层的 dense MLP 与 attention（计算量占比约 2%%，收益较小）",
    )
    parser.add_argument(
        "--quantize-indexer",
        action="store_true",
        help="量化 indexer 的 wk 与 wq_b。weights_proj 与 k_norm 始终保持 BF16。"
        "注意：vLLM 与 sglang 在 CUDA 下会把 wk 融合进 wk_weights_proj 并强制 BF16，"
        "此时量化 wk 不会生效",
    )
    return parser.parse_args()


ARGS = parse_args()

# 量化 routed experts 和 attention；其余模块默认全部跳过。
# 前 3 层、shared experts、indexer 是否量化由命令行参数决定。
# 注释“原始权重含 weight_scale_inv”表示这些层在源 checkpoint 中已经是
# FP8 block 权重，需要先由 FP8BlockDequantizer 还原后再重新量化。
IGNORE = [
    "lm_head",
    "model.embed_tokens",
    "model.norm",

    # Norm / 非 Linear 权重
    r"re:.*\.input_layernorm$",
    r"re:.*\.post_attention_layernorm$",
    r"re:.*\.self_attn\.q_a_layernorm$",
    r"re:.*\.self_attn\.kv_a_layernorm$",
    r"re:.*\.mlp\.gate$",
    r"re:.*\.mlp\.gate\.e_score_correction_bias$",
    # Indexer 中无论是否量化 wk / wq_b，这两项都保持 BF16
    # （weights_proj 源 checkpoint 无 scale，k_norm 是 LayerNorm）
    r"re:.*\.self_attn\.indexer\.weights_proj$",
    r"re:.*\.self_attn\.indexer\.k_norm$",

    # 兼容 config.json 中可能存在的旧命名
    r"re:.*\.self_attn\.indexers_proj$",
    r"re:.*\.eh_proj$",
    r"re:.*\.enorm$",
    r"re:.*\.hnorm$",
    r"re:.*\.shared_head\.norm$",
]

if not ARGS.quantize_shared_experts:
    IGNORE.append(r"re:.*\.mlp\.shared_experts\.(gate|up|down)_proj$")

if not ARGS.quantize_first_3_layers:
    # 只匹配 mlp 与 attention 投影，不波及前 3 层的 indexer，
    # 使 indexer 是否量化只由 --quantize-indexer 决定。
    IGNORE.append(r"re:^model\.layers\.[012]\.mlp\..*")
    IGNORE.append(
        r"re:^model\.layers\.[012]\.self_attn\."
        r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$"
    )

if not ARGS.quantize_indexer:
    IGNORE.append(r"re:.*\.self_attn\.indexer\.(wk|wq_b)$")

print(
    "量化选项: "
    f"shared_experts={ARGS.quantize_shared_experts}, "
    f"first_3_layers={ARGS.quantize_first_3_layers}, "
    f"indexer={ARGS.quantize_indexer}"
)

model_free_ptq(
    model_stub=ARGS.model_path,
    save_directory=ARGS.save_dir,
    scheme="FP8_DYNAMIC",
    ignore=IGNORE,
    converter=FP8BlockDequantizer(
        targets=FP8_BLOCK_TARGETS,
        weight_block_size=(128, 128),
    ),
    max_workers=8,
    device=None,
)

# 清理嵌套的旧 quantization_config，避免加载器读取旧配置。
config_path = os.path.join(ARGS.save_dir, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    cleaned = False
    for key in list(config.keys()):
        if key == "quantization_config":
            continue
        value = config[key]
        if isinstance(value, dict) and "quantization_config" in value:
            old = value.pop("quantization_config")
            print(
                f"已清理嵌套的旧 quantization_config: {key}.quantization_config "
                f"(format={old.get('format', 'unknown')})"
            )
            cleaned = True

    if cleaned:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print("config.json 已更新。")
