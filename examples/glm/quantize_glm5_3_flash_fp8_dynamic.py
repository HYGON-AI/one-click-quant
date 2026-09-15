# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3-Flash FP8 block -> FP8_DYNAMIC 量化脚本

流程：
  1. FP8BlockDequantizer: 将源 checkpoint 中的 FP8 block 权重还原为 BF16
  2. model_free_ptq: 重新量化语言模型中的 routed experts 和 attention

前置条件：
  - 源模型是 HuggingFace FP8 block checkpoint
  - llmcompressor / compressed-tensors 已安装，且支持 FP8BlockDequantizer

用法：
  python quantize_glm5_3_flash_fp8_dynamic.py
"""

from __future__ import annotations

import json
import os

from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from llmcompressor import model_free_ptq

MODEL_ID = "/models/GLM-5.3-Flash"
SAVE_DIR = "/models/GLM-5.3-Flash-FP8-DYNAMIC"

# Flash checkpoint 中带有 weight_scale_inv 的语言模型模块。
# 只匹配这些模块，避免为 lm_head、视觉模块等没有 block scale 的权重
# 创建错误依赖。
FP8_BLOCK_TARGETS = [
    r"re:^model\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj$",
    r"re:^model\.language_model\.layers\.\d+\.mlp\.experts\.\d+\.(gate|up|down)_proj$",
    r"re:^model\.language_model\.layers\.\d+\.mlp\.shared_experts\.(gate|up|down)_proj$",
    r"re:^model\.language_model\.layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)$",
]

# 保持 BF16 的模块；其余语言模型 Linear 模块重新量化为 FP8_DYNAMIC。
IGNORE = [
    "lm_head",
    "model.language_model.embed_tokens",
    "model.language_model.norm",

    # 前 3 层的全部模块保持 BF16，不进行量化。
    r"re:^model\.language_model\.layers\.[012]\..*",

    "model.language_model.layers.45.self_attn.indexer",
    "model.language_model.layers.45.self_attn.kv_b_proj",

    # Norm、门控和 hyper-connection 参数不是目标量化权重。
    r"re:^model\.language_model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)$",
    r"re:^model\.language_model\.layers\.(3|7|11|15|19|23|27|31|35|39|43|45)\.self_attn\.(q_a_layernorm|kv_a_layernorm|o_norm)$",
    r"re:^model\.language_model\.layers\.\d+\.mlp\.gate$",
    r"re:^model\.language_model\.layers\.\d+\.mlp\.gate\.e_score_correction_bias$",
    r"re:^model\.language_model\.layers\.\d+\.hc_(attn_base|attn_fn|attn_scale|ffn_base|ffn_fn|ffn_scale)$",

    # 线性注意力中的非 Linear / 状态参数保持 BF16。
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.(A_log|b_proj|dt_bias|f_a_proj|f_b_proj|g_a_proj|g_b_proj|k_proj|q_proj|v_proj)$",
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.o_norm$",
    r"re:^model\.language_model\.layers\.(0|1|2|4|5|6|8|9|10|12|13|14|16|17|18|20|21|22|24|25|26|28|29|30|32|33|34|36|37|38|40|41|42|44)\.self_attn\.o_proj$",
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.kv_b_proj$",

    # Flash 特有的输出与归一化参数保持 BF16。
    r"re:^model\.language_model\.layers\.\d+\.(eh_proj|enorm|hnorm)$",
    r"re:^model\.language_model\.layers\.\d+\.shared_head\.norm$",

    # 线性注意力卷积权重不是 2D Linear 权重，保持 BF16。
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.(q_conv1d|k_conv1d|v_conv1d)$",
    # Indexer 权重保持 BF16。
    r"re:^model\.language_model\.layers\.\d+\.self_attn\.indexer\..*$",

    # 视觉编码器保持 BF16。
    "model.visual",
    "visual",
    r"re:^model\.visual\..*$",
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

# 清理嵌套的旧 quantization_config，避免加载器读取旧配置。
config_path = os.path.join(SAVE_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    cleaned = False
    for key, value in config.items():
        if key == "quantization_config":
            if isinstance(value, dict) and isinstance(value.get("ignore"), list):
                value["ignore"] = [
                    item.replace("model.language_model.", "model.").replace(
                        r"model\.language_model\.", r"model\."
                    ).replace("model.visual", "visual")
                    for item in value["ignore"]
                ]
                value["ignore"].extend(
                    [
                        "model.eh_proj",
                        "model.enorm",
                        "model.hnorm",
                        "model.decoder.input_layernorm",
                        "model.decoder.post_attention_layernorm",
                        "model.decoder.mlp.gate",
                        "model.decoder.mlp.gate.e_score_correction_bias",
                        "model.decoder.self_attn.indexer",
                        "model.decoder.self_attn.kv_a_layernorm",
                        "model.decoder.self_attn.kv_b_proj",
                        "model.decoder.self_attn.q_a_layernorm",
                        "model.shared_head.norm",
                    ]
                )
                print("已将 quantization_config.ignore 转换为 SGLang 运行时模块名。")
                cleaned = True
            continue
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
