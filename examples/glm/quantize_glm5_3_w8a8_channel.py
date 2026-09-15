"""
GLM-5.3 FP8 block -> W8A8 channel-wise 量化脚本

流程：
  1. FP8BlockDequantizer: 将源 checkpoint 中的 FP8 block 权重还原为 BF16
  2. model_free_ptq: 重新量化为 W8A8，权重输出为 channel-wise INT8

前置条件：
  - 源模型是 HuggingFace FP8 block checkpoint
  - llmcompressor / compressed-tensors 已安装，且支持 FP8BlockDequantizer

用法：
  python3 quantize_glm5_3_w8a8_channel.py                                              # 仅量化 routed experts
  python3 quantize_glm5_3_w8a8_channel.py --quantize-attention                         # 额外量化 layer 3+ 的 attention 投影
  python3 quantize_glm5_3_w8a8_channel.py --quantize-shared-experts                    # 额外量化 layer 3+ 的 shared_experts
  python3 quantize_glm5_3_w8a8_channel.py --quantize-attention --quantize-shared-experts  # 两者叠加

  # 覆盖输入 / 输出路径（默认见 --help）
  python3 quantize_glm5_3_w8a8_channel.py --model-id /path/to/src --save-dir /path/to/dst

  # 指定若干层的 routed experts 保持 BF16（示例：层 3,4,5 与 60-65）
  python3 quantize_glm5_3_w8a8_channel.py --keep-routed-experts-bf16 3,4,5,60-65
"""

from __future__ import annotations

import argparse
import json
import os

from compressed_tensors.entrypoints.convert import FP8BlockDequantizer
from llmcompressor import model_free_ptq


def _parse_layer_ids(s: str) -> list[int]:
    """把 "3,7-10,15" 解析成排序后的层号列表 [3, 7, 8, 9, 10, 15]。"""
    result: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                raise argparse.ArgumentTypeError(
                    f"层号区间无效: {part} (lo > hi)"
                )
            result.update(range(lo, hi + 1))
        else:
            result.add(int(part))
    return sorted(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLM-5.3 FP8 block -> W8A8 channel-wise 量化"
    )
    parser.add_argument(
        "--model-id",
        default="/public/opendas/DL_DATA/llm-models/GLM-5.3",
        help="源 HuggingFace FP8 block checkpoint 路径（本地目录或 HF repo id）。",
    )
    parser.add_argument(
        "--save-dir",
        default="/work/models/GLM-5.3-W8A8-CHANNEL",
        help="量化后 W8A8 模型保存目录。多档 A/B 时请指定不同目录以避免覆盖。",
    )
    parser.add_argument(
        "--quantize-attention",
        action="store_true",
        help=(
            "同时把 layer 3+ 的 attention 5 个投影 "
            "(q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj) "
            "量化为 W8A8。默认关闭，仅量化 routed experts。"
            "indexer / norm / 前 3 层始终保持 BF16。"
        ),
    )
    parser.add_argument(
        "--quantize-shared-experts",
        action="store_true",
        help=(
            "同时把 layer 3+ 的 mlp.shared_experts (gate/up/down)_proj "
            "量化为 W8A8。默认关闭。前 3 层为 dense 结构，不含 shared_experts。"
        ),
    )
    parser.add_argument(
        "--keep-routed-experts-bf16",
        type=_parse_layer_ids,
        default=[],
        metavar="LAYERS",
        help=(
            "指定这些层的 routed experts (mlp.experts.N.(gate|up|down)_proj) "
            "不量化，保持 BF16。格式示例: '3,7-10,15'。默认为空（全部量化）。"
            "前 3 层本身为 dense、无 routed experts，指定它们无效果。"
        ),
    )
    return parser.parse_args()


args = _parse_args()

MODEL_ID = args.model_id
SAVE_DIR = args.save_dir

# 源 checkpoint 中所有带 weight_scale_inv 的层都需要先还原为 BF16。
# model_free_ptq 再根据 IGNORE 决定：routed experts 和 attention 重新量化，
# IGNORE 中的层保持 BF16。
# FP8BlockDequantizer 的 targets 必须覆盖所有实际含 weight_scale_inv 的模块，
# 但不能使用 r"re:.*"，否则会为 lm_head 等没有 scale 的权重创建错误依赖。
FP8_BLOCK_TARGETS = [
    # 仅量化 routed experts；attention 权重先反量化后由 IGNORE 保持 BF16。
    r"re:.*mlp\.experts\.\d+\.(gate|up|down)_proj$",

    # 源 attention 权重含 weight_scale_inv，必须先还原为 BF16。
    r"re:.*self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$",

    # 原始权重含 weight_scale_inv，反量化后由 IGNORE 保持 BF16
    r"re:^model\.layers\.[012]\.mlp\.(gate_proj|up_proj|down_proj)$",
    r"re:.*mlp\.shared_experts\.(gate|up|down)_proj$",
    r"re:.*self_attn\.indexer\.(wk|wq_b)$",
]

# 仅量化 routed experts；其余模块全部跳过。
# W8A8 的权重策略由 compressed-tensors / llmcompressor 的 W8A8 scheme
# 设置为 channel-wise，激活值保持 dynamic。
# 说明：
#   - attention 5 投影 (q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj)
#     是否量化由 --quantize-attention 控制；默认追加到 IGNORE。
#   - mlp.shared_experts.(gate|up|down)_proj 是否量化由 --quantize-shared-experts
#     控制；默认追加到 IGNORE。
IGNORE = [
    "lm_head",
    "model.embed_tokens",
    "model.norm",

    # 前 3 层：所有模块均保持 BF16
    r"re:^model\.layers\.[012]\..*",

    # Norm / 非 Linear 权重
    r"re:.*\.input_layernorm$",
    r"re:.*\.post_attention_layernorm$",
    r"re:.*\.self_attn\.q_a_layernorm$",
    r"re:.*\.self_attn\.kv_a_layernorm$",
    r"re:.*\.mlp\.gate$",
    r"re:.*\.mlp\.gate\.e_score_correction_bias$",
    # Indexer：全部跳过，不进行量化
    r"re:.*\.self_attn\.indexer\..*",

    # 兼容 config.json 中可能存在的旧命名
    r"re:.*\.self_attn\.indexers_proj$",
    r"re:.*\.eh_proj$",
    r"re:.*\.enorm$",
    r"re:.*\.hnorm$",
    r"re:.*\.shared_head\.norm$",
]

if not args.quantize_attention:
    # 默认路径：attention 权重反量化后不重新量化，保持 BF16。
    IGNORE.append(
        r"re:.*\.self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)$"
    )
    print("[quantize] attention 投影保持 BF16（仅量化 routed experts）")
else:
    # layer 3+ 的 attention 五个投影会被量化为 W8A8 channel-wise INT8。
    # layer 0/1/2 仍由上方 `re:^model\.layers\.[012]\..*` 保护为 BF16。
    # indexer 由 `re:.*\.self_attn\.indexer\..*` 保护为 BF16。
    print(
        "[quantize] attention 投影 (q_a_proj/q_b_proj/kv_a_proj_with_mqa/"
        "kv_b_proj/o_proj) 从 layer 3 起量化为 W8A8"
    )

if not args.quantize_shared_experts:
    # 默认路径：shared_experts 反量化后不重新量化，保持 BF16。
    IGNORE.append(
        r"re:.*\.mlp\.shared_experts\.(gate|up|down)_proj$"
    )
    print("[quantize] shared_experts 保持 BF16")
else:
    # layer 3+ 的 shared_experts (gate/up/down)_proj 会被量化为 W8A8。
    # 前 3 层是 dense（first_k_dense_replace=3），不含 shared_experts 子模块，
    # 无需额外保护。
    print("[quantize] shared_experts (gate/up/down)_proj 从 layer 3 起量化为 W8A8")

if args.keep_routed_experts_bf16:
    # 指定层的 routed experts 保持 BF16：拼接为 (i1|i2|...) 精确匹配层号。
    ids = "|".join(str(i) for i in args.keep_routed_experts_bf16)
    IGNORE.append(
        rf"re:^model\.layers\.({ids})\.mlp\.experts\.\d+\.(gate|up|down)_proj$"
    )
    print(
        f"[quantize] routed experts 保持 BF16 的层: "
        f"{args.keep_routed_experts_bf16}"
    )

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="W8A8",
    ignore=IGNORE,
    converter=FP8BlockDequantizer(
        targets=FP8_BLOCK_TARGETS,
        weight_block_size=(128, 128),
    ),
    max_workers=8,
    device=None,
)

# 清理嵌套的旧 quantization_config，避免加载器读取旧配置。
# W8A8 scheme 生成的权重配置使用 strategy=channel，即每个输出 channel 一个 scale。
config_path = os.path.join(SAVE_DIR, "config.json")
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
