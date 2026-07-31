"""
Kimi K3 MXFP4 → FP8_DYNAMIC 量化脚本

流程：
  1. CompressedTensorsDequantizer: 解压 MXFP4 packed 权重 → BF16
  2. model_free_ptq:                    重新量化为 FP8_DYNAMIC (per-channel weights + per-token dynamic activations)

前置条件：
  - compressed-tensors 已安装 MXFP4PackedCompressor.compression_param_names 修复
  - 完整模型文件已下载到 MODEL_ID 目录

用法：
  python quantize_kimi_k3_fp8_dynamic.py
"""

from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer
from llmcompressor import model_free_ptq
import json
import os

MODEL_ID = "/models/kimi-k3/Kimi-K3"
SAVE_DIR = "/models/kimi-k3/Kimi-K3-Channel-FP8-w8a8-2"

ignore = [
    # === 以下与原始 MXFP4 配置保持一致（保守策略） ===
    "re:.*self_attn.*",                                  # 所有注意力层 → BF16
    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",         # 密集 MLP 投影 → BF16
    "re:.*lm_head.*",                                    # 输出头 → BF16
    "re:.*vision_tower.*",                               # 视觉编码器 → BF16
    "re:.*mm_projector.*",                               # 多模态投影 → BF16

    # === 以下为额外排除（原始 BF16 / vLLM 不兼容的层） ===
    "re:.*embed_tokens.*",                               # 嵌入层 → BF16
    "re:.*block_sparse_moe\\.(gate|routed_expert_down_proj|routed_expert_up_proj).*",  # MoE 内特殊层：gate(非Linear) + latent MoE投影(vLLM兼容)
    "re:.*conv1d",                                       # Short convolution 核 (非 Linear)

    # === 对齐原始 MXFP4：残差投影保持 BF16 ===
    "re:.*mlp_res_proj.*",                               # 每层 MLP 残差 (7168→1)
    "re:.*self_attention_res_proj.*",                    # 每层 Attention 残差 (7168→1)
    "re:.*output_attn_res_proj.*",                       # 输出 Attention 残差 (7168→1)
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_DYNAMIC",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=8,
    device=None,
    #device=["cuda:0","cuda:1"],
)

# ── 清理嵌套的旧 quantization_config ──
# model_free_ptq 的 update_config 只写到顶层 config["quantization_config"]，
# 不会清理 text_config 等嵌套字段中遗留的旧 quantization_config。
# 这会导致 vLLM 加载时读到旧配置而出错，此处手动清理。
config_path = os.path.join(SAVE_DIR, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)

    cleaned = False
    for key in list(config.keys()):
        if key == "quantization_config":
            continue  # 保留顶层新的 FP8_DYNAMIC 配置
        value = config[key]
        if isinstance(value, dict) and "quantization_config" in value:
            old = value.pop("quantization_config")
            print(f"已清理嵌套的旧 quantization_config: {key}.quantization_config "
                  f"(format={old.get('format', 'unknown')})")
            cleaned = True

    if cleaned:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("config.json 已更新。")
