"""
DeepSeek-V4.1-Flash FP4/FP8 block -> FP8 channelwise 转换脚本

流程：
  1. 反量化原始 checkpoint 中的 FP4/FP8 block 权重为 FP32
  2. 重新量化为 FP8 channelwise 格式
  3. 更新 config.json 和 index 文件

原始模型格式（见 config.json）：
  - quant_method: "fp8", weight_block_size: [32, 32]
  - scale_fmt: "ue8m0"
  - expert_dtype: "fp4"  (routed experts 是 FP4 E2M1FN packed)
  - Attention + Shared Experts: FP8 E4M3FN blockwise

目标格式：
  - 所有量化层转为 FP8 E4M3FN channelwise (per-output-channel)
  - compressed-tensors float-quantized format

用法：
  python quantize_deepseek_v4_1_flash_fp8_channel.py
"""

from __future__ import annotations

import json
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# FP4 E2M1FN 查找表
FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

# 探测确认：DeepSeek-V4.1-Flash 所有量化层 block_size 都是 32×32
# - FP4 experts:      weight (out, in/2) int8 packed, scale (out, in/32) float8_e8m0fnu
# - FP8 attn/shared:  weight (out, in)   float8_e4m3fn, scale (out/32, in/32) float8_e8m0fnu
FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 32


def scale_name_for(weight_name: str) -> str:
    """根据 weight 名称生成对应的 scale 名称 (源 checkpoint 中的名字)"""
    return ".".join(weight_name.split(".")[:-1] + ["scale"])


def out_scale_name_for(weight_name: str) -> str:
    """输出 checkpoint 中 scale 的名字 (sglang 期望的 weight_scale_inv)"""
    return ".".join(weight_name.split(".")[:-1] + ["weight_scale_inv"])


def unpack_e2m1fn_to_float(x: torch.Tensor) -> torch.Tensor:
    """
    Unpack FP4 E2M1FN packed tensor (int8) to float32.
    每个 int8 字节存储 2 个 FP4 值（低4位和高4位）
    """
    assert x.dtype == torch.int8
    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    return torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)


def dequant_fp4_to_float(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    反量化 FP4 blockwise 权重为 float32.
    x: packed FP4 weight (int8), shape (out_dim, in_dim // 2)
    scale: UE8M0 scale, shape (out_dim, in_dim // FP4_BLOCK_SIZE)
    """
    values = unpack_e2m1fn_to_float(x).float()
    # scale shape=(out, in/32), float8_e8m0fnu -> 扩展为 (out, in)
    expanded_scale = scale.float().repeat_interleave(FP4_BLOCK_SIZE, dim=1)
    return values * expanded_scale


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    反量化 FP8 blockwise 权重为 float32.
    weight: FP8 E4M3FN, shape (out_dim, in_dim)
    scale:  float8_e8m0fnu, shape (out_dim // block_size, in_dim // block_size)

    如果不满足 (2D 且能被 block_size 整除) 则走 fallback：
    根据 scale shape 推断 block_size，或直接用广播反量化。
    """
    # float8_e8m0fnu 可直接转 float32
    scale = scale.float()

    # Fallback: 非 2D 或不能整除 -> 直接广播反量化
    if weight.ndim != 2 or scale.ndim != 2:
        # 尝试通过 numel 比例推断
        if weight.numel() % scale.numel() == 0:
            ratio = weight.numel() // scale.numel()
            # 简单展平 * 广播
            flat_w = weight.float().view(-1, ratio) if ratio > 1 else weight.float().view(-1, 1)
            flat_s = scale.float().view(-1, 1)
            return (flat_w * flat_s).view_as(weight).float()
        return weight.float() * scale.float().mean()

    out_dim, in_dim = weight.shape
    s_rows, s_cols = scale.shape

    # 由 scale shape 推断实际 block size（更鲁棒）
    if s_rows == 0 or s_cols == 0 or out_dim % s_rows != 0 or in_dim % s_cols != 0:
        # 不能整除 -> fallback
        return weight.float() * scale.float().mean()

    row_block = out_dim // s_rows
    col_block = in_dim // s_cols
    num_rows_blocks = s_rows
    num_cols_blocks = s_cols

    weight_blocks = weight.unflatten(0, (num_rows_blocks, row_block)).unflatten(
        2, (num_cols_blocks, col_block)
    )
    # shape: (num_rows_blocks, block_size, num_cols_blocks, block_size)
    
    # Expand scale: (num_rows_blocks, num_cols_blocks) -> (num_rows_blocks, 1, num_cols_blocks, 1)
    scale_expanded = scale[:, None, :, None]
    
    # Dequantize
    dequantized = weight_blocks.float() * scale_expanded
    
    # Reshape back
    return dequantized.flatten(0, 1).flatten(1, 2)


def quantize_fp8_channelwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    量化为 FP8 E4M3FN channelwise (per-output-channel).
    tensor: float32, shape (out_dim, in_dim)
    返回: (quantized, scale)
      - quantized: FP8 E4M3FN, shape (out_dim, in_dim)
      - scale: float32, shape (out_dim, 1)

    注意: 权重保持 [out, in] 布局。sglang 的 loader 在 dim=0 上融合 wq_a+wkv
    并按 output_dim=0 做 TP 分片,转置由 process_weights_after_loading 负责。
    """
    assert tensor.ndim == 2
    qmax = torch.finfo(torch.float8_e4m3fn).max
    abs_max = tensor.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = abs_max / qmax
    quantized = torch.clamp(tensor.float() / scale, -qmax, qmax)
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.float32)


def is_fp4_expert(name: str, tensor: torch.Tensor) -> bool:
    """判断是否是 FP4 量化的 expert 权重"""
    return tensor.dtype == torch.int8 and "experts" in name and name.endswith(".weight")


def is_engram_embed(name: str) -> bool:
    """engram.embed 在 sglang 里是硬编码 fp8+e8m0fnu blockwise 的 hash table
    (见 sglang/srt/layers/engram.py: EngramEmbedding 与 engram_gather kernel),
    不接受 quant_config,也没有 .scale -> .weight_scale_inv 的 remap。
    必须原样透传,不做任何重量化。"""
    return ".engram.embed." in name


def is_engram_wkv(name: str) -> bool:
    """engram.wkv 是标准 ReplicatedLinear,支持多种量化格式"""
    return ".engram.wkv." in name


def is_fp8_blockwise_weight(name: str, tensor: torch.Tensor, state_dict: dict) -> bool:
    """判断是否是 FP8 blockwise 量化的权重"""
    if is_engram_embed(name):
        # engram.embed 的 fp8 权重不参与 blockwise->channelwise 转换,单独透传
        return False
    # engram.wkv 在 channel 模式下才走这个转换分支 (blockwise/bf16 模式已在前面处理)
    if tensor.dtype != torch.float8_e4m3fn or not name.endswith(".weight"):
        return False
    # 检查是否有对应的 scale（可能是 .scale 或 .weight_scale_inv）
    scale_name = scale_name_for(name)
    alt_scale_name = out_scale_name_for(name)
    return scale_name in state_dict or alt_scale_name in state_dict


def is_wo_a_weight(name: str) -> bool:
    """wo_a 权重不量化（保持 BF16）"""
    return name.endswith("wo_a.weight") or name.endswith("attn.o_a_proj.weight")


def has_scale(name: str, state_dict: dict[str, torch.Tensor]) -> bool:
    """检查权重是否有对应的 scale"""
    return scale_name_for(name) in state_dict


def convert_one_file(input_path: str, output_path: str, engram_wkv_mode: str = "channel") -> None:
    """转换单个 safetensors 文件

    Args:
        engram_wkv_mode: engram.wkv 的处理模式 (channel/bf16/blockwise)
    """
    state_dict = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict = {}
    for name, tensor in state_dict.items():
        if name.endswith(".scale") or name.endswith(".weight_scale_inv"):
            # scale 会在处理 weight 时一起处理
            continue

        scale_name = scale_name_for(name)
        out_scale_name = out_scale_name_for(name)

        # 0. engram.embed: sglang 硬编码 fp8_e4m3fn + e8m0fnu blockwise,
        #    键名保持裸 ".scale"(不 remap 为 weight_scale_inv),必须原样透传
        if is_engram_embed(name):
            new_state_dict[name] = tensor
            if scale_name in state_dict:
                new_state_dict[scale_name] = state_dict[scale_name]
            continue

        # 0a. engram.wkv: 根据 engram_wkv_mode 走不同路径
        if is_engram_wkv(name):
            if engram_wkv_mode == "blockwise":
                # 原样透传 blockwise FP8 (保留 .scale 键名)
                new_state_dict[name] = tensor
                if scale_name in state_dict:
                    new_state_dict[scale_name] = state_dict[scale_name]
                continue
            elif engram_wkv_mode == "bf16":
                # 反量化为 BF16 (不输出 scale)
                if has_scale(name, state_dict):
                    scale = state_dict[scale_name]
                    weight_bf16 = dequant_fp8_blockwise(tensor, scale).bfloat16()
                    new_state_dict[name] = weight_bf16
                else:
                    new_state_dict[name] = tensor
                continue
            # else: engram_wkv_mode == "channel" -> 继续走下面的通用 blockwise 转换

        # 1. FP4 experts: 反量化 -> FP8 channelwise (MoE不转置)
        if is_fp4_expert(name, tensor):
            scale = state_dict[scale_name]
            # 反量化为 float32
            weight_fp32 = dequant_fp4_to_float(tensor, scale)
            # 重新量化为 FP8 channelwise
            weight_fp8, new_scale = quantize_fp8_channelwise(weight_fp32)
            new_state_dict[name] = weight_fp8
            new_state_dict[out_scale_name] = new_scale

        # 2. wo_a 权重：反量化为 BF16（不重新量化）
        elif is_wo_a_weight(name):
            if has_scale(name, state_dict):
                scale = state_dict[scale_name]
                weight_bf16 = dequant_fp8_blockwise(tensor, scale).bfloat16()
                new_state_dict[name] = weight_bf16
            else:
                new_state_dict[name] = tensor

        # 3. FP8 blockwise 权重: 反量化 -> FP8 channelwise
        elif is_fp8_blockwise_weight(name, tensor, state_dict):
            # 兼容两种 scale 命名：.scale 和 .weight_scale_inv
            scale_name_input = scale_name_for(name)
            if scale_name_input not in state_dict:
                scale_name_input = out_scale_name_for(name)
            scale = state_dict[scale_name_input]

            # 只处理标准 2D blockwise 权重；否则原样保留
            if tensor.ndim != 2 or scale.ndim != 2:
                print(f"[skip non-2D] {name}: weight={tuple(tensor.shape)}, scale={tuple(scale.shape)}")
                new_state_dict[name] = tensor
                new_state_dict[out_scale_name] = scale
                continue
            out_dim, in_dim = tensor.shape
            s_rows, s_cols = scale.shape
            if s_rows == 0 or s_cols == 0 or out_dim % s_rows != 0 or in_dim % s_cols != 0:
                print(f"[skip shape mismatch] {name}: w={tuple(tensor.shape)}, s={tuple(scale.shape)}")
                new_state_dict[name] = tensor
                new_state_dict[out_scale_name] = scale
                continue
            # 反量化为 float32
            weight_fp32 = dequant_fp8_blockwise(tensor, scale)
            # 重新量化为 FP8 channelwise
            weight_fp8, new_scale = quantize_fp8_channelwise(weight_fp32)
            new_state_dict[name] = weight_fp8
            new_state_dict[out_scale_name] = new_scale

        # 4. 其他权重：直接复制
        else:
            new_state_dict[name] = tensor

    save_file(new_state_dict, output_path)


def convert_model(input_dir: str, output_dir: str, engram_wkv_mode: str = "channel") -> None:
    """转换整个模型"""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    for path in tqdm(files, desc="Converting"):
        fname = os.path.basename(path)
        convert_one_file(path, os.path.join(output_dir, fname), engram_wkv_mode)


def fp8_compression_config() -> dict:
    """生成 native FP8 per-channel 配置 (sglang 兼容)"""
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "is_checkpoint_fp8_serialized": True,
        "is_per_channel": True,
    }


def update_index(output_dir: str, engram_wkv_mode: str = "channel") -> None:
    """更新 model.safetensors.index.json,移除反量化为 BF16 的层的 weight_scale_inv"""
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    # 移除反量化为 BF16 的层 (wo_a / o_a_proj, 以及 engram.wkv 在 bf16 模式) 的 scale
    def should_remove_scale(name: str) -> bool:
        if not name.endswith(".weight_scale_inv"):
            return False
        weight_name = name.replace(".weight_scale_inv", ".weight")
        if weight_name.endswith("wo_a.weight") or weight_name.endswith("o_a_proj.weight"):
            return True
        if engram_wkv_mode == "bf16" and ".engram.wkv." in weight_name:
            return True
        return False
    
    model_index["weight_map"] = {
        name: fname
        for name, fname in model_index["weight_map"].items()
        if not should_remove_scale(name)
    }

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_metadata(input_dir: str, output_dir: str, engram_wkv_mode: str = "channel") -> None:
    """复制配置文件并更新 quantization_config"""
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "configuration.json",
        "model.safetensors.index.json",
    ]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, fname))

    update_index(output_dir, engram_wkv_mode)

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 使用 sglang 兼容的 native FP8 per-channel 格式
    config.pop("compression_config", None)
    quant_config = fp8_compression_config()

    # bf16 模式下需要把 engram.wkv 加入 ignored_layers,让 sglang 跳过为其创建 fp8 scale
    if engram_wkv_mode == "bf16":
        quant_config["ignored_layers"] = ["*.engram.wkv"]

    config["quantization_config"] = quant_config

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def main() -> None:
    parser = ArgumentParser(
        description="Convert DeepSeek-V4.1-Flash to FP8 channelwise format."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"C:\chl\models\DeepSeek-V4.1-Flash",
        help="Path to DeepSeek-V4.1-Flash checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\chl\models\DeepSeek-V4.1-Flash-FP8-Channel",
        help="Path to output converted checkpoint directory.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Torch CPU thread count.",
    )
    parser.add_argument(
        "--engram-wkv-mode",
        type=str,
        choices=["channel", "bf16", "blockwise"],
        default="channel",
        help="engram.wkv 处理模式: "
             "channel=转为 FP8 per-channel (默认,最优存储/速度); "
             "bf16=反量化为 BF16 (最高精度,+314MB); "
             "blockwise=保留原始 FP8 blockwise (可能支持 MXFP8 优化)",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)

    print(f"Converting {args.input_dir} to FP8 channelwise format...")
    print(f"Output directory: {args.output_dir}")
    print(f"engram.wkv mode: {args.engram_wkv_mode}")

    convert_model(args.input_dir, args.output_dir, args.engram_wkv_mode)
    copy_metadata(args.input_dir, args.output_dir, args.engram_wkv_mode)

    print(f"\nDone! Converted FP8 channelwise model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
