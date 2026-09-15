# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import re
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import math
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file


def dequantize_block_fp8(input: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    # 获取输入维度
    K, N = input.size()
    K1, N1 = scale.size()
    # 创建填充后的张量 (复现参考代码的处理方式，这里使用zeros填充未覆盖区域，实际未参与计算)
    padded_input = torch.zeros([K1 * block_size, N1 * block_size], 
                                dtype=input.dtype, device=input.device)
    padded_input[:K, :N] = input
    # 1. 维度变换：View -> Permute
    # 将输入重排为 [K1, N1, block_size, block_size]，使 block 维度连续
    fp8_blocks = padded_input.view(K1, block_size, N1, block_size).permute(0, 2, 1, 3)
    # 2. 反量化计算
    # 扩展 scale 维度以匹配 block 维度 [K1, N1, 1, 1]
    scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
    # 将 FP8 转换为 float32 进行计算，乘以 scale
    dequant_values = fp8_blocks.float() * scale_expanded
    # 3. 维度逆变换：Permute -> Reshape
    # 恢复为原始的 [K1*block_size, N1*block_size] 布局
    restored_layout = dequant_values.permute(0, 2, 1, 3).reshape(padded_input.shape)
    # 4. 截取有效区域并转换为 BF16
    return restored_layout[:K, :N].to(torch.bfloat16)
 


def worker(
    rank: int,
    world_size: int,
    safetensor_files: list,
    weight_map: dict,
    input_dir: str,
    output_dir: str,
    quant_type: str,
    block_size: int,
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    local_files = safetensor_files[rank::world_size]
    loaded_files = {}

    def get_tensor(name):
        fname = weight_map[name]
        if fname not in loaded_files:
            loaded_files[fname] = load_file(
                os.path.join(input_dir, fname), device='cpu'
            )
        return loaded_files[fname][name].to(device)

    for safetensor_file in tqdm(local_files, position=rank):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device=device)
        new_state_dict = {}

        for weight_name, weight in state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            if "v_proj" in weight_name:
                continue
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map:
                if "k_proj" in weight_name:
                    scale_inv = get_tensor(scale_inv_name)
                    v_proj_weight_name = weight_name.replace("k_proj", "v_proj") #[768,4096]
                    v_proj_scale_inv_name = scale_inv_name.replace("k_proj", "v_proj") #[512,4096]

                    v_proj_weight = get_tensor(v_proj_weight_name) #[6,32]
                    v_proj_scale_inv = get_tensor(v_proj_scale_inv_name) #[4,32]

                    # === 1. TP 切分 ===
                    v_chunks = torch.chunk(v_proj_weight, 4, dim=0)   # 4 x [128, 4096]
                    k_chunks = torch.chunk(weight, 4, dim=0)          # 4 x [192, 4096]
                    v_s_chunks = torch.chunk(v_proj_scale_inv, 4, dim=0) #4 [1,32]
                    k_s_chunks = list(torch.chunk(scale_inv, 4, dim=0)) #3 [2,32]
                    new_k_chunks = []
                    new_v_chunks = []

                    # === 3. 每个 rank 拼接 → 反量化 → 再拆 ===
                    for i in range(4):
                        k_i = k_chunks[i]
                        v_i = v_chunks[i]
                        merged_kv = torch.cat([k_i, v_i], dim=0)   # [320, 4096]
                        merged_s  = torch.cat([k_s_chunks[i], v_s_chunks[i]], dim=0) # [3,32]

                        # === 反量化 ===
                        dequant_kv = dequantize_block_fp8(merged_kv, merged_s, block_size)

                        # === 拆回 K / V（关键！）===
                        k_size = k_i.size(0)   # 192
                        v_size = v_i.size(0)   # 128

                        new_k = dequant_kv[:k_size, :]
                        new_v = dequant_kv[k_size:k_size + v_size, :]

                        new_k_chunks.append(new_k)
                        new_v_chunks.append(new_v)

                    # === 4. 拼回完整 tensor ===
                    new_k_weight = torch.cat(new_k_chunks, dim=0)   # [768, 4096]
                    new_v_weight = torch.cat(new_v_chunks, dim=0)   # [512, 4096]
                    # === 5. 保存（保持原命名）===
                    new_state_dict[weight_name] = new_k_weight
                    v_weight_name = weight_name.replace("k_proj", "v_proj")
                    new_state_dict[v_weight_name] = new_v_weight
                    
                else:
                    scale_inv = get_tensor(scale_inv_name)
                    new_state_dict[weight_name] = dequantize_block_fp8(weight, scale_inv, block_size)
            else:
                new_state_dict[weight_name] = weight

        save_file(
            new_state_dict,
            os.path.join(output_dir, file_name),
        )


def main(input_path, output_path, quant_type, block_size):
    # os.makedirs(output_path, exist_ok=True)
    if not quant_type in ("int8", "fp8", "int4", "bf16"):
        raise f"UNSUPPORT TYPE:{quant_type}"
    
    src_dir = Path(input_path)
    dst_dir = Path(output_path)
    dst_dir.mkdir(exist_ok=True)
    for file in src_dir.rglob("*.json"):
        shutil.copy(file, dst_dir / file.name)
    for file in src_dir.rglob("*.py"):
        shutil.copy(file, dst_dir / file.name)
        
    index_path = os.path.join(output_path, "model.safetensors.index.json")
    config_path = os.path.join(output_path, "config.json")
    if not os.path.exists(index_path) or not os.path.exists(config_path):
        raise f"CAN NOT FIND FILE: {index_path} {config_path}"
    
    with open(index_path, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    print(f"safetensor_files:{input_path}")
    world_size = torch.cuda.device_count()
    assert world_size > 0, "No CUDA devices found"

    mp.spawn(
        worker,
        args=(world_size, safetensor_files, weight_map, input_path, output_path, quant_type, block_size),
        nprocs=world_size,
        join=True,
    )
    new_weight_map = {}
    for k, v in weight_map.items():
        if not("scale_inv" in k):
            new_weight_map[k] = v
        
    model_index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"model.safetensors.index.json modified and saved to {index_path}")
    
    
    with open(config_path, "r") as f:
        config = json.load(f)
    ignore = config["quantization_config"]["ignored_layers"]
    config.pop("quantization_config", None)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"config.json modified and saved to {config_path}")
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-block-hf-path", type=str, default="/models/DeepSeek-R1")
    parser.add_argument("--output-channel-hf-path", type=str, default="/models/DeepSeek-R1-Channel-int8")
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    main(args.input_block_hf_path, args.output_channel_hf_path, 'bf16',args.block_size)
    print("done")
