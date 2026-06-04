## 核心逻辑：Group-INT4 (G32) -> BF16 -> Channel-FP8
import os
import argparse
import sys
import json
import torch
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from pathlib import Path
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from utils.logging_config import get_logger
logger = get_logger(__name__)
# --- 基础量化工具函数 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def unpack_int4_from_int32(packed: torch.Tensor):
    """从脚本1继承：解包 int32 到 int8 (-8~7)"""
    M, K_packed = packed.shape
    K = K_packed * 8
    packed = packed.to(torch.int32)
    shifts = torch.arange(8, device=packed.device) * 4
    shifts = shifts.view(1, 1, 8)
    x = packed.unsqueeze(-1) >> shifts
    x = x & 0xF
    x = x.reshape(M, K).to(torch.int8)
    return x - 8

def weight_quant_fp8_channel(tensor: torch.Tensor):
    """从脚本2继承并优化：BF16 -> Channel FP8"""
    # 确保输入是 float 格式进行计算
    tensor = tensor.float()
    qmax = torch.finfo(torch.float8_e4m3fn).max
    # Per-channel (dim=1) 计算 abs_max
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)
    scale = abs_max / qmax
    
    quantized = tensor / scale
    quantized = torch.clamp(quantized, -qmax, qmax)
    
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.float32)
def weight_quant_int8(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)

# --- 核心 Worker 进程 ---

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
    
    local_files = safetensor_files[rank::world_size]
    
    # 预加载辅助字典，用于跨文件找 scale
    loaded_files = {}
    def get_tensor(name):
        fname = weight_map[name]
        if fname not in loaded_files:
            loaded_files[fname] = load_file(os.path.join(input_dir, fname), device='cpu')
        return loaded_files[fname][name].to(device)

    for safetensor_file in tqdm(local_files, position=rank, desc=f"GPU {rank}"):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device=device)
        new_state_dict = {}

        for weight_name, weight in state_dict.items():
            # 目标：处理专家的打包权重
            if ".mlp.experts." in weight_name and "weight_packed" in weight_name:
                scale_name = weight_name.replace("weight_packed", "weight_scale")
                
                # 1. 反量化至 BF16 (GroupSize 32)
                # 获取对应的 G32 scale
                old_scale = get_tensor(scale_name)
                w_int4 = unpack_int4_from_int32(weight)
                
                M, K = w_int4.shape
                group_size = block_size
                num_groups = K // group_size
                
                w_bf16 = w_int4.view(M, num_groups, group_size).float()
                w_bf16 = w_bf16 * old_scale.view(M, num_groups, 1)
                w_bf16 = w_bf16.reshape(M, K).to(torch.bfloat16)
                if quant_type == "int8":
                    # 2. 重新量化至 Channel INT8
                    q_new, new_scale = weight_quant_int8(w_bf16)

                else:   
                # 2. 重新量化至 Channel FP8
                    q_new, new_scale = weight_quant_fp8_channel(w_bf16)
                
                # 3. 构造新的 state_dict
                # 注意：名称通常需要从 weight_packed 改回 weight 以适配通用 FP8 算子
                base_name = weight_name.replace("_packed", "")
                new_scale_name = scale_name # 保持原名或根据需要修改
                
                new_state_dict[base_name] = q_new
                new_state_dict[new_scale_name] = new_scale
                
            elif ".mlp.experts." in weight_name and ("weight_scale" in weight_name or "weight_shape" in weight_name):
                # 跳过原始的 scale 和 shape，因为我们在上面处理 packed 时已经一并处理并重新存入
                continue
            else:
                # 保持非专家层权重不变（或者根据需求也可以在此加入非专家层的量化）
                new_state_dict[weight_name] = weight

        save_file(new_state_dict, os.path.join(output_dir, file_name))

# --- 主控逻辑 ---

def main(input_path, output_path, quant_type, block_size):
    if not quant_type in ("int8", "fp8"):
        raise f"UNSUPPORT TYPE:{quant_type}"
    src_dir = Path(input_path)
    dst_dir = Path(output_path)
    dst_dir.mkdir(exist_ok=True)
    
    # 复制配置文件
    for file in src_dir.rglob("*"):
        # 只要是 .py 文件或者 .json 文件就复制
        if file.suffix in [".py", ".json"]:
            # 保持文件名不变，复制到目标目录
            target_path = dst_dir / file.name
            shutil.copy(file, target_path)
            print(f"已复制: {file.name}")
            
    index_path = os.path.join(input_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]
    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    
    world_size = torch.cuda.device_count()
    
    print(f"Starting conversion: G32-INT4 -> Channel-FP8 using {world_size} GPUs")
    mp.spawn(
        worker,
        args=(world_size, safetensor_files, weight_map, input_path, output_path,quant_type, block_size),
        nprocs=world_size,
        join=True,
    )

    # 修改 model.safetensors.index.json
    # 因为我们将 weight_packed 变成了 weight
    new_weight_map = {}
    for k, v in weight_map.items():
        # 1️⃣ 跳过 scale / shape
        if "weight_shape" in k:
            continue

        # 2️⃣ packed → weight
        if "weight_packed" in k:
            new_k = k.replace("weight_packed", "weight")
        else:
            new_k = k

        new_weight_map[new_k] = v

    model_index["weight_map"] = new_weight_map
    with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
        json.dump(model_index, f, indent=2)

    # 修改 config.json 声明 FP8 格式
    config_path = os.path.join(input_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        qtype = "int" if quant_type == "int8" else "float"
        qformat = "int-quantized" if quant_type == "int8" else "float-quantized"
        quant_info = {
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": {
                        "dynamic": True,
                        "group_size": None,
                        "num_bits": 8,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "strategy": "token",
                        "symmetric": True,
                        "type": qtype,
                    },
                    "weights": {
                        "dynamic": False,
                        "group_size": None,
                        "num_bits": 8,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "strategy": "channel",
                        "symmetric": True,
                        "type": qtype,
                    },
                }
            },
            "format": qformat,
            "ignore": ["lm_head",
                    "re:.*self_attn.*",
                    "re:.*shared_experts.*",
                    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*"
                    ],
            "quant_method": "compressed-tensors",
        }

    if "text_config" in config:
        config["text_config"]["quantization_config"] = quant_info
    else:
        config["compression_config"] = quant_info
    
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
def run(args: argparse.Namespace) -> None:
    logger.info("w4_dequant_channel converting %s", args.model)
    quant_type = getattr(args, "quant_type", "fp8")
    block_size = getattr(args, "block_size", 32)
    src = args.model
    dst = args.save_dir or f"{src}-CHANNEL-{quant_type}"
    main(src, dst, quant_type=quant_type, block_size=block_size)
    logger.info("w4_dequant_channel done, output: %s", dst)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-block-hf-path", type=str, required=True, help="DeepSeek G32 INT4 模型路径")
    parser.add_argument("--output-channel-hf-path", type=str, required=True, help="输出 Channel FP8 模型路径")
    parser.add_argument("--quant-type", type=str, default="fp8")
    parser.add_argument("--block-size", type=int, default=32)
    args = parser.parse_args()

    main(args.input_block_hf_path, args.output_channel_hf_path, args.quant_type,args.block_size)
    print("Success: Transferred G32-INT4 to Channel-FP8")