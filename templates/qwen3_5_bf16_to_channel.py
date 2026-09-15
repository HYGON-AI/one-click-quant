# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from multiprocessing import Manager
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from utils.logging_config import get_logger

logger = get_logger(__name__)

import re
def weight_quant_int8(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)

def weight_quant_fp8(tensor: torch.Tensor):
    assert tensor.dim() == 2
    qmax = torch.finfo(torch.float8_e4m3fn).max
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0].clamp(min=1e-12)  # [rows, 1]
    scale = abs_max / qmax  # [rows, 1]
    assert scale.shape == (tensor.shape[0], 1)
    quantized = tensor / scale
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.float8_e4m3fn), scale.to(torch.float32)

def weight_quant_int4(tensor: torch.Tensor, percentile: float):
    assert tensor.dim() == 2
    qmax = 7.0
    abs_value=torch.abs(tensor)
    sorted_matrix,_ = torch.sort(abs_value, dim=1)
    k=tensor.shape[1]
    index=int(k*percentile)
    index = min(index, k - 1)
    abs_max=sorted_matrix[:,index].reshape(-1,1)
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -8, 7).to(torch.int8)
    dequant = (quantized * scale)
    quantized_int8=quantized.to(torch.uint8)
    n, k = quantized.size()
    new_shape = (n, k // 2)
    quantized_int4= torch.empty(new_shape, dtype=torch.int8, device=tensor.device)
    #pack |w0w1|w2w3|w4w5.....
    a=quantized_int8[..., ::2]
    b = quantized_int8[..., 1::2]
    a_4bit = a
    b_4bit = b & 0x0F
    quantized_int4 = (a_4bit << 4) |  b_4bit
    quantized_int4=quantized_int4.contiguous().to(torch.int8)

    return quantized_int4 , scale.to(torch.float32), dequant

def weight_quantint4_search_k(
    tensor: torch.Tensor,
    k_min: float = 0.97,
    k_max: float = 1.0,
    steps: int = 30,
    metric: str = "mse",  # "mse" or "l2"
):

    assert tensor.dim() == 2
    percentiles = torch.linspace(k_min, k_max, steps, device=tensor.device)
    best_error = float("inf")
    best_k = None
    best_quant = None
    best_scale = None

    # q_int4, scale, dequant = weight_quant_int4(tensor, 0.98)
    # return q_int4, scale, 0
    for p in percentiles:
        p = float(p.item())

        q_int4, scale, dequant = weight_quant_int4(tensor, p)

        if metric == "mse":
            error = torch.mean((tensor - dequant) ** 2).item()
        elif metric == "l2":
            error = torch.norm(tensor.to(torch.float16) - dequant.to(torch.float16)).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if error < best_error:
            best_error = error
            best_k = p
            best_quant = q_int4
            best_scale = scale
    # print(f"best_k:{best_k}")
    return best_quant, best_scale, best_k

ignore_layers = [
      "re:.*norm.weight.*",
      "re:mtp.pre_fc_norm.*",
      "re:.*embed_tokens.*",
      "re:.*input_layernorm.*",
      "re:.*post_attention_layernorm.*",
      "re:.*mlp.gate.*",
      "re:.*mlp.shared_expert_gate.*",
      "re:.*self_attn.k_norm.*",
      "re:.*self_attn.q_norm.*",
      "re:.*linear_attn.norm.*",
      "re:.*linear_attn.conv1d.*",
      "re:.*linear_attn.A_log.*",
      "re:.*linear_attn.dt_bias.*",
      "re:.*visual.*",
      "re:.*mtp.fc.*",
      "re:.*pos_embed.*",
      "re:.*lm_head.*",
      "re:.*linear_attn.in_proj_a.*",
      "re:.*linear_attn.in_proj_b.*"
]
def is_ignored(weight_name):
    for pattern in ignore_layers:
        if pattern.startswith("re:"):
            regex = pattern[3:]
            if re.search(regex, weight_name):
                return True
        else:
            if pattern in weight_name:
                return True
    return False

def worker(
    rank: int,
    world_size: int,
    safetensor_files: list,
    weight_map: dict,
    input_dir: str,
    output_dir: str,
    quant_type: str,
    shared_weight_map,
    not_quant_layers
):
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)
    torch.set_default_dtype(torch.bfloat16)

    local_files = safetensor_files[rank::world_size]

    for safetensor_file in tqdm(local_files, position=rank):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device=device)
        new_state_dict = {}
        not_quant_layers=[]
        for weight_name, weight in state_dict.items():
            # if  any(layer in weight_name for layer in ignore_layers):
            if is_ignored(weight_name):
                not_quant_layers.append(weight_name)
                new_state_dict[weight_name] = weight
                shared_weight_map[weight_name] = file_name
            else:
                assert weight.dim() != 1, f">>>>>Can not quant layer: {weight_name}!<<<<<"
                if weight.dim() == 3:
                    E, N, K = weight.shape
                    weight = weight.reshape(E * N, K)
                else:
                    E = N = K = None
                if quant_type == 'fp8':
                    q, s = weight_quant_fp8(weight)
                else:
                    try:
                        q, s = weight_quant_int8(weight)
                    except Exception as e:
                        not_quant_layers.append(weight_name)
                        new_state_dict[weight_name] = weight
                        continue
                    if quant_type == "int4" and ".mlp.experts." in weight_name:
                        if K is not None:
                            K = K//2
                        q, scale_int4, _ = weight_quantint4_search_k(q)
                        s = s * scale_int4 / 16
                        # reshape 回原结构
                if E is not None:
                    q = q.reshape(E, N, K)
                    s = s.reshape(E, N, 1)
                # print(f"rank{rank} quant {weight_name} with shape {weight.shape} to {q.shape} with scale {s.shape}")        
                new_state_dict[weight_name] = q
                new_scale_name = f"{weight_name}_scale"
                new_state_dict[new_scale_name] = s
                shared_weight_map[new_scale_name] = file_name
                shared_weight_map[weight_name] = file_name
                
        save_file(
            new_state_dict,
            os.path.join(output_dir, file_name),
        )
        # print(f"Process {rank} ignore {not_quant_layers}")
    


def main(input_path, output_path, quant_type):

    if not quant_type in ("int8", "fp8", "int4"):
        raise f"UNSUPPORT TYPE:{quant_type}"
    
    src_dir = Path(input_path)
    dst_dir = Path(output_path)
    dst_dir.mkdir(exist_ok=True)
    for file in src_dir.rglob("*.json"):
        shutil.copy(file, dst_dir / file.name)

    index_path = os.path.join(output_path, "model.safetensors.index.json")
    config_path = os.path.join(output_path, "config.json")

    if not os.path.exists(index_path) or not os.path.exists(config_path):
        raise f"CAN NOT FIND FILE: {index_path} {config_path}"
    with open(config_path, "r") as f:
        config = json.load(f)
    # not_quant_layers=config["quantization_config"]["modules_to_not_convert"]
    with open(index_path, "r") as f:
        model_index = json.load(f)

    weight_map = model_index["weight_map"]

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    print(f"safetensor_files:{input_path}")

    world_size = torch.cuda.device_count()
    assert world_size > 0, "No CUDA devices found"

    manager = Manager()
    shared_weight_map = manager.dict()

    mp.spawn(
        worker,
        args=(
            world_size,
            safetensor_files,
            weight_map,
            input_path,
            output_path,
            quant_type,
            shared_weight_map,
            []
        ),
        nprocs=world_size,
        join=True,
    )

    # 转成普通 dict
    new_weight_map = dict(shared_weight_map)

    model_index["weight_map"] = new_weight_map

    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)

    print(f"model.safetensors.index.json modified and saved to {index_path}")
    
    

    config.pop("quantization_config", None)
    if quant_type == "int4":
        config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "quant_method": "slimquant_w4a8",
        "modules_to_not_convert": ignore_layers,
        }
    else:
        qtype = "int" if quant_type == "int8" else "float"
        qformat = "int-quantized" if quant_type == "int8" else "float-quantized"
        config["compression_config"] = {
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
            "ignore": ignore_layers,
            "quant_method": "compressed-tensors",
        }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"config.json modified and saved to {config_path}")


def run(args: argparse.Namespace) -> None:
    logger.info("qwen3_5_bf16_to_channel converting %s", args.model)
    quant_type = getattr(args, "quant_type", "int8")
    src = args.model
    dst = args.save_dir or f"{src}-CHANNEL-{quant_type}"
    main(src, dst, quant_type)
    logger.info("qwen3_5_bf16_to_channel done, output: %s", dst)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, default="/models/Qwen3.5-397B-A17B")
    parser.add_argument("--output-path", type=str, default="/models/Qwen3.5-397B-A17B-Channel-int8")
    # parser.add_argument("--model-seri", type=str, default="qwen3.5")
    parser.add_argument("--quant-type", type=str, default="int8")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.quant_type)
    print("done")
