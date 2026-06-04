
import argparse
import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from utils.logging_config import get_logger

logger = get_logger(__name__)


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
    k_min: float = 0.95,
    k_max: float = 1.0,
    steps: int = 50,
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

def int8_block_to_channel(input: torch.Tensor, scale: torch.Tensor, quant_type: str = 'int8', block_size: int = 128):
    K, N = input.size()
    K1, N1 = scale.size()
    new_x = torch.empty([K1 * block_size, N1 * block_size],
                        dtype=input.dtype, device=input.device)
    new_x[:K, :N] = input

    fp8_v = new_x.view(K1, block_size, N1, block_size).permute(0, 2, 1, 3)
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    v = fp8_v.float() * scale
    v = v.permute(0, 2, 1, 3).reshape(new_x.shape)
    v = v[:K, :N]
    if quant_type in ["int8", "int4"]:
        q_weight, scale = weight_quant_int8(v)
    elif quant_type == "fp8":
        q_weight, scale = weight_quant_fp8(v)
    return q_weight, scale


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
            scale_inv_name = f"{weight_name}_scale_inv"
            if scale_inv_name in weight_map:
                # assert weight.element_size() == 2
                scale_inv = get_tensor(scale_inv_name)
                new_scale_name = scale_inv_name.replace("_scale_inv", "_scale")
                q, s = int8_block_to_channel(weight, scale_inv, quant_type, block_size)
                if quant_type == "int4" and ".mlp.experts." in weight_name:
                    q, scale_int4, _ = weight_quantint4_search_k(q)
                    s = s * scale_int4 / 16
                new_state_dict[weight_name] = q
                new_state_dict[new_scale_name] = s
            else:
                new_state_dict[weight_name] = weight

        save_file(
            new_state_dict,
            os.path.join(output_dir, file_name),
        )


def main(input_path, output_path, quant_type, block_size):
    # os.makedirs(output_path, exist_ok=True)
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
        new_k = k.replace("weight_scale_inv", "weight_scale")
        new_weight_map[new_k] = v
    model_index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"model.safetensors.index.json modified and saved to {index_path}")


    with open(config_path, "r") as f:
        config = json.load(f)
    config.pop("quantization_config", None)
    if quant_type == "int4":
        config["quantization_config"] = {
        "activation_scheme": "dynamic",
        "quant_method": "slimquant_w4a8",
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
            "ignore": ["lm_head"],
            "quant_method": "compressed-tensors",
        }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"config.json modified and saved to {config_path}")


def run(args: argparse.Namespace) -> None:
    logger.info("w8a8_block_cast_channel converting %s", args.model)
    quant_type = getattr(args, "quant_type", "int8")
    block_size = getattr(args, "block_size", 128)
    src = args.model
    dst = args.save_dir or f"{src}-CHANNEL-{quant_type}"
    main(src, dst, quant_type=quant_type, block_size=block_size)
    logger.info("w8a8_block_cast_channel done, output: %s", dst)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-block-hf-path", type=str, default="/models/DeepSeek-R1")
    parser.add_argument("--output-channel-hf-path", type=str, default="/models/DeepSeek-R1-Channel-int8")
    parser.add_argument("--quant-type", type=str, default="int8")
    parser.add_argument("--block-size", type=int, default=128)
    # parser.add_argument("--model-name", type=str, default="deepseek-ai/DeepSeek-R1")
    args = parser.parse_args()

    main(args.input_block_hf_path, args.output_channel_hf_path, args.quant_type,args.block_size)
    print("done")
