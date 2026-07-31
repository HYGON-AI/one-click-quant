
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

# from utils.logging_config import get_logger

# logger = get_logger(__name__)


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

def _detect_tp_shards(K: int, scale_rows: int, block_size: int = 128) -> int | None:
    """检测 TP-interleaved 布局。

    正常 block-FP8: scale_rows = ceil(K / block_size)
    TP-interleaved:  scale_rows = tp * ceil(K / (tp * block_size))

    返回 tp 值（>1 表示 interleaved），无法确定则返回 None。
    """
    if scale_rows == (K + block_size - 1) // block_size:
        return None  # 正常布局，无需特殊处理
    for tp in [8, 4, 2]:
        shard_rows = K // tp
        expected = tp * ((shard_rows + block_size - 1) // block_size)
        if expected == scale_rows:
            return tp
    return None


def _apply_per_shard(input: torch.Tensor, scale: torch.Tensor,
                     tp: int, fn, block_size: int = 128):
    """将 TP-interleaved 的 weight/scale 按 shard 拆分，逐 shard 调用 fn，再拼回。"""
    K, N = input.shape
    shard_rows = K // tp
    scale_shard_rows = scale.shape[0] // tp

    results_w = []
    results_s = []
    for r in range(tp):
        w_shard = input[r * shard_rows:(r + 1) * shard_rows, :]
        s_shard = scale[r * scale_shard_rows:(r + 1) * scale_shard_rows, :]
        res = fn(w_shard, s_shard, block_size)
        if isinstance(res, tuple):
            results_w.append(res[0])
            if len(res) > 1 and res[1] is not None:
                results_s.append(res[1])
        else:
            results_w.append(res)

    merged_w = torch.cat(results_w, dim=0)
    if results_s:
        merged_s = torch.cat(results_s, dim=0)
        return merged_w, merged_s
    return merged_w


def int8_block_to_channel(input: torch.Tensor, scale: torch.Tensor,
                          quant_type: str = 'int8', block_size: int = 128):
    """将 block-FP8 权重转换为 channel 量化。

    自动检测 TP-interleaved 布局（shard 独立量化后拼接），
    若检测到则逐 shard 独立转换。
    """
    K, N = input.size()
    tp = _detect_tp_shards(K, scale.shape[0], block_size)

    def _convert(w, s, bs):
        K1, N1 = s.shape
        new_x = torch.empty([K1 * bs, N1 * bs], dtype=w.dtype, device=w.device)
        new_x[:w.shape[0], :] = w
        fp8_v = new_x.view(K1, bs, N1, bs).permute(0, 2, 1, 3)
        s_exp = s.unsqueeze(-1).unsqueeze(-1)
        v = fp8_v.float() * s_exp
        v = v.permute(0, 2, 1, 3).reshape(new_x.shape)[:w.shape[0], :]
        if quant_type in ("int8", "int4"):
            return weight_quant_int8(v)[:2]
        elif quant_type == "fp8":
            return weight_quant_fp8(v)
        return v

    if tp:
        return _apply_per_shard(input, scale, tp, _convert, block_size)
    return _convert(input, scale, block_size)


def dequantize_block_fp8(input: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """将 block-FP8 权重反量化为 BF16。

    自动检测 TP-interleaved 布局（shard 独立量化后拼接），
    若检测到则逐 shard 独立反量化。
    """
    K, N = input.size()
    tp = _detect_tp_shards(K, scale.shape[0], block_size)

    def _dequant(w, s, bs):
        K1, N1 = s.shape
        padded = torch.zeros([K1 * bs, N1 * bs], dtype=w.dtype, device=w.device)
        padded[:w.shape[0], :] = w
        blocks = padded.view(K1, bs, N1, bs).permute(0, 2, 1, 3)
        dequant = (blocks.float() * s.unsqueeze(-1).unsqueeze(-1))
        return dequant.permute(0, 2, 1, 3).reshape(padded.shape)[:w.shape[0], :].to(torch.bfloat16)

    if tp:
        return _apply_per_shard(input, scale, tp, _dequant, block_size)
    return _dequant(input, scale, block_size)


def _should_skip_weight(weight_name: str, skip_prefixes: set[str] | None) -> bool:
    """检查权重名是否属于需要跳过的层。"""
    if not skip_prefixes:
        return False
    return any(prefix in weight_name for prefix in skip_prefixes)


def worker(
    rank: int,
    world_size: int,
    safetensor_files: list,
    weight_map: dict,
    input_dir: str,
    output_dir: str,
    quant_type: str,
    block_size: int,
    skip_prefixes: set[str] | None = None,
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
                scale_inv = get_tensor(scale_inv_name)
                if _should_skip_weight(weight_name, skip_prefixes):
                    # 跳过：反量化到 BF16，不保存 scale
                    dequant = dequantize_block_fp8(weight, scale_inv, block_size)
                    new_state_dict[weight_name] = dequant
                else:
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


def _get_ignore_list(weight_map: dict) -> list[str]:
    """扫描 weight_map，找出所有未量化的层前缀用于 ignore 列表。

    原理：有 _scale 或 _scale_inv 的权重是量化过的，没有的是 BF16。
    targets=["Linear"] 只匹配 nn.Linear，所以只需针对 Linear 层。
    提取 .weight 后缀的 key，检查是否有对应的 scale 条目，
    没有的话将层名前缀加入 ignore。
    """
    scale_set = {k for k in weight_map if k.endswith("_scale") or k.endswith("_scale_inv")}
    ignore = []
    for key in weight_map:
        if key.endswith(".weight") and f"{key}_scale" not in scale_set and f"{key}_scale_inv" not in scale_set:
            prefix = key[: -len(".weight")]
            ignore.append(prefix)
    return sorted(ignore)


def main(input_path, output_path, quant_type, block_size, skip_layer_indices: list[int] | None = None):
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

    # 构建 skip_prefixes：用于匹配权重名中是否包含跳过的层
    skip_prefixes: set[str] | None = None
    if skip_layer_indices:
        skip_prefixes = set()
        for idx in skip_layer_indices:
            skip_prefixes.add(f"model.layers.{idx}.")
        print(f"skip_layers: {skip_layer_indices} → {len(skip_prefixes)} prefixes")

    safetensor_files = sorted(glob(os.path.join(input_path, "*.safetensors")))
    print(f"safetensor_files:{input_path}")
    world_size = torch.cuda.device_count()
    assert world_size > 0, "No CUDA devices found"

    mp.spawn(
        worker,
        args=(world_size, safetensor_files, weight_map, input_path, output_path,
              quant_type, block_size, skip_prefixes),
        nprocs=world_size,
        join=True,
    )
    new_weight_map = {}
    for k, v in weight_map.items():
        new_k = k.replace("weight_scale_inv", "weight_scale")
        new_weight_map[new_k] = v

    # 清除跳过层的 scale 条目（它们没有被保存到 safetensors）
    if skip_prefixes:
        skip_scale_keys = [
            k for k in new_weight_map
            if k.endswith("_scale") and _should_skip_weight(k, skip_prefixes)
        ]
        for k in skip_scale_keys:
            del new_weight_map[k]
        print(f"removed {len(skip_scale_keys)} scale entries for skipped layers")

    model_index["weight_map"] = new_weight_map
    with open(index_path, "w") as f:
        json.dump(model_index, f, indent=2)
    print(f"model.safetensors.index.json modified and saved to {index_path}")


    with open(config_path, "r") as f:
        config = json.load(f)
    old_quant_config = config.pop("quantization_config", {})
    old_ignored = (
        old_quant_config.get("ignored_layers", [])
        if isinstance(old_quant_config, dict) else []
    )
    # 合并自动检测结果和原始 ignored_layers
    auto_ignore = _get_ignore_list(new_weight_map)
    merged_ignore = sorted(set(auto_ignore + old_ignored))
    if old_quant_config:
        print(
            "旧 quantization_config (quant_method=%s) 已被 compression_config 替代，"
            "自动检测 ignore=%d 项，原 ignored_layers=%d 项，合并后=%d 项。",
            old_quant_config.get("quant_method", "unknown"),
            len(auto_ignore),
            len(old_ignored),
            len(merged_ignore),
        )
    if quant_type == "int4":
        config["quantization_config"] = {
            "activation_scheme": "dynamic",
            "quant_method": "slimquant_w4a8",
            "modules_to_not_convert": merged_ignore,
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
            "ignore": merged_ignore,
            "quant_method": "compressed-tensors",
        }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"config.json modified and saved to {config_path}")


def run(args: argparse.Namespace) -> None:
    print("w8a8_block_cast_channel converting %s", args.model)
    quant_type = getattr(args, "quant_type", "int8")
    block_size = getattr(args, "block_size", 128)
    skip_layers_str = getattr(args, "skip_layers", "") or ""
    skip_layer_indices = None
    if skip_layers_str.strip():
        skip_layer_indices = [int(x.strip()) for x in skip_layers_str.split(",") if x.strip()]
        print(f"skip_layer_indices: {skip_layer_indices}")
    src = args.model
    dst = args.save_dir or f"{src}-CHANNEL-{quant_type}"
    main(src, dst, quant_type=quant_type, block_size=block_size, skip_layer_indices=skip_layer_indices)
    print("w8a8_block_cast_channel done, output: %s", dst)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-block-hf-path", type=str, default="/models/DeepSeek-R1")
    parser.add_argument("--output-channel-hf-path", type=str, default="/models/DeepSeek-R1-Channel-int8")
    parser.add_argument("--quant-type", type=str, default="int8")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--skip-layers",
        type=str,
        default="",
        help="Comma-separated layer indices to dequantize to BF16 instead of "
        "converting to channel-quantized format (e.g. '0,8,16').",
    )
    args = parser.parse_args()

    skip_layer_indices = None
    if args.skip_layers.strip():
        skip_layer_indices = [int(x.strip()) for x in args.skip_layers.split(",") if x.strip()]

    main(args.input_block_hf_path, args.output_channel_hf_path, args.quant_type,
         args.block_size, skip_layer_indices=skip_layer_indices)
    print("done")
