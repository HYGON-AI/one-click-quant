# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.3 (GlmMoeDsaForCausalLM) BF16 / FP8 -> 混合精度 INT4/INT8 W4A8+W8A8 转换脚本
(RTN, 无数据集校准; compressed-tensors 双 config_groups)

量化范围与 GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200
(quantize_glm5_3_flash_int4_channel.py --no-mla-to-bf16 --linear-attn-int8
 --kda-int8-scope qkvo --indexer-int8) 一一对应, 换到 GLM-5.3 的结构上:

  GLM-5.3 结构 (config.json):
    num_hidden_layers=78 (layers.0-77), num_nextn_predict_layers=1 (MTP = layers.78)
    mlp_layer_types: layers.0-2 dense, layers.3-78 sparse (256 routed + 1 shared)
    所有层都是 MLA + DSA, 没有线性注意力 (Flash 的 KDA q/k/v/o 在这里不存在)
    indexer_types: full 层 (0-2,6,10,...,74,78) 有 indexer 权重, shared 层复用前一层

  ┌────────────────────────────────────────────────┬──────────────┬────────────────────┐
  │ 模块                                            │ 本脚本        │ Flash 参考          │
  ├────────────────────────────────────────────────┼──────────────┼────────────────────┤
  │ routed experts gate/up/down (layers.3-77)       │ INT4 packed  │ INT4 (L3-44)       │
  │ routed experts (MTP layers.78)                  │ INT8         │ INT8 (L45)         │
  │ shared_experts gate/up/down (layers.3-78)       │ INT8         │ INT8               │
  │ dense MLP gate/up/down (layers.0-2)             │ INT8         │ INT8               │
  │ MLA q_a/q_b/kv_a_proj_with_mqa (layers.0-78)    │ INT8         │ INT8 (DSA 层)       │
  │ o_proj (layers.0-78)                            │ INT8         │ INT8               │
  │ indexer wq_b / wk (full 层 + MTP)               │ INT8         │ INT8               │
  │ kv_b_proj / indexer weights_proj                │ BF16 (ignore)│ BF16               │
  │ router gate / norms / indexer k_norm / embed /  │ BF16 (ignore)│ BF16               │
  │ lm_head / MTP eh_proj,enorm,hnorm,shared_head   │              │                    │
  └────────────────────────────────────────────────┴──────────────┴────────────────────┘
  可选 --mtp-int4: MTP 的 routed experts 也走 INT4 (默认关闭, 与参考一致为 INT8)。

量化公式 (与 quantize_glm5_3_flash_int4_channel.py / MoE-Quant 一致, 对称 per-channel):
  INT4: scale = absmax/7 (落盘 bf16), q = round(w/scale) ∈ [-8,7], 8 个沿 in_dim 打包进 int32
        (compressed-tensors pack-quantized, element i 在 bits [4i, 4i+4), 存 q+8)
        与 Flash 脚本的唯一差别: 先把 scale 取整到 bf16 再量化, 保证量化与反量化用的是
        同一个 scale (Flash 脚本用 fp32 scale 量化、存 bf16 scale)。
  INT8: scale = absmax/127 (落盘 fp32), q = round(w/scale) ∈ [-127,127] (int-quantized)

推理端 (DCU sglang, GlmMoeDsaForCausalLM / GlmMoeDsaForCausalLMNextN) 相关约束:
  - routed experts INT4 由 HCUCompressedTensorsW4A8Int8DynamicMoE 加载 (W4A8 动态 per-token)。
  - q_a_proj + kv_a_proj_with_mqa 运行时融合为 fused_qkv_a_proj_with_mqa, 两者同为 INT8。
  - shared experts 必须不融合进 routed experts 的 FusedMoE (精度不同)。DCU (gfx < 9.4)
    上 sglang 自动关闭该融合; 其它平台请加 --disable-shared-experts-fusion。
  - indexer wk 在 CUDA 上会与 weights_proj 融合成 bf16 的 wk_weights_proj, 而 INT8 wk 无法
    装进该融合参数; DCU 不融合。CUDA 上需设置 SGLANG_DISABLE_DSA_INDEXER_FUSION=1。
  - MTP 层在运行时不叫 model.layers.78: 解码块在 model.decoder.*, eh_proj/enorm/hnorm/
    shared_head 在 model.*。因此 ignore 规则不带 layers.N 锚点, group_int4 的 target 用显式
    层号 (3..77), MTP experts 自然落到 group_int8。脚本启动时会用运行时模块名做自检。

用法
====
  python quantize_glm5_3_int4_channel.py \\
      --input-dir  /llm-models/GLM-5.3-BF16 \\
      --output-dir /llm-models/GLM-5.3-CHANNEL-INT4-w4a8

  # 冒烟: 只转换指定 shard (不写 config / index)
  python quantize_glm5_3_int4_channel.py --input-dir ... --output-dir ... \\
      --only-files model-00001-of-00282.safetensors,model-00282-of-00282.safetensors

  # 只做自检和范围统计, 不写任何文件
  python quantize_glm5_3_int4_channel.py --input-dir ... --output-dir ... --dry-run

前置条件
========
  - torch, safetensors, tqdm
  - compressed_tensors 可选 (提供官方 pack_to_int32; 缺失时用位级等价的内置实现)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, BooleanOptionalAction
from collections import Counter
from dataclasses import dataclass
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

try:
    from compressed_tensors.compressors import pack_to_int32 as _ct_pack_to_int32
    _HAVE_COMPRESSED_TENSORS_PACK = True
except ImportError:
    _HAVE_COMPRESSED_TENSORS_PACK = False

    def _ct_pack_to_int32(value: torch.Tensor, num_bits: int) -> torch.Tensor:
        """Fallback for compressed_tensors pack_to_int32 (packed_dim=1, num_bits <= 8)."""
        if value.dtype is not torch.int8:
            raise ValueError("Tensor must be quantized to torch.int8 before packing")
        per_lane = 32 // num_bits
        if value.shape[-1] % per_lane != 0:
            raise ValueError(f"last dim must be divisible by {per_lane}, got {tuple(value.shape)}")
        lanes = (value.to(torch.int32) + (1 << (num_bits - 1))).view(*value.shape[:-1], -1, per_lane)
        shifts = torch.arange(per_lane, device=value.device, dtype=torch.int32) * num_bits
        return (lanes << shifts).sum(dim=-1).to(torch.int32)


# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

INT4_BITS = 4
INT4_QMAX = 7.0
INT8_QMAX = 127.0

_SAFETENSORS_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}


# --------------------------------------------------------------------------- #
# 层号: 全部由 config.json 推导, 不硬编码 78
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Layout:
    num_hidden_layers: int
    mtp_layers: tuple[int, ...]
    int4_layers: tuple[int, ...]   # routed experts 走 INT4 的层


def load_layout(input_dir: str, mtp_int4: bool) -> Layout:
    with open(os.path.join(input_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    num_hidden = int(text["num_hidden_layers"])
    mtp = tuple(range(num_hidden, num_hidden + int(text.get("num_nextn_predict_layers", 0))))
    mlp_types = list(text.get("mlp_layer_types") or [])
    if mlp_types:
        sparse = tuple(i for i, t in enumerate(mlp_types) if t == "sparse" and i < num_hidden)
    else:
        sparse = tuple(range(int(text.get("first_k_dense_replace", 0)), num_hidden))
    if not sparse:
        raise ValueError("config.json 中没有 sparse (MoE) 层")
    if mtp_int4 and not mtp:
        raise ValueError("--mtp-int4 但 config.json 的 num_nextn_predict_layers=0")
    return Layout(num_hidden, mtp, sparse + (mtp if mtp_int4 else ()))


# --------------------------------------------------------------------------- #
# checkpoint 键名分类
# --------------------------------------------------------------------------- #

_ROUTED_EXPERT_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
_INT8_RES = (
    # dense MLP (layers.0-2)
    re.compile(r"(?:^|\.)layers\.\d+\.mlp\.(?:gate_proj|up_proj|down_proj)\.weight$"),
    # shared experts
    re.compile(r"(?:^|\.)layers\.\d+\.mlp\.shared_experts\.(?:gate_proj|up_proj|down_proj)\.weight$"),
    # MLA: q_a / q_b / kv_a_proj_with_mqa / o_proj (kv_b_proj 保持 BF16)
    re.compile(r"(?:^|\.)layers\.\d+\.self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj)\.weight$"),
    # DSA indexer wq_b / wk (weights_proj 保持 BF16)
    re.compile(r"(?:^|\.)layers\.\d+\.self_attn\.indexer\.(?:wq_b|wk)\.weight$"),
)


def classify(name: str, layout: Layout) -> str:
    """checkpoint 键 -> 'int4' | 'int8' | 'keep'。只对 2D Linear 的 .weight 返回量化类别。"""
    m = _ROUTED_EXPERT_RE.search(name)
    if m is not None:
        layer = int(m.group(1))
        if layer in layout.int4_layers:
            return "int4"
        if layer in layout.mtp_layers:
            return "int8"   # MTP routed experts (对齐参考: moe_int8_layers=[MTP])
        raise ValueError(f"routed expert {name} 的层号 {layer} 既不是 MoE 主干也不是 MTP")
    if any(p.search(name) for p in _INT8_RES):
        return "int8"
    return "keep"


# --------------------------------------------------------------------------- #
# 运行时 (sglang) 模块名 + compressed-tensors config
# --------------------------------------------------------------------------- #

def runtime_module_name(ckpt_module: str, layout: Layout) -> str:
    """checkpoint 模块名 -> sglang 运行时模块名 (镜像 GlmMoeDsaForCausalLMNextN._map_mtp_ckpt_name)。"""
    for layer in layout.mtp_layers:
        prefix = f"model.layers.{layer}"
        if ckpt_module.startswith(prefix + "."):
            if any(p in ckpt_module for p in ("shared_head.norm", "eh_proj", "enorm", "hnorm")):
                return ckpt_module.replace(prefix, "model", 1)
            return ckpt_module.replace(prefix, "model.decoder", 1)
    return ckpt_module


def build_ignore_list() -> list[str]:
    """BF16 保留模块。规则不带 layers.N 锚点, 以同时覆盖 MTP 的 model.decoder.* / model.* 名。"""
    return [
        "lm_head",
        r"re:.*embed_tokens.*",
        r"re:^model\.norm$",
        r"re:.*(?:input_layernorm|post_attention_layernorm)$",
        # MoE router (vLLM/sglang 的 gate 本身不带 quant_config)
        r"re:.*mlp\.gate$",
        r"re:.*mlp\.gate\..*",
        # MLA
        r"re:.*self_attn\.(?:q_a_layernorm|kv_a_layernorm)$",
        r"re:.*self_attn\.kv_b_proj$",
        # DSA indexer: 只有 wq_b / wk 量化
        r"re:.*self_attn\.indexer\.k_norm(?:\..*)?$",
        r"re:.*self_attn\.indexer\.weights_proj$",
        r"re:.*self_attn\.indexer\.wk_weights_proj$",
        # MTP 专用模块 (运行时为 model.eh_proj / model.enorm / model.hnorm / model.shared_head.*)
        r"re:.*\.eh_proj$",
        r"re:.*\.enorm$",
        r"re:.*\.hnorm$",
        r"re:.*shared_head\..*",
    ]


def _int8_activations() -> dict:
    return {
        "dynamic": True, "group_size": None, "num_bits": 8,
        "observer": "minmax", "observer_kwargs": {},
        "strategy": "token", "symmetric": True, "type": "int",
    }


def _weights(num_bits: int) -> dict:
    return {
        "actorder": None, "block_structure": None, "dynamic": False,
        "group_size": -1, "num_bits": num_bits, "observer": "minmax",
        "observer_kwargs": {}, "strategy": "channel", "symmetric": True, "type": "int",
    }


def int4_target_regex(layout: Layout) -> str:
    return (
        r"re:.*\.layers\.(?:" + "|".join(str(i) for i in layout.int4_layers)
        + r")\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$"
    )


def build_compression_config(layout: Layout, mode: str) -> dict:
    """group_int4: 主干 routed experts (显式层号); group_int8: 其余未 ignore 的 Linear。"""
    return {
        "config_groups": {
            "group_int4": {
                "targets": [int4_target_regex(layout)],
                "input_activations": _int8_activations() if mode == "w4a8" else None,
                "output_activations": None,
                "format": "pack-quantized",
                "weights": _weights(4),
            },
            "group_int8": {
                "targets": ["Linear"],
                "input_activations": _int8_activations(),
                "output_activations": None,
                "format": "int-quantized",
                "weights": _weights(8),
            },
        },
        "format": "pack-quantized",
        "ignore": build_ignore_list(),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


def _match(name: str, rules: list[str]) -> str | None:
    """与 compressed-tensors / sglang 的匹配一致: re: 走 re.match, 否则精确相等。"""
    for rule in rules:
        if rule.startswith("re:"):
            if re.match(rule[3:], name):
                return rule
        elif rule == name:
            return rule
    return None


# --------------------------------------------------------------------------- #
# 自检: 在运行时模块名上, config 解析出的精度必须与脚本写出的精度一致
# --------------------------------------------------------------------------- #

def _shard_tensor_meta(input_dir: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    meta: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in sorted(glob(os.path.join(input_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                meta[key] = (sl.get_dtype(), tuple(sl.get_shape()))
    return meta


def self_check_scope(
    meta: dict[str, tuple[str, tuple[int, ...]]], layout: Layout, mode: str
) -> Counter:
    """遍历全部 2D .weight, 断言 '脚本写什么' == '运行时按 config 加载什么'。"""
    cfg = build_compression_config(layout, mode)
    ignore = cfg["ignore"]
    int4_target = cfg["config_groups"]["group_int4"]["targets"]
    stats: Counter = Counter()
    errors: list[str] = []
    for key, (_, shape) in meta.items():
        if not key.endswith(".weight") or len(shape) != 2:
            continue
        module = key[: -len(".weight")]
        if module.startswith("model.embed_tokens") or module == "lm_head":
            stats[("keep", "embed/lm_head")] += 1
            continue
        written = classify(key, layout)
        rt = runtime_module_name(module, layout)
        if _match(rt, ignore):
            expected = "keep"
        elif _match(rt, int4_target):
            expected = "int4"
        else:
            expected = "int8"   # group_int8 targets=["Linear"] 兜底
        if written != expected:
            errors.append(f"{key}: 脚本写 {written}, 运行时 {rt} 按 config 解析为 {expected}")
        generic = re.sub(r"experts\.\d+\.", "experts.E.", re.sub(r"layers\.\d+\.", "layers.N.", module))
        stats[(written, generic)] += 1
    if errors:
        raise AssertionError(
            f"量化范围自检失败 ({len(errors)} 处), 前 10 处:\n  " + "\n  ".join(errors[:10])
        )
    # INT4 packing 要求 in_dim % 8 == 0
    for key, (_, shape) in meta.items():
        if key.endswith(".weight") and len(shape) == 2 and classify(key, layout) == "int4":
            assert shape[1] % 8 == 0, f"{key} in_dim={shape[1]} 不能被 8 整除"
    return stats


def self_check_pack() -> None:
    """pack 位布局与 compressed-tensors 规范一致: q=[-8,-1,0,1,7,-8,7,0] -> 0x8F0F9870。"""
    q = torch.tensor([[-8, -1, 0, 1, 7, -8, 7, 0]], dtype=torch.int8)
    got = int(_ct_pack_to_int32(q, INT4_BITS).item())
    assert got == -1894803344, f"pack self-check failed: got={got}"


# --------------------------------------------------------------------------- #
# 反量化 / 量化
# --------------------------------------------------------------------------- #

def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8 blockwise (N,K) + scale (ceil(N/b), ceil(K/b)) -> fp32 (N,K); 兼容非整除尾块。"""
    s_rows, s_cols = scale.shape
    n, k = weight.shape
    rb, cb = -(-n // s_rows), -(-k // s_cols)
    s = scale.float().repeat_interleave(rb, 0)[:n].repeat_interleave(cb, 1)[:, :k]
    return weight.float() * s


def quantize_int8_channelwise(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 INT8 per-output-channel: int8 (N,K) ∈ [-127,127], fp32 scale (N,1)。"""
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT8_QMAX
    q = torch.round(w / scale).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale


def quantize_int4_channelwise_packed(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对称 INT4 per-output-channel + int32 pack: (packed (N,K/8), shape (2,), bf16 scale (N,1))。"""
    w = w.float()
    n, k = w.shape
    scale = (w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / INT4_QMAX).to(torch.bfloat16)
    q = torch.round(w / scale.float()).clamp(-8, 7).to(torch.int8)
    packed = _ct_pack_to_int32(q, INT4_BITS)
    return packed.contiguous(), torch.tensor([n, k], dtype=torch.int32), scale.contiguous()


# --------------------------------------------------------------------------- #
# 单 shard 转换
# --------------------------------------------------------------------------- #

_STATS_KEYS = ("int4", "int8", "fp8_to_bf16", "kept", "scales_dropped")


def convert_one_file(
    input_path: str, output_path: str, layout: Layout, device: torch.device | str = "cpu"
) -> dict[str, int]:
    """分支: .weight_scale_inv 丢弃 -> INT4 -> INT8 -> FP8 反量化为 BF16 -> 原样透传。"""
    device = torch.device(device)
    stats = {k: 0 for k in _STATS_KEYS}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        state_dict = {name: f.get_tensor(name) for name in f.keys()}

    out: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith(".weight_scale_inv"):
            stats["scales_dropped"] += 1
            continue
        kind = classify(name, layout) if (name.endswith(".weight") and tensor.ndim == 2) else "keep"
        scale_inv = state_dict.get(name + "_scale_inv")
        prefix = name[: -len(".weight")] if name.endswith(".weight") else name

        if kind == "keep":
            if scale_inv is not None:   # FP8 源里非量化目标 (理论上只有 kv_b_proj 等) -> BF16
                out[name] = dequant_fp8_blockwise(tensor.to(device), scale_inv.to(device)).bfloat16().cpu()
                stats["fp8_to_bf16"] += 1
            else:
                out[name] = tensor
                stats["kept"] += 1
            continue

        w = tensor.to(device)
        w = dequant_fp8_blockwise(w, scale_inv.to(device)) if scale_inv is not None else w.float()
        if kind == "int4":
            packed, shape, scale = quantize_int4_channelwise_packed(w)
            out[f"{prefix}.weight_packed"] = packed.cpu()
            out[f"{prefix}.weight_shape"] = shape
            out[f"{prefix}.weight_scale"] = scale.cpu()
        else:
            q, scale = quantize_int8_channelwise(w)
            out[name] = q.cpu()
            out[f"{prefix}.weight_scale"] = scale.cpu()
        stats[kind] += 1
        del w

    save_file(out, output_path, metadata={"format": "pt"})
    return stats


def _worker(task: tuple[str, str, Layout, int, int]) -> dict[str, int]:
    input_path, output_path, layout, gpu_id, num_threads = task
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    return convert_one_file(input_path, output_path, layout, torch.device(f"cuda:{gpu_id}"))


def convert_model(
    files: list[str], output_dir: str, layout: Layout,
    device_kind: str, gpu_ids: str | None, workers: int | None, num_threads: int,
) -> tuple[dict[str, int], list[str]]:
    os.makedirs(output_dir, exist_ok=True)
    stats = {k: 0 for k in _STATS_KEYS}
    use_cuda = device_kind == "cuda" or (device_kind == "auto" and torch.cuda.is_available())
    if device_kind == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 但 torch.cuda.is_available() == False")
    if not use_cuda:
        for path in tqdm(files, desc="Converting on cpu"):
            for k, v in convert_one_file(path, os.path.join(output_dir, os.path.basename(path)), layout).items():
                stats[k] += v
        return stats, ["cpu"]

    ids = [int(x) for x in gpu_ids.split(",")] if gpu_ids else list(range(torch.cuda.device_count()))
    n_workers = min(workers or len(ids), len(ids), max(1, len(files)))
    tasks = [
        (path, os.path.join(output_dir, os.path.basename(path)), layout, ids[i % len(ids)], num_threads)
        for i, path in enumerate(files)
    ]
    if n_workers <= 1:
        for task in tqdm(tasks, desc=f"Converting on cuda:{ids[0]}"):
            for k, v in _worker(task).items():
                stats[k] += v
    else:
        import torch.multiprocessing as mp
        with mp.get_context("spawn").Pool(processes=n_workers) as pool:
            for per in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks),
                            desc=f"Converting on {len(ids)} GPU(s) ({n_workers} workers)"):
                for k, v in per.items():
                    stats[k] += v
    return stats, [f"cuda:{i}" for i in ids]


# --------------------------------------------------------------------------- #
# config / index / summary
# --------------------------------------------------------------------------- #

def write_metadata(input_dir: str, output_dir: str, layout: Layout, mode: str) -> None:
    """复制非权重文件, 重建 index (只读 header), 写入 quantization_config。"""
    for path in glob(os.path.join(input_dir, "*")):
        name = os.path.basename(path)
        if os.path.isfile(path) and not name.endswith(".safetensors") and name != "model.safetensors.index.json":
            shutil.copyfile(path, os.path.join(output_dir, name))

    weight_map: dict[str, str] = {}
    total_size = 0
    for path in sorted(glob(os.path.join(output_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                numel = 1
                for d in sl.get_shape():
                    numel *= d
                weight_map[key] = os.path.basename(path)
                total_size += numel * _SAFETENSORS_ITEMSIZE[sl.get_dtype()]
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2, sort_keys=True)

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    config.pop("compression_config", None)
    config.pop("quantization_config", None)
    config["quantization_config"] = build_compression_config(layout, mode)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def scope_table(stats: Counter) -> list[str]:
    lines = []
    for (kind, generic), count in sorted(stats.items(), key=lambda x: (x[0][0], x[0][1])):
        lines.append(f"  {kind.upper():5s} x{count:<6d} {generic}")
    return lines


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = ArgumentParser(
        description="GLM-5.3 BF16/FP8 -> mixed INT4 (routed experts) / INT8 W4A8+W8A8, RTN channel-wise.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="/llm-models/GLM-5.3-BF16")
    parser.add_argument("--output-dir", default="/llm-models/GLM-5.3-CHANNEL-INT4-w4a8")
    parser.add_argument(
        "--mode", choices=["w4a8", "w4a16"], default="w4a8",
        help="INT4 组的激活声明 (权重字节相同)。INT8 组固定为 W8A8 动态 per-token。",
    )
    parser.add_argument(
        "--mtp-int4", action=BooleanOptionalAction, default=False,
        help="MTP 层的 routed experts 也量化为 INT4 (默认 INT8, 与 Flash 参考一致)。",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--gpu-ids", default=None, help="逗号分隔的 GPU 编号, 默认全部。")
    parser.add_argument("--workers", type=int, default=None, help="并行进程数, 默认 = GPU 数。")
    parser.add_argument("--num-threads", type=int, default=8, help="每个 worker 的 CPU 线程数。")
    parser.add_argument(
        "--only-files", default=None,
        help="逗号分隔的 shard 文件名, 只转换这些 (冒烟测试; 不写 config / index)。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只做自检并打印量化范围, 不写文件。")
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir 与 output-dir 不能相同")
    torch.set_num_threads(args.num_threads)

    layout = load_layout(args.input_dir, args.mtp_int4)
    self_check_pack()
    meta = _shard_tensor_meta(args.input_dir)
    stats_scope = self_check_scope(meta, layout, args.mode)

    print(f"GLM-5.3 -> mixed INT4/INT8 ({args.mode})")
    print(f"  input_dir   : {args.input_dir}")
    print(f"  output_dir  : {args.output_dir}")
    print(f"  INT4 layers : routed experts layers {list(layout.int4_layers)}")
    print(f"  MTP layers  : {list(layout.mtp_layers)} (routed experts {'INT4' if args.mtp_int4 else 'INT8'})")
    print("  scope self-check passed; 2D Linear weights by precision:")
    print("\n".join(scope_table(stats_scope)))
    if args.dry_run:
        return

    files = sorted(glob(os.path.join(args.input_dir, "*.safetensors")))
    if args.only_files:
        wanted = {x.strip() for x in args.only_files.split(",") if x.strip()}
        files = [p for p in files if os.path.basename(p) in wanted]
        missing = wanted - {os.path.basename(p) for p in files}
        if missing:
            raise FileNotFoundError(f"--only-files 中不存在的 shard: {sorted(missing)}")

    stats, devices = convert_model(
        files, args.output_dir, layout, args.device, args.gpu_ids, args.workers, args.num_threads
    )
    if args.only_files:
        print(json.dumps(stats, indent=2))
        print(f"\n[smoke] converted {len(files)} shard(s); config / index not written.")
        return

    write_metadata(args.input_dir, args.output_dir, layout, args.mode)
    expected = {
        "int4": sum(c for (k, _), c in stats_scope.items() if k == "int4"),
        "int8": sum(c for (k, _), c in stats_scope.items() if k == "int8"),
    }
    for kind, count in expected.items():
        if stats[kind] != count:
            raise RuntimeError(f"写出的 {kind} 张量数 {stats[kind]} != 自检统计 {count}")
    with open(os.path.join(args.output_dir, "int4_conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dir": args.input_dir,
                "mode": args.mode,
                "mtp_int4": args.mtp_int4,
                "int4_layers": list(layout.int4_layers),
                "mtp_layers": list(layout.mtp_layers),
                "devices": devices,
                "used_compressed_tensors_pack": _HAVE_COMPRESSED_TENSORS_PACK,
                "stats": stats,
                "scope": {f"{k}:{g}": c for (k, g), c in sorted(stats_scope.items())},
            },
            f, indent=2, ensure_ascii=False,
        )
    print(json.dumps(stats, indent=2))
    print(f"\nDone! saved to {args.output_dir}")


if __name__ == "__main__":
    main()
