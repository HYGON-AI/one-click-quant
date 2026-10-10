# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek-V4.1-Flash FP4/FP8 block -> 混合精度 INT4/INT8 转换脚本
(compressed-tensors 多 config_groups)

流程：
  1. 反量化原始 checkpoint 中的 FP4/FP8 block 权重为 FP32
  2. 主干 routed experts 重新量化为 INT4 (pack-quantized, int32-lane)
  3. 按优先级分层把部分 FP8 线层重新量化为 INT8 (int-quantized, channelwise)
  4. 其余特殊层反量化为 BF16;engram.embed 原样透传 FP8
  5. 更新 config.json 和 index 文件

原始模型格式（见 config.json）:
  - quant_method: "fp8", weight_block_size: [32, 32], scale_fmt: "ue8m0"
  - expert_dtype: "fp4"  ->  只有 routed experts 是 FP4 E2M1FN packed
  - 其余被量化的 Linear 都是 FP8 E4M3FN blockwise 32x32, 裸 `.scale` 键名
  - compressor / indexer.wk / indexer.weights_proj / router gate / 全部 norm /
    hc_* / attn_sink / vision 在原始 checkpoint 里**没有** `.scale`, 即未量化


量化优先级表（本脚本的 CLI 开关与之一一对应）
==============================================

【优先级一 — 默认开启，无开关】
  layers.*.ffn.experts.*.w[123]        原 FP4 -> INT4  (pack-quantized, int32-lane)
  layers.*.ffn.shared_experts.w[123]   原 FP8 -> INT8  (int-quantized, channelwise)

  依据:
    - routed experts 是原 checkpoint 唯一的 4bit 层, 参数量占绝对多数,
      且 MoE 稀疏激活 (top-6/384) 对量化误差容忍度高;
    - shared experts 是标准 dense SwiGLU FFN, vLLM 侧对应普通
      `DeepseekV4MLP` (gate_up_proj + down_proj), 无专用 kernel。

【优先级二 — --int8-p2 开启，默认关闭】
  layers.*.attn.wq_a                   原 FP8 -> INT8
  layers.*.attn.wkv                    原 FP8 -> INT8
  layers.*.attn.wq_b                   原 FP8 -> INT8
  mtp.*.attn.{wq_a,wkv,wq_b}           原 FP8 -> INT8  (同档, 见下)

  依据: 原始 checkpoint 就是 8bit, 数值上有先例。
  MTP 草稿层为何必须同档:
    dspark 把草稿层注册在 `model.layers.{num_hidden_layers + i}`
    (dspark.py:116), 即 layers.40/41/42 —— 运行时**不存在** `mtp.` 开头的
    模块名。所以 config_group 的 target 会同时命中主干和草稿, 而关掉 p2 时
    兜底 ignore (`re:.*attn\\.wq_b$` 等, 无 layers. 锚点) 同样同时覆盖两者。
    一旦开启 p2, 兜底 ignore 被移除, 草稿层立刻被 int8 组接管 —— 此时若
    草稿权重仍写 BF16, 就会出现"config 说 INT8、权重是 BF16"的错配:
    `weight` 被 `copy_` 静默截断成 int8, 而 `weight_scale` 因为 checkpoint
    里没有对应键, 停留在 `torch.empty` 的未初始化内存上。草稿主干因此被
    摧毁, 但它与目标共享 embedding / lm_head, 仍能输出合法 token, 只是每个
    都错 —— 表现为 MTP 接受率精确塌到 0, 而主模型输出仍然连贯。
    因此草稿的这些投影必须与主干同档同源。
  风险:
    - wq_a + wkv 在 vLLM 中被融合为 `attn.fused_wqa_wkv`
      (vllm/models/deepseek_v41/nvidia/model.py:1000-1001), 需要 fused
      MergedColumnParallelLinear 的 channelwise scale 分片加载正确;
    - wq_b 的输出直接进入 MLA Q norm / RoPE / sparse attention,
      部分 backend 还会对 wq_b 做 layout permutation
      (vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py:239-241)。

【优先级三 — --int8-p3 开启，默认关闭】
  layers.*.attn.wo_b                   原 FP8 -> INT8
  layers.*.attn.indexer.wq_b           原 FP8 -> INT8
  layers.*.engram.wkv                  原 FP8 -> INT8

  不含 layers.*.attn.wo_a: 默认为 BF16, 需单独的 --int8-wo-a 开关 (见下)。

  依据: 原始 checkpoint 都是 8bit。
  wo_a 为何排除:
    - wo_a 带 `is_bmm=True`, vLLM 走 FP8 BMM / einsum 专用路径
      (vllm/models/deepseek_v4/nvidia/ops/o_proj.py:30-95)。该路径用
      `use_fp8 = wo_a.weight.dtype == float8_e4m3fn` 分流, 只有 fp8 / bf16
      两个分支; INT8 会掉进 bf16 分支的 `.view()` + `torch.bmm`, 启动时
      在 profile_run 首次前向抛 "view size is not compatible with ..."。
    - 这条手写路径绕过 quant_method.apply(), linear 侧的
      CutlassInt8ScaledMMLinearKernel 对它无效, 只能自研 kernel。
    - 同一缺口在 ROCm / CPU / XPU 的 o_proj 路径同样存在, 所以在量化侧
      剔除是跨平台的, 而不是 CUDA 专属的 workaround。
  剩余风险:
    - wo_b 的输入可能是 `QuantizedActivation` / GEMM-RS 融合路径;
    - indexer.wq_b 影响 DSA sparse top-k 的离散选择, 需 top-k 一致性验证;
    - engram.wkv 输出进入 Engram 专用归一化门控 + Triton kernel。

【可选 — --int8-wo-a 开启，默认关闭，独立于 p2/p3】
  layers.*.attn.wo_a                   原 FP8 -> INT8
  mtp.*.attn.wo_a                      原 FP8 -> INT8  (同档, 理由同 p2 的 MTP 说明)

  !! 当前 vLLM 无法运行该选项的产物 !!
    wo_a 带 `is_bmm=True`, o_proj 走手写路径 deep_gemm_fp8_o_proj
    (vllm/models/deepseek_v4/nvidia/ops/o_proj.py:49-86), 只有 fp8 / bf16 两个
    分支。INT8 weight 经 CutlassInt8ScaledMMLinearKernel 加载后处理会被转置成
    (K, N) (scaled_mm/cutlass.py:50-55), 随后 `.view(n_groups, o_lora_rank, -1)`
    失败; 即使不失败, bf16 分支也不会乘 INT8 weight_scale。
    本开关只负责产出 "INT8 weight + fp32 per-out-channel weight_scale" 与对应
    config, 供已自行为 wo_a 补了 INT8 分组 BMM kernel 的引擎使用
    (需: o_proj 增加 INT8 分支 + 保持 weight 为 (N, K) 或按转置布局取用 +
     激活改为 per-token INT8)。未打补丁的 vLLM 上启用会在 profile_run 首次前向崩溃。
  数值: wo_a 形状为 (n_groups*o_lora_rank, n_heads*head_dim/n_groups), 每个输出行
    只属于一个 group, per-output-channel 量化与分组 BMM 兼容。
  config: 单独成组 group_int8_wo_a, 并从 ignore 中移除 `attn.wo_a`。
    wo_a 不在 packed_modules_mapping 里, 不会触发 partial-ignore。

【MTP (DSpark draft) routed experts — --mtp-experts，默认 int4】
  mtp.*.ffn.experts.*.w[123]           原 **MXFP4** (与主干 routed experts 同格式,
                                       E2M1 + E8M0 scale, block 32; 3 层 x 128 experts
                                       x 3 矩阵 = 1152 个张量, 全部带 .scale)

  注意: 这里原精度是 4bit, 因此三个档位里只有 int4 是"同档位换算",
  int8 / bf16 都是**精度提升**, 不是"保持原精度":
    int4  6.335 GiB (0.94x)  <- 默认, 与主干一致, 复用 --group-size
    int8 12.670 GiB (1.88x)
    bf16 25.312 GiB (3.76x)

  默认取 int4 的理由: 体积中性, 且 DeepSeek 参考实现本身在 4bit 下跑 draft,
  其公布接受率即在 MXFP4 档位达成。draft 精度直接决定投机解码接受率 alpha,
  若实测 alpha 在 int4 下退化, 再升到 int8 / bf16 (代价是可逆的;
  反过来默认 int8 的 +6.3 GiB 是无条件付出)。

  实现约束: MTP experts 走与主干完全相同的路径
  (vllm/models/deepseek_v41/nvidia/dspark.py:122-124 用 DeepseekV4DecoderLayer,
   :291 复用同一 quant_config), 因此要求 **不走 MegaMoE 后端** ——
  DeepseekV4MegaMoEExperts 自己 create_weights 硬编码 uint8 + hidden_size//2
  (MXFP4 布局), 绕过 compressed-tensors。此约束对主干 experts 同样存在。

【永远保留 (BF16 / 原 FP8 / 原 FP32) — 无开关】
  attn.compressor.wkv / wgate / norm   原始未量化, vLLM 里直接
                                       `torch.mm(x, weight.T)` 绕过 Linear
                                       (vllm/models/deepseek_v41/attention.py:916-922)
  attn.indexer.wk / weights_proj / k_norm   原始未量化, vLLM 显式 quant_config=None
  attn.q_norm / kv_norm / attn_norm / ffn_norm / 各层 RMSNorm
  ffn.gate (router)                    误差会改变 expert 选择
  hc_attn_* / hc_ffn_* / attn_sink     原始 FP32, 非 2D Linear
  engram.embed                         原样透传 FP8 + 裸 .scale (硬编码格式)
  engram.q_weight / k_weight           原始未量化
  mtp 的**非 experts** 部分 (shared_experts / attn / ffn.gate / main_proj /
      *_norm / markov_head / confidence_head / hc_* / engram)
                                       原 FP8 或无 scale -> BF16
  vision.* / aligner.* / image_*       主模型 loader 直接 skip
  embed / head (lm_head) / norm


用法
====
  # 只做优先级一 (routed experts INT4 + shared experts INT8)
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --input-dir  /path/DeepSeek-V4.1-Flash \\
      --output-dir /path/DeepSeek-V4.1-Flash-INT4-INT8

  # 加优先级二 (MLA 输入投影也 INT8)
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --int8-p2 --input-dir ... --output-dir ...

  # 加优先级三 (MLA 输出投影 wo_b + indexer.wq_b + engram.wkv 也 INT8)
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --int8-p2 --int8-p3 --input-dir ... --output-dir ...

  # 另外把 wo_a 也量化为 INT8 (需引擎侧自带 wo_a INT8 BMM kernel, 见上)
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --int8-p2 --int8-p3 --int8-wo-a --input-dir ... --output-dir ...

  # 显式覆盖 engram.wkv 策略 (默认 bf16; --int8-p3 时自动变 int8)
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --int8-p3 --engram-wkv-mode int8 --input-dir ... --output-dir ...

  # MTP experts 升到 INT8 (默认 int4)
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --mtp-experts int8 --input-dir ... --output-dir ...

  # MTP experts 升到 BF16
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py \\
      --mtp-experts bf16 --input-dir ... --output-dir ...

  # 只转换前 N 个 shard 做冒烟测试
  python quantize_deepseek_v4_1_flash_mixed_int4_int8.py --limit-files 2 ...


落盘键名约定
============
  INT4 (compressed-tensors pack-quantized):
    <prefix>.weight_packed   int32  (N, K/8)
    <prefix>.weight_scale    bf16   (N, 1) 或 (N, K/group_size)
    <prefix>.weight_shape    int64  (2,)
  INT8 (compressed-tensors int-quantized, channelwise):
    <prefix>.weight          int8   (N, K)   值域 [-127, 127]
    <prefix>.weight_scale    fp32   (N, 1)   per-output-channel
  BF16 / FP8 透传:
    <prefix>.weight          bf16 | float8_e4m3fn
    <prefix>.scale           (仅 engram.embed 与 engram-wkv-mode=blockwise 保留)


前置条件
========
  - torch >= 2.1 (含 torch.float8_e4m3fn dtype)
  - safetensors, tqdm
  - 不需要 compressed_tensors 包 (pack_to_int32 自带等价实现 + 位级自检)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

# FP4 E2M1FN 查找表
FP4_TABLE = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=torch.float32,
)

# DeepSeek-V4.1-Flash 所有量化层 block_size 都是 32x32
FP4_BLOCK_SIZE = 32
FP8_BLOCK_SIZE = 32

INT4_QMAX = 7
INT8_QMAX = 127.0

# engram.wkv 的三种处理方式
ENGRAM_WKV_MODES = ("bf16", "blockwise", "int8")

# MTP (DSpark draft) routed experts 的处理方式。
#
# 原始 checkpoint 中 mtp.*.ffn.experts.*.w[123] 与主干 routed experts 一样是
# MXFP4 (E2M1 值 + E8M0 scale, block 32), 共 3 层 x 128 experts x 3 矩阵 = 1152 个张量,
# 全部带 .scale。同一份 expert_dtype 由 vLLM 的 DSpark draft 复用
# (vllm/models/deepseek_v41/nvidia/dspark.py:291 用同一 quant_config,
#  :391-395 按 expert_dtype 决定 .weight_scale / .weight_scale_inv)。
#
# 三种落盘方式对应的体积 (13.590 B 参数):
#   int4 : 6.335 GiB  —— 与原始 MXFP4 的 6.724 GiB 基本持平 (同一 4bit 档位)  <- 默认
#   int8 : 12.670 GiB —— 精度升级 (4bit -> 8bit), 体积 1.88x
#   bf16 : 25.312 GiB —— 显著升级 (4bit -> 16bit), 体积 3.76x
#
# 默认取 int4: 体积中性, 且 DeepSeek 参考实现本身就在 4bit 下跑 draft,
# 改 bf16/int8 属于无条件付出体积换 margin, 需要实测接受率后才值得开。
MTP_EXPERTS_MODES = ("bf16", "int4", "int8")


# --------------------------------------------------------------------------- #
# 键名分类正则 (DeepSeek-V4.1-Flash 无 model. 前缀)
#
# 注意: 这些正则匹配的是 **checkpoint 键名**; vLLM/sglang 的 config targets
# 与 ignore 匹配的是 **运行时 module 名**, 两者命名不同, 见下方
# build_compression_config / build_ignore_list 的注释。
# --------------------------------------------------------------------------- #

# 主干 routed experts
ROUTED_EXPERT_RE = re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
# 主干 shared experts
# MTP 的 shared experts 一并处理: config 的 group_int8_shared_experts target
# (`re:.*shared_experts\.(?:gate_up_proj|down_proj|w[123])$`) 不限定 layers.*,
# 运行时会同时命中草稿层 model.layers.{40,41,42}.ffn.shared_experts。
# 草稿与主干必须同档同源, 否则草稿侧 int8 参数会拿到 BF16 权重。
SHARED_EXPERT_RE = re.compile(
    r"^(?:layers|mtp)\.\d+\.ffn\.shared_experts\.w[123]\.weight$"
)
# MTP experts (dspark)
MTP_EXPERT_RE = re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.w[123]\.weight$")
# engram
ENGRAM_EMBED_RE = re.compile(r"^(layers|mtp)\.\d+\.engram\.embed\.")
# engram.wkv 只处理主干。config 侧 group_int8_engram_wkv 的 target 是
# `re:.*layers\.\d+\.engram\.wkv$`, MTP 的 engram 子树整体在 ignore 里;
# 若这里放行 mtp.*, 就会出现"写成 INT8 但引擎按 BF16 加载"的不一致。
ENGRAM_WKV_RE = re.compile(r"^layers\.\d+\.engram\.wkv\.")

# 优先级二: MLA 输入侧三个投影
# MTP 一并处理, 理由同 SHARED_EXPERT_RE: config 的 p2 target 用 `layers.\d+`,
# 而 `\d+` 同样匹配草稿的层号 40/41/42, 运行时草稿层会按 INT8 建参数。
# 运行时真实层号见 build_ignore_list 的说明。
P2_ATTN_RE = re.compile(r"^(?:layers|mtp)\.\d+\.attn\.(?:wq_a|wkv|wq_b)\.weight$")
# 优先级三: MLA 输出侧的 down 投影 (wo_b)
# wo_a 不在此列: 它带 is_bmm=True, vLLM 的 deep_gemm_fp8_o_proj 直接读
# wo_a.weight 做 view/bmm, 只有 fp8 / bf16 分支, 没有 INT8。详见 build_ignore_list。
P3_ATTN_RE = re.compile(r"^(?:layers|mtp)\.\d+\.attn\.wo_b\.weight$")
# 可选 (--int8-wo-a): MLA 输出侧的分组低秩 down 投影 wo_a。
# MTP 一并处理, 理由同 P2_ATTN_RE: 运行时 target `layers.\d+` 同样命中草稿层 40/41/42。
WO_A_RE = re.compile(r"^(?:layers|mtp)\.\d+\.attn\.wo_a\.weight$")
# 优先级三: DSA indexer 的 query 投影
P3_INDEXER_RE = re.compile(r"^(?:layers|mtp)\.\d+\.attn\.indexer\.wq_b\.weight$")

# --------------------------------------------------------------------------- #
# MTP/DSpark 子树的运行时前缀 (用于 config target)
#
# vLLM  : model.mtp.{N}.ffn.experts.{E}.{gate,up,down}_proj
# sglang: stages.{N}.mlp.experts.{E}.*   (DSpark draft)
#
# 主干 routed experts 的 target 用负向先行断言排除该前缀, 这样:
#   - 主干与 MTP 即使量化档位相同也分属不同 group, 语义清晰;
#   - MTP 档位 (--mtp-experts) 改变时不需要重排 group 顺序,
#     因为 vLLM 的 find_matched_target 只取"第一个命中的 target",
#     若两个 target 都能命中就会依赖 dict 插入顺序。
# --------------------------------------------------------------------------- #
_MTP_RUNTIME_PREFIX = r"(?:mtp|stages)\.\d+\."

_BACKBONE_EXPERTS_TARGET = (
    r"re:(?!.*" + _MTP_RUNTIME_PREFIX + r")"
    r".*(?:mlp|ffn)\.experts\.\d+\.(?:gate_proj|up_proj|down_proj|w[123])$"
)
_MTP_EXPERTS_TARGET = (
    r"re:.*" + _MTP_RUNTIME_PREFIX
    + r"(?:mlp|ffn)\.experts\.\d+\.(?:gate_proj|up_proj|down_proj|w[123])$"
)


# --------------------------------------------------------------------------- #
# 转换选项 (可 pickle, 用于多进程 worker)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ConvOptions:
    """一次转换的全部开关。engram_wkv_mode 传入前必须已解析 (见 resolve_engram_wkv_mode)。"""

    group_size: int = -1          # INT4 沿输入维 K 的分组; -1 = per-channel
    mode: str = "w4a8"            # INT4 组的激活声明; w4a8(默认) 或 w4a16
    int8_p2: bool = False         # 优先级二: attn.wq_a / wkv / wq_b -> INT8
    int8_p3: bool = False         # 优先级三: attn.wo_b / indexer.wq_b / engram.wkv -> INT8
    int8_wo_a: bool = False       # 可选: attn.wo_a -> INT8 (vLLM 现有 o_proj 路径不支持)
    engram_wkv_mode: str = "bf16"  # bf16 | blockwise | int8
    mtp_experts: str = "int4"     # MTP routed experts: bf16 | int4 | int8


def resolve_engram_wkv_mode(int8_p3: bool, engram_wkv_mode: str | None) -> str:
    """--engram-wkv-mode 未显式给出时, 跟随 --int8-p3 (int8) 否则 bf16。

    engram.wkv 在原始 checkpoint 里是 FP8 (有 .scale), 因此 INT8 是"位宽守恒",
    而 bf16 反而是精度提升。把它挂在优先级三下, 与其它 8bit->8bit 的项同组。
    """
    if engram_wkv_mode is not None:
        return engram_wkv_mode
    return "int8" if int8_p3 else "bf16"


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def scale_name_for(weight_name: str) -> str:
    """源 checkpoint 中 scale 的名字 (.weight -> .scale)"""
    return ".".join(weight_name.split(".")[:-1] + ["scale"])


def has_scale(name: str, state_dict: dict[str, torch.Tensor]) -> bool:
    return scale_name_for(name) in state_dict


def weight_prefix(name: str) -> str:
    """去掉尾部 .weight 后的前缀; 非 .weight 键原样返回。"""
    return name[: -len(".weight")] if name.endswith(".weight") else name


# --------------------------------------------------------------------------- #
# 反量化: FP4 / FP8 blockwise -> FP32
# --------------------------------------------------------------------------- #

# FP4_TABLE 是查表反量化用的常量,GPU 场景下按 device 缓存一份, 避免每次搬运
_FP4_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}


def _fp4_table_on(device: torch.device) -> torch.Tensor:
    if device not in _FP4_TABLE_CACHE:
        _FP4_TABLE_CACHE[device] = FP4_TABLE.to(device)
    return _FP4_TABLE_CACHE[device]


def unpack_e2m1fn_to_float(x: torch.Tensor) -> torch.Tensor:
    """Unpack FP4 E2M1FN packed tensor (int8) to float32. 每字节 2 个 FP4 (低4/高4)."""
    assert x.dtype == torch.int8
    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    table = _fp4_table_on(x.device)
    return torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(1)


def dequant_fp4_to_float(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP4 blockwise (x: (N, K/2), scale: (N, K/32)) -> FP32 (N, K)."""
    values = unpack_e2m1fn_to_float(x).float()
    expanded_scale = scale.float().repeat_interleave(FP4_BLOCK_SIZE, dim=1)
    return values * expanded_scale


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    FP8 e4m3fn blockwise (weight (N,K), scale (N/b, K/b)) -> FP32 (N, K).
    根据 scale shape 推断 block size (V4.1-Flash 是 32x32),不整除走 fallback.
    """
    scale = scale.float()

    if weight.ndim != 2 or scale.ndim != 2:
        if weight.numel() % scale.numel() == 0:
            ratio = weight.numel() // scale.numel()
            flat_w = weight.float().view(-1, ratio) if ratio > 1 else weight.float().view(-1, 1)
            flat_s = scale.float().view(-1, 1)
            return (flat_w * flat_s).view_as(weight).float()
        return weight.float() * scale.float().mean()

    out_dim, in_dim = weight.shape
    s_rows, s_cols = scale.shape

    if s_rows == 0 or s_cols == 0 or out_dim % s_rows != 0 or in_dim % s_cols != 0:
        return weight.float() * scale.float().mean()

    row_block = out_dim // s_rows
    col_block = in_dim // s_cols

    weight_blocks = weight.unflatten(0, (s_rows, row_block)).unflatten(2, (s_cols, col_block))
    scale_expanded = scale[:, None, :, None]
    dequantized = weight_blocks.float() * scale_expanded
    return dequantized.flatten(0, 1).flatten(1, 2)


# --------------------------------------------------------------------------- #
# INT4 量化 + int32-lane packing (compressed-tensors pack-quantized 规范)
# --------------------------------------------------------------------------- #

def quantize_int4(
    tensor: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT4 量化, 值域 [-8, 7], scale = abs_max / 7.
      group_size = -1  -> per-channel (scale shape (N, 1))
      group_size >  0  -> per-group  (scale shape (N, K/g)),要求 K % g == 0
    返回 (q_int8 保存 [-8,7] 有符号值, scale float32).
    """
    assert tensor.ndim == 2, f"expect 2D tensor, got {tensor.shape}"
    N, K = tensor.shape
    w = tensor.float()

    if group_size == -1 or group_size is None:
        abs_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # (N, 1)
        scale = abs_max / INT4_QMAX
        q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
    else:
        assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
        wg = w.view(N, K // group_size, group_size)
        abs_max = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # (N, K/g, 1)
        scale_g = abs_max / INT4_QMAX
        q = torch.round(wg / scale_g).clamp(-8, 7).to(torch.int8)
        q = q.view(N, K)
        scale = scale_g.squeeze(-1)  # (N, K/g)

    return q, scale.to(torch.float32)


def pack_int4_to_int32(q: torch.Tensor) -> torch.Tensor:
    """
    Signed int4 (int8 存储, 值 [-8, 7]) -> compressed-tensors pack-quantized int32.
    沿最后一维 (input dim K) 每 8 个 int4 -> 1 个 int32. K 必须能被 8 整除.

    Bit 布局 (升序, 4-bit slot 连续, 与 compressed_tensors.pack_to_int32 位级等价):
      word = u0 | (u1<<4) | (u2<<8) | (u3<<12) | (u4<<16) | (u5<<20) | (u6<<24) | (u7<<28)
    其中 u_i = q_i + 8 in [0, 15] (signed -> unsigned).

    参考:
      compressed_tensors/compressors/pack_quantized/helpers.py:pack_to_int32
      utils/MoE-Quant/pack_quantized_model.py:pack_to_int32
    """
    assert q.dtype == torch.int8 and q.ndim == 2, f"expect 2D int8, got {q.shape}/{q.dtype}"
    N, K = q.shape
    assert K % 8 == 0, f"K={K} must be divisible by 8"

    # signed [-8, 7] -> unsigned [0, 15]; mask 高位, cast 到 int32
    u = ((q.to(torch.int32) + 8) & 0xF)  # (N, K)
    u = u.view(N, K // 8, 8)  # 每 8 个 nibble 装 1 个 int32
    shifts = torch.arange(8, dtype=torch.int32, device=q.device) * 4  # [0,4,8,...,28]
    packed = (u << shifts).sum(dim=-1).to(torch.int32)  # OR 等价于 SUM (bit slot 不相交)
    return packed.contiguous()


# --------------------------------------------------------------------------- #
# INT8 量化 (compressed-tensors int-quantized, channelwise symmetric)
# --------------------------------------------------------------------------- #

def quantize_int8_channelwise(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT8 channelwise (per-output-channel) 量化.
      tensor: float, shape (out_dim, in_dim)
    返回:
      quantized: int8,    shape (out_dim, in_dim), 值域 [-127, 127]
      scale:     float32, shape (out_dim, 1)

    对应 compressed-tensors 的 weights 声明:
      strategy="channel", symmetric=True, dynamic=False, num_bits=8, type="int"
    落盘后由 vLLM `CompressedTensorsW8A8Int8` 消费:
      vllm/model_executor/layers/quantization/compressed_tensors/schemes/
        compressed_tensors_w8a8_int8.py:58-82
      (weight 是 int8 ModelWeightParameter, weight_scale 是 fp32
       ChannelQuantScaleParameter, shape (sum(output_partition_sizes), 1))

    本函数与 quantize_glm5_3_flash_int4_channel.py:267-280 的公式一致:
      abs_max = max(|w|) along in_dim
      scale   = abs_max / 127
      q       = round(w / scale).clamp(-127, 127)
    """
    assert tensor.ndim == 2, f"expect 2D tensor, got {tensor.shape}"
    w = tensor.float()
    abs_max = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # (N, 1)
    scale = abs_max / INT8_QMAX
    q = torch.round(w / scale).clamp(-INT8_QMAX, INT8_QMAX).to(torch.int8)
    return q, scale.to(torch.float32)


# --------------------------------------------------------------------------- #
# 键名分类 (checkpoint 键名)
# --------------------------------------------------------------------------- #

def is_routed_expert_weight(name: str) -> bool:
    """主干 routed experts: layers.{L}.ffn.experts.{E}.w[123].weight"""
    return ROUTED_EXPERT_RE.match(name) is not None


def is_shared_expert_weight(name: str) -> bool:
    """主干 shared experts: layers.{L}.ffn.shared_experts.w[123].weight"""
    return SHARED_EXPERT_RE.match(name) is not None


def is_mtp_expert_weight(name: str) -> bool:
    """MTP experts: mtp.{M}.ffn.experts.{E}.w[123].weight (dspark)"""
    return MTP_EXPERT_RE.match(name) is not None


def is_engram_embed(name: str) -> bool:
    """engram.embed 是 sglang 硬编码 fp8+e8m0fnu blockwise hash table,原样透传"""
    return ENGRAM_EMBED_RE.match(name) is not None


def is_engram_wkv(name: str) -> bool:
    return ENGRAM_WKV_RE.match(name) is not None


def is_p2_attn_weight(name: str) -> bool:
    """优先级二: layers.{L}.attn.{wq_a,wkv,wq_b}.weight (不含 mtp.*)"""
    return P2_ATTN_RE.match(name) is not None


def is_p3_attn_weight(name: str) -> bool:
    """优先级三: layers.{L}.attn.wo_b.weight (不含 mtp.*; wo_a 见 P3_ATTN_RE 注释)"""
    return P3_ATTN_RE.match(name) is not None


def is_wo_a_weight(name: str) -> bool:
    """可选 (--int8-wo-a): {layers,mtp}.{L}.attn.wo_a.weight"""
    return WO_A_RE.match(name) is not None


def is_p3_indexer_weight(name: str) -> bool:
    """优先级三: layers.{L}.attn.indexer.wq_b.weight (不含 mtp.*)"""
    return P3_INDEXER_RE.match(name) is not None


def is_int8_target(name: str, opts: ConvOptions) -> bool:
    """该 checkpoint 键是否属于某个已启用的 INT8 目标。"""
    if is_shared_expert_weight(name):
        return True  # 优先级一, 始终启用
    if opts.int8_p2 and is_p2_attn_weight(name):
        return True
    if opts.int8_p3 and is_p3_attn_weight(name):
        return True
    if opts.int8_p3 and is_p3_indexer_weight(name):
        return True
    if opts.int8_wo_a and is_wo_a_weight(name):
        return True
    return False


def is_fp8_blockwise_to_bf16(
    name: str, tensor: torch.Tensor, state_dict: dict[str, torch.Tensor]
) -> bool:
    """
    需要 FP8->BF16 反量化的层: 所有未被 INT8 优先级命中的 FP8 blockwise 权重
    (attention MLA / compressor / indexer.wk / weights_proj / MTP attn / main_proj
     / engram.wkv(bfloat16 模式))。
    判据: FP8 e4m3fn dtype + .weight 结尾 + 有 .scale 兄弟,且不是 engram.embed.
    """
    if is_engram_embed(name):
        return False
    if tensor.dtype != torch.float8_e4m3fn:
        return False
    if not name.endswith(".weight"):
        return False
    return has_scale(name, state_dict)


# --------------------------------------------------------------------------- #
# 主转换流程
# --------------------------------------------------------------------------- #

def convert_one_file(
    input_path: str,
    output_path: str,
    opts: ConvOptions,
    stats: dict[str, int],
    device: torch.device | str = "cpu",
) -> None:
    """转换单个 safetensors 分片 (混合精度 INT4/INT8/BF16/FP8).

    分支顺序 (先窄后宽, 命中即出):
      0.  .scale                       -> 丢弃 (由各分支重写)
      1.  engram.embed                 -> 原样透传 FP8 + 保留裸 .scale
      2.  engram.wkv                   -> int8 / bf16 / blockwise(透传)
      3.  routed experts (主干)         -> INT4 pack-quantized
      4.  MTP experts                  -> BF16
      5.  shared experts (主干)         -> INT8            【优先级一, 始终】
      6.  attn.wq_a / wkv / wq_b       -> INT8 或落回 BF16 【优先级二, 可选】
      7.  attn.wo_b                    -> INT8 或落回 BF16 【优先级三, 可选】
          attn.wo_a                    -> INT8 或落回 BF16 【--int8-wo-a, 可选】
      8.  attn.indexer.wq_b            -> INT8 或落回 BF16 【优先级三, 可选】
      9.  其它 FP8 blockwise           -> BF16
      10. 其余 (BF16/FP32/非 2D)        -> 原样透传

    device: 计算所在 device (cpu 或 cuda:N). CPU tensor 从 safetensors 读出后临时搬到
    device 上做 dequant/quantize/pack, 保存前搬回 CPU (save_file 要求 CPU tensor).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device

    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t if t.device == device else t.to(device, non_blocking=False)

    def _to_cpu(t: torch.Tensor) -> torch.Tensor:
        return t if t.device.type == "cpu" else t.cpu()

    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(input_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            state_dict[name] = f.get_tensor(name)

    new_state_dict: dict[str, torch.Tensor] = {}

    for name, tensor in state_dict.items():
        # scale 键由 weight 分支处理; 未被处理的 scale 由后续分支判定是否保留
        if name.endswith(".scale"):
            continue

        scale_name = scale_name_for(name)
        prefix = weight_prefix(name)
        is_weight = name.endswith(".weight")

        # 1. engram.embed: 原样透传 FP8 + 保留裸 .scale (sglang 硬编码)
        if is_engram_embed(name):
            new_state_dict[name] = tensor
            if scale_name in state_dict:
                new_state_dict[scale_name] = state_dict[scale_name]
            stats["engram_embed_passthrough"] += 1
            continue

        # 2. engram.wkv: int8 / bf16 / blockwise(透传)
        if is_engram_wkv(name) and is_weight:
            if opts.engram_wkv_mode == "blockwise":
                new_state_dict[name] = tensor
                if scale_name in state_dict:
                    new_state_dict[scale_name] = state_dict[scale_name]
                stats["engram_wkv_passthrough"] += 1
                continue
            if opts.engram_wkv_mode == "int8" and scale_name in state_dict:
                weight_fp32 = dequant_fp8_blockwise(
                    _to_dev(tensor), _to_dev(state_dict[scale_name])
                )
                q_int8, new_scale = quantize_int8_channelwise(weight_fp32)
                new_state_dict[name] = _to_cpu(q_int8)
                new_state_dict[f"{prefix}.weight_scale"] = _to_cpu(new_scale)
                stats["engram_wkv_int8"] += 1
                continue
            # bf16 (默认), 或 int8 但缺 scale 时退化为 bf16
            if scale_name in state_dict:
                new_state_dict[name] = _to_cpu(
                    dequant_fp8_blockwise(
                        _to_dev(tensor), _to_dev(state_dict[scale_name])
                    ).bfloat16()
                )
            else:
                new_state_dict[name] = (
                    tensor.bfloat16() if tensor.dtype == torch.float8_e4m3fn else tensor
                )
            stats["engram_wkv_bf16"] += 1
            continue

        # 3. 主干 routed experts: FP4 -> INT4 pack-quantized
        if is_routed_expert_weight(name):
            if scale_name not in state_dict:
                # 理论上不该发生; 退化为透传避免整份任务崩掉
                new_state_dict[name] = tensor
                stats["routed_expert_no_scale"] += 1
                continue
            scale = _to_dev(state_dict[scale_name])
            weight_fp32 = dequant_fp4_to_float(_to_dev(tensor), scale)
            q_int8, q_scale = quantize_int4(weight_fp32, opts.group_size)
            packed = pack_int4_to_int32(q_int8)
            new_state_dict[f"{prefix}.weight_packed"] = _to_cpu(packed)
            new_state_dict[f"{prefix}.weight_scale"] = _to_cpu(q_scale.to(torch.bfloat16))
            new_state_dict[f"{prefix}.weight_shape"] = torch.tensor(
                list(weight_fp32.shape), dtype=torch.int64
            )
            stats["routed_expert_int4"] += 1
            continue

        # 4. MTP (DSpark draft) experts: 原始 MXFP4 -> INT4 / INT8 / BF16
        if is_mtp_expert_weight(name):
            if scale_name not in state_dict:
                # 理论上不该发生; 退化为透传避免整份任务崩掉
                new_state_dict[name] = tensor
                stats["mtp_expert_no_scale"] += 1
                continue

            weight_fp32 = dequant_fp4_to_float(
                _to_dev(tensor), _to_dev(state_dict[scale_name])
            )

            if opts.mtp_experts == "int4":
                # 与主干 routed experts 同格式, group_size 复用 --group-size
                q_int8, q_scale = quantize_int4(weight_fp32, opts.group_size)
                new_state_dict[f"{prefix}.weight_packed"] = _to_cpu(
                    pack_int4_to_int32(q_int8)
                )
                new_state_dict[f"{prefix}.weight_scale"] = _to_cpu(
                    q_scale.to(torch.bfloat16)
                )
                new_state_dict[f"{prefix}.weight_shape"] = torch.tensor(
                    list(weight_fp32.shape), dtype=torch.int64
                )
                stats["mtp_expert_int4"] += 1
            elif opts.mtp_experts == "int8":
                q_int8, q_scale = quantize_int8_channelwise(weight_fp32)
                new_state_dict[name] = _to_cpu(q_int8)
                new_state_dict[f"{prefix}.weight_scale"] = _to_cpu(q_scale)
                stats["mtp_expert_int8"] += 1
            else:  # bf16
                new_state_dict[name] = _to_cpu(weight_fp32.bfloat16())
                stats["mtp_expert_bf16"] += 1
            continue

        # 5-8. INT8 目标 (优先级一 / 二 / 三)
        if is_int8_target(name, opts) and is_fp8_blockwise_to_bf16(name, tensor, state_dict):
            weight_fp32 = dequant_fp8_blockwise(
                _to_dev(tensor), _to_dev(state_dict[scale_name])
            )
            q_int8, new_scale = quantize_int8_channelwise(weight_fp32)
            # compressed-tensors int-quantized: weight(int8) + weight_scale(fp32, (N,1))
            new_state_dict[name] = _to_cpu(q_int8)
            new_state_dict[f"{prefix}.weight_scale"] = _to_cpu(new_scale)
            if is_shared_expert_weight(name):
                stats["shared_expert_int8"] += 1
            elif is_p2_attn_weight(name):
                stats["attn_int8_p2"] += 1
            elif is_p3_attn_weight(name):
                stats["attn_int8_p3"] += 1
            elif is_p3_indexer_weight(name):
                stats["indexer_wqb_int8"] += 1
            elif is_wo_a_weight(name):
                stats["attn_wo_a_int8"] += 1
            continue

        # 9. 其它 FP8 blockwise (MLA 未启用部分 / compressor / indexer.wk /
        #    weights_proj / MTP attn / main_proj) -> BF16
        if is_fp8_blockwise_to_bf16(name, tensor, state_dict):
            scale = _to_dev(state_dict[scale_name])
            new_state_dict[name] = _to_cpu(
                dequant_fp8_blockwise(_to_dev(tensor), scale).bfloat16()
            )
            stats["fp8_to_bf16"] += 1
            continue

        # 10. 其余 (BF16 权重, gate, norm, embed, head, vision, attn_sink,
        #     hc_*, engram.q_weight/k_weight, ...) 透传
        new_state_dict[name] = tensor
        stats["kept"] += 1

    save_file(new_state_dict, output_path)


# --------------------------------------------------------------------------- #
# 设备解析 + 多 GPU worker
# --------------------------------------------------------------------------- #

_STATS_KEYS = (
    "routed_expert_int4",
    "routed_expert_no_scale",
    "shared_expert_int8",
    "attn_int8_p2",
    "attn_int8_p3",
    "indexer_wqb_int8",
    "attn_wo_a_int8",
    "engram_wkv_int8",
    "mtp_expert_int4",
    "mtp_expert_int8",
    "mtp_expert_bf16",
    "mtp_expert_no_scale",
    "fp8_to_bf16",
    "engram_embed_passthrough",
    "engram_wkv_bf16",
    "engram_wkv_passthrough",
    "kept",
)


def _empty_stats() -> dict[str, int]:
    return {k: 0 for k in _STATS_KEYS}


def _resolve_devices(
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_files: int,
) -> tuple[list[torch.device], int]:
    """根据 --device / --gpu-ids / --workers 参数解析实际 device 列表与并行度.

    Returns:
        (devices, effective_workers): devices 长度 >=1; effective_workers 是最终并行进程数
        (1 表示主进程顺序执行, >1 表示多进程池).
    """
    cuda_available = torch.cuda.is_available()

    if device_kind == "cpu":
        return [torch.device("cpu")], 1
    if device_kind == "cuda" and not cuda_available:
        raise RuntimeError(
            "--device cuda specified but torch.cuda.is_available() == False"
        )
    if device_kind == "auto" and not cuda_available:
        return [torch.device("cpu")], 1

    # GPU 分支
    if gpu_ids:
        try:
            ids = [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]
        except ValueError as e:
            raise RuntimeError(f"invalid --gpu-ids {gpu_ids!r}: {e}") from e
    else:
        ids = list(range(torch.cuda.device_count()))
    if not ids:
        raise RuntimeError("no CUDA device selected (device_count == 0)")

    devices = [torch.device(f"cuda:{i}") for i in ids]
    if workers is None:
        effective = min(len(devices), max(1, num_files))
    else:
        effective = min(max(1, workers), max(1, num_files))
    return devices, effective


def _worker_convert_shard(
    args: tuple[str, str, ConvOptions, int, int],
) -> dict[str, int]:
    """spawn worker 入口: 绑定一张 GPU, 处理一个 shard, 返回本 shard 的 stats.
    必须是模块顶层函数以支持 multiprocessing.spawn 的 pickle.
    """
    (input_path, output_path, opts, gpu_id, num_threads) = args
    torch.set_num_threads(max(1, num_threads))
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    stats = _empty_stats()
    convert_one_file(input_path, output_path, opts, stats, device)
    return stats


def convert_model(
    input_dir: str,
    output_dir: str,
    opts: ConvOptions,
    limit_files: int | None,
    device_kind: str,
    gpu_ids: str | None,
    workers: int | None,
    num_threads: int,
) -> tuple[dict[str, int], list[torch.device], int]:
    """遍历所有 shard, 单卡/CPU 顺序执行或多卡并行执行. 返回 (stats, devices, workers)."""
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(input_dir, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]

    devices, effective_workers = _resolve_devices(
        device_kind, gpu_ids, workers, len(files)
    )
    stats = _empty_stats()

    # 单进程路径: CPU 或单 GPU
    if effective_workers <= 1 or len(devices) == 1:
        device = devices[0]
        desc = f"Converting on {device}"
        for path in tqdm(files, desc=desc):
            fname = os.path.basename(path)
            convert_one_file(
                path,
                os.path.join(output_dir, fname),
                opts,
                stats,
                device,
            )
        return stats, devices, effective_workers

    # 多 GPU 并行: 每个 worker 绑定一张 GPU, 通过 mp.Pool + spawn 上下文调度 shard
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")

    tasks: list[tuple[str, str, ConvOptions, int, int]] = []
    for i, path in enumerate(files):
        gpu_id = devices[i % len(devices)].index
        fname = os.path.basename(path)
        tasks.append(
            (
                path,
                os.path.join(output_dir, fname),
                opts,
                int(gpu_id),
                num_threads,
            )
        )

    desc = f"Converting on {len(devices)} GPU(s) ({effective_workers} workers)"
    with ctx.Pool(processes=effective_workers) as pool:
        for per_stats in tqdm(
            pool.imap_unordered(_worker_convert_shard, tasks),
            total=len(tasks),
            desc=desc,
        ):
            for k, v in per_stats.items():
                stats[k] = stats.get(k, 0) + v

    return stats, devices, effective_workers


# --------------------------------------------------------------------------- #
# compressed-tensors config 生成
#
# 要点: targets / ignore 匹配的是 **推理引擎实例化时的 module 名**, 不是
# checkpoint 键名。两者在 DeepSeek-V4.1 上差异很大:
#   checkpoint                      vLLM runtime
#   layers.N.attn.wq_a        ->    model.layers.N.attn.fused_wqa_wkv   (融合)
#   layers.N.attn.wkv         ->    model.layers.N.attn.fused_wqa_wkv   (融合)
#   layers.N.attn.wq_b        ->    model.layers.N.attn.wq_b
#   layers.N.attn.wo_a/wo_b   ->    model.layers.N.attn.wo_a / wo_b
#   layers.N.attn.compressor.wkv/wgate
#                             ->    model.layers.N.attn.compressor.fused_wkv_wgate
#   layers.N.ffn.shared_experts.w1/w3
#                             ->    model.layers.N.ffn.shared_experts.gate_up_proj
#   layers.N.ffn.shared_experts.w2
#                             ->    model.layers.N.ffn.shared_experts.down_proj
#
# 融合关系来自 DeepseekV41LLMForCausalLM.packed_modules_mapping
#   (vllm/models/deepseek_v41/nvidia/model.py:1315-1319):
#     {"gate_up_proj": ["w1","w3"],
#      "fused_wqa_wkv": ["wq_a","wkv"],
#      "fused_wkv_wgate": ["wkv","wgate"]}
#
# 关键约束 (vllm/model_executor/layers/quantization/compressed_tensors/utils.py:53-70):
#   should_ignore_layer 对一个融合层做 packed_modules_mapping 展开后,
#   如果**只有部分 shard 命中 ignore**, 会直接 raise ValueError。
#   因此 wq_a 与 wkv 必须同进同出 (同属 --int8-p2),
#   compressor.wkv 与 compressor.wgate 必须同时留在 ignore 里。
# --------------------------------------------------------------------------- #

def _int4_weights_arg(group_size: int) -> dict:
    if group_size == -1 or group_size is None:
        strategy = "channel"
        gs = -1
    else:
        strategy = "group"
        gs = int(group_size)
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": gs,
        "num_bits": 4,
        "observer": "minmax",
        "observer_kwargs": {},
        "strategy": strategy,
        "symmetric": True,
        "type": "int",
    }


def _int8_weights_arg() -> dict:
    """INT8 channelwise 对称权重 (与 vLLM CompressedTensorsW8A8Int8 对应)。"""
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": False,
        "group_size": -1,
        "num_bits": 8,
        "observer": "minmax",
        "observer_kwargs": {},
        "strategy": "channel",
        "symmetric": True,
        "type": "int",
    }


def _int8_input_activations_arg() -> dict:
    """INT8 动态 per-token 对称激活。

    注意 1: compressed-tensors 的 format 为 int-quantized 时,
    `_quantization_scheme_map_from_config` 要求 input_activations 存在
    (compressed_tensors.py:346-369), 否则会断言权重类型必须是 FLOAT。

    注意 2: dynamic=True 时 compressed-tensors 会强制 observer=None
    (quant_args.py: "No observer is used for dynamic quant."), 这里直接写
    规范形式, 避免每次加载刷警告。权重侧仍保留 "minmax"。
    """
    return {
        "actorder": None,
        "block_structure": None,
        "dynamic": True,
        "group_size": None,
        "num_bits": 8,
        "observer": None,
        "observer_kwargs": {},
        "strategy": "token",
        "symmetric": True,
        "type": "int",
    }


def _input_activations_arg(mode: str) -> dict | None:
    """INT4 组的激活声明 (只影响 config, 权重字节不变)。"""
    if mode == "w4a16":
        return None
    if mode == "w4a8":
        return _int8_input_activations_arg()
    raise ValueError(f"unknown mode: {mode}")


def build_ignore_list(opts: ConvOptions) -> list[str]:
    """ignore 列表 — 覆盖所有 BF16/FP32/FP8 保留张量, 避免 compressed-tensors 误配。

    组织方式:
      (a) 永远 ignore 的类别 (MTP / compressor / indexer 非目标 / norm / router /
          vision / engram 非目标 / 顶层);
      (b) 由开关决定的类别 (未启用的优先级二/三, 以及 engram.wkv)。

    采用 `re:` 通配前缀同时覆盖 vLLM 的 `attn.*` 与 sglang 的 `self_attn.*` 命名。
    """
    ignore = [
        # ---- MoE router gate (不是 experts 内部的 gate_proj) ----
        r"re:.*\.ffn\.gate$",
        r"re:.*\.ffn\.gate\..*",
        r"re:.*\.mlp\.gate$",
        r"re:.*\.mlp\.gate\..*",

        # ---- compressor: 原始 checkpoint 无 .scale, vLLM 里 direct torch.mm ----
        # 必须成对出现, 否则 fused_wkv_wgate 的部分 ignore 会 raise
        r"re:.*attn\.compressor\.wkv$",
        r"re:.*attn\.compressor\.wgate$",
        r"re:.*attn\.compressor\.fused_wkv_wgate$",
        r"re:.*attn\.compressor\.norm$",

        # ---- DSA indexer 的非 INT8 部分 (原始 checkpoint 未量化) ----
        r"re:.*attn\.indexer\.wk$",
        r"re:.*attn\.indexer\.weights_proj$",
        r"re:.*attn\.indexer\.wk_weights_proj$",
        r"re:.*attn\.indexer\.k_norm$",
        r"re:.*attn\.indexer\.k_norm\..*$",

        # ---- 归一化参数 (非 2D Linear, 防御性列出) ----
        r"re:.*attn\.q_norm$",
        r"re:.*attn\.kv_norm$",
        r"re:.*attn_norm$",
        r"re:.*ffn_norm$",

        # ---- engram 的非目标部分 ----
        r"re:.*engram\.embed$",
        r"re:.*engram\.embed_tokens$",
        r"re:.*engram\.q_weight$",
        r"re:.*engram\.k_weight$",

        # ---- MTP 特殊模块 ----
        r"re:.*main_proj.*",
        r"re:.*markov_head.*",
        r"re:.*confidence_head.*",

        # ---- 视觉 / aligner / 顶层 ----
        r"re:.*vision\..*",
        r"re:.*aligner\..*",
        "lm_head",
        "head",
        "embed",
        "norm",
    ]

    # ---- (b) MTP / DSpark 子树 ----
    # sglang DSpark draft 把 MTP 挂在 `stages.{N}.mlp.*` 下, vLLM 用 `model.mtp.{N}.*`;
    # 两种前缀都列。注意 sglang 侧 MLP 叫 `mlp`, vLLM 侧叫 `ffn`。
    #
    # !! 这些 `mtp.` / `stages.` 前缀的 ignore 只对"本脚本写什么 dtype"生效,
    # !! 对 vLLM 的运行时方案解析是**无效**的: dspark 把草稿层注册在
    # !! `model.layers.{num_hidden_layers + i}` (dspark.py:116), 即 layers.40/41/42,
    # !! 运行时不存在 `mtp.` 开头的模块名。真正决定运行时方案的是 config_group
    # !! 的 target, 而它们用 `\d+`, 会同时命中草稿层。
    # !! 因此下面这些规则必须与"草稿层实际写出的 dtype"保持一致 ——
    # !! 放行的分支 (shared_experts / p2 / p3 投影) 在写侧也必须放行,
    # !! 否则就会出现"config 说 INT8、权重却是 BF16"的错配,
    # !! 草稿主干被污染后接受率会塌到 0 (主模型输出仍连贯, 因为它共享
    # !! embedding 和 lm_head)。
    ignore.extend([
        r"re:.*(?:mtp|stages)\.\d+\.(?:ffn|mlp)\.shared_experts.*",
        r"re:.*(?:mtp|stages)\.\d+\.(?:attn|self_attn)\..*",
        r"re:.*(?:mtp|stages)\.\d+\.(?:ffn|mlp)\.gate$",
        r"re:.*(?:mtp|stages)\.\d+\.(?:ffn|mlp)\.gate\..*",
        r"re:.*(?:mtp|stages)\.\d+\.(?:attn_norm|ffn_norm|main_norm|norm)$",
        r"re:.*(?:mtp|stages)\.\d+\.main_proj$",
        r"re:.*(?:mtp|stages)\.\d+\.markov_head\..*",
        r"re:.*(?:mtp|stages)\.\d+\.confidence_head\..*",
        r"re:.*(?:mtp|stages)\.\d+\.hc_.*",
        r"re:.*(?:mtp|stages)\.\d+\.embed.*",
        # MTP 的 engram 子树 (embed / wkv / q_weight / k_weight) 一律 BF16,
        # 与 ENGRAM_WKV_RE 只放行 layers.* 保持一致
        r"re:.*(?:mtp|stages)\.\d+\.engram\..*",
    ])

    if opts.mtp_experts == "bf16":
        # MTP routed experts 也保持 BF16: 直接整棵 MTP 子树 ignore。
        # targets 侧无需改动, 因为主 experts 的 target 已用负向先行断言排除 MTP。
        ignore.extend([
            r"re:.*stages\.\d+\..*",
            r"re:.*mtp\.\d+\..*",
        ])

    # ---- (c) 由开关决定的类别 ----
    if not opts.int8_p2:
        # 未启用优先级二: 三个 MLA 输入投影全部 ignore。
        # 同时覆盖 vLLM 的 `attn.fused_wqa_wkv` 与 sglang 的 `self_attn.wqkv_a`。
        ignore.extend([
            r"re:.*attn\.fused_wqa_wkv$",
            r"re:.*self_attn\.wqkv_a$",
            r"re:.*attn\.wq_a$",
            r"re:.*attn\.wkv$",
            r"re:.*attn\.wq_b$",
            r"re:.*self_attn\.wq_b$",
        ])
    # wo_a 默认 BF16, 不受 --int8-p3 影响, 仅由 --int8-wo-a 单独放行。
    # 它带 is_bmm=True, vLLM 的 deep_gemm_fp8_o_proj 直接读 wo_a.weight 做
    # .view(n_groups, o_lora_rank, -1) + torch.bmm, 只有 fp8 / bf16 两个分支,
    # INT8 会掉进 bf16 分支并在 view 处抛
    # "view size is not compatible with input tensor's size and stride"。
    # 这条手写路径绕过了 quant_method.apply(), 所以 linear 侧的
    # CutlassInt8ScaledMMLinearKernel 救不了它。同一缺口在 ROCm
    # (rocm_aiter_mla_sparse) / CPU (cpu_sparse) / XPU (xpu_sparse) 的
    # o_proj 路径同样存在, 因此默认在量化侧剔除是跨平台的。
    # 开启 --int8-wo-a 即表示引擎侧已自备 INT8 wo_a kernel, 此时不再 ignore。
    # wo_a 不在 packed_modules_mapping 里, 单独 ignore 不会触发 partial-ignore。
    if not opts.int8_wo_a:
        ignore.append(r"re:.*attn\.wo_a$")
    if not opts.int8_p3:
        # 未启用优先级三: MLA 输出投影 + indexer.wq_b 全部 ignore。
        ignore.extend([
            r"re:.*attn\.wo_b$",
            r"re:.*attn\.indexer\.wq_b$",
        ])
    if opts.engram_wkv_mode != "int8":
        ignore.append(r"re:.*engram\.wkv$")

    return ignore


def build_compression_config(opts: ConvOptions) -> dict:
    """生成 compressed-tensors 多 config_groups 配置。

    每个 group 显式声明 `format`, 因为:
      - INT4 必须 pack-quantized, INT8 必须 int-quantized;
      - vLLM `_get_scheme_from_parts` 会先按 format 分流
        (compressed_tensors.py:744-905): 若 INT8 组误用顶层 pack-quantized,
        会命中 `_is_wNaM_int` 而落到 WNA8Int(Humming) 而不是 W8A8Int8。

    顶层 format 作为兜底取 pack-quantized, 与 GLM 脚本一致。
    """
    int8_group_kwargs = {
        "input_activations": _int8_input_activations_arg(),
        "output_activations": None,
        "weights": _int8_weights_arg(),
        "format": "int-quantized",
    }

    groups: dict[str, dict] = {
        # ---- 优先级一: 主干 routed experts INT4 (已排除 MTP 子树) ----
        "group_int4_routed_experts": {
            "targets": [_BACKBONE_EXPERTS_TARGET],
            "input_activations": _input_activations_arg(opts.mode),
            "output_activations": None,
            "weights": _int4_weights_arg(opts.group_size),
            "format": "pack-quantized",
        },
        # ---- 优先级一: shared experts INT8 ----
        # 注意 MTP 的 ffn.shared_experts 会被这个 target 命中, 所以它必须留在
        # ignore 里 (见 build_ignore_list 的 MTP 分节)。
        "group_int8_shared_experts": {
            "targets": [
                r"re:.*shared_experts\.(?:gate_up_proj|down_proj|w[123])$",
            ],
            **int8_group_kwargs,
        },
    }

    # ---- MTP (DSpark draft) routed experts: 独立 group ----
    # 单独成组而不是复用主干 group, 这样 --mtp-experts 的档位变化只影响本组,
    # 且从 config 一眼能看出 MTP 的档位。
    if opts.mtp_experts == "int4":
        groups["group_int4_mtp_experts"] = {
            "targets": [_MTP_EXPERTS_TARGET],
            "input_activations": _input_activations_arg(opts.mode),
            "output_activations": None,
            "weights": _int4_weights_arg(opts.group_size),
            "format": "pack-quantized",
        }
    elif opts.mtp_experts == "int8":
        groups["group_int8_mtp_experts"] = {
            "targets": [_MTP_EXPERTS_TARGET],
            **int8_group_kwargs,
        }
    # bf16: 不加 group; MTP 整棵子树已在 ignore 里

    # ---- 优先级二 (可选): MLA 输入投影 ----
    if opts.int8_p2:
        groups["group_int8_attn_input"] = {
            "targets": [
                # vLLM 融合名 + sglang 融合名 + checkpoint 原始名, 同时覆盖
                r"re:.*layers\.\d+\.attn\.(?:fused_wqa_wkv|wq_a|wkv|wq_b)$",
                r"re:.*layers\.\d+\.self_attn\.(?:wqkv_a|wq_b)$",
            ],
            **int8_group_kwargs,
        }

    # ---- 优先级三 (可选): MLA 输出投影 wo_b + indexer.wq_b + engram.wkv ----
    # wo_a 不在此组: vLLM 的 o_proj 手写路径只支持 fp8/bf16, 见 build_ignore_list;
    # 如需量化请用 --int8-wo-a (下方 group_int8_wo_a)。
    if opts.int8_p3:
        groups["group_int8_attn_output"] = {
            "targets": [
                r"re:.*layers\.\d+\.(?:attn|self_attn)\.wo_b$",
            ],
            **int8_group_kwargs,
        }
        groups["group_int8_indexer_wq_b"] = {
            "targets": [
                r"re:.*layers\.\d+\.(?:attn|self_attn)\.indexer\.wq_b$",
            ],
            **int8_group_kwargs,
        }
        if opts.engram_wkv_mode == "int8":
            groups["group_int8_engram_wkv"] = {
                "targets": [r"re:.*layers\.\d+\.engram\.wkv$"],
                **int8_group_kwargs,
            }

    # ---- 可选: wo_a (分组低秩 O 投影) ----
    # 独立于 p3, 单独成组便于从 config 一眼看出; 需引擎侧自备 INT8 BMM kernel。
    if opts.int8_wo_a:
        groups["group_int8_wo_a"] = {
            "targets": [r"re:.*layers\.\d+\.(?:attn|self_attn)\.wo_a$"],
            **int8_group_kwargs,
        }

    return {
        "config_groups": groups,
        "format": "pack-quantized",
        "ignore": build_ignore_list(opts),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
    }


# --------------------------------------------------------------------------- #
# index.json / config.json 更新
# --------------------------------------------------------------------------- #

def _rebuild_index(
    output_dir: str,
    limit_files: int | None,
) -> None:
    """
    重建 model.safetensors.index.json 的 weight_map:
      读取每个新 shard 的实际 tensor keys, 覆盖旧 weight_map.
    这样自动:
      - 加入 routed experts 新键 (weight_packed / weight_scale / weight_shape)
      - 加入 INT8 层的 weight (int8) + weight_scale (fp32)
      - 剔除所有已反量化为 BF16 层的旧 .scale
      - 保留 engram.embed 的 .scale 透传
    """
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return

    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)

    files = sorted(glob(os.path.join(output_dir, "*.safetensors")))
    if limit_files is not None:
        files = files[:limit_files]

    new_weight_map: dict[str, str] = {}
    total_size = 0
    for path in files:
        fname = os.path.basename(path)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_weight_map[key] = fname
                t = f.get_tensor(key)
                total_size += t.numel() * t.element_size()

    model_index["weight_map"] = new_weight_map
    if "metadata" in model_index and isinstance(model_index["metadata"], dict):
        model_index["metadata"]["total_size"] = total_size
    else:
        model_index["metadata"] = {"total_size": total_size}

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(model_index, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_metadata(
    input_dir: str,
    output_dir: str,
    opts: ConvOptions,
    limit_files: int | None,
) -> None:
    """复制配置文件, 更新 quantization_config, 并重建 index."""
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "configuration.json",
        "model.safetensors.index.json",
        "README.md",
        "LICENSE",
    ]:
        src = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(output_dir, fname))

    _rebuild_index(output_dir, limit_files)

    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 移除原始 fp8/fp4 声明,替换为 compressed-tensors
    config.pop("compression_config", None)
    config.pop("quantization_config", None)

    config["quantization_config"] = build_compression_config(opts)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, sort_keys=True)


def _engram_wkv_desc(mode: str) -> str:
    return {
        "bf16": "BF16 (from FP8)",
        "blockwise": "FP8 blockwise (passthrough)",
        "int8": "INT8 channelwise (from FP8)",
    }[mode]


def _mtp_desc(mode: str) -> str:
    return {
        "int4": "INT4 pack-quantized (原 MXFP4 -> 同 4bit 档位; 6.34 GiB vs 原始 6.72 GiB)",
        "int8": "INT8 int-quantized (原 MXFP4 -> 升级到 8bit; 12.67 GiB, 1.88x)",
        "bf16": "BF16 (原 MXFP4 -> 升级到 16bit; 25.31 GiB, 3.76x)",
    }[mode]


def write_summary(
    output_dir: str,
    stats: dict[str, int],
    args,
    opts: ConvOptions,
    devices: list[torch.device],
    workers: int,
) -> None:
    int4_targets = [
        "主干 routed experts layers.*.ffn.experts.*.w[123]"
    ]
    int8_targets = ["shared_experts (layers.*.ffn.shared_experts.w[123])"]
    if opts.mtp_experts == "int4":
        int4_targets.append("MTP experts mtp.*.ffn.experts.*.w[123]")
    elif opts.mtp_experts == "int8":
        int8_targets.append("MTP experts mtp.*.ffn.experts.*.w[123]")
    if opts.int8_p2:
        int8_targets.append("MLA 输入投影 (attn.wq_a / attn.wkv / attn.wq_b)")
    if opts.int8_wo_a:
        int8_targets.append("MLA 分组 O 投影 (attn.wo_a, 需引擎自带 INT8 BMM kernel)")
    if opts.int8_p3:
        int8_targets.append("MLA 输出投影 (attn.wo_b)")
        int8_targets.append("DSA indexer (attn.indexer.wq_b)")
        if opts.engram_wkv_mode == "int8":
            int8_targets.append("engram.wkv")

    summary = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "priority_1_always_on": True,
        "priority_2_enabled": opts.int8_p2,
        "priority_3_enabled": opts.int8_p3,
        "int8_wo_a_enabled": opts.int8_wo_a,
        "mode": opts.mode,
        "group_size": opts.group_size,
        "engram_wkv_mode": opts.engram_wkv_mode,
        "mtp_experts": opts.mtp_experts,
        "mtp_experts_original_dtype": "MXFP4 (E2M1 + E8M0 scale, block 32)",
        "limit_files": args.limit_files,
        "compute": {
            "device": args.device,
            "gpu_ids": args.gpu_ids,
            "workers_requested": args.workers,
            "workers_effective": workers,
            "devices_used": [str(d) for d in devices],
        },
        "layout": {
            "int4_targets": (
                "; ".join(int4_targets)
                + " : INT4 pack-quantized (int32-lane, per-channel or per-group)"
            ),
            "int8_targets": (
                "; ".join(int8_targets)
                + " : INT8 int-quantized channelwise "
                  "(int8 weight + fp32 per-out-channel scale, per-token dynamic act)"
            ),
            "mtp_experts": _mtp_desc(opts.mtp_experts),
            "mtp_non_expert": (
                "BF16 (原 FP8 blockwise 或无 scale): MTP 的 shared_experts / attn / "
                "ffn.gate / main_proj / main_norm / attn_norm / ffn_norm / norm / "
                "markov_head / confidence_head / hc_* / engram"
            ),
            "kept_bf16": (
                "attn.wo_a (除非 --int8-wo-a) / "
                "compressor (wkv/wgate/norm) / indexer.wk / indexer.weights_proj / "
                "all RMSNorm / MoE router gate / hc_* / attn_sink / "
                "engram.q_weight / engram.k_weight / vision / aligner / embed / head"
            ),
            "engram.embed": "FP8 e4m3fn + ue8m0 scale (passthrough)",
            "engram.wkv": _engram_wkv_desc(opts.engram_wkv_mode),
            "shared_experts_original_dtype": "FP8 E4M3FN blockwise 32x32 (原始有 .scale)",
        },
        "stats": stats,
    }
    with open(os.path.join(output_dir, "conversion_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #

def _self_check_pack() -> None:
    """确认 pack_int4_to_int32 的位布局符合 compressed-tensors 规范.

    输入   q = [-8, -1, 0, 1, 7, -8, 7, 0]
    +8 -> u = [ 0,  7, 8, 9,15,  0,15, 8]
    word (LSB first, 4-bit slot 升序):
      0 | (7<<4) | (8<<8) | (9<<12) | (15<<16) | (0<<20) | (15<<24) | (8<<28)
      = 0x8F0F9870 (unsigned 2400163952) = -1894803344 (signed int32)
    """
    q = torch.tensor([[-8, -1, 0, 1, 7, -8, 7, 0]], dtype=torch.int8)
    expected = -1894803344
    got = int(pack_int4_to_int32(q).item())
    assert got == expected, f"pack self-check failed: got={got}, expected={expected}"


def _self_check_int8() -> None:
    """确认 INT8 channelwise 量化的 shape / dtype / round-trip 误差上界。

    scale = abs_max / 127 => 单元素最大绝对误差 <= scale/2 = abs_max/254。
    """
    torch.manual_seed(0)
    w = torch.randn(8, 64) * 3.0
    q, scale = quantize_int8_channelwise(w)

    assert q.dtype == torch.int8, f"expect int8, got {q.dtype}"
    assert q.shape == w.shape, f"shape mismatch: {q.shape} vs {w.shape}"
    assert scale.dtype == torch.float32, f"expect fp32 scale, got {scale.dtype}"
    assert scale.shape == (8, 1), f"expect scale shape (8,1), got {tuple(scale.shape)}"
    assert int(q.abs().max()) <= 127, f"int8 out of range: {int(q.abs().max())}"

    deq = q.float() * scale
    err = (deq - w).abs()
    bound = scale.expand_as(w).abs() / 2.0 + 1e-6
    assert bool((err <= bound).all()), "int8 round-trip error exceeds scale/2 bound"

    # 全零行不应产生 NaN (clamp(min=1e-12) 保证)
    z_q, z_s = quantize_int8_channelwise(torch.zeros(2, 4))
    assert bool(torch.isfinite(z_s).all()), "zero row produced non-finite scale"
    assert int(z_q.abs().max()) == 0, "zero row produced non-zero codes"


def _self_check_ignore_partial(opts: ConvOptions, model_dir: str) -> None:
    """静态检查: 融合层的 ignore 不能出现"部分命中"。

    vLLM `should_ignore_layer` 对 packed_modules_mapping 展开后的 shard
    若只有部分命中 ignore, 会 raise ValueError。这里按
    DeepseekV41LLMForCausalLM.packed_modules_mapping 的三个融合层做一次
    本地模拟, 提前在转换阶段发现问题, 而不是等到推理加载时才崩。
    """
    ignore = build_ignore_list(opts)
    fused_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    def _matches(value: str) -> bool:
        # 与 vLLM `is_equal_or_regex_match` 一致: re: 前缀走 re.match (前缀锚定),
        # 其余走精确相等 (config_utils.py:306-323)。
        for pattern in ignore:
            if pattern.startswith("re:"):
                if re.match(pattern[3:], value):
                    return True
            elif value == pattern:
                return True
        return False

    probes = [
        "model.layers.0.ffn.shared_experts.gate_up_proj",
        "model.layers.0.attn.fused_wqa_wkv",
        "model.layers.2.attn.compressor.fused_wkv_wgate",
        # 草稿侧同样存在 fused 层 (draft 用同一个 DeepseekV4DecoderLayer)。
        # 必须用运行时层号而非 `mtp.` 前缀 —— vLLM 从不产生 `mtp.` 模块名,
        # 用 checkpoint 名探测会恒真空通过, 参见 _draft_runtime_layers。
        *(f"model.layers.{L}.{proj}"
          for L in _draft_runtime_layers(model_dir)
          for proj in ("ffn.shared_experts.gate_up_proj", "attn.fused_wqa_wkv")),
    ]
    for layer_name in probes:
        proj_name = layer_name.split(".")[-1]
        shards = [
            layer_name.replace(proj_name, s) for s in fused_mapping[proj_name]
        ]
        hits = [_matches(s) for s in shards]
        assert all(hits) or not any(hits), (
            f"partial ignore on fused layer {layer_name}: "
            f"{dict(zip(shards, hits))} -> vLLM would raise ValueError"
        )


def _self_check_mtp_ignore(opts: ConvOptions) -> None:
    """断言 MTP/DSpark 子树的 ignore 不变量。

    这是本脚本最容易出错、且失败代价最高的一处:
      - MTP 的 ffn.shared_experts 会被 group_int8_shared_experts 的 target 命中
        (该 target 未限定 layers.*), 而我们把 MTP shared experts 写成 BF16。
        若漏 ignore -> 加载时 int8 scheme 套在 bf16 权重上 -> 崩。
      - MTP 的 experts 则相反: 必须"不被 ignore"才能拿到 group_int4/int8_mtp_experts。

    两种前缀 (vLLM `model.mtp.N.*` / sglang `stages.N.*`) 都要检查。
    """
    ignore = build_ignore_list(opts)

    # 与 vLLM `is_equal_or_regex_match` 一致
    def _ignored(value: str) -> bool:
        for pattern in ignore:
            if pattern.startswith("re:"):
                if re.match(pattern[3:], value):
                    return True
            elif value == pattern:
                return True
        return False

    # MTP 的非 experts 部分: 必须全部 ignore (我们写 BF16)
    must_ignore = [
        "{p}ffn.shared_experts.gate_up_proj",
        "{p}ffn.shared_experts.down_proj",
        "{p}mlp.shared_experts.gate_up_proj",   # sglang 命名
        "{p}attn.fused_wqa_wkv",
        "{p}attn.wo_a",
        "{p}attn.wo_b",
        "{p}attn.indexer.wq_b",
        "{p}ffn.gate",
        "{p}main_proj",
        "{p}main_norm",
        "{p}norm",
        "{p}attn_norm",
        "{p}ffn_norm",
        "{p}markov_head.head",
        "{p}confidence_head.proj",
        "{p}engram.wkv",                        # MTP 的 engram 不参与 INT8
    ]
    for prefix in ("model.mtp.0.", "stages.0."):
        for tmpl in must_ignore:
            name = tmpl.format(p=prefix)
            assert _ignored(name), (
                f"MTP 非 experts 模块未被 ignore: {name}\n"
                f"  -> 会被某个 config_group 的 target 命中, "
                f"而本脚本把它写成 BF16, 加载时会 dtype 不匹配"
            )

    # MTP 的 experts: bf16 档位必须 ignore, int4/int8 档位必须不 ignore
    for prefix in ("model.mtp.0.", "stages.0."):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            name = f"{prefix}ffn.experts.0.{proj}"
            if opts.mtp_experts == "bf16":
                assert _ignored(name), (
                    f"--mtp-experts bf16 但 {name} 未被 ignore -> 会被 INT4/INT8 group 命中"
                )
            else:
                assert not _ignored(name), (
                    f"--mtp-experts {opts.mtp_experts} 但 {name} 被 ignore -> 写了 "
                    f"{opts.mtp_experts} 却拿不到对应 scheme, 加载时会 dtype 不匹配"
                )


def _draft_runtime_layers(model_dir: str) -> list[int]:
    """草稿层在 vLLM 运行时占用的层号。

    dspark 把草稿 decoder 注册在 `layers.{num_hidden_layers + i}`
    (vllm/models/deepseek_v41/nvidia/dspark.py:116), 即 layers.40/41/42 ——
    **不是** `mtp.*`。config_group 的 target 与 ignore 都在运行时模块名上匹配,
    所以任何 MTP 相关的静态检查都必须用这组层号, 用 `mtp.` 前缀会恒真空通过。
    """
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    n_layers = int(text_cfg["num_hidden_layers"])
    n_draft = int(text_cfg.get("num_nextn_predict_layers", 0))
    return list(range(n_layers, n_layers + n_draft))


def _self_check_draft_runtime_names(opts: ConvOptions, model_dir: str) -> None:
    """静态检查: config 在**运行时模块名**上的解析结果, 与脚本实际写出的 dtype 一致。

    这是 `_self_check_mtp_ignore` 的运行时版本。后者用 `mtp.` 前缀断言, 而 vLLM
    从不产生这种模块名, 所以它对本类 bug 完全无感。

    错配后果: config 说 INT8 而 checkpoint 里是 BF16, 草稿主干被污染。
    草稿与目标共享 embedding 和 lm_head, 所以仍能输出合法 token 但每个都错 ——
    表现为 MTP 接受率塌到 0, 而主模型输出仍然连贯。
    """
    ignore = build_ignore_list(opts)
    cfg = build_compression_config(opts)
    groups = cfg["config_groups"]
    groups = list(groups.values()) if isinstance(groups, dict) else groups
    targets = [t for g in groups for t in g["targets"]]

    def _ignored(name: str) -> bool:
        for p in ignore:
            if p.startswith("re:"):
                if re.match(p[3:], name):
                    return True
            elif name == p:
                return True
        return False

    def _matched_target(name: str) -> str | None:
        for t in targets:  # vLLM 取第一个命中 (compressed_tensors/utils.py:73)
            if t.startswith("re:"):
                if re.match(t[3:], name):
                    return t
            elif t == name:
                return t
        return None

    draft_layers = _draft_runtime_layers(model_dir)
    assert draft_layers, "未能从 config.json 解析出草稿层号"

    # (proj, 是否应在"开启对应开关时"被量化)
    attn_cases = [
        ("attn.fused_wqa_wkv", opts.int8_p2, "p2"),
        ("attn.wq_b", opts.int8_p2, "p2"),
        ("attn.wo_b", opts.int8_p3, "p3"),
        ("attn.indexer.wq_b", opts.int8_p3, "p3"),
        ("attn.wo_a", opts.int8_wo_a, "int8-wo-a"),
    ]
    for layer in draft_layers:
        for suffix, enabled, flagname in attn_cases:
            name = f"model.layers.{layer}.{suffix}"
            tgt = _matched_target(name)
            if enabled:
                assert not _ignored(name) and tgt is not None, (
                    f"草稿层 {name} 在 --{flagname} 开启时既被 ignore 也没有 config_group "
                    f"命中 -> 运行时按 BF16 建参数, 但脚本已把它写成 INT8"
                )
            else:
                assert _ignored(name), (
                    f"草稿层 {name} 未被 ignore, 但 --{flagname} 是关的 -> 运行时会被 "
                    f"{tgt!r} 命中并按 INT8 建参数, 而脚本写的是 BF16"
                )

        # wo_a 的放行与否已由上面的 attn_cases (--int8-wo-a) 覆盖;
        # 关闭时必须 ignore (deep_gemm_fp8_o_proj 没有 INT8 分支), 开启时必须被 group_int8_wo_a 命中。

        # shared_experts 的 int8 组是无条件的, 草稿层必须同样放行
        shared = f"model.layers.{layer}.ffn.shared_experts.gate_up_proj"
        assert not _ignored(shared) and _matched_target(shared) is not None, (
            f"草稿层 {shared} 未被 shared_experts 组命中或被 ignore -> "
            f"config 与脚本写出的 dtype 不一致"
        )

    # 写侧: 这些张量必须真的被量化 (谓词认 mtp. 前缀)
    for layer in draft_layers:
        for suffix, enabled in [
            ("attn.wq_b.weight", opts.int8_p2),
            ("attn.wo_a.weight", opts.int8_wo_a),
            ("ffn.shared_experts.w1.weight", True),
        ]:
            ckpt = f"mtp.0.{suffix}"
            assert is_int8_target(ckpt, opts) == enabled, (
                f"{ckpt} 的 is_int8_target={is_int8_target(ckpt, opts)}, 期望 {enabled}"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = ArgumentParser(
        description=(
            "Convert DeepSeek-V4.1-Flash to mixed-precision INT4/INT8 "
            "(compressed-tensors multi config_groups). Priority 1 "
            "(routed experts INT4 + shared experts INT8) is always on; "
            "priority 2 / 3 are opt-in and default off."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=r"/models/DeepSeek-V4.1-Flash",
        help="Path to DeepSeek-V4.1-Flash checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"/models/DeepSeek-V4.1-Flash-INT4-INT8",
        help="Path to output converted checkpoint directory.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["w4a8", "w4a16"],
        default="w4a8",
        help="INT4 group activation declaration (weight bytes identical, only config differs). "
             "w4a8 (default) declares INT8 dynamic per-token symmetric activations, matching "
             "the CompressedTensorsW4A8Int8 MoE path; w4a16 declares BF16 activations.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=-1,
        help="INT4 group size along input dim K. -1 = per-channel (default). "
        "Positive values (e.g. 64, 128) enable per-group quantization; K must be divisible.",
    )
    parser.add_argument(
        "--int8-p2",
        action="store_true",
        default=False,
        help=(
            "Enable priority 2: quantize the MLA input projections "
            "layers.*.attn.wq_a / wkv / wq_b from FP8 to INT8 channelwise. "
            "Default: off. Risk: wq_a+wkv are runtime-fused into "
            "attn.fused_wqa_wkv, and wq_b feeds MLA Q-norm / RoPE / sparse "
            "attention with backend-specific layout permutation. Verify the "
            "engine's fused loader and MLA kernels before use."
        ),
    )
    parser.add_argument(
        "--int8-p3",
        action="store_true",
        default=False,
        help=(
            "Enable priority 3: quantize the MLA output projection "
            "layers.*.attn.wo_b, the DSA indexer query "
            "layers.*.attn.indexer.wq_b, and engram.wkv from FP8 to INT8 "
            "channelwise. Default: off. layers.*.attn.wo_a is always left "
            "BF16: it is excluded from this group because vLLM's dedicated "
            "o_proj path (deep_gemm_fp8_o_proj) bypasses quant_method.apply() "
            "and only implements fp8/bf16 branches. Remaining risks: wo_b may "
            "receive a QuantizedActivation / GEMM-RS input, and indexer.wq_b "
            "affects the discrete DSA top-k selection. Requires parity "
            "evaluation."
        ),
    )
    parser.add_argument(
        "--int8-wo-a",
        action="store_true",
        default=False,
        help=(
            "Quantize layers.*.attn.wo_a (and the MTP draft's wo_a) from FP8 to "
            "INT8 channelwise, as its own config group. Independent of "
            "--int8-p2/--int8-p3. Default: off (wo_a stays BF16). WARNING: stock "
            "vLLM cannot run the result: wo_a is_bmm=True and its o_proj path "
            "(deep_gemm_fp8_o_proj) only implements fp8/bf16 branches, so an "
            "INT8 wo_a fails at the first forward (profile_run). Enable only if "
            "your engine provides a grouped INT8 BMM kernel for wo_a."
        ),
    )
    parser.add_argument(
        "--engram-wkv-mode",
        type=str,
        choices=list(ENGRAM_WKV_MODES),
        default=None,
        help=(
            "Explicit override for engram.wkv. If omitted, it follows "
            "--int8-p3 (int8) or defaults to bf16. Note the original checkpoint "
            "stores engram.wkv as FP8, so 'int8' preserves the bit width while "
            "'bf16' is a precision upgrade; 'blockwise' passes the raw FP8 "
            "through with its bare .scale."
        ),
    )
    parser.add_argument(
        "--mtp-experts",
        type=str,
        choices=list(MTP_EXPERTS_MODES),
        default="int4",
        help=(
            "MTP (DSpark draft) routed experts format, for "
            "mtp.*.ffn.experts.*.w[123]. The original checkpoint stores them as "
            "MXFP4 (E2M1 + E8M0, block 32), same as the backbone routed experts. "
            "'int4' (default) keeps the same 4-bit tier at 6.34 GiB vs the "
            "original 6.72 GiB, and matches the backbone; 'int8' is a precision "
            "upgrade at 12.67 GiB (1.88x); 'bf16' at 25.31 GiB (3.76x). "
            "Draft precision drives the speculative-decoding acceptance rate, so "
            "raise this only if measured acceptance regresses at int4. "
            "int4 shares --group-size with the backbone experts."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="Torch CPU thread count (per worker in multi-GPU mode).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device: auto (GPU if torch.cuda.is_available() else CPU), "
        "cpu (force CPU), cuda (require GPU, error if unavailable).",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA device indices to use, e.g. '0,1,3'. "
        "Default: all detected GPUs (torch.cuda.device_count()).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker processes across GPUs. "
        "Default: min(#GPUs, #shards). Ignored in CPU / single-GPU mode.",
    )
    parser.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="Convert only first N shard files (smoke testing).",
    )
    args = parser.parse_args()

    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        raise ValueError("input-dir and output-dir must be different")

    engram_wkv_mode = resolve_engram_wkv_mode(args.int8_p3, args.engram_wkv_mode)
    opts = ConvOptions(
        group_size=args.group_size,
        mode=args.mode,
        int8_p2=args.int8_p2,
        int8_p3=args.int8_p3,
        int8_wo_a=args.int8_wo_a,
        engram_wkv_mode=engram_wkv_mode,
        mtp_experts=args.mtp_experts,
    )

    torch.set_num_threads(args.num_threads)
    _self_check_pack()
    _self_check_int8()
    _self_check_ignore_partial(opts, args.input_dir)
    _self_check_mtp_ignore(opts)
    _self_check_draft_runtime_names(opts, args.input_dir)

    print(f"Converting {args.input_dir} to mixed INT4/INT8 format...")
    print(f"  output_dir        : {args.output_dir}")
    print(f"  mode (INT4 group) : {opts.mode}")
    print(
        f"  group_size        : {opts.group_size} "
        f"({'per-channel' if opts.group_size == -1 else 'per-group'})"
    )
    print(f"  priority 1        : ON  (routed experts INT4 + shared experts INT8)")
    print(f"  priority 2        : {'ON ' if opts.int8_p2 else 'off'} (attn.wq_a/wkv/wq_b -> INT8)")
    print(f"  priority 3        : {'ON ' if opts.int8_p3 else 'off'} (attn.wo_b, indexer.wq_b -> INT8)")
    print(f"  int8 wo_a         : {'ON ' if opts.int8_wo_a else 'off'} (attn.wo_a -> INT8; off = BF16)")
    if opts.int8_wo_a:
        print("  !! WARNING: --int8-wo-a 产物需要引擎自带 wo_a INT8 BMM kernel, "
              "未打补丁的 vLLM 会在首次前向崩溃 (见脚本顶部说明)")
    print(f"  engram_wkv_mode   : {opts.engram_wkv_mode}")
    print(
        f"  mtp_experts       : {opts.mtp_experts}"
        f"{'  (原 MXFP4 -> 同档位 INT4)' if opts.mtp_experts == 'int4' else ''}"
    )
    print(f"  device            : {args.device}"
          + (f" (cuda_available={torch.cuda.is_available()},"
             f" device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0})"))
    if args.gpu_ids:
        print(f"  gpu_ids           : {args.gpu_ids}")
    if args.workers is not None:
        print(f"  workers           : {args.workers}")

    stats, devices, workers = convert_model(
        args.input_dir,
        args.output_dir,
        opts,
        args.limit_files,
        args.device,
        args.gpu_ids,
        args.workers,
        args.num_threads,
    )
    copy_metadata(args.input_dir, args.output_dir, opts, args.limit_files)
    write_summary(args.output_dir, stats, args, opts, devices, workers)

    print()
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True, indent=2))
    print(f"\nDone! Mixed INT4/INT8 model saved to {args.output_dir}")
    print(f"  devices used   : {[str(d) for d in devices]}")
    print(f"  workers used   : {workers}")


if __name__ == "__main__":
    main()
