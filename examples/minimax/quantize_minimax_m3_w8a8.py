#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 独立 W8A8 PTQ 量化脚本（支持 INT8 / FP8 两种配方）。

--scheme W8A8 / INT8 : per-channel INT8 权重 + per-token 动态 INT8 激活
--scheme FP8_DYNAMIC : per-channel FP8(e4m3) 权重 + per-token 动态 FP8 激活（即 W8A8-FP8）
两种配方的激活量化都是 dynamic，因此都不需要校准数据集。

关于专家层（重点）
------------------
M3 绝大部分权重在 MoE 专家里，脚本用 targets="Linear" 就够了：llmcompressor 会在
oneshot() 内部自动把专家线性化成 nn.Linear 再量化，无需额外配置。

产物里的专家键名是线性化命名：
    ...block_sparse_moe.experts.N.{gate_proj,up_proj,down_proj}.weight[_scale]
而不是官方 ckpt 的 ...experts.N.{w1,w3,w2}.weight。因此**回读产物必须走
llmcompressor 的线性化加载路径**（本脚本会把 minimax_m3_vl 注册进
load_quantizable_moe 后加载；vLLM 的 compressed-tensors 走同一套）。
直接用普通 transformers.from_pretrained 读会报 experts.gate_up_proj/down_proj MISSING，
专家权重被随机初始化。

为什么不复用 main.py / llmc_oneshot.py
--------------------------------------
M3 的 checkpoint `model_type == "minimax_m3_vl"`：
  1) `auto_map` 只声明了 `AutoConfig`，目录里没有 `modeling_*.py`，所以
     `has_remote_code` 判定为 False；
  2) transformers 的 `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES` 里只有
     `minimax_m3_vl_text`，没有 `minimax_m3_vl`，所以 `has_local_code` 也为 False。

于是 llmc_oneshot.py 里固定的
    AutoModelForCausalLM.from_pretrained(...)
会直接抛 `ValueError: Unrecognized configuration class MiniMaxM3VLConfig`。

本脚本按 `model_type` 分派加载器（多模态走 AutoModelForImageTextToText），
其余流程（QuantizationModifier + oneshot + save_compressed）与 llmc_oneshot 保持一致。

用法
----
    # 预检：只解析 config、检查加载器、列出将被量化/被忽略的模块（不加载权重）
    python3 quantize_minimax_m3_w8a8.py --model /llm_models_1/MiniMax-M3 --dry-run

    # 数据无关的 W8A8(INT8) PTQ（默认）
    python3 quantize_minimax_m3_w8a8.py \
        --model /llm_models/MiniMax-M3 \
        --save-dir /quant_models/MiniMax-M3-W8A8

    # 数据无关的 W8A8-FP8 PTQ
    python3 quantize_minimax_m3_w8a8.py \
        --model /llm_models/MiniMax-M3 \
        --scheme FP8_DYNAMIC \
        --save-dir /quant_models/MiniMax-M3-W8A8-FP8

    # 显存不够时（428B BF16，建议至少用 auto_offload；或直接用 torchrun 多卡）
    python3 quantize_minimax_m3_w8a8.py --model /llm_models/MiniMax-M3 \
        --device-map auto_offload --save-dir /quant_models/MiniMax-M3-W8A8

    torchrun --nproc_per_node=8 quantize_minimax_m3_w8a8.py \
        --model /llm_models/MiniMax-M3 --save-dir /quant_models/MiniMax-M3-W8A8
"""

import argparse
import contextlib
import os
import re
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

# 多模态 model_type：AutoModelForCausalLM 不认，必须走 ImageTextToText
MULTIMODAL_MODEL_TYPES = {"minimax_m3_vl", "minimax_m2_vl", "minimax_m2_mini_vl"}

# M3 默认忽略列表（显式 --ignore 会整体覆盖它）。
# 模块名以 transformers 的模块树为准（官方 ckpt 的键名会由 transformers 的
# conversion_mapping.py:607 "minimax_m3_vl" 规则自动重排，例如
#   block_sparse_moe.  ->  .mlp.            （所以不能写 moe.gate）
#   index_q_proj.     ->  indexer.q_proj.
#   vision_tower.vision_model.encoder.layers. -> vision_tower.layers.
DEFAULT_IGNORE = [
    "re:.*lm_head$",                       # 输出头保持 BF16
    r"re:.*mlp\.gate$",                   # 路由层 TopKRouter（weight 是裸 Parameter，非 Linear）
    "re:.*vision_tower.*",                 # 视觉塔，与文本精度解耦
    "re:.*multi_modal_projector.*",        # 多模态投影
    # MSA 稀疏索引分支，若精度不达标可一并放开（默认参与量化）
    # r"re:.*self_attn\.indexer\..*$",
]

# 专家模块（MiniMaxM3VLExperts）在 checkpoint / 未线性化模型里是堆叠的 3D nn.Parameter：
#     mlp.experts.gate_up_proj  (num_experts, 2*intermediate, hidden)
#     mlp.experts.down_proj     (num_experts, hidden, intermediate)
#
# 它们不是 nn.Linear，但 llmcompressor 在 oneshot() 里会自动把 MoE 线性化
# （3D 参数 -> 每个专家一组 gate_proj/up_proj/down_proj nn.Linear），因此
# targets="Linear" **能覆盖专家层**（M3：57 个 MoE 层 x 128 专家全部参与量化）。
#
# 注意产物键名：线性化后保存出来是
#     ...block_sparse_moe.experts.N.{gate_proj,up_proj,down_proj}.weight[_scale]
# 而官方 ckpt 是 ...experts.N.{w1,w3,w2}.weight。所以产物必须走 llmcompressor 的
# 线性化加载路径（load_quantizable_moe；vLLM 的 compressed-tensors 走同一套）回读，
# 用普通 transformers.from_pretrained 直读会丢专家权重（UNEXPECTED + MISSING）。
EXPERT_MODULE_SUFFIX = ".mlp.experts"

# 把 minimax_m3_vl 注册进 llmcompressor 的 MoE 线性化表（0.13.0 尚无该条目）。
# 注册后才能用 load_quantizable_moe 一边加载一边线性化，并注册保存用的键名映射。
MOE_ARCH_IMPORT_PATHS = (
    "transformers.models.minimax_m3_vl.configuration_minimax_m3_vl.MiniMaxM3VLTextConfig",
    "transformers.models.minimax_m3_vl.modeling_minimax_m3_vl.MiniMaxM3VLExperts",
)
# 官方 ckpt（2D per-expert）-> 线性化后的 2D 模块名：w1=gate, w3=up, w2=down
MOE_ARCH_2D_RENAMINGS = (
    ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    [
        (r"\.mlp\.experts\.(\d+)\.w1\.", r".mlp.experts.\1.gate_proj."),
        (r"\.mlp\.experts\.(\d+)\.w2\.", r".mlp.experts.\1.down_proj."),
        (r"\.mlp\.experts\.(\d+)\.w3\.", r".mlp.experts.\1.up_proj."),
    ],
)


def _register_moe_linearization() -> bool:
    """向 llmcompressor 注册 M3 的 MoE 线性化映射，返回是否可用。"""
    try:
        from llmcompressor.modeling.moe.conversion_mappings import (
            ARCH_TO_IMPORT_PATHS,
            ARCH_TO_2D_MAPPINGS,
        )
        from llmcompressor.modeling.moe.linearize import has_linearize_load_mappings
    except ImportError:
        return False

    from transformers.core_model_loading import WeightRenaming

    remove_targets, renamings = MOE_ARCH_2D_RENAMINGS
    ARCH_TO_IMPORT_PATHS.setdefault("minimax_m3_vl", MOE_ARCH_IMPORT_PATHS)
    ARCH_TO_2D_MAPPINGS.setdefault(
        "minimax_m3_vl",
        (list(remove_targets), [WeightRenaming(s, t) for s, t in renamings]),
    )
    return has_linearize_load_mappings("minimax_m3_vl")


# ============================================================
# ignore 匹配（语义与 compressed_tensors.utils.match.match_name 一致）
# ============================================================

def _match_one(name: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        # 注意：llmcompressor 用的是 re.match（从头锚定），不是 re.search
        return re.match(pattern[len("re:"):], name) is not None
    return pattern == name


def _is_ignored(name: str, ignore: list[str]) -> bool:
    return any(_match_one(name, p) for p in ignore)


# ============================================================
# 环境自检：torch 的共享内存管理程序
# ============================================================

def _ensure_torch_shm_manager() -> None:
    """确保 torch 自带的 `torch_shm_manager` 可执行。

    分布式（torchrun）下只要有任意一层被 offload 到 CPU，compressed_tensors 就会选中
    DistributedCPUCache（见 OffloadCache.cls_from_device：("cpu", True) ->
    DistributedCPUCache）。rank0 会用 torch 的 managed shared memory 把权重放进
    /dev/shm 共享给其它 rank，这一步需要 execl `torch/bin/torch_shm_manager`。

    部分 DTK / torch wheel 或镜像构建会丢掉 `torch/bin/*` 的可执行位，于是 rank0 抛
        RuntimeError: torch_shm_manager ...: execl failed: Permission denied
    其余 rank 只是被 torchrun 连带 SIGTERM，日志被噪声淹没。这里提前补齐权限，
    补不了就给出可操作的报错。
    """
    bin_dir = Path(torch.__file__).parent / "bin"
    manager = bin_dir / "torch_shm_manager"
    if not manager.exists() or os.access(manager, os.X_OK):
        return

    fixed: list[str] = []
    for entry in sorted(bin_dir.iterdir()):
        if not entry.is_file() or os.access(entry, os.X_OK):
            continue
        try:
            entry.chmod(entry.stat().st_mode | 0o111)
            fixed.append(entry.name)
        except OSError as exc:
            print(f"[WARN] 无法为 {entry} 补可执行位：{exc}")

    if os.access(manager, os.X_OK):
        print(f"[INFO] 已为 torch/bin 下 {len(fixed)} 个文件补上可执行位"
              f"（含 torch_shm_manager）；此修复不持久，镜像重建后需重做")
        return

    raise SystemExit(
        f"[FAIL] {manager} 不可执行，分布式 + CPU offload 必然失败于\n"
        f"       RuntimeError: torch_shm_manager ... execl failed: Permission denied\n"
        f"       请任选其一：\n"
        f"         1) chmod +x {bin_dir}/*\n"
        f"         2) 去掉 torchrun，改单进程 + --device-map auto_offload\n"
        f"            （非分布式走 CPUCache，不使用 shm manager，可绕开该限制）"
    )


# ============================================================
# 配置解析 / 加载器分派
# ============================================================

def _load_config(model_path: str):
    """优先用 transformers 内置 config，失败再退回 remote code。

    M3 仓库里自带的 `configuration_minimax_m3_vl.py` 是从 `sglang.srt.configs.minimax_vl`
    抄过来的镜像版，只用于 AutoConfig 兼容，**不能**用来构建 transformers 的建模类：
    它会把 vision_config/text_config 退化成裸 PretrainedConfig，导致
    `AttributeError: 'PreTrainedConfig' object has no attribute 'temporal_patch_size'`。
    所以这里必须优先走内置 config（trust_remote_code=False）。
    """
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        print(f"[INFO] 使用 transformers 内置 config: {type(config).__name__}")
        return config
    except Exception as exc:
        print(f"[WARN] 内置 config 不可用（{type(exc).__name__}: {exc}），回退 remote code")
        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def _resolve_auto_class(config) -> tuple[type, str]:
    """按 model_type 选择 AutoModel 入口，返回 (AutoModel 类, 理由)。"""
    model_type = getattr(config, "model_type", None)
    if model_type in MULTIMODAL_MODEL_TYPES:
        return (
            AutoModelForImageTextToText,
            f"model_type={model_type!r} 未注册到 AutoModelForCausalLM，改用多模态入口",
        )
    return AutoModelForCausalLM, f"model_type={model_type!r} 走 CausalLM 入口"


def _check_auto_class_supported(auto_cls, config) -> None:
    """复刻 transformers 的类解析分支，提前给出可读的报错。"""
    auto_map = getattr(config, "auto_map", None) or {}
    has_remote_code = auto_cls.__name__ in auto_map
    has_local_code = type(config) in auto_cls._model_mapping
    if has_remote_code or has_local_code:
        return
    raise SystemExit(
        f"[FAIL] {auto_cls.__name__} 无法识别该 config：\n"
        f"       config class = {type(config).__name__}, model_type = {getattr(config, 'model_type', None)}\n"
        f"       auto_map     = {list(auto_map) or '(空)'}\n"
        f"       请确认 transformers 版本是否支持该 model_type，或调整 _resolve_auto_class() 的分派。"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniMax-M3 W8A8 (INT8 / FP8, PTQ) 独立量化脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="M3 模型目录（含 config.json）")
    parser.add_argument("--scheme", type=str, default="W8A8",
                        choices=["W8A8", "INT8", "FP8_DYNAMIC", "FP8_BLOCK"],
                        help="量化配方（均为免校准数据的 PTQ）：\n"
                             "  W8A8 / INT8  - per-channel INT8 权重 + per-token 动态 INT8 激活\n"
                             "  FP8_DYNAMIC  - per-channel FP8(e4m3) 权重 + per-token 动态 FP8 激活（W8A8-FP8）\n"
                             "  FP8_BLOCK    - 128x128 block FP8 权重 + per-token 动态 FP8 激活")
    parser.add_argument("--save-dir", type=str, default=None, help="量化产物输出目录")
    parser.add_argument("--ignore", type=str, default=None,
                        help="逗号分隔的 ignore 规则；显式指定会整体覆盖脚本内置默认值")
    parser.add_argument("--device-map", type=str, default="auto",
                        help="'auto' / 'cpu' / 'cuda:0' / 'auto_offload'")
    parser.add_argument("--pipeline", type=str, default="independent",
                        help="oneshot 流水线：basic / datafree / sequential / independent")
    parser.add_argument("--sequential-targets", type=str, default=None,
                        help="pipeline=sequential 时逐层处理的目标，如 'Linear'")
    parser.add_argument("--num-threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--targets", type=str, default="Linear",
                        help="量化目标模块类型。保持 'Linear' 即可：llmcompressor 会先把 M3 的 MoE "
                             "专家自动线性化成 nn.Linear，专家层同样会被量化，无需改成别的别名")
    parser.add_argument("--dry-run", action="store_true",
                        help="只做预检：解析 config、打印加载器与将量化/忽略的模块，不加载权重")
    return parser.parse_args()


# ============================================================
# 预检
# ============================================================

def _dry_run(config, auto_cls, ignore: list[str]) -> None:
    print("=" * 72)
    print("DRY RUN — 仅检查 config / 加载器 / 模块匹配，不读取权重")
    print("=" * 72)
    print(f"config class  : {type(config).__module__}.{type(config).__name__}")
    print(f"model_type    : {config.model_type}")
    print(f"architectures : {getattr(config, 'architectures', None)}")
    print(f"auto_map keys : {list(getattr(config, 'auto_map', None) or {})}")
    print(f"loader        : {auto_cls.__name__}")
    print(f"ignore ({len(ignore)}) :")
    for p in ignore:
        print(f"    {p}")

    try:
        from accelerate import init_empty_weights
    except ImportError:
        print("\n[WARN] 未安装 accelerate，跳过模块枚举。")
        return

    print("\n" + "=" * 72)
    print("模块匹配结果（空权重构建模型，仅看 nn.Linear）")
    print("=" * 72)
    with init_empty_weights():
        model = auto_cls.from_config(config, trust_remote_code=False)

    quantized, skipped = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        (skipped if _is_ignored(name, ignore) else quantized).append(name)

    def _summarize(items: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for n in items:
            key = re.sub(r"\.\d+\.", ".N.", n)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    print(f"\n>>> 将被量化 ({len(quantized)} 个 Linear) 按模块类型汇总：")
    for k, v in _summarize(quantized).items():
        print(f"    {v:>6}  {k}")
    print(f"\n>>> 被 ignore 跳过 ({len(skipped)} 个 Linear)：")
    for k, v in _summarize(skipped).items():
        print(f"    {v:>6}  {k}")

    # MoE 专家在未线性化时是 3D Parameter，但 oneshot() 内部会自动线性化后再量化
    n_experts = sum(
        1 for n, mod in model.named_modules()
        if n.endswith(EXPERT_MODULE_SUFFIX) and mod is not model
    )
    if n_experts:
        got = next(mod for n, mod in model.named_modules() if n.endswith(EXPERT_MODULE_SUFFIX))
        params = dict(got.named_parameters(recurse=False))
        shapes = {k: tuple(v.shape) for k, v in params.items()}
        n_mat = sum(v.shape[0] * (2 if k == "gate_up_proj" else 1) for k, v in params.items())
        print(
            f"\n>>> [INFO] 检测到 {n_experts} 个 MoE 专家模块（{EXPERT_MODULE_SUFFIX}），"
            f"未线性化时是堆叠的 3D Parameter: {shapes}\n"
            f"           oneshot() 会自动把它们线性化成 nn.Linear（每个专家一组 "
            f"gate_proj/up_proj/down_proj），\n"
            f"           因此 targets='Linear' **会量化专家层**"
            f"（共约 {n_experts * n_mat} 个专家矩阵，占模型绝大部分权重）。"
        )

    if not quantized:
        print("\n[WARN] 没有被量化的 Linear，请检查 --ignore 是否过宽。")
    print("\n预检完成。")


# ============================================================
# 模型 / 分词器
# ============================================================

def _should_init_dist() -> bool:
    return any(
        key in os.environ
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    )


def _load_model(args, config, auto_cls):
    use_distributed = _should_init_dist()
    if use_distributed:
        from compressed_tensors.offload import init_dist, load_offloaded_model

        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "7200")
        os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
        init_dist()
        rank = torch.distributed.get_rank()
        # 必须显式传 auto_cls：load_offloaded_model 只 patch 传入的 AutoModel 类，
        # M3 走的是 AutoModelForImageTextToText，用默认的 AutoModelForCausalLM 不保险
        load_ctx = load_offloaded_model(auto_cls)
    elif args.device_map == "auto_offload":
        from compressed_tensors.offload import load_offloaded_model

        rank = 0
        load_ctx = load_offloaded_model(auto_cls)
    else:
        rank = 0
        load_ctx = contextlib.nullcontext()

    # M3 的 MoE 专家在 ckpt 里是 2D per-expert（w1/w3/w2），未线性化模型里是 3D 堆叠参数。
    # 用 llmcompressor 的线性化加载路径：加载时就拆成 2D nn.Linear，并注册保存用的键名映射，
    # 否则产物会以“线性化命名”落盘、无法被普通 transformers 回读（专家权重全丢）。
    moe_linearized = False
    loader_ctx = contextlib.nullcontext()
    if _register_moe_linearization():
        from llmcompressor.modeling.moe.linearize import load_quantizable_moe

        loader_ctx = load_quantizable_moe(auto_cls)
        moe_linearized = True
        print("[INFO] 启用 MoE 线性化加载（load_quantizable_moe）")

    with load_ctx, loader_ctx:
        kwargs = dict(dtype="auto", device_map=args.device_map, trust_remote_code=False)
        if moe_linearized:
            # load_quantizable_moe 内部会自己解析 config，不能再传 config=
            model = auto_cls.from_pretrained(args.model, **kwargs)
        else:
            model = auto_cls.from_pretrained(args.model, config=config, **kwargs)
    return model, rank, use_distributed


def _load_tokenizer(model_path: str):
    """M3 是多模态，依次尝试：内置 processor -> remote processor -> tokenizer。"""
    from transformers import AutoProcessor

    for cls in (AutoProcessor, AutoTokenizer):
        for trc in (False, True):
            try:
                obj = cls.from_pretrained(model_path, trust_remote_code=trc)
                print(f"[INFO] 使用 {cls.__name__}(trust_remote_code={trc})")
                return obj
            except Exception as exc:
                print(f"[WARN] {cls.__name__}(trust_remote_code={trc}) 失败: {type(exc).__name__}")
    raise RuntimeError("无法加载 tokenizer/processor")


def _copy_non_weight_files(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        return
    for src in src_root.rglob("*"):
        if not src.is_file() or src.suffix == ".safetensors":
            continue
        if src.name == "model.safetensors.index.json":
            continue
        dst = dst_root / src.relative_to(src_root)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _rewrite_saved_ignore(save_dir: str, model) -> None:
    """把 config.json 里 quantization_config.ignore 换写成“模型内部模块名”。

    保存时 compressed_tensors 会按反向键名映射重写 ignore 列表，M3 的模块名与 ckpt 键名
    不一致（vision_tower / lm_head 会被改名，例如写成
    `vision_tower.vision_model.encoder.layers.0.self_attn.k_proj`、`language_model.lm_head`），
    回读时按模型内部名匹配不上，于是这些本该跳过的模块被当成“应量化”，
    报 `weight_scale MISSING`（实测回读后 lm_head 变 float32 + 假 scale，前向输出直接爆掉）。

    这里用模型里实际没有 quantization_scheme 的 Linear 模块名覆盖它，保证回读一致。
    """
    import json

    cfg_path = Path(save_dir) / "config.json"
    if not cfg_path.exists():
        return
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    qc = data.get("quantization_config")
    if not qc:
        return

    skipped = sorted(
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and getattr(mod, "quantization_scheme", None) is None
    )
    if not skipped:
        return

    old = qc.get("ignore") or []
    qc["ignore"] = skipped
    cfg_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] 已修正 quantization_config.ignore：{len(old)} -> {len(skipped)} 条"
          f"（改用模型内部模块名，避免回读时把被 ignore 的模块误判为量化）")


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    args = _parse_args()

    # torchrun + CPU offload 依赖 libshm 的 torch_shm_manager，先补权限再加载
    _ensure_torch_shm_manager()

    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    torch.set_grad_enabled(False)

    config = _load_config(args.model)
    auto_cls, reason = _resolve_auto_class(config)
    _check_auto_class_supported(auto_cls, config)
    print(f"[INFO] loader={auto_cls.__name__} ({reason})")

    ignore = (
        [x.strip() for x in args.ignore.split(",") if x.strip()]
        if args.ignore else list(DEFAULT_IGNORE)
    )

    if args.dry_run:
        _dry_run(config, auto_cls, ignore)
        return

    # llmcompressor 只在正式量化时导入，保证 --dry-run 在轻量环境也能跑
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    model, rank, use_distributed = _load_model(args, config, auto_cls)
    tokenizer = _load_tokenizer(args.model)

    # W8A8 / FP8_DYNAMIC 都是“权重 per-channel 静态 + 激活 per-token dynamic”，
    # 不需要校准数据：llmcompressor 会据此推断出 DataFreePipeline，故不传 dataset。
    recipe = QuantizationModifier(targets=args.targets, scheme=args.scheme, ignore=ignore)
    print(f"[INFO] recipe: QuantizationModifier(targets={args.targets!r}, scheme={args.scheme!r})"
          f"（免校准数据）")

    sequential_targets = None
    if args.sequential_targets:
        sequential_targets = [x.strip() for x in args.sequential_targets.split(",") if x.strip()]

    oneshot(
        model=model,
        recipe=recipe,
        trust_remote_code_model=True,
        pipeline=args.pipeline,
        sequential_targets=sequential_targets,
    )

    if use_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        if rank != 0:
            return
        torch.distributed.init_process_group(
            backend="gloo", world_size=1, rank=0, timeout=timedelta(seconds=3600),
        )

    save_dir = args.save_dir or f"{args.model}-{args.scheme}"
    os.makedirs(save_dir, exist_ok=True)
    model.requires_grad_(False)
    model.save_pretrained(save_dir, save_compressed=True)
    _rewrite_saved_ignore(save_dir, model)
    tokenizer.save_pretrained(save_dir)
    _copy_non_weight_files(Path(args.model), Path(save_dir))

    print(f"[INFO] 量化完成，产物已保存到 {save_dir}")

    if use_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
