"""
Kimi K3 量化精度对比测试

对比两个量化格式的 mini 模型在相同输入下的输出差异（cosine similarity / 相对误差），
用于定位精度下降的具体层段。

默认对比指标: hidden_states[-1]（最后一层隐状态，cos 更能反映真实量化误差）
默认输入来源: tokenizer 编码真实 prompt（避免随机 id 触发 MoE 极端路由）

可选调试项：
  - --num-cases N: 只跑 _REAL_PROMPTS 的前 N 条测试
  - --dump-per-layer: 逐层打印 hidden state cosine
  - --dump-weight-stats: 直接从 checkpoint 读取关键权重并打印统计/cos
  - --dump-submodule: hook 指定层内部子模块输出并打印 cosine
  - --log-routing: 打印每层 MoE gate 路由重叠率

"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── ANSI 颜色 ───
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def _is_decoder_layer_module(mod: torch.nn.Module) -> bool:
    """匹配 Kimi/Qwen/Llama 等常见 DecoderLayer，供逐层 hidden state hook 使用。"""
    cls = type(mod).__name__
    return cls.endswith("DecoderLayer") or cls in {
        "KimiDecoderLayer",
        "Qwen3_5MoeDecoderLayer",
        "Qwen3MoeDecoderLayer",
    }


def _is_moe_gate_module(name: str, mod: torch.nn.Module) -> bool:
    """匹配 Kimi/Qwen MoE router/gate。"""
    cls = type(mod).__name__.lower()
    lname = name.lower()
    return (
        type(mod).__name__ == "KimiMoEGate"
        or lname.endswith(".mlp.gate")
        or "router" in lname
        or "gate" in cls
        or "router" in cls
    )


def _infer_gate_topk(module: torch.nn.Module, logits: torch.Tensor, top_k: int | None = None) -> int:
    if isinstance(top_k, int) and top_k > 0:
        return min(top_k, logits.shape[-1])
    for attr in ("top_k", "topk", "num_experts_per_tok", "n_routed_experts"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return min(value, logits.shape[-1])
    return min(2, logits.shape[-1])


def _infer_model_topk(model: torch.nn.Module) -> int | None:
    """从模型 config 推断 MoE top-k；Qwen 的 gate 常是 Linear，本身没有 top-k 属性。"""
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is None:
            continue
        for attr in ("num_experts_per_tok", "moe_top_k", "top_k", "topk"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


def _routing_pair_to_cpu(indices: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """把 routing tensor 归一成 CPU 上的 [n_tokens, top_k]。"""
    idx = indices.detach().cpu()
    w = weights.detach().cpu()
    if idx.dim() > 2:
        idx = idx.reshape(-1, idx.shape[-1])
        w = w.reshape(-1, w.shape[-1])
    return idx, w


def _normalize_routing_output(module: torch.nn.Module, out, top_k: int | None = None):
    """把不同模型 gate/router 输出尽量归一成 (topk_idx, topk_weight)。"""
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        first, second = out[0], out[1]
        if torch.is_tensor(first) and torch.is_tensor(second):
            # Kimi: (topk_idx, topk_weight)
            if first.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                return _routing_pair_to_cpu(first, second)
            # 一些 router 可能返回 (weights, indices)
            if second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                return _routing_pair_to_cpu(second, first)
    if torch.is_tensor(out) and out.dim() >= 2:
        # Qwen gate 常见输出为 router logits；本地 topk 用于路由重叠粗略分析。
        k = _infer_gate_topk(module, out, top_k)
        weights, indices = torch.topk(out.detach().float(), k=k, dim=-1)
        return _routing_pair_to_cpu(indices, weights)
    return None


def _build_max_memory(max_memory_per_gpu: str | None):
    if max_memory_per_gpu is None or not torch.cuda.is_available():
        return None
    return {i: max_memory_per_gpu for i in range(torch.cuda.device_count())}


def _fix_shape_mismatch_from_ckpt(model: torch.nn.Module, model_path: Path):
    """加载后修正 shape mismatch 参数：从 checkpoint 中 slice 前 N 个元素覆盖随机初始化值."""
    import glob
    from safetensors import safe_open

    shard_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not shard_files:
        return

    param_dict = dict(model.named_parameters())
    fixed = 0
    for shard_file in shard_files:
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in param_dict:
                    continue
                param = param_dict[key]
                ckpt_t = f.get_tensor(key)
                if ckpt_t.shape == param.shape:
                    continue
                # 仅当 checkpoint 每维都 >= model 时才 slice（截头取用）
                if all(c >= m for c, m in zip(ckpt_t.shape, param.shape)):
                    slices = tuple(slice(0, s) for s in param.shape)
                    param.data.copy_(ckpt_t[slices].to(dtype=param.dtype, device=param.device))
                    print(f"  [fix_shape] {key}: ckpt {tuple(ckpt_t.shape)} → model {tuple(param.shape)}")
                    fixed += 1
    if fixed:
        print(f"  [fix_shape] 共修正 {fixed} 个 shape mismatch 参数")


def _find_last_layer_device(device_map: dict):
    """返回 device_map 中最后一个 `layers.<n>` 所在设备（用于模型尾部模块对齐执行设备）。"""
    import re
    last_idx = -1
    last_device = 0
    for key, device in device_map.items():
        match = re.search(r"(^|\.)layers\.(\d+)($|\.)", key)
        if match:
            idx = int(match.group(2))
            if idx > last_idx:
                last_idx = idx
                last_device = device
    return last_device


def _apply_module_device_overrides(device_map: dict, overrides: dict[str, str | int]):
    """在 dispatch 前按显式映射覆盖模块设备；设备值支持特殊值 `last-layer`。"""
    for module_name, placement in overrides.items():
        if placement == "last-layer":
            placement = _find_last_layer_device(device_map)
        device_map[module_name] = placement
    if overrides:
        print(f"  [offload] 模块设备覆盖: {overrides}")


def _coalesce_no_split_modules(device_map: dict, model: torch.nn.Module, no_split_classes: list[str]):
    """将 no-split 类模块（如 DecoderLayer）合并为单个整层 device_map 条目。

    若只对齐子模块条目，dispatch_model 的 hook 会挂在子模块上，层主模块的 forward
    输入不会被移动到层设备，层内直接访问权重的自定义算子仍会跨设备。因此这里删除
    该层所有子条目，只保留一个 `层名 -> 设备` 条目。
    """
    if not no_split_classes:
        return
    coalesced = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Module) or module.__class__.__name__ not in no_split_classes:
            continue
        prefix = name + "."
        child_entries = {k: v for k, v in device_map.items() if k.startswith(prefix)}
        if not child_entries:
            continue
        if name in device_map:
            target = device_map[name]
        else:
            import collections
            target = collections.Counter(child_entries.values()).most_common(1)[0][0]
        for child_name in child_entries:
            del device_map[child_name]
            coalesced += 1
        device_map[name] = target
    if coalesced:
        print(f"  [offload] 将 {coalesced} 个子模块条目合并为整层条目 (no-split 层 {len(device_map)} 个条目)")



def load_model(
    model_path: Path,
    device: str,
    max_memory_per_gpu: str | None = None,
    cpu_convert_then_dispatch: bool = False,
    module_device_overrides: dict[str, str | int] | None = None,
):
    """加载 mini 模型；支持 device_map=auto 多卡分配

    大 mini (含 W4A16/MXFP4 quantization_config 且 checkpoint > 50GB) 解压后可能到
    数百 GB, 单卡装不下。当 device="auto" 时采用两阶段策略:
      1. 不使用 device_map，先把 packed checkpoint 完整加载到 CPU，避免 accelerate
         按 packed tensor 大小低估显存并在加载阶段 OOM
      2. 在 CPU 上主动解压为 dense BF16，再按真实 tensor 大小 dispatch 到多 GPU + CPU
      - 禁用 transformers `caching_allocator_warmup`，避免不可靠的预热估算导致 OOM
      - CPU RAM 需要同时容纳 packed/dense 过渡期间的峰值；当前机器应有约 2TB RAM
    """
    import json as _json
    print(f"  加载: {model_path}")
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        model = model.to("cpu")
    else:
        # 检查 checkpoint 大小 + 是否量化过, 决定是否用 CPU offload 策略
        cfg_path = Path(model_path) / "config.json"
        is_quantized = False
        if cfg_path.exists():
            with open(cfg_path) as _f:
                _cfg = _json.load(_f)
            is_quantized = "quantization_config" in _cfg or any(
                isinstance(v, dict) and "quantization_config" in v
                for v in _cfg.values()
            )

        ckpt_size_gb = sum(
            f.stat().st_size for f in Path(model_path).glob("*.safetensors")
        ) / 1e9

        # 指定模块设备覆盖时，必须由本脚本执行 dispatch，才能将覆盖写入 device map。
        need_offload = cpu_convert_then_dispatch or bool(module_device_overrides)
        max_memory = _build_max_memory(max_memory_per_gpu)
        if need_offload and device != "cpu":
            mode_reason = "指定模块设备覆盖" if module_device_overrides else "CPU 转换"
            print(f"  [offload] 量化 checkpoint {ckpt_size_gb:.1f}GB, "
                  f"启用 CPU 加载 → 主动解压 → GPU/CPU dispatch ({mode_reason})")

            # ── Monkey-patch caching_allocator_warmup ──
            # transformers 加载/dispatch 大模型时可能按估算大小预热 CUDA cache。
            # 对 packed quantized 权重估算不可靠，预热本身可能 OOM；禁用更稳。
            import transformers.modeling_utils as _tmu2
            if not getattr(_tmu2, "_warmup_patched", False):
                _tmu2.caching_allocator_warmup = lambda *a, **kw: None
                _tmu2._warmup_patched = True
                print("  [offload] 已禁用 caching_allocator_warmup")

            # ── 最终 dispatch 的真实显存预算 ──
            # 之前用每卡 8GiB 是为了逼 accelerate 不把 packed W4A16 过量塞进 GPU，
            # 但在主动 CPU 解压成 dense BF16 后，module_sizes 已经准确，再继续用
            # 8GiB 会导致几乎全模型驻 CPU，推理极慢。这里按真实显存留 10%/16GiB
            # 余量，H20 144GB 上约 125GiB/card。
            n_gpu = torch.cuda.device_count()
            if max_memory is not None:
                dispatch_max_memory = dict(max_memory)
                dispatch_max_memory.setdefault("cpu", "1800GiB")
            else:
                dispatch_max_memory = {}
                for i in range(n_gpu):
                    total_gib = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    budget_gib = int(max(1, min(total_gib * 0.90, total_gib - 16)))
                    dispatch_max_memory[i] = f"{budget_gib}GiB"
                dispatch_max_memory["cpu"] = "1800GiB"
            print(f"  [offload] dispatch max_memory: {dispatch_max_memory}")

            # ── 先只加载到 CPU ──
            # 不在初始 from_pretrained 里使用 device_map=auto，避免 accelerate 对 packed
            # 权重低估大小后产生 GPU OOM，也避免 ct_decompress_hook 看到 meta tensor。
            print("  [offload] step 1: 加载 checkpoint 到 CPU...")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                ignore_mismatched_sizes=True,
            )

            # ── 如有 compressed-tensors hook，在 CPU 上主动 decompress ──
            if hasattr(model, "ct_decompress_hook"):
                print("  [offload] step 2: CPU 上主动 decompress W4A16 packed → BF16...")
                from compressed_tensors.compressors.model_compressors.model_compressor import (
                    ModelCompressor,
                )
                compressor = ModelCompressor.from_pretrained_model(model)
                if compressor is not None:
                    compressor.decompress_model(model)
                    compressor.remove_decompression_hook(model)
                    print("  [offload]   decompress 完成, 权重现在是 dense BF16")
                else:
                    print("  [offload]   无需 decompress")
            else:
                print("  [offload] step 2: 未发现 ct_decompress_hook，跳过 decompress")

            # ── 再按真实 dense 权重大小 dispatch 到多 GPU + CPU ──
            print("  [offload] step 3: dispatch dense/CPU 权重到 GPU + CPU...")
            from accelerate import dispatch_model
            from accelerate.utils import infer_auto_device_map
            # infer_auto_device_map 默认不读取 model._no_split_modules，必须显式传入，
            # 且必须为 list/tuple：accelerate 对非 list/tuple 会整体包成单元素列表，
            # 传入 set 会导致类名匹配失败、no-split 不生效。
            no_split_classes = (
                getattr(model, "_no_split_modules", None)
                or getattr(model.__class__, "_no_split_modules", None)
            )
            no_split_classes = list(no_split_classes) if no_split_classes else []
            print(f"  [offload] no_split_module_classes: {no_split_classes}")
            _device_map = infer_auto_device_map(
                model,
                max_memory=dispatch_max_memory,
                dtype=torch.bfloat16,
                no_split_module_classes=no_split_classes,
            )
            _coalesce_no_split_modules(_device_map, model, no_split_classes or [])
            _apply_module_device_overrides(_device_map, module_device_overrides or {})
            dispatch_model(
                model,
                device_map=_device_map,
                offload_buffers=False,
            )
            print(f"  [offload] dispatch 完成 ({len(_device_map)} 个 module)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device,
                max_memory=max_memory,
                ignore_mismatched_sizes=True,
            )
    model.eval()
    _fix_shape_mismatch_from_ckpt(model, model_path)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count / 1e9:.2f}B")
    return model


# ─── 真实 prompt (改动 1: 避免 random id 触发 MoE 极端路由 & 量化误差放大) ───
# 覆盖中英文 / 短长 / 代码等常见分布, 每条 prompt 直接作为输入, 不 repeat/padding/截断
_REAL_PROMPTS = [
    "The capital of France is",
    "The capital of France is Paris, would you agree?",
    "中国的首都是",
    "秦皇汉武，唐宋明清",
    "中国地大物博，人口众多，文化丰富，具有丰富的历史和文化，秦皇汉武，唐宋明清。它的首都是",
    "The capital of France is Paris, there are many cities in France, such as Marseille, Nice, and Lyon, would you agree?",
    "The Kimi-K3 model uses MoE architecture with 896 experts, each activated conditionally based on the routing gate, and each expert has 128 parameters, with a total of 114.5B parameters.",
    
    "Large language models have transformed natural language processing. "
    "Modern architectures combine self-attention with mixture-of-experts layers "
    "to scale parameters efficiently. Quantization techniques such as W4A16, "
    "W8A8, and FP8 further reduce memory footprint while preserving accuracy. "
    "Kimi-K3, developed by Moonshot AI, is a state-of-the-art open-source model "
    "that leverages these techniques to deliver high performance at lower cost.",
    # "The model supports long context lengths up to one million tokens, "
    # "enabling applications like document analysis and code understanding.",
]


def build_prompt_cases(
    tokenizer,
    device: torch.device,
    use_random: bool,
    vocab_size: int,
    num_cases: int | None = None,
) -> list[tuple[torch.Tensor, str]]:
    """直接用 _REAL_PROMPTS 构造推理输入, 每条 prompt 一个 case (bs=1).

    不做任何 repeat / padding / 截断: input_ids 就是 tokenize 后的原始序列.
    use_random=True 或 tokenizer 缺失时, 按每条 prompt 的编码长度生成随机 id.
    num_cases 指定时，只取 _REAL_PROMPTS 的前 N 条。
    """
    if num_cases is not None and num_cases < 1:
        raise ValueError("--num-cases 必须大于 0")

    prompts = _REAL_PROMPTS if num_cases is None else _REAL_PROMPTS[:num_cases]
    cases = []
    for i, prompt in enumerate(prompts):
        if use_random or tokenizer is None:
            enc_len = (
                len(tokenizer(prompt, add_special_tokens=False).input_ids)
                if tokenizer is not None
                else 32
            )
            input_ids = torch.randint(0, vocab_size, (1, enc_len), device=device)
        else:
            input_ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
        desc = f"prompt[{i}] len={input_ids.shape[1]}: {prompt}"
        cases.append((input_ids, desc))
    return cases


def _get_compare_tensor(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    compare_logits: bool,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """获取用于对比的张量.

    inputs_embeds 不为 None 时直接作为模型输入（级联模式），忽略 input_ids.

    优先级:
      1. compare_logits=True → logits
      2. output_hidden_states=True 且模型支持 → hidden_states[-1]
      3. 模型不支持 output_hidden_states (返回 None) → lm_head forward-hook 捕获输入
    """
    dev = next(model.parameters()).device

    def _call(extra_kw=None):
        kw = {"use_cache": False}
        if extra_kw:
            kw.update(extra_kw)
        if inputs_embeds is not None:
            return model(inputs_embeds=inputs_embeds.to(dev), **kw)
        return model(input_ids, **kw)

    if compare_logits:
        return _call().logits.float().cpu()

    result = _call({"output_hidden_states": True})
    if result.hidden_states is not None:
        return result.hidden_states[-1].float().cpu()

    # Kimi-K3 自定义 forward 不支持 output_hidden_states → 用 hook 捕获 lm_head 输入
    print("  [compare] output_hidden_states 不可用, 改用 lm_head hook 捕获 hidden state")
    lm_head = None
    for name, mod in model.named_modules():
        if "lm_head" in name:
            lm_head = mod
            print(f"  [compare] 找到 lm_head: {name}")
            break
    if lm_head is None:
        raise RuntimeError(
            "output_hidden_states 不可用且找不到 lm_head, 请加 --compare-logits"
        )
    captured: dict = {}

    def _hook(module, inp, out):
        captured["hs"] = inp[0].detach()

    hook = lm_head.register_forward_hook(_hook)
    try:
        _call()
    finally:
        hook.remove()

    if "hs" not in captured:
        raise RuntimeError(
            "lm_head hook 未捕获到 hidden state, 请加 --compare-logits"
        )
    return captured["hs"].float().cpu()


def _collect_routing(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor | None = None,
) -> dict:
    """Hook 所有 MoE gate/router，收集每层的 topk_idx + topk_weight.

    返回: {layer_name: (topk_idx, topk_weight)}  均为 cpu tensor, shape [n_tokens, top_k]
    """
    hooks = []
    routing: dict = {}
    dev = next(model.parameters()).device

    gate_topk = _infer_model_topk(model)
    for name, mod in model.named_modules():
        if _is_moe_gate_module(name, mod):
            def _make_hook(n):
                def _hook(module, inp, out):
                    normalized = _normalize_routing_output(module, out, gate_topk)
                    if normalized is not None:
                        routing[n] = normalized
                return _hook
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        if inputs_embeds is not None:
            model(inputs_embeds=inputs_embeds.to(dev), use_cache=False)
        else:
            model(input_ids, use_cache=False)

    for h in hooks:
        h.remove()
    return routing


def _apply_fixed_routing(model: torch.nn.Module, fixed_routing: dict) -> list:
    """给 model_b 的 MoE gate/router 注册 hook，尽量复用 model_a 的路由。

    Kimi gate 通常返回 (topk_idx, topk_weight)，可完整覆盖；Qwen gate 常见返回
    router logits，只能通过构造稀疏 logits 强制 top-k expert，权重仅近似。
    返回 hook 列表，调用方负责 remove().
    """
    hooks = []
    for name, mod in model.named_modules():
        if not _is_moe_gate_module(name, mod):
            continue
        if name not in fixed_routing:
            continue

        def _make_hook(fixed_idx, fixed_w):
            def _hook(module, inp, out):
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    first, second = out[0], out[1]
                    if torch.is_tensor(first) and torch.is_tensor(second):
                        idx = fixed_idx.to(first.device)
                        w = fixed_w.to(device=second.device, dtype=second.dtype)
                        replaced = list(out)
                        if first.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                            replaced[0], replaced[1] = idx, w
                        elif second.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                            replaced[0], replaced[1] = w.to(dtype=first.dtype), idx.to(second.device)
                        return tuple(replaced) if isinstance(out, tuple) else replaced

                if torch.is_tensor(out) and out.dim() >= 2:
                    # Qwen gate 若输出 logits，则用稀疏 logits 强制 top-k expert。
                    idx = fixed_idx.to(out.device)
                    values = fixed_w.to(device=out.device, dtype=out.dtype)
                    if idx.shape[:-1] != out.shape[:-1]:
                        idx = idx.reshape(*out.shape[:-1], idx.shape[-1])
                        values = values.reshape(*out.shape[:-1], values.shape[-1])
                    forced = torch.full_like(out, torch.finfo(out.dtype).min)
                    forced.scatter_(-1, idx.long(), values)
                    return forced

                return out
            return _hook

        fixed_idx, fixed_w = fixed_routing[name]
        hooks.append(mod.register_forward_hook(_make_hook(fixed_idx, fixed_w)))
    return hooks


def _get_compare_tensor_fixed_routing(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    input_ids: torch.Tensor,
    compare_logits: bool,
    inputs_embeds: torch.Tensor | None = None,
):
    """先跑 model_a 收集路由，再以固定路由跑 model_b，返回 (out_a, out_b, routing_a)."""
    hooks_collect = []
    routing_a: dict = {}
    dev_a = next(model_a.parameters()).device
    gate_topk = _infer_model_topk(model_a)
    for name, mod in model_a.named_modules():
        if _is_moe_gate_module(name, mod):
            def _make_c(n):
                def _h(module, inp, out):
                    normalized = _normalize_routing_output(module, out, gate_topk)
                    if normalized is not None:
                        routing_a[n] = normalized
                return _h
            hooks_collect.append(mod.register_forward_hook(_make_c(name)))

    with torch.no_grad():
        out_a = _get_compare_tensor(model_a, input_ids, compare_logits, inputs_embeds)

    for h in hooks_collect:
        h.remove()

    hooks_fix = _apply_fixed_routing(model_b, routing_a)
    with torch.no_grad():
        out_b = _get_compare_tensor(model_b, input_ids, compare_logits, inputs_embeds)
    for h in hooks_fix:
        h.remove()

    return out_a, out_b, routing_a


def _dump_per_layer_cos(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    input_ids: torch.Tensor,
    fix_routing: bool,
    inputs_embeds: torch.Tensor | None = None,
):
    """逐层收集两模型的 hidden state，打印每层 cosine similarity.

    hook 点: DecoderLayer 的 forward 输出第 0 个元素（hidden_states）.
    fix_routing=True 时 model_b 沿用 model_a 的路由.
    inputs_embeds 不为 None 时用于级联模式，替代 input_ids 作为输入.
    """
    def _collect_layer_hs(
        model: torch.nn.Module,
        emb: torch.Tensor | None = None,
    ) -> dict:
        dev = next(model.parameters()).device
        hooks = []
        layer_hs: dict = {}
        for name, mod in model.named_modules():
            if _is_decoder_layer_module(mod):
                def _make_hook(n):
                    def _h(module, inp, out):
                        hs = out[0] if isinstance(out, tuple) else out
                        layer_hs[n] = hs.detach().float().cpu()
                    return _h
                hooks.append(mod.register_forward_hook(_make_hook(name)))
        with torch.no_grad():
            if emb is not None:
                model(inputs_embeds=emb.to(dev), use_cache=False)
            else:
                model(input_ids, use_cache=False)
        for h in hooks:
            h.remove()
        return layer_hs

    # model_a 正常跑，收集路由 + 逐层 hidden state
    hooks_collect = []
    routing_a: dict = {}
    gate_topk = _infer_model_topk(model_a)
    for name, mod in model_a.named_modules():
        if _is_moe_gate_module(name, mod):
            def _make_c(n):
                def _h(module, inp, out):
                    normalized = _normalize_routing_output(module, out, gate_topk)
                    if normalized is not None:
                        routing_a[n] = normalized
                return _h
            hooks_collect.append(mod.register_forward_hook(_make_c(name)))
    hs_a = _collect_layer_hs(model_a, inputs_embeds)
    for h in hooks_collect:
        h.remove()

    # model_b 跑（可选固定路由），收集逐层 hidden state
    hooks_fix = _apply_fixed_routing(model_b, routing_a) if fix_routing else []
    hs_b = _collect_layer_hs(model_b, inputs_embeds)
    for h in hooks_fix:
        h.remove()

    # 逐层打印 cos
    keys = sorted(hs_a.keys())
    fix_tag = " [fix_routing]" if fix_routing else ""
    print(f"  [per_layer_cos]{fix_tag} 共 {len(keys)} 层:")
    for k in keys:
        ha = hs_a[k]
        hb = hs_b.get(k)
        if hb is None:
            print(f"    {k}: model_b 无此层")
            continue
        cos = torch.nn.functional.cosine_similarity(
            ha.flatten().unsqueeze(0),
            hb.flatten().unsqueeze(0),
        ).item()
        marker = "" if cos >= 0.95 else "  ⚠"
        short = ".".join(k.split(".")[-2:])
        print(f"    {short}: cos={cos:.6f}{marker}")



def _parse_debug_experts(value: str | None, default_count: int) -> list[int]:
    """解析 --debug-experts；未指定时默认取前 default_count 个 expert。"""
    if value is None or not value.strip():
        return list(range(max(0, default_count)))
    experts = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_s, end_s = part.split(":", 1)
            experts.extend(range(int(start_s), int(end_s)))
        else:
            experts.append(int(part))
    return sorted(set(experts))


def _checkpoint_weight_map(model_path: Path) -> dict[str, str]:
    """读取 safetensors index；没有 index 时扫描目录内 safetensors。"""
    import json as _json
    from safetensors import safe_open

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            index = _json.load(f)
        weight_map = index.get("weight_map", {})
        if isinstance(weight_map, dict) and weight_map:
            return weight_map

    weight_map: dict[str, str] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = shard.name
    return weight_map


def _load_checkpoint_tensor(model_path: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    from safetensors import safe_open

    shard = weight_map[key]
    with safe_open(str(model_path / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _debug_weight_selected(key: str, layer_idx: int, experts: set[int]) -> bool:
    prefix = f"model.layers.{layer_idx}."
    if not key.startswith(prefix):
        return False
    suffix = key[len(prefix):]
    if suffix.startswith("linear_attn."):
        return key.endswith((".weight", ".bias")) or ".weight" in key
    if suffix.startswith("self_attn."):
        return key.endswith((".weight", ".bias")) or ".weight" in key
    if suffix.startswith("mlp.gate."):
        return True
    if suffix.startswith("mlp.shared_expert.") or suffix.startswith("mlp.shared_expert_gate."):
        return key.endswith((".weight", ".bias")) or ".weight" in key
    expert_match = __import__("re").match(r"mlp\.experts\.(\d+)\.", suffix)
    if expert_match and int(expert_match.group(1)) in experts:
        return key.endswith((".weight", ".bias")) or ".weight" in key
    return False


def _tensor_summary(t: torch.Tensor) -> str:
    x = t.detach().float()
    finite = torch.isfinite(x)
    finite_ratio = finite.float().mean().item() if x.numel() else 1.0
    if finite.any():
        xf = x[finite]
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"mean={xf.mean().item():+.3e} std={xf.std(unbiased=False).item():.3e} "
            f"maxabs={xf.abs().max().item():.3e} norm={torch.linalg.vector_norm(xf).item():.3e} "
            f"finite={finite_ratio:.3f}"
        )
    return f"shape={tuple(t.shape)} dtype={t.dtype} all_nonfinite"


def _compare_tensor_summary(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    if af.numel() == 0 or bf.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    cos = torch.nn.functional.cosine_similarity(af.unsqueeze(0), bf.unsqueeze(0)).item()
    rel = ((af - bf).abs() / (af.abs() + 1e-8)).mean().item()
    max_diff = (af - bf).abs().max().item()
    return cos, rel, max_diff


def dump_checkpoint_weight_stats(
    model_a_path: Path,
    model_b_path: Path,
    layer_idx: int,
    experts: list[int],
    max_tensors: int,
) -> None:
    """直接对比两个 checkpoint 中指定层的关键权重，用于定位 W4→BF16 反量化是否已异常。"""
    weight_map_a = _checkpoint_weight_map(model_a_path)
    weight_map_b = _checkpoint_weight_map(model_b_path)
    expert_set = set(experts)
    keys_a = {k for k in weight_map_a if _debug_weight_selected(k, layer_idx, expert_set)}
    keys_b = {k for k in weight_map_b if _debug_weight_selected(k, layer_idx, expert_set)}
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print("\n─── 调试: checkpoint 权重统计 ───")
    print(f"  layer={layer_idx} experts={experts} common={len(common)} only_a={len(only_a)} only_b={len(only_b)}")
    if only_a[:5]:
        print(f"  only model_a 示例: {only_a[:5]}")
    if only_b[:5]:
        print(f"  only model_b 示例: {only_b[:5]}")
    if not common:
        print("  ⚠ 未找到可对比权重；请检查 --debug-layer / --debug-experts 或 checkpoint 命名。")
        return

    for key in common[:max_tensors]:
        a = _load_checkpoint_tensor(model_a_path, weight_map_a, key)
        b = _load_checkpoint_tensor(model_b_path, weight_map_b, key)
        print(f"\n  [weight] {key}")
        print(f"    A: {_tensor_summary(a)}")
        print(f"    B: {_tensor_summary(b)}")
        if a.shape != b.shape:
            print(f"    ❌ shape mismatch: A={tuple(a.shape)} B={tuple(b.shape)}")
            continue
        cos, rel, max_diff = _compare_tensor_summary(a, b)
        marker = "" if cos >= 0.95 else "  ⚠"
        print(f"    cmp: cos={cos:.6f}{marker} rel_mean={rel:.3e} max_diff={max_diff:.3e}")
    if len(common) > max_tensors:
        print(f"  ... 还有 {len(common) - max_tensors} 个 tensor 未打印，可增大 --debug-max-tensors")


def _first_tensor(obj):
    """从 module output 中取第一个 tensor，兼容 tuple/list/dict。"""
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, (tuple, list)):
        for item in obj:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for item in obj.values():
            t = _first_tensor(item)
            if t is not None:
                return t
    return None


def _is_debug_submodule(name: str, layer_idx: int, experts: set[int]) -> bool:
    prefix = f"model.layers.{layer_idx}."
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix):]
    if suffix == "linear_attn" or suffix.startswith("linear_attn."):
        return True
    if suffix in {"mlp", "mlp.gate", "mlp.experts", "mlp.shared_expert", "mlp.shared_expert_gate"}:
        return True
    if suffix.startswith("mlp.shared_expert."):
        return True
    expert_match = __import__("re").match(r"mlp\.experts\.(\d+)(?:\.|$)", suffix)
    if expert_match and int(expert_match.group(1)) in experts:
        return True
    return False


def _register_submodule_debug_hooks(
    model: torch.nn.Module,
    layer_idx: int,
    experts: list[int],
    captured: dict,
) -> list:
    hooks = []
    expert_set = set(experts)
    for name, mod in model.named_modules():
        if not _is_debug_submodule(name, layer_idx, expert_set):
            continue

        def _make_hook(n):
            def _h(module, inp, out):
                t = _first_tensor(out)
                if t is not None:
                    captured[n] = t.detach().float().cpu()
            return _h

        hooks.append(mod.register_forward_hook(_make_hook(name)))
    return hooks


def _print_submodule_cos(sub_a: dict, sub_b: dict) -> None:
    keys = sorted(sub_a.keys())
    if not keys:
        print("  [submodule_cos] 未捕获到指定子模块输出")
        return
    print(f"  [submodule_cos] 共 {len(keys)} 个子模块:")
    for key in keys:
        a = sub_a[key]
        b = sub_b.get(key)
        short = ".".join(key.split(".")[-4:])
        if b is None:
            print(f"    {short}: model_b 无此输出")
            continue
        if a.shape != b.shape:
            print(f"    {short}: shape mismatch A={tuple(a.shape)} B={tuple(b.shape)}")
            continue
        cos, rel, max_diff = _compare_tensor_summary(a, b)
        marker = "" if cos >= 0.95 else "  ⚠"
        print(f"    {short}: cos={cos:.6f}{marker} rel_mean={rel:.3e} max_diff={max_diff:.3e}")



def _run_and_collect(
    model: torch.nn.Module,
    tokenizer,
    vocab_size: int,
    use_random: bool,
    compare_logits: bool,
    dump_per_layer: bool,
    log_routing: bool = False,
    dump_submodule: bool = False,
    debug_layer: int = 0,
    debug_experts: list[int] | None = None,
    num_cases: int | None = None,
) -> list[dict]:
    """跑一个 model, 收集每个 case 的最终输出 + 逐层 / 子模块 hidden state.

    case 输入直接来自 _REAL_PROMPTS (bs=1, 不 repeat/padding/截断).
    返回: [{desc, input_ids, out, layer_hs, routing, submodule_hs}, ...]
      - out: shape 与 _get_compare_tensor 返回一致, CPU float
      - layer_hs: {layer_name: cpu_tensor} 或 None (dump_per_layer=False 时)
      - routing: {gate_name: (topk_idx, topk_weight)} 或 None (log_routing=False 时)
      - submodule_hs: {module_name: cpu_tensor} 或 None (dump_submodule=False 时)
    """
    exec_device = next(model.parameters()).device
    records = []
    debug_experts = debug_experts or []

    cases = build_prompt_cases(tokenizer, exec_device, use_random, vocab_size, num_cases=num_cases)
    for input_ids, desc in cases:
        print(f"\n[{desc}]")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")

        layer_hs = None
        submodule_hs = None
        if dump_per_layer or dump_submodule:
            hs_dict: dict = {}
            sub_dict: dict = {}
            hooks = []
            if dump_per_layer:
                # 挂 hook 收集每层 DecoderLayer 输出, 与 _dump_per_layer_cos 里
                # _collect_layer_hs 完全一致
                for name, mod in model.named_modules():
                    if _is_decoder_layer_module(mod):
                        def _make_hook(n):
                            def _h(module, inp, out):
                                hs = out[0] if isinstance(out, tuple) else out
                                hs_dict[n] = hs.detach().float().cpu()
                            return _h
                        hooks.append(mod.register_forward_hook(_make_hook(name)))
            if dump_submodule:
                hooks.extend(_register_submodule_debug_hooks(model, debug_layer, debug_experts, sub_dict))
            try:
                with torch.no_grad():
                    out = _get_compare_tensor(model, input_ids, compare_logits)
            finally:
                for h in hooks:
                    h.remove()
            layer_hs = hs_dict if dump_per_layer else None
            submodule_hs = sub_dict if dump_submodule else None
        else:
            with torch.no_grad():
                out = _get_compare_tensor(model, input_ids, compare_logits)

        routing = _collect_routing(model, input_ids) if log_routing else None

        # input_ids 转到 CPU 保存, 避免模型释放后引用 GPU tensor 悬空
        records.append({
            "desc": desc,
            "input_ids": input_ids.cpu(),
            "out": out,
            "layer_hs": layer_hs,
            "routing": routing,
            "submodule_hs": submodule_hs,
        })

    return records


def _compare_collected(
    records_a: list[dict],
    records_b: list[dict],
    threshold: float,
    print_logits: bool,
    compare_logits: bool,
    log_routing: bool = False,
):
    """对比两组已收集的输出记录。"""
    target_name = "logits" if compare_logits else "hidden_states[-1]"
    print(f"\n对比目标: {target_name}")

    for rec_a, rec_b in zip(records_a, records_b):
        assert rec_a["desc"] == rec_b["desc"], "test case 顺序不一致"
        print(f"\n[{rec_a['desc']}]")
        print(f"  input_ids shape: {rec_a['input_ids'].shape}")

        if log_routing:
            routing_a = rec_a.get("routing")
            routing_b = rec_b.get("routing")
            if routing_a is not None and routing_b is not None:
                _log_routing_diff(routing_a, routing_b)

        out_a, out_b = rec_a["out"], rec_b["out"]

        # 逐层 cos (若两侧都有)
        if rec_a["layer_hs"] is not None and rec_b["layer_hs"] is not None:
            hs_a, hs_b = rec_a["layer_hs"], rec_b["layer_hs"]
            keys = sorted(hs_a.keys())
            print(f"  [per_layer_cos] 共 {len(keys)} 层:")
            for k in keys:
                ha = hs_a[k]
                hb = hs_b.get(k)
                if hb is None:
                    print(f"    {k}: model_b 无此层")
                    continue
                cos = torch.nn.functional.cosine_similarity(
                    ha.flatten().unsqueeze(0),
                    hb.flatten().unsqueeze(0),
                ).item()
                marker = "" if cos >= 0.95 else "  ⚠"
                short = ".".join(k.split(".")[-2:])
                print(f"    {short}: cos={cos:.6f}{marker}")

        # 子模块 cos (若两侧都有)
        if rec_a.get("submodule_hs") is not None and rec_b.get("submodule_hs") is not None:
            _print_submodule_cos(rec_a["submodule_hs"], rec_b["submodule_hs"])

        # NaN/Inf 检查
        for label, out in [("model_a", out_a), ("model_b", out_b)]:
            nan_count = torch.isnan(out).sum().item()
            inf_count = torch.isinf(out).sum().item()
            if nan_count or inf_count:
                print(f"  ❌ {label}: NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"  ✅ {label}: 无 NaN/Inf")

        if print_logits:
            print(f"\n  --- out_a (model_a): shape={out_a.shape}, sum={out_a.sum().item():.6f} ---")
            print(out_a)
            print(f"\n  --- out_b (model_b): shape={out_b.shape}, sum={out_b.sum().item():.6f} ---")
            print(out_b)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            out_a.flatten().unsqueeze(0),
            out_b.flatten().unsqueeze(0),
        ).item()
        if cos_sim >= threshold:
            print(colored(f"  Cosine similarity: {cos_sim:.6f}  ✅ >= {threshold}", GREEN))
        else:
            print(colored(f"  Cosine similarity: {cos_sim:.6f}  ❌ < {threshold}", RED))

        # 相对误差 + 最大差异
        rel_err = ((out_a - out_b).abs() / (out_a.abs() + 1e-8)).mean().item()
        print(f"  相对误差 mean:   {rel_err:.6e}")
        max_diff = (out_a - out_b).abs().max().item()
        print(f"  最大绝对差异:    {max_diff:.6e}")


def _wait_gpu_free(timeout: float = 10.0, idle_mb: float = 100.0):
    """轮询 CUDA 显存，最多等待 timeout 秒，所有卡占用均低于 idle_mb MB 时提前退出."""
    if not torch.cuda.is_available():
        return
    import time
    n = torch.cuda.device_count()
    deadline = time.monotonic() + timeout
    while True:
        used = [torch.cuda.memory_allocated(i) / 1024**2 for i in range(n)]
        max_used = max(used)
        print(f"  [gpu] " + "  ".join(f"gpu{i}={v:.0f}MB" for i, v in enumerate(used)))
        if max_used < idle_mb:
            print(f"  [gpu] 所有卡显存 < {idle_mb:.0f}MB，继续")
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"  [gpu] 等待超时 ({timeout:.0f}s)，最大占用 {max_used:.0f}MB，继续")
            break
        time.sleep(min(0.5, remaining))


def _release_model(model):
    """释放 CPU/GPU 资源。调用方必须在此调用后立即将自己的引用置 None."""
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _worker_run_and_collect(
    model_path: Path,
    device: str,
    tokenizer,
    vocab_size: int,
    use_random: bool,
    compare_logits: bool,
    dump_per_layer: bool,
    log_routing: bool,
    dump_submodule: bool,
    debug_layer: int,
    debug_experts: list[int],
    num_cases: int | None,
    max_memory_per_gpu: str | None,
    cpu_convert_then_dispatch: bool,
    module_device_overrides: dict[str, str | int],
    output_path: str,
    queue,  # multiprocessing.Queue; 只传字符串状态，不传 tensor
):
    """子进程入口: 加载模型、跑推理、把 records 写入文件后退出.

    注意不要通过 multiprocessing.Queue 直接传 torch.Tensor；PyTorch 会用
    resource_sharer 传 storage fd，子进程退出后父进程反序列化可能 EOF。
    写临时文件后再让父进程 torch.load，可以确保进程退出释放 CUDA context。
    """
    try:
        torch.manual_seed(42)
        model = load_model(
            model_path,
            device,
            max_memory_per_gpu=max_memory_per_gpu,
            cpu_convert_then_dispatch=cpu_convert_then_dispatch,
            module_device_overrides=module_device_overrides,
        )
        torch.manual_seed(42)
        records = _run_and_collect(
            model, tokenizer, vocab_size,
            use_random, compare_logits, dump_per_layer,
            log_routing=log_routing,
            dump_submodule=dump_submodule,
            debug_layer=debug_layer,
            debug_experts=debug_experts,
            num_cases=num_cases,
        )
        torch.save(records, output_path)
        queue.put(("ok", output_path))
    except Exception:
        import traceback
        queue.put(("err", traceback.format_exc()))


def _sequential_compare(
    model_a_path: Path,
    model_b_path: Path,
    device: str,
    tokenizer,
    vocab_size: int,
    threshold: float,
    print_logits: bool,
    use_random: bool,
    compare_logits: bool,
    dump_per_layer: bool,
    log_routing: bool,
    dump_submodule: bool = False,
    debug_layer: int = 0,
    debug_experts: list[int] | None = None,
    num_cases: int | None = None,
    max_memory_per_gpu: str | None = None,
    cpu_convert_then_dispatch: bool = False,
    module_device_overrides: dict[str, str | int] | None = None,
):
    """串行加载: 子进程跑 model_a → 进程退出释放 GPU → 子进程跑 model_b → 对比.

    每个模型在独立子进程中加载和推理，进程退出后 OS 强制回收 CUDA context，
    GPU 显存彻底释放。"""
    import multiprocessing as mp

    debug_experts = debug_experts or []
    module_device_overrides = module_device_overrides or {}

    def _run_in_subprocess(label: str, model_path: Path):
        import tempfile
        import os
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        fd, output_path = tempfile.mkstemp(prefix=f"compare_mini_{label}_", suffix=".pt")
        os.close(fd)
        p = ctx.Process(
            target=_worker_run_and_collect,
            args=(model_path, device, tokenizer, vocab_size,
                  use_random, compare_logits, dump_per_layer, log_routing,
                  dump_submodule, debug_layer, debug_experts, num_cases,
                  max_memory_per_gpu, cpu_convert_then_dispatch,
                  module_device_overrides, output_path, q),
            daemon=False,
        )
        print(f"\n─── 阶段: 加载 {label} 并收集 (子进程) ───")
        p.start()
        # Queue 只传字符串，不传 tensor；records 由子进程 torch.save 到 output_path
        status, payload = q.get()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"{label} 子进程异常退出 exitcode={p.exitcode}")
        if status == "err":
            raise RuntimeError(f"{label} 子进程推理失败:\n{payload}")
        print(f"─── {label} 子进程已退出，GPU 显存已释放 ───")
        _wait_gpu_free()
        import time
        print(f"  [gpu] 等待 10 秒，请用 nvidia-smi 确认显存已彻底释放 ...")
        time.sleep(10)
        try:
            records = torch.load(payload, map_location="cpu")
        finally:
            try:
                os.remove(output_path)
            except OSError:
                pass
        return records  # records list

    records_a = _run_in_subprocess("model_a", model_a_path)
    records_b = _run_in_subprocess("model_b", model_b_path)

    print("\n─── 阶段 3: 对比 ───")
    _compare_collected(
        records_a, records_b, threshold, print_logits, compare_logits,
        log_routing=log_routing,
    )


def _log_routing_diff(routing_a: dict, routing_b: dict):
    """逐层打印两模型路由重叠率."""
    keys = sorted(routing_a.keys())
    if not keys:
        print("  [routing] 未找到任何 MoE gate/router, 跳过路由分析")
        return
    print(f"  [routing] 共 {len(keys)} 个 MoE 层, 逐层路由重叠率 (expert_overlap):")
    for k in keys:
        idx_a = routing_a[k][0]   # [n_tokens, top_k]
        entry_b = routing_b.get(k)
        if entry_b is None:
            print(f"    {k}: model_b 无此层")
            continue
        idx_b = entry_b[0]
        n_tokens, top_k = idx_a.shape
        match = 0
        for t in range(n_tokens):
            match += len(set(idx_a[t].tolist()) & set(idx_b[t].tolist()))
        total = n_tokens * top_k
        overlap = match / total if total > 0 else 0.0
        marker = "" if overlap >= 0.95 else "  ⚠"
        short = ".".join(k.split(".")[-3:])
        print(f"    {short}: overlap={overlap:.3f} ({match}/{total}){marker}")



def main():
    parser = argparse.ArgumentParser(
        description="Kimi K3 量化精度对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-a", type=str,
                        default="/models/kimi-k3/Kimi-K3-Mini-MXFP4",
                        help="模型 A 路径")
    parser.add_argument("--model-b", type=str,
                        default="/models/kimi-k3/Kimi-K3-Mini-FP8",
                        help="模型 B 路径")
    parser.add_argument("--device",
                        default="auto" if torch.cuda.is_available() else "cpu",
                        help="推理设备 / device_map (auto=多卡自动分配, balanced=更均衡分配, cuda:0=单卡, cpu=纯CPU)")
    parser.add_argument("--max-memory-per-gpu", type=str, default=None,
                        help="传给 from_pretrained/max_memory 的每卡显存预算，例如 120GiB；用于约束 auto/balanced 分配")
    parser.add_argument("--cpu-convert-then-dispatch", action="store_true",
                        help="先在 CPU 完成 checkpoint conversion，再 dispatch 到 GPU；用于 full-expert Qwen3.8 mini 避免加载阶段 GPU OOM")
    parser.add_argument("--module-device-overrides", type=str, default=None,
                        help="逗号分隔的 完整模块名=设备 映射；在自定义 dispatch 前覆盖模块设备，例如 module_a=cuda:0,module_b=cpu。设备可用 last-layer 表示最后一个 decoder 层所在设备。适用于 Tensor device 不一致错误")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Cosine similarity 阈值 (默认 0.95)")
    parser.add_argument("--print-logits", action="store_true",
                        help="打印 out_a 和 out_b 的完整输出张量 (默认关闭)")
    parser.add_argument("--use-random", action="store_true",
                        help="用随机 id 替代真实 prompt (旧行为, 不推荐)")
    parser.add_argument("--compare-logits", action="store_true",
                        help="对比 logits 替代 hidden_states[-1] (旧行为, 尺度大易夸大误差)")
    parser.add_argument("--log-routing", action="store_true",
                        help="打印每层 MoE gate 路由重叠率, 用于确认两模型路由是否一致")
    parser.add_argument("--dump-per-layer", action="store_true",
                        help="逐层打印 hidden state cosine similarity, 定位误差首次显著下跌的层")
    parser.add_argument("--dump-weight-stats", action="store_true",
                        help="直接从两个 checkpoint 读取指定层关键权重并打印统计/cos，用于定位 W4→BF16 反量化异常")
    parser.add_argument("--dump-submodule", action="store_true",
                        help="hook 指定层内部子模块输出并打印 cosine，用于定位 layer 内首个突变点")
    parser.add_argument("--debug-layer", type=int, default=0,
                        help="--dump-weight-stats/--dump-submodule 使用的层号，默认 0")
    parser.add_argument("--debug-experts", type=str, default=None,
                        help="调试 expert 列表，逗号或区间分隔，例如 0,1,2 或 0:4；默认取前 --debug-num-experts 个")
    parser.add_argument("--debug-num-experts", type=int, default=4,
                        help="未指定 --debug-experts 时默认调试前 N 个 expert，默认 4")
    parser.add_argument("--debug-max-tensors", type=int, default=80,
                        help="--dump-weight-stats 最多打印多少个 tensor，默认 80")
    parser.add_argument("--num-cases", type=int, default=None,
                        help="只跑 _REAL_PROMPTS 的前 N 条测试；默认跑全部")
    args = parser.parse_args()

    model_a_path = Path(args.model_a)
    model_b_path = Path(args.model_b)
    debug_experts = _parse_debug_experts(args.debug_experts, args.debug_num_experts)
    if args.module_device_overrides:
        module_device_overrides = {}
        for item in args.module_device_overrides.split(","):
            module_name, separator, placement = item.strip().partition("=")
            if not separator or not module_name or not placement:
                parser.error("--module-device-overrides 格式应为 完整模块名=设备，多个映射以逗号分隔")
            module_device_overrides[module_name] = placement
    else:
        module_device_overrides = {}
    for path, label in [(model_a_path, "模型 A"), (model_b_path, "模型 B")]:
        if not path.exists():
            print(f"❌ {label} 路径不存在: {path}")
            print("   请先运行 build_mini_kimi_k3.py 构建 mini 模型。")
            sys.exit(1)

    print(f"设备: {args.device}")
    print(f"Cosine 阈值: {args.threshold}")
    if args.dump_weight_stats or args.dump_submodule:
        print(f"调试层: {args.debug_layer}  调试 experts: {debug_experts}")
    print()

    if args.dump_weight_stats:
        dump_checkpoint_weight_stats(
            model_a_path,
            model_b_path,
            args.debug_layer,
            debug_experts,
            args.debug_max_tensors,
        )
        print()

    tokenizer = None
    if not args.use_random:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_a_path), trust_remote_code=True
            )
            print(f"  Tokenizer 加载成功 (vocab_size={tokenizer.vocab_size})")
        except Exception as e:
            print(f"  ⚠ Tokenizer 加载失败, 退回随机 id: {e}")

    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(str(model_a_path), trust_remote_code=True)
    vocab_size = _cfg.text_config.vocab_size if hasattr(_cfg, "text_config") else _cfg.vocab_size
    print(f"vocab_size: {vocab_size}")
    if args.num_cases is not None:
        print(f"测试用例数: 前 {args.num_cases} 条 _REAL_PROMPTS")

    print(f"\n{'=' * 60}")
    print("串行加载对比测试")
    print("=" * 60)
    _sequential_compare(
        model_a_path, model_b_path, args.device,
        tokenizer, vocab_size,
        args.threshold, args.print_logits,
        use_random=args.use_random,
        compare_logits=args.compare_logits,
        dump_per_layer=args.dump_per_layer,
        log_routing=args.log_routing,
        dump_submodule=args.dump_submodule,
        debug_layer=args.debug_layer,
        debug_experts=debug_experts,
        num_cases=args.num_cases,
        max_memory_per_gpu=args.max_memory_per_gpu,
        cpu_convert_then_dispatch=args.cpu_convert_then_dispatch,
        module_device_overrides=module_device_overrides,
    )

    print(f"\n{'=' * 60}")
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
