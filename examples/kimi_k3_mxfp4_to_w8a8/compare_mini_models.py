#!/usr/bin/env python3
"""
Kimi K3 量化精度对比测试

从原始 MXFP4 和量化后 FP8_DYNAMIC 分别构建 mini 模型，
用相同输入推理并对比输出 logits。

构建步骤（先运行）:
  python build_mini_kimi_k3.py --src /models/kimi-k3/Kimi-K3 --dst .../Mini-MXFP4 --src-format mxfp4 -n 4
  python build_mini_kimi_k3.py --src /model/Kimi-K3-FP8-DYNAMIC --dst .../Mini-FP8 --src-format fp8 -n 4

然后运行:
  python compare_mini_models.py --model-a .../Mini-MXFP4 --model-b .../Mini-FP8
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

# ─── ANSI 颜色 ───
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def colored(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def load_model(model_path: Path, device: str):
    """加载 mini 模型；支持 device_map=auto 多卡分配"""
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
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            ignore_mismatched_sizes=True,
        )
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count / 1e9:.2f}B")
    return model


def compare_models(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    vocab_size: int,
    threshold: float,
    print_logits: bool,
):
    """用相同输入对比两个模型的输出"""
    exec_device = next(model_a.parameters()).device

    test_cases = [
        (1, 16,  "短序列 (bs=1, len=16)"),
        (1, 64,  "中等序列 (bs=1, len=64)"),
        (2, 32,  "batch (bs=2, len=32)"),
        (1, 128, "长序列 (bs=1, len=128)"),
    ]

    for batch_size, seq_len, desc in test_cases:
        print(f"\n[{desc}]")
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=exec_device)
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  input_ids range: [{input_ids.min().item()}, {input_ids.max().item()}]")

        with torch.no_grad():
            out_a = model_a(input_ids, use_cache=False).logits.float()
            out_b = model_b(input_ids, use_cache=False).logits.float()

        # NaN/Inf 检查
        for label, out in [("MXFP4", out_a), ("FP8", out_b)]:
            nan_count = torch.isnan(out).sum().item()
            inf_count = torch.isinf(out).sum().item()
            if nan_count or inf_count:
                print(f"  ❌ {label}: NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"  ✅ {label}: 无 NaN/Inf")

        if print_logits:
            print(f"\n  --- out_a (MXFP4): shape={out_a.shape}, dtype={out_a.dtype}, sum={out_a.sum().item():.6f} ---")
            print(out_a)
            print(f"\n  --- out_b (FP8): shape={out_b.shape}, dtype={out_b.dtype}, sum={out_b.sum().item():.6f} ---")
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

        # 相对误差
        rel_err = ((out_a - out_b).abs() / (out_a.abs() + 1e-8)).mean().item()
        print(f"  相对误差 mean:   {rel_err:.6e}")

        # 最大值差异
        max_diff = (out_a - out_b).abs().max().item()
        print(f"  最大绝对差异:    {max_diff:.6e}")

        # Top-K 重叠率（预测 token 一致性）
        for k in (1, 5, 10):
            topk_a = out_a.topk(k, dim=-1).indices
            topk_b = out_b.topk(k, dim=-1).indices
            overlap = 0
            for t in range(topk_a.shape[1]):
                overlap += len(set(topk_a[0, t].tolist()) & set(topk_b[0, t].tolist()))
            total = k * topk_a.shape[1]
            print(f"  Top-{k:2d} 重叠率:      {overlap}/{total} ({100*overlap/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Kimi K3 量化精度对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-a", type=Path,
                        default=Path("/models/kimi-k3/Kimi-K3-Mini-MXFP4"),
                        help="原始 MXFP4 构建的 mini 模型路径")
    parser.add_argument("--model-b", type=Path,
                        default=Path("/models/kimi-k3/Kimi-K3-Mini-FP8"),
                        help="量化后 FP8_DYNAMIC 构建的 mini 模型路径")
    parser.add_argument("--device",
                        default="auto" if torch.cuda.is_available() else "cpu",
                        help="推理设备 (auto=多卡自动分配, cuda:0=单卡, cpu=纯CPU)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Cosine similarity 阈值 (默认 0.95)")
    parser.add_argument("--print-logits", action="store_true",
                        help="打印 out_a 和 out_b 的完整 logits (默认关闭)")
    args = parser.parse_args()

    for path, label in [(args.model_a, "模型 A (MXFP4)"), (args.model_b, "模型 B (FP8)")]:
        if not path.exists():
            print(f"❌ {label} 路径不存在: {path}")
            print("   请先运行 build_mini_kimi_k3.py 构建 mini 模型。")
            sys.exit(1)

    print(f"设备: {args.device}")
    print(f"Cosine 阈值: {args.threshold}")
    print()

    # 加载模型
    print("加载模型...")
    model_a = load_model(args.model_a, args.device)
    model_b = load_model(args.model_b, args.device)

    # A_log 同步
    synced = 0
    for (na, pa), (nb, pb) in zip(model_a.named_parameters(), model_b.named_parameters()):
        if na.endswith(".A_log"):
            pb.data.copy_(pa.data)
            synced += 1
    if synced > 0:
        print(f"  同步了 {synced} 个 A_log 参数 (model_a → model_b)")

    # 获取 vocab_size
    vocab_size = model_a.config.text_config.vocab_size if hasattr(model_a.config, "text_config") else model_a.config.vocab_size
    print(f"\nvocab_size: {vocab_size}")

    # 设置随机种子以保证可复现
    torch.manual_seed(42)

    # 对比
    print(f"\n{'=' * 60}")
    print("开始对比测试")
    print("=" * 60)
    compare_models(model_a, model_b, vocab_size, args.threshold, args.print_logits)

    print(f"\n{'=' * 60}")
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
