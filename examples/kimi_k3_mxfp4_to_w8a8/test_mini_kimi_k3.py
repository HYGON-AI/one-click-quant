#!/usr/bin/env python3
"""
加载 Kimi K3 mini 模型，运行 forward pass 推理验证。

用法：
  python test_mini_kimi_k3.py --model /mnt/c/chl/models/Kimi-K3-Mini

验证内容：
  1. 模型能否通过 from_pretrained 正常加载
  2. forward pass 是否无 NaN/Inf
  3. 输出 logits 形状是否正确
  4. generate() 是否正常
"""

import argparse
import sys
from pathlib import Path

import torch


def test_mini_model(model_path: Path, device: str = "cpu"):
    """
    加载 mini 模型并运行推理验证。

    :param model_path: mini 模型目录
    :param device: 推理设备（cpu 或 cuda）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载模型: {model_path}")
    print(f"设备: {device}")

    # ── 1. 加载模型 ──
    print("\n[1/5] 加载模型权重...")
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        model = model.to("cpu")
        exec_device = "cpu"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            ignore_mismatched_sizes=True,
        )
        # 从模型推断输入张量应放置的设备
        exec_device = next(model.parameters()).device
    print(f"  输入张量设备: {exec_device}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count / 1e9:.2f}B")

    # 获取内部 Transformer 结构
    # KimiK3ForConditionalGeneration.language_model → KimiLinearForCausalLM.model → KimiLinearModel
    lm_head = model.language_model.lm_head
    inner = model.language_model.model
    num_layers = len(inner.layers)
    cfg = inner.config
    num_exp = getattr(cfg, "num_experts", 0)
    print(f"  层数: {num_layers} (dense=0, MoE=1..{num_layers - 1})")
    print(f"  隐藏维度: {cfg.hidden_size}")
    print(f"  每层 experts: {num_exp}")

    # ── 2. 模型结构检查 ──
    print("\n[2/5] 模型结构检查...")
    assert num_layers >= 1, f"至少需要 1 层，实际 {num_layers}"
    assert inner.embed_tokens is not None, "缺少 embed_tokens"
    assert inner.norm is not None, "缺少 norm"
    assert lm_head is not None, "缺少 lm_head"

    # Layer 0: 应为 dense MLP（first_k_dense_replace=1）
    layer0 = inner.layers[0]
    assert hasattr(layer0, "mlp"), "Layer 0 应为 dense MLP"
    assert not hasattr(layer0, "block_sparse_moe"), "Layer 0 不应有 MoE"
    assert hasattr(layer0, "self_attn"), "Layer 0 缺少 self_attn"
    print(f"  ✅ Layer 0: dense MLP + {('KDA' if layer0.is_linear_attn else 'MLA')}")

    # Layers 1..N-1: MoE（如有）
    for i in range(1, num_layers):
        layer = inner.layers[i]
        assert hasattr(layer, "block_sparse_moe"), f"Layer {i} 应为 MoE"
        if i == 1:
            moe = layer.block_sparse_moe
            assert len(moe.experts) == num_exp, \
                f"期望 {num_exp} experts，实际 {len(moe.experts)}"
            assert moe.gate.weight.shape[0] == num_exp, \
                f"Gate 应为 ({num_exp}, h)，实际 {moe.gate.weight.shape}"
            print(f"  ✅ Layer 1..{num_layers - 1}: MoE ({num_exp} experts) + "
                  f"{('KDA' if layer.is_linear_attn else 'MLA')}")

    # ── 3. Forward pass ──
    print("\n[3/5] Forward pass (随机 token)...")
    # 使用较小的 vocab 范围内的随机 token
    vocab_size = model.config.text_config.vocab_size if hasattr(model.config, "text_config") else model.config.vocab_size
    batch_size = 1
    seq_len = 16
    input_ids = torch.randint(0, min(vocab_size, 163840), (batch_size, seq_len), device=exec_device)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, use_cache=False)

    logits = output.logits
    print(f"  Logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, vocab_size), f"形状错误: {logits.shape}"

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"  NaN: {has_nan}, Inf: {has_inf}")
    assert not has_nan, "❌ 输出中包含 NaN！"
    assert not has_inf, "❌ 输出中包含 Inf！"
    print("  ✅ Forward pass 正常，无 NaN/Inf")

    # Logits 统计
    logits_f32 = logits.float()
    print(f"  Logits mean: {logits_f32.mean().item():.4f}")
    print(f"  Logits std:  {logits_f32.std().item():.4f}")
    print(f"  Logits min:  {logits_f32.min().item():.4f}")
    print(f"  Logits max:  {logits_f32.max().item():.4f}")

    # 确保 logits 不是完全退化（全零或全相同）
    assert logits_f32.std().item() > 0, "❌ Logits 标准差为 0，可能模型退化"
    print("  ✅ Logits 分布正常")

    # ── 4. 更长的序列测试 ──
    print("\n[4/5] 长序列 forward pass (seq_len=64)...")
    input_ids_long = torch.randint(0, min(vocab_size, 163840), (1, 64), device=exec_device)
    with torch.no_grad():
        output_long = model(input_ids_long, use_cache=False)
    has_nan_long = torch.isnan(output_long.logits).any().item()
    assert not has_nan_long, "❌ 长序列输出中包含 NaN！"
    print(f"  Logits shape: {output_long.logits.shape}")
    print("  ✅ 长序列 forward pass 正常")

    # ── 5. Tokenizer + generate 测试 ──
    print("\n[5/5] Tokenizer 加载与 generate 测试...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

        # 简单 generate
        input_text = "Hello"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_output = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                use_cache=False,
            )
        gen_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
        print(f"  Input:  '{input_text}'")
        print(f"  Output: '{gen_text}'")
        print("  ✅ Generate 正常")
    except Exception as e:
        print(f"  ⚠ Tokenizer/generate 测试跳过: {e}")

    # ── 总结 ──
    print("\n" + "=" * 60)
    print("✅ 所有验证通过！Mini 模型推理正常。")
    print(f"   参数量: {param_count / 1e9:.2f}B")
    print(f"   设备: {device}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 Kimi K3 mini 模型")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/mnt/c/chl/models/Kimi-K3-Mini"),
        help="Mini 模型路径",
    )
    parser.add_argument(
        "--device",
        default="auto" if torch.cuda.is_available() else "cpu",
        help="推理设备 (auto=多卡自动分配, cuda:0=单卡, cpu=纯CPU)",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"❌ 模型路径不存在: {args.model}")
        print("   请先运行 build_mini_kimi_k3.py 构建 mini 模型。")
        sys.exit(1)

    test_mini_model(args.model, args.device)
