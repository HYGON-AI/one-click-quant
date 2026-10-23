# Qwen3.8-Flash-Next 离线混合量化（W4 专家 + W8 指定层）

本目录脚本把 **Qwen3.8-Flash-Next** BF16 权重离线压成 slimquant 风格混合 checkpoint：

1. 主网 fused experts → **INT4**
2. 主网 Attention QKV / shared expert / linear_attn 主投影 → **INT8**
3. （可选）PLE `ngram_embedding` → **INT8**
4. （可选）MTP 块：experts → **INT4**，attn QKV / shared_expert → **INT8**

带 `-MSE` 的脚本只做还原误差检查，不写权重。

默认路径可按机器修改；默认使用8卡量化，可以使用环境变量`HIP_VISIBLE_DEVICES`控制卡号。

## 目录与产物

| 阶段 | 脚本 | 默认输入 | 默认输出 |
|------|------|----------|----------|
| 1 | `qwen38_flash_next_bf16_experts_to_int4.py` | `/model/Qwen/Qwen3.8-Flash-Next` | `/module/Qwen3.8-Flash-Next-INT4` |
| 1 校验 | `qwen38_flash_next_bf16_experts_to_int4-MSE.py` | 上表 BF16 / INT4 | 打印 MSE |
| 2 | `qwen38_flash_next_int4_to_int4_8.py` | `/module/Qwen3.8-Flash-Next-INT4` | `/module/Qwen3.8-Flash-Next-w4a8-slimquant` |
| 2 校验 | `qwen38_flash_next_int4_to_int4_8-MSE.py` | Stage1 INT4（W8 层仍为 BF16）/ Stage2 | 打印 MSE |
| 3 可选 | `qwen38_flash_next_ngram_to_int8.py` | Stage2 混合权重 | `/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8` |
| 3 校验 | `qwen38_flash_next_ngram_to_int8-MSE.py` | Stage2 / Stage3 | 打印 MSE |
| 4 可选 | `qwen38_flash_next_mtp_to_w4a8.py` | Stage2 或 Stage3（MTP 仍为 BF16） | `/module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp` |
| 4 校验 | `qwen38_flash_next_mtp_to_w4a8-MSE.py` | 含 BF16 MTP 的源 / Stage4 | 打印 MSE |

## 推荐流程

在仓库根或本目录执行均可（下面以本目录为例）。

### Stage 1：主网专家 BF16 → INT4

量化 `language_model.layers.*.mlp.experts.{gate_up_proj,down_proj}`（fused MoE）。`mtp.*` 保持 BF16；MTP 混合量化见 Stage 4。

```bash
cd examples/qwen/qwen3_8-flash-next

# 可选：只扫描、不写盘
python3 qwen38_flash_next_bf16_experts_to_int4.py --dry-run

python3 qwen38_flash_next_bf16_experts_to_int4.py \
  --input-path /model/Qwen/Qwen3.8-Flash-Next \
  --output-path /module/Qwen3.8-Flash-Next-INT4

# 校验 INT4 解包还原 vs 原 BF16
python3 qwen38_flash_next_bf16_experts_to_int4-MSE.py \
  --src /model/Qwen/Qwen3.8-Flash-Next \
  --dst /module/Qwen3.8-Flash-Next-INT4
```


### Stage 2：主网指定层 BF16 → INT8（专家 INT4 保留）

在 Stage1 结果上量化：

- `self_attn.{q,k,v}_proj`
- `mlp.shared_expert.{gate,up,down}_proj`
- `linear_attn.{in_proj_qkv,in_proj_z}`

`o_proj` / `out_proj`、router、hyper_connection、indexer、PLE、visual、embed、`lm_head`、**整段 MTP** 等保持浮点。

```bash
python3 qwen38_flash_next_int4_to_int4_8.py --dry-run

python3 qwen38_flash_next_int4_to_int4_8.py \
  --input-path /module/Qwen3.8-Flash-Next-INT4 \
  --output-path /module/Qwen3.8-Flash-Next-w4a8-slimquant

python3 qwen38_flash_next_int4_to_int4_8-MSE.py \
  --src /module/Qwen3.8-Flash-Next-INT4 \
  --dst /module/Qwen3.8-Flash-Next-w4a8-slimquant
```

不需要 ngram INT8、也不量化 MTP 时，Stage2 目录即可作为推理产物。

### Stage 3（可选）：PLE ngram → INT8

把约 102GiB 的 `ple_embedding.ngram_embedding.shard_*.weight` 压成 INT8。当前 sglang PLE 路径通常仍吃 BF16/FP8，本步主要是出权重；服务端需另有 INT8（或加载时反量化）支持。

```bash
python3 qwen38_flash_next_ngram_to_int8.py --dry-run

# 默认 per_tensor：整表一个 scale（更接近官方 FP8 PLE）
python3 qwen38_flash_next_ngram_to_int8.py \
  --input-path /module/Qwen3.8-Flash-Next-w4a8-slimquant \
  --output-path /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8 \
  --scale-mode per_tensor

# 或 per_channel：每行一个 scale，精度更高
python3 qwen38_flash_next_ngram_to_int8.py \
  --input-path /module/Qwen3.8-Flash-Next-w4a8-slimquant \
  --output-path /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8 \
  --scale-mode per_channel

python3 qwen38_flash_next_ngram_to_int8-MSE.py \
  --src /module/Qwen3.8-Flash-Next-w4a8-slimquant \
  --dst /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8
```

### Stage 4（可选）：MTP → W4 专家 + W8 指定层

在主网已量化、**MTP 仍为 BF16** 的 checkpoint 上（推荐 Stage2 或 Stage3 产物）对齐主网方案：

- INT4：`mtp.layers.*.mlp.experts.{gate_up_proj,down_proj}`
- INT8：`mtp.layers.*.self_attn.{q,k,v}_proj`、`mtp.layers.*.mlp.shared_expert.{gate,up,down}_proj`
- 其余 MTP（`o_proj`、router、hyper_connection、`fc_*`、norm 等）保持 BF16
- 主网已有 INT4/INT8/ngram 张量原样拷贝
- 更新 `compression_config`：去掉整段 `re:^mtp\..*` ignore，把 MTP 写入 `experts` / `w8a8_include`，并加上细粒度 MTP ignore

```bash
python3 qwen38_flash_next_mtp_to_w4a8.py --dry-run

# 输入可以是 Stage2 或 Stage3；默认会尝试 ngram-int8 路径
python3 qwen38_flash_next_mtp_to_w4a8.py \
  --input-path /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8 \
  --output-path /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp

python3 qwen38_flash_next_mtp_to_w4a8-MSE.py \
  --src /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8 \
  --dst /module/Qwen3.8-Flash-Next-w4a8-slimquant-ngram-int8-mtp
```

## 说明

- 各转换脚本跑完会在输出目录写 `QUANTIZATION_DONE.txt` 和更新后的 `config.json` / `model.safetensors.index.json`。
- 可用 GPU 数由当前可见 CUDA 设备决定；无 GPU 时请勿直接跑写盘转换（可先 `--dry-run`）。
