# DeepSeek-V3：预处理 + MoE-Quant W4A8

两段式：先用 `preprocess` 把官方 block FP8 解包并写成 BF16 目录，再用 `utils/MoE-Quant` 做校准 GPTQ（experts INT4，Attention 与 shared experts INT8），最后 pack 成 W4A8。

## 1. 预处理

```bash
cd one-click-quant

python3 main.py --alg preprocess \
  --model /public/opendas/DL_DATA/llm-models/deepseek-ai/DeepSeek-V3 \
  --save-dir /module/DeepSeek-V3-preprocessed \
  --recipe preprocess/recipes/deepseek_v3.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8
```

官方权重是 block FP8（`e4m3` + `weight_scale_inv`，128×128）。`unpack_fp8: true` 会先解包到 `/module/DeepSeek-V3-preprocessed.bf16_staging`（约 1.3T+），再 8 卡按层做 QuaRot / Flex。`--save-dir` 必须与 `--model` 不同。`--preprocess-gpus` 按层分卡改权重。

默认 `use_calib_acts: false`，用权重侧 proxy 搜 scale，不需要 `acts.pt`。

| Transform | 作用 |
|-----------|------|
| `quarot` | RMSNorm fuse + Hadamard。覆盖 hidden、MLA `q_lora` / `kv_lora` |
| `flex_awq_ssz` | dense `mlp.up_proj`、`shared_experts.up_proj`、`experts.*.up_proj` |
| `flex_smooth_quant` | LN 种子由 adapter 展开全部下游；MoE `post_attention_layernorm` 只在前 8 个消费者上搜 scale，再吸到全部消费者 |

Adapter 还覆盖 dense MLP（`first_k_dense_replace`）、逐专家 `experts.{i}.*`、`shared_experts`，以及 MTP 层 `shared_head`。`enorm`/`hnorm`→`eh_proj` 不是单 Norm→Linear，不做 Norm fuse。

## 2. MoE-Quant

在预处理后的 BF16 目录上做。`--mode mixed_w4_w8`：路由专家 INT4，Attention（含 MLA `q_a/q_b/kv_a/kv_b`）和 `shared_experts` INT8，`lm_head`、embedding、router `mlp.gate` 以及 dense `gate/up/down_proj` 保持浮点。校准集关键字 `open-platypus` 对应 `garage-bAInd/Open-Platypus`。

```bash
cd one-click-quant/utils/MoE-Quant

torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path /module/DeepSeek-V3-preprocessed \
  --dataset_name_or_path open-platypus \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --mode mixed_w4_w8 \
  --bits 4 \
  --group_size 128 \
  --rel_damp 0.1 \
  --sym \
  --dtype bfloat16 \
  --attn_implementation eager \
  --save_dir /module/DeepSeek-V3-w4a8-MOE-Quant-gptq
```

这一步写出 GPTQ 中间结果（约 642G），还不是推理用的 packed 权重。

## 3. 打包

```bash
python3 pack_quantized_model.py \
  --model_name_or_path /module/DeepSeek-V3-preprocessed \
  --quantized_model_path /module/DeepSeek-V3-w4a8-MOE-Quant-gptq \
  --packed_model_path /module/DeepSeek-V3-w4a8-MOE-Quant \
  --dtype bfloat16 \
  --activation-bits 8 \
  --mode mixed_w4_w8
```

`--activation-bits 8` 得到 W4A8 / W8A8。pack 结果在 `/module/DeepSeek-V3-w4a8-MOE-Quant`（约 334G）。`mixed_w4_w8` 会写出 `group_w4` 与 `group_w8` 两组 `config_groups`。打包格式是 GPTQ `weight_packed`（int32）。

## 4. 若 commit_inplace 在 merge 后失败（Directory not empty: rank*_spill）

8 卡 transform + merge 已完成、只卡在 `rmtree(save_dir)` 时，**不必重跑整段预处理**。确认 staging 仍在：

- `/module/DeepSeek-V3-preprocessed.bf16_staging`（含 `UNPACK_FP8_DONE.json`）
- `save_dir/.preprocess_shards/rank{0..7}.json` 与 rank spill（rank7 的 chunk 若已丢，脚本会用旧 `merge_spill` 补）

```bash
cd one-click-quant
python3 preprocess/resume_commit_inplace.py \
  --staging /module/DeepSeek-V3-preprocessed.bf16_staging \
  --save-dir /module/DeepSeek-V3-preprocessed \
  --recipe preprocess/recipes/deepseek_v3.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  --gpus 8
```

成功后会写出最终目录和 `PREPROCESS_DONE.json`。验证无再删备份：

```bash
rm -rf /module/.preprocess_shards_bak_DeepSeek-V3-preprocessed
```

## 5. 导入失败：NumPy / SciPy

若 import 时报 `All ufuncs must have type numpy.ufunc`，是容器里 SciPy 与当前 NumPy 不匹配：

```bash
pip install --force-reinstall --no-cache-dir "numpy==2.2.6" "scipy==1.15.3"
```
