## MoE-Quant
---
这是一个基于[IST-DASLab/MoE-Quant](https://github.com/IST-DASLab/MoE-Quant)改造的量化工具，目标是在单机环境下可以对大权重模型(如5T级别BF16)做数据集校准的int4量化转换。

### 特性
- 支持GPTQ量化算法。
- 支持逐层量化，节省显存占用。
- 量化流程与模型适配进行抽象分离。
- 模型适配时支持EP并行，减少单卡显存占用，提高计算性能。
- 量化输出策略支持group和channel。
- 支持W4A16和W4A8输出配置。

### 安装依赖
```shell
pip install flash-attn --no-build-isolation
pip install causal-conv1d --no-build-isolation
pip install flash-linear-attention --no-build-isolation
```

### 模型量化命令
```shell
torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path $MODEL_PATH \
  --dataset_name_or_path open-platypus \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --bits 4 \
  --group_size 128 \
  --rel_damp 0.1 \
  --sym \
  --quantize_only_experts \
  --dtype bfloat16 \
  --save_dir $QUANTIZED_MODEL_PATH
```

### 量化范围控制（`--ignore`）

`--quantize_only_experts` 是历史开关（只量化路由专家）。需要量化 attention /
linear_attn 投影等其它层时，改用 `--ignore`：

```shell
torchrun --nnodes=1 --nproc-per-node=8 --master_port 29501 quant.py \
  --model_name_or_path $MODEL_PATH \
  --dataset_name_or_path open-platypus \
  --num_calibration_samples 512 \
  --max_sequence_length 4096 \
  --bits 4 \
  --group_size 128 \
  --rel_damp 0.1 \
  --sym \
  --ignore "re:.*self_attn\.(q|k|v|o)_proj$" \
  --dtype bfloat16 \
  --save_dir $QUANTIZED_MODEL_PATH
```

量化范围 = **全部 `nn.Linear` − 适配器结构默认项 − `--ignore` 追加项**。
规则语法与 compressed-tensors 一致：`re:<regex>` 走 `re.match`（从头匹配），
裸字符串为精确等值。注意 vLLM 会融合 `q/k/v_proj` 与 `gate/up_proj`，
这三者必须同进同退，否则加载时报 `vLLM requires all to use the same scheme`。

> **W4A8 说明**：DCU 上 sglang 的 `HCUCompressedTensorsW4A8Int8DynamicMoE` 可加载
> compressed-tensors INT4（pack-quantized, channel）路由专家 + 动态 per-token INT8 激活，
> 其它 Linear 需为 INT8 W8A8（int-quantized）。用 `--force_int8` 指定 GPTQ 量化为 INT8 的层，
> 打包时加 `--activation-bits 8`，会生成 `group_int4` + `group_int8` 两个 config_groups。

### 混合精度（`--force_int8`）

`--force_int8` 与 `--ignore` 语法相同，匹配到的层按 INT8 做 GPTQ，其余未 ignore 的层用 `--bits`；
`--ignore` 优先级高于 `--force_int8`。GLM-5.3 的 MTP 层 78、GLM-5.3-Flash 的 MTP 层 45 不参与 GPTQ，打包时加
`--mtp-int8` 以 RTN INT8 量化（与参考脚本逐字节一致）。
普通 GLM-5.3 还支持打包时使用 `--mtp-int4`：MTP routed experts 为 INT4 RTN，
其它未 ignore 的 MTP Linear 为 INT8 RTN。两个 MTP 参数互斥；均不指定时 MTP 保持 BF16。

与 `GLM-5.3-Flash-CHANNEL-INT4-w4a8-v3-same-as-iter_0033200` 量化范围一致的完整命令见
[tests/run_glm5_flash_w4a8.sh](./tests/run_glm5_flash_w4a8.sh)。

### GLM-5.3 BF16 → 混合 INT4/INT8 W4A8

与 `examples/glm/quantize_glm5_3_int4_channel.py` 默认命令的量化范围、channel 粒度、
compressed-tensors 加载格式一致：主干 layers.3–77 routed experts 为 INT4，dense/shared MLP、
MLA q_a/q_b/kv_a/o、indexer wq_b/wk 为 INT8；MTP layers.78 的相应 Linear 为 INT8，
router、kv_b_proj、indexer weights_proj、norm、embedding、lm_head 和 MTP 专用投影/头保持 BF16。
主干使用数据集校准的 GPTQ，MTP 使用 RTN，因此主干权重数值不会与原 RTN 模型逐字节一致，
任务精度需要另行对比。激活的 INT8 动态量化在推理时执行，GPTQ 校准本身只量化权重。

```bash
cd /quant_models/one-click-quant/utils/MoE-Quant
bash tests/run_glm5_3_w4a8.sh
```

脚本默认读取 `/llm-models/GLM-5.3-BF16` 和本地 `OpenThoughts-114k`，使用 8 卡、
512 条校准数据、最大长度 4096。生成的 GPTQ 中间结果为
`/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-intermediate`，最终可加载模型为
`/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ`。
可通过 `MODEL_PATH`、`DATASET`、`SAVE_MODEL_PATH`、`PACKED_MODEL_PATH`、
`NPROC_PER_NODE`、`NUM_CALIBRATION_SAMPLES`、`MAX_SEQUENCE_LENGTH`、`MASTER_PORT` 覆盖默认值。
卡数须整除 256 个专家，校准样本数应为卡数的倍数；Python 环境需要能导入
`transformers.models.glm_moe_dsa`，以及 torch、triton、accelerate、datasets、safetensors、compressed-tensors。

这里不能加 `--quantize_only_experts`（会把 dense/shared/attention/indexer 留在 BF16），
不能设置 `--group_size`（目标是 channel），打包使用 `--activation-bits 8`，
并选择 `--mtp-int8`（默认脚本）或 `--mtp-int4`。
完整两阶段命令见 [tests/run_glm5_3_w4a8.sh](./tests/run_glm5_3_w4a8.sh)。

若需要把 MTP routed experts 也改成 INT4，运行：

```bash
bash tests/run_glm5_3_w4a8.sh --mtp-int4
```

该命令的默认输出为 `/quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-MTP-INT4`。
主干校准和中间结果与默认 INT8 MTP 方案相同。若已有中间结果，只需重新打包：

```bash
python pack_quantized_model.py \
  --model_name_or_path /llm-models/GLM-5.3-BF16 \
  --quantized_model_path /quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-intermediate \
  --packed_model_path /quant_models/GLM-5.3-CHANNEL-INT4-w4a8-GPTQ-MTP-INT4 \
  --dtype bfloat16 \
  --activation-bits 8 \
  --mtp-int4
```

`--mtp-int4` 仅改变 MTP 的 routed experts（gate/up/down），不覆盖其它 MTP 模块的 INT8/BF16
设置，也不对 MTP 做 GPTQ 校准。INT4 config targets 同时覆盖 checkpoint 的
`model.layers.78.mlp.experts.*` 和 sglang NextN 的 `model.decoder.mlp.experts.*`。
当前此参数仅支持普通 GLM-5.3（`glm_moe_dsa`），Flash 仍使用 `--mtp-int8`。

CPU 回归检查：`python tests/test_glm5_3_mtp_int4.py`，覆盖量化数值、打包、MTP 命名匹配和参数互斥。

> 数据集下载和离线使用，可能需要设置环境变量 HF_ENDPOINT=https://hf-mirror.com 或 HF_DATASETS_OFFLINE=1

### 模型打包命令
```shell
python pack_quantized_model.py \
    --model_name_or_path $MODEL_PATH \
    --quantized_model_path $QUANTIZED_MODEL_PATH \
    --packed_model_path $QUANTIZED_MODEL_PATH-packed \
    --dtype bfloat16
```

### 已支持模型
- DeepSeek-V3 (DeepseekV3ForCausalLM)
- Qwen3.8-2.4T-A95B-FP8 (Qwen3_5MoeForCausalLM)
- Qwen3.5-35B-A3B / Qwen3.5-397B-A17B (Qwen3_5MoeForConditionalGeneration，多模态包装类)
- Kimi-K3 (KimiK3ForConditionalGeneration)
- GLM-5.3 (GlmMoeDsaForCausalLM)
- GLM-5.3-Flash (Glm5NextForConditionalGeneration)

> 多模态包装类只量化文本主干；视觉塔 (`model.visual.*`) 与 MTP 头 (`mtp.*`)
> 以 BF16 原样搬运到打包产物，不参与 GPTQ，也不会被丢弃。

详见 [MoE-Quant-ignore改造说明.md](./MoE-Quant-ignore改造说明.md)。

