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

> ⚠️ **W4A8 说明**：MoE-Quant 的打包链路走 compressed-tensors，而 Hygon DCU 上
> compressed-tensors 的 W4A8 kernel 不可用（upstream 的 W4A8-INT8 MoE 仅 CPU/Arm，
> W4A8-FP8 仅 SM90）。**DCU 上可用的是 W4A16 / W8A8**；若确需 W4A8，请使用
> `python main.py --alg slimquant_ptq --scheme W4A8`（输出 `quant_method: slimquant_w4a8`）。

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

