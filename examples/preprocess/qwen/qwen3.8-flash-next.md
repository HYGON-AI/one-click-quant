# Qwen3.8-Flash-Next：预处理 + slimquant W4A8

两段式：先用 `preprocess` 写出仍是 BF16 的目录，再用 `slimquant_ptq` 做 W4A8。

Qwen3.8-Flash-Next架构是 `Qwen4ExpForConditionalGeneration`。权重已是 BF16，recipe `unpack_mxfp4: false`。残差是 4×H hyper-connection，`hc_norm` 为 Gemma（`y = rms(x) * (1+γ)`），对角 `(1+γ)` 与 Hadamard 不对易，**默认不做 QuaRot**。默认 transform：`flex_awq_ssz` + `flex_smooth_quant`。Smooth 只做 MTP `pre_fc_norm_*`，层内 `hc_norm` 不做。Flex AWQ 覆盖 `shared_expert` 与融合专家 `gate_up_proj` 的 up 半。详见 `preprocess/recipes/qwen3_8_flash_next.yaml`。

## 1. 预处理

```bash
cd one-click-quant

python3 main.py --alg preprocess \
  --model /parastor/opendas/DL_DATA/llm-module/Qwen/Qwen3.8-Flash-Next \
  --save-dir /module/Qwen3.8-Flash-Next-preprocessed \
  --recipe preprocess/recipes/qwen3_8_flash_next.yaml \
  --transforms flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8
```

`--save-dir` 必须与 `--model` 不同。`--preprocess-gpus` 按层分卡改权重。默认 `use_calib_acts: false`。产出是浮点 HF 目录，没有 `compression_config`。

## 2. slimquant W4A8

`slimquant_ptq` 不加载模型、不需要校准数据，直接改 safetensors。`--scheme W4A8`：名字含 `.mlp.experts.` 的权重量化成 INT4 并打包，其余 Linear 做 INT8。

该架构不在 `templates/slimquant_ptq.py` 的内置 ignore 表里，只自动跳过 `lm_head`、`re:.*mlp.gate$`、`re:.*embed_tokens.*`，以及模块名以 `norm` / `layernorm` 结尾的权重。下面三类仍会被送进量化并因不是二维 Linear 而失败，需要用 `--ignore` 追加（追加，不替换默认项）：

| 张量 | 形状 | 原因 |
|------|------|------|
| `*.linear_attn.conv1d.weight` | `(10240, 1, 4)` | 一维卷积核 |
| `mtp.pre_fc_norm_embedding.weight` | `(2560,)` | 名字不以 `norm` 结尾的一维 γ |
| `model.visual.patch_embed.proj.weight` | `(1152, 3, 2, 16, 16)` | 视觉 patch 卷积 |

```bash
python3 main.py --alg slimquant_ptq \
  --model /module/Qwen3.8-Flash-Next-preprocessed \
  --scheme W4A8 \
  --save-dir /module/Qwen3.8-Flash-Next-preprocessed-w4a8 \
  --from-preprocessed \
  --ignore "re:.*conv1d.*,re:.*norm.*,re:.*visual.*" \
  --device-map cuda:0
```

`--from-preprocessed` 会检查输入是浮点权重且不含 `compression_config`。`--device-map cuda:N` 指定单卡；不写时 `auto` 会被改成 `cuda:0`。这个算法不会多卡切分。

## 3. 导入失败：NumPy / SciPy

若在 `from compressed_tensors` 处报 `All ufuncs must have type numpy.ufunc`，是容器里 SciPy 与当前 NumPy 不匹配：

```bash
pip install --force-reinstall --no-cache-dir "numpy==2.2.6" "scipy==1.15.3"
```
