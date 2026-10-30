# AutoQuantize 精度分析工具

## 简介

one-click-quant 中的混合精度分析工具。基于 NVIDIA ModelOpt，输入模型和校准数据，自动选择哪些目标模块使用 **INT4**、哪些使用 **INT8**，输出 `autoquantize_plan.json`，供后续 RTN/GPTQ 量化使用。

自动保留源配置的 `ignore/ignored_layers` 等排除规则，Q/K/V、gate/up 等按融合组统一决策。**这里只生成分配计划，不导出量化模型。**

支持浮点模型（BF16/FP16/FP32），以及已接入源编码的 Block FP8、MXFP4/FP8/BF16 混合模型。模型架构需被 Transformers 和 ModelOpt 支持；量化源权重需为 safetensors 格式。
默认单设备模式需要容纳完整解码模型及评分开销。MiMo 可通过本工具的 `--parallel ep` 使用多卡专家分片、逐块加载和反向重算，无需加载完整浮点模型到单卡。

## 使用方法

在已安装适配当前 NVIDIA/DCU 的 PyTorch 环境中安装依赖，保留厂商适配的 PyTorch：

```bash
python -m pip install "nvidia-modelopt[hf]==0.47.0"
```

从 **one-click-quant 根目录**运行，替换模型目录和输出路径：

```bash
python -m utils.auto_quantize \
  --model /path/to/model \
  --calibration-data open-platypus \
  --calibration-samples 8 \
  --calibration-seq-length 512 \
  --effective-bits 4.2 \
  --device cuda:0 \
  --output /path/to/analysis/autoquantize_plan.json
```

完整 MiMo 使用多卡 EP，例如以下两卡命令（替换成空闲卡）：

```bash
CUDA_VISIBLE_DEVICES=4,5 HIP_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m utils.auto_quantize \
  --parallel ep --trust-remote-code \
  --model /llm-models-1/XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --calibration-data open-platypus \
  --calibration-samples 8 --calibration-seq-length 512 \
  --effective-bits 4.2 --device cuda:0 \
  --output /quant_models/models/MiMo-AutoQuantize-EP/autoquantize_plan.json
```

EP 支持 MiMo 和 Kimi-K3，未实现通用 TP；专家数需能被进程数整除，实际校准序列块数需为进程数的正整数倍。八卡使用 `--nproc_per_node=8` 和八个可见设备。增加 `--dry-run` 可只检查元数据与分片，不解码大权重、不输出计划。

Kimi-K3 使用相同命令，将 `--model` 改为 `/llm_models_2/Kimi-K3`，并选择新的输出路径。运行环境需有源模型依赖的 `fla-core`，且 KDA 算子支持反向传播。该路径只构造文本模型，按 decoder block 加载本 rank 的专家和共享权重，在块内复用，前向结束释放，反向重算时重新加载；支持源目录的嵌套 MXFP4 配置及 `weight_packed`/`weight_scale`。函数式 attention residual 投影保持浮点，不参与搜索。

模型使用本地 HuggingFace 目录，包含配置、权重和 tokenizer。需要模型自定义代码时，增加 `--trust-remote-code`。DCU 同样使用 `cuda:0` 设备名。

`open-platypus` 首次使用需要下载或已有缓存，也可将 `--calibration-data` 替换成本地 `.txt`、`.json`、`.jsonl` 文件。JSONL 每行一条文本，例如：

```json
{"text": "请解释二分查找，并写一个 Python 实现。"}
```

## 参数与结果

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--model` | 必填 | 本地模型目录 |
| `--output` | 必填 | 输出 JSON 路径，不能覆盖已有文件 |
| `--calibration-data` | 必填 | 校准数据集或本地文本路径 |
| `--calibration-samples` | `8` | 最多使用的校准序列块数 |
| `--calibration-seq-length` | `512` | 每个序列块的最大 token 数 |
| `--effective-bits` | `4.2` | 搜索权重的平均位宽上限，范围 4～8 |
| `--device` | `cuda:0` | 分析设备；EP 模式按 LOCAL_RANK 选择对应可见卡 |
| `--parallel` | `single` | `ep` 启用 MiMo 专家分片与逐块加载 |
| `--dry-run` | 关闭 | EP 元数据与分片预检，不执行评分 |
| `--dtype` | `bfloat16` | 分析工作精度，可选 BF16/FP16/FP32 对应的 `bfloat16/float16/float32` |
| `--exclude` | 无额外排除 | 额外保留浮点的模块，例如 `--exclude lm_head` |
| `--include` | 不限制范围 | 只搜索匹配的模块，例如 `--include 'model.layers.0.*'` |
| `--policy` | 不使用 | 复用包含 include/exclude/分组规则的 JSON 配置文件 |

通常不需要设置 `exclude/include/policy`。排除规则优先，include 不会重新启用源配置已排除的层。

`effective-bits` 先用默认 `4.2` 即可。调低会减少 INT8 的预算；设为 `4` 要求搜索目标全部 INT4，受后端约束时可能不可行。预算不包含排除层、scale 和缓存，不能直接当作模型文件大小或精度保证。

输出json主要查看：

| 字段 | 含义 |
|---|---|
| `module_bits` | 每个搜索目标的位宽，`4` 为 INT4、`8` 为 INT8 |
| `excluded_modules` | 保留浮点、不参与搜索的目标及原因 |
| `actual_effective_bits` | 分配后的实际平均位宽 |

命令还会输出 `autoquantize_plan.sensitivity.json` 和可阅读的 `.sensitivity.txt`，按 INT4 候选敏感度从高到低列出融合组、INT4/INT8 分数、分数下降量 `S4-S8` 及最终位宽。EP 模式仅由 rank 0 求解和保存。分数下降量不等于实测精度提升，终端显示前 10 名。

后续由支持此计划格式的 RTN/GPTQ 脚本读取 JSON，执行量化并评测精度。**当前通用计划不能直接交给旧 MiMo 专用 RTN/GPTQ 脚本。**

完整参数使用 `python -m utils.auto_quantize --help` 查看。