# 扩展：MoE-Quant（`utils/MoE-Quant/`）

> **权威来源**：`utils/MoE-Quant/README.md`、`src/models/base.py`（adapter 契约）、
> `src/models/registry.py`（注册与顺序）、`quant.py` / `pack_quantized_model.py` 的 `parse_args`。
> 本文只写**两阶段流程、adapter 契约、位宽约束、新增模型改哪里**。

## 1. 定位与前置

单机量化超大 MoE（BF16 到 T 级）的**有校准** INT4 GPTQ 链路。与前两条链路的区别：

| | slimquant | llmcompressor | **MoE-Quant** |
|---|---|---|---|
| 需要校准数据 | 否 | 是 | **是** |
| 逐层省显存 | 不涉及 | 部分 | **是（逐 block）** |
| 产物 | HF safetensors | HF safetensors | **先 per-block `.pt`，再打包成 HF** |

**工作目录必须是 `utils/MoE-Quant/`**（两个脚本都 `from src import ...`），
或在仓库根设 `PYTHONPATH=utils/MoE-Quant`。依赖见其 README（flash-attn / causal-conv1d / flash-linear-attention）。

## 2. 两阶段

```bash
cd utils/MoE-Quant

# ① GPTQ：逐 block 量化，产物是 <save_dir>/<layer_name>/quantized_weight.pt
#    （每 block 一份）+ <save_dir>/metadata.pt（全量一份，记录 bits/group_size/quantize_only_experts）
# NPROC / MASTER_PORT / CALIB_SET 为示例，按卡数与本地数据改，不要写死 8 / 29501 / open-platypus
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
CALIB_SET="${CALIB_SET:-open-platypus}"
torchrun --nnodes=1 --nproc-per-node="$NPROC" --master_port "$MASTER_PORT" quant.py \
  --model_name_or_path "$MODEL" --dataset_name_or_path "$CALIB_SET" \
  --num_calibration_samples 512 --max_sequence_length 4096 \
  --bits 4 --group_size 128 --rel_damp 0.1 --sym \
  --quantize_only_experts --dtype bfloat16 --save_dir "$QUANT"

# ② 打包：写成 HF 格式 + quantization_config；activation-bits 决定 W4A16 / W8A8
python3 pack_quantized_model.py \
  --model_name_or_path "$MODEL" --quantized_model_path "$QUANT" \
  --packed_model_path "$QUANT-packed" --dtype bfloat16 --activation-bits 16
```

- ① 的产物**不是**能直接加载的 HF 模型，必须跑 ② 才有 config / index / safetensors
- `--resume` 从 `save_dir` 里已有的最大 block 号继续（每个 block 单独落盘，所以可断点续跑）
- ② 的 `--model_name_or_path` 是**原始 BF16 源模型**（不是 ① 的产物），它会从源目录读 safetensors index
- `metadata.pt` 记录 `bits` / `group_size` / `quantize_only_experts`，② 会**读回并覆盖 CLI 值**，
  且强制 `sym = True`（不支持非对称）。所以这三个参数**只在 ① 生效**，② 再传无效
- `--dtype` 在 ① 的默认值是 `float16`。源权重是 BF16 时必须显式 `--dtype bfloat16`，
  否则权重会按 fp16 处理（GLM / Kimi 的 `validate_packing_args` 也强制要求 bfloat16）

## 3. 位宽与格式（改参数前先看这张）

| 维度 | 取值 | 决定什么 |
|---|---|---|
| `--bits`（①） | 4 / 8 | 权重位宽；各 adapter 自行校验（见下） |
| `--group_size`（①） | 省略 = per-channel；整数 = group-wise | 必须是量化层输入维的约数 |
| `--activation-bits`（②） | `16`（默认）→ `pack-quantized`；`8` → `int-quantized` | **W4A16 vs W8A8 在这里定**，不在 ① |
| `--quantize_only_experts`（①） | 开关 | 决定 ② 写入的 ignore 规则（见 §4） |

硬约束（**以 adapter 内的报错为准**，别按模型族记忆）：

| 模型族 | `--bits` | `--quantize_only_experts` / `--sym` | 打包额外要求 |
|---|---|---|---|
| DeepSeek-V3 | 仅 `4` | 不校验 | 不校验 |
| Qwen3.5 / 3.8 MoE | 仅 `4` | 不校验 | 不校验 |
| Kimi-K3 | 仅 `4` | **两者都必须给** | `--dtype bfloat16` + `--activation-bits 16`（只支持 W4A16） |
| GLM-5.3 / GLM-5.3-Flash | `4` 或 `8` | **两者都必须给** | `--dtype bfloat16`；`--activation-bits 8` 必须配 `--bits 8` |

**这条链路不存在 W4A8**：`int-quantized` 格式在 `pack_weight` 里硬性要求 8bit，
`--bits 4 --activation-bits 8` 会直接抛错。要 W4A8 走 `--alg slimquant_ptq`。

## 4. `quantization_config` 由谁写

`pack_quantized_model.py::prepare_quantization_config` 一次性写出：

- `quant_method: compressed-tensors`、`quantization_status: compressed`
- `format`：`int-quantized`（activation-bits 8）/ `pack-quantized`（16）
- `weights.strategy`：`channel`（`group_size` 省略）/ `group`
- **`group_size` 省略时落盘写 `-1` 而不是 `null`**（vLLM 的 WNA16 路径断言 `== -1`）
- `ignore`：来自 `adapter.get_quantization_ignore(quantize_only_experts)`

`get_quantization_ignore` 的默认实现（`base.py`）：

- 总是忽略 `lm_head`
- `quantize_only_experts=True` 时再加 `re:.*self_attn.*`、`re:.*shared_experts.*`、
  `re:.*mlp\.(gate|up|gate_up|down)_proj.*`，规则名 `default_experts_only`

也就是说 **非 expert 层保持浮点**是这一档的默认行为，不是 bug。

## 5. Adapter 契约

模型差异全部收在 `src/models/<model>_adapter.py`，引擎（GPTQ、校准、通信、序列化）不动。
必须/常用钩子：

| 方法 | 作用 |
|---|---|
| `matches(config)` | 类方法，判断是否接管该 config；默认返回 `False` |
| `prepare_config` | 加载前的分布式相关配置改写 |
| `checkpoint_keys_for_model_key` | 逻辑 key → 物理 checkpoint key（fused 张量、别名） |
| `materialize_state_dict` | 用物理张量拼出逻辑 state_dict；默认实现会**先把 FP8 反量化**（缺 `_scale_inv` 直接报错） |
| `build_empty_model` / `build_packing_model` | 空模型构造（`trust_remote_code=True`） |
| `get_transformer_layers` / `get_layer_prefix` / `get_block_index_from_layer_name` | 定位 block |
| `is_routed_expert` | 默认按 `mlp.experts.<i>.{down,gate,up}_proj` 正则 |
| `get_quantization_ignore` | 决定 ② 写出的 `ignore` |
| `expected_quantized_layer_names` | 返回非空集合时，缺层直接报错（完整性校验） |
| `count_extra_shards` / `save_extra_weights` / `copy_artifacts` | 额外分片与 remote-code 文件的落盘 |

## 6. 新增一个模型

1. 写 `src/models/<model>_adapter.py`，继承 `ModelAdapter`，实现 `matches` 与必需的 key 映射
2. 注册到 `registry.py` 的 `_ADAPTER_CLASSES` —— **顺序敏感**：更具体的必须排在前面，
   `get_model_adapter` 是**首个 `matches` 命中即返回**
3. 在该 adapter 的 `validate_quantization_args` / `validate_packing_args` 里写位宽约束（报错信息即文档）
4. 支持哪些模型**以 registry 为准**：`grep -n -A10 "_ADAPTER_CLASSES:" src/models/registry.py`

**不要改 `quant.py` 主流程**。两套注册表互不相干：这里的 `registry.py` 与 `preprocess/adapters/` 完全独立，
接一个模型不会让另一条链路支持它。

## 7. 验收

- 产物 `quant_method: compressed-tensors`，`format` 与 `--activation-bits` 对应
- 权重是 `weight_packed`（int32，pack-quantized）或 int8（int-quantized），且**每个都有配对 scale**
- `ignore` 与 `quantize_only_experts` 的预期一致（非 expert 层为 bf16）
- `scripts/check_artifact.py <产物目录>` 过，再按 `04-verify-eval.md` 做 PPL 对比
