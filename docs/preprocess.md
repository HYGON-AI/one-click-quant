# 独立预处理模块

预处理与量化解耦：先写出仍是 BF16/FP16 的 HF 目录，再交给 `slimquant_ptq` / `model_free_ptq` / MoE-Quant。

## 设计目标

```text
通用框架
  WeightStore / pipeline / transform 注册 / absorb 语义
        │
        ├── 每模型 recipe YAML     （adapter 名、transform 顺序、include 种子）
        └── 每模型 adapter 代码    （LN map / rotate pairs / UpDown 配对）
```

| 层 | 职责 | 跨模型复用？ |
|----|------|--------------|
| `WeightStore` / `pipeline` | 读写 safetensors、跑 transform、写出浮点 ckpt | 是 |
| `absorb.py` | UpDown / NormLinear 吸收 | 是 |
| `quarot` / `flex_*` transforms | 搜索 + 调用 adapter + absorb | 是（逻辑通用） |
| `adapters/<model>.py` | 模块命名、LN 消费者表、旋转对、`w3→w2` | **否，每模型一份** |
| `recipes/<model>.yaml` | `adapter:` + include 种子 + 超参 | **否，每模型一份** |

新模型需要补充新adapter和新recipe。

---

## 流程

```text
  假设原始模型是 MXFP4 / 浮点权重
        │
        ▼
  adapter.prepare_weights（Kimi：MXFP4→BF16 unpack）
        │
        ▼
  pipeline 加载 recipe → 依次跑 transforms
        │  写出 BF16/FP16 HF 目录
        ▼
  slimquant_ptq / model_free_ptq / …
```

### 支持的预处理功能

| Transform | 子图 | 吸收规则 | 模块名从哪来 |
|-----------|------|----------|--------------|
| `flex_awq_ssz` | UpDown | `W_up /= s`(out)、`W_down *= s`(in) | recipe `include` 种子 + `adapter.resolve_up_down` |
| `flex_smooth_quant` | NormLinear | 全部下游 `W *= s`(in); RMSNorm:`γ/=s`; Gemma:`γ=(1+γ)/s-1` | recipe 种子 + `adapter.build_ln_fuse_map` |
| `quarot` | Hadamard + LN→Linear | 吸收成功后 RMSNorm γ 置 1、Gemma γ 置 0； | `adapter.build_ln_fuse_map` / `build_rotate_pairs` |

---

## 包结构

```text
preprocess/
  pipeline.py              # 通用入口：load recipe → adapter → transforms → save
  io.py                    # WeightStore
  calib.py                 # 读取 --act-stats（acts.pt）
  adapters/
    base.py                # ModelAdapter 接口
    qwen3_8_flash_next.py  # Qwen3.8-Flash-Next (qwen4_exp)
    deepseek_v3.py         # DeepSeek-V3 (MLA MoE structure glue)
    __init__.py            # ADAPTER_REGISTRY
  unpack/
    shard_io.py            # 通用 safetensors 写盘 / rank manifest
    fp8.py                 # DeepSeek block FP8 → BF16（1:1 分片，可多卡）
  transforms/
    absorb.py              # 通用吸收
    quarot.py / flex_*.py  # 通用 transform（只调 adapter）
    hadamard.py
  recipes/
    qwen3_8_flash_next.yaml   # Qwen3.8-Flash-Next
    deepseek_v3.yaml          # DeepSeek-V3
```

---

## 示例：DeepSeek-V3

```bash
python3 main.py --alg preprocess \
  --model /public_model/deepseek-ai/DeepSeek-V3 \
  --save-dir /module/DeepSeek-V3-preprocessed \
  --recipe preprocess/recipes/deepseek_v3.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8
```

说明：官方权重是 block FP8，所以流水线会先按 `--preprocess-gpus` 多卡解包成bf16到 `{save-dir}.bf16_staging`，再 8 卡按层做 QuaRot / AWQ / Smooth Quant。

---

## 示例：Qwen3.8-Flash-Next

```bash
python3 main.py --alg preprocess \
  --model /parastor/opendas/DL_DATA/llm-module/Qwen/Qwen3.8-Flash-Next \
  --save-dir /path/Qwen3.8-Flash-Next-preprocessed \
  --recipe preprocess/recipes/qwen3_8_flash_next.yaml \
  --transforms flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8
```

说明：权重已是 BF16，所以无需解包操作。
`hc_norm` 是 Gemma（`y = rms(x) * (1+γ)`）：mix 里既有 Linear 门控，又有 `gates * hyper_input_normed` 逐元素乘，因此不对 `hc_norm` 做 fuse+γ=0。
残差本身就是 4×H 的 hyper-connection，对角 `(1+γ)` 与 Hadamard 不对易，故默认不做 QuaRot 旋转。
默认 transforms：`flex_awq_ssz` + `flex_smooth_quant`

---




---

## 接入新模型（checklist）

1. **新增** `preprocess/adapters/<model>.py`，实现 `ModelAdapter`：
   - `build_ln_fuse_map` / `build_rotate_pairs`（若不用 QuaRot 可返回空表）
   - `resolve_up_down`（若有 UpDown Flex）
2. 在 `adapters/__init__.py` 的 `ADAPTER_REGISTRY` 注册
3. **新增** `preprocess/recipes/<model>.yaml`：`adapter: <name>` + 各 transform 的 `include` 种子
4. 用小层数 smoke（recipe `quarot.max_layers`）验证写出目录可被下游 PTQ 加载

模板骨架：

```python
# preprocess/adapters/my_model.py
from preprocess.adapters.base import ModelAdapter

class MyModelAdapter(ModelAdapter):
    name = "my_model"

    def build_ln_fuse_map(self, cfg, num_layers=None):
        ...

    def build_rotate_pairs(self, cfg, *, block_size=32, num_layers=None, num_experts=None):
        return [], {}

    def resolve_up_down(self, up_module: str):
        # e.g. "*.up_proj" -> "*.down_proj"
        ...
```

```yaml
# preprocess/recipes/my_model.yaml
adapter: my_model
transforms: [flex_smooth_quant]
flex_smooth_quant:
  include: ["model.layers.*.mlp.up_proj"]
```

---

## 验收

- 预处理目录可被现有 `slimquant_ptq` 直接量化（`--from-preprocessed`）
- 开关 preprocess 两套产物可对比 PPL
- 换模型只需新 adapter + recipe，**不改** absorb / pipeline / transform 核心
