# 独立预处理模块

预处理与量化解耦：先写出**仍是 BF16/FP16 的 HF 目录**（旋转 / smooth scale 已吸收进 `.weight`），再交给 `slimquant_ptq` / `model_free_ptq` / MoE-Quant。

## 设计目标（不是「一个 yaml 跑通全球模型」）

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

**禁止**：把别的模型权重丢进 `kimi_k3.yaml` 期望跑通。换模型 = 新 adapter + 新 recipe。

---

## 流程

```text
原始 MXFP4 / 浮点权重
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

约束：

1. 产出必须是浮点 `.weight`，不写 INT4/INT8 / `compression_config`
2. Smooth / Flex 的 scale 吸收进权重（及相邻 Linear/Norm）
3. Kimi MXFP4：默认 `unpack_mxfp4: true`，在 **单进程流式** 写出 `{save-dir}.bf16_staging` 后再跑多卡 QuaRot/Flex（避免把整网 BF16 堆进内存导致 mmap OOM）。输入已是 BF16 时可设 `unpack_mxfp4: false`
4. `--save-dir` 必须与 `--model` 不同（禁止原地覆盖）
5. recipe 必须声明 `adapter:`（或可被 `model_type` 弱自动识别）

**磁盘提示**：staging + 最终目录都需要很大空间（往往数 T）；可在 unpack 成功后自行删 staging。

### 子图分工（吸收语义）

| Transform | 子图 | 吸收规则 | 模块名从哪来 |
|-----------|------|----------|--------------|
| `flex_awq_ssz` | UpDown | `W_up /= s`（out）、`W_down *= s`（in）；融合 3D 专家只改 `gate_up` 的 up 半 | recipe `include` 种子 + `adapter.resolve_up_down` |
| `flex_smooth_quant` | NormLinear | 全部下游 `W *= s`（in）。RMSNorm：`γ /= s`；Gemma：`γ = (1+γ)/s - 1` | recipe 种子 + `adapter.build_ln_fuse_map` |
| `quarot` | Hadamard + LN→Linear | 吸收成功后 RMSNorm γ 置 1、Gemma γ 置 0；2D/3D 都可旋；左旋同步 bias | `adapter.build_ln_fuse_map` / `build_rotate_pairs` |

**禁止**把 UpDown 种子（如 `experts.*.w3`）放进 `flex_smooth_quant`：无 Norm 前驱时 orphan 缩放会改网络。

---

## 包结构

```text
preprocess/
  pipeline.py              # 通用入口：load recipe → adapter → transforms → save
  io.py                    # WeightStore
  calib.py                 # 读取 --act-stats（acts.pt）
  collect_acts_kimi_k3.py  # Kimi-K3 采集 acts.pt 样例（单进程 hook）
  adapters/
    base.py                # ModelAdapter 接口
    kimi_k3.py             # Kimi-K3
    qwen3_8_flash_next.py  # Qwen3.8-Flash-Next (qwen4_exp)
    deepseek_v3.py         # DeepSeek-V3 (MLA MoE + FP8 unpack)
    __init__.py            # ADAPTER_REGISTRY
  transforms/
    absorb.py              # 通用吸收
    quarot.py / flex_*.py  # 通用 transform（只调 adapter）
    hadamard.py
  recipes/
    kimi_k3.yaml              # Kimi-K3
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

说明：官方权重是 **block FP8**（`e4m3` + `weight_scale_inv`，128×128）。流水线会先单进程流式解包到 `{save-dir}.bf16_staging`（约 1.3T+），再 8 卡按层做 QuaRot / Flex。Adapter 覆盖 MLA（`q_a/q_b`、`kv_a/kv_b`）、dense MLP（`first_k_dense_replace`）、逐专家 `experts.{i}.*`、`shared_experts`，以及 `layers.{num_hidden_layers}` 的 MTP（`shared_head`；`enorm`/`hnorm`→`eh_proj` 不做 Norm fuse，因此 **不旋转** `eh_proj`）。缺 `weight_scale_inv` 时解包会直接报错，不会把「半解包」目录标成完成。解包后 `config.quantization_config` 会去掉。磁盘需预留 staging + 最终目录空间。

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

说明：权重已是 BF16（`unpack_mxfp4: false`）。`hc_norm` 是 Gemma（`y = rms(x) * (1+γ)`）：mix 里既有 Linear 门控，又有 `gates * hyper_input_normed` 逐元素乘，因此 **不对 `hc_norm` 做 fuse+γ=0**。残差本身就是 4×H 的 hyper-connection，对角 `(1+γ)` 与 Hadamard 不对易，故 **默认不做 QuaRot 旋转**（adapter 返回空 rotate map）。默认 transforms：`flex_awq_ssz` + `flex_smooth_quant`（Smooth 仅 MTP `pre_fc_norm_*`；层内 `hc_norm` 默认关掉做对照）。Flex AWQ 覆盖 `shared_expert`（2D）与路由专家（`gate_up` 后一半为 up）。

---

## 示例：Kimi-K3

### 1. Recipe（`recipes/kimi_k3.yaml`）

只写**选择**，不写架构表：

```yaml
adapter: kimi_k3          # → preprocess.adapters.kimi_k3.KimiK3Adapter

transforms:
  - quarot
  - flex_awq_ssz
  - flex_smooth_quant

flex_awq_ssz:
  include:
    - language_model.model.layers.*.block_sparse_moe.experts.*.w3

flex_smooth_quant:
  include:
    # 种子；完整消费者由 adapter LN map 展开
    - language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj
    - language_model.model.layers.0.mlp.up_proj
    - language_model.model.layers.*.block_sparse_moe.routed_expert_up_proj
```

### 2. Adapter（`adapters/kimi_k3.py`）里有什么

| 方法 | Kimi-K3 含义 |
|------|----------------|
| `build_ln_fuse_map` | 如 `post_attention_layernorm → [gate, shared gate/up, routed_expert_down_proj]` |
| `build_rotate_pairs` | hidden / q_lora / kv_lora / latent MoE 的左右旋模块表 |
| `resolve_up_down` | `experts.*.w3 → w2`；`*.up_proj → *.down_proj` |
| `resolve_norm_linear` | 默认查 LN map（Smooth 与 QuaRot LN fuse **同一张表**） |

子图分工要点：

- AWQ 只挂 routed expert 的 **UpDown**（`w3`），不重复挂到 Smooth
- Smooth 挂 **NormLinear** 种子；共享 Norm 的全部下游由 adapter 补全
- Latent MoE：`routed_expert_norm → routed_expert_up_proj`（`latent_moe_use_norm` 时）

### 3. CLI

```bash
# 单卡 / CPU
python3 main.py --alg preprocess \
  --model /path/Kimi-K3 \
  --save-dir /path/Kimi-K3-preprocessed \
  --recipe preprocess/recipes/kimi_k3.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  [--preprocess-device cuda:0] [--act-stats /path/acts.pt]

# 多卡按层并行写权重（非 MoE EP；每卡处理 layers 的子集）
python3 main.py --alg preprocess \
  --model /path/Kimi-K3 \
  --save-dir /path/Kimi-K3-preprocessed \
  --recipe preprocess/recipes/kimi_k3.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8

python3 main.py --alg slimquant_ptq \
  --model /path/Kimi-K3-preprocessed --scheme W4A8 \
  --save-dir /path/out --from-preprocessed
```

`--preprocess-gpus N`（N>1）用 `torch.multiprocessing` 起 N 个进程，层号 `i % N == rank` 分到各卡，各 rank 写出 dirty 分片后由主进程合并落盘。这与 MoE 专家并行（EP）无关，也不做在线采 act。

Flex 默认 `use_calib_acts: false`（权重侧 act proxy）。要用真实校准激活，见下一节采集 `acts.pt`，再设 recipe `use_calib_acts: true` 并传 `--act-stats`。

### 4. 采集 `acts.pt`（样例）

#### `acts.pt` 是做什么的

`acts.pt` 是 **离线校准激活统计**：对选定 Linear（或其 Norm）在校准文本上前向时，把**输入激活**按模块名存成文件，供后续 `flex_awq_ssz` / `flex_smooth_quant` 使用。

| 环节 | 不用 `acts.pt`（默认） | 使用 `acts.pt` |
|------|------------------------|----------------|
| 开关 | recipe `use_calib_acts: false` | `use_calib_acts: true` + CLI `--act-stats /path/acts.pt` |
| 通道画像 `a` 从哪来 | 权重侧 proxy（如 `|W_down|` / `|W|` 列 max） | 真实校准激活的 per-channel 幅度（再 `amax` 等） |
| 搜 `alpha` / `(α,β)` | 仍可网格搜，但尖刺位置是「猜」的 | 按真实激活尖刺通道定 `s = a^α …`，更贴近推理 |

具体用途：

1. **`flex_awq_ssz`**：用 down（或 up）入口激活构造中间维画像，搜 `α`，再吸进 `W_up/W_down`。  
2. **`flex_smooth_quant`**：用 Norm 出口或某个消费者入口激活构造 Norm 维画像，搜 `(α,β)`，再吸进全部下游 Linear 与 γ。  
3. **不参与** QuaRot（旋转矩阵与校准 token 无关），也不写入最终 INT4/INT8；只影响预处理后的**浮点**权重形状。

没有 `acts.pt` 时流水线仍可跑通（默认 proxy）；要压真实激活 outlier、对齐「activation-aware」效果时再采。

| 文件 | 作用 |
|------|------|
| [`preprocess/collect_acts_kimi_k3.py`](../preprocess/collect_acts_kimi_k3.py) | 加载 HF 模型，对目标 Linear 挂 `forward_hook`，用校准文本前向，写出 `acts.pt` |
| [`preprocess/calib.py`](../preprocess/calib.py) | preprocess 运行时通过 `--act-stats` **读取**该文件（不负责采集） |

**输出格式**：`torch.save({module_name: Tensor, ...}, acts.pt)`，key 为模块名（无 `.weight`），value 为 float CPU 张量。默认 `--reduce channel_max` 存 per-channel absmax `[K]`；`--reduce raw` 存截断后的 `[N,K]`。加载时 `[K]` 会规范成 `[1,K]`，Flex 搜索按样本维截断后再 `amax(dim=0)`，不会把通道数误当成 batch 截断。

默认 hook 与 recipe Flex 对齐：`experts.*.w2`（AWQ 用 down 输入）、`shared_experts.up_proj` / `mlp.up_proj` / `routed_expert_up_proj`。

```bash
cd one-click-quant
# 建议先用 Mini / 已解包浮点；全量 Kimi 需本机能 load
python3 preprocess/collect_acts_kimi_k3.py \
  --model /path/Kimi-K3-Mini \
  --out /path/acts.pt \
  --device cuda:0 \
  --max-layers 2 --max-experts 8

# recipe 中 flex_* .use_calib_acts: true
python3 main.py --alg preprocess \
  --model /path/Kimi-K3 \
  --save-dir /path/Kimi-K3-preprocessed \
  --recipe preprocess/recipes/kimi_k3.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  --act-stats /path/acts.pt
```

**限制（刻意不做）**：无多卡 EP、跨卡 `sync_act_stats`、多模态图像校准。全量真校准可自行扩展本样例。MoE 未路由到的专家可能采不到激活。

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
