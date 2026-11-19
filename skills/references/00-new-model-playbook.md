# 模型方案入口：先判适配 → 已适配可选 / 客户自提评估 / 未适配走新模型流程

> **何时用**：客户给出一个 HF 模型目录（或明确模型路径），要量化、要方案、或不确定本仓库是否已适配。
> **权威对照**：能力表见 [`../README.md`](../README.md)；执行细节见 `02`/`03`/`10`–`13`。
> **本阶段不做**：BF16 基线 PPL / 精度门禁（留给后续补强 `04`/`SKILL`）。产出的是**工程可行结论**，不是精度承诺。

## Agent 硬规则

1. **必须先读客户给出的真实模型目录**（`config.json` + 权重索引/分片），禁止只凭模型名字猜。
2. **必须先判定「已适配 / 未适配」**，再分支；不得跳过判定直接开跑。
3. 未得到客户确认前：**不得**开跑量化 / 不得擅自大规模改核心逻辑。
4. 「可行」= **当前仓库代码已有入口，或文档化可适配路径**；不要列理论上的任意 PTQ。
5. **MoE-Quant 永不推荐 / 永不判为可行 W4A8**（`--bits 4 --activation-bits 8` 在 pack 阶段硬失败）。要 W4A8 → `slimquant_ptq`（或 `examples/` 混合脚本）。
6. 两套 adapter 注册表独立：`preprocess/adapters/` ≠ `utils/MoE-Quant/src/models/registry.py`。

## 总流程（每次客户给模型都走）

```text
① 取证（阶段 A）
② 判定适配状态（阶段 S）
        ├─ 已适配 ──→ 列出「已适配方案」给客户选（阶段 T）
        │                 ├─ 客户选已适配方案 → 确认后进 02/03 执行
        │                 └─ 客户不选、自提方案 → 评估自提方案（阶段 U）
        │                         ├─ 可行 → 说明差异/风险，确认后执行（或先小改 ignore/recipe）
        │                         └─ 不可行 / 需大改 → 说明原因 + 给出替代（可回到已适配列表或走适配清单）
        └─ 未适配 ──→ 新模型全流程（阶段 B→C→D→E）：可行列表 → preprocess 建议 → 主推荐 → 确认后适配/执行
```

---

## 阶段 A：取证包（自行读取）

### A.1 必读文件

| 来源 | 提取 |
|------|------|
| `<model>/config.json` | `model_type`、`architectures`、`hidden_size`、`num_hidden_layers`、MoE 字段（`num_experts` / `n_routed_experts` / `decoder_sparse_step` / `moe_intermediate_size`）、MLA/MTP/VL 线索（`q_lora_rank`、`num_nextn_predict_layers`、`vision_*`、嵌套 `text_config`） |
| `model.safetensors.index.json` 或单分片 `model*.safetensors` | 键名模式抽样；是否存在 `weight_scale_inv` / `weight_packed` / per-expert `experts.<i>.*` / fused `experts.gate_up_proj` |
| 抽样 1–2 个张量（可用 `tools/model_xray.py --output_dir <非模型目录>`） | dtype、大致 shape；**不要**把 xray JSON 写进模型目录（会干扰 slimquant） |
| 仓库对照（以代码为准） | preprocess `ADAPTER_REGISTRY` + `preprocess/recipes/`；MoE-Quant `_ADAPTER_CLASSES`；`templates/quant_templates.yaml`；`templates/slimquant_ptq.py` 的 `_ARCH_IGNORE`；`examples/<family>/`；[`../README.md`](../README.md) 能力表 |

### A.2 结构标签（打勾用）

- 权重：`bf16_fp16` / `block_fp8` / `mxfp4` / `already_quantized`
- 拓扑：`dense` / `moe` / `mla` / `linear_attn` / `mtp` / `vl`
- expert 落盘：`per_expert_2d` / `fused_3d` / `unknown`
- 仓库状态：`has_preprocess` / `has_moe_quant` / `has_scheme_yaml` / `has_arch_ignore` / `has_examples` / `none`

### A.3 取证摘要（客户可读，5–10 行）

必须包含：路径、架构、规模（层×hidden×experts）、权重形态、仓库命中的入口（逐条列出有/无）。

---

## 阶段 S：判定是否「已适配」

### S.1 判据（满足任一条专用入口 → 该路径算「已适配」；至少一条专用入口 → 模型算「已适配」）

| 专用入口 | 判定方式 |
|----------|----------|
| Preprocess | `model_type` / arch 能命中 `ADAPTER_REGISTRY`，或存在对应 `preprocess/recipes/<name>.yaml` |
| MoE-Quant | `get_model_adapter(config)` 不抛错 / `_ADAPTER_CLASSES` 中有 `matches` |
| 定制 scheme | `quant_templates.yaml` 有该 `architectures`（及尺寸签名，若 YAML 要求精确签名） |
| SlimQuant 专用 ignore | `_ARCH_IGNORE` 有该 arch（及签名）；**仅有 `_DEFAULT_IGNORE` 不算专用适配**，但 MoE+`.mlp.experts.` 仍可标「通用 slimquant 可试」 |
| examples | `examples/<family>/` 有明确对该模型/族的脚本或 README 条目 |

> **「已适配」≠ 万能**：只表示仓库有**至少一条**对该结构的专用或文档化可跑路径。  
> 通用可试路径（`FP8_DYNAMIC` / `W8A8` / `model_free_ptq` / `FP8_TO_BF16`）**单独列出**，标成「通用可试」，不要假装成「本模型专用适配」。

### S.2 对照命令

```bash
python3 -c "from preprocess.adapters import ADAPTER_REGISTRY as R; print(sorted(set(R)))"
ls preprocess/recipes/

grep -n "_ADAPTER_CLASSES" -A20 utils/MoE-Quant/src/models/registry.py
# 或：用 AutoConfig 加载后 from src.models import get_model_adapter

# 读 templates/quant_templates.yaml
grep -n "_ARCH_IGNORE" -A40 templates/slimquant_ptq.py | head -60
ls examples/
```

### S.3 分支

- **已适配**（≥1 条专用入口）→ 阶段 T  
- **未适配**（无专用入口；最多只有通用可试）→ 跳过 T，直接走阶段 B→C→D（新模型流程）

输出一句给客户：`适配结论：已适配（命中：…）| 未适配`。

---

## 阶段 T：已适配 — 给出可选方案（等客户选）

### T.1 列出「已适配方案」菜单

只列阶段 S 命中的专用路径 + 与之匹配的通用可试（若适用）。每条写清：

- 方案名（如 preprocess + slimquant W4A8 / MoE-Quant W8A8 / YAML scheme …）
- 是否需要 preprocess、对应 recipe
- 命令骨架、预期 `quant_method` / format
- 限制（如 experts-only、相对路径、无 W4A8 等）

### T.2 交付模板（已适配 · 待选择）

```text
【取证摘要】
- …

【适配结论】已适配
- 命中入口：preprocess=… / MoE-Quant=… / scheme YAML=… / slimquant _ARCH_IGNORE=… / examples=…

【已适配方案（请选择编号）】
1. …
2. …
（可选）通用可试：…

【说明】
- 请回复方案编号以继续执行；或给出你希望的自定义方案（alg/scheme/是否 preprocess/位宽等），将进入可行性评估。
- 确认前不执行量化。精度基线对比：本阶段不做。
```

### T.3 客户行为

| 客户做法 | Agent 行为 |
|----------|------------|
| 选择已适配方案编号 | 确认细节（产物目录、卡数等）后进 `02`/`03`/`10`–`13` 执行 |
| 不选、提出自己的方案 | 进入阶段 U（自提方案评估） |
| 既不选也不提方案 | 追问一次；仍无答复则停，不开跑 |

---

## 阶段 U：客户自提方案 — 可行性评估（借鉴新模型方法）

在**已适配**前提下，客户坚持自己的路径（例如：已有 MoE-Quant W8A8，却要求 MoE-Quant W4A8；或要求未挂上的 scheme）。

### U.1 评估维度（逐条写结论）

1. **目标格式是否被本仓库支持**（尤其禁止 MoE-Quant W4A8）  
2. **与当前模型结构是否匹配**（MoE / fused experts / FP8 / 相对路径要求等）  
3. **是否已有代码入口**；若无，属于「小改」（ignore/recipe/YAML 签名）还是「大改」（新 adapter / 新 examples）  
4. **是否建议 / 需要 preprocess**（复用阶段 C 规则）  
5. **相对已适配菜单的差异与风险**（静默降级、非 expert 仍为 BF16、资源需求等）

### U.2 评估结论三态

| 结论 | 含义 | 下一步 |
|------|------|--------|
| **可行（可直接跑）** | 入口已在，参数组合合法 | 给出命令骨架，确认后执行 |
| **可行（需小改/适配）** | 文档化扩展点可完成 | 列出改哪些文件（`10`/`11`/`13`），确认后改再跑 |
| **不可行** | 硬约束或无合理扩展路径 | 说明原因；**回流**已适配菜单或给出最接近的替代主方案 |

### U.3 交付模板（自提方案评估）

```text
【客户自提方案】
- …

【可行性】可行（可直接跑）| 可行（需小改/适配）| 不可行
- 理由（逐条）：
- 与已适配方案的差异 / 风险：
- 是否 preprocess：建议 | 可选 | 不建议 | 有限 — 理由…

【若不可行：替代建议】
- 回到已适配方案：编号 …
- 或最接近的可行动作：…

【下一步】
- 请确认：执行自提（及所需改动）/ 改选已适配编号 / 停止。确认前不执行量化。
```

---

## 阶段 B：未适配 — 可行量化方案列表

对每个候选标三态之一：**可直接跑** / **需适配后跑** / **明确不支持**。

每条至少写：路径名、前置条件、预期 `quant_method`/format、主要限制。

### B.1 候选池（按取证裁剪，无信号则删除整行）

| ID | 路径 | 列入条件 | 预期产物要点 | 常见限制 |
|----|------|----------|--------------|----------|
| G1 | `--scheme FP8_DYNAMIC` / `W8A8`（默认 alg） | `bf16_fp16` 或可先解包；transformers 能加载 | compressed-tensors 类 | 结构奇异可能加载失败 |
| G2 | `--alg model_free_ptq` | 同上；要免校准 | 同上 | `--model` **必须相对路径**（见 `03` 3.0） |
| G3 | `--scheme FP8_TO_BF16` | `block_fp8` / 需解包 | 浮点权重 | 只是解包不是量化 |
| S1 | `--alg slimquant_ptq --scheme W4A8` | MoE 且 expert 键可匹配 `.mlp.experts.`（含将 fused 拆开） | `quant_method: slimquant_w4a8`；routed experts INT4，其它多为 INT8 | 非 GPTQ；相对路径；无专用 `_ARCH_IGNORE` 时走默认 ignore |
| M1 | MoE-Quant GPTQ `bits=4` → pack `activation-bits=16` | MoE；registry 已有或可按 `13` 适配 | W4A16 / `pack-quantized` | 通常 `--quantize_only_experts`；非 expert 保持浮点 |
| M2 | MoE-Quant GPTQ `bits=8` → pack `activation-bits=8` | 同上，且 adapter 允许 bits=8 | W8A8 / `int-quantized` | **禁止** bits=4+act=8 |
| M∅ | MoE-Quant W4A8 | — | — | **明确不支持**，必须写进「不可行」 |
| Y1 | `quant_templates.yaml` 定制 scheme | YAML 有该 arch/签名 | 以该 scheme 为准 | 签名（hidden×layers）要对上 |
| E1 | `examples/<family>/` 脚本 | 目录下有对应该族的脚本/README | 以脚本为准 | 非统一 `main.py` CLI |
| P→Q | preprocess 后再走 S1/G1/… | 见阶段 C「建议」 | 浮点 preprocess 目录 + 下游量化产物 | 必须 `--from-preprocessed`（下游）且专用 recipe |

---

## 阶段 C：是否推荐 preprocess（未适配主流程；自提评估也可复用）

| 信号 | 倾向 |
|------|------|
| 客户目标含 W4* 且口头在意掉点 | **建议** |
| `moe` + 大规模 experts / 难量化结构线索 | **建议** |
| 已有同族 preprocess adapter/recipe（`has_preprocess`） | **强烈建议**（复用或小改） |
| 只要快速 W8A8/FP8、掉点要求松 | **可选 / 不建议** |
| `already_quantized` 或只要解包（G3） | **不建议** |
| 已知与 QuaRot/残差不兼容的结构（如部分 HC + Gemma norm） | **有限 preprocess**（recipe 里关 quarot / 少 fuse）或跳过 |

输出必须是一句：**建议 / 可选 / 不建议 / 有限 preprocess** + 理由 +  
`recipe` 状态（已有文件名 / 需新建 → 指向 `10-ext-preprocess.md`）。

---

## 阶段 D：未适配 — 综合推荐主方案

按优先级叙述（不必数值打分）：

1. **目标格式一票否决**（客户要 W4A8 就不能推 MoE-Quant）
2. **适配成本**：已有入口 > 改 ignore/recipe > 新建 adapter > 新 examples 脚本
3. **工程风险**：通用路径稳但粗；专用路径准但要适配
4. **资源**：slimquant 偏单卡流式；MoE-Quant 需多卡 EP（`num_experts % nproc == 0`）
5. **preprocess 性价比**：若阶段 C 为「建议」，主方案写成 **preprocess → 量化** 两段

主方案必须带：**命令骨架**、产物目录约定、若需适配则列出要改的文件。另给 **1 个备选**。

### 交付模板（未适配 · 新模型）

```text
【取证摘要】
- …

【适配结论】未适配
- 命中入口：无专用（或仅通用可试：…）

【可行方案】
1. …（可直接跑 / 需适配 / 不支持）— 限制…
2. …
（必须单独列出明确不支持项，如 MoE-Quant W4A8）

【是否 preprocess】
- 结论：建议 | 可选 | 不建议 | 有限 preprocess
- 理由：
- recipe：已有 <file> | 需新建（见 10-ext-preprocess.md）

【推荐主方案】
- 路径：
- 命令骨架：
- 产物目录：
- 若需适配：改哪些文件 / 注册表

【备选】
- …

【下一步】
- 请确认主方案（或指定备选）。确认前不执行量化。
- 精度基线对比：本阶段不做。
```

---

## 阶段 E：客户确认之后

1. **需适配**：只改文档指定的扩展点；preprocess 先 `check_recipe.py`；禁止改 absorb/pipeline/transform 核心绕过。
2. **执行**：环境见 `01`；mini/解包见 `02`；命令见 `03`；MoE-Quant 见 `13` + `utils/MoE-Quant/README.md`。
3. **契约验收**：量化产物 `check_artifact.py`；preprocess 浮点按 `10`（无量化配置键）。
4. 本 playbook **仍不要求**精度对比；若客户主动要 PPL，再按 `04` 做，并标明「非本阶段门禁」。

---

## 客户侧示例 prompt

**通用（先判适配）：**

```text
读取 <模型绝对路径>，按 one-click-quant skills 的 00-new-model-playbook：
先判断该模型在仓库里是否已适配；若已适配，列出已适配方案供我选择；
若我另行提出方案，请评估是否可行。若未适配，按新模型流程给出可行方案、是否 preprocess 与主推荐。
先不要执行量化。
```

**已知未适配 / 只要新模型推荐：**

```text
读取 <模型绝对路径> 的 config 与权重索引，按 00 新模型分支：
列出可实现方案（含不可行）、是否建议 preprocess、推荐主方案。先不要执行量化。
```

---

## 与其它 reference 的分工

| 文档 | 职责 |
|------|------|
| **本文 `00`** | 任意模型入口：判适配 → 已适配菜单 / 自提评估 / 未适配推荐 |
| `02` | 确认方案后的解包 / mini / 结构覆盖 |
| `03` | 确认方案后的具体命令与路径坑 |
| `10`/`11`/`13` | 「需适配」时怎么改代码 |
| `04`/`05` | 契约与精度排查（精度门禁后续加强，不阻塞本 playbook） |
