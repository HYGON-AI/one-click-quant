# 预处理理论基础

本文档说明 `one-click-quant/preprocess` 中 float 预处理（QuaRot / Flex Smooth / Flex AWQ-SSZ）的理论基础。实现细节见 [preprocess.md](./preprocess.md)；工程调用见仓库 README。

---

## 第一章 QuaRot：原理、条件与模型适配

### 1.1 要解决什么问题

激活里常有少数通道特别大（outlier）。按通道 / token 量化时，scale 往往由最大值决定：尖刺把整行 `S` 拉爆，其它数被挤成 0，量化误差很大。

**QuaRot（Quantization with Rotation）** 用正交旋转把尖刺能量摊到多个通道上，使各通道 `|max|` 更接近，再去做后续 AWQ / Smooth / INT4·INT8 量化。旋转矩阵 `Q` 在预处理时生成一次并吸进权重；推理图仍是原来的 Linear，热路径不额外乘 `×Q` GEMM。

---

### 1.2 计算原理

#### 1.2.1 正交不变式

对激活与权重乘一对正交矩阵（Hadamard 一类，`Q Qᵀ = I`）：

```text
X' = X Q
W' = Qᵀ W
⇒  X' W' = X W
```

浮点结果不变。通道离群是少数 hidden 维特别大；`X' = X Q` 在特征维上线性混合：新第 `j` 通道是旧各维按 `Q[:, j]` 加权求和。Hadamard / 正交 `Q` 的每一列把很多旧维掺在一起，尖刺按系数分到多个新通道。正交保 L2 范数，能量几乎不变，但不再集中在某一个 max 通道。

- `block_size: 32`：每 32 维一块小 `Q`，块内搅拌、块间不搅（也可用整维 Hadamard，视 size/mode）。
- **`W'` 不随 token 变**：`Q` 与某个 `X` 无关，写进 checkpoint 后冻住；每个新 token 的 `X` 会变，但推理始终用同一份已旋转（再量化）的权重。
- **不要把 `Q Qᵀ W` 融进同一块权重**：`Q Qᵀ = I` 会变回原 `W`，激活仍是尖刺，等于没转。恒等式必须拆开：`(X Q)(Qᵀ W) = X W`。量化需要变均匀的是激活侧 `X Q`，本层权重只配套改成 `Qᵀ W`。

#### 1.2.2 离线吸收：推理图里没有单独的 `×Q`

显式做法是每个 token 先算 `X' = X Q` 再用 `W'`，多一次 GEMM。离线 QuaRot 把 `Q` 拆到**相邻两层**权重里：

原来：

```text
X_prev = H · W_prev    # 上一跳输出
Y      = X_prev · W    # 本层输出
```

量化预处理时改：

```text
W_prev_fused ← W_prev Q      # 上一层右乘：写出侧吸进 Q
W_fused      ← Qᵀ W          # 本层左乘：输入侧吸进 Qᵀ
```

推理仍是两次 Linear：

```text
X_new = H · W_prev_fused = H · W_prev Q = X_prev Q
Y     = X_new · W_fused  = (X_prev Q)(Qᵀ W) = X_prev W
```

与代码对应（权重布局 `[out, in]`，见 `preprocess/transforms/hadamard.py` 的 `rotate_weight`）：

| 侧 | 操作 | 含义 |
|----|------|------|
| 右旋 `right_rotate` | `W ← W @ R` | 旋**输入维**（上一跳写出侧 / 本层读残差） |
| 左旋 `left_rotate` | `W ← Rᵀ @ W` | 旋**输出维**（本层写残差）；有 bias 时同步 `b ← Rᵀ b` |

`optional/quarot.safetensors`（`global_rotation`）只是本次用的 `Q` 存档，用于对齐 / 调试；热路径加载的是已经旋转再量化的权重，**不**拿该文件去乘激活。

#### 1.2.3 第 0 层与残差坐标系

第 0 层前面没有 Decoder Linear：`Q` 吸进 embedding。查表 `X = one_hot · E`，预处理 `E ← E Q`，推理仍是普通 embedding：`X0 = one_hot · (E Q) = X Q`。

中间各层残差一直活在旋转空间：

- **读残差**的层（QKV / `q_a`·`kv_a`、MLP `gate`/`up`、MoE 相关输入等）→ **右乘**
- **写残差**的层（`o_proj`、`down_proj` 等）→ **左乘**
- 最后 `lm_head` 再右乘同一 `Q`，把表示转回可算 logits 的坐标系（或与训练时一致的出口约定）

#### 1.2.4 为何必须先做 LN → Linear fuse

RMSNorm：`y = rms(x) * γ`，再接 `Linear(y) = W y`。等价吸收（按 in_features 逐元素乘）：

```text
W' = W ⊙ γ
```

吸收后把 Norm 的 γ 置为恒等（标准 RMSNorm：`γ → 1`；Gemma：`y = rms(x)*(1+γ)`，则 `W' = W*(1+γ)`，`γ → 0`）。  
原因：对角缩放与 Hadamard **不对易**（`diag·R ≠ R·diag`）。若不先 fuse，残差上仍挂着 γ，离线旋转无法保持数学等价。

实现上两步固定顺序（`preprocess/transforms/quarot.py`）：

1. **LN fuse**：adapter 给 `norm → [linears]`，`W' = W * γ`，再清 γ  
2. **旋转吸收**：adapter 给 `(left_map, right_map)` + 旋转 meta，按右/左旋改权重  

`one-click-quant` 直接改 safetensors 权重，纯离线；推理图不变。

#### 1.2.5 极简数值例子（4 维 hidden，手算串起 1.2.1–1.2.4）

取归一化 4 维 Hadamard（对称正交：`Qᵀ = Q`，`Qᵀ Q = Q² = I`）：

```text
Q = (1/2) * [[ 1,  1,  1,  1],
             [ 1, -1,  1, -1],
             [ 1,  1, -1, -1],
             [ 1, -1, -1,  1]]
```

##### （1）正交不变式：把尖刺摊平（对应 1.2.1）

某个 token 激活：

```text
X = [100, 1, 1, 1]
```

通道 0 是明显离群。旋转 `X' = X Q`：

- 第 0 列：`(100+1+1+1)/2 = 51.5`
- 第 1 列：`(100-1+1-1)/2 = 49.5`
- 第 2 列：`(100+1-1-1)/2 = 49.5`
- 第 3 列：`(100-1-1+1)/2 = 49.5`

```text
X' = [51.5, 49.5, 49.5, 49.5]
```

动态范围从约 `100:1` 变成约 `51.5:49.5 ≈ 1.04:1`。L2 范数几乎不变：

```text
||X||₂ ≈ 100.015
||X'||₂ ≈ 100.015
```

能量没丢，但不再集中在单个通道。

权重侧配套：设本层 `W = I₄`，则 `W' = Qᵀ W = Q`。输出：

```text
Y' = X' W' = (X Q) Qᵀ = X = [100, 1, 1, 1]
```

浮点完全不变。**关键**：不能把 `Q Qᵀ` 合并回单位阵，否则 `W'` 变回 `I`，激活仍是尖刺，等于没转。必须让激活真正变成 `X Q`，权重配套变成 `Qᵀ W`。

##### （2）离线吸收：推理图里没有单独的 `×Q`（对应 1.2.2）

两层 Linear：`X₁ = H W₁`，`Y = X₁ W₂`。

**显式做法**：每 token 先算 `X₁' = X₁ Q`，再用 `W₂' = Qᵀ W₂` → 多一次 GEMM。

**离线吸收**：预处理改权重

```text
W₁_fused = W₁ Q       # 上一层输出侧右乘
W₂_fused = Qᵀ W₂      # 本层输入侧左乘
```

推理仍是两个普通 Linear：

```text
X_new = H W₁_fused = H W₁ Q = X₁ Q
Y     = X_new W₂_fused = (X₁ Q)(Qᵀ W₂) = X₁ W₂
```

权重布局 `[out, in]`：右旋 `W ← W @ R`（旋输入维）；左旋 `W ← Rᵀ @ W`（旋输出维；bias 同步 `b ← Rᵀ b`）。`optional/quarot.safetensors` 只是 `Q` 存档，热路径用已旋转再量化的权重。

##### （3）第 0 层与残差坐标系（对应 1.2.3）

第 0 层前没有 Decoder Linear，`Q` 吸进 embedding：`E ← E Q`。查表：

```text
X₀ = one_hot · (E Q) = X Q
```

例如某 token 的 embedding 行 `Eᵢ = [100, 1, 1, 1]`，预处理后：

```text
Eᵢ' = Eᵢ Q = [51.5, 49.5, 49.5, 49.5]
```

推理查表直接得到旋转后激活，无需再乘 `Q`。中间层残差活在旋转空间：读残差 → 右旋 `W ← W Q`；写残差 → 左旋 `W ← Qᵀ W`；`lm_head` 再右乘同一 `Q` 转回 logits 坐标系。

##### （4）为何必须先做 LN → Linear fuse（对应 1.2.4）

`diag(γ)` 与 Hadamard **不对易**：`diag(γ)·Q ≠ Q·diag(γ)`。残差上还挂着 γ 时，旋转不能简单穿过。

正确：先 `W' = W ⊙ γ`，再 `γ → 1`（Gemma：`W' = W ⊙ (1+γ)`，`γ → 0`），Norm 只做纯 `rms(x)`。

**数值例子。** 设 `x = [100, 1, 1, 1]`，`γ = [2, 1, 1, 1]`。简化 RMS：

```text
rms(x) = x / sqrt(mean(x²)) = x / sqrt(2500.75)
       ≈ [1.9997, 0.02, 0.02, 0.02]
y = rms(x) ⊙ γ ≈ [3.9994, 0.02, 0.02, 0.02]
```

接 `W = I` 时输出即 `y`。

**错误**：先旋转 `x` 得 `[51.5, 49.5, 49.5, 49.5]`，再对旋转后通道乘**原通道**的 γ —— 通道已混合，γ 不再对应，结果与原始 `y` 不等价。

**正确（先 fuse）**：`W' = diag([2,1,1,1])`，Norm 的 γ=1，只输出 `rms(x)`。再旋转：

```text
rms(x)' = rms(x) Q
       ≈ [1.03, 0.99, 0.99, 0.99]   # 非常均匀
```

权重配套：`W'' = Qᵀ W' = Qᵀ diag(γ)`。最终：

```text
rms(x)' · W'' = (rms(x) Q)(Qᵀ diag(γ)) = rms(x) diag(γ) = y
```

与原始一致。**结论**：必须先 LN fuse，消掉与 Hadamard 不对易的对角缩放，旋转才能安全左右吸收。

##### （5）例子小结

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1.2.1 正交不变式 | `X'=XQ`，`W'=QᵀW` | 尖刺摊到多通道，浮点不变 |
| 1.2.2 离线吸收 | 上一跳 `W_prev←W_prev Q`，本层 `W←QᵀW` | 推理无额外 `×Q` GEMM |
| 1.2.3 第 0 层与残差 | embedding `E←EQ`，lm_head 右乘 `Q` | 残差全程在旋转空间，出口转回 |
| 1.2.4 LN fuse | `W←W⊙γ`，`γ→1` | 消掉与 Hadamard 不对易的对角缩放 |

核心：**用正交旋转打散离群值，同时把旋转离线融进权重和 embedding，保证推理图不变、数学等价。**

---

### 1.3 使用条件（何时可以开 QuaRot）

**开 QuaRot，当且仅当同时满足：**

1. **残差路径与 hidden 维一致**  
   普通结构：`x ← x + Attn(...)`、`x ← x + MLP(...)`，残差流维数 = `hidden_size`（或可 block-diag 重复同一 `R` 的整数倍，且无额外对角挂在残差上）。

2. **Norm 是可 fuse 的 RMSNorm**  
   fuse 后 γ→1（或 Gemma γ→0）在数学上等价；Norm 输出不能再被非 Linear 路径复用（例如 `gates * normed` 逐元素乘）。

3. **adapter 能给出完整左右旋成对表**  
   `build_rotate_pairs`：每个旋转空间上，写出侧右旋与读入侧左旋成对，中间只有已 fuse 的 RMS 或恒等。

4. **没有「对角缩放挂在残差上且与 Hadamard 不对易」的结构**  
   典型反例：hyper-connection + 未 fuse 的 Gemma `hc_norm`。

满足四条 → **原则上可用**；仍须按架构手写 fuse/rotate 表（不是换权重目录自动生效）。  
「几个 rot」指的是 **几把旋转矩阵 / 几个正交作用空间**，不是「只旋几个权重」——每把矩阵会乘到许多 Linear 上。

---

### 1.4 DeepSeek-V3：对哪些空间做了 QuaRot

代码：`preprocess/adapters/deepseek_v3.py`（`build_rotate_pairs` / `build_ln_fuse_map`）。

| 旋转矩阵 | 尺寸 | 模式 | 作用通路 | 典型成对模块 |
|----------|------|------|----------|----------------|
| `rot` | `hidden_size` | Hadamard | 主残差流 | embed / lm_head 右旋；`q_a`/`kv_a`、MLP gate·up 右旋；`o_proj`、down 左旋；MTP 只旋 `embed_tokens` / `shared_head.head`（**不旋** `eh_proj`：`enorm`/`hnorm` 双 Norm 无法 fuse，对角 γ 与 Hadamard 不对易） |
| `rot_b_proj` | `q_lora_rank` | Block-Hadamard-Shifted | MLA Q 压缩 | `q_a_proj` **左** ↔ `q_b_proj` **右** |
| `rot_kv` / `rot_kv_block` | `kv_lora_rank` | Hadamard | MLA KV 压缩 | `kv_a` **左**（`block_diag(R, I_rope)`，RoPE 维不动）↔ `kv_b` **右** |

本仓库 preprocess 离线路径**不单独做** V→O 的 `rot_uv`；MLA 的 V 经 `o_proj` 仍由主残差 `rot` 覆盖写出侧。

LN fuse 示例：`input_layernorm → {q_a, kv_a}`，`q_a_layernorm → q_b`，`kv_a_layernorm → kv_b`，`post_attention_layernorm → gate/up`（及专家）。

```text
hidden ──rot──► q_a ──rot_b_proj──► q_b ──► Q
       └──rot──► kv_a ──rot_kv──► kv_b ──► K / V
                                    └──► o_proj ──rot──► hidden
```

---

### 1.5 Kimi-K3：对哪些空间做了 QuaRot

代码：`preprocess/adapters/kimi_k3.py`。

| 旋转矩阵 | 尺寸 | 模式 | 作用通路 | 说明 |
|----------|------|------|----------|------|
| `rot` | `hidden_size` | Hadamard | 主残差流 | `embed_tokens` / `lm_head` 右旋；`mm_projector.rot_proj` 左旋（视觉进同一坐标系）；层内读/写残差的 Linear 成对；可选 attn residual 出口 |
| `rot_b_proj` | `q_lora_rank` | Block-Hadamard-Shifted | MLA Q | 非 KDA 层：`q_a` 左 ↔ `q_b` 右 |
| `rot_kv` / `rot_kv_block` | `kv_lora_rank` | Hadamard | MLA KV | 非 KDA 层：`kv_a` 左（+ `I_rope`）↔ `kv_b` 右 |
| `rot_latent` | `routed_expert_hidden_size` | Hadamard | Latent MoE | `routed_expert_up` / `experts.w1·w3` 右旋 ↔ `routed_expert_down` / `w2` 左旋 |

与 DeepSeek 的差异要点：

- 用 **`rot_latent`** 覆盖 Latent MoE 中间维（DeepSeek adapter 无此空间）。
- **KDA 层**跳过 `rot_b_proj` / `rot_kv`。
- 多模态入口：`mm_projector.rot_proj` 与 `embed_tokens` 共用 `rot`。

---

### 1.6 Qwen3.8-Flash-Next：为什么不能做 QuaRot

Recipe：`preprocess/recipes/qwen3_8_flash_next.yaml`；Adapter：`preprocess/adapters/qwen3_8_flash_next.py`。

默认 transforms **不含** `quarot`，只跑 Flex AWQ / Smooth。即便强制加 `quarot`，`build_rotate_pairs` 也返回**空旋转表**（最多 fuse MTP 的 `pre_fc_norm_embedding → fc_embedding`）。

**根因（同时违反条件 1、2、4）：**

1. **残差不是普通 `hidden`，而是 hyper-connection 的 `hc_count × H`（如 4×2560）**  
   残差流本身是宽 HC 路径，不能按「维数 = hidden、普通 x+attn/mlp」那套全局 `rot` 直接套。

2. **`hc_norm` 是 Gemma：`y = rms(x)*(1+γ)`，且不能 fuse**  
   `GatedResidual.mix` 在 Linear 门控之外还有 `gates * hyper_input_normed` **逐元素乘**，复用了 norm 后的向量。fuse 并把 γ→0 会破坏该支路（先前 dump 出现过 logits collapse）。于是对角 `(1+γ)` **留在残差上**。

3. **对角与 Hadamard 不对易**  
   在未吸收 γ 的残差上做 `X→XR` 再指望权重里 `Rᵀ` 抵消，数学不等价 → **整段不转**。

结论：该模型用 Flex（AWQ-SSZ / Smooth）做 outlier 相关预处理，**不用** QuaRot 旋转。

---

### 1.7 给新模型怎么分析（含 GLM-5.2 示例）

满足四条件只说明「可以设计」；具体要画图、写 adapter。推荐步骤：

#### 步骤

1. **读 `config.json` + 一层前向**  
   标清：残差怎么加、Norm 类型（RMS / Gemma / LayerNorm）、是否 MLA、是否 MoE、是否 MTP / indexer / hyper-connection。

2. **核对四条使用条件**  
   - 残差维是否 = `hidden_size`（或可安全 block-diag）？  
   - 每个 Norm 的输出是否**只**进 Linear 消费者？能否 fuse？  
   - 是否存在挂在残差上的对角缩放 / elementwise 复用？  
   - 任一否 → 该全局路径不能做 offline QuaRot（局部子空间仍可能可旋，需单独证明）。

3. **划分正交作用空间**（有几把 `rot*`）  
   常见：`hidden`、`q_lora_rank`、`kv_lora_rank`、专家中间维等。RoPE 维一律用单位阵，不旋。

4. **为每个空间填左右旋成对表**  
   对每个 Linear：旋转作用在输入 → 右旋；作用在输出 → 左旋。中间算子破坏等价则排除。

5. **写 `build_ln_fuse_map` + `build_rotate_pairs`**  
   小层 smoke（`quarot.max_layers`）对比旋转前后浮点 logits / 中间激活，再全量预处理。

#### 示例：`/parastor/opendas/DL_DATA/llm-models/GLM-5.2`

从 config 可见（节选）：

| 字段 | 值 | 对 QuaRot 的含义 |
|------|-----|------------------|
| `model_type` | `glm_moe_dsa` | GLM MoE + DSA indexer |
| `hidden_size` | 6144 | 主残差 `rot` 尺寸 |
| `q_lora_rank` / `kv_lora_rank` | 2048 / 512 | MLA：`rot_b_proj` / `rot_kv` |
| `v_head_dim` / `qk_rope_head_dim` | 256 / 64 | RoPE 维不旋；V 维可由 `o_proj` 侧残差 `rot` 覆盖 |
| `rms_norm_eps` | 有 | 标准 RMS 路径，可按 DeepSeek adapter 方式 fuse |
| `first_k_dense_replace` | 3 | 前几层 dense MLP，其后 routed MoE |
| `indexer_types` | full/shared 交替 | **额外**读残差的 indexer 线性层，须并入 `rot` / `rot_b_proj` 右旋表 |

**条件判断：** 普通残差 + MLA + RMSNorm fuse，**无** Qwen 式 HC/`hc_norm` → **可以做 QuaRot**（与 DeepSeek-V3 同族）。

**空间与成对表（接入 preprocess 时可仿 `deepseek_v3.py`，并补 indexer）：**

| 空间 | 成对要点 |
|------|----------|
| `rot`（6144） | embed / lm_head；`q_a`/`kv_a`、MLP·专家 gate/up 右旋；`o_proj`/down 左旋；**另**：`indexer.wk`、`indexer.weights_proj` 等读残差模块右旋 |
| `rot_b_proj`（2048） | `q_a` 左 ↔ `q_b` 右；indexer 的 `wq_b` 等 Q 压缩消费者右旋 |
| `rot_kv`（512） | `kv_a` 左（+ `I_rope`）↔ `kv_b` 右 |

若要把 GLM-5.2 接入 `one-click-quant/preprocess`：新增 adapter（仿 `deepseek_v3.py`），补上 indexer 模块名与 MTP 层命名，再在 recipe 里打开 `quarot`。**不要**假设「有 MLA 就零改动复用 DeepSeek 表」——索引器与 checkpoint 前缀必须按实模核对。

#### 决策速查

```text
新模型
  ├─ 残差是 HC / 宽连接，且 Norm 输出被 elementwise 复用？ → 不做全局 QuaRot（如 Qwen3.8-Flash-Next）
  ├─ 普通 x+attn/mlp + 可 fuse RMSNorm？
  │     ├─ 有 MLA？ → rot + rot_b_proj + rot_kv
  │     ├─ 有 Latent MoE 中间维？ → + rot_latent（如 Kimi-K3）
  │     ├─ 有额外读残差头（indexer 等）？ → 并入对应空间的右旋表（如 GLM-5.2）
  │     └─ 写完整 fuse/rotate 表 → smoke → 全量
  └─ 仅局部子空间可证等价？ → 可只旋局部；残差全局仍关
```

---

### 1.8 本章小结

| 问题 | 结论 |
|------|------|
| QuaRot 算什么 | 正交旋转摊平激活 outlier；`X'=X Q`，`W'=Qᵀ W`；离线吸进相邻权重 |
| 何时能开 | 四条件：残差≈hidden、RMS 可 fuse、成对表完整、无残差对角与 R 不对易 |
| DeepSeek-V3 | `rot` / `rot_b_proj` / `rot_kv`（三空间；见 `adapters/deepseek_v3.py`） |
| Kimi-K3 | `rot` / `rot_b_proj` / `rot_kv` / `rot_latent`（四空间；KDA 跳过部分） |
| Qwen3.8-Flash-Next | HC 4×H + 不可 fuse 的 Gemma `hc_norm` → 不能做 |
| 新模型（如 GLM-5.2） | 先判四条件，再按子空间画左右旋；同族可做，须含 indexer |

---

## 第二章 Flex AWQ-SSZ：原理、条件与模型适配

本章对应 transform：`flex_awq_ssz`（`preprocess/transforms/flex_awq_ssz.py`）。它与 QuaRot 正交：QuaRot 换坐标系摊尖刺；AWQ 在 **Up–Down 子图**上按通道做对角缩放，把激活难度挪到权重上。本步只改浮点权重形状，**不**写出 INT4；真正的 INT4/INT8 落盘在后续量化阶段。

---

### 2.1 要解决什么问题

FFN / MoE 专家里，**up 投影之后、down 投影之前**的中间激活往往特别尖（尤其路由专家）。后面若对 down 的输入做动态 INT8、对权重做 INT4：

- 少数中间通道 max 很大 → 激活 scale 被拉爆，其它通道挤成 0；
- 若只压激活、不管权重，浮点就不等价。

**Flex AWQ-SSZ** 在中间维上找一组正数 `s`，做：

```text
X' = X / s          # 尖通道变矮，利于激活量化
W' = W * s          # 同通道权重变胖，利于用更大的 per-channel 权重量化步长消化
⇒  X' W' = X W      # 浮点乘积不变
```

离线时把 `1/s`、`s` **吸进相邻的 up / down 权重**，推理仍是原来的两次 Linear（+ 中间激活函数），没有显式的 `/s`。

与 QuaRot 对比：

| | QuaRot | Flex AWQ-SSZ |
|--|--------|----------------|
| 变换 | 正交旋转（混通道） | 对角缩放（不混通道） |
| 作用面 | 残差 / MLA 等多空间 | 仅 **Up–Down** 中间维 |
| 搜索 | 固定 Hadamard（seed） | 网格搜 `alpha`，用假量化 MSE 打分 |
| 是否写 INT | 否（浮点预处理） | 否（浮点预处理） |

---

### 2.2 计算原理

思路是 `Y = (X / s) · (s · W)`：激活除以 `s`（尖刺变矮），权重乘 `s`（变胖），浮点仍相等。子图类型 **Up–Down** 表示缩放可以吸进相邻的 up / down，整体仍等价。以 Kimi 为例，本步常只作用在专家 `w3`（up 侧）：搜 `alpha` 得到通道缩放 `s`，再用 INT4 SSZ 假量化选 MSE 最小的 `alpha`（`beta` 固定为 0）。专家里 up/gate 激活往往更尖，所以先打 w3；w1/w2 这一步还不动。

#### 2.2.1 为什么要 `/s` 和 `×s`

对每个中间通道一个正数 `s_j`：

```text
X'_j = X_j / s_j          # 激活第 j 通道
W' 的第 j 个输入通道 × s_j
⇒  X' W' = X W
```

`s_j > 1` 时该通道激活变矮、权重变胖。动态 INT8 的 scale 常被少数尖通道的 max 拉爆，其它维挤成 0；把尖通道除掉较大的 `s`，激活更好量化。权重变胖由后续 per-channel INT4 单独用更大的 `S_w` 消化。

- QuaRot：换坐标系**混通道**摊尖刺。
- 本步：按通道**伸缩**，专门打 FFN Up–Down。只改浮点形状，不把权重写成 INT4，让后面量化更好活。

离线吸收（`absorb_up_down`；权重布局 `[out, in]`）：

```text
W_up   ← W_up  / s     # 沿 out_features（up 写出侧吸进 1/s）
W_down ← W_down * s    # 沿 in_features（down 读入侧吸进 s）
```

中间再经 SiLU / SwiGLU 等时，工程上仍把 `s` 吸在 up 写出 / down 读入上。假量化打分打在 **down** 上（误差出在整条 FFN 出口）；融合后中间维上 `/s` 与 `×s` 抵消，推理不再显式除 `s`。

融合 3D 专家（`gate_up_proj` 形如 `[E, 2I, H]`）时：只除 **up 半**（后一半），gate 半不动；`down_proj` `[E, H, I]` 在最后一维乘 `s`（`absorb_fused_gate_up_down`）。

#### 2.2.2 `mean` / `max` 与 `alpha` 怎么定 `s`

校准激活 `X` 为 `[token, 通道]`。通道缩放（本步 **固定 `beta = 0`**，不除权重幅度）：

```text
s_j = a_j ^ alpha
```

实现接口：`make_scales_from_alpha_beta(act_profile, w_max, alpha, 0.0)` → `s = act^α / w^β`，此处 `β≡0`。

- **`a_j`（通道画像）**：回答「第 j 通道相对别的通道有多尖」→ 决定各通道 `s` 的**相对比例**。尖的通道 `a_j` 大，钝的小。
  - 理想落地常用 `a_j = mean_token |X_{:,j}|`（均值抗校准噪声，避免偶然极值把某通道 `s` 永久拉高）。
  - 搜分阶段常用 `a_j = max_token |X_{:,j}|`（动态 INT8 和尖刺伤害都跟峰值有关，用 `max` 时不同 `α` 在尖通道上差距更大，更好选出能压离群的档）。
- **`alpha`**：全层（该子图）共用一个全局油门 → 决定这份相对差异要执行到几成。
  - `α = 0` → `s ≡ 1`（不搬）；
  - `α = 1` → `s = a`（按画像幅度几乎搬满）；
  - `α = 0.5` 等为开方折中。
- **不是**每个通道手搜一个 `s_j`，而是只定一个 `α`，再代入 `s_j = a_j^α`。

Flex Smooth 用 `s ∝ a^α / w^β`（权重本身很大时少搬一点）。本步写死 `β = 0`，只按激活搬；权重好不好量化交给假量化 MSE 去约束 `α`。

**搜 `α` 用 `max`、落地用 `mean`** 是常见工程启发式（搜与落地画像可不一致）。本仓库 `flex_awq_ssz.py` 在已有校准激活时，搜分与吸收两侧目前都用 `|X|` 的 **amax**；无校准时用 `|W_down|` 列 max 作代理（见 2.2.4）。也可以全程用 `mean` 搜，但对尖刺的分辨力通常更弱。

#### 2.2.3 搜 `alpha`：假量化 + SSZ 打分

`alpha` 是网格试出来的，不是闭式算出来的。`search_alpha_awq_ssz` 对

```text
α ∈ {0, search_step, …, 1.0}
```

搜索（recipe 默认 `search_step: 0.1`；更细时可设 `0.05`，约 21 档）。每一档：

```text
s    = act_profile ^ α          # 搜分阶段常用 max_token(|X|)^α
X̃   = X / s
W̃   = fake_quant_SSZ(W_down * s)   # 权重 INT4 SSZ 假量化（ssz_step≤10）
Y_fp = X · W_downᵀ                 # 原始浮点金标准（未缩放）
Y_q  = Quant_INT8_sym(X̃) · W̃ᵀ     # 激活 per-token 对称 INT8 × 伪 INT4 权重
score = RMS(Y_q - Y_fp) / RMS(Y_fp)
```

取 `score` 最小的 `α`。没有 `α = f(X,W)` 的解析解：好坏取决于 INT4 SSZ + 动态 INT8 之后的输出误差，只能用量化误差当裁判。若配置写死固定 `alpha`，则可跳过搜索。

**分数怎么算（归一化 RMSE，不是未开方的 `mean((·)²)`）：**

```text
Y_fp  = X · Wᵀ
normal = sqrt( mean(Y_fp²) ) = RMS(Y_fp)
score  = sqrt( mean( (Y_q - Y_fp)² ) ) / normal
       = RMS(Y_q - Y_fp) / RMS(Y_fp)
```

除以 `RMS(Y_fp)` 是为了跨层 / 跨专家时输出量纲可比。比较谁大谁小时与普通 MSE 单调一致，选出的最优 `α` 相同。

- 打分打在 **down**（如 Kimi 的 w2）的输出上。
- `fake_quant_weight_ssz`：按通道 absmax 试若干 clip 比例，选重建 MSE 最小的伪 INT4（`ssz_step` 默认 10）。这是**搜索用的裁判**，不是最终落盘量化器。

#### 2.2.4 校准激活 vs 权重代理

| `use_calib_acts` | 行为 |
|------------------|------|
| `true` + 提供 `--act-stats` | 用 down（或 up）入口的校准激活搜 `alpha` 并算通道画像 |
| `false`（各 recipe 默认） | 用 `|W_down|` 列 max 当作 `[1, K]` 激活代理（无校准也能跑，与「真激活 AWQ」有差距） |

有校准数据时更贴近真实尖刺；无校准时仍可做一轮结构上的通道搬家，便于流水线打通。

#### 2.2.5 落地顺序（与 recipe）

典型顺序：`quarot` → **`flex_awq_ssz`** → `flex_smooth_quant`。  
AWQ 在 QuaRot 之后的旋转空间里搜；Smooth 是另一类 **Norm–Linear** 子图（第三章），**不要**把 UpDown 种子（如 `experts.*.w3`）放进 Smooth 的 `include`。

流程：

1. recipe `include` 列出 **up** 模块 glob；
2. `adapter.resolve_up_down(up)` 得到 **down**；
3. 搜 `alpha` → 算 `s` → `absorb_up_down`（或 fused 版）。

#### 2.2.6 极简数值例子（中间维 = 4，打在专家 `w3`）

下面用手算把 **Flex AWQ SSZ 打在专家 `w3`（up 侧）** 走一遍；中间维缩小到 4。

##### （1）场景设定

Kimi 专家是 SwiGLU：

```text
h = silu(w1(x)) * w3(x)   # gate × up
y = w2(h)                 # down
```

AWQ 这步只作用在 **`w3` 的输出通道**，也就是中间维 `h` 上：

```text
h'  = h / s
w2' = w2 * s              # 按输入通道（列）乘 s
⇒  h' · w2' = (h / s) · (w2 * s) = h · w2
```

浮点完全等价，但 `h` 的尖刺被 `s` 压矮，`w2` 对应通道变胖，量化更好做。

##### （2）校准激活：中间维 `h`

假设校准集有 3 个 token：

| token | 通道1 | 通道2 | 通道3 | 通道4 |
|-------|-------|-------|-------|-------|
| t1    | 100   | 1     | 1     | 1     |
| t2    | 90    | 2     | 1     | 1     |
| t3    | 110   | 1     | 1     | 1     |

通道1 明显是尖刺，其他通道正常。

每通道画像 `a_j = mean_token |X_{:,j}|`：

- 通道1：`(100+90+110)/3 = 100`
- 通道2：`(1+2+1)/3 = 1.333`
- 通道3：`(1+1+1)/3 = 1`
- 通道4：`(1+1+1)/3 = 1`

```text
a = [100, 1.333, 1, 1]
```

##### （3）通道缩放 `s_j = a_j^α`

`α` 是全局油门：`α=0` → `s≡1`；`α=1` → `s=a`；`α=0.5` → 开方折中。

| α    | s₁ = 100^α | s₂ = 1.333^α | s₃ | s₄ |
|------|------------|--------------|----|----|
| 0    | 1          | 1            | 1  | 1  |
| 0.5  | 10         | 1.155        | 1  | 1  |
| 1    | 100        | 1.333        | 1  | 1  |

##### （4）缩放激活 `h' = h / s`

**α = 0（不缩放）**：`h' = h`，通道1 仍约 100。

**α = 0.5**，`s = [10, 1.155, 1, 1]`：

| token | 通道1 | 通道2 | 通道3 | 通道4 |
|-------|-------|-------|-------|-------|
| t1    | 10    | 0.866 | 1     | 1     |
| t2    | 9     | 1.732 | 1     | 1     |
| t3    | 11    | 0.866 | 1     | 1     |

通道1 最大值从 110 降到 11，动态范围约 11:1。

**α = 1**，`s = [100, 1.333, 1, 1]`：

| token | 通道1 | 通道2 | 通道3 | 通道4 |
|-------|-------|-------|-------|-------|
| t1    | 1     | 0.75  | 1     | 1     |
| t2    | 0.9   | 1.5   | 1     | 1     |
| t3    | 1.1   | 0.75  | 1     | 1     |

所有通道大约在 0.75～1.5，非常均匀。

##### （5）权重侧：`w2' = w2 * s`

假设 `w2`（输出 hidden=2，输入中间维=4）：

```text
w2 = [[1, 1, 1, 1],
      [1, 1, 1, 1]]
```

按列乘 `s`：

- α = 0：`w2' = w2`
- α = 0.5：每行变成 `[10, 1.155, 1, 1]`
- α = 1：每行变成 `[100, 1.333, 1, 1]`

##### （6）假量化与打分

对每个 α：

1. 激活 `h'` 用 **per-token INT8** 量化；
2. 权重 `w2'` 用 **per-channel INT4 SSZ** 假量化（最多 10 步）；
3. `Y_q = Quant(h') · Quant_SSZ(w2')`；
4. 与浮点金标准 `Y_fp = h · w2` 比较：

```text
score = RMS(Y_q - Y_fp) / RMS(Y_fp)
```

选 `score` 最小的 `α`。

**手算近似（简化量化，仅帮助直觉）：**

| α | 现象（粗略） | 约略误差 |
|---|--------------|----------|
| 0 | `h` 通道1≈100，INT8 per-token scale 很大，其它通道被挤扁；`w2` 全 1，INT4 基本不变；`Y_q≈[102.36,102.36]`，`Y_fp=[103,103]` | ~0.64 |
| 0.5 | `h'` 通道1≈10，激活更好量化；`w2'` 第 1 列变 10，INT4 后其它列略被压；`Y_q≈[104.16,104.16]` | ~1.16 |
| 1 | `h'` 很匀，但 `w2'` 第 1 列=100，INT4 scale 巨大，其它列≈0；`Y_q≈[100,100]` | ~3 |

本简化例子里 α=0 误差最小，是因为 `w2` 太简单。实际中 `w2` 各列不同、尖刺通道不是唯一重要通道时，α=0.5 往往更好。**关键点：搜索 α 用真实的 INT4 SSZ 假量化器算 MSE，不是靠手算猜。**

##### （7）搜 α 用 max，落地用 mean

**搜索阶段**（更好分辨尖刺）用每通道最大绝对值：

```text
a_max = [max(100,90,110), max(1,2,1), 1, 1] = [110, 2, 1, 1]
s = a_max^α
```

不同 α 在尖通道上的差距更大，更容易选出能压离群的档。

**落地阶段**（冻进模型）用均值：

```text
a_mean = [100, 1.333, 1, 1]
s = a_mean^α
```

均值抗校准噪声，避免某个 token 的偶然极值把某通道 `s` 永久拉高。

（本仓库 `flex_awq_ssz` 当前搜/吸两侧多用 amax，见 2.2.2；上表是理想启发式。）

##### （8）融合：吸进 `w3` 和 `w2`

选定 `α` 后得到最终 `s`：

- `w3` 输出通道除以 `s`（每一输出行 / `s_j`）→ `w3(x)` 输出变成原来的 `1/s`；
- `w2` 输入通道乘以 `s`（每一输入列 × `s_j`）。

中间维上：

```text
(silu(w1) * w3 / s) · (w2 * s) = silu(w1) * w3 · w2
```

推理时不再显式除 `s`，中间维激活自动被压平。

##### （9）例子小结

1. **算画像**：校准中间维 `h`，`a_j = mean(|h|)`（或搜分用 max）。
2. **搜 α**：网格试 `α`，用假量化 `h/s` 与 `w2*s`，选输出 score 最小的档。
3. **定 s**：落地用选定画像算 `s = a^α`。
4. **融合**：`w3` 出通道 `/s`，`w2` 入通道 `×s`，中间维抵消。
5. **效果**：尖刺压矮，权重侧由 per-channel INT4 消化，量化误差下降。

一句话：**通过搜索一个全局 α，按通道把中间维激活除、权重乘，把 Up–Down 子图的量化误差压到最小，且不改变浮点等价性。**

---

### 2.3 使用条件（何时可以开 Flex AWQ-SSZ）

**开 Flex AWQ-SSZ，当且仅当同时满足：**

1. **存在成对的 Up–Down，且中间维一致**  
   `up.out_features == down.in_features`（或 fused：`gate_up` 的 up 半宽度 = `down` 最后一维）。adapter 必须实现 `resolve_up_down`。

2. **中间维上的对角缩放可离线吸收**  
   缩放只改 up 写出通道与 down 读入通道；中间若有 SiLU/SwiGLU，仍按 UpDown 惯例吸收（不要求残差维 = hidden）。

3. **recipe 能给出明确的 up 种子 `include`**  
   一般只打「激活最尖」的 up（如专家 `w3` / `up_proj`），避免无配对的 orphan 缩放。

4. **权重已是浮点**  
   MXFP4 / FP8 须先 unpack；3D fused 专家走 `absorb_fused_gate_up_down`，2D 走 `absorb_up_down`。

与 QuaRot 四条件的关键区别：

- **不要求**残差与 `hidden_size` 一致，也**不要求**全局 RMSNorm fuse。
- 因此 **Qwen3.8-Flash-Next 不能做 QuaRot，但仍可做 Flex AWQ**（打 shared / fused 专家 Up–Down）。

「能不能做」取决于有没有合法 Up–Down；「打哪些」由 recipe `include` + `resolve_up_down` 决定。

---

### 2.4 DeepSeek-V3：对哪些权重做了 Flex AWQ

Recipe：`preprocess/recipes/deepseek_v3.yaml`；配对：`DeepSeekV3Adapter.resolve_up_down`。

| include 种子（up） | 配对 down | 说明 |
|--------------------|-----------|------|
| `model.layers.*.mlp.up_proj` | `...mlp.down_proj` | 前 `first_k_dense_replace` 层 dense MLP |
| `model.layers.*.mlp.shared_experts.up_proj` | `...shared_experts.down_proj` | 共享专家 |
| `model.layers.*.mlp.experts.*.up_proj` | `...experts.*.down_proj` | 路由专家（逐专家 2D） |

```text
… → up_proj ──(中间维 / s)──► SiLU/SwiGLU ──► down_proj (× s) → …
```

Attention / `gate_proj` / router **不在**本步 `include` 里（gate 与 up 并行时，只缩放 up 半边对应的中间通道逻辑由「只列 up」保证）。

---

### 2.5 Kimi-K3：对哪些权重做了 Flex AWQ

Recipe：`preprocess/recipes/kimi_k3.yaml`；配对：`KimiK3Adapter.resolve_up_down`（`w3→w2`，以及通用 `up_proj→down_proj`）。

| include 种子（up） | 配对 down | 说明 |
|--------------------|-----------|------|
| `...block_sparse_moe.experts.*.w3` | `...experts.*.w2` | 路由专家 up；默认**只打这一类** |

```text
w3 (up) ── /s ──► 中间激活 ──► w2 (down) × s
```

- 专家里 up/gate 激活往往更尖，所以先打 **w3**；**w1 / w2 本步不改**（w1 是 gate 侧）。
- 共享专家 `up_proj`、layer0 dense `up_proj`、`routed_expert_up_proj` 放在 **Flex Smooth** 的 `include` 里，不在 AWQ 重复挂，避免双重搬家语义混乱。

---

### 2.6 Qwen3.8-Flash-Next：为什么仍可做 Flex AWQ

Qwen 因 HC + 未 fuse 的 Gemma `hc_norm` **不能**做全局 QuaRot，但专家 FFN 仍是标准 Up–Down，满足 §2.3。

Recipe：`preprocess/recipes/qwen3_8_flash_next.yaml`。

| include 种子（up） | 配对 down | 说明 |
|--------------------|-----------|------|
| `...mlp.shared_expert.up_proj` | `...shared_expert.down_proj` | 2D 共享专家 |
| `...mlp.experts.gate_up_proj` | `...experts.down_proj` | 融合 3D：只缩放 up 半 |
| MTP 层同上 | 同上 | `mtp.layers.0.mlp.*` |

默认 transforms：`flex_awq_ssz` + `flex_smooth_quant`（无 `quarot`）。AWQ 专打 MLP 中间维，与残差上的 HC/`hc_norm` 解耦。

---

### 2.7 给新模型怎么分析（含 GLM-5.2 示例）

#### 步骤

1. **找 FFN / 专家里的 Up–Down**  
   命名常见：`up_proj`/`down_proj`，`w3`/`w2`，或 fused `gate_up_proj`/`down_proj`。确认中间维相等。

2. **实现 `resolve_up_down`**  
   输入 up 模块名，返回 down；无法配对则返回 `None`（该种子会被 skip）。

3. **选 `include` 种子**  
   优先激活最尖、且后续会做低比特的 up（常为路由专家 up）。不要把无 down 的层、也不要把纯 Norm 放进 AWQ。

4. **核对 2D vs 3D**  
   逐专家 2D → `absorb_up_down`；整层 fused `[E, 2I, H]` → 走 fused 吸收，且 `include` 写 `gate_up_proj` 而不是不存在的 `experts.*.up_proj`。

5. **校准（可选）**  
   `use_calib_acts: true` + 采集 down 入口激活，搜索更稳；默认代理模式可先 smoke。

6. **与 Smooth 分工**  
   UpDown → `flex_awq_ssz`；Norm→Linear → `flex_smooth_quant`。同一 up 不要两边各搬一次除非有意叠加且清楚顺序。

#### 示例：GLM-5.2（`/parastor/opendas/DL_DATA/llm-models/GLM-5.2`）

架构：`glm_moe_dsa`，前几层 dense MLP，其后 MoE + shared experts（命名接近 DeepSeek：`mlp.up_proj` / `experts.*.up_proj` / `shared_experts.up_proj`）。

**条件判断：** 存在标准 Up–Down → **可以做 Flex AWQ**（与能否做 QuaRot 独立；GLM-5.2 两者通常都可以）。

**建议 `include`（接入 preprocess 时仿 DeepSeek recipe）：**

```yaml
flex_awq_ssz:
  search_step: 0.1
  ssz_step: 10
  use_calib_acts: false
  include:
    - model.layers.*.mlp.up_proj
    - model.layers.*.mlp.shared_experts.up_proj
    - model.layers.*.mlp.experts.*.up_proj
```

`resolve_up_down` 与 DeepSeek 相同模式即可；**indexer 不参与 AWQ**（不是 Up–Down 中间维问题）。核对 checkpoint 里专家是 2D 逐专家还是 fused 3D，再决定是否要用 fused 吸收路径。

#### 决策速查

```text
新模型
  ├─ 有 up/down（或 w3/w2、gate_up/down）且中间维一致？
  │     ├─ 否 → 不做 flex_awq_ssz
  │     └─ 是 → 写 resolve_up_down + include
  │           ├─ 路由专家 up 最尖？ → 优先列入（如 Kimi 只打 w3）
  │           ├─ dense / shared 也要 W4？ → 一并列入（如 DeepSeek / Qwen）
  │           └─ fused 3D？ → include gate_up，用 fused absorb
  └─ 与 QuaRot / Smooth：可叠加；UpDown 种子勿塞进 Smooth
```

---

### 2.8 本章小结

| 问题 | 结论 |
|------|------|
| AWQ 算什么 | 中间维对角 `X/s`、`W*s`；吸进 up/down；用 INT4 SSZ 假量化搜 `alpha` |
| 何时能开 | 成对 Up–Down、可吸收、include 明确、浮点权重 |
| DeepSeek-V3 | dense / shared / 路由专家的 `up_proj→down_proj` |
| Kimi-K3 | 默认仅路由专家 `w3→w2` |
| Qwen3.8-Flash-Next | 不做 QuaRot，但做 shared + fused 专家 AWQ |
| 新模型（如 GLM-5.2） | 仿 DeepSeek UpDown；indexer 不进 AWQ |

---

## 第三章 Flex Smooth Quant：原理、条件与模型适配

本章对应 transform：`flex_smooth_quant`（`preprocess/transforms/flex_smooth.py`）。它与 Flex AWQ 同属「对角缩放搬家」，但子图不同：

| | Flex AWQ-SSZ（第二章） | Flex Smooth Quant（本章） |
|--|------------------------|---------------------------|
| 子图 | **Up–Down** | **Norm–Linear** |
| 缩放公式 | `s ∝ a^α`（固定 `β=0`） | `s ∝ a^α / w^β`（同时搜 `α`、`β`） |
| 假量化裁判 | 激活 INT8 × 权重 **INT4 SSZ** | 激活 INT8 × 权重 **INT8** |
| 吸收位置 | `W_up /= s`（out），`W_down *= s`（in） | 全部下游 `W *= s`（in）；Norm `γ /= s`（或 Gemma 等价式） |
| 典型种子 | 专家 `w3` / `up_proj` | Norm 名，或 Norm 的任一 Linear 消费者 |

本步只改浮点权重 / γ，**不**写出 INT4/INT8；为后续 W8A8（以及已做过 AWQ 的路径上再压一轮激活尖刺）铺路。

---

### 3.1 要解决什么问题

很多层的难点不在 FFN 中间维，而在 **RMSNorm（或 Gemma Norm）刚输出、还没进 Linear** 的激活：少数通道特别大，后面所有读该 Norm 的 Linear（QKV、gate/up、router、shared expert…）共享同一份尖刺。若只对某一条 Up–Down 做 AWQ，管不到 Attention / shared / router 这些边。

**Flex SmoothQuant** 在 Norm 输出维上找一组正数 `s`，做经典搬家：

```text
X' = X / s          # Norm 后激活变矮
W' = W * s          # 每个下游 Linear 的输入通道变胖
γ' = γ / s          #（RMS）或 Gemma：γ' = (1+γ)/s - 1
⇒  浮点 (γ ⊙ rms(x)) 进 Linear 的结果不变
```

与 AWQ 对比：AWQ 专打 **一条** up→down 链，并用 INT4 假量化当裁判；Smooth 专打 **一个 Norm 扇出的整组 Linear**，裁判是 INT8×INT8，覆盖面更宽、位宽假设更偏 W8。

---

### 3.2 计算原理

#### 3.2.1 Norm–Linear 恒等式

标准 RMSNorm：`y = rms(x) ⊙ γ`，再 `z = W y`（`W` 布局 `[out, in]`）。对每个 in 通道 `j`：

```text
y'_j = y_j / s_j
W'_{*,j} = W_{*,j} * s_j
γ'_j = γ_j / s_j
```

则 `y' = rms(x) ⊙ γ'`，且 `W' y' = W y`。Gemma（`y = rms(x) ⊙ (1+γ)`）用：

```text
γ' = (1+γ)/s - 1
```

使 `1+γ' = (1+γ)/s`，同样保持等价（`absorb_norm_linear`）。

**关键**：同一 Norm 的**全部** Linear 消费者必须乘同一套 `s`。只改其中一个 Linear、不改兄弟层，等价性破坏。实现上以 adapter 的 `build_ln_fuse_map` / `resolve_norm_linear` 为消费者全集；recipe `include` 只提供**种子**（Norm 名或任一消费者），再展开成 `(norm, siblings)` 组。

#### 3.2.2 为什么是 `s ∝ a^α / w^β`

通道画像：

```text
a_j = max_token |X_{:,j}|     # 实现里搜分与落地多用 amax
w_j = max_out |W_{*,j}|       # 对参与搜索的 Linear（或多 Linear 拼/逐个 max）按 in 通道
s_j = a_j^α / w_j^β
```

- **`α`**：激活尖刺要搬多少。`α=0` → 不看激活；`α=1` → 按激活幅度几乎搬满。
- **`β`**：权重本身已经很大的通道少搬一点，避免「激活变好、权重炸坏」。`β=0` 时退化为纯按激活（接近 AWQ 的 β 设定）；`β>0` 是 Smooth 相对 AWQ 的核心差别。
- 不是每通道手搜 `s_j`，而是搜两个全局标量 `(α, β)`，再代入上式。

#### 3.2.3 搜 `(α, β)`：两阶段网格 + INT8 假量化

`search_alpha_beta`（`flex_smooth.py`）：

1. **先搜 α**，令 `β = 1-α`（经典 SmoothQuant 对角线约束上的一维搜索）：
   ```text
   α ∈ {0, search_step, …, 1.0}
   ```
2. **固定最优 α，再扫 β**：
   ```text
   β ∈ {0, search_step, …, 1.0}
   ```

每一档：

```text
s    = a^α / w^β
X̃   = X / s
W̃   = W * s
Y_fp = X · Wᵀ
Y_q  = Quant_INT8(X̃) · Quant_INT8(W̃)ᵀ
score = RMS(Y_q - Y_fp) / RMS(Y_fp)
```

取 score 最小的 `(α, β)`。与 AWQ 的差别：这里权重假量化是 **对称 INT8**，不是 INT4 SSZ——因为 Smooth 服务的对象经常是后续 W8 层，且 Norm 扇出里混有不会走 W4 的 Linear。

#### 3.2.4 种子 → Norm 组 → 搜索消费者 → 吸收全体

流程（`FlexSmoothTransform.apply`）：

1. `include` glob → 匹配到的模块名（Norm 或 Linear）；
2. `adapter.resolve_norm_linear(seed)` → `(norm, all_consumers)`；解析失败 → **orphan，跳过**（并提示 UpDown 种子应去 `flex_awq_ssz`）；
3. 按 `norm` 分组；同一 Norm 只处理一次；
4. 若消费者极多（如 DeepSeek `post_attention_layernorm` 挂数百专家）：`max_search_consumers`（默认 8）只在子集上搜 `(α,β)`（优先非 `.experts.`），**吸收仍对全部 present consumers**；
5. `absorb_norm_linear(norm, present, scales, norm_style)`。

Gemma + `hc_count>1` 且 Norm 维 = `hc*hidden` 时走 **tied** 路径：`absorb_gemma_tied_hc`——hidden 维一套 `s`，宽 Linear 用 `s.repeat(hc)`，并要求 `hidden_smooth_partners`（mix 后的 hidden 消费者）齐全，否则 skip。

#### 3.2.5 校准激活 vs 权重代理

| `use_calib_acts` | 行为 |
|------------------|------|
| `true` + `--act-stats` | 优先用 seed Linear 或 Norm 的校准激活 |
| `false`（默认） | `|W|` 列 max 作 `[1,K]` 激活代理 |

与 AWQ 相同：无校准也能跑通；有校准更贴近真实 Norm 出口尖刺。

#### 3.2.6 与 AWQ / QuaRot 的顺序与禁忌

典型顺序：`quarot` → `flex_awq_ssz` → **`flex_smooth_quant`**。

| 规则 | 原因 |
|------|------|
| **禁止**把纯 UpDown 种子（如仅 `experts.*.w3` 且不在 LN map 里）塞进 Smooth | `resolve_norm_linear` 失败 → orphan；强行单边缩放改网络 |
| Smooth 的 `include` 可以是 Norm，也可以是消费者 | 二者都会展开成同一 Norm 组 |
| 同一条 up 若已在 AWQ 吸过中间维，Smooth 再动的是 **Norm→该 Linear 的输入维**（hidden），与中间维不同轴 | 一般可叠加，但要清楚改的是哪一维 |
| one-click 的 Kimi recipe **不把** `experts.*.w3` 放进 Smooth | w3 不在 `post_attention_layernorm` 消费者表里；AWQ 已处理 `w3↔w2`。勿与其它工具的 Smooth include 名单混用 |

#### 3.2.7 极简数值例子（Norm 维 = 4，两个下游 Linear）

##### （1）场景

```text
y = rms(x) ⊙ γ
z1 = W1 y      # 例如 gate
z2 = W2 y      # 例如 up
```

校准后 Norm 出口激活（3 token × 4 通道）尖在通道 0：

```text
a = max_token|X| = [80, 2, 1, 1]
```

两个下游拼在一起后，in 通道权重画像：

```text
w = [1, 10, 1, 1]    # 通道 1 权重本身很大
```

##### （2）不同 `(α, β)` 下的 `s = a^α / w^β`

| (α, β) | s₀ | s₁ | s₂ | s₃ | 含义 |
|--------|----|----|----|----|------|
| (0, 0) | 1 | 1 | 1 | 1 | 不搬 |
| (1, 0) | 80 | 2 | 1 | 1 | 只按激活搬满（类 AWQ） |
| (0.5, 0.5) | √80≈8.94 | √(2/10)≈0.45 | 1 | 1 | 激活尖搬一点；权重大的通道少搬 |
| (0, 1) | 1 | 0.1 | 1 | 1 | 只按权重「反缩放」（少见最优） |

通道 0：激活尖 → `s` 宜大。通道 1：激活不大但 `w` 大 → `β>0` 时 `s` 被压小，避免权重再乘大数后 INT8 更差。

##### （3）吸收

设选中 `α=0.5, β=0.5`，`s≈[8.94, 0.45, 1, 1]`：

```text
W1 ← W1 * s     # 沿 in
W2 ← W2 * s
γ  ← γ / s      # RMS
```

浮点 `W1 (rms⊙γ)` 不变；推理时 Norm 出口自动变矮。

##### （4）假量化打分（直觉）

对每个候选 `(α,β)`：用 `X/s` 与 `W*s` 做对称 INT8 假量化，算相对 `X W` 的归一化 RMSE，取最小者。实现里先沿 `β=1-α` 搜 α，再固定 α 扫 β。

##### （5）例子小结

1. 由种子找到 Norm 及全部消费者；  
2. 用激活/权重画像搜 `(α,β)` → `s`；  
3. 全部 Linear `×s`，γ 反缩放；  
4. 激活尖刺下移到权重，且权重大的通道被 `β` 保护。

---

### 3.3 使用条件（何时可以开 Flex Smooth）

**开 Flex Smooth，当且仅当同时满足：**

1. **存在 Norm→Linear 扇出，且消费者表可枚举**  
   adapter `build_ln_fuse_map` / `resolve_norm_linear` 能给出完整 `(norm, linears)`；遗漏兄弟 Linear 会破坏等价。

2. **缩放轴是 Norm 输出维 = 各 Linear 的 in_features**  
   （Gemma HC 特例：`hc*hidden` Norm + tied `s.repeat(hc)`，另需 hidden partners。）

3. **recipe `include` 种子能解析进某 Norm 组**  
   种子可以是 Norm 或消费者；解析失败则跳过，不会静默改单边权重。

4. **权重与 γ 已是浮点**  
   与 AWQ 相同，须先 unpack。

与 AWQ / QuaRot 的关系：

- **不要求** Up–Down 对；**不要求**残差可旋转。  
- Norm 若被 elementwise 复用（Qwen `hc_norm`），**不能**做 QuaRot fuse，但 **仍可能**做 Smooth（不把 γ→0，只做 `γ'=(1+γ)/s-1` 的 tied 吸收）——是否开启由 recipe 决定（Qwen 默认关掉层内 `hc_norm`，只 Smooth MTP）。

「值不值得做」：Norm 出口激活 outlier 重、或下游大量 W8 Linear 时收益大；若尖刺几乎只在专家中间维，优先 AWQ 即可。

---

### 3.4 DeepSeek-V3：对哪些 Norm / 权重做了 Flex Smooth

Recipe：`preprocess/recipes/deepseek_v3.yaml`；消费者表：`DeepSeekV3Adapter.build_ln_fuse_map`。

`include` **直接列 Norm**（种子即左侧）：

| include 种子（Norm） | 典型消费者（由 LN map 展开） |
|----------------------|------------------------------|
| `model.layers.*.input_layernorm` | `q_a_proj`、`kv_a_proj_with_mqa` |
| `model.layers.*.post_attention_layernorm` | dense：`gate/up`；MoE：`gate`、shared `gate/up`、全体 `experts.*.{gate,up}_proj` 等 |
| `model.layers.*.self_attn.q_a_layernorm` | `q_b_proj` |
| `model.layers.*.self_attn.kv_a_layernorm` | `kv_b_proj` |
| `model.norm` | `lm_head` |
| `model.layers.*.shared_head.norm` | MTP `shared_head.head` |

```text
input_layernorm ── /s ──► q_a / kv_a (×s)
post_attn_ln    ── /s ──► gate / up / experts… (×s)   # 消费者极多
q_a_layernorm   ── /s ──► q_b
kv_a_layernorm  ── /s ──► kv_b
```

`max_search_consumers: 8`：MoE 的 `post_attention_layernorm` 先在 gate/shared 等子集上搜 `(α,β)`，再把同一 `s` 吸进全部专家消费者。

`enorm`/`hnorm`→`eh_proj` 不在 fuse/Smooth 表里（拼接输入，不是单 Norm→Linear）。

---

### 3.5 Kimi-K3：对哪些 Norm / 权重做了 Flex Smooth

Recipe：`preprocess/recipes/kimi_k3.yaml`；LN map：`KimiK3Adapter.build_ln_fuse_map`。

`include` **列的是消费者种子**（经 `resolve_norm_linear` 反查 Norm）：

| include 种子（Linear） | 解析到的 Norm | 同组其它消费者（吸收时一并 ×s） |
|------------------------|---------------|----------------------------------|
| `...shared_experts.up_proj` | `post_attention_layernorm` | `gate`、shared `gate_proj`、`routed_expert_down_proj` 等 |
| `...layers.0.mlp.up_proj` | layer0 `post_attention_layernorm` | layer0 `gate_proj` / `up_proj` |
| `...routed_expert_up_proj` | `routed_expert_norm`（若 `latent_moe_use_norm`） | `routed_expert_up_proj` |

```text
post_attn_ln ── /s ──► shared up/gate / router / routed_expert_down …
routed_expert_norm ── /s ──► routed_expert_up
```

- **不**把 `experts.*.w3` 放进 Smooth：`w3` 不在上述 LN 消费者表中；中间维已由 AWQ 的 `w3→w2` 处理。  
- 若其它工具的 Smooth `include` 写了 `experts.*.w3`，其语义可能不同；one-click 中该种子不在 LN map 里，会被当作 orphan 跳过。

---

### 3.6 Qwen3.8-Flash-Next：为什么只做有限 Smooth

不能做全局 QuaRot（HC + 未 fuse `hc_norm`），但 Norm–Linear Smooth **局部仍可能**。

Recipe 默认 `include` **仅 MTP**：

| include | 说明 |
|---------|------|
| `mtp.pre_fc_norm_embedding` | LN fuse map 内 → `mtp.fc_embedding`；可标准 RMS/Gemma 吸收 |
| `mtp.pre_fc_norm_hidden` | 特殊：shared `fc_hidden`，`resolve_norm_linear` 返回空消费者列表，避免错误 fold |

**层内 `hc_norm` 默认关掉**（对照实验）：虽有 `_hc_smooth_consumers` + `hidden_smooth_partners` 的 tied Gemma 路径，但 mix 里 elementwise 复用 normed 向量，Smooth 需非常小心；默认 recipe 不列入，避免 collapse。若强制打开，走 `absorb_gemma_tied_hc`，而不是 QuaRot 式 `γ→0`。

默认 transforms：`flex_awq_ssz` + `flex_smooth_quant`（无 `quarot`）——AWQ 打专家 Up–Down，Smooth 只动 MTP 安全 Norm。

---

### 3.7 给新模型怎么分析（含 GLM-5.2 示例）

#### 步骤

1. **画出每个 Norm 的消费者**  
   只进 Linear？有无 `gates * normed`、拼接、多输入？后者不能当标准 Norm–Linear（或只能 Smooth 特例 / 直接跳过）。

2. **实现 `build_ln_fuse_map`（及必要时 `resolve_norm_linear`）**  
   Smooth 与 QuaRot fuse **共用**这张表时最省事；若某 Norm 可 Smooth 不可 fuse（如 Qwen `hc_norm`），再 override `resolve_norm_linear`。

3. **选 `include` 种子**  
   写 Norm 名或任一消费者均可；优先激活尖、下游要 W8/敏感的组。MoE 扇出巨大时设 `max_search_consumers`。

4. **Norm 风格**  
   `norm_affine()`：`rms` 或 `gemma`；HC 宽 Norm 实现 `hidden_smooth_partners`。

5. **与 AWQ 分工**  
   中间维 UpDown → AWQ；Norm 出口扇出 → Smooth。不要把无 Norm 前驱的 up 种子丢进 Smooth。

6. **校准（可选）**  
   hook Norm 出口或首个消费者输入，写入 `acts.pt`。

#### 示例：GLM-5.2

普通残差 + RMSNorm + MLA + MoE，LN 拓扑接近 DeepSeek：

**可做 Flex Smooth。** 建议 `include`（仿 DeepSeek，并视需要加 indexer 前 Norm——若 indexer 前另有 Norm 且只喂 Linear）：

```yaml
flex_smooth_quant:
  search_step: 0.1
  use_calib_acts: false
  max_search_consumers: 8
  include:
    - model.layers.*.input_layernorm
    - model.layers.*.post_attention_layernorm
    - model.layers.*.self_attn.q_a_layernorm
    - model.layers.*.self_attn.kv_a_layernorm
    - model.norm
```

`post_attention_layernorm` 的专家消费者必须在 adapter LN map 里列全；搜索用子集、吸收用全集。Indexer 若共享 `input_layernorm`，应出现在同一消费者列表中，否则 Smooth 会漏乘 `s`。

#### 决策速查

```text
新模型
  ├─ Norm 输出是否只进 Linear（可枚举全集）？
  │     ├─ 否（elementwise / 拼接 / 多输入）→ 跳过或走特例（tied Gemma HC）
  │     └─ 是 → 写 LN map + include
  │           ├─ 扇出含大量专家？ → max_search_consumers
  │           ├─ Gemma / hc*hidden？ → norm_affine + partners
  │           └─ 与 AWQ：UpDown 归 AWQ，Norm 扇出归 Smooth
  └─ 仅 MTP / 局部 Norm 安全？ → 像 Qwen 一样缩小 include
```

---

### 3.8 本章小结

| 问题 | 结论 |
|------|------|
| Smooth 算什么 | Norm–Linear 上 `X/s`、`W*s`、`γ` 反缩放；搜 `(α,β)`，INT8×INT8 假量化打分 |
| 何时能开 | 消费者表完整、轴匹配、种子可解析、浮点权重；不要求可旋转 / 有 UpDown |
| DeepSeek-V3 | 多种 Norm 种子；MoE post-attn 扇出极大，限制 search 消费者数 |
| Kimi-K3 | 以 shared / layer0 / latent up 等为种子反查 Norm；专家 `w3` 不进 Smooth |
| Qwen3.8-Flash-Next | 默认仅 MTP Smooth；层内 `hc_norm` 关掉 |
| 新模型（如 GLM-5.2） | 仿 DeepSeek LN map；indexer 若同 Norm 须列入消费者 |

---

## 总表：三种预处理怎么选

| | QuaRot | Flex AWQ-SSZ | Flex Smooth |
|--|--------|--------------|-------------|
| 变换 | 正交旋转 | 对角缩放（中间维） | 对角缩放（Norm 维） |
| 子图 | 残差 / MLA / latent 成对 Linear | Up–Down | Norm–Linear |
| 硬条件 | 可 fuse、可旋转、无残差对角冲突 | 成对 up/down | 完整 Norm 消费者表 |
| DeepSeek | ✅ | ✅ | ✅ |
| Kimi | ✅ | ✅（默认 w3→w2） | ✅（shared/latent 等） |
| Qwen3.8-FN | ❌ | ✅ | ⚠️ 仅 MTP（默认可） |
| 推荐顺序 | ① | ② | ③ |
