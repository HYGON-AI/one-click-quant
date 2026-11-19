# 阶段4 产物校验与评测

> **权威来源**：仓库根 `README.md`「PPL评估」、`utils/analyze_quant.py`、`utils/lm_eval.py`、
> `utils/torch_infer.py`、`examples/**/*-MSE.py`。
> 顺序：**先产物契约 → 再张量 dtype → 再 MSE → 最后 PPL / 推理**。前一步不过不要往下走。

## 1. 产物契约（目录级）

产物契约按下面四条逐项看。**只有前两条是硬门禁**，后两条按方案而定：

- `config.json`：含量化配置块，键名与内容见下节
- **权重文件二选一**，不要只认 index：
  - 多分片：`model.safetensors.index.json` + `weight_map` 里列出的**每个分片都真实存在**
  - 单分片（小模型常见，<10GB 规模）：**只有 `model.safetensors`，没有 index.json**
- tokenizer 相关文件（`tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` 等）：
  **随源模型一起复制**，源头没有则产物也没有 —— 这种缺失不算量化失败，但下游加载会受影响，需要单独补
- llmcompressor 路径还会写 `recipe.yaml`（量化配方快照）；自定义脚本可能写 `QUANTIZATION_DONE.txt`

### 1.1 配置键与 `quant_method` 必须对得上

| 产出路径 | 配置键 | 需要核对 |
|---|---|---|
| llmcompressor（`FP8_DYNAMIC` / `W8A8` / `W4A16` / 裸 `--scheme W4A8`） | 通常是 `quantization_config` | compressed-tensors 格式，有 `config_groups`，`quant_method: compressed-tensors` |
| `--alg slimquant_ptq --scheme W4A8`（int4） | **固定 `quantization_config`**（按 `quant_type` 写死，不随版本变） | `quant_method: slimquant_w4a8`。若见到 `compression_config` + 该值，说明产物来自 `examples/` 多阶段 / Kimi 脚本 |
| `--alg slimquant_ptq` 的 int8 / fp8 模式 | `compression_config` | `quant_method: compressed-tensors`、`format: int-quantized` / `float-quantized` |
| `examples/` 下手写转换脚本 | **多为 `quantization_config`**；`examples/kimi/kimi_k3_mxfp4_to_int4.py` 与 qwen38 多阶段脚本用 `compression_config` | 各脚本自定，**以脚本实现为准**，不要按本条推断 |
| `examples/qwen/qwen3_8-flash-next/` 多阶段混合精度 | `compression_config`，**且可能同时复制进 `text_config`** | `quant_method: slimquant_w4a8` + `mixed_precision.{experts,w8a8_include}`；两处内容应一致 |

只看"有没有这个键"不够：**键对了但 `quant_method` 不对，说明走了另一条路线**（典型是裸 `--scheme W4A8` 与
`--alg slimquant_ptq --scheme W4A8` 混淆，见 `03-quant-runbook.md` 第 3.1 节）。

> **配置键名本身不是判据**：`quantization_config` 与 `compression_config` 在流通的产物里都能见到
> （不同 alg / 不同 `examples/` 脚本写不同的键）。校验脚本因此两个键都接受，但必须打印并核对 `quant_method`。

> **不要因为缺少 `model.safetensors.index.json` 就判定产物失败**：权重能装进单个分片时
> `save_pretrained` 只写 `model.safetensors`，不生成索引。按 index 判断会把正常产物误判成坏的。

`QUANTIZATION_DONE.txt` **不是 PTQ 判据**：它只由 `examples/` 下少数模型专属转换脚本写出（内含 stats JSON），
`main.py` 的任何 PTQ 主链路都不产出该文件。哪些脚本会写，以 `grep -rl QUANTIZATION_DONE --include=*.py .` 的实际结果为准，
不要记忆这份名单。

手工校验（把产物目录作为参数传进去，兼容单分片与多分片）：

```bash
python3 - "<产物目录>" <<'PY'
import glob, json, os, sys

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
has_q = "quantization_config" in cfg or "compression_config" in cfg
qkey = "quantization_config" if "quantization_config" in cfg else "compression_config"

idx_path = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx_path):
    wm = json.load(open(idx_path))["weight_map"]
    missing = sorted({s for s in wm.values() if not os.path.exists(os.path.join(d, s))})
    print("layout: sharded  tensors=%d shards=%d missing=%s"
          % (len(wm), len(set(wm.values())), missing))
else:
    single = glob.glob(os.path.join(d, "model*.safetensors"))
    files = [os.path.basename(x) for x in single]
    missing = [] if single else ["NO WEIGHT FILE"]
    print("layout: single   files=%s missing=%s" % (files, missing))

print("quant config:", qkey if has_q else "MISSING",
      "| quant_method =", cfg.get(qkey, {}).get("quant_method") if has_q else None,
      "| format =", cfg.get(qkey, {}).get("format") if has_q else None)
print("tokenizer:", os.path.exists(os.path.join(d, "tokenizer.json")))
print("recipe.yaml:", os.path.exists(os.path.join(d, "recipe.yaml")))
PY
```

输出示例（单分片产物）：

```text
layout: single   files=['model.safetensors'] missing=[]
quant config: quantization_config | quant_method = <本方案应有的值> | format = None
tokenizer: True
recipe.yaml: True
```

判读：`missing` 必须为空；`quant_method` 必须与方案对应（对照 1.1 的表）。
`slimquant_ptq` 无论产物几个分片都会写 `model.safetensors.index.json`，所以那种情况下会打印 `layout: sharded ... shards=1`。

**优先直接跑校验脚本**（等价于上面这段手工检查，另加分片存在性与 dtype 抽样）：

```bash
python3 ./skills/scripts/check_artifact.py <产物目录> --expect-quant-method <期望值>
```

上面的 heredoc 只在脚本不可用时用；`--expect-quant-method` 省略则只报告实际值不比对。

> **适用范围（L5/C2）**：`check_artifact.py` 只用于**量化产物**
> （`main.py` PTQ / slimquant / MoE-Quant 打包 / `examples/` 量化脚本，含多阶段最后一档）。
> 下列情况**不要**跑它当门禁：
> - `--alg preprocess` 浮点目录（无量化配置块 → 必然 FAIL；判据见 `10-ext-preprocess.md`）
> - 只改了 `config.json`、权重未变的中间手术步（先按该脚本 README / `21` 的阶段约定验）
> - 尚未写完分片的半成品目录

## 2. 张量 dtype（结构级）

```bash
python3 tools/model_xray.py --input_dir <产物目录> --output_dir <xray输出目录>
```

逐条核对**方案预期**，而不是"看起来像就行"。不同方案的落盘形态差别很大，**别用一套键名去套所有方案**：

| 方案 | 权重键名 | 权重 dtype | scale 键名 | scale dtype |
|---|---|---|---|---|
| `FP8_DYNAMIC` | `...weight` | `float8_e4m3fn` | `...weight_scale` | bf16 `[N,1]` |
| `W8A8`（per-channel INT8） | `...weight` | `int8` | `...weight_scale` | bf16 |
| **`slimquant_ptq` W4A8 · 非 MoE** | `...weight` | `int8` | `...weight_scale` | **float32 `[N,1]`** |
| **`slimquant_ptq` W4A8 · MoE experts** | per-expert `...experts.<i>.<proj>.weight`（2D，本链路唯一形态）；若见到 fused `...experts.gate_up_proj`（3D、无 `.weight`）则不是本链路产物 | `int8`，**K 维减半** | 同前缀 `.weight_scale` | **float32**（`[N,1]`，fused 形态为 `[E,N,1]`） |
| `MoE-Quant` / WNA16（pack-quantized） | `...weight_packed` | **int32**（8×int4 装进 1 个 int32） | `...weight_scale` | bf16 |
| **`examples/` 多阶段混合精度**（Qwen3.8-Flash-Next 类） | 同一产物里**多套并存**：experts 同 slimquant INT4、指定层 INT8、其余 bf16 | 逐类按上面各行核对 | 按类配对 | 按类 |

> 多阶段混合精度产物的配置是 `compression_config.mixed_precision.{experts, w8a8_include}` 分组结构，
> **不能**用"全模型一个 dtype"的眼光看。分组字段语义、每阶段该动哪些层，见 `21-hybrid-quant.md`。

落盘形态示例（各方案差异很大，请对照自己的产物核对，不要照搬 shape）：

```text
# FP8_DYNAMIC
lm_head.weight                            bfloat16      [152064, 3584]   ← --ignore lm_head 生效
model.layers.0.mlp.down_proj.weight       float8_e4m3fn [3584, 18944]
model.layers.0.mlp.down_proj.weight_scale bfloat16      [3584, 1]

# slimquant W4A8，非 MoE 模型 → 全 INT8，K 不减半
model.layers.0.mlp.down_proj.weight       int8          [3584, 18944]
model.embed_tokens.weight                 bfloat16      ← 不在 targets=Linear 内

# slimquant W4A8，MoE 模型 → experts 判据是 dtype + K 减半；本链路产物必为 per-expert 2D

# 形态 A（fused）：源权重是 bf16 裸 3D；带 int8 + float32 scale 的 fused 只可能来自其它工具
model.layers.0.mlp.experts.gate_up_proj              int8    [E, 2I, H/2]
model.layers.0.mlp.experts.gate_up_proj.weight_scale float32 [E, 2I, 1]
model.layers.0.mlp.experts.down_proj                 int8    [E, H, I/2]
model.layers.0.mlp.experts.down_proj.weight_scale    float32 [E, H, 1]

# 形态 B（per-expert，split_fused_moe_experts 之后）：本链路产物的唯一形态
model.layers.0.mlp.experts.0.gate_proj.weight        int8    [I, H/2]
model.layers.0.mlp.experts.0.gate_proj.weight_scale  float32 [I, 1]
model.layers.0.mlp.experts.0.down_proj.weight        int8    [H, I/2]
model.layers.0.mlp.experts.0.down_proj.weight_scale  float32 [H, 1]

model.layers.0.mlp.gate.weight                       bfloat16 [E, H]   ← re:.*mlp.gate$ 忽略
```

判据：

- 该量化的层：权重 dtype 是 int8 / fp8 / int32-packed，且**必有配对 scale**
- 不该量化的层（`modules_to_not_convert` / `ignore` / `targets` 之外，以及所有 `*norm`、`embed_tokens`、`lm_head`）：仍是 bf16
- 不存在残留 `weight_scale_inv`（block FP8 源格式痕迹，说明解包没做干净）
- MoE 的 K 维是否减半：int8 packed → K/2；int32 packed → K/8；纯 INT8 → K 不变。**用 `weight_shape` 或已知 `intermediate_size` 交叉验证**
- **slimquant W4A8 产物的 experts 必为 per-expert 2D**（`split_fused_moe_experts` 无条件先拆）。
  若见到 fused 3D int8 + scale，说明产物不是本链路产出的（其它工具 / 仓库外分支），仍按上面三条不变式验收，
  不要因为"键名没有 `.weight`"或"键数变多"就判异常；反之，若 experts 仍是 **bf16** 3D 未量化，说明该层没被 INT4 处理

判据不一致 → 回阶段3 查 ignore 列表与 `targets`，不要在错的产物上做评测。

## 3. MSE 验证（单层数值）

`find examples -name "*-MSE.py"` 找到的脚本，用同一条反量化路径把产物还原成浮点，与 BF16 源逐 tensor 对比，输出 `mse` 与 `rel_l2` 的 mean/max/worst。

**MSE 脚本与转换脚本是一一配对的**，先查有哪些：

```bash
find examples -name "*-MSE.py"      # 注意在第三层目录，ls examples/*/*-MSE.py 匹配不到
```

> 这些 MSE 脚本只校验**同名转换脚本**的产物。`main.py` 的 PTQ 产物（含 slimquant W4A8）
> 没有现成 MSE 脚本，不要硬套 —— 那种情况改用 dtype 结构校验 + PPL/推理对比。

> **必须显式传 `--src` / `--dst`**：脚本自带 `DEFAULT_SRC` / `DEFAULT_DST` 硬编码路径，
> 那是仓库作者环境里的位置，换台机器就是错的。不传会安静地读到不存在或别人的目录。

```bash
python3 examples/qwen/qwen3_8-flash-next/qwen38_flash_next_bf16_experts_to_int4-MSE.py \
  --src <BF16源目录> --dst <INT4产物目录>
```

- `--src` 必须是**同结构、同层数的 BF16 源**（MSE 脚本按 index 名字对齐，不做层重映射）
- 脚本会强校验 `packed.shape[-1] * 2 == bf16.shape[-1]`，shape 不符直接报错 —— 这是好事，别绕过
- 关注 `max` 与 `worst_rel_l2`，不看单层均值；异常 tensor 再去查该层的 scale 计算与 packing
- 每个方案的反量化公式不同（per-channel scale、`scale/16`、nibble 顺序），**换方案要换对应脚本**

## 4. PPL / 下游评测

`utils/lm_eval.py` 的硬约束（不是建议，也**不是**可调 CLI 参数）：

- 强制 `HF_DATASETS_OFFLINE=1`，**必须离线**
- 数据集缓存必须已在 `HF_DATASETS_CACHE`（默认 `./datasets`）下的
  `EleutherAI___wikitext_document_level`；缺失直接 `sys.exit(1)`
- backend / 任务 / batch **写死**为 `lm_eval --model vllm --tasks wikitext --batch_size 1`

因此：**vLLM 起不来、或没有离线数据集时，经 `main.py --lm-eval` 这条路走不通**，不要反复重试。
若要换任务、换 backend（如 `hf`）、或改 batch，必须**绕过** `main.py`，直接调 `lm_eval` CLI
或自建脚本——本仓库不保证那条路径，报告里要写明。

### 4.1 回退路径：`utils/torch_infer.py`

vLLM 不可用 / 只想看生成是否退化时，用纯 PyTorch 反量化推理：

```bash
python3 -m utils.torch_infer --model <产物目录>
python3 -m utils.torch_infer --model <产物目录> --prompt "你好,1+1等于几" --max-tokens 64
python3 -m utils.torch_infer --model <产物目录> --no-dispatch
```

- `--prompt` 逗号分隔多个 prompt；不传则跑内置中英测试集
- `--no-dispatch`：`compressed-tensors` 的 `dispatch_model` 报错时使用
- `--dtype` / `--device` 可覆盖（CPU 可跑，但慢）

已知坑（遇到时**先别判产物坏**）：

- 部分 W8A16 / `compressed-tensors` 产物的 `config_groups.*.input_activations` 为 `null` 时，
  `_print_model_info` 会 `AttributeError: 'NoneType' object has no attribute 'get'`
  （`utils/torch_infer.py`）。这是工具打印配置时的空指针，不是权重损坏；可换 GPU/`--lm-eval`，
  或等上游修（`a = gcfg.get(...) or {}`）
- 在**非目标加速器**或纯 CPU 上回退路径偶发乱码输出：**乱码 ≠ 量化失败**。
  先用 `check_artifact` + `model_xray` 核对契约与 dtype，再在目标加速器上复跑或改看 PPL

回退路径只证明"能加载、能生成、无明显退化"，**不能替代 PPL 数值结论**，报告里要写清楚用的是哪条路径。

### 4.2 精度对比报告（ΔPPL / ΔNLL，无阈值）

做完产物契约（§1）与 dtype（§2）之后，若要做精度验证，按下面流程出**对比报告**。
**不对 ΔPPL / ΔNLL 设阈值，也不因 Δ 大小判 PASS/FAIL**；硬门禁仍只在契约 / dtype。

| 符号 | 定义 |
|------|------|
| PPL | lm_eval wikitext 表格里的 `word_perplexity` |
| `PPL_b` / `PPL_q` | baseline / 量化产物的 word_perplexity |
| ΔPPL | `PPL_q − PPL_b` |
| ΔNLL | `ln(PPL_q) − ln(PPL_b)`（nats） |

**完成精度验证的最低交付**：四个数都要有 —— `PPL_b`、`PPL_q`、`ΔPPL`、`ΔNLL`。只报单侧 PPL 不算完成。

Agent 流程：

1. `python3 ./skills/scripts/check_artifact.py <产物目录>` 退出码 0
2. 确认 `HF_DATASETS_CACHE`（默认 `./datasets`）下存在 `EleutherAI___wikitext_document_level`
3. 对 **baseline** 与 **产物** 各跑一次 `--lm-eval`（同一 `tasks=wikitext`、同一离线缓存；
   `tensor_parallel_size` / `dtype` / `gpu_memory_utilization` 尽量一致），从输出表摘 `word_perplexity`
4. 用脚本算差并打印（脚本**不**跑评测、**不**设阈值）：

```bash
# baseline / 产物各跑一次；pretrained= 换成真实目录
python3 main.py --lm-eval "pretrained=<baseline>,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"
python3 main.py --lm-eval "pretrained=<产物>,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"

python3 ./skills/scripts/report_ppl_delta.py \
  --baseline-ppl <PPL_b> --quant-ppl <PPL_q>
```

报告模板（可原样贴给客户）：

```text
PPL_b   = <baseline word_perplexity>
PPL_q   = <quant word_perplexity>
ΔPPL    = PPL_q - PPL_b
ΔNLL    = ln(PPL_q) - ln(PPL_b)   # nats
```

客户问「是否达标」时：说明本 skill **无 Δ 阈值门禁**，只提供数值；若要排查涨点，再进 `05-precision-debug.md`。

## 5. 评测对比基线的纪律

- 对比必须是**同结构同层数**的 BF16 源（或同结构预处理浮点），不能拿全量模型 PPL 去比 mini 产物
- 精度结论必须给出 §4.2 的四个数；缺任一等于没结论
- **不对 ΔPPL / ΔNLL 设阈值**；涨点排查走阶段5 `05-precision-debug.md`，不要直接改方案
