# 阶段2 模型准备

> **权威来源**：各 `examples/*/build_mini_*.py` 与 `tools/model_xray.py` 的 `--help`；参数以当次 `--help` 为准。
> 本文只写判断顺序、关键 flag 与门禁。
>
> **尚未选定量化路径时**：先走 [00-new-model-playbook.md](00-new-model-playbook.md)
> （判适配 → 已适配菜单或未适配推荐 / 自提评估），客户确认后再做本文的解包 / mini。

## 何时用本文

- 量化方案已定（或客户已确认 `00` 的推荐），需要解包 / 建 mini / 核对结构覆盖
- 不确定权重是 BF16 / block FP8 / MXFP4 / 已量化（也可作为 `00` 取证的补充手段）
- 需要一个小模型来验证量化链路（而不是动辄几 T 的全量权重）
- 复用别的模型的 mini 做冒烟，但担心结构覆盖不全

## 步骤 1：权重取证

```bash
python3 tools/model_xray.py --input_dir <模型目录> --output_dir <输出目录>
```

产出每个分片的 `<shard>.metadata.json`，记录每个 tensor 的 dtype 与 shape。

三个必须知道的行为（都影响怎么用、怎么判）：

- 它逐个 `get_tensor` 读出 dtype/shape，**不是只读 header** —— 不会常驻整个模型，但会加载**当前分片**的全部张量
- `--output_dir` 省略时，JSON **写回模型目录本身**。而 `--alg slimquant_ptq` 要求模型目录里没有非权重杂项
  （见 `03-quant-runbook.md` 3.0），所以**必须显式给 `--output_dir`**，否则下次量化会报
  `OSError: ... does not appear to have a file named <shard>.metadata.json`
- 它**不返回非零退出码**（出错只打印 `❌`）。判据是生成的 JSON 内容，不是 exit code

判据：

| 观察到 | 结论 | 下一步 |
|---|---|---|
| 有 `weight_scale_inv` | block FP8（如 DeepSeek 官方） | 先解包到 BF16 |
| 有 `weight_packed` + `weight_shape` + `weight_scale`，`weight_packed` 是 **int32**，`weight_scale` 是 **bf16** | compressed-tensors WNA16 pack-quantized（MoE-Quant / W4A16） | 解量化或 `--keep-quant` |
| 有 `weight_packed` + `weight_scale`，且 `weight_scale` 是 **uint8** | MXFP4（E8M0 指数，group_size=32） | 解包；Kimi 可用 `build_mini_kimi_k3.py --src-format mxfp4` |
| `config.json` 里 `quantization_config` **或** `compression_config` 的 `quant_method == "slimquant_w4a8"` | slimquant W4A8 产物 | 已是成品，别当源模型再量化 |
| 有 `...experts.<i>.<proj>.weight`（int8）+ 配对 `.weight_scale`（**float32**） | slimquant int4 的 per-expert 形态（本链路唯一产物形态） | 同上 |
| 有 `...experts.gate_up_proj` / `...experts.down_proj` 3D 键，dtype **int8** + 同名 `weight_scale` float32 | int8 的 fused 形态，**不是本仓库链路产出** | 同上，按不变式验收 |
| 有 `quantization_config` / `compression_config`，权重是 int8 / fp8 | 已量化（llmcompressor 或 slimquant int8/fp8） | 同上 |
| dtype 全 bf16/fp16，无 scale | 浮点权重 | 可直接量化 |
| 有 3D 键 `...mlp.experts.gate_up_proj` / `...down_proj`，dtype bf16、无 scale | 浮点权重，且 experts 是 **fused** 形态 | 可直接量化；`slimquant_ptq` 会先拆成逐 expert 再量化，**产物必定是 per-expert 2D** |

判据要点：

- **只看 `weight_packed` 分不出 MXFP4 和 INT4** —— 要看 `weight_scale` 的 dtype（uint8=E8M0 → MXFP4；浮点 → affine）
- **scale 的浮点 dtype 不是固定 float32**：WNA16 产物是 bf16，slimquant 产物是 float32。判「是否 affine 量化」只看"是不是浮点"，别写死 float32
- **slimquant 产物没有 `weight_packed`**：非 MoE 是 `<module>.weight` + `.weight_scale`（float32）；
  experts 在当前链路里**必定是 per-expert 2D**（`...experts.<i>.<proj>.weight`）——
  `templates/slimquant/process.py` 在匹配前**无条件**调用 `split_fused_moe_experts`，
  自仓库首个提交起就是这样
- **fused 3D 只会出现在两处**：源模型的 bf16 裸权重（`...experts.gate_up_proj` / `...experts.down_proj`，无 `.weight` 后缀），
  或其它工具产出的已量化形态（int8 + float32 scale）。本仓库链路两者都会先拆开再量化。
  所以若在某个"产物"里看到 3D int8 experts，它**不是**本仓库链路产出的 ——
  按 `04-verify-eval.md` 第 2 节的三条不变式验收即可，但**不要**据此推断本仓库会产出这种形态
- **判"是不是已量化 / 走了哪条路线"最可靠的是看 `quant_method`**，而不是猜键名形态：
  配置键由 alg/quant_type 决定（int4 → `quantization_config`；int8/fp8 → `compression_config`；
  `examples/` 脚本自定义），键名不是判据，`quant_method` 才是下游加载器真正读的字段
  （对照 `04-verify-eval.md` 1.1 的表）

## 步骤 2：解包 / 反量化分支

| 源格式 | 机制 | 落点 |
|---|---|---|
| DeepSeek block FP8 | recipe 中 `unpack_fp8`（**默认就是 `true`**，设 `false` 才关闭），多卡解包到 `{save-dir}.bf16_staging`，分片 1:1 | `preprocess/unpack/fp8.py` |
| 已量化 W4A16 | `build_mini_*` 可识别 `--src-format w4a16` | `examples/*/build_mini_*.py` |
| 已量化 MXFP4 | 一次性转换脚本，直接读 safetensors 重打包 | `examples/kimi/kimi_k3_mxfp4_to_int4.py` |

- `unpack_fp8_gpus` 可覆盖解包卡数（默认跟随 `--preprocess-gpus`）。
- 极小冒烟模型不想走 staging 时，DeepSeek recipe 可设 `unpack_fp8_inline: true`，在 adapter 内原地解包。

**Kimi MXFP4 没有 preprocess 路径**：`preprocess/adapters/kimi_k3.py` 与 `preprocess/recipes/kimi_k3.yaml`
都不在仓库里，但 `preprocess/pipeline.py` / `preprocess/adapters/__init__.py` 可能仍残留对 `kimi_k3` 的引用，
一旦被调用就会 `ImportError` / `KeyError`。Kimi 走 `examples/kimi/kimi_k3_mxfp4_to_int4.py` 或 `utils/MoE-Quant/`；
要恢复 preprocess 支持需自己实现 adapter，见 `10-ext-preprocess.md`。

> 判据是**目录里有没有这个文件**（`ls preprocess/adapters/`、`ls preprocess/recipes/`），
> 不是任何文档里写的"支持/不支持"。接入或排障前先列一遍目录，不要凭记忆假设某个 adapter 还在。

## 步骤 3：构建减层 mini

脚本清单以 `ls examples/*/build_mini_*.py` 为准，`-e` 默认值以各脚本 `--help` 为准（下表只做概览）。
**脚本与目标模型族必须对上**：例如 `build_mini_qwen3_8.py` 只服务 Qwen3.8 MoE，
不能拿它去减 `Qwen3-30B-A3B`（`Qwen3MoeForCausalLM`）当正式冒烟——dry-run 偶发能过也不算覆盖正确。

| 脚本 | 目标模型 | `-e` 默认 |
|---|---|---|
| `examples/qwen/build_mini_qwen3_8.py` | Qwen3.8 MoE | 0（全部） |
| `examples/deepseek/build_mini_deepseek_v3.py` | DeepSeek-V3 | 8 |
| `examples/glm/build_mini_glm5_3.py` | GLM-5.3 | 0（全部） |
| `examples/kimi/build_mini_kimi_k3.py` | Kimi-K3 | 0（全部） |

公共参数：

- `--src` / `--dst`：源模型目录 / 输出目录
- `--src-format`：`auto`（默认；具体可选值因脚本而异，见「差异」）
- `-n/--num-layers`：保留前 N 层，默认 4
- `--layer-range START:END`：截取源层区间（半开）
- `--layer-ids 0,12,45,92`：显式指定不连续层号（与上两项互斥）
- `-e/--num-experts`：每层保留 expert 数；Qwen3.8 每层 512 experts，快速验证建议 `-e 8`
- `--keep-quant`：保留源量化张量与 `quantization_config`；默认解量化为 BF16

差异（**不要一刀切**）：

- `--src-format` 的可选值各脚本不同（多一个 `mxfp4`、`w8a8` 之类就会报 invalid choice）：
  先 `python3 <脚本> --help` 确认，别把一个脚本的可选值套到另一个上
- `--dry-run`（只统计筛选、不读写 safetensors）：Qwen / DeepSeek / GLM 有，**Kimi 没有**
- `--no-mtp`：仅 Qwen
- `--include-nextn`：DeepSeek / GLM

三个行为要点：

- 保留层会被**重映射为连续** `model.layers.0..N-1`，层号不再是源码里的原编号
- 默认输出纯 BF16：解量化后 `quantization_config` 会被 `pop` 掉（四个脚本都是）
- **`compression_config` 不会被 `pop`**（四个 `build_mini_*` 都不处理这个键）。源模型是
  `compression_config` 风格的量化产物（Kimi / GLM / 多阶段产物）时，默认解量化后 `config.json` 里会**残留**
  一个"声称已量化"的配置，下游会照它去反量化纯浮点权重 —— 交付前必须手工删掉这个键，
  或用 `10-ext-preprocess.md` 的验收判据（两个量化键都不存在）复核

**限制**：脚本读 `config.json` 的**顶层** `num_hidden_layers` / `num_experts`。对 config 嵌套在 `text_config`
里的模型（如 `Qwen3.8-27B`、`Qwen4ExpForConditionalGeneration` 的 `Qwen3.8-Flash-Next-*`）会直接
`KeyError: 'num_hidden_layers'`。扁平 config（如 `Qwen3MoeForCausalLM` 的 `Qwen3-30B-A3B`）可正常 dry-run。

**例外是 Kimi**：`build_mini_kimi_k3.py` 内部用 `config.get("text_config", config)` 回退，嵌套 config 可用，
且顶层与 `text_config` 都会改。其余三族（Qwen / DeepSeek / GLM）不支持嵌套，要先把 `text_config`
的字段拍平到顶层或自行加工 —— 不要以为是参数写错。

```bash
# 例：Qwen3.8 取 4 层 × 8 experts（src 必须是扁平 config，否则先按上面的限制处理）
python3 examples/qwen/build_mini_qwen3_8.py \
  --src <真实源模型目录> --dst <真实输出目录> \
  -n 4 -e 8 --dry-run

python3 examples/qwen/build_mini_qwen3_8.py \
  --src <真实源模型目录> --dst <真实输出目录> \
  -n 4 -e 8
```

## 结构覆盖要求（硬门禁）

> 减层**不等于**取前 N 层。必须覆盖**该模型实际拥有**、且接入路径会碰到的每一种结构单元。
> 候选清单（有则覆盖，无则跳过，不要为不存在的结构硬凑层）：
> MLA attention / dense MLP / MoE experts / shared_experts / MTP / shared_head.norm / linear_attn。

对 **dense 小模型**（无 MoE / 无 MTP / 无 MLA）只需覆盖其真实模块（如 `self_attn` + `mlp` + norm），
对照 `config.json` / `model_xray` 键名勾选即可。对 MoE / 多模态 / MTP 模型，只取前 4 层时常漏掉
晚出现结构，冒烟会假通过——层号不确定时用 `--layer-ids`，或 `--dry-run` 先看筛选结果。

## 门禁

- [ ] `--dst` 下 `config.json` 可解析，`num_hidden_layers` 等于设定值
- [ ] `model_xray` 复查：dtype 全为 bf16/fp16，无 `weight_scale_inv`（看 JSON，不看退出码）
- [ ] 权重文件存在且自洽：多分片时 `model.safetensors.index.json` 的 `weight_map` 指向的**每个分片都真实存在**；
      单分片（小模型常见）只有 `model.safetensors`、**没有 index.json**，这不是失败（判据见 `04-verify-eval.md`）
- [ ] 该模型**实际存在**的结构单元至少各出现一次（无则跳过，有则必须覆盖）
