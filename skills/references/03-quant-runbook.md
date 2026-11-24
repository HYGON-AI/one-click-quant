# 阶段3 量化执行手册

> **权威来源**（本文与代码冲突时，以这些为准）：
> `python3 main.py --help`、仓库根 `README.md`（含「常见故障排查」）、`utils/MoE-Quant/README.md`、
> `templates/quant_templates.yaml`。
> 本文只写**路径怎么选、命令怎么写、哪些是坑**。
>
> **方案未定**：先走 [00-new-model-playbook.md](00-new-model-playbook.md)（判适配 / 菜单 / 自提评估 / 未适配推荐）
> 让客户确认主方案，再回到本文写具体命令。未确认前不要擅自选定 `--alg` / `--scheme` 开跑。

## 1. 路径选择

| 目标 | 推荐路径 |
|------|----------|
| BF16 → per-channel FP8 | `--scheme FP8_DYNAMIC` |
| BF16 → per-channel INT8 | `--scheme W8A8` |
| Block FP8 → BF16 | `--scheme FP8_TO_BF16` |
| DeepSeek-V3 block FP8 → 逐通道 INT8/FP8 | `--scheme DeepseekV3_FP8_TO_INT8` / `DeepseekV3_FP8_TO_FP8` |
| Kimi W4 → 逐通道 INT8/FP8 | `--scheme Kimi_W4_TO_INT8` / `Kimi_W4_TO_FP8` |
| Qwen3.5 MoE BF16 → INT8 | `--scheme Qwen35_MOE_BF16_TO_INT8` |
| 快速免数据 PTQ | `--alg model_free_ptq`（配合 scheme） |
| INT8 base + MoE INT4 打包 | `--alg slimquant_ptq --scheme W4A8` |
| 先做浮点预处理再量化 | `--alg preprocess` → 再用 slimquant/model_free 并加 `--from-preprocessed` |
| 单机超大 MoE GPTQ（数据校准） | `utils/MoE-Quant/`（见第 4 节） |
| 模型专属的一次性转换脚本 | `examples/<family>/` |

自定义 scheme 通过 `templates/quant_templates.yaml` 解析（架构 + 尺寸签名）。更多 scheme 可参考 compressed-tensors 上游的 `quant_scheme`。

> **`--help` 里的 scheme 清单不是全集**：`main.py` 的 `--scheme` help 只列了 8 个，
> 但 `llmc_oneshot.py` 的配方表里还有 `W4A16` / `W4A16_ASYM` / `W4A8` / `W4AFP8`（都走 `smoothquant_gptq`），
> 只有落在 `quant_templates.yaml` 里的才会显示成独立 scheme。判断"支持不支持"看配方表与 YAML，别只看 help。

## 2. 最简命令

路径由用户提供。不清楚就先问，或用 `$MODEL` 模板 —— 不要编造 `/models/...`，也不要扫盘。

```bash
MODEL="<请替换为实际 HF 目录>"

# 通用 FP8 / INT8
python3 main.py --model "$MODEL" --scheme FP8_DYNAMIC --save-dir "${MODEL}-FP8_DYNAMIC"
python3 main.py --model "$MODEL" --scheme W8A8        --save-dir "${MODEL}-W8A8"

# 强制指定算法（覆盖 scheme 推导）
# model_free_ptq 的 --model 必须是相对路径（见 3.0）；下面假设已按 3.0 建好软链
python3 main.py --model .models/clean-model --scheme W8A8 --alg model_free_ptq \
  --save-dir "${MODEL}-W8A8-mf"
python3 main.py --model "$MODEL" --scheme W8A8 --alg ptq

# SlimQuant W4A8 —— --model 必须是相对路径，见 3.0 节
python3 main.py --alg slimquant_ptq --scheme W4A8 \
  --model .models/clean-model --save-dir "${MODEL}-W4A8"

# 两段式：预处理 → 量化
python3 main.py --alg preprocess \
  --model "$MODEL" --save-dir "${MODEL}-preprocessed" \
  --recipe preprocess/recipes/<model>.yaml \
  --transforms quarot,flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8

# 第二步同样要相对路径（先按 3.0 节把 -preprocessed 链进 .models/）
python3 main.py --alg slimquant_ptq --scheme W4A8 \
  --model .models/clean-preprocessed --save-dir "${MODEL}-W4A8" \
  --from-preprocessed
```

> **`--save-dir` 可以用绝对路径**（内部走 `shutil.copyfile`，不经过 `cached_file`）。
> **`--model` 路径规则**：
> - **必须相对路径**：`slimquant_ptq`、`model_free_ptq`（见 3.0）
> - **其它 alg**：默认可绝对路径；一旦出现 `does not appear to have a file named`，
>   优先按 3.0 建相对软链再跑（同一类 `cached_file` 问题），不要先怀疑权重损坏

> `--alg preprocess` **必须显式给 `--recipe`**：省略时会落到一个仓库里并不存在的默认 recipe，
> 直接 `FileNotFoundError`。先 `ls preprocess/recipes/` 看有哪些可用。见 `10-ext-preprocess.md`。

不指定 `--save-dir` 时默认落在 `"{--model}-{--scheme}"`（slimquant 为 `-slimquant-<type>`），见第 5 节。
`--from-preprocessed` 会做浮点权重自检，失败即退出 —— 不要把量化产物再喂给它。

## 3. slimquant_ptq 的 INT4 语义（**最容易理解错的一处**）

> 本节只讲**语义与坑**；内部执行链、INT4 数值细节、新 `quant_type` 怎么加 → `12-ext-slimquant.md`。

### 3.0 相对路径与干净软链（`slimquant_ptq` / `model_free_ptq` 共用）

两条 alg 都走 `get_checkpoint_files` → `transformers.utils.cached_file`，因此有相同硬约束：

1. **必须在仓库根目录运行**（`main.py` / 模板用相对路径加载）。
2. **`--model` 必须是相对路径，不能是绝对路径**。绝对路径经 `transformers.utils.cached_file`
   时常退化成 `OSError: /abs/path does not appear to have a file named <basename>`
   （在常见 transformers 版本上复现过；以当次报错为准，不要死记某一大版本号）。
3. **模型目录里不能有断链软链接或非权重杂项文件**（`os.walk` 会一并交给 `cached_file`）。
   报错里点出的那个名字通常就是多余项。

满足方式：在**仓库根**建只含必要文件的软链目录。**必须在仓库根展开 `$SRC` 的 glob**——
先 `cd` 进空目录再写 `*.safetensors` 会匹配不到源分片，导致只链上 config/tokenizer、权重全缺。

```bash
# SRC = 源模型绝对路径；--model 只用相对路径
SRC="<源模型绝对路径>"
cd <仓库根>
mkdir -p .models/clean-model
for f in config.json generation_config.json model.safetensors.index.json \
         tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
  [ -e "$SRC/$f" ] && ln -sf "$SRC/$f" .models/clean-model/
done
# 分片 glob 必须对 SRC 展开，不要在空的 clean-model 里展开
ln -sf "$SRC"/*.safetensors .models/clean-model/

python3 main.py --alg slimquant_ptq --scheme W4A8 \
  --model .models/clean-model --save-dir <产物目录>
# model_free_ptq 同理：--model .models/clean-model
```

### 3.1 精度语义

`--alg slimquant_ptq --scheme W4A8` **不是**"所有层都 INT4"。实际是两段式：

> **先分清两条同名路线**：不带 `--alg slimquant_ptq` 时，`--scheme W4A8` 走的是 `llmc_oneshot.py`
> 里的 `smoothquant_gptq`（SmoothQuant + GPTQ 的 4bit weight / int8 act），跟下面这套
> "INT8 base + MoE INT4 打包"**完全不是同一个东西**。判"是哪条路线"只看 `config.json` 的 `quant_method`：
> llmcompressor 那条是 compressed-tensors 风格（`pack-quantized` + `group_size`），
> 本节这条是 `quant_method: slimquant_w4a8`。**不要用键名形态来区分**（见下）。

| 层类型 | 精度 | 落盘形态 |
|---|---|---|
| **非 MoE** 的所有 Linear | **per-channel INT8** | `<module>.weight`（int8）+ `<module>.weight_scale`（**float32 `[N,1]`**） |
| **MoE routed experts** | **INT4**（percentile 网格搜索，`k∈[k_min,k_max]`） | 2 个 int4 打包进 1 个 int8，**K 维减半** + 配对 float32 scale。**键形态见下（本链路产物固定 per-expert 2D）** |

**experts 的键形态：源侧可能 fused，产物侧必定 per-expert**（判据见下）：

| 形态 | 键名 | 形状 |
|---|---|---|
| **fused**（源权重裸 3D，或其它工具产出的已量化形态；**不是本链路产物**） | `...experts.gate_up_proj` / `...experts.down_proj`（**无 `.weight` 后缀**）+ 同名 `.weight_scale` | 3D `[E, N, K/2]`；scale `[E, N, 1]` |
| **per-expert**（`process.py` 调 `split_fused_moe_experts` 后，即本链路产物） | `...experts.<i>.<proj>.weight` + `.weight_scale` | 2D `[N, K/2]`；scale `[N, 1]` |

**不变式只有三条**（不依赖键形态）：experts 权重 dtype 是 int8；K 维相对源减半；有同前缀配对的 **float32** scale。

要点：

- `quant_method` 写为 `slimquant_w4a8`，所以叫 "W4A8"，但非 MoE 底座是 INT8
- **fused experts 一定被拆开**：`process.py` 在匹配前**无条件**调用 `split_fused_moe_experts`
  （自仓库首个提交起就是这样），所以**本链路产物必定是 per-expert 2D**。
  源 `...experts.gate_up_proj` `[E, 2I, H]` → `gate_proj` + `up_proj` 两半，各自 `[I, H/2]`；
  源 `...experts.down_proj` `[E, H, I]` → `...experts.<i>.down_proj.weight` `[H, I/2]`。
  若在某个"产物"里看到 fused 3D int8 + scale，说明它**不是**本仓库链路产出的；
  仍可接受为量化产物，但判据只用下面三条不变式，不能拿"有没有 `.weight` 后缀"或"键数是否变多"当判据
- 同理 `config.json` 的键名由 **alg/quant_type** 决定，不是随版本变：`--alg slimquant_ptq` 的 int4 一定写
  `quantization_config`（`update_slimquant_config` 里按 `quant_type` 分支写死，自仓库首个提交起未变）；
  int8/fp8 一定写 `compression_config`。若见到 `compression_config` + `quant_method: slimquant_w4a8`，
  说明产物来自 `examples/` 的多阶段脚本或 Kimi 转换脚本，**不是** slimquant 的 int4 路径。
  **唯一稳定的判据仍是 `quant_method == "slimquant_w4a8"`**
- INT4 打包靠 `templates/slimquant/helpers.py` 的 `pack_int4_to_int8`；`weight_scale` 已含 `/16`，
  反量化时要乘回（反量化参考 `templates/slimquant/helpers.py::unpack_int4_from_int8`）
- MoE 层靠 `moe_pattern` 命中，**默认值是 `.mlp.experts.`，且 `main.py` 没有 `--moe-pattern` 参数**
  （只有 `getattr(args, "moe_pattern", ...)` 兜底）。所以 MoE 命名不是 `.mlp.experts.` 的模型
  （如 `.block_sparse_moe.experts.`）**不会触发 INT4，会静默降级成"全 INT8"**——
  产物看起来"成功了"，但没有 W4。判据：产物里 experts 的 dtype 是 int8 且 K 维减半才算走了 INT4；
  K 维没变就是没命中。要支持别的命名只能改代码，不是加参数。
- `ignore` 的默认表来自 `templates/slimquant_ptq.py::_ARCH_IGNORE`，**两级键**：外层 `config["architectures"][0]`、
  内层 `(hidden_size, num_hidden_layers)` 签名；内层查不到用 `_WILDCARD_SIGNATURE = (-1,-1)`，
  外层架构都没有才用 `_DEFAULT_IGNORE`。默认表是 `lm_head`、`re:.*mlp.gate$`、`re:.*embed_tokens.*`；
  用户 `--ignore` 与默认表**取并集**。int4 产物一定含 `modules_to_not_convert`（`update_slimquant_config`
  的 int4 分支写死）；`examples/` 链路的产物用 `ignore` 字段。字段缺失说明产物不是本 alg 产出的，
  判据应回到权重 dtype
- 以 `norm` / `layernorm` 结尾的模块**自动跳过**（`match_quantizable_tensors` 内建），与 `ignore` 无关
- 非 safetensors 文件（tokenizer、generation_config 等）会**原样复制**到输出目录；索引与 config 由
  `update_safetensors_index` / `update_slimquant_config` 重写
- 直接读 safetensors，**不实例化 HF 模型**；但**逐分片把张量加载到设备**上算：
  `--device-map` 为默认 `auto` 时会被模板换成 `cuda:0`，所以显存需求 ≈ **单个分片**规模（不是整模型）。
  纯 CPU 跑须显式 `--device-map cpu`（慢很多，但可行）—— 不要以为它"只在内存里跑"
- 只是 INT8/FP8 需求时别走这条：用 `FP8_DYNAMIC` / `W8A8`，除非模型不被 transformers 支持

## 4. 单机超大 MoE GPTQ：`utils/MoE-Quant/`

需要**数据校准**的 INT4（GPTQ）才用它，分两步。**两条命令都以 `utils/MoE-Quant/` 为工作目录**——
两个脚本都是 `from src import ...`，不在该目录下跑会 `ModuleNotFoundError`（或设 `PYTHONPATH=utils/MoE-Quant`）。
接新模型 / 改 adapter → `13-ext-moe-quant.md`。

```bash
cd utils/MoE-Quant

# ① 逐层 GPTQ（EP 并行省显存，支持 --resume）
# NPROC / MASTER_PORT / 校准集名为示例，按机器卡数与本地数据改
NPROC="${NPROC:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
CALIB_SET="${CALIB_SET:-open-platypus}"   # 或本地数据集路径
torchrun --nnodes=1 --nproc-per-node="$NPROC" --master_port "$MASTER_PORT" quant.py \
  --model_name_or_path "$MODEL" \
  --dataset_name_or_path "$CALIB_SET" \
  --num_calibration_samples 512 --max_sequence_length 4096 \
  --bits 4 --group_size 128 --rel_damp 0.1 --sym \
  --quantize_only_experts \
  --dtype bfloat16 --save_dir "$QUANTIZED_MODEL_PATH"

# ② 打包成 compressed-tensors 产物；activation-bits 16 → W4A16，8 → W8A8（要求 ① 用 --bits 8）
python3 pack_quantized_model.py \
  --model_name_or_path "$MODEL" \
  --quantized_model_path "$QUANTIZED_MODEL_PATH" \
  --packed_model_path "$QUANTIZED_MODEL_PATH-packed" \
  --dtype bfloat16 --activation-bits 16
```

- **W4A16 / W8A8 是打包阶段决定的**（`--activation-bits`），不是 `quant.py` 阶段
- **不存在 W4A8 这一档**：`--activation-bits 8` 走 `int-quantized` 格式，而
  `pack_weight` 里硬性要求 `bits == 8`，`--bits 4 --activation-bits 8` 会直接
  `ValueError: int-quantized format only supports 8-bit weights`。要 W4A8 走第 3 节的 `slimquant_ptq`
- `--bits` / `--group_size` / `--quantize_only_experts` 从 ① 的 `metadata.pt` 读回并**覆盖**② 的 CLI 值
- 模型适配在独立注册表 `src/models/registry.py`（**与 `preprocess/adapters/` 不是同一套**），顺序敏感：更具体的 adapter 必须排在前面
- 支持哪些模型**以 registry 为准**：`grep -n -A10 "_ADAPTER_CLASSES:" src/models/registry.py`
- 各 adapter 的 bit 约束不同（**以 adapter 内的报错信息为准**，不要按模型族记忆）：
  - DeepSeek-V3 / Qwen3.5-3.8 MoE / Kimi-K3：仅 `--bits 4`，且只保证 **W4A16**
  - GLM-5.3 / GLM-5.3-Flash：`--bits 4`（W4A16）或 `--bits 8`（W8A16）；W8A8 打包要求 `--bits 8`
  - **这条链路没有 W4A8**：`--activation-bits 8` 需配 `--bits 8`，否则打包直接报错。要 W4A8 请走第 3 节的 `slimquant_ptq`
- 新模型接入方式见 **`13-ext-moe-quant.md`**（adapter 契约 + 注册顺序 + 位宽约束写法），不要改 `quant.py` 主流程

> 以上路径相对 `utils/MoE-Quant/`（即上面 `cd` 之后的位置）；在仓库根引用时前面加 `utils/MoE-Quant/`。

## 5. 常用覆盖参数

| 参数 | 适用场景 |
|------|----------|
| `--needs-data true\|false` | 强制 / 跳过校准数据集 |
| `--dataset` | 校准数据路径（默认 `./datasets/ultrachat_200k`） |
| `--ignore` | 见下方"`--ignore` 语义分路径"，三条路径行为不同，别当成一回事 |
| `--pipeline sequential` + `--sequential-targets Linear` | 省显存（更慢） |
| `--device-map cpu` / `--device-map auto_offload` | 省显存（`auto_offload` 先落 CPU/磁盘，按需 onload） |
| `--offload-hessians` | GPTQ 类路径计算 hessians 时省显存 |
| `--patch-for-llmc` | `save_pretrained` 阶段 OOM |
| `--print-model` | 加载后打印模型结构 |
| `torchrun --nproc_per_node=N main.py …` | 多卡（llmcompressor；>100B 建议 N=2 或 4） |

### `--ignore` 语义分路径（**三套代码，别当成一回事**）

| 路径 | `--ignore` 行为 | 依据 |
|---|---|---|
| 默认 alg / llmcompressor（`FP8_DYNAMIC`、`W8A8`、`W4A16`…） | **整体替换**内置默认值 | `llmc_oneshot.py`：`if args.ignore: ignore = [...]` / `else: ignore = [默认]` |
| `--alg model_free_ptq` | **整体替换**，且它有自己的 `_ARCH_IGNORE` / `_DEFAULT_IGNORE` | `templates/model_free_ptq.py`：`if args.ignore: ignore = [...]` |
| `--alg slimquant_ptq` | **并集**：先取架构默认，再 `update(用户列表)` | `templates/slimquant_ptq.py`：`ignore.update(user_ignore)` |

`model_free_ptq` 与 llmcompressor 的默认列表**内容还不一样**（前者多 `mtp.pre_fc_norm_embedding` 等项），
所以「默认就会忽略哪些层」必须看当次 log 里打印的 `ignore=[...]`，不要凭记忆。

匹配规则（**只对 `slimquant_ptq` 路径确认过**，实现见 `templates/slimquant/_vendor.py::_match_name`）：

- 带 `re:` 前缀 → `re.match`，即**前缀匹配**；要整名匹配必须自己加 `$`（默认表的 `re:.*mlp.gate$` 就是这么写的）
- 不带 `re:` → **整名相等**，不是子串包含。所以 `embed_tokens` 匹配不到 `model.embed_tokens`，
  默认表才用 `re:.*embed_tokens.*`
- 匹配对象是**模块名**（去掉末尾 `.weight`），不是张量名

> llmcompressor 与 `model_free_ptq` 的 ignore 匹配走**上游** `compressed_tensors/utils/match.py`，
> 本仓库不保证与 `_match_name` 逐字一致（`_match_name` 是自研实现，见 `THIRD_PARTY_NOTICES.md`）。
> 这两条路径上的实际效果**以产物 dtype 为准**核对，不要只凭模式写法推断。
> 共同点是 `re:` 前缀约定，可先按上面写，再用 `04-verify-eval.md` 第 2 节的 dtype 表验证。

结论：
- slimquant 路径想"救回"某层，加进 `--ignore` 即可，默认项不会被挤掉
- 另外两条路径这么写会把默认保护**全部丢掉**，必须自己把需要的默认项一起写上

> **config.json 里的 ignore 已被"清洗"，不要抄回去当 `--ignore` 用**：
> `templates/slimquant/config.py::_clean_ignore_patterns` 写盘时会把
> `re:.*mlp.gate$` → `mlp.gate`、`re:.*embed_tokens.*` → `embed_tokens`（去掉 `re:`、前缀 `.*`、尾部 `.*`/`$`）。
> 于是 `modules_to_not_convert` / `ignore` 里看到的是**裸名字**，而裸名字是**整名相等**匹配，
> 抄回去只能匹配到字面同名的模块，`model.layers.0.mlp.gate` 匹配不上。要复用请自己补回 `re:.*...$`。

### `--save-dir` 省略时的落点

不是"随便一个目录"，而是按 alg 拼出来（日志会 `warning` 打出来）：

| alg | 默认落点 |
|---|---|
| 通用 / llmcompressor | `{--model}-{--scheme}` |
| `slimquant_ptq` | `{--model}-slimquant-{quant_type}` |
| `preprocess` | `{--model}-preprocessed` |

即 `--model /data/Qwen3-4B --scheme W8A8` → `/data/Qwen3-4B-W8A8`。
**默认值会写在 `--model` 同级目录**，所以 `--model` 是相对路径时产物也落在相对位置，容易找不到 —— 建议始终显式给 `--save-dir`。

## 6. 跑完立刻做什么

不要直接报"完成"。先过 `references/04-verify-eval.md` 的产物契约与 dtype 校验，
再做 PPL / 推理对比。方案与产物 dtype 不符时，回查 `--ignore` 与 `targets`。

PPL 命令（baseline 与产物各跑一次，摘 `word_perplexity` 后算差，见 `04` §4.2）：

```bash
python3 main.py --lm-eval "pretrained=$SAVE,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"
python3 ./skills/scripts/report_ppl_delta.py --baseline-ppl <PPL_b> --quant-ppl <PPL_q>
```

OOM → 降 `gpu_memory_utilization`；vLLM 不可用时改用 `utils/torch_infer.py`（见 `04-verify-eval.md`）。
`--lm-eval` 强制离线且要求本地已有 wikitext 缓存，缺数据集会直接退出 —— 先读 `04-verify-eval.md` 第 4 节
的前置条件，不要把它当成"多试几次就能过"。ΔPPL / ΔNLL **无阈值**，只展示。

## 7. examples 索引

模型族目录以 `ls examples/` 为准（下表会随仓库演进过期）；每个目录下通常有 README 与可跑脚本：

| 模型族 | 目录 |
|--------|------|
| DeepSeek | `examples/deepseek/` |
| Qwen / Flash-Next | `examples/qwen/` |
| GLM | `examples/glm/` |
| Kimi | `examples/kimi/` |
| MiMo / MiniMax / Hy4 | `examples/mimo/`、`examples/minimax/`、`examples/hy4/` |
| 预处理说明 | `examples/preprocess/` |
