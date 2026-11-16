# 混合精度多阶段量化

> **权威来源**：各 `examples/<family>/` 下 README 与转换脚本（以目录实际文件为准）。
> 本文分两层：§1–§2 **跨脚本通用约定**；§3 起是 **Qwen3.8-Flash-Next / Kimi 范本**（其它族勿照搬层名）。
> 这类链路是 `examples/` 离线脚本，**不是** `main.py` 的 `--alg` 路径。

## 1. 什么时候需要它

`--alg slimquant_ptq --scheme W4A8` 是**固定规则**（非 MoE 全 INT8 + MoE experts INT4，见 `03` 第 3 节）。
当需要「不同层不同位宽」的定制组合时（如 experts W4、attention/shared_expert W8、MTP 单独处理），
才走多阶段脚本。判定标志：`compression_config.mixed_precision` 下存在
`experts` / `w8a8_include` 这类**分组字段**，而不是只有单一 `quant_method`。

入口怎么找（不要默认只有 Flash-Next）：

```bash
ls examples/                    # 模型族目录
find examples -name "*mixed*" -o -name "*int4*" -o -name "*w4a8*" | head
# 常见：examples/qwen/qwen3_8-flash-next/、examples/kimi/、
#       examples/deepseek/*mixed*、examples/hy4/、examples/mimo/ …
```

选中脚本后以该目录 README 为准；下面「通用约定」适用于多数多阶段脚本，层名/阶段数以脚本为准。

**配置结构**（改配置前先认清层级；字段名因脚本而异，以产物 `config.json` 为准）：

```text
compression_config
├── quant_method: slimquant_w4a8   # 或脚本自定义值
├── activation_scheme: ...
├── ignore: [...]
└── mixed_precision
    ├── experts       {weight_bits: 4, ... modules: [...]}
    └── w8a8_include  {weight_bits: 8, ...}   # 分组名可能不同
```

> 分组在 `mixed_precision` **里面**，不在 `compression_config` 顶层。
> 部分脚本（如 Flash-Next）会把同一对象写进顶层与 `text_config`——手改时**两处都要改**。

## 2. 跨脚本通用约定

1. **有 `--dry-run` 就先扫不写盘**：确认命中层数与 dtype 分布，再真跑（以该脚本 `--help` 为准）
2. **有配对 `-MSE.py` 就每阶段跑**：`--src` / `--dst` 必须显式给（脚本自带 `DEFAULT_*` 是作者环境路径）；
   `--src` 通常是上一阶段产物，不是原始 BF16
3. **产物怎么算「这阶段做完」**：以该脚本 README / 打印的 DONE 文件为准；常见是更新后的
   `config.json` + `model.safetensors.index.json`（若脚本要求 index）。缺文件按脚本报错修，
   不要拿 `04`「单分片无 index 也合法」套到硬性要求 index 的脚本上
4. **无加速器时不要跑写盘转换**（可先 dry-run）。可见设备用当前栈的环境变量：
   `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / 厂商文档指定变量——不要写死某一种
5. **每个阶段写一份完整副本**：磁盘按「阶段数 × 源模型大小」预留；中间产物勿立刻删（下一阶段要读，也要能回退）
6. **验收**：最终档过 `04-verify-eval.md` 第 1、2 节（`check_artifact` + dtype）；中间档按脚本约定。
   记录每阶段脚本名与参数——多阶段产物无法反推来源

## 3. 范本：Qwen3.8-Flash-Next 四阶段

> 以下层名与阶段划分**仅适用于** `examples/qwen/qwen3_8-flash-next/`。
> 其它族（deepseek mixed / hy4 / mimo / kimi）只复用 §2，阶段表以各自 README 为准。

工作目录：`examples/qwen/qwen3_8-flash-next/`。每个阶段「独立脚本 + 配对 `-MSE.py`」。

| Stage | 内容 | 校验 |
|---|---|---|
| 1 | 主网 fused experts → INT4 | `*-MSE.py` |
| 2 | attn QKV / shared_expert / linear_attn → INT8 | `*-MSE.py` |
| 3（可选） | PLE `ngram_embedding` → INT8（`--scale-mode per_tensor/per_channel`） | `*-MSE.py` |
| 4（可选） | MTP → W4 experts + W8 指定层 | `*-MSE.py` |

各阶段保留浮点的部分（**不要事后才发现被量化了**）：

- Stage 1 只动 `language_model.layers.*.mlp.experts.{gate_up_proj,down_proj}`；`mtp.*` 整体保持 BF16
- Stage 2 只动 `self_attn.{q,k,v}_proj`、`mlp.shared_expert.{gate,up,down}_proj`、
  `linear_attn.{in_proj_qkv,in_proj_z}`；`o_proj`/`out_proj`、router、hyper_connection、indexer、
  PLE、visual、embed、`lm_head`、整段 MTP 全部浮点
- Stage 4 只动 MTP 的 experts（INT4）与 `self_attn.{q,k,v}_proj` + `shared_expert.*`（INT8）；
  MTP 的 `o_proj`、router、hyper_connection、`fc_*`、norm 保持 BF16

Stage 2 产物在不需要 ngram/MTP 时即可直接用于推理。

本目录脚本**硬性要求**输入带 `model.safetensors.index.json`；单分片无 index 会
`FileNotFoundError`——小模型单分片产物不在这些脚本支持范围内。

`qwen38_flash_next_int4_to_int4_8.py` 会把同一个 `compression` 写进顶层与 `text_config`，
手改配置时两处都要改。

## 4. 范本：`compression_config` 手术（Kimi / Flash-Next MTP）

多阶段链路的最后一个阶段经常需要**改配置而不改权重**，这是最容易出错的一步。

**Kimi（手动改 `ignore`，必做）**：`examples/kimi/kimi_k3_mxfp4_to_int4.py` 的 `build_output_config`
只写 `compression_config = {"quant_method": "slimquant_w4a8"}` —— **不含 `ignore`**。
跑完必须手编 `config.json`，补上 README 给的 6 条 `ignore`
（`self_attn` / `shared_experts` / `mlp.(gate|up|gate_up|down)_proj` / `lm_head` / `vision_tower` / `mm_projector`）。
不补的话配置声明「全量化」而实际有大量 BF16 层，加载器无法正确反量化。

**Flash-Next MTP（自动改）**：`qwen38_flash_next_mtp_to_w4a8.py` 会做三件事，且**顺序不能反**：

1. 从 `ignore` 里**删掉整段** `re:^mtp\..*`（否则新量化的 MTP 层同时被 ignore，等于白量化）
2. 把 MTP 模块并入 `mixed_precision.experts.modules`（W4）与 `mixed_precision.w8a8_include.modules`（W8）
3. 补**细粒度** MTP ignore：`o_proj`、`mlp.gate`、`shared_expert_gate`、`indexer`、
   `hyper_connection`、`fc_*`、`pre_fc_norm_*`、`q_norm`/`k_norm`

脚本还会保留输入里已有的其他 `mixed_precision` 子块（如 ngram），并逐个 `setdefault` 补齐缺省字段 ——
所以**手改时不要整块替换 `mixed_precision`**，那会把已有的 ngram 等子块一起丢掉。

手改配置时的判据：

- 改完必须用 `04-verify-eval.md` 第 2 节的 dtype 表**逐类核对**：每一组都该有对应的 int8/int4 + scale，
  BF16 层仍是 bf16
- `ignore` 是**正则列表**：`re:` 前缀走正则（`slimquant_ptq` 路径下是 `re.match` **前缀匹配**，
  要整名匹配必须自己加 `$`），裸字符串走**整名相等**。所以 `embed_tokens` 匹配不到
  `model.embed_tokens`，默认表才写成 `re:.*embed_tokens.*`。写错前缀或漏 `$` 都会**静默不生效**
  （精确语义与上游差异见 `03-quant-runbook.md` 第 5 节）
- 分阶段脚本可能留下**过期 scale**（上一轮量化过、这轮改回浮点），脚本会显式丢弃并计入 stats，
  手改配置时不会自动清理，必须自己确认

## 5. 验收

- 有 `-MSE.py` 的阶段：`max` / `worst_rel_l2` 达标（见 `04-verify-eval.md` 第 3 节）
- 最终产物过 `04` 第 1、2 节，并给出与 BF16 源的 PPL（或声明用了 `torch_infer` 回退）
- 记录每个阶段的脚本名与参数
