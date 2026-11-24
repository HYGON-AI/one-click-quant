# 扩展：slimquant（`--alg slimquant_ptq`）

> **权威来源**：`templates/slimquant/` 源码（`__init__.py` / `process.py` / `helpers.py` / `config.py` / `_vendor.py`）。
> 本文只写**内部执行链、数值细节、往哪加代码**。用法与语义见 `03-quant-runbook.md` 第 3 节。

## 1. 定位

Data-free 权重量化：**不实例化 HF 模型、不用校准数据**，直接读 safetensors 写新 safetensors。
显存需求由**单个分片**决定（逐分片加载到设备），不是整模型；`--device-map cpu` 可纯 CPU 跑。
代价是**无法**用校准激活、hook、GPTQ 之类需要前向的手段。

## 2. 执行链（`slimquant_ptq()` 五步）

```text
① get_checkpoint_files(model_stub)        # os.walk 或 list_repo_files + cached_file
② 非 safetensors 文件原样 copyfile 到 save_directory
③ build_inverse_weight_maps → 每分片一个 job
④ exec_jobs(process_file_slimquant, ...)   # 逐分片量化，默认 max_workers=1（串行）
⑤ update_slimquant_config + update_safetensors_index
```

单分片输入是**支持**的：`get_weight_map` 在没有 index 时回退到 `model.safetensors`
（但那时文件名必须正好是 `model.safetensors`，否则 `ValueError`）。

## 3. 三种 `quant_type`

| `quant_type` | scheme | 权重 dtype | scale dtype | config |
|---|---|---|---|---|
| `int8` | `_INT8_SCHEME`（per-channel INT8 权重 + per-token 动态 INT8 激活声明） | int8 | float32 `[N,1]` | `compression_config`，`compressed-tensors` + `int-quantized` |
| `fp8` | `_FP8_SCHEME`（E4M3） | float8_e4m3fn | float32 `[N,1]` | 同上，`float-quantized` |
| `int4` | **复用 `_INT8_SCHEME`** | 非 MoE int8 / MoE experts 打包 int8 | float32 | `quantization_config`，`quant_method: slimquant_w4a8` |

`int4` 复用 INT8 scheme 是刻意的：INT4 只是在 INT8 之上对 MoE experts 再做一段网格搜索。
`_SCHEME_TO_QUANT_TYPE` 把 CLI 的 `W4A8` / `W8A8` / `FP8_DYNAMIC` 映射到这三类。

写盘前 `update_slimquant_config` 会**先删掉**已有 `quantization_config` / `compression_config`，
所以产物里只会有一个配置块。ignore 的落点也不同：

| `quant_type` | 配置块 | ignore 键名 |
|---|---|---|
| `int4` | `quantization_config` | `modules_to_not_convert` |
| `int8` / `fp8` | `compression_config` | `ignore` |

（`check_artifact.py` 两个键都认，不必记。）

## 4. int4 的数值细节（改代码前必须懂）

对每个 2D 权重：

1. 先做 per-channel INT8：`scale = absmax(w)/127`，`q = clamp(round(w/scale), -127, 127)`
2. 若模块名命中 `moe_pattern`（子串匹配），在**已量化的 int8 值**上做 INT4：
   - 每行按 `|w|` 升序排序，取 `index = int(k * K)` 处的值作为 absmax（`k` = percentile）
   - `scale4 = absmax/7`，`q4 = clamp(round(w/scale4), -8, 7)`
   - `k` 由 `torch.linspace(k_min, k_max, k_steps)` 网格搜索，按 `search_metric`（`mse` / `l2`）取最优
   - 最终写入的 scale 是 **`scale4 * scale8 / 16`**（`/16` 来自 nibble 打包的读出约定）
3. 打包：每两个 int4 塞进一个 int8，**高位是偶数下标、低位是奇数下标**，
   所以 K 维减半；反量化见 `helpers.py::unpack_int4_from_int8`（注意要减 16 的借用处理）

`weight_scale` 已含 `/16`，下游反量化直接 `w_int4 * scale` 即可，不要再除一次。

**全零行是已知边界**：`weight_quant_int8` / `weight_quant_int4` 的 `scale` 没有 `clamp(min=1e-12)`
（只有 `weight_quant_fp8` 有）。某行权重全 0 时 `scale = 0`，`w/0` 得 `nan`，
`nan` 过 `clamp` 后仍是 `nan`，转 int8 结果未定义。剪枝过或空 expert 的模型要留意这一点。

## 5. 扩展点（按改动代价排序）

| 想做的事 | 改哪里 | 注意 |
|---|---|---|
| 某架构默认忽略某些层 | `templates/slimquant_ptq.py::_ARCH_IGNORE`（**外层键是 `config["architectures"][0]`，内层键是 `(hidden_size, num_hidden_layers)`**） | 查不到内层键 → 退 `_WILDCARD_SIGNATURE = (-1, -1)` 通配；连架构都没有 → 退 `_DEFAULT_IGNORE`。签名取值支持 `text_config` 嵌套。用户 `--ignore` 是逗号分隔字符串，与默认表**取并集** |
| 换 MoE 层命名 | `moe_pattern` 参数（`slimquant_ptq()` 已开放） | CLI **没有** `--moe-pattern`；模板里用 `getattr(args, "moe_pattern", ".mlp.experts.")` 兜底，要开放得先在 `slimquant_ptq.py` 注册参数 |
| 新增 `quant_type` | `__init__.py::_SCHEMES` + `_SCHEME_TO_QUANT_TYPE` + `process.py` 的 `if quant_type ==` 分支 | 三处都要加，漏一处会 `ValueError` 或走错分支 |
| 量化前改张量 | 实现 `Converter` 协议（`process` / `validate` / `get_dependencies`）并传给 `slimquant_ptq(converter=...)` | 需同时声明依赖张量，跨分片依赖由 `build_inverse_weight_maps` 处理 |
| 换量化算法 | `helpers.py` 的 `weight_quant_int8` / `weight_quant_fp8` / `weight_quant_int4` | 前两个返回 `(q, scale)`；`weight_quant_int4` 多返回一个 `dequant` 供网格搜索算误差。共同契约：scale 是 `[N,1]` float32 |

前两行是**数据驱动**的（改表即可），后三行是**改逻辑**，改完必须重跑 `04-verify-eval.md` 第 1、2 节。

## 6. 能力边界（不要试图让它做这些）

- **没有 group-wise**：只有 per-channel（一行一个 scale）。需要 group 走 MoE-Quant / llmcompressor
- **没有激活量化校准**：`int8` / `fp8` 的 config 里虽有 `input_activations` 声明，但权重是直接算的，
  不跑校准；真实激活精度靠下游推理框架
- **不实例化模型**：拿不到 module 类型，`targets` 只是名义上的 `["Linear"]`，
  实际靠"张量名以 `.weight` 结尾且不以 `norm`/`layernorm` 结尾"判定
- **experts 会被拆开**：`split_fused_moe_experts` 在匹配前无条件执行（不区分是否 MoE），
  3D 的 `...gate_up_proj` 会变成逐 expert 的 `gate_proj` / `up_proj` 两个 2D 键。
  因此**本链路产物必定是 per-expert 2D**，不存在 fused 3D 产物形态

## 7. 验收

- `config.json` 的 `quant_method` 与 `quant_type` 对应（`int4` → `slimquant_w4a8`）
- 非 MoE 权重 int8 + float32 scale；MoE experts int8 且 **K 维减半**（没减半 = 没命中 `moe_pattern`）
- 不存在残留 `weight_scale_inv`（说明输入是已解包的浮点权重）
- 用 `scripts/check_artifact.py <产物目录>` 一次过完上面几条
