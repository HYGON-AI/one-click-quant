# Hy4-preview：主干与 MTP 的 channel-wise GPTQ / W4A8

这是从实际运行版本整理的 Hy4 专用适配。量化仍调用本项目 `src/gptq_loop.py`，不是 LLM Compressor，也不是重新训练。源模型只读，不提交模型权重、校准原文、凭据或历史实验目录。

## 支持范围与限制

- 78 个主干层、1 个 MTP；第 0 层 dense FFN 保留原精度。
- 77 个主干 MoE 和 1 个 MTP 的路由/共享专家 gate/up/down，共 60,138 个投影。
- 对称 INT4 `[-7,7]`，每输出通道一个 FP32 scale，`group_size=None`。
- GPTQ block size 128 是算法处理块，不是量化分组。阻尼依次 0.01/0.03/0.1，零覆盖/分解失败有明确 RTN 回退记录。
- 校准传播使用量化权重的浮点模拟值，**没有联合模拟 A8 误差**。整数推理才采用动态 per-token INT8 `[-127,127]`，down 前重新量化，INT32 累加。
- 注意力、iHC、路由器、嵌入、输出头及其他非目标张量保留原精度。
- 校准只支持 1 或 8 rank；8 rank 是文本并行、加权 Hessian 合并和唯一 owner 转换，不是复制完整 BF16 模型。
- 当前推理适配固定 TP=8、EP=1、PP=1，目标 H20。**不是任意四卡部署方案，也不是官方 vLLM/SGLang 免补丁格式。**
- MTP 量化包含在内；默认服务不启用推测解码。MTP 校准的完整原生 GPU 数值等价性未正式验收。

## 目录与上游兼容性

`quant.py` 和 `pack_quantized_model.py` 是原入口；`src/hy4/` 仅收纳实际使用的适配依赖。`src/hy4/runtime/` 是独立推理支持。模型注册按 `hy_v4` 延迟导入，不让其他架构强制依赖 Hy4 的 Transformers 模块；GLM-5.3-Flash 等原有注册保留。

这一提交优先保留已运行的模块边界，没有把约 200 个历史脚本全部复制进来，也没有为了减少文件数把模型、恢复和内核拼在同一大文件。后续接口重构应单独提交并回归测试。

## 环境

运行于 Linux；源读取使用 POSIX 文件接口。原量化环境：Python 3.12、PyTorch 2.13.0+cu130、Transformers 5.18.0.dev0（必须含 `transformers.models.hy_v4`）、Triton 和 safetensors。必须固定实际依赖构建；仅版本字符串不能唯一定位开发版源码。

已运行量化镜像本地 ID：

`sha256:708168d70285624324f3c993dbe3e875cfdd6afbd474f5bb2c70842b14217db0`

已运行 SGLang 镜像本地 ID：

`sha256:a33925f7c456f133083780ab9a1970e62c6980640c55b7c8b19d707afd3a94ac`

该镜像内 SGLang commit：`dc2157dcd62d5fb1bc5317fcf8765ebfcd8a8dad`。

这些 ID 是复现来源，不是可公开拉取的镜像地址。**本仓库尚不提供从公共基础镜像完整复建原依赖的保证。** 使用前需获取原环境导出的镜像，或自行验证含上述组件的环境；不能用不含 Hy4 的普通 Transformers 替代。

以下命令均在 `utils/MoE-Quant` 目录运行。将路径变量设置到自己的目录，不修改源模型。

## 1. 准备数据与参数清单

优先使用原冻结 corpus（zh/en/math/code.json）和对应 manifest。新数据可用 `prepare_hy4_calibration.py`，输入是四个 JSONL 文件，每行 `{ "id": "...", "text": "..." }`，另有 `sources.json`，例如：

```json
{"zh":{"dataset":"your-dataset","revision":"immutable-revision","split":"train","license":"verified-license"}}
```

实际 sources.json 必须同时包含 zh/en/math/code 四项。用户负责数据许可及与评测题去重；脚本仅做规范化整文去重，不声称能检测训练污染。

```bash
export PYTHONPATH="$PWD"
python prepare_hy4_calibration.py --model "$MODEL" --input "$TEXTS" \
  --output "$CORPUS" --per-domain 32 --max-length 2048
python -m src.hy4.hy4_source --model "$MODEL" --manifest "$TARGET_MANIFEST"
```

新数据准备会显式截取最多 2048 tokens，写入冻结记录；后续校准不重新截断。目标清单只读分片头，不代替全模型 SHA 校验。

## 2. 逐层校准与 GPTQ

先复核 GPU 无资源冲突；容器建议源目录只读挂载。原八卡使用 16 CPU、1 TiB 主机内存限制，每卡任务分配预算 64 GiB，磁盘至少保留 300 GiB。

```bash
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 --master-port=29459 \
  quant.py --model_name_or_path "$MODEL" --dataset_name_or_path "$CORPUS" \
  --calibration_manifest "$CORPUS/manifest.json" \
  --num_calibration_samples 128 --max_sequence_length 2048 \
  --bits 4 --sym --dtype bfloat16 \
  --quantize_scope routed_shared_experts --include_mtp \
  --activation_bits 8 --weight_range narrow \
  --offload_activations --tie_gptq_handles --attn_implementation eager \
  --gpu_memory_budget_gib 64 --disk_reserve_gib 300 --stage_disk_headroom_gib 50 \
  --save_dir "$CANDIDATE"
```

不要传 `--group_size` 或 `--quantize_only_experts`。后者会排除本方案必须量化的共享专家。Hy4 要求 `rel_damp=0.01`、`block_size=128`、`quantization_scale=absmax`、`quantization_order=default`（均为默认值），不支持的设置直接拒绝，实际阻尼重试策略以导出 recipe 为准。

恢复在同一代码/模型/数据/分区下添加 `--resume`。重组源码后哈希发生变化，不能覆盖旧快照恢复旧运行。全局 layer commit 成功后才允许恢复；不凭目录存在跳过。

## 3. 独立打包

```bash
python pack_quantized_model.py --model_name_or_path "$MODEL" \
  --quantized_model_path "$CANDIDATE" --packed_model_path "$PACKED" \
  --dtype bfloat16 --activation-bits 8 \
  --hy4-target-manifest "$TARGET_MANIFEST" \
  --hy4-calibration-manifest "$CORPUS/manifest.json" --preflight
# 预检通过后执行同一命令，去掉 --preflight。
```

输出 `hy4_w4a8_v1`：uint8 packed、FP32 scale、保留张量、索引、配方、覆盖率及来源。偶数输入列低 nibble，奇数列高 nibble；负值补码表示，奇数宽度尾部补零。推理 bank 当前要求实际 Hy4 的偶数宽度。

## 4. 专用 SGLang 推理

从含上述固定 SGLang 源码与依赖的原环境镜像构建代码覆盖层：

```bash
docker build -f inference/hy4/Dockerfile --build-arg BASE_IMAGE="$HY4_BASE_IMAGE" \
  -t hy4-gptq-sglang:local .
docker run --rm --network none --gpus '"device=0,1,2,3,4,5,6,7"' \
  --cpus 32 --memory 256g --shm-size 8g \
  --mount "type=bind,source=$PACKED,target=/checkpoint,readonly" \
  --mount "type=bind,source=$REPORTS,target=/runtime" \
  -e HY4_GPTQ_ROOT=/checkpoint -e HY4_GPTQ_REPORT_DIR=/runtime -e OMP_NUM_THREADS=2 \
  --entrypoint bash hy4-gptq-sglang:local /opt/hy4-bootstrap/serve.sh
```

REPORTS 必须事先创建，且与权重路径分开。服务绑定容器内 127.0.0.1:31108，宿主无法直接访问；本地测试可通过 `docker exec` 或共享容器网络访问。默认 decode CUDA Graph、prefill eager，不开启 MTP 推测解码。

该格式不能使用 `--quantization w4a8_int4`；sitecustomize 显式加载专用适配。入口失败会终止，不静默回落 BF16。当前发布不提供未经新包验证的 MTP 推测解码启动参数。

## 5. 验证与证据边界

```bash
python -m unittest discover -s tests/hy4
python tests/hy4/check_merged_gptq.py
python tests/hy4/check_hf_hy4_expert_capture.py
python tests/hy4/check_hf_hy4_layer_state.py
python tests/hy4/check_hf_hy4_mtp.py
GLOO_SOCKET_IFNAME=lo torchrun --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29589 tests/hy4/check_hf_hy4_distributed_hessian.py
# 以下在显式分配的空闲 GPU 上运行；包含真实权重读取与校验开销。
python tests/hy4/check_gptq_real_integer.py --candidate "$PACKED" --report "$REPORTS/integer.json"
python tests/hy4/check_gptq_grouped.py --candidate "$PACKED" --report "$REPORTS/grouped.json"
```

2026-09-14：整理后的 Windows/Linux 格式、恢复、数据冻结共 4 项 unittest、Linux CPU 专家采集、连续 full/shared 层、MTP 前向及两进程 Gloo 合并通过；两个 CLI help 通过。新包完整量化、GPU 内核及全模型服务尚未重新运行，不能把历史镜像的结果冒充为整理后新包实测。

历史原快照：60,123 GPTQ 投影、15 个零覆盖 RTN；MTP 占 762 GPTQ＋9 RTN。完整 no_think/4K HumanEval 为 154/164，但没有完整 BF16 对照或 PPL 验收，不保证精度损失门槛。新包的模型级精度状态为 `NOT_EVALUATED`。

## 来源

量化代码来自原 `readme-native-full128-v1-code`，打包来自 `readme-native-full128-pack-v1-code`，推理来自上述固定镜像。与原实现相比主要调整包导入、延迟注册、删除本机硬编码并补齐可用命令；未换用新算法。上游 MoE-Quant 作者及第三方组件的许可/署名保持不变。
