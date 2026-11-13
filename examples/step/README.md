### Step5 混合精度 GPTQ W4A8 / W8A8

在仓库根目录运行，按需设置模型路径和本地校准数据文件：

```bash
MODEL_PATH=/path/to/step5-fp8 DATASET=/path/to/calibration.jsonl RECIPE=experts \
  bash utils/MoE-Quant/tests/run_step5_w4a8_gptq.sh --run
MODEL_PATH=/path/to/step5-fp8 DATASET=/path/to/calibration.jsonl RECIPE=mixed \
  bash utils/MoE-Quant/tests/run_step5_w4a8_gptq.sh --run
```

| 配方 | 路由专家 | Attention / 普通 MLP / 共享 MLP | lm_head / MTP | 默认打包产物 |
|---|---|---|---|---|
| `experts` | INT4 GPTQ | BF16 | BF16 | `step-5-preview-channel-w4a8-GPTQ` |
| `mixed`（默认） | INT4 GPTQ | INT8 GPTQ | INT8 RTN | `step-5-preview-int4-int8-GPTQ` |

两种配方均保留 router、norm、embedding、eh_proj、视觉模块、DSA 索引器及 88–90 层路由专家的源精度。
量化范围由配方决定，不接受 `--force_int8`。默认输出在 `/quant_models`，中间结果目录与打包产物同名并追加 `-intermediate`。
设备和输出路径可以用 `HIP_VISIBLE_DEVICES`、`NPROC_PER_NODE`、`SAVE_MODEL_PATH` 和 `PACKED_MODEL_PATH` 覆盖。
