### QMiniMax-M2.5-W8A8-INT8

```bash
vllm v0.19.0 or
http://10.16.1.152:5000/jenkins/model_test_env/sglang:0.5.12-ubuntu22.04-dtk26041-py3.10-scalex40-sf_b020-20260810-0214
pip install -i https://mirrors.aliyun.com/pypi/simple llmcompressor==0.11.0
pip install -i https://mirrors.aliyun.com/pypi/simple numpy==1.25.0
python3.10 -m pip install -U "huggingface-hub>=1.5.0,<2.0" "kernels>=0.12.0,<0.13"
python3.10 -m pip install "huggingface-hub>=0.34.0,<1.0"
python3 main.py --model /models/MiniMax-M2.5 --scheme FP8_TO_BF16 --save-dir /models/MiniMax-M2.5-BF16
python main.py --model /models/MiniMax-M2.5-BF16 --scheme W8A8 --alg ptq --needs-data false --ignore 'lm_head,re:.*moe.gate+,re:.*moe.e_score_correction_bias+' --save-dir /models/MiniMax-M2.5-W8A8
```

### MiniMax M3.1 Flash native quantization

Run from `examples/minimax` with Python 3.10+, PyTorch with FP8 support and `safetensors`. Use fresh output directories.

```bash
src=/models/MiniMax-M3.1-Flash
out=/models/m31-exports

# Main W4A8
python quantize_minimax_m31_native.py --source "$src" \
  --recipe w4a8 --device cuda:0 --output "$out/main-w4a8"

# DSpark FP8 or direct INT8
python quantize_minimax_m31_native.py --source "$src/dspark" \
  --recipe dspark-fp8 --output "$out/dspark-fp8"

python quantize_minimax_m31_native.py --source "$src/dspark" \
  --recipe dspark-int8 --output "$out/dspark-int8-direct"
```

#### DSpark Smooth INT8

For fresh calibration, export the BF16 draft, follow the [server setup](collect_minimax_m31_dspark_activations.py) and run the collector. With saved calibration files, skip to the Smooth INT8 export.

```bash
acts="$out/dspark-activations"

python quantize_minimax_m31_native.py --source "$src/dspark" \
  --recipe dspark-bf16 --output "$out/dspark-bf16-calibration"

# Start the calibration server before running the collector.
python collect_minimax_m31_dspark_activations.py \
  --dataset /datasets/wikitext_document_level-train.arrow \
  --tokenizer "$src" --output "$acts"

python quantize_minimax_m31_dspark_smooth.py \
  --source "$src/dspark" --activations "$acts" \
  --calibration-manifest "$acts/calibration_manifest.json" \
  --output "$out/dspark-int8-smooth"
```

Pass the exported draft to SGLang with `--speculative-draft-model-path`.


### MiniMax-M3.1 NVFP4 → channel FP8 转换验证

依赖：`torch`、`safetensors`、`compressed-tensors`（已验证版本分别为
2.11.0、0.8.0、0.17.0）。在仓库根目录运行：

```bash
python examples/minimax/nvfp4_to_channel_fp8_probe.py \
    --model /path/to/MiniMax-M3.1 \
    --threads 4 \
    --output ./minimax-m3.1-nvfp4-channel-fp8.json
```

仅在 CPU 上验证 9 个 NVFP4 专家矩阵，输出 JSON 误差报告；不导出模型，
不处理原生 FP8 权重或激活 A8。输入约定和验证细节见脚本顶部说明。
