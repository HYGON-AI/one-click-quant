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

These examples start from the original MiniMax M3.1 Flash checkpoint. Run them from `examples/minimax` with Python 3.10+, PyTorch with `float8_e4m3fn` support and `safetensors`. Use a fresh directory for each export. The main model is processed one shard at a time and requires disk space for the output and RAM for a full output shard.

| Recipe | Weight conversion | Runtime activation quantization |
| --- | --- | --- |
| Main W4A8 | NVFP4 experts → per-channel packed INT4 with weight-MSE clipping. MXFP8 linears → per-channel INT8. | Dynamic per-token INT8 |
| DSpark FP8 | MXFP8 → per-channel E4M3FN | Dynamic per-token FP8 |
| DSpark direct INT8 | MXFP8 → per-channel INT8 | Dynamic per-token INT8 |
| DSpark Smooth INT8 | Calibrate on the BF16 draft, fold scales into RMSNorm and the affected weight columns, then export per-channel INT8. | Dynamic per-token INT8 |

The export scripts reconstruct native matrices in FP32 from the packed weights and scales. W4A8 uses FP64 for the clipping search and rounding. Only the calibration path materializes a BF16 draft. Unquantized tensors keep their source dtype, except for the DSpark norm weights updated by smoothing, which stay BF16 after the fold. The main exporter writes SlimQuant metadata and the draft exporters write compressed-tensors metadata for compatible SGLang loaders.

#### Export the main model and optional draft variants

```bash
src=/models/MiniMax-M3.1-Flash
out=/models/m31-exports

python quantize_minimax_m31_native.py --source "$src" \
  --recipe w4a8 --device cuda:0 --output "$out/main-w4a8"

python quantize_minimax_m31_native.py --source "$src/dspark" \
  --recipe dspark-fp8 --output "$out/dspark-fp8"

python quantize_minimax_m31_native.py --source "$src/dspark" \
  --recipe dspark-int8 --output "$out/dspark-int8-direct"
```

`--device cpu` is also supported. The W4A8 output contains the main model. Select the draft separately through SGLang's `--speculative-draft-model-path`.

#### Collect and export DSpark Smooth INT8

Use Smooth INT8 for the calibrated DSpark recipe. To reuse existing calibration, supply the rank files `activations_*.pt` and their `calibration_manifest.json` directly to the Smooth exporter.

To collect new activations, first export the BF16 draft used for calibration.

```bash
python quantize_minimax_m31_native.py --source "$src/dspark" \
  --recipe dspark-bf16 --output "$out/dspark-bf16-calibration"
```

Use an isolated SGLang calibration server with the W4A8 main model and this BF16 draft. In the launcher, prepend the absolute path to `dspark_calibration` to the effective `PYTHONPATH` **after** the runtime paths are assigned. Set `M31_CALIB_OUTPUT` to a new shared directory that the server and client can both access. Disable prefill and decode graphs. The reference calibration used DP8/EP8, context length 163840, chunk size 4096, max total tokens 524288, and static memory fraction 0.90.

Run the client with `aiohttp`, `pyarrow` and `transformers` installed. It uses 128 WikiText training pages and routes requests across all eight ranks.

```bash
acts="$out/dspark-activations"
python collect_minimax_m31_dspark_activations.py \
  --dataset /datasets/wikitext_document_level-train.arrow \
  --tokenizer "$src" --output "$acts" --dp-size 8 \
  --url http://127.0.0.1:11010

python quantize_minimax_m31_dspark_smooth.py \
  --source "$src/dspark" --activations "$acts" \
  --calibration-manifest "$acts/calibration_manifest.json" \
  --output "$out/dspark-int8-smooth"
```

Use `dspark-int8-smooth` as the draft model path for inference. The exporter records the calibration inputs and selected fold settings in its local `quantization_report.json`.
