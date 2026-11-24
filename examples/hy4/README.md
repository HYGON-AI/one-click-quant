### Hy4-preview-Channel-INT4-w4a8

```bash
  cd one-click-quant
  python3 examples/hy4/quantize_hy4_preview_mixed_int4_int8.py \
    --input-dir /models/Hy4-preview \
    --output-dir /models/Hy4-preview-Channel-INT4-w4a8 \
    --int8-mlp --int8-mla --int8-indexer-wq-b --int8-indexer-wk
```

### Hy4-flash-Channel-FP8-w8a8

```bash
  python3 quantize_hy4_flash_fp8_channel.py --input-dir /models/Hy4-flash --output-dir /models/Hy4-flash-channel-fp8
```

### Hy4-flash-Channel-INT4-w4a8

```bash
  python quantize_hy4_flash_mixed_int4_int8.py \
    --input-dir /models/Hy4-flash \
    --output-dir /models/Hy4-flash-Channel-INT4-w4a8 \
    --int8-attn --int8-linear-gate --int8-dense-mlp --int8-shared-experts
```

### Hy4-Channel-FP8-w8a8

```bash
  python3 quantize_hy4_fp8_channel.py \
    --input-dir /models/Hy4 \
    --output-dir /models/Hy4-Channel-FP8-w8a8 \
    --fp8-o-proj --fp8-linear-gate --fp8-dfly

# OR

  python3 quantize_hy4_fp8_channel.py \
    --input-dir /models/Hy4 \
    --output-dir /models/Hy4-Channel-FP8-w8a8 \
    --fp8-o-proj --fp8-linear-gate --fp8-dfly \
    --fp8-mla --fp8-indexer-wq-b --fp8-mlp --fp8-indexer-wk
```

### Hy4-Channel-INT4-w4a8

```bash
  python3 quantize_hy4_mixed_int4_int8.py \
    --input-dir /models/Hy4 \
    --output-dir /models/Hy4-Channel-INT4-w4a8 \
    --int8-mlp --int8-mla --int8-o-proj --int8-dfly \
    --int8-indexer-wq-b  --int8-indexer-wk --int8-linear-gate
```
