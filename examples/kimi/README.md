### Kimi-K3-W4A8-INT4

```bash
cd /one-click-quant-main
python examples/kimi/kimi_k3_mxfp4_to_int4.py \
  --input-path /llm_models/Kimi-K3 \
  --dry-run

python examples/kimi/kimi_k3_mxfp4_to_int4.py \
  --input-path /llm_models/Kimi-K3 \
  --output-path  /llm_models/Kimi-K3-INT4

cp /llm_models/Kimi-K3/tiktoken.model \
   /llm_models/Kimi-K3-INT4/
```

在 `/llm_models/Kimi-K3-INT4/config.json` 中找到 `compression_config` 字段，替换为：

```json
"compression_config": {
  "quant_method": "slimquant_w4a8",
  "ignore": [
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*"
  ]
},
```

也可以直接用脚本替换：

```bash
python3 - <<'PY'
import json

path = "/llm_models/Kimi-K3-INT4/config.json"
with open(path, "r", encoding="utf-8") as f:
    config = json.load(f)

config["compression_config"] = {
    "quant_method": "slimquant_w4a8",
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
}

with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"updated: {path}")
PY
```
