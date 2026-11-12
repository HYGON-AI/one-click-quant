"""Read-only source/prepared-output checks for the Step-5 W4A8 GPTQ recipe."""

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import get_model_adapter
from src.models.step5_checkpoint import expected_schema, validate_schema
from src.models.step5_utils import Step5DecoderLayer
from src.models.step5_adapter import get_reference_converter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--packed")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(args.source, trust_remote_code=True)
    adapter = get_model_adapter(config)
    # Check every decoder's logical/source shape mapping without materializing
    # the multi-hundred-billion-parameter model or reading tensor payloads.
    count = 0
    for i in range(config.text_config.num_hidden_layers):
        with torch.device("meta"):
            block = Step5DecoderLayer(config.text_config, i)
        for relative, tensor in block.state_dict().items():
            logical = f"model.layers.{i}.{relative}"
            physical = adapter.checkpoint_keys_for_model_key(logical, adapter.weight_map)[0]
            if physical not in adapter.meta:
                raise ValueError(f"Missing source key: {logical} -> {physical}")
            dtype, shape = adapter.meta[physical]
            if ".moe.experts." in logical and physical != logical:
                shape = shape[1:]
            if tuple(shape) != tuple(tensor.shape):
                raise ValueError(f"Source shape mismatch: {logical}, {shape}, {tuple(tensor.shape)}")
            count += 1
        del block
    assert len(adapter.default_ignore_rules()) == 12
    print(f"Step5 source OK: {count} logical decoder tensors; {len(adapter.layer_bits)} GPTQ targets; original 12 ignore rules", flush=True)
    if args.packed:
        root = Path(args.packed)
        if (root / ".incomplete").exists():
            raise ValueError("Step5 output is still incomplete")
        index = json.loads((root / "model.safetensors.index.json").read_text())
        validate_schema(root, index["weight_map"], expected_schema(adapter))
        saved = json.loads((root / "config.json").read_text())
        expected_config = adapter.raw_config.copy()
        expected_config["quantization_config"] = get_reference_converter().compression_config(
            adapter.raw_config, adapter.plan
        )
        if saved != expected_config:
            raise ValueError("Packed Step5 config differs from the source-preserving GPTQ recipe")
        print(f"Step5 packed output OK: {root}; {len(index['weight_map'])} tensors", flush=True)


if __name__ == "__main__":
    main()
