"""Step-5 routing, EP ownership, source decoding, and export regression checks."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models import get_model_adapter
from src.models.step5_utils import Step5MoE, Step5RMSNorm
from src.models.step5_checkpoint import pack_checkpoint, pack_artifact


def config(ep_size=1):
    return SimpleNamespace(
        ep_size=ep_size, hidden_size=8, moe_num_experts=4, moe_intermediate_size=8,
        moe_top_k=2, norm_expert_weight=True, moe_router_scaling_factor=3.0,
    )


def fill(model):
    # Global IDs define weights; EP ranks must see identical expert functions.
    with torch.no_grad():
        model.gate.weight.copy_(torch.arange(32).reshape(4, 8).float().sin() * 0.3)
        model.router_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.1]))
        for eid, expert in enumerate(model.experts):
            if expert is not None:
                for i, p in enumerate(expert.parameters()):
                    p.copy_((torch.arange(p.numel()).reshape(p.shape) + eid * 7 + i).float().cos() * 0.2)


def ep_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        distributed = Step5MoE(config(2), 0).eval()
        reference = Step5MoE(config(1), 0).eval()
        fill(distributed)
        fill(reference)
        x = torch.arange((rank + 2) * 8).reshape(1, rank + 2, 8).float().cos()
        with torch.no_grad():
            actual, expected = distributed(x), reference(x)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        owned = [i for i, expert in enumerate(distributed.experts) if expert is not None]
        assert owned == list(range(rank * 2, rank * 2 + 2)), owned
        assert all(f"experts.{eid}." in " ".join(distributed.state_dict()) for eid in owned)
    finally:
        dist.destroy_process_group()


class Step5Tests(unittest.TestCase):
    def test_norm_and_selection_only_router_bias(self):
        norm = Step5RMSNorm(8, 1e-5)
        norm.weight.data.fill_(-0.25)
        x = torch.arange(16).reshape(2, 8).float() / 10
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * 0.75
        torch.testing.assert_close(norm(x), expected)
        moe = Step5MoE(config(), 0)
        fill(moe)
        ids, weights = moe._route(x)
        scores = F.linear(x, moe.gate.weight).sigmoid()
        wanted_ids = (scores + moe.router_bias).topk(2, dim=-1).indices
        torch.testing.assert_close(ids, wanted_ids)
        selected = scores.gather(1, ids)
        torch.testing.assert_close(weights, selected / selected.sum(-1, keepdim=True) * 3)

    def test_variable_token_ep_matches_all_experts(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(ep_worker, args=(f"file://{root}/rendezvous",), nprocs=2, join=True)

    def test_fp8_bank_decode_and_full_streaming_export(self):
        with tempfile.TemporaryDirectory() as root:
            source, intermediate, output = [Path(root) / name for name in ("source", "gptq", "output")]
            source.mkdir()
            intermediate.mkdir()
            ignore = [r"re:.*lm_head$", r"re:.*self_attn.*", r"re:.*share_expert.*",
                      r"re:.*\.moe\.gate$", r"re:.*vision_model.*", r"re:.*\.mlp.*"]
            raw = {
                "model_type": "step5_vl",
                "text_config": {
                    "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                    "vocab_size": 16, "moe_num_experts": 4, "moe_intermediate_size": 8,
                    "moe_layer_list": [0], "moe_top_k": 2,
                },
                "quantization_config": {
                    "quant_method": "compressed-tensors", "ignore": ignore,
                    "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                        "num_bits": 8, "type": "float", "strategy": "channel",
                        "symmetric": True, "dynamic": False,
                    }}},
                },
            }
            tensors = {
                "lm_head.weight": torch.arange(128).reshape(16, 8).bfloat16(),
                "model.layers.1.mlp.gate_proj.weight": torch.ones(16, 8, dtype=torch.bfloat16),
                "vision_model.weight": torch.arange(8).float(),
            }
            for proj in ("gate_proj", "up_proj", "down_proj"):
                name = f"model.layers.0.moe.{proj}"
                tensors[name + ".weight"] = torch.arange(256).reshape(4, 8, 8).remainder(8).to(torch.float8_e4m3fn)
                tensors[name + ".weight_scale"] = torch.arange(1, 33).reshape(4, 8, 1).float() / 100
            save_file(tensors, str(source / "model.safetensors"))
            (source / "config.json").write_text(json.dumps(raw))
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {k: "model.safetensors" for k in tensors},
            }))
            c = AutoConfig.from_pretrained(source)
            adapter = get_model_adapter(c)
            key = "model.layers.0.moe.experts.2.gate_proj.weight"
            state = adapter.materialize_state_dict(tensors, [key], torch.bfloat16)
            expected = tensors["model.layers.0.moe.gate_proj.weight"][2].float() * tensors["model.layers.0.moe.gate_proj.weight_scale"][2]
            torch.testing.assert_close(state[key], expected.bfloat16())
            shapes = {}
            for plan_item in adapter.plan:
                module = plan_item.output.removesuffix(".weight")
                if module in adapter.layer_bits:
                    _, item_shape = adapter.meta[plan_item.source]
                    shapes[module] = item_shape[1:] if plan_item.expert is not None else item_shape
            for name in adapter.layer_bits:
                bits = adapter.layer_bits[name]
                center = 1 << (bits - 1)
                rows, cols = shapes[name]
                directory = intermediate / name
                directory.mkdir()
                torch.save({
                    "bits": bits, "qweight": torch.arange(rows * cols).reshape(rows, cols).remainder(2 * center).byte(),
                    "scale": torch.ones(rows, 1, dtype=torch.bfloat16) / center,
                    "zero": torch.full((rows, 1), center, dtype=torch.bfloat16),
                }, directory / "quantized_weight.pt")
            args = SimpleNamespace(quantized_model_path=str(intermediate), packed_model_path=str(output), max_shard_size_mb=1)
            pack_checkpoint(adapter, args)
            saved = json.loads((output / "config.json").read_text())
            # lm_head is quantized (int8), so compression_config() drops its ignore rule.
            self.assertEqual(saved["quantization_config"]["ignore"], [r for r in ignore if r != r"re:.*lm_head$"])
            index = json.loads((output / "model.safetensors.index.json").read_text())["weight_map"]
            self.assertEqual(sum(k.endswith("weight_packed") for k in index), 12)
            self.assertIn("lm_head.weight", index)
            self.assertNotIn("lm_head.weight_packed", index)
            self.assertFalse((output / ".incomplete").exists())
            for key in ("model.layers.1.mlp.gate_proj.weight", "vision_model.weight"):
                with safe_open(str(output / index[key]), framework="pt") as handle:
                    torch.testing.assert_close(handle.get_tensor(key), tensors[key], rtol=0, atol=0)
            name = next(iter(adapter.layer_bits))
            shape = shapes[name]
            packed = pack_artifact(intermediate / name / "quantized_weight.pt", name, shape, adapter.layer_bits[name])
            shifts = torch.arange(8, dtype=torch.int64) * 4
            restored = ((packed[name + ".weight_packed"].long().unsqueeze(-1) >> shifts) & 15).reshape(8, 8)
            torch.testing.assert_close(restored, torch.arange(64).reshape(8, 8).remainder(16))


if __name__ == "__main__":
    unittest.main()
