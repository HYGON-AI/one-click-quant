"""Small MXFP4 K3.1 end-to-end EP test, including KDA backward on accelerator."""

import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from utils.auto_quantize.api import analyze_model, validate_plan
from utils.auto_quantize.ep import merge_ep_plans
from utils.auto_quantize.model_load.kimi import (
    KimiRuntime, _KimiShortContextIndexer, _KimiTextModel, _kimi_modeling,
)


def make_fixture(source, destination, cpu):
    destination.mkdir(parents=True, exist_ok=True)
    source_config = json.loads((source / "config.json").read_text())
    data = copy.deepcopy(source_config["text_config"])
    data.update(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=2, moe_intermediate_size=32,
                num_experts=8, num_experts_per_token=2, num_shared_experts=1,
                routed_expert_hidden_size=32, q_lora_rank=32, kv_lora_rank=16,
                qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                attn_res_block_size=2, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    full_layers = [1, 2, 3, 4] if cpu else [4]
    data["linear_attn_config"].update(num_heads=2, head_dim=32,
                                      kda_layers=[] if cpu else [1, 2, 3], full_attn_layers=full_layers)
    data["dsa_config"].update(dsa_layers=full_layers, index_topk=64, index_head_dim=16,
                              index_num_heads=2)
    source_config["text_config"] = data
    source_config["model_type"] = "kimi_k31"
    (destination / "config.json").write_text(json.dumps(source_config))
    (destination / "configuration_kimi_k3.py").write_bytes((source / "configuration_kimi_k3.py").read_bytes())
    modeling = _kimi_modeling(destination)
    construction = copy.deepcopy(data)
    construction.pop("quantization_config", None)
    config = modeling.KimiLinearConfig(**construction)
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    model = _KimiTextModel(modeling.KimiLinearForCausalLM(config)).eval()
    for layer in full_layers:
        model.model.layers[layer - 1].self_attn.indexer = _KimiShortContextIndexer(config)
    weights = {}
    for name, parameter in model.named_parameters():
        if ".experts." in name and name.endswith(".weight"):
            n, k = parameter.shape
            assert k % 32 == 0
            weights[name.removesuffix(".weight") + ".weight_packed"] = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8)
            weights[name + "_scale"] = torch.full((n, k // 32), 120, dtype=torch.uint8)
        else:
            value = parameter.detach().clone()
            if name.endswith("e_score_correction_bias") or name.endswith("dt_bias"):
                value.zero_()
            if name.endswith("A_log"):
                value.zero_()
            weights[name] = value.float() if name.endswith(("A_log", "dt_bias", "e_score_correction_bias")) else value.bfloat16()
    save_file(weights, destination / "model.safetensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true", help="CPU MLA/DSA path; accelerator run also tests KDA")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = "cpu" if args.cpu else f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    if not args.cpu:
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("gloo" if args.cpu else "nccl", timeout=timedelta(minutes=15))
    try:
        if rank == 0:
            make_fixture(args.source, args.work_dir, args.cpu)
        if world > 1:
            dist.barrier()
        runtime = KimiRuntime(args.work_dir, device=device,
                              dtype=torch.float32 if args.cpu else torch.bfloat16,
                              rank=rank, world_size=world, trust_remote_code=True)
        inventory = runtime.global_inventory(exclude=[], include=[], grouping_rules=[]) if rank == 0 else None
        # Reject a length for which dense DSA would cease to be equivalent.
        try:
            runtime.forward(runtime.model, torch.zeros((1, 65), dtype=torch.long))
        except ValueError as error:
            assert "index_topk" in str(error)
        else:
            raise AssertionError("Long DSA sequence was not rejected")
        torch.manual_seed(42 + rank)
        batch = torch.randint(1, 128, (1, 16))
        with runtime:
            local = analyze_model(runtime.model, [batch], effective_bits=4.2, adapter=runtime.adapter,
                                  source_config=runtime.source_config, forward_step=runtime.forward,
                                  _runtime=runtime)
        assert all(n >= 2 for n in runtime.block_loads.values()), runtime.block_loads
        assert all(p.is_meta for p in runtime.model.parameters()), "Source parameters remained resident"
        received = [None] * world if rank == 0 else None
        if world > 1:
            dist.gather_object(local, received, dst=0)
        else:
            received = [local]
        if rank == 0:
            plan = merge_ep_plans(received, inventory, effective_bits=4.2, owners=runtime.owners)
            validate_plan(plan)
            searched = [g for g in plan["groups"].values() if g["status"] == "searched"]
            assert len(searched) == 3
            assert any(g["scores"]["4"] > 0 for g in searched), "No gradient sensitivity measured"
            assert plan["policy"]["activation_bits"] == 8
            assert len(plan["module_bits"]) == 3 * 8 * 3
            summary = {"status": "passed", "world_size": world, "kda_backward": not args.cpu,
                       "searched_modules": len(plan["module_bits"]), "groups": len(searched),
                       "actual_effective_bits": plan["actual_effective_bits"],
                       "block_loads": runtime.block_loads,
                       "scores": {k: v["scores"] for k, v in plan["groups"].items() if v["status"] == "searched"}}
            (args.work_dir / "result.json").write_text(json.dumps(summary, indent=2))
            print("SMOKE PASSED:", json.dumps(summary), flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
