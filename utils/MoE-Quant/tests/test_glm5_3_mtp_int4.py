# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression checks: python tests/test_glm5_3_mtp_int4.py."""

import argparse
import contextlib
import functools
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pack_quantized_model as packer
from src.models.glm_moe_dsa_adapter import GlmMoeDsaAdapter


class MtpInt4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cpu = patch("torch.cuda.is_available", return_value=False)
        cpu.start()
        self.addCleanup(cpu.stop)
        self.adapter = GlmMoeDsaAdapter(SimpleNamespace(
            num_hidden_layers=2, num_nextn_predict_layers=1,
            hidden_size=16, moe_intermediate_size=16, n_routed_experts=4,
            mlp_layer_types=["dense", "sparse"],
        ))
        self.args = argparse.Namespace(
            bits=4, sym=True, group_size=None, dtype="bfloat16",
            activation_bits=8, mtp_int4=True, mtp_int8=False,
            quantize_only_experts=False,
            ignore=self.adapter.resolve_ignore([r"re:.*\.self_attn\.kv_b_proj$"]),
        )

    @staticmethod
    def unpack_int4(packed):
        shifts = torch.arange(8) * 4
        return (((packed.long().unsqueeze(-1) >> shifts) & 15) - 8).flatten(1).to(torch.int8)

    def precision(self, config, name):
        if self.adapter.match_ignore(name, config["ignore"]):
            return None
        for group in config["config_groups"].values():
            if self.adapter.match_ignore(name, group["targets"]) or "Linear" in group["targets"]:
                return group["weights"]["num_bits"]
        self.fail(f"No quantization scheme for {name}")

    def test_int4_round_trip_and_stored_scale(self):
        weight = torch.randn(4, 16).bfloat16()
        weight[0].zero_()
        result = packer.pack_int4_rtn(weight, torch.bfloat16)
        expected_scale = (weight.float().abs().amax(1, keepdim=True).clamp(min=1e-12) / 7).bfloat16()
        expected_q = torch.round(weight.float() / expected_scale.float()).clamp(-8, 7).to(torch.int8)
        self.assertEqual(result["weight_packed"].dtype, torch.int32)
        self.assertEqual(result["weight_scale"].dtype, torch.bfloat16)
        self.assertEqual(result["weight_shape"].tolist(), [4, 16])
        torch.testing.assert_close(self.unpack_int4(result["weight_packed"]), expected_q, rtol=0, atol=0)
        torch.testing.assert_close(result["weight_scale"], expected_scale, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(result["weight_scale"]).all())
        for invalid in (torch.zeros(16), torch.zeros(4, 7)):
            with self.assertRaisesRegex(ValueError, "divisible by 8"):
                packer.pack_int4_rtn(invalid, torch.bfloat16)

    def test_mtp_shards_and_precision_config_agree(self):
        tensors = {
            "model.layers.2.mlp.experts.0.gate_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.mlp.experts.0.up_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.mlp.experts.0.down_proj.weight": torch.randn(128, 128).to(torch.float8_e4m3fn),
            "model.layers.2.mlp.experts.0.down_proj.weight_scale_inv": torch.tensor([[0.5]]),
            "model.layers.2.mlp.shared_experts.up_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.self_attn.q_a_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.self_attn.indexer.wk.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.self_attn.kv_b_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.self_attn.indexer.weights_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.mlp.gate.weight": torch.randn(4, 16).bfloat16(),
            "model.layers.2.eh_proj.weight": torch.randn(8, 16).bfloat16(),
            "model.layers.2.enorm.weight": torch.ones(16, dtype=torch.bfloat16),
            "model.layers.2.shared_head.head.weight": torch.randn(8, 16).bfloat16(),
            # This isn't the MTP in the tiny config and must not be copied.
            "model.layers.78.mlp.experts.0.gate_proj.weight": torch.randn(8, 16).bfloat16(),
        }
        config = packer.prepare_quantization_config(self.args, self.adapter, {
            "model.layers.1.mlp.experts.0.gate_proj": 4,
            "model.layers.0.self_attn.q_a_proj": 8,
        })
        _, ignore = packer.resolve_ignore_for_packing(self.args, self.adapter)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            save_file(tensors, tmp / "source.safetensors")
            weight_map = {key: "source.safetensors" for key in tensors}
            index = {}
            out = tmp / "packed"
            out.mkdir()
            self.assertEqual(self.adapter.count_extra_shards(weight_map), 1)
            next_id = self.adapter.save_extra_weights(
                str(tmp), weight_map, str(out), 1, 1, index,
                mtp_int8_ignore=ignore,
                mtp_pack_int8=functools.partial(packer.pack_int8_rtn,
                    packing_format="int-quantized", dtype=torch.bfloat16),
                mtp_pack_int4=functools.partial(packer.pack_int4_rtn, dtype=torch.bfloat16),
            )
            self.assertEqual(next_id, 2)
            result = load_file(str(out / "model-00001-of-00001.safetensors"))
            self.assertEqual(result.keys(), index.keys())
            for key, source in tensors.items():
                if key.endswith("_scale_inv") or not key.startswith("model.layers.2."):
                    self.assertNotIn(key, result)
                    continue
                name = key[:-len(".weight")]
                bits = self.precision(config, name)
                weight = source.float()
                if key + "_scale_inv" in tensors:
                    weight = weight * tensors[key + "_scale_inv"]
                if bits is None:
                    torch.testing.assert_close(result[key], weight.bfloat16(), rtol=0, atol=0)
                elif bits == 4:
                    self.assertNotIn(key, result)
                    self.assertEqual(result[name + ".weight_packed"].dtype, torch.int32)
                    scale = (weight.abs().amax(1, keepdim=True).clamp(min=1e-12) / 7).bfloat16()
                    expected = torch.round(weight / scale.float()).clamp(-8, 7).to(torch.int8)
                    torch.testing.assert_close(self.unpack_int4(result[name + ".weight_packed"]), expected,
                        rtol=0, atol=0)
                    torch.testing.assert_close(result[name + ".weight_scale"], scale, rtol=0, atol=0)
                else:
                    self.assertEqual(result[key].dtype, torch.int8)
                    self.assertEqual(result[name + ".weight_scale"].dtype, torch.float32)
                    scale = weight.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
                    expected = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
                    torch.testing.assert_close(result[key], expected, rtol=0, atol=0)

    def test_runtime_targets_and_other_mtp_modules(self):
        for backbone_bits in (4, 8):
            config = packer.prepare_quantization_config(self.args, self.adapter, {
                "model.layers.1.mlp.experts.0.gate_proj": backbone_bits,
                "model.layers.0.self_attn.q_a_proj": 8,
            })
            for prefix in ("model.layers.2", "model.decoder"):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    self.assertEqual(self.precision(config, f"{prefix}.mlp.experts.0.{proj}"), 4)
                    self.assertEqual(self.precision(config, f"{prefix}.mlp.shared_experts.{proj}"), 8)
                self.assertEqual(self.precision(config, f"{prefix}.self_attn.q_a_proj"), 8)
                self.assertIsNone(self.precision(config, f"{prefix}.self_attn.kv_b_proj"))
            self.assertIsNone(self.precision(config, "model.eh_proj"))
            self.assertEqual(self.precision(config, "model.layers.1.mlp.experts.0.gate_proj"), backbone_bits)
            self.assertEqual(self.precision(config, "model.layers.3.mlp.experts.0.gate_proj"), 8)
            self.assertEqual(config["config_groups"]["group_int4"]["format"], "pack-quantized")
            self.assertEqual(config["config_groups"]["group_int8"]["format"], "int-quantized")

    def test_ignore_still_wins(self):
        self.args.ignore.append(r"re:.*mlp\.experts\.0\.gate_proj$")
        config = packer.prepare_quantization_config(self.args, self.adapter, {})
        for prefix in ("model.layers.2", "model.decoder"):
            self.assertIsNone(self.precision(config, f"{prefix}.mlp.experts.0.gate_proj"))
            self.assertEqual(self.precision(config, f"{prefix}.mlp.experts.0.up_proj"), 4)

    def test_int8_and_bf16_modes_remain_available(self):
        self.args.mtp_int4 = False
        for int8 in (False, True):
            self.args.mtp_int8 = int8
            config = packer.prepare_quantization_config(self.args, self.adapter, {
                "model.layers.1.mlp.experts.0.gate_proj": 4,
                "model.layers.0.self_attn.q_a_proj": 8,
            })
            self.assertEqual(self.precision(config, "model.layers.2.mlp.experts.0.gate_proj"),
                8 if int8 else None)
            if int8:
                self.assertEqual(self.precision(config, "model.decoder.mlp.experts.0.gate_proj"), 8)

    def test_validation_and_flag_exclusivity(self):
        self.adapter.validate_packing_args(self.args)
        for override in (dict(mtp_int8=True), dict(group_size=16), dict(quantize_only_experts=True)):
            bad = argparse.Namespace(**(vars(self.args) | override))
            with self.assertRaises(ValueError):
                self.adapter.validate_packing_args(bad)
        self.adapter.config.num_nextn_predict_layers = 0
        with self.assertRaisesRegex(ValueError, "num_nextn_predict_layers"):
            self.adapter.validate_packing_args(self.args)
        argv = ["pack_quantized_model.py", "--model_name_or_path", "source",
            "--quantized_model_path", "gptq", "--packed_model_path", "packed"]
        with patch.object(sys, "argv", argv + ["--mtp-int4"]):
            parsed = packer.parse_args()
            self.assertTrue(parsed.mtp_int4)
            self.assertFalse(parsed.mtp_int8)
        with patch.object(sys, "argv", argv + ["--mtp-int4", "--mtp-int8"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                packer.parse_args()
            self.assertEqual(error.exception.code, 2)

    def test_launcher_forwards_only_the_packing_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            bins = tmp / "bin"
            bins.mkdir()
            for executable, output in (("torchrun", "quant"), ("python", "pack")):
                stub = bins / executable
                stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$MTP_CAPTURE_DIR/' + output + '"\n')
                stub.chmod(0o755)
            env = os.environ | {
                "PATH": str(bins) + os.pathsep + os.environ["PATH"],
                "MTP_CAPTURE_DIR": str(tmp), "LOG_DIR": str(tmp / "logs"),
                "SAVE_MODEL_PATH": str(tmp / "intermediate"),
                "PACKED_MODEL_PATH": str(tmp / "packed"),
            }
            for flags in ([], ["--mtp-int4"]):
                result = subprocess.run(["bash", str(ROOT / "tests/run_glm5_3_w4a8.sh"), *flags],
                    env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                quant = (tmp / "quant").read_text().splitlines()
                pack = (tmp / "pack").read_text().splitlines()
                self.assertNotIn("--mtp-int4", quant)
                self.assertNotIn("--mtp-int8", quant)
                self.assertIn("--mtp-int4" if flags else "--mtp-int8", pack)
                self.assertNotIn("--mtp-int8" if flags else "--mtp-int4", pack)


if __name__ == "__main__":
    unittest.main(verbosity=2)
