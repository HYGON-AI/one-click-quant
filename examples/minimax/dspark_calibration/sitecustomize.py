# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in DSpark activation hooks for an isolated, graph-disabled SGLang server."""

import importlib.abc
import importlib.machinery
import os
import sys
from functools import wraps
from pathlib import Path

TARGET = "sglang.srt.models.minimax_m3_dspark"


def install(module):
    import torch

    cls = module.DSparkMiniMaxDraftModel
    original_init = cls.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        output = Path(os.environ["M31_CALIB_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        enabled = output / "collect.enabled"
        trigger = output / "dump.trigger"
        destination = output / f"activations_{os.getpid()}.pt"
        stats = {}
        dumped = False

        def collect(name, value):
            nonlocal dumped
            if dumped or not enabled.is_file():
                return
            if trigger.is_file():
                if len(stats) == 11:
                    payload = {
                        key: dict(
                            entry,
                            amax=entry["amax"].cpu(),
                            samples=torch.cat(entry["samples"]).cpu(),
                        )
                        for key, entry in stats.items()
                    }
                    temporary = destination.with_suffix(".tmp")
                    torch.save(payload, temporary)
                    temporary.rename(destination)
                    dumped = True
                    stats.clear()
                return
            if not value.numel():
                return
            value = value.detach().reshape(-1, value.shape[-1])
            maximum = value.float().abs().amax(0)
            entry = stats.setdefault(
                name, dict(amax=maximum, samples=[], calls=0, rows=0)
            )
            entry["amax"] = torch.maximum(entry["amax"], maximum)
            entry["calls"] += 1
            entry["rows"] += value.shape[0]
            indices = torch.linspace(
                0, value.shape[0] - 1, min(4, value.shape[0]), device=value.device
            ).long()
            sample = value.index_select(0, indices)
            if len(entry["samples"]) < 64:
                entry["samples"].append(sample)
            elif entry["calls"] % 16 == 0:
                entry["samples"][(entry["calls"] // 16) % 64] = sample

        names = {
            f"layers.{index}.{projection}"
            for index in range(5)
            for projection in ("self_attn.qkv_proj", "mlp.gate_up_proj")
        }
        modules = dict(self.named_modules())
        if not names <= modules.keys():
            raise ValueError(
                f"Missing DSpark calibration modules: {sorted(names - modules.keys())}"
            )
        for name in sorted(names):
            modules[name].register_forward_pre_hook(
                lambda layer, args, key=name: collect(key, args[0])
            )
        self.hidden_norm.register_forward_hook(
            lambda layer, args, value: collect("hidden_norm", value)
        )

    cls.__init__ = init


class CalibrationLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        install(module)


class CalibrationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.loader:
            spec.loader = CalibrationLoader(spec.loader)
        return spec


if os.environ.get("M31_CALIB_OUTPUT"):
    sys.meta_path.insert(0, CalibrationFinder())
