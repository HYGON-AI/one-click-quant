# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path

from compressed_tensors.quantization import QuantizationScheme


def _clean_ignore_patterns(patterns: list[str]) -> list[str]:
    """Strip ``re:`` and regex wrapping for config.json.

    ``"re:.*mlp.gate$"`` → ``"mlp.gate"``
    ``"re:.*embed_tokens.*"`` → ``"embed_tokens"``
    """
    cleaned = []
    for p in patterns:
        if p.startswith("re:"):
            p = p[3:].removeprefix(".*").removesuffix(".*").removesuffix("$")
        cleaned.append(p)
    return cleaned


def update_slimquant_config(
    save_directory: str | os.PathLike,
    scheme: QuantizationScheme,
    ignore: set[str] | None = None,
    quant_type: str = "int4",
):
    """Write quantization config into the model's config.json.

    Writes different format depending on ``quant_type``:

    - ``"int4"``: ``quant_method: "slimquant_w4a8"``
    - ``"int8"``: ``quant_method: "compressed-tensors"``, format ``int-quantized``
    - ``"fp8"``:  ``quant_method: "compressed-tensors"``, format ``float-quantized``

    :param save_directory: output checkpoint directory
    :param scheme_name: name of the quantization scheme preset
    :param scheme: QuantizationScheme used for quantization
    :param ignore: set of module patterns that were skipped
    :param quant_type: "int8", "fp8", or "int4"
    :param moe_pattern: pattern used to identify MoE expert layers (int4 only)
    """
    config_path = Path(save_directory) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {save_directory}")

    with open(config_path, "r") as f:
        config = json.load(f)

    config.pop("quantization_config", None)
    config.pop("compression_config", None)

    ignore_list = _clean_ignore_patterns(list(ignore) if ignore else [])

    if quant_type == "int4":
        config["quantization_config"] = {
            "activation_scheme": "dynamic",
            "quant_method": "slimquant_w4a8",
            "modules_to_not_convert": ignore_list,
        }
    else:
        qtype = "int" if quant_type == "int8" else "float"
        qformat = "int-quantized" if quant_type == "int8" else "float-quantized"

        weights_scheme = scheme.weights
        weights_args = {
            "dynamic": False,
            "group_size": None,
            "num_bits": weights_scheme.num_bits if weights_scheme else 8,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": weights_scheme.strategy if weights_scheme and weights_scheme.strategy else "channel",
            "symmetric": weights_scheme.symmetric if weights_scheme else True,
            "type": qtype,
        }

        input_activations_scheme = scheme.input_activations
        group_0_config: dict = {
            "targets": scheme.targets,
            "weights": weights_args,
        }

        if input_activations_scheme is not None:
            group_0_config["input_activations"] = {
                "dynamic": True,
                "group_size": None,
                "num_bits": input_activations_scheme.num_bits,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": input_activations_scheme.strategy if input_activations_scheme.strategy else "token",
                "symmetric": input_activations_scheme.symmetric,
                "type": qtype,
            }

        config["compression_config"] = {
            "config_groups": {
                "group_0": group_0_config,
            },
            "format": qformat,
            "ignore": ignore_list,
            "quant_method": "compressed-tensors",
        }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
