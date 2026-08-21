from typing import List, Dict, Optional, Union, Optional, Sequence, Tuple

import re
import numpy as np
import torch.nn as nn


LINEAR_LAYERS = (nn.Linear,)


def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])


def select_layers(
    model: nn.Module,
    layer_prefix: Optional[str] = "",
    layer_regex: str = ".*",
    layer_classes: Union[nn.Module, List[nn.Module]] = nn.Module,
) -> Dict[str, nn.Module]:
    layers = {}
    for layer_name, layer in model.named_modules():
        if (
            isinstance(layer, layer_classes)
            and re.search(layer_regex, layer_name)
            and layer_name.startswith(layer_prefix)
        ):
            layers[layer_name] = layer
    return layers
