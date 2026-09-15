# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch import Tensor


__all__ = ["inv_sym"]


def inv_sym(X: Tensor):
    """
    More efficient and stable inversion of symmetric matrices.
    """
    return torch.cholesky_inverse(torch.linalg.cholesky(X))