from __future__ import annotations

__all__ = ["RMSNorm"]

import onnx_ir as ir
import numpy as np
import torch
from torch import nn
from torch_onnx_models.components._rms_norm_utils import apply_rms_norm
from torch_onnx_models import BuilderModule
from torch_onnx_models._builder import get_current_builder


class RMSNorm(BuilderModule):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # Mark: weights
        self.weight = ir.Tensor(np.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        weight = self.builder.initializer(self.weight, name="weight")
        return apply_rms_norm(
            x=hidden_states, weight=weight, eps=self.variance_epsilon
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
