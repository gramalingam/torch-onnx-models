from __future__ import annotations

__all__ = ["RMSNorm"]

import onnx_ir as ir
import numpy as np
import torch
from torch import nn
from onnx_models.components._rms_norm_utils import apply_rms_norm
from onnx_models import BuilderModule
from onnx_models._builder import get_current_builder
import onnx_models.utils as utils

class RMSNorm(BuilderModule):
    def __init__(self, hidden_size: int, eps: float = 1e-6, name: str | None = None):
        super().__init__(name)
        self.weight = utils.make_external_tensor("weight", ir.DataType.FLOAT, (hidden_size,))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return apply_rms_norm(
            x=hidden_states, weight=self.weight, eps=self.variance_epsilon
        )

