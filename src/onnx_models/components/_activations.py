# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import math
from collections import OrderedDict

import onnx_ir as ir

from onnx_models import BuilderModule, OpBuilder

logger = logging.getLogger(__name__)

# Note: As activations do not have trained weights, we can map them to (model builder) functions
# rather than (model builder) classes.

def GELUTanh(op: OpBuilder, input: ir.Value) -> ir.Value:
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """
    return op.Gelu(input, approximate="tanh", _version=20)


def GELUActivation(op: OpBuilder, input: ir.Value) -> ir.Value:
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return op.Gelu(input)


def QuickGELUActivation(op: OpBuilder, input: ir.Value) -> ir.Value:
    # TODO: return input * torch.sigmoid(1.702 * input)
    raise NotImplementedError("QuickGELUActivation is not implemented yet.")


def MsftQuickGELUActivation(op: OpBuilder, input: ir.Value) -> ir.Value:
    return op.QuickGelu(input, _domain="com.microsoft")

class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_fast": GELUTanh,
    "gelu_new": GELUActivation,
    "gelu_pytorch_tanh": GELUTanh,
    "quick_gelu": QuickGELUActivation,
}
_ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in _ACT2FN:
        return _ACT2FN[activation_string]
    else:
        return GELUTanh
        # raise KeyError(
        #     f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        # )


# gelu_new = get_activation("gelu_new")
# gelu = get_activation("gelu")
# gelu_fast = get_activation("gelu_fast")
# quick_gelu = get_activation("quick_gelu")
# silu = get_activation("silu")
# mish = get_activation("mish")
# linear_act = get_activation("linear")
