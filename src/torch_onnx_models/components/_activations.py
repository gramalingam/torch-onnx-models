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

import torch
from torch import ir.Value, nn

from torch_onnx_models import _barrier

logger = logging.getLogger(__name__)


class GELUTanh(BuilderModule):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def forward(self, input: ir.Value) -> ir.Value:
        return self.op.Gelu(input, approximate="tanh", _version=20)


class GELUActivation(BuilderModule):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: ir.Value) -> ir.Value:
        return self.op.Gelu(input)


@_barrier.with_barrier(
    {
        "region": "quick_gelu",
    }
)
def quick_gelu(input: ir.Value) -> ir.Value:
    return input * torch.sigmoid(1.702 * input)


@_barrier.with_barrier(
    {
        "region": "quick_gelu",
    }
)
def quick_gelu_msft(input: ir.Value) -> ir.Value:
    return torch.onnx.ops.symbolic(
        "com.microsoft::QuickGelu",
        [input],
        dtype=input.dtype,
        shape=input.shape,
        version=1,
    )


class QuickGELUActivation(BuilderModule):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: ir.Value) -> ir.Value:
        return quick_gelu(input)


class MsftQuickGELUActivation(BuilderModule):
    def forward(self, input: ir.Value) -> ir.Value:
        return quick_gelu_msft(input)


class ClippedGELUActivation(BuilderModule):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: ir.Value) -> ir.Value:
        return torch.clip(nn.functional.gelu(x), self.min, self.max)


class AccurateGELUActivation(BuilderModule):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: ir.Value) -> ir.Value:
        return (
            0.5
            * input
            * (
                1
                + torch.tanh(
                    self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))
                )
            )
        )


class MishActivation(BuilderModule):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def forward(self, input: ir.Value) -> ir.Value:
        return nn.functional.mish(input)


class LinearActivation(BuilderModule):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: ir.Value) -> ir.Value:
        return input


class LaplaceActivation(BuilderModule):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(input))


class ReLUSquaredActivation(BuilderModule):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = self.op.Mul(relu_applied, relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": GELUTanh,
    "gelu_new": GELUActivation,
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
}
_ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in _ACT2FN:
        return _ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


# gelu_new = get_activation("gelu_new")
# gelu = get_activation("gelu")
# gelu_fast = get_activation("gelu_fast")
# quick_gelu = get_activation("quick_gelu")
# silu = get_activation("silu")
# mish = get_activation("mish")
# linear_act = get_activation("linear")
