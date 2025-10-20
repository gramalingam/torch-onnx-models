from __future__ import annotations

import onnx_ir as ir

from onnx_models._builder import OpBuilder

# this uses the float32 as the data type until the final multiplication with weight
# TODO(jambayk): expose dtype as an argument if needed
def apply_rms_norm(
    op: OpBuilder,
    *,
    x: ir.Value,
    weight: ir.Value,
    eps: float = 1e-5,
) -> ir.Value:
    """
    Apply RMS Normalization to the input hidden states.

    This function normalizes the input hidden states using the RMS normalization technique,
    which scales the input by the root mean square of its elements, followed by a learnable
    weight parameter.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, hidden_size).
        weight (ir.Value): The learnable weight tensor of shape (hidden_size,).
        eps (float): A small value to avoid division by zero (default is 1e-6).

    Returns:
        ir.Value: The normalized hidden states with the same shape as input.
    """
    # assumes opset 23 will be used during export
    return op.RMSNormalization(x, weight, epsilon=eps)

def apply_rms_norm_decomposed(
    op: OpBuilder,
    *,
    x: ir.Value,
    weight: ir.Value,
    eps: float = 1e-5,
) -> ir.Value:
    """
    Apply RMS Normalization to the input hidden states.

    This function normalizes the input hidden states using the RMS normalization technique,
    which scales the input by the root mean square of its elements, followed by a learnable
    weight parameter.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, hidden_size).
        weight (ir.Value): The learnable weight tensor of shape (hidden_size,).
        eps (float): A small value to avoid division by zero (default is 1e-6).

    Returns:
        ir.Value: The normalized hidden states with the same shape as input.
    """
    # x_dtype = x.dtype
    # x = x.to(torch.float32)
    # variance = x.pow(2).mean(-1, keepdim=True)
    # x = x * torch.rsqrt(variance + eps)
    # return weight * x.to(x_dtype)
    raise NotImplementedError("Decomposed RMSNorm not yet implemented.")


def apply_rms_norm_contrib(
    op,
    *,
    x: ir.Value,
    weight: ir.Value,
    eps: float = 1e-5,
) -> ir.Value:
    """
    Apply RMS Normalization to the input hidden states.

    This function normalizes the input hidden states using the RMS normalization technique,
    which scales the input by the root mean square of its elements, followed by a learnable
    weight parameter.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, hidden_size).
        weight (ir.Value): The learnable weight tensor of shape (hidden_size,).
        eps (float): A small value to avoid division by zero (default is 1e-6).

    Returns:
        ir.Value: The normalized hidden states with the same shape as input.
    """
    
    # SimplifiedLayerNormalization is a contrib op but it is miscongured as ai.onnx in ORT
    return op.SimplifiedLayerNormalization(x, weight, epsilon=eps)