from __future__ import annotations

import onnx_ir as ir
import torch
from torch import nn
from torch_onnx_models._builder import get_current_builder, get_current_op_builder


def get_rotary_pos_emb(
    position_ids: ir.Value, cos_cache: torch.Tensor, sin_cache: torch.Tensor
) -> tuple[ir.Value, ir.Value]:
    """
    Retrieve the cosine and sine positional embeddings based on the provided position IDs.

    Args:
        position_ids (ir.Value): The position IDs tensor of shape (batch_size, seq_length).
        cos_cache (ir.Value): The cosine cache tensor of shape (max_position_embeddings, head_dim).
        sin_cache (ir.Value): The sine cache tensor of shape (max_position_embeddings, head_dim).

    Returns:
        tuple[ir.Value, ir.Value]: A tuple containing the cosine and sine embeddings,
                                           each of shape (batch_size, seq_length, head_dim).
    """
    # Get the current builder
    builder = get_current_builder()
    if builder is None:
        raise RuntimeError("No active IRModelBuilder found in context.")
    
    # Convert torch tensors to ONNX initializers
    tensor = ir.Tensor(cos_cache, dtype=ir.DataType.FLOAT)
    cos_cache_init = builder.initializer(tensor, "cos_cache")
    sin_cache_init = builder.initializer(ir.Tensor(sin_cache, dtype=ir.DataType.FLOAT), "sin_cache")
    
    # Use ONNX Gather operations instead of nn.functional.embedding
    cos_embeddings = builder.op_builder.Gather(cos_cache_init, position_ids, axis=0)
    sin_embeddings = builder.op_builder.Gather(sin_cache_init, position_ids, axis=0)
    
    return cos_embeddings, sin_embeddings


# TODO(jambayk): add support for interleaved format if needed
# requires torch 2.9+
# this can actually be fused with get_rotary_pos_emb as well but we keep them separate for clarity
# as well as to align the model architecture with transformers
# is it really worth separating this from get_rotary_pos_emb? shape doesn't match transformers
# where position embeddings are (batch_size, seq_length, rotary_embedding_dim)
def apply_rotary_pos_emb(
    *,
    x: ir.Value,
    position_embeddings: tuple[ir.Value, ir.Value],
    num_heads: int,
    rotary_embedding_dim: int = 0,
) -> ir.Value:
    """
    Apply Rotary Positional Embedding (RoPE) to the input hidden states.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        position_embeddings (tuple[ir.Value, ir.Value]): The cosine and sine position embeddings.
            Each tensor should be of shape (batch_size, seq_length, rotary_embedding_dim // 2).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        ir.Value: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    op = get_current_op_builder()
    (cos, sin) = position_embeddings
    return op.RotaryEmbedding(x, cos, sin, num_heads=num_heads, rotary_embedding_dim=rotary_embedding_dim)



def apply_rotary_pos_emb_decomposed(
    *,
    x: ir.Value,
    position_embeddings: tuple[ir.Value, ir.Value],
    num_heads: int,
    rotary_embedding_dim: int = 0,
) -> ir.Value:
    """
    Apply Rotary Positional Embedding (RoPE) to the input hidden states.

    This function modifies the input hidden states by applying the RoPE transformation
    using the provided cosine and sine caches based on the given position IDs.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        position_embeddings (tuple[ir.Value, ir.Value]): The cosine and sine position embeddings.
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        ir.Value: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    batch_size, seq_length, _ = x.shape
    x = x.reshape(batch_size, seq_length, num_heads, -1)
    # doing conditionals so that the graph is cleaner when rotary_embedding_dim is 0
    if rotary_embedding_dim == 0:
        x_rot, x_pass = x, None
    else:
        x_rot, x_pass = x[..., :rotary_embedding_dim], x[..., rotary_embedding_dim:]

    cos, sin = position_embeddings
    cos = cos.unsqueeze(2)  # (batch_size, seq_length, 1, head_dim)
    sin = sin.unsqueeze(2)  # (batch_size, seq_length, 1, head_dim)

    x1, x2 = x_rot.chunk(2, dim=-1)

    real = cos * x1 - sin * x2
    imag = sin * x1 + cos * x2

    x_applied = torch.cat((real, imag), dim=-1)

    if x_pass is not None:
        return torch.cat([x_applied, x_pass], dim=-1)

    return x_applied.reshape(batch_size, seq_length, -1)


# this is a fused version of get_rotary_pos_emb + apply_rotary_pos_emb
def fused_rotary_emb_contrib(
    *,
    x: ir.Value,
    cos_cache: ir.Value,
    sin_cache: ir.Value,
    position_ids: ir.Value,
    num_heads: int,
    rotary_embedding_dim: int = 0,
) -> ir.Value:
    """
    Apply Rotary Positional Embedding (RoPE) to the input hidden states.

    This function modifies the input hidden states by applying the RoPE transformation
    using the provided cosine and sine caches based on the given position IDs.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        cos_cache (ir.Value): The cosine cache tensor of shape (max_position_embeddings, head_dim).
        sin_cache (ir.Value): The sine cache tensor of shape (max_position_embeddings, head_dim).
        position_ids (ir.Value): The position IDs tensor of shape (batch_size, seq_length).
        num_heads (int): The number of attention heads.
        rotary_embedding_dim (int): The dimension of the rotary embeddings for partial embedding (default is 0 equivalent to head_dim, meaning full embedding).

    Returns:
        ir.Value: The transformed hidden states with RoPE applied, of the same shape as input.
    """
    return torch.onnx.ops.symbolic(
        "com.microsoft::RotaryEmbedding",
        [x, position_ids, cos_cache, sin_cache],
        attrs={"num_heads": num_heads, "rotary_embedding_dim": rotary_embedding_dim},
        dtype=x.dtype,
        shape=x.shape,
        version=1,
    )
