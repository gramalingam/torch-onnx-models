from __future__ import annotations

import onnx_ir as ir
import torch
from torch import nn
from .._builder import OpBuilder


# TODO(jambayk): generalize to include sliding window
def create_attention_bias(
    op: OpBuilder,
    *,
    attention_mask: ir.Value,
    query_length: ir.Value,
    mask_value: float = torch.finfo(torch.float32).min
) -> ir.Value:
    """
    Create attention bias for use in attention mechanisms.

    Args:
        attention_mask (ir.Value): The attention mask tensor of shape (batch_size, total_length).
        query_length (ir.Value): The length of the query sequence, as a 1D tensor.
        dtype (torch.dtype): The desired data type for the output tensor.
        mask_value (float, optional): The value to use for masked positions. If None, uses the minimum value for the specified dtype.

    Returns:
        ir.Value: The attention bias tensor reshaped and cast to the specified dtype of shape (batch_size, 1, query_length, total_length).
    """
   
    # all_indices = attention_mask.cumsum(-1)
    one_0d = op.Constant(value_int=1)
    one_1d = op.Constant(value_ints=[1])
    all_indices = op.CumSum(attention_mask, one_0d)
    
    # kv_indices = torch.unsqueeze(all_indices, 1)
    kv_indices = op.Unsqueeze(all_indices, one_1d)
    
    # q_indices = all_indices[:, -query_length:]
    # For data-dependent slicing, we need to compute the start index
    total_length = op.Shape(attention_mask, start=1)
    start_idx = op.Sub(total_length, query_length)
    
    q_indices = op.Slice(
        all_indices,
        start_idx,
        total_length,
        one_1d
    )
    
    # q_indices = torch.unsqueeze(q_indices, -1)
    minus1_1d = op.Constant(value_ints=[-1])
    q_indices = op.Unsqueeze(q_indices, minus1_1d)
    
    # full_mask = q_indices >= kv_indices
    full_mask = op.GreaterOrEqual(q_indices, kv_indices)
    
    # torch.unsqueeze(attention_mask, 1).to(torch.bool)
    attention_mask_unsqueezed = op.Unsqueeze(attention_mask, one_1d)
    attention_mask_bool = op.Cast(attention_mask_unsqueezed, to=ir.DataType.BOOL)
    
    # full_mask = torch.logical_and(attention_mask_bool, full_mask)
    full_mask = op.And(attention_mask_bool, full_mask)
    
    # torch.where(full_mask, 0.0, mask_value)
    zero_tensor = op.Constant(value_float=0.0)
    mask_value_tensor = op.Constant(value_float=mask_value)
    result = op.Where(full_mask, zero_tensor, mask_value_tensor)
    
    # return torch.unsqueeze(result, 1)
    return op.Unsqueeze(result, one_1d)


# requires latest nightly ort to run inference correctly on exported model
# GQA case is incorrect in stable releases
def attention(
    op: OpBuilder,
    *,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    # rename back to attention_mask?
    bias: ir.Value,
    past_key: ir.Value | None = None,
    past_value: ir.Value | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    scale: float,
) -> tuple[ir.Value, ir.Value | None, ir.Value | None]:
    """
    Perform attention operation using ONNX Attention operator

    Args:
        query (ir.Value): The query tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
        key (ir.Value): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (ir.Value): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        bias (ir.Value): The attention bias tensor of shape (batch_size or 1, q_num_heads or 1, seq_length, seq_length + past_length).
        past_key (ir.Value | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (ir.Value | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[ir.Value, ir.Value, ir.Value]: A tuple containing the attention output, present key, and present value.
            attention_output (ir.Value): The output tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
            present_key (ir.Value): The present key tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
            present_value (ir.Value): The present value tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
    """
    if past_key is None:
        assert past_value is None
        attn = op.Attention(
            query, key, value, bias,
            kv_num_heads=kv_num_heads, q_num_heads=q_num_heads, scale=scale,
        )
        return (attn, None, None)
    else:
        assert past_value is not None
        return op.Attention(
            query, key, value, bias, past_key, past_value,
            kv_num_heads=kv_num_heads, q_num_heads=q_num_heads, scale=scale,
            _outputs=3
        )

# TODO(rama): Not yet migrated.
def _reshape_3d_to_4d(
    x: ir.Value, batch_size: int, seq_length: int, num_heads: int
) -> ir.Value:
    """
    Reshape a 3D tensor to a 4D tensor for multi-head attention.

    Args:
        x (ir.Value): The input tensor of shape (batch_size, seq_length, num_heads * head_dim).
        batch_size (int): The batch size.
        seq_length (int): The sequence length.
        num_heads (int): The number of attention heads.

    Returns:
        ir.Value: The reshaped tensor of shape (batch_size, num_heads, seq_length, head_dim).
    """
    return x.reshape(batch_size, seq_length, num_heads, -1).transpose(1, 2).contiguous()

# TODO(rama): Not yet migrated.
def _prepare_kv_mha(
    *,
    key: ir.Value,
    value: ir.Value,
    past_key: ir.Value | None = None,
    past_value: ir.Value | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    batch_size: int,
    seq_length: int,
) -> tuple[ir.Value, ir.Value, ir.Value, ir.Value]:
    """
    Prepare key and value tensors for Multi-Head Attention (MHA) operation.

    Args:
        key (ir.Value): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (ir.Value): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        past_key (ir.Value | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (ir.Value | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.

    Returns:
        tuple[ir.Value, ir.Value, ir.Value, ir.Value]: A tuple containing the prepared key, value, present key, and present value tensors.
            key (ir.Value): The prepared key tensor of shape (batch_size, q_num_heads, total_length, head_dim).
            value (ir.Value): The prepared value tensor of shape (batch_size, q_num_heads, total_length, head_dim).
            present_key (ir.Value): The present key tensor for caching of shape (batch_size, kv_num_heads, total_length, head_dim).
            present_value (ir.Value): The present value tensor for caching of shape (batch_size, kv_num_heads, total_length, head_dim).
    """
    key = _reshape_3d_to_4d(key, batch_size, seq_length, kv_num_heads)
    value = _reshape_3d_to_4d(value, batch_size, seq_length, kv_num_heads)

    # TODO(jambayk): put some guidance that there should not be data-dependent conditionals in general but None checks are ok
    if past_key is not None and past_value is not None:
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
    present_key = key
    present_value = value

    if q_num_heads != kv_num_heads:
        key = key.repeat_interleave(q_num_heads // kv_num_heads, dim=1)
        value = value.repeat_interleave(q_num_heads // kv_num_heads, dim=1)
    return key, value, present_key, present_value

# TODO(rama): Not yet migrated.
def attention_decomposed(
    *,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    bias: ir.Value,
    past_key: ir.Value | None = None,
    past_value: ir.Value | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    scale: float,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """
    Perform attention operation using ONNX Attention operator

    Args:
        query (ir.Value): The query tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
        key (ir.Value): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (ir.Value): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        bias (ir.Value): The attention bias tensor of shape (batch_size or 1, q_num_heads or 1, seq_length, seq_length + past_length).
        past_key (ir.Value | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (ir.Value | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[ir.Value, ir.Value, ir.Value]: A tuple containing the attention output, present key, and present value.
            attention_output (ir.Value): The output tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
            present_key (ir.Value): The present key tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
            present_value (ir.Value): The present value tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
    """
    batch_size, seq_length, _ = query.shape
    query = _reshape_3d_to_4d(query, batch_size, seq_length, q_num_heads)
    key, value, present_key, present_value = _prepare_kv_mha(
        key=key,
        value=value,
        past_key=past_key,
        past_value=past_value,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    attn_weight = torch.matmul(query, key.transpose(2, 3)) * scale
    if torch.onnx.is_in_onnx_export():
        # export is failing due to shape mismatch which shouldn't be happening
        attn_weight = torch.onnx.ops.symbolic(
            "Add",
            [attn_weight, bias],
            attrs={},
            dtype=attn_weight.dtype,
            shape=attn_weight.shape,
        )
    else:
        attn_weight = attn_weight + bias

    attn_weights = nn.functional.softmax(attn_weight, dim=-1)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = (
        attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_length, -1)
    )
    return attn_output, present_key, present_value

# TODO(rama): Not yet migrated.
def attention_contrib_mha(
    *,
    query: ir.Value,
    key: ir.Value,
    value: ir.Value,
    bias: ir.Value,
    past_key: ir.Value | None = None,
    past_value: ir.Value | None = None,
    q_num_heads: int,
    kv_num_heads: int,
    scale: float,
) -> tuple[ir.Value, ir.Value, ir.Value]:
    """
    Perform attention operation using ONNX Attention operator

    Args:
        query (ir.Value): The query tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
        key (ir.Value): The key tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        value (ir.Value): The value tensor of shape (batch_size, seq_length, kv_num_heads * head_dim).
        bias (ir.Value): The attention bias tensor of shape (batch_size or 1, q_num_heads or 1, seq_length, seq_length + past_length).
        past_key (ir.Value | None): The past key tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        past_value (ir.Value | None): The past value tensor for caching of shape (batch_size, kv_num_heads, past_length, head_dim).
        q_num_heads (int): The number of query attention heads.
        kv_num_heads (int): The number of key-value heads.
        scale (float): The scaling factor for the attention scores.

    Returns:
        tuple[ir.Value, ir.Value, ir.Value]: A tuple containing the attention output, present key, and present value.
            attention_output (ir.Value): The output tensor of shape (batch_size, seq_length, q_num_heads * head_dim).
            present_key (ir.Value): The present key tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
            present_value (ir.Value): The present value tensor for caching of shape (batch_size, kv_num_heads, seq_length + past_length, head_dim).
    """
    batch_size, seq_length, _ = query.shape
    key, value, present_key, present_value = _prepare_kv_mha(
        key=key,
        value=value,
        past_key=past_key,
        past_value=past_value,
        q_num_heads=q_num_heads,
        kv_num_heads=kv_num_heads,
        batch_size=batch_size,
        seq_length=seq_length,
    )

    return (
        torch.onnx.ops.symbolic(
            "com.microsoft::MultiHeadAttention",
            [query, key, value, None, None, bias],
            attrs={"num_heads": q_num_heads, "scale": scale},
            dtype=value.dtype,
            shape=(batch_size, seq_length, q_num_heads * value.shape[-1]),
            version=1,
        ),
        present_key,
        present_value,
    )

