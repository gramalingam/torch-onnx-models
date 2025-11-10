from __future__ import annotations

import torch
from torch import nn

import onnxscript

def batched_attention(query_states, key_states, value_states, cu_seqlens):
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    splits = [
        torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
    ]

    attn_outputs = [
        attention(
            q,
            k,
            v,
        )[0]
        for q, k, v in zip(*splits)
    ]
    attn_output = torch.cat(attn_outputs, dim=1)
    return attn_output

op = onnxscript.opset(23)

@onnxscript.script()
def batched_attention1(query_states, key_states, value_states, cu_seqlens):
    num_batches = op.Size(cu_seqlens) - 1
    batching_axis = op.Constant(value_ints=[2])
    attn_output = op.Slice(query_states, [0], [0], batching_axis)  # Initialize empty output
    for i in range(num_batches):
        i_1d = op.Reshape(i, [1])
        i_plus_1_1d = i_1d + 1
        start = op.Gather(cu_seqlens, i_1d, axis=0)
        end = op.Gather(cu_seqlens, i_plus_1_1d, axis=0)
        q_batch = op.Slice(query_states, start, end, batching_axis)
        k_batch = op.Slice(key_states, start, end, batching_axis)
        v_batch = op.Slice(value_states, start, end, batching_axis)
        attn_output_batch = op.Attention(q_batch, k_batch, v_batch)
        attn_output = op.Concat(attn_output, attn_output_batch, axis=2)
    return attn_output

class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1  # needed for eager attention
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.attention_dropout = 0.0
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
        ]

        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output