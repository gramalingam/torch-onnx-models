from __future__ import annotations

import onnx_ir as ir

from torch_onnx_models import _configs
from torch_onnx_models.components._attention import Attention
from torch_onnx_models.components._mlp import MLP
from torch_onnx_models.components._rms_norm import RMSNorm


class DecoderLayer(BuilderModule):
    # take in layer_idx since newer models have hybrid layers
    # sliding window attention, no rope, etc.
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: ir.Value,
        attention_bias: ir.Value,
        position_embeddings: tuple[ir.Value, ir.Value],
        past_key_value: tuple[ir.Value, ir.Value] | None,
    ) -> tuple[ir.Value, tuple[ir.Value, ir.Value]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_bias=attention_bias,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_key_value
