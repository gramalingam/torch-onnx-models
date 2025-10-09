from __future__ import annotations

import onnx_ir as ir

from torch_onnx_models import _configs
from torch_onnx_models.components._attention_utils import create_attention_bias
from torch_onnx_models.components._decoder import DecoderLayer
from torch_onnx_models.components._rms_norm import RMSNorm
from torch_onnx_models.components._rotary_embedding import initialize_rope
from torch_onnx_models import BuilderModule
from torch_onnx_models.components._standard import Linear, Embedding

class TextModel(BuilderModule):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = initialize_rope(config)

    def forward(
        self,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]]]:
        # embed tokens and positions
        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(position_ids)
        query_length = self.op.Shape(input_ids, start=1)

        # get the attention bias
        attention_bias = create_attention_bias(
            attention_mask=attention_mask,
            query_length=query_length,
        )

        present_key_values = []
        for layer, past_key_value in zip(
            self.layers, past_key_values or [None] * len(self.layers)
        ):
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                attention_bias=attention_bias,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
            )
            present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values


class CausalLMModel(BuilderModule):
    def __init__(self, config: _configs.ArchitectureConfig):
        super().__init__()
        self.model = TextModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: ir.Value,
        attention_mask: ir.Value,
        position_ids: ir.Value,
        past_key_values: list[tuple[ir.Value, ir.Value]] | None = None,
    ) -> tuple[ir.Value, list[tuple[ir.Value, ir.Value]]]:
        hidden_states, present_key_values = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        logits = self.lm_head(hidden_states)
        return logits, present_key_values

    def unflatten_inputs(
        self, inputs: list[ir.Value]
    ) -> tuple[
        ir.Value,
        ir.Value,
        ir.Value,
        list[tuple[ir.Value, ir.Value]] | None,
    ]:
        input_ids = inputs[0]
        attention_mask = inputs[1]
        position_ids = inputs[2]
        if len(inputs) > 3:
            past_key_values = []
            for i in range(3, len(inputs), 2):
                key = inputs[i]
                value = inputs[i + 1]
                past_key_values.append((key, value))
        else:
            past_key_values = None
        return input_ids, attention_mask, position_ids, past_key_values

    def flatten_outputs(
        self, outputs: tuple[ir.Value, list[tuple[ir.Value, ir.Value]]]
    ) -> list[ir.Value]:
        logits, present_key_values = outputs
        flat_present = []
        for key, value in present_key_values:
            flat_present.extend([key, value])
        return [logits] + flat_present
