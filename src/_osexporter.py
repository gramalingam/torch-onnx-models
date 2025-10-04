from __future__ import annotations

__all__ = ["convert_hf_model"]

import json
import logging

import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import torch
import tqdm
from onnx_ir import tensor_adapters
from torch._subclasses.fake_tensor import FakeTensorMode

from torch_onnx_models import _configs, onnx_passes
from torch_onnx_models.components._model import CausalLMModel

logger = logging.getLogger(__name__)

def _create_inputs_outputs(
    config: _configs.ArchitectureConfig, export_config: _configs.ExportConfig
):
    """Create ONNX IR Values for inputs and outputs based on the model configuration."""
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    head_dim = config.head_dim
    
    # Create input Values with appropriate shapes and types
    input_values = {}
    
    # Input IDs: (batch_size, sequence_length) - int64
    input_ids = ir.Value(
        name="input_ids",
        shape=ir.Shape(["batch", "sequence_len"]),
        dtype=ir.DataType.INT64
    )
    
    # Attention mask: (batch_size, past_sequence_len + sequence_len) - int64  
    attention_mask = ir.Value(
        name="attention_mask", 
        shape=ir.Shape(["batch", "past_sequence_len+sequence_len"]),
        dtype=ir.DataType.INT64
    )
    
    # Position IDs: (batch_size, sequence_length) - int64
    position_ids = ir.Value(
        name="position_ids",
        shape=ir.Shape(["batch", "sequence_len"]), 
        dtype=ir.DataType.INT64
    )
    
    input_values = [input_ids, attention_mask, position_ids]

    # Past key values: list of (key, value) tuples for each layer
    # Each has shape (batch_size, num_key_value_heads, past_sequence_len, head_dim) - float32
    # Create flat list directly
    for i in range(num_hidden_layers):
        input_values.extend([
            ir.Value(
                name=f"past_key_values.{i}.key",
                shape=ir.Shape(["batch", num_key_value_heads, "past_sequence_len", head_dim]),
                dtype=ir.DataType.FLOAT
            ),
            ir.Value(
                name=f"past_key_values.{i}.value", 
                shape=ir.Shape(["batch", num_key_value_heads, "past_sequence_len", head_dim]),
                dtype=ir.DataType.FLOAT
            )
        ])
    
    # Create output Values
    
    # Logits: (batch_size, sequence_length, vocab_size) - float32
    logits = ir.Value(
        name="logits",
        shape=ir.Shape(["batch", "sequence_len", config.vocab_size]),
        dtype=ir.DataType.FLOAT
    )
    
    # Present key values: list of (key, value) tuples for each layer  
    # Each has shape (batch_size, num_key_value_heads, sequence_len, head_dim) - float32
    output_values = [logits]
    for i in range(num_hidden_layers):
        output_values.extend([
            ir.Value(
                name=f"present.{i}.key",
                shape=ir.Shape(["batch", num_key_value_heads, "sequence_len", head_dim]),
                dtype=ir.DataType.FLOAT
            ),
            ir.Value(
                name=f"present.{i}.value",
                shape=ir.Shape(["batch", num_key_value_heads, "sequence_len", head_dim]), 
                dtype=ir.DataType.FLOAT
            )
        ])
    
    return input_values, output_values


def convert_hf_model(
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct",
    load_weights: bool = True,
    clear_metadata: bool = False,
) -> ir.Model:
    """Convert a HuggingFace model to ONNX.

    Args:
        model_id: The model ID on HuggingFace Hub.
        load_weights: Whether to load the pretrained weights from the HuggingFace model.
        clear_metadata: Whether to clear debugging metadata from the ONNX model.
    """
    import transformers

    # Need to use transformers to load config because transformers has additional
    # logic to standardize the config field names.
    config = transformers.AutoConfig.from_pretrained(model_id)
    architecture_config = _configs.ArchitectureConfig.from_transformers(config)
    
    model = CausalLMModel(architecture_config)
    with IRModelBuilder() as builder:
        model_inputs, model_outputs = _create_inputs_outputs(
            architecture_config, None
        )
        outputs = model(model_inputs)
        onnx_model = builder.get_model(outputs)
 
    #     opset_version=23,

    onnx_model.producer_name = "torch_onnx_models"
    onnx_model.producer_version = "0.1.0"
    onnx_model.graph.name = model_id

    passes = ir.passes.PassManager(
        [
            onnx_passes.AssignNamesPass(),
            onnx_passes.FoldTransposePass(),
            common_passes.RemoveUnusedNodesPass(),
            common_passes.RemoveUnusedFunctionsPass(),
            common_passes.RemoveUnusedOpsetsPass(),
            common_passes.DeduplicateInitializersPass(),
            common_passes.CommonSubexpressionEliminationPass(),
            # onnx_passes.RemoveBarrierPass()
        ]
    )

    passes(onnx_model)

    if clear_metadata:
        common_passes.ClearMetadataAndDocStringPass()(onnx_model)

    return onnx_model
