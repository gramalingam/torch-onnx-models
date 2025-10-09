from __future__ import annotations

__all__ = ["Linear", "Embedding"]

import numpy as np
import torch
import onnx_ir as ir
from torch_onnx_models import BuilderModule


class Linear(BuilderModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # These will be set when the builder is available during forward()
        self._weight = None
        self._bias = None
    
    def forward(self, input: ir.Value) -> ir.Value:
        # Create weight initializer if not already created
        if self._weight is None:
            # Create a weight matrix of shape (in_features, out_features)
            # The actual values don't matter as they'll be loaded from trained model
            weight_data = ir.Tensor(np.zeros((self.in_features, self.out_features), dtype=np.float32))
            self._weight = self.builder.initializer(weight_data, name="weight")
        
        # Perform matrix multiplication: input @ weight
        # ONNX MatMul expects input shape (..., in_features) and weight shape (in_features, out_features)
        output = self.op.MatMul(input, self._weight)
        
        # Add bias if enabled
        if self.use_bias:
            if self._bias is None:
                # Create bias vector of shape (out_features,)
                bias_data = torch.zeros(self.out_features)
                self._bias = self.builder.initializer("bias", bias_data)
            
            # Add bias using ONNX Add operation
            output = self.op.Add(output, self._bias)
        
        return output
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}"


class Embedding(BuilderModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self._weight = ir.ExternalTensor("",None, None, ir.DataType.FLOAT,
                                         shape=ir.Shape([num_embeddings, embedding_dim]),
                                         name="weight")
    
    def forward(self, input: ir.Value) -> ir.Value:
        weight = self.builder.initializer(self._weight, name="weight")
        output = self.op.Gather(weight, input, axis=0)       
        return output
    
    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        return s
