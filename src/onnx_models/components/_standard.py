from __future__ import annotations

__all__ = ["Linear", "Embedding"]

import onnx_ir as ir
from onnx_models import BuilderModule
import onnx_models.utils as utils


class Linear(BuilderModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name: str | None = None):
        super().__init__(name)
        self._weight = utils.make_external_tensor("weight", ir.DataType.FLOAT, (in_features, out_features))
        if bias:
            self._bias = utils.make_external_tensor("bias", ir.DataType.FLOAT, (out_features,))
        else:
            self._bias = None
    
    def forward(self, input: ir.Value) -> ir.Value:
        output = self.op.MatMul(input, self._weight)
        
        # Add bias if enabled
        if self._bias is not None:
            output = self.op.Add(output, self._bias)
        
        return output



class Embedding(BuilderModule):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, name: str | None = None):
        super().__init__(name)
        self._weight = utils.make_external_tensor("weight", ir.DataType.FLOAT,(num_embeddings, embedding_dim))
    
    def forward(self, input: ir.Value) -> ir.Value:
        return self.op.Gather(self._weight, input, axis=0)

