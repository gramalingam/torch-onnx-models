# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx_ir as ir
import onnx_models

# Monkey patch class Value:
ir.Value._op_builder = property(lambda self: onnx_models.get_current_op_builder())

# Monkey patch ir.Value with operator overloads
ir.Value.__add__ = lambda self, other: self._op_builder.Add(self, other)
ir.Value.__sub__ = lambda self, other: self._op_builder.Sub(self, other)
ir.Value.__mul__ = lambda self, other: self._op_builder.Mul(self, other)
ir.Value.__truediv__ = lambda self, other: self._op_builder.Div(self, other)
ir.Value.__matmul__ = lambda self, other: self._op_builder.MatMul(self, other)
ir.Value.__neg__ = lambda self: self._op_builder.Neg(self)

ir.Value.__radd__ = lambda self, other: self._op_builder.Add(other, self)
ir.Value.__rsub__ = lambda self, other: self._op_builder.Sub(other, self)
ir.Value.__rmul__ = lambda self, other: self._op_builder.Mul(other, self)
ir.Value.__rtruediv__ = lambda self, other: self._op_builder.Div(other, self)
ir.Value.__rmatmul__ = lambda self, other: self._op_builder.MatMul(other, self)

ir.Value.__lt__ = lambda self, other: self._op_builder.Less(self, other)
ir.Value.__le__ = lambda self, other: self._op_builder.LessOrEqual(self, other)
ir.Value.__gt__ = lambda self, other: self._op_builder.Greater(self, other)
ir.Value.__ge__ = lambda self, other: self._op_builder.GreaterOrEqual(self, other)
ir.Value.__eq__ = lambda self, other: self._op_builder.Equal(self, other)