from __future__ import annotations

from typing import Iterable

import onnx_ir as ir

def make_external_tensor(name: str, dtype: ir.DataType, shape: Iterable[int | str]) -> ir.ExternalTensor:
    return ir.ExternalTensor(
        location=name,
        offset=None,
        length=None,
        dtype=dtype,
        shape=ir.Shape(list(shape)),
        name=name
    )