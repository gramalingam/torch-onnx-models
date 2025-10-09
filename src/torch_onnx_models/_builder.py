from __future__ import annotations

from typing import Any, Callable, Sequence
from contextlib import contextmanager
from threading import local

import onnx_ir as ir
import onnx_ir.passes.common as common_passes

class IRModelBuilder(ir.tape.Tape):
    def __init__(self) -> None:
        super().__init__()
        self.op_builder = OpBuilder(self)

    def add_output(self, output_parameter: ir.Value, computed_value: ir.Value) -> None:
        self.op("Identity", inputs=[computed_value], output=output_parameter)

# Global thread-local storage for builder stack
_thread_local = local()


def get_current_builder() -> IRModelBuilder | None:
    """Get the current IRModelBuilder from the context stack."""
    stack = getattr(_thread_local, 'builder_stack', None)
    if stack:
        return stack[-1]
    return None

def get_current_op_builder() -> OpBuilder | None:
    """Get the current OpBuilder from the context stack."""
    builder = get_current_builder()
    if builder:
        return builder.op_builder
    return None

@contextmanager
def builder_context(builder: IRModelBuilder):
    """Context manager to set the current IRModelBuilder.
    
    Args:
        builder: The IRModelBuilder to set as current
        
    Usage:
        with builder_context(my_builder):
            # my_builder is now the current builder
            current = get_current_builder()
    """
    # Initialize the stack if it doesn't exist
    if not hasattr(_thread_local, 'builder_stack'):
        _thread_local.builder_stack = []
    
    # Push the new builder onto the stack
    _thread_local.builder_stack.append(builder)
    try:
        yield builder
    finally:
        # Pop the builder from the stack
        _thread_local.builder_stack.pop()

class OpBuilder:
    def __init__(self, builder: IRModelBuilder) -> None:
        self._builder = builder
    
    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def _make_node(self, op_type: str, inputs: Sequence[ir.Value], kwargs: dict[str, Any]):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)
        if isinstance(outputs, Sequence):
            num_outputs = len(outputs)
        else:
            assert isinstance(outputs, int)
            num_outputs = outputs

        if num_outputs == 1:
            value = self._builder.op(
                op_type, inputs=inputs, attributes=kwargs, domain=domain, version=version
            )
            if isinstance(outputs, Sequence):
                value.name = outputs[0]
            return value
        values = self._builder.op_multi_out(
            op_type,
            inputs=inputs,
            attributes=kwargs,
            domain=domain,
            version=version,
            num_outputs=num_outputs,
        )
        if isinstance(outputs, Sequence):
            for value, name in zip(values, outputs):
                value.name = name
        return values

class BuilderModule:
    def __call__(self, *args, **kwargs):
        """Delegate calls to the forward method."""
        self.builder = get_current_builder()
        if self.builder is None:
            raise RuntimeError("No active IRModelBuilder found in context.")
        self.op = get_current_op_builder()

        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method")
    

GraphBuilderFunction = Callable[[Sequence[ir.Value]], Sequence[ir.Value]]

def export(model: GraphBuilderFunction, model_inputs: Sequence[ir.Value], model_outputs: Sequence[ir.Value], model_id: str) -> ir.Model:
    # opset_version=23,
    builder = IRModelBuilder()
    with builder_context(builder):
        outputs = model(model_inputs)
        assert len(outputs) == len(model_outputs), "Output length mismatch"
        for output_parameter, computed_value in zip(model_outputs, outputs):
            builder.add_output(output_parameter=output_parameter, computed_value=computed_value)
        graph = ir.Graph(
            name=f"{model_id}",
            inputs=model_inputs,
            outputs=model_outputs,
            nodes=builder.nodes,
            initializers=builder.initializers,
            opset_imports={"": 23},
        )
        onnx_model = ir.Model(
            graph=graph,
            ir_version=11,
            producer_name="torch_onnx_models",
            producer_version="0.1.0",
        )

    # passes = ir.passes.PassManager(
    #     [
    #         # onnx_passes.AssignNamesPass(),
    #         # onnx_passes.FoldTransposePass(),
    #         common_passes.RemoveUnusedNodesPass(),
    #         common_passes.RemoveUnusedFunctionsPass(),
    #         common_passes.RemoveUnusedOpsetsPass(),
    #         common_passes.DeduplicateInitializersPass(),
    #         common_passes.CommonSubexpressionEliminationPass(),
    #         # onnx_passes.RemoveBarrierPass()
    #     ]
    # )

    # passes(onnx_model)
    return onnx_model
