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
        self._module_stack : list[BuilderModule] = []

    def push_module(self, module: str) -> None:
        self._module_stack.append(module)

    def pop_module(self) -> None:
        self._module_stack.pop()

    def context_name(self) -> str:
        return ".".join([m.name for m in self._module_stack if m.name])

    def connect_output(self, output_parameter: ir.Value, computed_value: ir.Value) -> None:
        """Connect a computed value to an output parameter of the graph.
        
        Args:
            output_parameter: This specifies the name, type, and shape of the output.
            computed_value: The computed value to connect to the output parameter.
        """
        if computed_value.producer() is None:
            # Graph input. Can't use _is_graph_input method yet.
            # Identity-elimination is not possible in this case
            self.op("Identity", inputs=[computed_value], output=output_parameter)
        else:
            # Avoid unnecessary Identity nodes
            # TODO: Check type and shape compatibility
            computed_value.name = output_parameter.name
            computed_value.type = output_parameter.type
            computed_value.shape = output_parameter.shape

    def initializer(self, tensor: ir.TensorProtocol, name: str | None = None) -> ir.Value:
        if name is None:
            name = tensor.name
        prefix = self.context_name()
        if prefix:
            name = f"{prefix}.{name}"
            # TODO: set tensor name as well
        return super().initializer(tensor, name=name)

# Global thread-local storage for builder stack
_thread_local = local()


def get_current_builder() -> IRModelBuilder:
    """Get the current IRModelBuilder from the context stack."""
    stack = getattr(_thread_local, 'builder_stack', None)
    if stack:
        return stack[-1]
    else:
        raise RuntimeError("No active IRModelBuilder found in context.")

def get_current_op_builder() -> OpBuilder:
    """Get the current OpBuilder from the context stack."""
    builder = get_current_builder()
    return builder.op_builder

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

    def _adapt_input(self, value: ir.Value | ir.TensorProtocol) -> ir.Value:
        if not isinstance(value, ir.Value):
            # TODO: We could using caching to avoid duplicate initializers. However, it seems unlikely
            # to be useful in practice, as shared use of a stateful module is rare.
            return self._builder.initializer(value)
        return value

    def _adapt_outputs(self, outputs: int | Sequence[str | ir.Value]) -> Sequence[ir.Value]:
        prefix = self._builder.context_name()
        if isinstance(outputs, int):
            count = len(self._builder.nodes)
            name = f"{prefix}.val_{count}" if prefix else "val_{count}"
            if outputs == 1:
                return [ir.Value(name=name)]
            else:
                return [ir.Value(name=f"{name}.{i}") for i in range(outputs)]
        adapted_outputs = []
        for output in outputs:
            if isinstance(output, ir.Value):
                adapted_outputs.append(output)
            else:
                adapted_outputs.append(ir.Value(name=output))
        return adapted_outputs
    
    def _make_node(self, op_type: str, inputs: Sequence[ir.Value | ir.TensorProtocol], kwargs: dict[str, Any]):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)

        output_values = self._adapt_outputs(outputs)

        inputs = [self._adapt_input(i) for i in inputs]

        if len(output_values) == 1:
            value = self._builder.op(
                op_type, inputs=inputs, attributes=kwargs, domain=domain, version=version, output=output_values[0]
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
            outputs=output_values,
        )
        return values

class BuilderModule:
    def __init__(self, name: str | None = None):
        """Initialize BuilderModule with optional name.
        
        Args:
            name: Optional name for the module. If None, uses the class name.
        """
        # self.name = name if name is not None else self.__class__.__name__
        self.name = name

    def __call__(self, *args, **kwargs):
        """Delegate calls to the forward method."""
        self.builder = get_current_builder()
        self.op = get_current_op_builder()
        self.builder.push_module(self)
        result = self.forward(*args, **kwargs)
        self.builder.pop_module()
        return result
    
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
            builder.connect_output(output_parameter=output_parameter, computed_value=computed_value)
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
