from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence
from contextlib import contextmanager
from threading import local

import onnxscript.values
import onnxscript.optimizer
import onnx_ir as ir
import onnx_ir.passes.common as common_passes
import onnx_models._inference as inference
import onnx_models._inliner as inliner

# A type representing the domains/versions used in creating nodes in IR.
UsedOpsets = set[tuple[str, Optional[int]]]

class Tape:
    def __init__(self) -> None:
        self._nodes: list[ir.Node] = []
        self._initializers: list[ir.Value] = []
        self._used_opsets: UsedOpsets = set()

    @property
    def nodes(self) -> Sequence[ir.Node]:
        return tuple(self._nodes)

    def append_node(self, node: ir.Node) -> None:
        self._nodes.append(node)
        self._used_opsets.add((node.domain, node.version))

    @property
    def initializers(self) -> Sequence[ir.Value]:
        return tuple(self._initializers)

    @property
    def used_opsets(self) -> UsedOpsets:
        return self._used_opsets

    def add_node(
        self,
        op_type: str,
        inputs: Sequence[ir.Value | None],
        attributes: Mapping[str, ir._convenience.SupportedAttrTypes] | None = None,
        *,
        num_outputs: int | None = None,
        outputs: Sequence[ir.Value] | None = None,
        domain: str = "",
        overload: str = "",
        version: int | None = None,
        graph: ir.Graph | None = None,
        name: str | None = None,
        doc_string: str | None = None,
        metadata_props: dict[str, str] | None = None,
    ) -> ir.Node:
        if num_outputs is None and outputs is None:
            raise ValueError("Either num_outputs or outputs must be provided.")
        if num_outputs is not None and outputs is not None:
            raise ValueError("Both num_outputs and outputs cannot be provided simultaneously.")
        output_kwargs: dict[str, Any]
        if outputs is None:
            output_kwargs = dict(num_outputs=num_outputs)
        else:
            output_kwargs = dict(outputs=outputs)
        if attributes is None:
            attrs: Sequence[ir.Attr] = ()
        else:
            attrs = ir._convenience.convert_attributes(attributes)
        node = ir.Node(
            domain,
            op_type,
            inputs,
            attributes=attrs,
            **output_kwargs,
            overload=overload,
            version=version,
            graph=graph,
            name=name,
            doc_string=doc_string,
            metadata_props=metadata_props,
        )
        self.append_node(node)
        return node

    def initializer(self, tensor: ir.TensorProtocol, name: str) -> ir.Value:
        shape = ir.Shape((d if isinstance(d, int) else d.value) for d in tensor.shape.dims)
        value = ir.Value(
            name=name, shape=shape, type=ir.TensorType(tensor.dtype), const_value=tensor
        )
        self._initializers.append(value)
        return value
    
class IRModelBuilder:
    def __init__(self) -> None:
        super().__init__()
        self._tape = Tape()
        self._op_builder = OpBuilder(self)
        self._module_stack : list[BuilderModule] = []

    @property
    def op(self) -> OpBuilder:
        return self._op_builder

    @property
    def tape(self) -> Tape:
        return self._tape

    def initializer(self, tensor: ir.TensorProtocol, name: str | None = None) -> ir.Value:
        if name is None:
            name = tensor.name
        prefix = self.context_name()
        if prefix:
            name = f"{prefix}.{name}"
            # TODO: set tensor name as well
        return self._tape.initializer(tensor, name=name)
    
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
    return builder.op

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

    @property
    def builder(self) -> IRModelBuilder:
        return self._builder

    def __getattr__(self, op_type: str) -> Any:
        return lambda *args, **kwargs: self._make_node(op_type, args, kwargs)

    def initializer(self, tensor: ir.TensorProtocol, name: str | None = None) -> ir.Value:
        return self._builder.initializer(tensor, name)

    def _adapt_input(self, value: ir.Value | ir.TensorProtocol) -> ir.Value:
        if not isinstance(value, ir.Value):
            # TODO: We could using caching to avoid duplicate initializers. However, it seems unlikely
            # to be useful in practice, as shared use of a stateful module is rare.
            return self._builder.initializer(value)
        return value

    def _adapt_outputs(self, outputs: int | Sequence[str | ir.Value]) -> Sequence[ir.Value]:
        prefix = self._builder.context_name()
        if isinstance(outputs, int):
            count = len(self._builder.tape.nodes)
            name = f"{prefix}.val_{count}" if prefix else f"val_{count}"
            if outputs == 1:
                return [ir.Value(name=name)]
            else:
                return [ir.Value(name=f"{name}.{i}") for i in range(outputs)]
        adapted_outputs = []
        for output in outputs:
            if isinstance(output, ir.Value):
                adapted_outputs.append(output)
            elif isinstance(output, str):
                adapted_outputs.append(ir.Value(name=output))
            else:
                raise TypeError(f"Output type not supported.")
        return adapted_outputs
    
    def _make_node(self, op_type: str, inputs: Sequence[ir.Value | ir.TensorProtocol], kwargs: dict[str, Any]):
        domain = kwargs.pop("_domain", "")
        version = kwargs.pop("_version", None)
        outputs = kwargs.pop("_outputs", 1)

        output_values = self._adapt_outputs(outputs)

        inputs = [self._adapt_input(i) for i in inputs]

        node = self._builder.tape.add_node(
                op_type,
                inputs=inputs,
                attributes=kwargs,
                domain=domain,
                version=version,
                outputs=output_values,
            )
        if domain == "":
            onnxscript.optimizer.basic_constant_propagation([node])
            inference.infer_outputs(node, 23)
        return node.outputs if len(node.outputs) > 1 else node.outputs[0]

    def call(self, function, *args, **kwargs):
        if isinstance(function, ir.Function):
            function_ir = function
        elif isinstance(function, onnxscript.values.OnnxFunction):
            function_proto = function.to_function_proto()
            function_ir = ir.serde.deserialize_function(function_proto)
        else:
            raise TypeError("Function must be an ir.Function or onnxscript.ONNXFunction")
        nodes, outputs = inliner.instantiate(function_ir, args, kwargs)
        for node in nodes:
            self._builder.tape.append_node(node)
            onnxscript.optimizer.basic_constant_propagation([node])
            inference.infer_outputs(node, 23)
        return outputs if len(outputs) > 1 else outputs[0]

class BuilderModule:
    def __init__(self, name: str | None = None):
        """Initialize BuilderModule with optional name.
        
        Args:
            name: Optional name for the module. If None, uses the class name.
        """
        # self.name = name if name is not None else self.__class__.__name__
        self.name = name

    def __call__(self, op: OpBuilder, *args, **kwargs):
        """Delegate calls to the forward method."""
        # Following not used currently: this is not necessary if we explicitly pass op/builder around.
        # self.builder = get_current_builder()
        # self.op = get_current_op_builder()
        assert isinstance(op, OpBuilder), "First argument must be an OpBuilder"
        op.builder.push_module(self)
        result = self.forward(op, *args, **kwargs)
        op.builder.pop_module()
        return result
    
    def forward(self, *args, **kwargs):
        """Forward method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method")
    

GraphBuilderFunction = Callable[[Sequence[ir.Value]], Sequence[ir.Value]]

def export(model: GraphBuilderFunction, model_inputs: Sequence[ir.Value], model_outputs: Sequence[ir.Value], model_id: str) -> ir.Model:
    # opset_version=23,
    builder = IRModelBuilder()
    with builder_context(builder):
        outputs = model(builder.op, model_inputs)
        assert len(outputs) == len(model_outputs), "Output length mismatch"
        for output_parameter, computed_value in zip(model_outputs, outputs):
            builder.connect_output(output_parameter=output_parameter, computed_value=computed_value)
        graph = ir.Graph(
            name=f"{model_id}",
            inputs=model_inputs,
            outputs=model_outputs,
            nodes=builder.tape.nodes,
            initializers=builder.tape.initializers,
            opset_imports={"": 23},
        )
        onnx_model = ir.Model(
            graph=graph,
            ir_version=11,
            producer_name="onnx_models",
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
