from __future__ import annotations


from onnx_models.components import _activations
from onnx_models import _configs
from onnx_models import BuilderModule, OpBuilder
from onnx_models.components._standard import Linear

class MLP(BuilderModule):
    def __init__(self, config: _configs.ArchitectureConfig, name: str | None = None):
        super().__init__(name)

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        bias = config.mlp_bias
        self.gate_proj = Linear(
            hidden_size, intermediate_size, bias=bias, name="GateProj"
        )
        self.up_proj = Linear(
            hidden_size, intermediate_size, bias=bias, name="UpProj"
        )
        self.down_proj = Linear(
            intermediate_size, hidden_size, bias=bias, name="DownProj"
        )
        self.act_fn = _activations.get_activation(config.hidden_act)

    def forward(self, op, x):
        gate = self.gate_proj(op, x)
        up = self.up_proj(op, x) 
        gate_activated = self.act_fn(op, gate)
        gated = op.Mul(gate_activated, up)
        down_proj = self.down_proj(op, gated)
        return down_proj
