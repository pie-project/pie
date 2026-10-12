use poem_ir::{
    Attention, Collective, Elementwise, Fused, Layout, Linear, Node, Operation, Spatial,
};

use crate::error::KernelError;

pub trait DispatchAttention {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError>;
}

pub trait DispatchLinear {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError>;
}

pub trait DispatchElementwise {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError>;
}

pub trait DispatchLayout {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError>;
}

pub trait DispatchCollective {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError>;
}

pub trait DispatchFused {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError>;
}

pub trait DispatchSpatial {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError>;
}

pub trait DispatchProbe {
    fn probe(&mut self, _node: &Node) {}
}

pub trait Dispatch:
    DispatchAttention
    + DispatchLinear
    + DispatchElementwise
    + DispatchLayout
    + DispatchCollective
    + DispatchFused
    + DispatchSpatial
    + DispatchProbe
{
    fn exec(&mut self, node: &Node) -> Result<(), KernelError> {
        let outcome = match &node.op {
            Operation::Attention(op) => DispatchAttention::dispatch(self, op),
            Operation::Linear(op) => DispatchLinear::dispatch(self, op),
            Operation::Elementwise(op) => DispatchElementwise::dispatch(self, op),
            Operation::Layout(op) => DispatchLayout::dispatch(self, op),
            Operation::Collective(op) => DispatchCollective::dispatch(self, op),
            Operation::Fused(op) => DispatchFused::dispatch(self, op),
            Operation::Spatial(op) => DispatchSpatial::dispatch(self, op),
        };
        if outcome.is_ok() {
            self.probe(node);
        }
        outcome
    }
}

impl<T> Dispatch for T where
    T: DispatchAttention
        + DispatchLinear
        + DispatchElementwise
        + DispatchLayout
        + DispatchCollective
        + DispatchFused
        + DispatchSpatial
        + DispatchProbe
{
}
