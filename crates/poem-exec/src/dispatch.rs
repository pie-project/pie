use poem_ir::{
    Attention, Collective, Conv, Elementwise, Fused, Gather, Linear, Node, Operation, Sampling, Ssm,
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

pub trait DispatchGather {
    fn dispatch(&mut self, op: &Gather) -> Result<(), KernelError>;
}

pub trait DispatchCollective {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError>;
}

pub trait DispatchSampling {
    fn dispatch(&mut self, op: &Sampling) -> Result<(), KernelError>;
}

pub trait DispatchSsm {
    fn dispatch(&mut self, op: &Ssm) -> Result<(), KernelError>;
}

pub trait DispatchFused {
    fn dispatch(&mut self, op: &Fused) -> Result<(), KernelError>;
}

pub trait DispatchConv {
    fn dispatch(&mut self, op: &Conv) -> Result<(), KernelError>;
}

pub trait DispatchProbe {
    fn probe(&mut self, _node: &Node) {}
}

pub trait Dispatch:
    DispatchAttention
    + DispatchLinear
    + DispatchElementwise
    + DispatchGather
    + DispatchCollective
    + DispatchSampling
    + DispatchSsm
    + DispatchFused
    + DispatchConv
    + DispatchProbe
{
    fn exec(&mut self, node: &Node) -> Result<(), KernelError> {
        let outcome = match &node.op {
            Operation::Attention(op) => DispatchAttention::dispatch(self, op),
            Operation::Linear(op) => DispatchLinear::dispatch(self, op),
            Operation::Elementwise(op) => DispatchElementwise::dispatch(self, op),
            Operation::Gather(op) => DispatchGather::dispatch(self, op),
            Operation::Collective(op) => DispatchCollective::dispatch(self, op),
            Operation::Sampling(op) => DispatchSampling::dispatch(self, op),
            Operation::Ssm(op) => DispatchSsm::dispatch(self, op),
            Operation::Fused(op) => DispatchFused::dispatch(self, op),
            Operation::Conv(op) => DispatchConv::dispatch(self, op),
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
        + DispatchGather
        + DispatchCollective
        + DispatchSampling
        + DispatchSsm
        + DispatchFused
        + DispatchConv
        + DispatchProbe
{
}
