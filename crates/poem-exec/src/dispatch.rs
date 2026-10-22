use poem_ir::{
    Attn, Collective, Conv, Elemwise, Fused, Gather, Linear, Node, Operation, Sampling, Ssm,
};

use crate::error::KernelError;

pub trait DispatchAttn {
    fn dispatch(&mut self, op: &Attn) -> Result<(), KernelError>;
}

pub trait DispatchLinear {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError>;
}

pub trait DispatchElemwise {
    fn dispatch(&mut self, op: &Elemwise) -> Result<(), KernelError>;
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
    DispatchAttn
    + DispatchLinear
    + DispatchElemwise
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
            Operation::Attn(op) => DispatchAttn::dispatch(self, op),
            Operation::Linear(op) => DispatchLinear::dispatch(self, op),
            Operation::Elemwise(op) => DispatchElemwise::dispatch(self, op),
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
    T: DispatchAttn
        + DispatchLinear
        + DispatchElemwise
        + DispatchGather
        + DispatchCollective
        + DispatchSampling
        + DispatchSsm
        + DispatchFused
        + DispatchConv
        + DispatchProbe
{
}
