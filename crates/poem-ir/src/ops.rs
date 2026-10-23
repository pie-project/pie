use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

pub mod attn;
pub mod collective;
pub mod conv;
pub mod elemwise;
pub mod fused;
pub mod gather;
pub mod linear;
pub mod sampling;
pub mod ssm;

pub use attn::{Attn, IndexScore, RaggedMask};
pub use collective::Collective;
pub use conv::{Conv, GridRule, TimePad, VoxelSegment};
pub use elemwise::{
    ActFn, BinaryOp, Bounds, Elemwise, GateActivation, ModulateForm, MropeForm, NormAxis, NormNext,
    NormWeight, ResidualGate, RopeForm, RopeFreqs, Turn, Yarn,
};
pub use fused::{Fused, NormKind};
pub use gather::{Gather, PixelOrder, TapKind};
pub use linear::{ActQuant, GluAct, Linear, RouteScore};
pub use sampling::Sampling;
pub use ssm::Ssm;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    Attn(Attn),
    Linear(Linear),
    Elemwise(Elemwise),
    Gather(Gather),
    Collective(Collective),
    Sampling(Sampling),
    Ssm(Ssm),
    Fused(Fused),
    Conv(Conv),
}

impl Operation {
    fn operands(&self) -> &dyn Operands {
        match self {
            Self::Attn(op) => op,
            Self::Linear(op) => op,
            Self::Elemwise(op) => op,
            Self::Gather(op) => op,
            Self::Collective(op) => op,
            Self::Sampling(op) => op,
            Self::Ssm(op) => op,
            Self::Fused(op) => op,
            Self::Conv(op) => op,
        }
    }
}

impl Operands for Operation {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        self.operands().inputs(sink);
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        self.operands().outputs(sink);
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        self.operands().aliases(sink);
    }
    fn name(&self) -> &'static str {
        self.operands().name()
    }
    fn ports(&self, sink: &mut Vec<(&'static str, ValueId, crate::check::Expect)>) {
        self.operands().ports(sink);
    }
}

impl From<Attn> for Operation {
    fn from(op: Attn) -> Self {
        Self::Attn(op)
    }
}
impl From<Linear> for Operation {
    fn from(op: Linear) -> Self {
        Self::Linear(op)
    }
}
impl From<Elemwise> for Operation {
    fn from(op: Elemwise) -> Self {
        Self::Elemwise(op)
    }
}
impl From<Gather> for Operation {
    fn from(op: Gather) -> Self {
        Self::Gather(op)
    }
}
impl From<Collective> for Operation {
    fn from(op: Collective) -> Self {
        Self::Collective(op)
    }
}
impl From<Sampling> for Operation {
    fn from(op: Sampling) -> Self {
        Self::Sampling(op)
    }
}
impl From<Ssm> for Operation {
    fn from(op: Ssm) -> Self {
        Self::Ssm(op)
    }
}
impl From<Fused> for Operation {
    fn from(op: Fused) -> Self {
        Self::Fused(op)
    }
}
impl From<Conv> for Operation {
    fn from(op: Conv) -> Self {
        Self::Conv(op)
    }
}
