use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

pub mod attn;
pub mod collective;
pub mod custom_cuda;
pub mod elemwise;
pub mod layout;
pub mod linear;
pub mod spatial;

pub use attn::{Attention, RaggedMask};
pub use collective::Collective;
pub use custom_cuda::CustomCuda;
pub use elemwise::{Elementwise, GateActivation, ModulateForm, MropeForm, NormKind, RopeForm};
pub use layout::Layout;
pub use linear::Linear;
pub use spatial::{GridRule, Spatial, TimePad, VoxelSegment};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    Attention(Attention),
    Linear(Linear),
    Elementwise(Elementwise),
    Layout(Layout),
    Collective(Collective),
    CustomCuda(CustomCuda),
    Spatial(Spatial),
}

impl Operation {
    fn operands(&self) -> &dyn Operands {
        match self {
            Self::Attention(op) => op,
            Self::Linear(op) => op,
            Self::Elementwise(op) => op,
            Self::Layout(op) => op,
            Self::Collective(op) => op,
            Self::CustomCuda(op) => op,
            Self::Spatial(op) => op,
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
}

impl From<Attention> for Operation {
    fn from(op: Attention) -> Self {
        Self::Attention(op)
    }
}
impl From<Linear> for Operation {
    fn from(op: Linear) -> Self {
        Self::Linear(op)
    }
}
impl From<Elementwise> for Operation {
    fn from(op: Elementwise) -> Self {
        Self::Elementwise(op)
    }
}
impl From<Layout> for Operation {
    fn from(op: Layout) -> Self {
        Self::Layout(op)
    }
}
impl From<Collective> for Operation {
    fn from(op: Collective) -> Self {
        Self::Collective(op)
    }
}
impl From<CustomCuda> for Operation {
    fn from(op: CustomCuda) -> Self {
        Self::CustomCuda(op)
    }
}
impl From<Spatial> for Operation {
    fn from(op: Spatial) -> Self {
        Self::Spatial(op)
    }
}
