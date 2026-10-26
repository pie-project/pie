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

pub use attn::{Attn, Blocks, IndexScore, RaggedMask, Relative};
pub use collective::Collective;
pub use conv::{Conv, GridRule, TimePad, VoxelSegment};
pub use elemwise::{
    ActFn, BinaryOp, Bounds, Elemwise, GateActivation, GateHeads, ModulateForm, MropeForm,
    NormAxis, NormNext, NormWeight, ResidualGate, RopeForm, RopeFreqs, Turn, Yarn,
};
pub use fused::{Fused, NormKind, RegionRead};
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
    /// Every op the IR has, by name: each group's, in group order.
    pub fn vocabulary() -> impl Iterator<Item = &'static str> {
        [
            Attn::NAMES,
            Linear::NAMES,
            Elemwise::NAMES,
            Gather::NAMES,
            Collective::NAMES,
            Sampling::NAMES,
            Ssm::NAMES,
            Fused::NAMES,
            Conv::NAMES,
        ]
        .into_iter()
        .flatten()
        .copied()
    }

    /// Whether `name` is an op's name.
    #[must_use]
    pub fn names_an_op(name: &str) -> bool {
        Self::vocabulary().any(|op| op == name)
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Each op's name is its group's prefix and its variant in snake case,
    /// and no two ops share one: a name keys every table that keys on ops.
    #[test]
    fn every_op_has_a_name_of_its_own() {
        let groups = [
            ("attn", Attn::NAMES),
            ("linear", Linear::NAMES),
            ("elemwise", Elemwise::NAMES),
            ("gather", Gather::NAMES),
            ("collective", Collective::NAMES),
            ("sampling", Sampling::NAMES),
            ("ssm", Ssm::NAMES),
            ("fused", Fused::NAMES),
            ("conv", Conv::NAMES),
        ];
        let mut seen = std::collections::BTreeSet::new();
        for (group, names) in groups {
            assert!(!names.is_empty(), "`{group}` names no op");
            for name in names {
                let variant = name
                    .strip_prefix(group)
                    .and_then(|rest| rest.strip_prefix('.'))
                    .unwrap_or_else(|| panic!("`{name}` is not under `{group}.`"));
                assert!(
                    !variant.is_empty()
                        && variant
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                    "`{name}` is not snake case"
                );
                assert!(seen.insert(*name), "two ops are named `{name}`");
            }
        }
        assert_eq!(seen.len(), Operation::vocabulary().count());
        assert!(Operation::names_an_op("elemwise.residual_add"));
        assert!(!Operation::names_an_op("elemwise.gated_residual_add"));
    }
}
