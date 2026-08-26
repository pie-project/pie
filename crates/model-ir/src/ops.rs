//! The op vocabulary: one enum per family, variants carrying typed fields.
//! Backend-specific families (`CustomCuda`) live here ungated — they are pure
//! data; only their dispatch impls are gated. Each family is declared by hand —
//! the enum and its `Operands` impl side by side, so the field list and the
//! def-use reading of it sit in one file; exhaustive matches (no `_` arms) force
//! every new variant through all four methods.
//!
//! Six families, and which one a new op belongs to is settled by running this
//! ordered procedure — first match wins, so a variant that satisfies two
//! criteria lands in the earlier family:
//!
//! 1. Tokens interact, or a sequence cache is touched (kv, ssm state, indexer,
//!    compressor pool) → `Attention`.
//! 2. Learned-weight channel mixing, and its epilogues → `Linear`.
//! 3. Movement, split, or select without compute → `Layout`.
//! 4. Per-token local math, tokens independent — per-token reductions like
//!    rmsnorm's mean-of-squares included → `Elementwise`.
//! 5. Crosses devices — the SPMD tensor-parallel collectives → `Collective`.
//! 6. Inexpressible portably, one backend's fusion → `Custom<Backend>`, today
//!    only `CustomCuda`.

use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

pub mod attn;
pub mod collective;
pub mod custom_cuda;
pub mod elemwise;
pub mod layout;
pub mod linear;

pub use attn::Attention;
pub use collective::Collective;
pub use custom_cuda::CustomCuda;
pub use elemwise::Elementwise;
pub use layout::Layout;
pub use linear::Linear;

/// One variant per family, so "does this backend cover this op" is a missing
/// match arm in its `Dispatch` impl, caught at compile time. Written out by
/// hand — the whole vocabulary in one glance, no macro between the reader and
/// the list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    Attention(Attention),
    Linear(Linear),
    Elementwise(Elementwise),
    Layout(Layout),
    Collective(Collective),
    CustomCuda(CustomCuda),
}

impl Operation {
    /// The family payload, seen through the one trait they all share — the
    /// single match every `Operands` method delegates through, so the four
    /// cannot drift from each other.
    fn operands(&self) -> &dyn Operands {
        match self {
            Self::Attention(op) => op,
            Self::Linear(op) => op,
            Self::Elementwise(op) => op,
            Self::Layout(op) => op,
            Self::Collective(op) => op,
            Self::CustomCuda(op) => op,
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
