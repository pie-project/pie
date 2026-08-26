//! The op vocabulary: one enum per family, variants carrying typed fields.
//! Backend-specific families (`Cuda`) live here ungated — they are pure data;
//! only their dispatch impls are gated.

use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

pub mod attention;
pub mod cuda;
pub mod dist;
pub mod gate;
pub mod gemm;
pub mod hc;
pub mod index;
pub mod layout;
pub mod mla;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod pool;
pub mod rope;
pub mod ssm;

pub use attention::Attention;
pub use cuda::Cuda;
pub use dist::Dist;
pub use gate::Gate;
pub use gemm::Gemm;
pub use hc::Hc;
pub use index::Index;
pub use layout::Layout;
pub use mla::Mla;
pub use mlp::Mlp;
pub use moe::Moe;
pub use norm::Norm;
pub use pool::Pool;
pub use rope::Rope;
pub use ssm::Ssm;

/// One variant per family, so "does this backend cover this op" is a missing
/// match arm in its `Dispatch` impl, caught at compile time. The `Operands`
/// impl and the `From<family>` conversions are generated together from one
/// variant list — the enum cannot gain a family the delegation misses.
macro_rules! operation {
    ($($family:ident),* $(,)?) => {
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub enum Operation {
            $($family($family),)*
        }

        impl Operands for Operation {
            fn inputs(&self, sink: &mut Vec<ValueId>) {
                match self { $(Self::$family(op) => op.inputs(sink),)* }
            }
            fn outputs(&self, sink: &mut Vec<ValueId>) {
                match self { $(Self::$family(op) => op.outputs(sink),)* }
            }
            fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
                match self { $(Self::$family(op) => op.aliases(sink),)* }
            }
            fn name(&self) -> &'static str {
                match self { $(Self::$family(op) => op.name(),)* }
            }
        }

        $(impl From<$family> for Operation {
            fn from(op: $family) -> Self {
                Self::$family(op)
            }
        })*
    };
}

operation!(
    Norm, Mlp, Gemm, Dist, Rope, Moe, Gate, Layout, Ssm, Attention, Mla, Index, Pool, Hc, Cuda,
);
