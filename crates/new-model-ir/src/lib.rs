//! The canonical IR of a traced forward pass: typed op enums, `Def` × `Ty`
//! value declarations, guard conditions, and the serializable `Plan` that
//! carries them. Backend-free by construction — a leaf of the workspace, pure
//! data plus serde, so every crate that reads a plan reads only this one.

pub mod check;
pub mod cond;
pub mod operands;
pub mod ops;
pub mod plan;
pub mod value;

pub use check::{Fault, check, checked};
pub use cond::Cond;
pub use operands::Operands;
pub use ops::{
    Attention, Cuda, Dist, Gate, Gemm, Hc, Index, Layout, Mla, Mlp, Moe, Norm, Operation, Pool,
    Rope, Ssm,
};
pub use plan::{CacheRow, Node, Param, Plan, Plane, Repr, Seam, Shard};
pub use value::{Def, Dim, Dtype, GeomKind, RuntimeInput, StructKind, Ty, ValueDecl, ValueId};

/// The derive, re-exported so downstream writes `#[derive(Operands)]` against
/// this crate alone. Shares the trait's name across namespaces, serde-style.
pub use new_model_ir_macros::Operands;
