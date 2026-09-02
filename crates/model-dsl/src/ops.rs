//! Kernel wrappers: the model author's surface over the op enums.
//!
//! Every wrapper computes its output shape in plain Rust, declares the
//! outputs with `Recorder::fresh`, pushes the typed op, and returns the
//! fresh [`Value`]s. Raggedness is ambient: prefill/chunked wrappers take the
//! fire-aligned tensor directly, with no indptr plumbing. Each of the six
//! wrapper modules corresponds to one IR op family, and each wrapper carries
//! the name of the variant it pushes (e.g. `Attention::MlaDecode` is
//! [`attn::mla_decode`]).

use crate::declare::Weight;
use crate::record::Value;
use model_ir::{
    Attention, Collective, CustomCuda, Dim, Dtype, Elementwise, GateActivation, Layout, Linear,
    MropeForm, StructKind, Ty, ValueId,
};

pub mod attn;
pub mod collective;
pub mod custom;
pub mod elemwise;
pub mod layout;
pub mod linear;

/// A two-axis tensor type: the whole surviving shape algebra is `[rows, width]`.
fn tensor(rows: Dim, width: impl Into<u64>, dtype: Dtype) -> Ty {
    Ty::Tensor {
        shape: vec![rows, Dim::Const(width.into())],
        dtype,
    }
}
