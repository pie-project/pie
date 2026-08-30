//! Kernel wrappers: the model author's surface over the op enums.
//!
//! Every wrapper computes its output shape in plain Rust (design §4), declares
//! the outputs with `Recorder::fresh`, pushes the typed op, and returns the
//! fresh [`Value`]s — the model author never writes a shape. In-place kernels
//! construct the SSA pair: the enum's `*_out` field names a fresh id and the
//! compiler folds the pair onto one arena slot, so the wrapper still returns a
//! fresh `Value` (§2). Raggedness is ambient (§5): prefill/chunked wrappers
//! take the fire-aligned tensor directly, with no indptr plumbing. Caches are
//! storage-only ids; geometry enters the graph through
//! [`Input`](crate::Input)'s accessors as declared runtime inputs — a runtime
//! input is something the engine binds, not something a kernel computes, so
//! its declaration belongs beside the handle a forward reaches for it with —
//! and the plan ops here are pure functions of them (§6, §7).
//!
//! The six wrapper modules below ARE the IR's six op families, and each
//! wrapper carries the name of the one variant it pushes: `Attention::MlaDecode`
//! is [`attn::mla_decode`], `Elementwise::RopeYarn` is [`elemwise::rope_yarn`],
//! `CustomCuda::QkvFusedQknormRopeVnormWrite` is
//! [`custom::qkv_fused_qknorm_rope_vnorm_write`]. The surface is the taxonomy:
//! a call site reads as the op's name, so one vocabulary runs from the model
//! text through the plan to the engine arm.

use crate::declare::Weight;
use crate::record::Value;
use model_ir::{
    Attention, Collective, CustomCuda, Dim, Dtype, Elementwise, Layout, Linear, MropeForm,
    StructKind, Ty, ValueId,
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
