//! Kernel wrappers: the model author's surface over the op enums.
//!
//! Every wrapper computes its output shape in plain Rust (design §4), declares
//! the outputs with `Recorder::fresh`, pushes the typed op, and returns the
//! fresh [`Value`]s — the model author never writes a shape. In-place kernels
//! construct the SSA pair: the enum's `*_out` field names a fresh id and the
//! compiler folds the pair onto one arena slot, so the wrapper still returns a
//! fresh `Value` (§2). Raggedness is ambient (§5): prefill/chunked wrappers
//! take the fire-aligned tensor directly, with no indptr plumbing. Caches are
//! storage-only ids; geometry enters the graph through [`geometry`] as
//! declared runtime inputs, and plan ops are pure functions of them (§6, §7).
//!
//! The six wrapper modules below ARE the IR's six op families, and each
//! wrapper carries the name of the one variant it pushes: `Attention::MlaDecode`
//! is [`attn::mla_decode`], `Elementwise::RopeYarn` is [`elemwise::rope_yarn`],
//! `CustomCuda::QkvFusedQknormRopeVnormWrite` is
//! [`custom::qkv_fused_qknorm_rope_vnorm_write`]. The surface is the taxonomy:
//! a call site reads as the op's name, so one vocabulary runs from the model
//! text through the plan to the driver arm.

use crate::declare::Weight;
use crate::record::{Recorder, Value};
use model_ir::{
    Attention, Collective, CustomCuda, Dim, Dtype, Elementwise, GeomKind, Layout, Linear,
    RuntimeInput, StructKind, Ty, ValueId,
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

/// Declares one geometry vector of kv space `space` as a runtime input (§7):
/// indptr-shaped vectors are `lanes + 1` long, the page-table vectors are
/// per-lane, and the fire tables — the padding mask, the token→lane map, and
/// the write addressing — are per-token. Everything is `i32` except
/// `RowValid`, the packed `u8` graph-padding mask. The plan wrappers fetch
/// their own; forwards call this for the write geometry a `kv_append` takes.
pub fn geometry(r: &Recorder, space: u32, kind: GeomKind) -> Value {
    let (rows, dtype) = match kind {
        GeomKind::Indptr => (Dim::LanesPlus(1), Dtype::I32),
        GeomKind::Indices | GeomKind::SeqLens | GeomKind::LastPageLen | GeomKind::KvLen => {
            (Dim::Lanes, Dtype::I32)
        }
        GeomKind::RowValid => (Dim::Tokens, Dtype::U8),
        GeomKind::RequestOfToken | GeomKind::WritePage | GeomKind::WriteOffset => {
            (Dim::Tokens, Dtype::I32)
        }
    };
    r.input(
        RuntimeInput::Geometry { space, kind },
        Ty::Tensor {
            shape: vec![rows],
            dtype,
        },
    )
}

/// Declares kv space `space`'s custom attention mask as a runtime input:
/// packed `u8` mask bits, token-aligned, read by `attention.masked`. Both
/// planes carry the bits this way — metal's fire tables and the cuda plan's
/// `Mask` pair; the per-request enabled bits and spans stay driver-derived
/// for now.
pub fn mask(r: &Recorder, space: u32) -> Value {
    r.input(
        RuntimeInput::Mask { space },
        Ty::Tensor {
            shape: vec![Dim::Tokens],
            dtype: Dtype::U8,
        },
    )
}
