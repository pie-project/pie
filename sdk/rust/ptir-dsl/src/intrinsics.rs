//! `intrinsics::*` — first-party stage-scoped values + model constants (overview
//! §4). Model constants are functions (a runtime value can't be a bare path in
//! Rust; deviation approved). Stage-scoped values emit echo's
//! [`Op::IntrinsicVal`](pie_ptir::op::Op::IntrinsicVal) with the
//! trace-known shape/dtype the registry checks. `intrinsics::kernel::*` second-
//! party surface: a minimal `attn_page_mask` sink now; full rollout in P7.

use pie_ptir::op::IntrinsicId;
use pie_ptir::types::{DType, Shape};

use crate::context::current_rows;
use crate::model;
use crate::value::{intrinsic_val, Tensor};

/// Model vocabulary size (trace-known; `[intrinsics::vocab()]` in shapes).
pub fn vocab() -> u32 {
    model::vocab()
}
/// Tokens per KV page (trace-known).
pub fn page_size() -> u32 {
    model::page_size()
}
/// Number of transformer layers (trace-known).
pub fn num_layers() -> u32 {
    model::num_layers()
}
/// The late-bound backend activation dtype (`intrinsics::activation_type`, §4).
#[allow(non_upper_case_globals)]
pub const activation_type: DType = DType::F32;

fn logits_shape() -> Shape {
    let rows = current_rows();
    let v = vocab();
    Shape::matrix(rows.max(1), v)
}

/// `intrinsics::logits()` — the LM-head logits, `[n_out, vocab]` F32 (§5.1). For
/// a single read-out row the SDK reshapes to `[vocab]` so single-position
/// samplers read a vector (echo's §3 golden does the same).
pub fn logits() -> Tensor {
    let t = intrinsic_val(IntrinsicId::Logits, logits_shape(), DType::F32);
    single_row_reshape(t)
}
/// `intrinsics::mtp_logits(k)` — the model's `k` draft/future-token heads (§4),
/// decl'd `[k, vocab]` regardless of the embed row count. echo's §6.1 contract:
/// the classic `K` drafts vs `K+1` verify window are DISTINCT shapes — charlie's
/// Stage-2 resolves the MtpLogits rows FROM THIS DECL (`mtp_draft_row .. +k`), so
/// a `[K+1,V]` decl would request more rows than the head produces. Model-gated
/// on `has_mtp_logits`. Mirrors the eDSL's `intrinsic_mtp_logits_matrix_dyn(k)`.
pub fn mtp_logits(k: u32) -> Tensor {
    intrinsic_val(IntrinsicId::MtpLogits, Shape::matrix(k, vocab()), DType::F32)
}
/// `intrinsics::hidden()` — the residual stream at read-out (epilogue).
pub fn hidden() -> Tensor {
    let rows = current_rows().max(1);
    // Hidden width is a model constant; modeled here as the activation rows.
    intrinsic_val(IntrinsicId::Hidden, Shape::matrix(rows, vocab()), activation_type)
}
/// `intrinsics::query()` — this layer's projected query (attn taps, §5.3).
pub fn query() -> Tensor {
    intrinsic_val(IntrinsicId::Query, Shape::vector(vocab()), activation_type)
}
/// `intrinsics::value_head()` — model-gated scalar value head (epilogue).
pub fn value_head() -> Tensor {
    intrinsic_val(IntrinsicId::ValueHead, Shape::vector(current_rows().max(1)), DType::F32)
}
/// `intrinsics::layer` — the invocation's layer index (attn taps; U32 scalar).
pub fn layer() -> Tensor {
    intrinsic_val(IntrinsicId::Layer, Shape::SCALAR, DType::U32)
}

/// Reshape a `[1, vocab]` logits matrix to `[vocab]` for the single-row case
/// (matches echo's §3 golden). Multi-row passes keep the matrix.
fn single_row_reshape(t: Tensor) -> Tensor {
    let s = t.shape();
    if s.rank() == 2 && s.dims()[0] == 1 {
        crate::value::reshape(t, [s.dims()[1]])
    } else {
        t
    }
}

/// Second-party kernels (`intrinsics::kernel::*`, overview §4). Full rollout is
/// P7; a minimal `attn_page_mask` sink exists now so T11 is enforceable.
pub mod kernel {
    use crate::context::record_sink;
    use crate::error::Span;
    use crate::value::AsTensor;
    use alloc::string::String;
    use pie_ptir::registry::SinkScope;

    /// `attn_page_mask(mask)` — a configuration sink (overview §6.1): this
    /// layer's attention consumes the page mask. Returns nothing. Recorded for
    /// T11 precedence (must precede this layer's attention).
    #[track_caller]
    pub fn attn_page_mask(mask: impl AsTensor) {
        let span = Span::here();
        let _ = mask.to_arg();
        record_sink(String::from("attn_page_mask"), span, SinkScope::Attention);
    }
}
