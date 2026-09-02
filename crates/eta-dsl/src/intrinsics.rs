//! `intrinsics::*`: first-party stage-scoped values and model constants
//! (functions, since a runtime value can't be a bare path in Rust). Stage-
//! scoped values emit the IR's
//! [`Op::IntrinsicVal`](eta_ir::op::Op::IntrinsicVal). `intrinsics::kernel::*`
//! is the second-party surface, minimal for now.

use eta_ir::op::IntrinsicId;
use eta_ir::types::{Dtype, Shape};

use crate::context::current_rows;
use crate::model;
use crate::value::{Tensor, intrinsic_val};

/// Model vocabulary size (trace-known; `[intrinsics::vocab()]` in shapes).
pub fn vocab() -> u32 {
    model::vocab()
}
/// Tokens per KV page (trace-known).
pub fn page_size() -> u32 {
    model::page_size()
}
/// The interpreter-visible activation dtype. Always F32: a backend storing
/// bf16/fp8 does so beneath this, and nothing in a trace observes that
/// choice.
#[allow(non_upper_case_globals)]
pub const activation_type: Dtype = Dtype::F32;

fn logits_shape() -> Shape {
    let rows = current_rows();
    let v = vocab();
    Shape::matrix(rows.max(1), v)
}

/// `intrinsics::logits()` — the LM-head logits, `[n_out, vocab]` F32. For
/// a single read-out row the SDK reshapes to `[vocab]` so single-position
/// samplers read a vector (the IR's golden reference does the same).
pub fn logits() -> Tensor {
    let t = intrinsic_val(IntrinsicId::Logits, logits_shape(), Dtype::F32);
    single_row_reshape(t)
}
/// `intrinsics::mtp_logits(k)` — the model's `k` draft/future-token heads,
/// decl'd `[k, vocab]` regardless of the embed row count. The classic `K`
/// drafts vs `K+1` verify window are distinct shapes; a `[K+1,V]` decl would
/// request more rows than the head produces. Model-gated on `has_mtp_logits`.
pub fn mtp_logits(k: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::MtpLogits,
        Shape::matrix(k, vocab()),
        Dtype::F32,
    )
}
/// `intrinsics::mtp_drafts(n)` — the draft head's token ids for the fire's
/// readout rows, `[n]` I32 with `n = n_out × depth`, row-major: readout row
/// `i`'s chain is `[i·depth, (i+1)·depth)`, and it is conditioned on the
/// trunk's argmax at row `i` (the same argmax `reduce_argmax(logits())`
/// reads), so a verifier that accepts row `m` continues with the chain at
/// `m`. `depth` is the model's (`model::mtp_depth()`), zero without a head.
/// Model-gated on `mtp_depth > 0`.
pub fn mtp_drafts(n: u32) -> Tensor {
    intrinsic_val(IntrinsicId::MtpDrafts, Shape::vector(n.max(1)), Dtype::I32)
}
/// `intrinsics::hidden(width)` — the residual stream at read-out (epilogue),
/// `[n_out, width]`. `width` is a parameter because the hidden size is not
/// in [`ModelProfile`](eta_ir::registry::ModelProfile); `bind` checks only
/// rank and row count, so a wrong width is carried into the plan's extents
/// rather than refused.
pub fn hidden(width: u32) -> Tensor {
    let rows = current_rows().max(1);
    intrinsic_val(
        IntrinsicId::Hidden,
        Shape::matrix(rows, width.max(1)),
        activation_type,
    )
}
/// `intrinsics::query(width)` — this layer's projected query (attn taps),
/// `[width]`. Declared, not derived, for the same reason as [`hidden`].
pub fn query(width: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::Query,
        Shape::vector(width.max(1)),
        activation_type,
    )
}
/// `intrinsics::value_head()` — model-gated scalar value head (epilogue).
pub fn value_head() -> Tensor {
    intrinsic_val(
        IntrinsicId::ValueHead,
        Shape::vector(current_rows().max(1)),
        Dtype::F32,
    )
}
/// `intrinsics::layer` — the invocation's layer index (attn taps; U32 scalar).
pub fn layer() -> Tensor {
    intrinsic_val(IntrinsicId::Layer, Shape::SCALAR, Dtype::U32)
}
/// `intrinsics::attn_score(planes)` — how much attention every exported layer
/// paid to each live KV position this fire, `[planes, ATTN_SCORE_KV_MAX]` F32.
/// Readable at the epilogue, model-gated on `has_attn_score`. Not computed
/// here: the attention capture arm accumulated it as it ran, one plane per
/// (exported layer, query head), handed to the epilogue as a device tensor.
///
/// # The rectangle
///
/// Row `layer * heads + head` is that (layer, head)'s distribution, rows run
/// layer-major. Per row: `i < kv_len` is the attention probability assigned
/// to KV position `i` (live prefix sums to 1); `kv_len <= i < ATTN_SCORE_KV_MAX`
/// is exactly `0.0`, written every fire (never a stale tail).
///
/// # Why per-head, and why the program folds
///
/// Folding in the kernel would make head-folded free and per-head
/// impossible, so the rectangle is per-head; a program wanting TOVA's or
/// H2O's quantity means `mean` over its own heads, in-graph at the epilogue.
///
/// `planes` is declared, like `hidden`'s width, since the load's plane count
/// is not in [`ModelProfile`](eta_ir::registry::ModelProfile). The width is
/// not declared: see [`ATTN_SCORE_KV_MAX`](eta_ir::registry::ATTN_SCORE_KV_MAX).
pub fn attn_score(planes: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::AttnScore,
        Shape::matrix(planes.max(1), eta_ir::registry::ATTN_SCORE_KV_MAX),
        Dtype::F32,
    )
}

/// The width of every [`attn_score`] row — re-exported so an author spells the
/// ceiling once and gets the one the backend carved.
pub const fn attn_score_kv_max() -> u32 {
    eta_ir::registry::ATTN_SCORE_KV_MAX
}

/// Reshape a `[1, vocab]` logits matrix to `[vocab]` for the single-row case
/// (matches the IR's golden reference). Multi-row passes keep the matrix.
fn single_row_reshape(t: Tensor) -> Tensor {
    let s = t.shape();
    if s.rank() == 2 && s.dims()[0] == 1 {
        crate::value::reshape(t, [s.dims()[1]])
    } else {
        t
    }
}

/// Second-party kernels (`intrinsics::kernel::*`). A minimal `attn_page_mask`
/// sink exists now so the sink stage-precedence lint is enforceable.
pub mod kernel {
    use crate::context::{emit, intern_name, record_sink};
    use crate::error::Span;
    use crate::value::{AsTensor, Tensor};
    use alloc::string::String;
    use alloc::vec;
    use eta_ir::op::{IntrinsicId, Op};
    use eta_ir::registry::SinkScope;
    use eta_ir::types::{Dtype, Shape, ValueType};

    /// `envelope_dot(p_max)` — Quest page criticality for this layer, `[p_max]`
    /// F32 (Tang et al., arXiv:2406.10774). Model-gated on the backend's
    /// `envelope_dot` kernel. Per slot: a finalized owned page gets its
    /// criticality upper bound; a page still filling gets `+inf` (always
    /// selected); a slot past the request's page list gets `-inf`. `p_max`
    /// is the program's own page ceiling, refused rather than truncated if
    /// exceeded. Takes no argument even though the op is
    /// `KernelCall(envelope_dot, [query])`: the query's width is a backend
    /// constant ETA has no extent for, so the declared query is a handle
    /// the backend binds the real row onto.
    #[track_caller]
    pub fn envelope_dot(p_max: u32) -> Tensor {
        let query_ty = ValueType::new(Shape::vector(1), super::activation_type);
        let query = emit(
            Op::IntrinsicVal {
                intr: IntrinsicId::Query,
                shape: query_ty.shape,
                dtype: query_ty.dtype,
            },
            &[query_ty],
        );
        let score_ty = ValueType::new(Shape::vector(p_max), Dtype::F32);
        let name = intern_name("envelope_dot");
        Tensor::node(
            emit(
                Op::KernelCall {
                    name,
                    args: vec![query],
                    shape: score_ty.shape,
                    dtype: score_ty.dtype,
                },
                &[score_ty],
            ),
            score_ty,
        )
    }

    /// `attn_page_mask(mask)` — a configuration sink: this layer's attention
    /// consumes the page mask. `mask` is `[p_max]`, one entry per page of
    /// the request's page list in order; nonzero keeps the page. Recorded
    /// both as an `Op::SinkCall` (so the value reaches the backend) and in
    /// the session's sink list (so T11 can check it precedes the layer's
    /// attention).
    #[track_caller]
    pub fn attn_page_mask(mask: impl AsTensor) {
        let span = Span::here();
        let (mask, _) = mask.to_arg().materialize();
        let name = intern_name("attn_page_mask");
        emit(
            Op::SinkCall {
                name,
                args: vec![mask],
            },
            &[],
        );
        record_sink(String::from("attn_page_mask"), span, SinkScope::Attention);
    }

    /// `lora(a, b, sites)` — a pass-wide configuration sink: the whole forward
    /// applies the low-rank delta `W'x = Wx + B(Ax)` at the declared
    /// projection sites. Legal only in the pass prologue (T11). `a` is
    /// `[num_layers, R, d]` and `b` is `[num_layers, d_out, R]`, rank `R`
    /// trace-known (a different rank re-traces); the LoRA scale `alpha/R`
    /// is folded into `b`'s contents; `sites` is a trace-known constant over
    /// the model's site vocabulary (placement is structure, weights are
    /// contents).
    #[track_caller]
    pub fn lora(a: impl AsTensor, b: impl AsTensor, sites: impl AsTensor) {
        let span = Span::here();
        let (a, _) = a.to_arg().materialize();
        let (b, _) = b.to_arg().materialize();
        let (sites, _) = sites.to_arg().materialize();
        let name = intern_name("lora");
        emit(
            Op::SinkCall {
                name,
                args: vec![a, b, sites],
            },
            &[],
        );
        record_sink(String::from("lora"), span, SinkScope::PassWide);
    }

    /// `adapter_scale(l, sites)` — the SCALE form of the adapter sink
    /// (IA3): the whole forward applies `y = l ⊙ y` at the declared
    /// sites. `l` is `[num_layers, d_out]` with any static scale folded
    /// into its contents; `sites` is the trace-known placement constant.
    /// Wire-encodes as the 2-argument `lora` sink (arity selects the
    /// form — the engine's resolver branches on it).
    #[track_caller]
    pub fn adapter_scale(l: impl AsTensor, sites: impl AsTensor) {
        let span = Span::here();
        let (l, _) = l.to_arg().materialize();
        let (sites, _) = sites.to_arg().materialize();
        let name = intern_name("lora");
        emit(
            Op::SinkCall {
                name,
                args: vec![l, sites],
            },
            &[],
        );
        record_sink(String::from("lora"), span, SinkScope::PassWide);
    }
}
