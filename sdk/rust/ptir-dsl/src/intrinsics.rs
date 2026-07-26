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
    use crate::context::{emit, intern_name, record_sink};
    use crate::error::Span;
    use crate::value::{AsTensor, Tensor};
    use alloc::string::String;
    use alloc::vec;
    use pie_ptir::op::{IntrinsicId, Op};
    use pie_ptir::registry::SinkScope;
    use pie_ptir::types::{DType, Shape, ValueType};

    /// `envelope_dot(p_max)` — Quest page criticality for THIS layer, `[p_max]`
    /// F32 (Tang et al., arXiv:2406.10774). Model-gated on the backend's
    /// `envelope_dot` kernel; a backend without per-page key envelopes refuses
    /// the program at bind.
    ///
    /// Slot semantics, so a consumer can be written without guessing:
    ///   - a page this request owns whose contents are final -> its criticality
    ///     upper bound, `max` over kv heads of `Σ_qh Σ_d max(q·kmin, q·kmax)`;
    ///   - a page the current forward is still filling -> `+inf`, i.e. always
    ///     selected. Quest keeps the local window anyway, and a stale bound
    ///     would be a silent mis-rank;
    ///   - a slot past this request's page list -> `-inf`, so a `rank_le` never
    ///     picks one.
    ///
    /// `p_max` is the program's own page ceiling; the backend refuses a request
    /// whose page list is longer, rather than truncating it.
    ///
    /// Takes no argument even though the underlying op is
    /// `KernelCall(envelope_dot, [query])`: the query it scores is the model's
    /// projected query for this fire's last token, whose width
    /// (`num_q_heads * head_dim`) is a backend constant PTIR has no extent for.
    /// The declared query is therefore a HANDLE — the backend binds the real
    /// row — and the DSL emits it rather than asking an author to invent a
    /// width.
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
        let score_ty = ValueType::new(Shape::vector(p_max), DType::F32);
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

    /// `attn_page_mask(mask)` — a configuration sink (overview §6.1): this
    /// layer's attention consumes the page mask. Returns nothing.
    ///
    /// `mask` is `[p_max]`, one entry per page of the request's page list in
    /// order; a nonzero entry keeps the page. It is recorded twice on purpose:
    /// as an `Op::SinkCall` so the mask VALUE reaches the backend, and in the
    /// session's sink list so T11 can check this call precedes the layer's
    /// attention. Dropping the argument (as this did before it had a lowering)
    /// makes the sink a no-op that still type-checks.
    #[track_caller]
    pub fn attn_page_mask(mask: impl AsTensor) {
        let span = Span::here();
        let (mask, _) = mask.to_arg().materialize();
        let name = intern_name("attn_page_mask");
        emit(Op::SinkCall { name, args: vec![mask] }, &[]);
        record_sink(String::from("attn_page_mask"), span, SinkScope::Attention);
    }
}
