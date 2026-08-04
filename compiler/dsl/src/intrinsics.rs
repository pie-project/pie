//! `intrinsics::*` — first-party stage-scoped values + model constants.
//! Model constants are functions (a runtime value can't be a bare path in
//! Rust; deviation approved). Stage-scoped values emit the IR's
//! [`Op::IntrinsicVal`](pie_ir::op::Op::IntrinsicVal) with the
//! trace-known shape/dtype the registry checks. `intrinsics::kernel::*` second-
//! party surface: a minimal `attn_page_mask` sink now; full rollout deferred.

use pie_ir::op::IntrinsicId;
use pie_ir::types::{DType, Shape};

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
/// The interpreter-visible activation dtype.
///
/// Named `activation_type` because the backend's activation dtype is
/// late-bound, but this constant is not late-bound and never was: the
/// materialization every intrinsic declares — and every tier-0 run produces —
/// is F32. A backend storing bf16/fp8 does so beneath this; nothing in a trace
/// observes that choice, so nothing here can vary with it.
#[allow(non_upper_case_globals)]
pub const activation_type: DType = DType::F32;

fn logits_shape() -> Shape {
    let rows = current_rows();
    let v = vocab();
    Shape::matrix(rows.max(1), v)
}

/// `intrinsics::logits()` — the LM-head logits, `[n_out, vocab]` F32. For
/// a single read-out row the SDK reshapes to `[vocab]` so single-position
/// samplers read a vector (the IR's golden reference does the same).
pub fn logits() -> Tensor {
    let t = intrinsic_val(IntrinsicId::Logits, logits_shape(), DType::F32);
    single_row_reshape(t)
}
/// `intrinsics::mtp_logits(k)` — the model's `k` draft/future-token heads,
/// decl'd `[k, vocab]` regardless of the embed row count. The contract:
/// the classic `K` drafts vs `K+1` verify window are DISTINCT shapes — the CUDA driver's
/// Stage-2 resolves the MtpLogits rows FROM THIS DECL (`mtp_draft_row .. +k`), so
/// a `[K+1,V]` decl would request more rows than the head produces. Model-gated
/// on `has_mtp_logits`. Mirrors the eDSL's `intrinsic_mtp_logits_matrix_dyn(k)`.
pub fn mtp_logits(k: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::MtpLogits,
        Shape::matrix(k, vocab()),
        DType::F32,
    )
}
/// `intrinsics::hidden(width)` — the residual stream at read-out (epilogue),
/// `[n_out, width]`.
///
/// `width` is a parameter because the hidden size is not in
/// [`ModelProfile`](pie_ir::registry::ModelProfile) and the SDK cannot derive
/// it. `bind` deliberately checks only the rank and row count for this
/// intrinsic, so a wrong width is not refused — it is carried into the plan's
/// extents. Pass the model's hidden size; it is the same kind of declared
/// ceiling as `mtp_logits`'s `k` and `attn_score`'s `kv_max`.
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
        DType::F32,
    )
}
/// `intrinsics::layer` — the invocation's layer index (attn taps; U32 scalar).
pub fn layer() -> Tensor {
    intrinsic_val(IntrinsicId::Layer, Shape::SCALAR, DType::U32)
}
/// `intrinsics::attn_score(kv_max)` — how much attention THIS layer paid to
/// each live KV position, `[kv_max]` F32. Readable only at `on_attn` (the
/// scores do not exist until the layer's attention has run) and model-gated on
/// `has_attn_score`.
///
/// The value is the attention probability the request's MOST RECENT query
/// token assigned to KV position `i`, averaged over the query heads. That is
/// TOVA's quantity exactly (Oren et al., arXiv:2401.06104), H2O's per-step
/// increment (Zhang et al., arXiv:2306.14048), and SnapKV's observation window
/// at width 1.
///
/// Heads are folded by the backend, not by the program, for the same reason
/// Quest's `envelope_dot` folds them: the paged KV layout carries ONE page list
/// per request, so eviction is a per-request decision and a per-head score has
/// no representable consumer. Folding in the kernel also avoids materialising
/// `[num_heads, kv_max]` per layer.
///
/// Slot semantics:
///   - `i < kv_len` -> the mean attention probability at that position; the
///     live prefix sums to 1;
///   - `kv_len <= i < kv_max` -> `0.0`. A position that does not exist received
///     no attention, so it sorts to the bottom of every eviction ranking
///     without a sentinel.
///
/// `kv_max` is the program's own ceiling (`prompt + max_tokens`, mirroring
/// `envelope_dot`'s `p_max`); the backend refuses a request longer than it
/// rather than truncating.
pub fn attn_score(kv_max: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::AttnScore,
        Shape::vector(kv_max.max(1)),
        DType::F32,
    )
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
    use pie_ir::op::{IntrinsicId, Op};
    use pie_ir::registry::SinkScope;
    use pie_ir::types::{DType, Shape, ValueType};

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

    /// `attn_page_mask(mask)` — a configuration sink: this
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
    /// projection sites. Returns nothing; legal only in the pass prologue
    /// (T11 — a pass-wide sink must precede everything that consumes it).
    ///
    /// Three invariants carried from the design (`tensor-ir-log.md` §6.5):
    ///
    /// * `a` is `[num_layers, R, d]` and `b` is `[num_layers, d_out, R]`, with
    ///   the rank `R` trace-known — a different rank is a different traced
    ///   program (a different bucket). The weight *contents* are data (fed
    ///   through channels or computed in-graph), so swapping an adapter is
    ///   re-seeding, never re-tracing.
    /// * The LoRA scale `alpha/R` is folded into `b`'s contents — per-adapter
    ///   data, so there is no scalar argument here.
    /// * `sites` is a trace-known constant over the model's site vocabulary
    ///   (q/k/v/o/up/..): placement is structure, weights are contents.
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
    /// form — the driver's resolver branches on it).
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
