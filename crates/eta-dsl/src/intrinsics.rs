//! `intrinsics::*` — first-party stage-scoped values + model constants.
//! Model constants are functions (a runtime value can't be a bare path in
//! Rust; deviation approved). Stage-scoped values emit the IR's
//! [`Op::IntrinsicVal`](eta_ir::op::Op::IntrinsicVal) with the
//! trace-known shape/dtype the registry checks. `intrinsics::kernel::*` second-
//! party surface: a minimal `attn_page_mask` sink now; full rollout deferred.

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
/// The interpreter-visible activation dtype.
///
/// Named `activation_type` because the backend's activation dtype is
/// late-bound, but this constant is not late-bound and never was: the
/// materialization every intrinsic declares — and every tier-0 run produces —
/// is F32. A backend storing bf16/fp8 does so beneath this; nothing in a trace
/// observes that choice, so nothing here can vary with it.
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
/// decl'd `[k, vocab]` regardless of the embed row count. The contract:
/// the classic `K` drafts vs `K+1` verify window are DISTINCT shapes — the CUDA engine's
/// Stage-2 resolves the MtpLogits rows FROM THIS DECL (`mtp_draft_row .. +k`), so
/// a `[K+1,V]` decl would request more rows than the head produces. Model-gated
/// on `has_mtp_logits`. Mirrors the eDSL's `intrinsic_mtp_logits_matrix_dyn(k)`.
pub fn mtp_logits(k: u32) -> Tensor {
    intrinsic_val(
        IntrinsicId::MtpLogits,
        Shape::matrix(k, vocab()),
        Dtype::F32,
    )
}
/// `intrinsics::hidden(width)` — the residual stream at read-out (epilogue),
/// `[n_out, width]`.
///
/// `width` is a parameter because the hidden size is not in
/// [`ModelProfile`](eta_ir::registry::ModelProfile) and the SDK cannot derive
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
        Dtype::F32,
    )
}
/// `intrinsics::layer` — the invocation's layer index (attn taps; U32 scalar).
pub fn layer() -> Tensor {
    intrinsic_val(IntrinsicId::Layer, Shape::SCALAR, Dtype::U32)
}
/// `intrinsics::attn_score(planes)` — how much attention EVERY exported layer
/// paid to each live KV position this fire, `[planes, ATTN_SCORE_KV_MAX]` F32.
/// Readable at the **epilogue** and model-gated on `has_attn_score`.
///
/// **THE GRAPH WROTE IT; THIS READS IT** (`.wiki/alto/attn-score.md` §4).
/// The rectangle is not computed here and it is not recomputed anywhere: the
/// attention capture arm accumulated it as it ran, one plane per (exported
/// layer, query head), and the epilogue is handed the whole thing as a device
/// tensor. So there is no per-layer tap, no mid-forward stage, and no host in
/// the loop — the three things the C++ lineage's `attn_score` needed and the
/// three things alto's one-captured-graph fire cannot give it.
///
/// # The rectangle
///
/// Row `layer * heads + head` is that (layer, head)'s distribution, and rows
/// run LAYER-MAJOR so a program that declares fewer planes than the load
/// exports reads a prefix of the layers rather than a stripe of the heads.
/// Slot semantics per row:
///
///   - `i < kv_len` → the attention probability that (layer, head) assigned to
///     KV position `i`, averaged over the observation window's query rows. The
///     live prefix sums to 1;
///   - `kv_len <= i < ATTN_SCORE_KV_MAX` → exactly `0.0`. A position that does
///     not exist received no attention, so it sorts to the bottom of every
///     eviction ranking without a sentinel — and the backend writes the whole
///     row every fire, so this is never a stale tail.
///
/// # Why per-head, and why the program folds
///
/// Observability wants per-head (§4's table: "per-head is the better answer");
/// the eviction papers want it head-folded. Folding in the kernel would make
/// the second free and the first impossible, so the rectangle is per-head and
/// a program that wants TOVA's or H2O's quantity means `mean` over its own
/// heads — one in-graph reduction at the epilogue, on the device, which is
/// where §4 puts every reduction anyway.
///
/// # `planes` is declared, exactly like `hidden`'s width
///
/// The load's plane count (`exported attention layers × query heads`) is not
/// in [`ModelProfile`](eta_ir::registry::ModelProfile) and the SDK has no host
/// call for it, so the program states it and the backend refuses a claim
/// larger than it exports — the same contract `hidden(width)` and
/// `mtp_logits(k)` already carry. The WIDTH is not declared: see
/// [`ATTN_SCORE_KV_MAX`](eta_ir::registry::ATTN_SCORE_KV_MAX) for why a slab
/// pitch cannot be a per-program number.
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
    /// (`num_q_heads * head_dim`) is a backend constant ETA has no extent for.
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
    /// Three invariants carried from the design (`eta-ir-log.md` §6.5 —
    /// doc not in tree; the three are stated in full below):
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
