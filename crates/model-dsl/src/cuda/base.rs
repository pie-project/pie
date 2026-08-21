//! CUDA generation statements: projections, attention, rope, KV writes,
//! recurrent updates, and the surrounding launches.

use super::*;

/// `kernels::moe::token_batched_weighted_sum`, or the `..._add` form when
/// the residual folds into the same launch — a SYMBOL CHOICE, which is why
/// this wrapper survives B4-gen step 7; each arm is the generated fn.
pub fn weighted_sum(weights: &Val, x: &Val, hidden: u32, residual: Option<&Val>) -> Val {
    match residual {
        // The `_add` form aliases its result over the residual; the plain
        // form's result is ruled `rows(weights) x width(src)` by nothing —
        // the caller still states it (`Unstated`, dtype trap: the rows
        // carrier is f32 while the result is bf16).
        Some(r) => generated::token_batched_weighted_sum_add(x, weights, r, weights.layer, None),
        None => generated::token_batched_weighted_sum(
            (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16),
            x,
            weights,
            weights.layer,
            None,
        ),
    }
}

/// `kernels::attn::attention_xqa_decode_bf16_prepared` (whose contract includes the fire-wide
/// XQA prepare — and which is therefore declared `whole`; see [`model_ir::kernels`]).
///
/// The swept signature: `q` and the KV view as operands, and a five-scalar
/// run `[num_q_heads, num_kv_heads, head_dim, sm_scale, num_requests]` —
/// the workspace carve is launch-local `ctx.scratch` at the ROUTINE's own
/// capacities (substrate, not a model fact), and the request count is a
/// fire extent the lowering splices.
#[allow(clippy::too_many_arguments)]
pub fn attention_xqa_decode(
    q: &Val,
    kv: &Kv,
    window_left: i32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sm_scale: f32,
) -> Option<Val> {
    // THE WINDOW IS A FACT NOW, NOT A SCALAR: this routine attends the whole
    // context and declares no window `Const`. It stays on the signature
    // because a caller states the deployment's window here and the guard
    // predicates above still read the same number.
    let _ = window_left;
    let kvc = rt_object(&q.t, "kv_cache", Some(kv.l));
    let out = q.t.inner.borrow().inside_value_region();
    let shape = (!out).then(|| q.t.inner.borrow().value_shape(q.id));
    record_with_extents(
        &q.t,
        Some(kv.l),
        "attn::attention_xqa_decode_bf16_prepared",
        vec![],
        kv_state(kv),
        vec![num_q_heads, num_kv_heads, head_dim, sm_scale.to_bits(), 0],
        vec![requests_extent(6)],
        vec![q.id, kvc],
        shape.map(|s| (s, DType::BF16)),
    )
}

/// State that this fire needs the paged-DECODE schedule raised.
///
/// The dispatches below read a plan the driver walks on the CPU from this
/// batch's page CSR. It used to find out by looking for the dispatch symbol
/// in the lowered kernel table; a text that states the preparation is the
/// text saying what it needs, and the driver stops knowing which symbol
/// implies which schedule.
///
/// Stated ONCE per fire, outside the layer loop: a schedule is a property of
/// the batch, not of a layer.
///
/// `full_attention` picks between the two plans a deployment may run. A model
/// whose layers disagree on head dim — gemma-4's sliding against its full —
/// states this twice, once per dim, because the planner bakes the number in.
/// Stated by the three decode helpers below rather than by hand, so a text
/// cannot state the dispatch and forget the schedule it reads. It is `pub`
/// because a text that reaches the dispatch another way still has to say so.
///
/// Restating it costs nothing: the driver asks whether ANY prep names the
/// decode schedule, and raises one plan either way.
pub fn decode_plan(t: &Trace, head_dim: u32, full_attention: bool) -> model_ir::trace::ValueId {
    t.with(None, |b| {
        b.push_prep(model_ir::trace::PrepKind::DecodeAttention {
            head_dim,
            full_attention,
        })
    })
}

/// [`decode_plan`]'s prefill twin, stated by the three PLANNED prefill helpers.
///
/// The planless pair state nothing: they walk their own schedule from the host
/// CSR mirrors when the statement runs, so there is none to raise.
pub fn prefill_plan(t: &Trace, head_dim: u32) -> model_ir::trace::ValueId {
    t.with(None, |b| {
        b.push_prep(model_ir::trace::PrepKind::PrefillAttention { head_dim })
    })
}

/// `kernels::attn::dispatch_attention_flashinfer_decode` against the decode plan its contract
/// obligates. `[window_left, logits_soft_cap, sm_scale]` is the run the
/// routine's three `Const` marks claim, in their order.
pub fn attention_flashinfer_decode(
    q: &Val,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
) -> Option<Val> {
    let plan = decode_plan(&q.t, head_dim, window_left == -1);
    let kvc = rt_object(&q.t, "kv_cache", Some(kv.l));
    attn_at_planned(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_decode",
        {
            let mut p = vec![window_left as u32];
            p.extend(cap_and_scale(soft_cap, sm_scale, head_dim));
            p
        },
        Some(plan),
        vec![kvc],
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_prefill_bf16` — the dispatch alone.
pub fn attention_flashinfer_prefill(
    q: &RaggedVal,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
) -> Option<Val> {
    // THE WINDOW IS A FACT NOW, NOT A SCALAR. Every routine below asks
    // for it through `keys::WindowLeft`; the statement used to carry it
    // and carried it into the slot a soft cap is read from. It stays on
    // the signature because a caller states the deployment's window here
    // and the guard predicates above still read the same number.
    let _ = window_left;
    // THE EDGE, WRITTEN DOWN. The prep raised this fire's prefill schedule and
    // the statement below executes it; until the value existed the driver had
    // to recover which was which from a family string. See
    // `.wiki/designs/design-struct.md`.
    let plan = prefill_plan(&q.data.t, head_dim);
    let kvc = rt_object(&q.data.t, "kv_cache", Some(kv.l));
    attn_at_planned(
        &q.data,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_bf16",
        cap_and_scale(soft_cap, sm_scale, head_dim),
        Some(plan),
        vec![q.indptr.id, kvc],
    )
}

/// `kernels::attn::attention_flashinfer_prefill` — plan-free prefill wrapper.
///
/// Plan-FREE means it walks its own schedule at fire time, and the swept
/// signature says from what: the KV view, the qo CSR (device and host
/// mirrors), the page-CSR host mirror and the raised plan CACHE are all
/// operands now, and the run carries `[logits_soft_cap, sm_scale,
/// head_dim, kv_num_heads, window_left]` — the request count is the CSR
/// operand's own row count.
pub fn attention_flashinfer_prefill_planless(
    q: &RaggedVal,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
    kv_num_heads: u32,
) -> Option<Val> {
    let inputs = vec![
        q.data.id,
        rt_object(&q.data.t, "kv_cache", Some(kv.l)),
        q.indptr.id,
        rt_object(&q.data.t, "fa2.prefill", None),
        rt_object(&q.data.t, "qo_indptr.host", None),
        rt_object(&q.data.t, "kv_page_indptr.host", None),
    ];
    let mut params = cap_and_scale(soft_cap, sm_scale, head_dim);
    params.extend([head_dim, kv_num_heads, window_left as u32]);
    let out = q.data.t.inner.borrow().inside_value_region();
    let shape = (!out).then(|| q.data.t.inner.borrow().value_shape(q.data.id));
    record_with_extents(
        &q.data.t,
        Some(kv.l),
        "attn::attention_flashinfer_prefill",
        vec![],
        kv_state(kv),
        params,
        vec![],
        inputs,
        shape.map(|s| (s, DType::BF16)),
    )
}

/// Class-dependent attention helper; arm order is the contract.
#[allow(clippy::too_many_arguments)]
pub fn attention_for(
    class: model_ir::trace::FireClass,
    q: &Val,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
    kv_num_heads: u32,
) -> Option<Val> {
    match class {
        model_ir::trace::FireClass::Decode => {
            attention_flashinfer_decode(q, kv, window_left, head_dim, soft_cap, sm_scale)
        }
        // The planless arm states no prep -- it plans from the host CSR when
        // the statement runs -- but `head_dim` is read here even so: the
        // routine takes an `sm_scale`, and a caller that states none gets
        // `1/sqrt(head_dim)`.
        _ => attention_flashinfer_prefill_planless(
            &crate::runtime::query_windows(q),
            kv,
            window_left,
            head_dim,
            soft_cap,
            sm_scale,
            kv_num_heads,
        ),
    }
}

/// [`attention_for`], asked for its LSE — the sink families' form.
#[allow(clippy::too_many_arguments)]
pub fn attention_for_lse(
    class: model_ir::trace::FireClass,
    q: &Val,
    kv: &Kv,
    q_heads: u32,
    head_dim: u32,
    kv_num_heads: u32,
    window_left: i32,
    soft_cap: f32,
    sm_scale: f32,
) -> (Val, Val) {
    match class {
        model_ir::trace::FireClass::Decode => attention_flashinfer_decode_lse(
            q,
            kv,
            q_heads,
            head_dim,
            window_left,
            soft_cap,
            sm_scale,
        ),
        // As `attention_for`: the planless prefill raises nothing.
        _ => attention_flashinfer_prefill_lse(
            &crate::runtime::query_windows(q),
            kv,
            q_heads,
            head_dim,
            kv_num_heads,
            window_left,
            soft_cap,
            sm_scale,
        ),
    }
}

/// FlashInfer decode with LSE: produces `(o, lse)`, LSE is last positional arg.
pub fn attention_flashinfer_decode_lse(
    q: &Val,
    kv: &Kv,
    q_heads: u32,
    head_dim: u32,
    window_left: i32,
    soft_cap: f32,
    sm_scale: f32,
) -> (Val, Val) {
    // The window picks the SCHEDULE: a sliding family's LSE decode (gpt-oss)
    // plans against its window, a whole-context one against the full plan.
    // The old form hardcoded full — the B5-2 sweep's flagged defect.
    let plan = decode_plan(&q.t, head_dim, window_left < 0);
    let kvc = rt_object(&q.t, "kv_cache", Some(kv.l));
    let shape = q.t.inner.borrow().value_shape(q.id);
    let ids = q.t.with(Some(kv.l), |b| {
        b.launch_with_params(
            "attn::dispatch_attention_flashinfer_decode_lse",
            vec![],
            kv_state(kv),
            // Declared UNREAD by the routine (the dispatch reads the plan),
            // but the run says what the statement means: THIS window, cap
            // and scale — not a hardcoded whole-context spelling.
            vec![window_left as u32, soft_cap.to_bits(), sm_scale.to_bits()],
            vec![q.id, plan, kvc],
            vec![
                (shape, DType::BF16),
                (Shape(vec![Dim::Tokens, Dim::Const(q_heads)]), DType::F32),
            ],
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::attn::attention_flashinfer_prefill_lse` — the planless prefill over a statement
/// that declares both of its results, and the prefill twin of
/// [`attention_flashinfer_decode_lse`]. Operands and run as
/// [`attention_flashinfer_prefill_planless`]'s.
#[allow(clippy::too_many_arguments)]
pub fn attention_flashinfer_prefill_lse(
    q: &RaggedVal,
    kv: &Kv,
    q_heads: u32,
    head_dim: u32,
    kv_num_heads: u32,
    window_left: i32,
    soft_cap: f32,
    sm_scale: f32,
) -> (Val, Val) {
    let inputs = vec![
        q.data.id,
        rt_object(&q.data.t, "kv_cache", Some(kv.l)),
        q.indptr.id,
        rt_object(&q.data.t, "fa2.prefill", None),
        rt_object(&q.data.t, "qo_indptr.host", None),
        rt_object(&q.data.t, "kv_page_indptr.host", None),
    ];
    let mut params = cap_and_scale(soft_cap, sm_scale, head_dim);
    params.extend([head_dim, kv_num_heads, window_left as u32]);
    let shape = q.data.t.inner.borrow().value_shape(q.data.id);
    let outs = record_many_with_extents(
        &q.data.t,
        Some(kv.l),
        "attn::attention_flashinfer_prefill_lse",
        vec![],
        kv_state(kv),
        params,
        vec![],
        inputs,
        vec![
            (shape, DType::BF16),
            (Shape(vec![Dim::Tokens, Dim::Const(q_heads)]), DType::F32),
        ],
    );
    let mut it = outs.into_iter();
    let o = it.next().expect("the attention states two outputs");
    let lse = it.next().expect("the attention states two outputs");
    (o, lse)
}

/// `kernels::attn::attention_naive_paged` — the fallback prefill for a head dim flashinfer's TC
/// prefill template rejects. The KV view and qo CSR are operands; the run is
/// `[head_dim, num_kv_heads, window_left, sm_scale, logits_soft_cap]` — the
/// request count is the CSR operand's own row count. The optional LSE
/// out stays undeclared: no caller of the naive fallback reads one.
#[allow(clippy::too_many_arguments)]
pub fn attention_naive_paged(
    q: &RaggedVal,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    num_kv_heads: u32,
    sm_scale: f32,
    soft_cap: f32,
) -> Option<Val> {
    let inputs = vec![
        q.data.id,
        rt_object(&q.data.t, "kv_cache", Some(kv.l)),
        q.indptr.id,
    ];
    let out = q.data.t.inner.borrow().inside_value_region();
    let shape = (!out).then(|| q.data.t.inner.borrow().value_shape(q.data.id));
    record_with_extents(
        &q.data.t,
        Some(kv.l),
        "attn::attention_naive_paged",
        vec![],
        kv_state(kv),
        vec![
            head_dim,
            num_kv_heads,
            window_left as u32,
            sm_scale.to_bits(),
            soft_cap.to_bits(),
        ],
        vec![],
        inputs,
        shape.map(|s| (s, DType::BF16)),
    )
}

/// `kernels::attn::write_kv_to_pages`: the page-derived append (position re-derived from the
/// page table). The `HasWriteDesc` guard's else-region.
///
/// # Why this one is hand-written where its sibling is generated
///
/// `attn::write_kv_to_pages` is a declaration standing for a CHOICE. The
/// name resolves at LOAD, through `Boot::route`, to `write_kv_to_pages_bf16`
/// or `write_kv_to_pages_quantised` from the KV dtype the checkpoint settles
/// — so a model text states the outer name and no text ever states either
/// body. A generated wrapper is named for a body, which is exactly what this
/// statement must not name.
///
/// # What it must still state
///
/// Being hand-written does not make it exempt from the column. The routine's
/// operands are derived from its `fn` signature and bound POSITIONALLY, so
/// this statement owes the same list the generated wrapper would write: the
/// two token planes it is handed, then the layer's KV view, the write origin,
/// the query CSR and the row-validity mask — and the two head numbers as
/// launch params. Stating `[k, v]` and no params, which is what stood here,
/// bound `kvc` off the third input slot of a two-input fire and refused
/// every prefill that took this leg with "the fire does not carry an input
/// operand". The refusal aborted the pass, so the epilogue that samples the
/// first token never ran, and every device-carried decode behind it read an
/// empty ring.
pub fn write_kv_to_pages(k: &Val, v: &Val, kv: &Kv, num_kv_heads: u32, head_dim: u32) {
    record_with_params(
        &kv.t,
        Some(kv.l),
        "attn::write_kv_to_pages",
        vec![],
        kv_state(kv),
        vec![num_kv_heads, head_dim],
        vec![
            k.id,
            v.id,
            rt_object(&kv.t, "kv_cache", Some(kv.l)),
            rt_requests(&kv.t, "first_token"),
            rt_requests(&kv.t, "qo_indptr"),
            rt_tokens(&kv.t, "row_valid"),
        ],
        None,
    );
}

/// THE GDN GEOMETRY THE STATEMENT CARRIES.
///
/// Seven numbers the CHECKPOINT fixes at load. They reached the CUDA kernels
/// as `keys::Gdn*` -- eleven keys resolving through one `f.cx.gdn()` -- while
/// `kernels-metal` and `kernels-vulkan` have spelled the same numbers
/// `Const<i32>` since §11.12 ruled that a constant belongs in the statement.
/// One fact, two spellings; this is the CUDA side catching up.
///
/// A struct and not seven arguments, for `kernels-metal`'s `GdnShape`'s
/// reason: they travel together, and a caller free to pass them separately is
/// a caller free to pass them inconsistently.
#[derive(Clone, Copy, Debug)]
pub struct GdnShape {
    /// Key heads (compact, pre-GQA-repeat).
    pub k_heads: u32,
    /// Value heads.
    pub v_heads: u32,
    /// Key head width.
    pub k_dim: u32,
    /// Value head width.
    pub v_dim: u32,
    /// Channels the causal convolution carries: `2*Hk*Dk + Hv*Dv`.
    pub conv_dim: u32,
    /// Convolution kernel width.
    pub conv_k: u32,
}

impl GdnShape {
    /// `[k_heads, v_heads, k_dim, v_dim]` -- the GQA recurrence forms' run.
    #[must_use]
    fn gqa(self) -> Vec<u32> {
        vec![self.k_heads, self.v_heads, self.k_dim, self.v_dim]
    }

    /// `[v_heads, k_dim, v_dim]` -- the forms that index the compact K layout
    /// directly and never need the key head count.
    #[must_use]
    fn compact(self) -> Vec<u32> {
        vec![self.v_heads, self.k_dim, self.v_dim]
    }
}

/// `kernels::ssm::recurrent_gated_delta_step_batched[_gqa][_state_bf16]`: the one-token decode
/// recurrence step against the layer's per-request recurrent state. `gqa` states the
/// compact-K_h-indexing GQA variant (value heads != key heads); `state_bf16` the store dtype.
/// Output = the semantic [`gated_delta`](crate::gated_delta)'s: the core keeps v's `[Tokens,
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_batched(
    q: &Val,
    k: &Val,
    v: &Val,
    g: &Val,
    beta: &Val,
    rs: &Rs,
    gqa: bool,
    state_bf16: bool,
    geom: GdnShape,
) -> Val {
    let kernel = match (gqa, state_bf16) {
        (true, true) => "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
        (true, false) => "ssm::recurrent_gated_delta_step_batched_gqa",
        (false, true) => "ssm::recurrent_gated_delta_step_batched_state_bf16",
        (false, false) => "ssm::recurrent_gated_delta_step_batched",
    };
    // THE RUN THE ROUTINE'S MARKS DECLARE, and the two differ: the GQA
    // forms take the key head count and the compact ones index that
    // layout directly and never need it. Both end on `r`, the request
    // count, which is the fire's and spliced by the lowering.
    let mut params = if gqa { geom.gqa() } else { geom.compact() };
    let r_at = params.len() as u8;
    params.push(0);
    let rsv = rt_object(&q.t, "recurrent_state", Some(rs.l));
    let ids = q.t.with(Some(rs.l), |b| {
        let shape = b.value_shape(v.id);
        b.launch_with_extents(
            kernel,
            vec![],
            rs_state(rs),
            params,
            vec![requests_extent(r_at)],
            vec![q.id, k.id, v.id, g.id, beta.id, rsv],
            vec![(shape, DType::F32)],
        )
    });
    Val {
        t: q.t.clone(),
        id: ids[0],
        layer: Some(rs.l),
    }
}

/// Prefill recurrence guard launch: output-less; guard owns the value.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_warp_tiled(
    q: &Val,
    k: &Val,
    v: &Val,
    g: &Val,
    beta: &Val,
    rs: &Rs,
    state_bf16: bool,
    geom: GdnShape,
    write_state: bool,
) {
    // One arm per state dtype; no exported non-GQA duplicate.
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel, geom.gqa(), write_state);
}

/// Cached prefill recurrence; indexes repeated `[Vh]`, so the guard materializes heads first.
/// Guard-region launch, output-less like the warp-tiled form.
pub fn gdn_prefill_cached(
    q: &Val,
    k: &Val,
    v: &Val,
    g: &Val,
    beta: &Val,
    rs: &Rs,
    state_bf16: bool,
    geom: GdnShape,
    write_state: bool,
) {
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched_cached"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel, geom.compact(), write_state);
}

/// `kernels::ssm::chunk_gated_delta_prefill_batched[_state_bf16]`: the batched GQA-aware FLA
/// prefill recurrence — the fallback arm (it indexes the compact K_h layout directly, so no
/// repeats). Guard-region launch, output-less like the warp-tiled form.
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_fla(
    q: &Val,
    k: &Val,
    v: &Val,
    g: &Val,
    beta: &Val,
    rs: &Rs,
    state_bf16: bool,
    geom: GdnShape,
    write_state: bool,
) {
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel, geom.gqa(), write_state);
}

#[allow(clippy::too_many_arguments)]
fn gdn_prefill(
    q: &Val,
    k: &Val,
    v: &Val,
    g: &Val,
    beta: &Val,
    rs: &Rs,
    kernel: &str,
    mut params: Vec<u32>,
    write_state: bool,
) {
    // The geometry run ends on `[write_state]`, the persist flag the
    // caller's — the request count is the CSR operand's own row count. The
    // recurrent view and the qo CSR are the operands after `beta`.
    params.push(u32::from(write_state));
    let rsv = rt_object(&q.t, "recurrent_state", Some(rs.l));
    let qo_indptr = rt_requests(&q.t, "qo_indptr");
    record_with_extents(
        &q.t,
        Some(rs.l),
        kernel,
        vec![],
        rs_state(rs),
        params,
        vec![],
        vec![q.id, k.id, v.id, g.id, beta.id, rsv, qo_indptr],
        None,
    );
}

/// `verify_stash_load`: pseudo-symbol; no inputs, three bf16 outputs `[qkv, a, b]`.
pub fn verify_stash_load(t: &Trace, rs: &Rs, conv_dim: u32, value_heads: u32) -> (Val, Val, Val) {
    let ids = t.with(Some(rs.l), |b| {
        b.launch(
            "ssm::verify_stash_load",
            vec![],
            rs_state(rs),
            vec![],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(conv_dim)]), DType::BF16),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(value_heads)]),
                    DType::BF16,
                ),
                (
                    Shape(vec![Dim::Tokens, Dim::Const(value_heads)]),
                    DType::BF16,
                ),
            ],
        )
    });
    let mk = |id| Val {
        t: t.clone(),
        id,
        layer: Some(rs.l),
    };
    (mk(ids[0]), mk(ids[1]), mk(ids[2]))
}

/// `verify_stash_store`: pseudo-symbol; output-less per-request stash write.
pub fn verify_stash_store(qkv: &Val, a: &Val, b: &Val, rs: &Rs) {
    record(
        &qkv.t,
        Some(rs.l),
        "ssm::verify_stash_store",
        vec![],
        rs_state(rs),
        vec![qkv.id, a.id, b.id],
        None,
    );
}

/// `kernels::attn::dispatch_attention_flashinfer_decode_capture`: the score-capturing decode
/// dispatch (the OnAttn sideband's producer; its contract includes the capture publish against
/// the possibly page-mask-compacted CSR). Region launch of the WantsAttnScore guard —
/// output-less; the guard owns the attention output.
pub fn attention_flashinfer_decode_capture(
    q: &Val,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
) -> Option<Val> {
    let plan = decode_plan(&q.t, head_dim, window_left == -1);
    let kvc = rt_object(&q.t, "kv_cache", Some(kv.l));
    let score = rt_object(&q.t, "attn.score", None);
    attn_at_planned(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_decode_capture",
        // THIS one declares the window as a `Const` and the other decode does
        // not, which is the whole reason the run is spelled per routine.
        {
            let mut p = vec![window_left as u32];
            p.extend(cap_and_scale(soft_cap, sm_scale, head_dim));
            p
        },
        Some(plan),
        vec![kvc, score],
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16` — the prefill
/// counterpart, same guard-region contract.
#[allow(clippy::too_many_arguments)]
pub fn attention_flashinfer_prefill_capture(
    q: &RaggedVal,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
) -> Option<Val> {
    // THE WINDOW IS A FACT NOW, NOT A SCALAR. Every routine below asks
    // for it through `keys::WindowLeft`; the statement used to carry it
    // and carried it into the slot a soft cap is read from. It stays on
    // the signature because a caller states the deployment's window here
    // and the guard predicates above still read the same number.
    let _ = window_left;
    let plan = prefill_plan(&q.data.t, head_dim);
    let kvc = rt_object(&q.data.t, "kv_cache", Some(kv.l));
    // The score observation is ONE driver-owned view (CSR + window): the
    // window is boot policy and the CSR is the fire's, so neither is the
    // statement's to state — the old form carried a Const zero and minted
    // a stream no driver answered, which was the capture path dead twice.
    let score = rt_object(&q.data.t, "attn.score", None);
    attn_at_planned(
        &q.data,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        cap_and_scale(soft_cap, sm_scale, head_dim),
        Some(plan),
        vec![kvc, q.indptr.id, score],
    )
}

/// Output-less peel-prefix QK-norm/rope/KV write; peel owns q rows.
/// Operands and run as [`super::qkv_decode_qk_norm_rope_write_kv`]'s — and
/// as there, the table mark is not nullable: a `None` leaves the statement
/// one operand short and `check_plan` refuses the plan.
pub fn qkv_decode_qk_norm_rope_write_kv_region(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    table: Option<&Val>,
    num_kv_heads: u32,
    theta: f32,
) {
    let mut inputs = vec![packed.id];
    if let Some(t) = table {
        inputs.push(t.id);
    }
    inputs.push(rt_object(&packed.t, "kv_cache", Some(kv.l)));
    inputs.push(rt_tokens(&packed.t, "positions"));
    inputs.push(rt_tokens(&packed.t, "row_valid"));
    let head_dim = q_norm
        .per_head
        .expect("a per-head q norm carries its head dim");
    record_with_params(
        &packed.t,
        Some(kv.l),
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
        vec![
            num_kv_heads,
            head_dim,
            theta.to_bits(),
            q_norm.eps.to_bits(),
        ],
        inputs,
        None,
    );
}

/// `dispatch_attention_flashinfer_prefill_custom`: custom-mask prefill; mask data is runtime args.
pub fn attention_flashinfer_prefill_custom(
    q: &RaggedVal,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
) -> Option<Val> {
    // THE WINDOW IS A FACT NOW, NOT A SCALAR. Every routine below asks
    // for it through `keys::WindowLeft`; the statement used to carry it
    // and carried it into the slot a soft cap is read from. It stays on
    // the signature because a caller states the deployment's window here
    // and the guard predicates above still read the same number.
    let _ = window_left;
    let plan = prefill_plan(&q.data.t, head_dim);
    let maskv = rt_object(&q.data.t, "attention_mask", None);
    let kvc = rt_object(&q.data.t, "kv_cache", Some(kv.l));
    attn_at_planned(
        &q.data,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_custom",
        cap_and_scale(soft_cap, sm_scale, head_dim),
        Some(plan),
        vec![maskv, kvc, q.indptr.id],
    )
}

/// `gemm::lora_qkv_correction`: in-place q/v adapter correction inside `HasLora`.
pub fn lora_qkv_correction(q: &Val, v: &Val, l: u32) {
    record(
        &q.t,
        Some(l),
        "gemm::lora_qkv_correction",
        vec![],
        None,
        vec![q.id, v.id],
        None,
    );
}

/// Attention dispatch shape: q input, cache state, output unless a guard/peel
/// owns it. `params` is the run the named routine's `Const` marks declare, in
/// their order.
///
/// IT USED TO BE `vec![window_left as u32]`, AND THAT IS NOT WHAT ANY OF THESE
/// ROUTINES READ THERE. Every flashinfer dispatch declares `logits_soft_cap:
/// Const<f32>` at slot 0 and asks for the window through `keys::WindowLeft`
/// instead -- so the statement put `-1` where a float's BITS are read, which
/// is `NaN`, and a soft cap of NaN takes every logit with it. The window
/// itself was stated at a slot nothing reads.
///
/// It could not fault and it could not be seen: one `u32` is as wide as
/// another, the count matched for the routines that take exactly one scalar,
/// and `check_plan`'s params rule only ever counted them.

/// [`attn_at`], with the raise the statement names.
///
/// `plan` is placed as input 1, after `q`. The order is the routine's: `q`
/// keeps `In(0)`, which half this family's comments and every one of its
/// tests state, and the raise takes the slot after it. A launcher that does
/// not yet take the operand passes `None` and places what it always did --
/// which is what keeps the five siblings firing while this one moves.
fn attn_at_planned(
    q: &Val,
    kv: &Kv,
    kernel: &str,
    params: Vec<u32>,
    plan: Option<model_ir::trace::ValueId>,
    tail: Vec<model_ir::trace::ValueId>,
) -> Option<Val> {
    let out = q.t.inner.borrow().inside_value_region();
    let shape = (!out).then(|| q.t.inner.borrow().value_shape(q.id));
    let mut inputs = vec![q.id];
    inputs.extend(plan);
    inputs.extend(tail);
    record_with_params(
        &q.t,
        Some(kv.l),
        kernel,
        vec![],
        kv_state(kv),
        params,
        inputs,
        shape.map(|s| (s, DType::BF16)),
    )
}

/// `[logits_soft_cap, sm_scale]`, the run five of the flashinfer dispatches
/// declare and nothing else.
///
/// `sm_scale` is derived from the head dim when the caller states none, which
/// is llama's `1/sqrt(d)` rule and NOT attention's -- gemma-3 publishes
/// `query_pre_attn_scalar` and gemma-4 publishes 1.0, because its per-head
/// q/k norms have already divided by what this would divide by again. A
/// statement that derives it cannot serve a family that states it, so a
/// positive `sm_scale` always wins.
fn cap_and_scale(soft_cap: f32, sm_scale: f32, head_dim: u32) -> Vec<u32> {
    let scale = if sm_scale > 0.0 {
        sm_scale
    } else {
        1.0f32 / (head_dim as f32).sqrt()
    };
    vec![soft_cap.to_bits(), scale.to_bits()]
}
