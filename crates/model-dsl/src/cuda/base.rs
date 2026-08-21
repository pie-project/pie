//! CUDA generation statements: projections, attention, rope, KV writes,
//! recurrent updates, and the surrounding launches.

use super::*;


builder! {
    /// `kernels::mlp::sigmoid_dot_scalar_gate_add`: the shared expert's landing with its
    /// gate logit folded in — one launch that dots `norm_x` with the `[1, H]` gate row,
    /// sigmoids the scalar, and accumulates `shared` into the stream.
    pub fn sigmoid_dot_scalar_gate_add(
        x: &Val,
        gate: &MatW,
        shared: &Val,
        base: &Val,
        hidden: u32,
    ) -> Val {
        symbol: "mlp::sigmoid_dot_scalar_gate_add",
        on: x,
        weights: [gate.name],
        layer: gate.layer,
        inputs: [x, base, shared],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the shared-expert landing produces its value",
    }


    /// `kernels::mlp::chunked_swiglu` over routed `N * k` rows; keeps expert dim.
    pub fn swiglu_routed(x: &Val, top_k: u32, intermediate: u32) -> Val {
        symbol: "mlp::chunked_swiglu",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(top_k), Dim::Const(intermediate)] as BF16,
        made: "the routed activation produces its value",
    }
}

/// `kernels::moe::token_batched_weighted_sum`, or the `..._add_bf16` form when the
/// residual folds into the same launch.
pub fn weighted_sum(weights: &Val, x: &Val, hidden: u32, residual: Option<&Val>) -> Val {
    let mut inputs = vec![x.id, weights.id];
    if let Some(r) = residual {
        inputs.push(r.id);
    }
    record(
        &weights.t,
        weights.layer,
        if residual.is_some() {
            "moe::token_batched_weighted_sum_add"
        } else {
            "moe::token_batched_weighted_sum"
        },
        vec![],
        None,
        inputs,
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the combine produces its value")
}

builder! {
    /// The MLP activation, stating which of the two swiglu kernels runs.
    pub fn swiglu(x: &Val, intermediate: u32) -> Val {
        symbol: "mlp::chunked_swiglu",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `kernels::mlp::chunked_swiglu_into` over the ALIGNED leg's block-major
    /// staging: [`swiglu`](crate::cuda::swiglu)'s shape, plus the destination the
    /// pointer build named.
    ///
    /// A DIFFERENT SYMBOL from [`swiglu`](crate::cuda::swiglu), and the second
    /// operand is why: this states its destination, so the routine's result
    /// must BE it (`InOut`), while the dense form places no destination at all
    /// and its result is the arena's to put anywhere.
    pub fn swiglu_aligned(x: &Val, stage: &Val, aligned: Dim, intermediate: u32) -> Val {
        symbol: "mlp::chunked_swiglu_into",
        on: x,
        inputs: [x, stage],
        out: [aligned, Dim::Const(intermediate)] as BF16,
        made: "the aligned activation produces its value",
    }


    /// `kernels::mlp::swiglu` in its PAIR form: two operands, the gate and the up
    /// projection, into one activation.
    pub fn swiglu_pair(gate: &Val, up: &Val, intermediate: u32) -> Val {
        symbol: "mlp::swiglu",
        on: gate,
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }
}

/// `kernels::rope::qk_rmsnorm_rope_bf16`: fused per-head q/k norm + Standard rope.
///
/// `[head_dim, theta, eps]` is the run the routine's three `Const` marks
/// claim; the positions stream is minted by name. The head dim and epsilon
/// ride the norm handle — `theta` is the deployment's and the caller states it.
pub fn qk_rmsnorm_rope(
    q: &Val,
    k: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    theta: f32,
) -> (Val, Val) {
    let head_dim = q_norm
        .per_head
        .expect("a per-head q norm carries its head dim");
    let ids = q.t.with(q.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch_with_params(
            "rope::qk_rmsnorm_rope_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            None,
            vec![head_dim, theta.to_bits(), q_norm.eps.to_bits()],
            vec![q.id, k.id, positions],
            vec![(q_sh, DType::BF16), (k_sh, DType::BF16)],
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::attn::attention_xqa_decode_bf16_prepared` (whose contract includes the fire-wide
/// XQA prepare — and which is therefore declared `whole`; see [`model_ir::kernels`]).
///
/// The swept signature: `q` and the KV view as operands, and a seven-scalar
/// run `[num_q_heads, num_kv_heads, head_dim, sm_scale, float_bytes,
/// int_bytes, num_requests]` — the workspace byte counts are the statement's
/// now (the carve is launch-local `ctx.scratch`), and the request count is a
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
    float_bytes: u32,
    int_bytes: u32,
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
        vec![
            num_q_heads,
            num_kv_heads,
            head_dim,
            sm_scale.to_bits(),
            float_bytes,
            int_bytes,
            0,
        ],
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
        b.push_prep(model_ir::trace::PrepKind::DecodeAttention { head_dim, full_attention })
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
        model_ir::trace::FireClass::Decode => {
            attention_flashinfer_decode_lse(q, kv, q_heads, head_dim, window_left, soft_cap, sm_scale)
        }
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

/// `kernels::rope::rope_yarn_original_bf16`: the YaRN-paper rope — a dim-index ramp between
/// interpolated and extrapolated frequencies, plus an `attention_factor` magnitude scale.
///
/// The eight YaRN numbers are the checkpoint's and ride the params run in
/// the order the routine's `Const` marks claim: `[head_dim, theta, factor,
/// beta_fast, beta_slow, attention_factor, original_max_position,
/// interleaved]`. The positions stream is minted by name.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_original(
    q: &Val,
    k: &Val,
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
) -> (Val, Val) {
    let (q_sh, k_sh) = {
        let b = q.t.inner.borrow();
        (b.value_shape(q.id), b.value_shape(k.id))
    };
    let ids = q.t.with(q.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        b.launch_with_params(
            "rope::rope_yarn_original_bf16",
            vec![],
            None,
            vec![
                head_dim,
                theta.to_bits(),
                factor.to_bits(),
                beta_fast.to_bits(),
                beta_slow.to_bits(),
                attention_factor.to_bits(),
                original_max_position,
                u32::from(interleaved),
            ],
            vec![q.id, k.id, positions],
            vec![(q_sh, DType::BF16), (k_sh, DType::BF16)],
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::rope::rope_bf16`: the full rotation, named.
/// The semantic [`super::rope`] carries a `RopeKind` and a rotary width, and the driver's arm
/// asked whether the width was zero to decide between two launchers.
pub fn rope(
    q: &Val,
    k: &Val,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> (Val, Val) {
    // `[num_q_heads, num_kv_heads, head_dim, theta, interleaved]`, which is
    // the run the routine's five `Const` marks claim, in order — the same
    // run the tier-1 `dsl::rope` states.
    rope_launch(
        q,
        k,
        "rope::rope_bf16",
        vec![
            q_heads,
            kv_heads,
            head_dim,
            theta.to_bits(),
            u32::from(interleaved),
        ],
    )
}

/// Partial rope: `[rotary_dim, head_dim, theta]` rides `params`; q/k are
/// full-width operands.
pub fn rope_partial(q: &Val, k: &Val, rotary_dim: u32, head_dim: u32, theta: f32) -> (Val, Val) {
    assert!(
        rotary_dim > 0,
        "a partial rotation with no channels is the full one; state \
         `cuda::rope`"
    );
    rope_launch(
        q,
        k,
        "rope::rope_partial_bf16",
        vec![rotary_dim, head_dim, theta.to_bits()],
    )
}

/// Rope launch shape: q,k inputs, then the minted positions stream; q,k
/// outputs; both pairs alias in place.
fn rope_launch(q: &Val, k: &Val, symbol: &str, params: Vec<u32>) -> (Val, Val) {
    let (q_sh, k_sh) = {
        let b = q.t.inner.borrow();
        (b.value_shape(q.id), b.value_shape(k.id))
    };
    let ids = q.t.with(q.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        b.launch_with_params(
            symbol,
            vec![],
            None,
            params,
            vec![q.id, k.id, positions],
            vec![(q_sh, DType::BF16), (k_sh, DType::BF16)],
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

builder! {
    /// `kernels::gemm::act_x_wt_bias_bf16`: projection with bias in the epilogue.
    pub fn gemm_bias(x: &Val, w: &MatW, bias: &MatW) -> Val {
        symbol: "gemm::act_x_wt_bias_bf16",
        on: x,
        weights: [w.name, bias.name],
        layer: w.layer,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(w.width)] as BF16,
        made: "a biased projection produces its value",
    }
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

/// Attention sink rescale is in place; sink logit rides the weight slot and
/// `[num_q_heads, head_dim]` the params run.
pub fn attention_sink_rescale(
    o: &Val,
    lse: &Val,
    sinks: &MatW,
    num_q_heads: u32,
    head_dim: u32,
) -> Val {
    let shape = o.t.inner.borrow().value_shape(o.id);
    record_with_params(
        &o.t,
        sinks.layer,
        "attn::attention_sink_rescale",
        vec![sinks.name.clone()],
        None,
        vec![num_q_heads, head_dim],
        vec![o.id, lse.id],
        Some((shape, DType::BF16)),
    )
    .expect("the sink rescale produces its value")
}

/// `kernels::quant::bf16_to_fp16`: the activation cast the MXFP4 routed GEMVs want on their
/// input.
pub fn bf16_to_fp16(x: &Val) -> Val {
    let shape = x.t.inner.borrow().value_shape(x.id);
    record(
        &x.t,
        x.layer,
        "quant::bf16_to_fp16",
        vec![],
        None,
        vec![x.id],
        Some((shape, DType::F16)),
    )
    .expect("the cast produces its value")
}

/// MXFP4 fused gate/up: weight slot names per-expert pointer bank; the
/// resident expert-weight view is the third operand and `[glu_limit,
/// glu_alpha]` the params run. Returns `(gate, up)`.
pub fn mxfp4_moe_gate_up_decode(
    x: &Val,
    experts: &Val,
    bank: &MatW,
    top_k: u32,
    intermediate: u32,
    limit: f32,
    alpha: f32,
) -> (Val, Val) {
    let shape = || {
        (
            Shape(vec![
                Dim::Tokens,
                Dim::Const(top_k),
                Dim::Const(intermediate),
            ]),
            DType::BF16,
        )
    };
    let ew = rt_object(&x.t, "expert_weights", bank.layer);
    let ids = x.t.with(bank.layer, |b| {
        b.launch_with_params(
            "quant::mxfp4_moe_gate_up_decode_bf16",
            vec![bank.name.clone()],
            None,
            vec![limit.to_bits(), alpha.to_bits()],
            vec![experts.id, x.id, ew],
            vec![shape(), shape()],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: bank.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// `kernels::quant::mxfp4_moe_down_decode_bf16`: the routed down projection, the same bank
/// convention as [`mxfp4_moe_gate_up_decode`] — expert-weight view included.
pub fn mxfp4_moe_down_decode(
    x: &Val,
    experts: &Val,
    bank: &MatW,
    top_k: u32,
    hidden: u32,
) -> Val {
    let ew = rt_object(&x.t, "expert_weights", bank.layer);
    record(
        &x.t,
        bank.layer,
        "quant::mxfp4_moe_down_decode_bf16",
        vec![bank.name.clone()],
        None,
        vec![experts.id, x.id, ew],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the routed down projection produces its value")
}

builder! {
    /// GPT-OSS GLU over routed `[Tokens, k, intermediate]`; `limit` and the
    /// gate's `alpha` ride params, in the order the routine's two `Const<f32>`
    /// marks claim them.
    ///
    /// `alpha` was stated nowhere and the routine takes it, so the fire read a
    /// zero at that slot -- `x * sigmoid(0)` is `x/2`, an activation that is
    /// finite, varied and wrong, which is the failure mode this whole check
    /// exists to catch.
    pub fn gpt_oss_glu(
        gate: &Val,
        up: &Val,
        top_k: u32,
        intermediate: u32,
        limit: f32,
        alpha: f32,
    ) -> Val {
        symbol: "mlp::gpt_oss_glu",
        on: gate,
        params: [limit.to_bits(), alpha.to_bits()],
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(top_k), Dim::Const(intermediate)] as BF16,
        made: "the clamped GLU produces its value",
    }
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

/// `kernels::attn::write_kv_explicit_bf16`: the explicit-descriptor KV write (graph-replay
/// steering; N cells, one per query token). Stated inside the `HasWriteDesc` guard's
/// then-region. The KV view and row validity are operands; `[num_kv_heads,
/// head_dim]` rides the params run.
pub fn write_kv_explicit(k: &Val, v: &Val, kv: &Kv, num_kv_heads: u32, head_dim: u32) {
    let kvc = rt_object(&kv.t, "kv_cache", Some(kv.l));
    let row_valid = rt_tokens(&kv.t, "row_valid");
    record_with_params(
        &kv.t,
        Some(kv.l),
        "attn::write_kv_explicit_bf16",
        vec![],
        kv_state(kv),
        vec![num_kv_heads, head_dim],
        vec![k.id, v.id, kvc, row_valid],
        None,
    );
}

/// `kernels::attn::write_kv_to_pages`: the page-derived append (position re-derived from the
/// page table). The `HasWriteDesc` guard's else-region.
pub fn write_kv_to_pages(k: &Val, v: &Val, kv: &Kv) {
    record(
        &kv.t,
        Some(kv.l),
        "attn::write_kv_to_pages",
        vec![],
        kv_state(kv),
        vec![k.id, v.id],
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

    /// `[conv_dim, conv_k]` -- the two conv walks'.
    #[must_use]
    fn conv(self) -> Vec<u32> {
        vec![self.conv_dim, self.conv_k]
    }
}

/// `kernels::ssm::causal_conv1d_update_batched`: the slot-indirected decode conv update (+
/// fused SiLU) against the layer's per-request conv slab. Shape-preserving, like the semantic
/// [`causal_conv1d`](crate::causal_conv1d) it lowers.
pub fn gdn_conv_update_batched(x: &Val, w: &ConvW, rs: &Rs, shape: GdnShape) -> Val {
    gdn_conv(
        x,
        w,
        rs,
        "ssm::causal_conv1d_update_batched",
        shape.conv(),
        vec![],
        vec![],
    )
}

/// `kernels::ssm::causal_conv1d_prefill_batched`: the batched prefill conv walk (each
/// request walking its qo_indptr window and persisting the trailing K-window into the slab).
/// The run appends `[write_state]`, the persist flag the caller's — the
/// request count is the CSR operand's own row count.
pub fn gdn_conv_prefill_batched(
    x: &Val,
    w: &ConvW,
    rs: &Rs,
    shape: GdnShape,
    write_state: bool,
) -> Val {
    let mut params = shape.conv();
    params.push(u32::from(write_state));
    let qo_indptr = rt_requests(&x.t, "qo_indptr");
    gdn_conv(
        x,
        w,
        rs,
        "ssm::causal_conv1d_prefill_batched",
        params,
        vec![],
        vec![qo_indptr],
    )
}

fn gdn_conv(
    x: &Val,
    w: &ConvW,
    rs: &Rs,
    kernel: &str,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    tail_inputs: Vec<model_ir::trace::ValueId>,
) -> Val {
    // The recurrent view sits between `x` and the CSR in the swept
    // signature; the bias plane is a nullable second weight the statement
    // places only when the checkpoint has one.
    let rsv = rt_object(&x.t, "recurrent_state", Some(rs.l));
    let mut weights = vec![w.name.clone()];
    weights.extend(w.bias.clone());
    let mut inputs = vec![x.id, rsv];
    inputs.extend(tail_inputs);
    let ids = x.t.with(Some(w.layer), |b| {
        let shape = b.value_shape(x.id);
        b.launch_with_extents(
            kernel,
            weights,
            rs_state(rs),
            params,
            param_extents,
            inputs,
            vec![(shape, DType::BF16)],
        )
    });
    Val {
        t: x.t.clone(),
        id: ids[0],
        layer: Some(w.layer),
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

builder! {
    /// `repeat_interleave_heads_fp32`: produces `[Tokens, value_heads, key_dim]` f32 dataflow.
    pub fn repeat_interleave_heads(x: &Val, key_heads: u32, value_heads: u32, key_dim: u32) -> Val {
        symbol: "ssm::repeat_interleave_heads_fp32",
        on: x,
        // `[k_heads, v_heads, key_dim]`, the repeat's own contract: it walks
        // the compact K layout and writes each key head `v_heads / k_heads`
        // times, so it needs both counts and the width. `key_dim` was already
        // the caller's -- it states the output shape below with it.
        params: [key_heads, value_heads, key_dim],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(value_heads), Dim::Const(key_dim)] as F32,
        made: "the head repeat produces its value",
    }
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
    let score_indptr = rt_requests(&q.t, "attn.score_indptr");
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
        vec![kvc, score_indptr],
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
    score_window: u32,
) -> Option<Val> {
    // THE WINDOW IS A FACT NOW, NOT A SCALAR. Every routine below asks
    // for it through `keys::WindowLeft`; the statement used to carry it
    // and carried it into the slot a soft cap is read from. It stays on
    // the signature because a caller states the deployment's window here
    // and the guard predicates above still read the same number.
    let _ = window_left;
    let plan = prefill_plan(&q.data.t, head_dim);
    let kvc = rt_object(&q.data.t, "kv_cache", Some(kv.l));
    // The score CSR stays a loose mint: its data half (`score_out`) is
    // driver-owned, not a statement value, so there is no Val to pair.
    let score_indptr = rt_requests(&q.data.t, "attn.score_indptr");
    attn_at_planned(
        &q.data,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        {
            let mut p = cap_and_scale(soft_cap, sm_scale, head_dim);
            p.push(score_window);
            p
        },
        Some(plan),
        vec![kvc, q.indptr.id, score_indptr],
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

/// KV-cache dequant staging before a prefill-shaped dispatch; its own statement.
/// The KV view is the one operand and `[num_kv_heads, head_dim]` the run.
pub fn dequant_only(kv: &Kv, num_kv_heads: u32, head_dim: u32) {
    let kvc = rt_object(&kv.t, "kv_cache", Some(kv.l));
    record_with_params(
        &kv.t,
        Some(kv.l),
        "attn::dequant_kv_cache_layer_to_bf16_active",
        vec![],
        kv_state(kv),
        vec![num_kv_heads, head_dim],
        vec![kvc],
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
