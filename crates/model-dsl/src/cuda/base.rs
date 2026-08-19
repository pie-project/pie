//! CUDA generation statements: projections, attention, rope, KV writes,
//! recurrent updates, and the surrounding launches.

use super::*;


builder! {
    /// `kernels::mlp::sigmoid_dot_scalar_gate_add_bf16`: the shared expert's landing with its
    /// gate logit folded in — one launch that dots `norm_x` with the `[1, H]` gate row,
    /// sigmoids the scalar, and accumulates `shared` into the stream.
    pub fn sigmoid_dot_scalar_gate_add(
        x: &Val,
        gate: &MatW,
        shared: &Val,
        base: &Val,
        hidden: u32,
    ) -> Val {
        symbol: "mlp::sigmoid_dot_scalar_gate_add_bf16",
        on: x,
        weights: [gate.name],
        layer: gate.layer,
        inputs: [x, base, shared],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the shared-expert landing produces its value",
    }


    /// `kernels::mlp::chunked_swiglu_bf16` over routed `N * k` rows; keeps expert dim.
    pub fn swiglu_routed(x: &Val, top_k: u32, intermediate: u32) -> Val {
        symbol: "mlp::chunked_swiglu_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(top_k), Dim::Const(intermediate)] as BF16,
        made: "the routed activation produces its value",
    }
}

/// `kernels::moe::token_batched_weighted_sum_bf16`, or the `..._add_bf16` form when the
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
            "moe::token_batched_weighted_sum_add_bf16"
        } else {
            "moe::token_batched_weighted_sum_bf16"
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
        symbol: "mlp::chunked_swiglu_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }


    /// `kernels::mlp::chunked_swiglu_into_bf16` over the ALIGNED leg's block-major
    /// staging: [`swiglu`](crate::cuda::swiglu)'s shape, plus the destination the
    /// pointer build named.
    ///
    /// A DIFFERENT SYMBOL from [`swiglu`](crate::cuda::swiglu), and the second
    /// operand is why: this states its destination, so the routine's result
    /// must BE it (`InOut`), while the dense form places no destination at all
    /// and its result is the arena's to put anywhere.
    pub fn swiglu_aligned(x: &Val, stage: &Val, aligned: Dim, intermediate: u32) -> Val {
        symbol: "mlp::chunked_swiglu_into_bf16",
        on: x,
        inputs: [x, stage],
        out: [aligned, Dim::Const(intermediate)] as BF16,
        made: "the aligned activation produces its value",
    }


    /// `kernels::mlp::swiglu_bf16` in its PAIR form: two operands, the gate and the up
    /// projection, into one activation.
    pub fn swiglu_pair(gate: &Val, up: &Val, intermediate: u32) -> Val {
        symbol: "mlp::swiglu_bf16",
        on: gate,
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(intermediate)] as BF16,
        made: "the activation produces its value",
    }
}

/// `kernels::rope::qk_rmsnorm_rope_bf16`: fused per-head q/k norm + Standard rope.
pub fn qk_rmsnorm_rope(q: &Val, k: &Val, q_norm: &NormW, k_norm: &NormW) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch(
            "rope::qk_rmsnorm_rope_bf16",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            None,
            vec![q.id, k.id],
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
pub fn attention_xqa_decode(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    // THE WINDOW IS A FACT NOW, NOT A SCALAR. Every routine below asks
    // for it through `keys::WindowLeft`; the statement used to carry it
    // and carried it into the slot a soft cap is read from. It stays on
    // the signature because a caller states the deployment's window here
    // and the guard predicates above still read the same number.
    let _ = window_left;
    attn_at(
        q,
        kv,
        "attn::attention_xqa_decode_bf16_prepared",
        // NOTHING, AND THAT IS THE HONEST STATE. This routine declares the
        // KV POOL's geometry -- page size, scheme byte, storage dtype,
        // block size -- and a trace text holds none of it: the pool is the
        // allocator's and the text never sees a deployment. It stated
        // `[window_left]` before, which put one number at the page-size slot
        // and left the rest zero; an empty run at least does not claim a
        // page size of -1 -- and the routine asks the fire for the pool's
        // numbers again, which is where they were before the marks and where
        // `driver-cuda` has always answered them.
        vec![],
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
pub fn decode_plan(t: &Trace, head_dim: u32, full_attention: bool) {
    t.with(None, |b| {
        b.push_prep(model_ir::trace::PrepKind::DecodeAttention { head_dim, full_attention });
    });
}

/// [`decode_plan`]'s prefill twin, stated by the three PLANNED prefill helpers.
///
/// The planless pair state nothing: they walk their own schedule from the host
/// CSR mirrors when the statement runs, so there is none to raise.
pub fn prefill_plan(t: &Trace, head_dim: u32) {
    t.with(None, |b| {
        b.push_prep(model_ir::trace::PrepKind::PrefillAttention { head_dim });
    });
}

/// `kernels::attn::dispatch_attention_flashinfer_decode` against the decode plan its contract
/// obligates.
pub fn attention_flashinfer_decode(
    q: &Val,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
) -> Option<Val> {
    decode_plan(&q.t, head_dim, window_left == -1);
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_decode",
        // The plain decode declares no `Const` at all -- every number it uses
        // is a fact it asks for -- so its statement carries none.
        vec![],
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_prefill_bf16` — the dispatch alone.
pub fn attention_flashinfer_prefill(
    q: &Val,
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
    prefill_plan(&q.t, head_dim);
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_bf16",
        cap_and_scale(soft_cap, sm_scale, head_dim),
    )
}

/// `kernels::attn::attention_flashinfer_prefill` — plan-free prefill wrapper.
pub fn attention_flashinfer_prefill_planless(
    q: &Val,
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
    attn_at(q, kv, "attn::attention_flashinfer_prefill",
        cap_and_scale(soft_cap, sm_scale, head_dim),
    )
}

/// Class-dependent attention helper; arm order is the contract.
pub fn attention_for(
    class: model_ir::trace::FireClass,
    q: &Val,
    kv: &Kv,
    window_left: i32,
    head_dim: u32,
    soft_cap: f32,
    sm_scale: f32,
) -> Option<Val> {
    match class {
        model_ir::trace::FireClass::Decode => {
            attention_flashinfer_decode(q, kv, window_left, head_dim)
        }
        // The planless arm states no prep -- it plans from the host CSR when
        // the statement runs -- but `head_dim` is read here even so: the
        // routine takes an `sm_scale`, and a caller that states none gets
        // `1/sqrt(head_dim)`.
        _ => attention_flashinfer_prefill_planless(
            q,
            kv,
            window_left,
            head_dim,
            soft_cap,
            sm_scale,
        ),
    }
}

/// [`attention_for`], asked for its LSE — the sink families' form.
pub fn attention_for_lse(
    class: model_ir::trace::FireClass,
    q: &Val,
    kv: &Kv,
    q_heads: u32,
    head_dim: u32,
) -> (Val, Val) {
    match class {
        model_ir::trace::FireClass::Decode => {
            attention_flashinfer_decode_lse(q, kv, q_heads, head_dim)
        }
        // As `attention_for`: the planless prefill raises nothing.
        _ => attention_flashinfer_prefill_lse(q, kv, q_heads),
    }
}

/// FlashInfer decode with LSE: produces `(o, lse)`, LSE is last positional arg.
pub fn attention_flashinfer_decode_lse(
    q: &Val,
    kv: &Kv,
    q_heads: u32,
    head_dim: u32,
) -> (Val, Val) {
    // No window here: the `_lse` decode is stated by families that attend the
    // whole context, and `attn_plan` picks the full plan on an unbounded one.
    decode_plan(&q.t, head_dim, true);
    let shape = q.t.inner.borrow().value_shape(q.id);
    let ids = q.t.with(Some(kv.l), |b| {
        b.launch(
            "attn::dispatch_attention_flashinfer_decode_lse",
            vec![],
            kv_state(kv),
            vec![q.id],
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
pub fn rope_yarn_original(q: &Val, k: &Val) -> (Val, Val) {
    let (q_sh, k_sh) = {
        let b = q.t.inner.borrow();
        (b.value_shape(q.id), b.value_shape(k.id))
    };
    let ids = q.t.with(q.layer, |b| {
        b.launch(
            "rope::rope_yarn_original_bf16",
            vec![],
            None,
            vec![q.id, k.id],
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
pub fn rope(q: &Val, k: &Val, q_heads: u32, kv_heads: u32, head_dim: u32) -> (Val, Val) {
    // `[num_q_heads, num_kv_heads, head_dim]`, which is the run the routine's
    // three `Const` marks claim, in order. It was `vec![0]` -- one placeholder
    // where three extents are read -- so every rotation counted zero heads.
    // The theta and the interleave flag are NOT here: nothing on `Deployment`
    // states the flag, so the body asks for both.
    rope_launch(q, k, "rope::rope_bf16", vec![q_heads, kv_heads, head_dim])
}

/// Partial rope: `rotary_dim` rides `params`; q/k are full-width operands.
pub fn rope_partial(q: &Val, k: &Val, rotary_dim: u32) -> (Val, Val) {
    assert!(
        rotary_dim > 0,
        "a partial rotation with no channels is the full one; state \
         `cuda::rope`"
    );
    rope_launch(q, k, "rope::rope_partial_bf16", vec![rotary_dim])
}

/// Rope launch shape: q,k inputs; q,k outputs; both pairs alias in place.
fn rope_launch(q: &Val, k: &Val, symbol: &str, params: Vec<u32>) -> (Val, Val) {
    let (q_sh, k_sh) = {
        let b = q.t.inner.borrow();
        (b.value_shape(q.id), b.value_shape(k.id))
    };
    let ids = q.t.with(q.layer, |b| {
        b.launch_with_params(
            symbol,
            vec![],
            None,
            params,
            vec![q.id, k.id],
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
/// [`attention_flashinfer_decode_lse`].
pub fn attention_flashinfer_prefill_lse(q: &Val, kv: &Kv, q_heads: u32) -> (Val, Val) {
    let shape = q.t.inner.borrow().value_shape(q.id);
    let ids = q.t.with(Some(kv.l), |b| {
        b.launch(
            "attn::attention_flashinfer_prefill_lse",
            vec![],
            kv_state(kv),
            vec![q.id],
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

/// Attention sink rescale is in place; sink logit rides the weight slot.
pub fn attention_sink_rescale(o: &Val, lse: &Val, sinks: &MatW) -> Val {
    let shape = o.t.inner.borrow().value_shape(o.id);
    record(
        &o.t,
        sinks.layer,
        "attn::attention_sink_rescale_bf16",
        vec![sinks.name.clone()],
        None,
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

/// MXFP4 fused gate/up: weight slot names per-expert pointer bank; returns `(gate, up)`.
pub fn mxfp4_moe_gate_up_decode(
    x: &Val,
    experts: &Val,
    bank: &MatW,
    top_k: u32,
    intermediate: u32,
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
    let ids = x.t.with(bank.layer, |b| {
        b.launch(
            "quant::mxfp4_moe_gate_up_decode_bf16",
            vec![bank.name.clone()],
            None,
            vec![experts.id, x.id],
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

builder! {
    /// `kernels::quant::mxfp4_moe_down_decode_bf16`: the routed down projection, the same bank
    /// convention as [`mxfp4_moe_gate_up_decode`].
    pub fn mxfp4_moe_down_decode(
        x: &Val,
        experts: &Val,
        bank: &MatW,
        top_k: u32,
        hidden: u32,
    ) -> Val {
        symbol: "quant::mxfp4_moe_down_decode_bf16",
        on: x,
        weights: [bank.name],
        layer: bank.layer,
        inputs: [experts, x],
        out: [Dim::Tokens, Dim::Const(top_k), Dim::Const(hidden)] as BF16,
        made: "the routed down projection produces its value",
    }


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
        symbol: "mlp::gpt_oss_glu_bf16",
        on: gate,
        params: [limit.to_bits(), alpha.to_bits()],
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(top_k), Dim::Const(intermediate)] as BF16,
        made: "the clamped GLU produces its value",
    }
}

/// `kernels::attn::attention_naive_paged` — the fallback prefill for a head dim flashinfer's TC
/// prefill template rejects.
pub fn attention_naive_paged(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    let _ = window_left;
    attn_at(
        q,
        kv,
        "attn::attention_naive_paged",
        // NOTHING, AND THAT IS THE HONEST STATE. This routine declares the
        // KV POOL's geometry -- page size, scheme byte, storage dtype,
        // block size -- and a trace text holds none of it: the pool is the
        // allocator's and the text never sees a deployment. It stated
        // `[window_left]` before, which put one number at the page-size slot
        // and left the rest zero; an empty run at least does not claim a
        // page size of -1 -- and the routine asks the fire for the pool's
        // numbers again, which is where they were before the marks and where
        // `driver-cuda` has always answered them.
        vec![],
    )
}

/// `kernels::attn::write_kv_explicit_bf16`: the explicit-descriptor KV write (graph-replay
/// steering; N cells, one per query token). Stated inside the `HasWriteDesc` guard's
/// then-region.
pub fn write_kv_explicit(k: &Val, v: &Val, kv: &Kv) {
    record(
        &kv.t,
        Some(kv.l),
        "attn::write_kv_explicit_bf16",
        vec![],
        kv_state(kv),
        vec![k.id, v.id],
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

/// `kernels::ssm::causal_conv1d_update_batched_bf16`: the slot-indirected decode conv update (+
/// fused SiLU) against the layer's per-request conv slab. Shape-preserving, like the semantic
/// [`causal_conv1d`](crate::causal_conv1d) it lowers.
pub fn gdn_conv_update_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
    gdn_conv(x, w, rs, "ssm::causal_conv1d_update_batched_bf16")
}

/// `kernels::ssm::causal_conv1d_prefill_batched_bf16`: the batched prefill conv walk (each
/// request walking its qo_indptr window and persisting the trailing K-window into the slab).
pub fn gdn_conv_prefill_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
    gdn_conv(x, w, rs, "ssm::causal_conv1d_prefill_batched_bf16")
}

fn gdn_conv(x: &Val, w: &ConvW, rs: &Rs, kernel: &str) -> Val {
    let ids = x.t.with(Some(w.layer), |b| {
        let shape = b.value_shape(x.id);
        b.launch(
            kernel,
            vec![w.name.clone()],
            rs_state(rs),
            vec![x.id],
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
) -> Val {
    let kernel = match (gqa, state_bf16) {
        (true, true) => "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16",
        (true, false) => "ssm::recurrent_gated_delta_step_batched_gqa",
        (false, true) => "ssm::recurrent_gated_delta_step_batched_state_bf16",
        (false, false) => "ssm::recurrent_gated_delta_step_batched",
    };
    let ids = q.t.with(Some(rs.l), |b| {
        let shape = b.value_shape(v.id);
        b.launch(
            kernel,
            vec![],
            rs_state(rs),
            vec![q.id, k.id, v.id, g.id, beta.id],
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
) {
    // One arm per state dtype; no exported non-GQA duplicate.
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel);
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
) {
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched_cached"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel);
}

/// `kernels::ssm::chunk_gated_delta_prefill_batched[_state_bf16]`: the batched GQA-aware FLA
/// prefill recurrence — the fallback arm (it indexes the compact K_h layout directly, so no
/// repeats). Guard-region launch, output-less like the warp-tiled form.
pub fn gdn_prefill_fla(q: &Val, k: &Val, v: &Val, g: &Val, beta: &Val, rs: &Rs, state_bf16: bool) {
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel);
}

fn gdn_prefill(q: &Val, k: &Val, v: &Val, g: &Val, beta: &Val, rs: &Rs, kernel: &str) {
    record(
        &q.t,
        Some(rs.l),
        kernel,
        vec![],
        rs_state(rs),
        vec![q.id, k.id, v.id, g.id, beta.id],
        None,
    );
}

builder! {
    /// `repeat_interleave_heads_fp32`: produces `[Tokens, value_heads, key_dim]` f32 dataflow.
    pub fn repeat_interleave_heads(x: &Val, value_heads: u32, key_dim: u32) -> Val {
        symbol: "ssm::repeat_interleave_heads_fp32",
        on: x,
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
    decode_plan(&q.t, head_dim, window_left == -1);
    attn_at(
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
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16` — the prefill
/// counterpart, same guard-region contract.
pub fn attention_flashinfer_prefill_capture(
    q: &Val,
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
    prefill_plan(&q.t, head_dim);
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        cap_and_scale(soft_cap, sm_scale, head_dim),
    )
}

/// Output-less peel-prefix QK-norm/rope/KV write; peel owns q rows.
pub fn qkv_decode_qk_norm_rope_write_kv_region(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    table: Option<&Val>,
) {
    let mut inputs = vec![packed.id];
    if let Some(t) = table {
        inputs.push(t.id);
    }
    record(
        &packed.t,
        Some(kv.l),
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
        inputs,
        None,
    );
}

/// `dispatch_attention_flashinfer_prefill_custom`: custom-mask prefill; mask data is runtime args.
pub fn attention_flashinfer_prefill_custom(
    q: &Val,
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
    prefill_plan(&q.t, head_dim);
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_custom",
        cap_and_scale(soft_cap, sm_scale, head_dim),
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
pub fn dequant_only(kv: &Kv) {
    record(
        &kv.t,
        Some(kv.l),
        "attn::dequant_kv_cache_layer_to_bf16_active",
        vec![],
        kv_state(kv),
        vec![],
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
fn attn_at(q: &Val, kv: &Kv, kernel: &str, params: Vec<u32>) -> Option<Val> {
    let out = q.t.inner.borrow().inside_value_region();
    let shape = (!out).then(|| q.t.inner.borrow().value_shape(q.id));
    record_with_params(
        &q.t,
        Some(kv.l),
        kernel,
        vec![],
        kv_state(kv),
        params,
        vec![q.id],
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
