//! THE STATEMENTS EVERY GENERATION MAKES — the projections' landing,
//! the attention dispatches, rope, the KV writes, the recurrent
//! updates, and the odds and ends that travel with them.
//!
//! The most-reached-for file in the surface, and in the single-file
//! `cuda.rs` it was the LAST 1,129 lines, under a `TENSOR PARALLELISM`
//! heading it had nothing to do with. Nothing here moved relative to
//! anything else here; what changed is that it is no longer filed
//! under the wrong word.

use super::*;

/// `kernels::mlp::sigmoid_dot_scalar_gate_add_bf16`: the shared
/// expert's landing with its gate logit folded in — one launch that
/// dots `norm_x` with the `[1, H]` gate row, sigmoids the scalar, and
/// accumulates `shared` into the stream.
///
/// The general form is a `[Tokens, 1]` GEMM followed by a scalar-gated
/// add; this fused form runs when
/// the gate weight is bound unquantized and `N` is within the decode
/// fast path's bound (1024). Every fire this text covers is under
/// `cutlass_max_rows` (<= 512), so within the declaration's own row
/// range the fused form is not a guarded arm but the only arm.
pub fn sigmoid_dot_scalar_gate_add(
    x: &Val,
    gate: &MatW,
    shared: &Val,
    base: &Val,
    hidden: u32,
) -> Val {
    record(
        &x.t,
        gate.layer,
        "mlp::sigmoid_dot_scalar_gate_add_bf16",
        vec![gate.name.clone()],
        None,
        vec![x.id, base.id, shared.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the shared-expert landing produces its value")
}

/// `kernels::mlp::chunked_swiglu_bf16` over the routed rows — the
/// same kernel [`swiglu`]'s packed arm names, launched with `N * k`
/// rows instead of `N`. A separate statement because the SHAPE
/// differs, not the kernel: the routed value keeps its expert dim.
pub fn swiglu_routed(x: &Val, top_k: u32, intermediate: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "mlp::chunked_swiglu_bf16",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(top_k),
                Dim::Const(intermediate),
            ]),
            DType::BF16,
        )),
    )
    .expect("the routed activation produces its value")
}

/// `kernels::moe::token_batched_weighted_sum_bf16`, or the
/// `..._add_bf16` form when the residual folds into the same launch.
///
/// The combine collapses `[Tokens, k, H]` to `[Tokens, H]` under the
/// router's weights. `fold_residual` is the hand-written pass's
/// `add_to_residual`: at tp=1 the MoE output lands straight on the
/// residual stream, so the add is not a second launch. Stating it
/// here is what lets the body emit ONE op where the semantic text
/// emits a WeightedSum and a ResidualAdd — the fusion is a kernel
/// fact, so it belongs in the CUDA reading, not in the trace shape.
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

/// The MLP activation, stating which of the two swiglu kernels runs.
///
/// `packed` is `model::…::LlamaLikeCudaFacts::gate_up_fused`: a
/// checkpoint that bound the packed gate‖up bank lands the projection
/// in one buffer and takes the CHUNKED kernel; one that did not lands
/// two and takes the pair form. Same arithmetic, different addressing
/// — which is exactly the kind of choice that used to sit in the
/// executor (`declared::arm_swiglu`) and in the generated file (a
/// per-layer `if (gate_up_fused_N)`), reading a workspace to decide
/// what the binding had already decided at load.
///
/// One value either way: the trace declares ONE packed matmul before
/// this, and whether the binding materialised it as one buffer or two
/// is a BUFFER question, which is `lower::Buffers`'.
pub fn swiglu(x: &Val, intermediate: u32, packed: bool) -> Val {
    record(
        &x.t,
        x.layer,
        if packed {
            "mlp::chunked_swiglu_bf16"
        } else {
            "mlp::swiglu_bf16"
        },
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

/// `kernels::mlp::chunked_swiglu_bf16` over the ALIGNED leg's
/// block-major staging: [`swiglu`](crate::cuda::swiglu)'s shape, plus the
/// destination the pointer build named.
///
/// Its own statement because the destination is not this call's to
/// choose. The aligned staging's addresses are baked into the
/// pointer arrays, so the activation has to land on the buffer
/// `build_moe_ptrs_aligned` declared — an operand, written in place,
/// exactly as the two grouped GEMMs around it do. Stating it any
/// other way puts the activation somewhere the down projection's
/// pointers do not point.
pub fn swiglu_aligned(x: &Val, stage: &Val, aligned: Dim, intermediate: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "mlp::chunked_swiglu_bf16",
        vec![],
        None,
        vec![x.id, stage.id],
        Some((Shape(vec![aligned, Dim::Const(intermediate)]), DType::BF16)),
    )
    .expect("the aligned activation produces its value")
}

/// `kernels::mlp::swiglu_bf16` in its PAIR form: two operands, the
/// gate and the up projection, into one activation.
///
/// The spelling an UNFUSED gate_up binding actually fires, and the
/// one the declaration could not carry until now. [`swiglu`](crate::cuda::swiglu)
/// above states one packed operand either way and lets `packed` pick
/// the kernel — which left the pair form reading two workspace
/// buffers (`ws.gate`, `ws.up`) that no traced value described, so
/// the executor had to keep that convention and cross-check it
/// against the fact on every launch.
///
/// With the projections stated as two matmuls the two operands ARE
/// values, and the whole `gate_up_used_fused` correspondence between
/// the Matmul arm and this one disappears: each statement says what
/// it reads.
pub fn swiglu_pair(gate: &Val, up: &Val, intermediate: u32) -> Val {
    record(
        &gate.t,
        gate.layer,
        "mlp::swiglu_bf16",
        vec![],
        None,
        vec![gate.id, up.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(intermediate)]),
            DType::BF16,
        )),
    )
    .expect("the activation produces its value")
}

/// `kernels::rope::qk_rmsnorm_rope_bf16`: the fused per-head q/k
/// norm + Standard rope, one launch — the hand-written
/// `fuse_qk_norm_rope` branch. bf16 rounding differs between this
/// kernel and the norm+rope triple it replaces, so parity requires
/// stating it wherever the hand-written path takes it: every lowered
/// arm with per-head qk-norm and Standard rope that did not take the
/// fully-fused decode post. In place on q and k; SSA-wise two fresh
/// values.
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

/// `kernels::attn::attention_xqa_decode_bf16_prepared` (whose contract
/// includes the fire-wide XQA prepare — and which is therefore
/// declared `whole`; see [`model_ir::kernels`]).
pub fn attention_xqa_decode(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(
        q,
        kv,
        "attn::attention_xqa_decode_bf16_prepared",
        window_left,
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_decode` against the decode
/// plan its contract obligates.
pub fn attention_flashinfer_decode(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_decode",
        window_left,
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_prefill_bf16` — the dispatch
/// ALONE.
///
/// Three wrappers used to differ here only by whether they also
/// launched the dequant staging: llama_like's cache may be
/// quantized, so its prefill-shaped arms dequant the layer first,
/// while qwen3_5's full-attention path gates on a native-bf16 cache
/// and launches only the dispatch. That is not a property of this
/// kernel — it is a second STATEMENT the text either makes or does
/// not, so the text makes it ([`dequant_only`] beside this call).
pub fn attention_flashinfer_prefill(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_bf16",
        window_left,
    )
}

/// `kernels::attn::attention_flashinfer_prefill` — the PLAN-FREE
/// prefill wrapper, which builds its own R-shaped plan from the host
/// indptrs on the way in.
///
/// A DIFFERENT statement from [`attention_flashinfer_prefill`], not
/// a spelling of it: that one names the dispatch alone and its
/// caller owes it a plan, while this one owes nothing and cannot be
/// given a row window (the plan it builds spans all R requests).
/// gemma-4's prefill fires this; llama_like's fires the other. The
/// two are one call apart in C++ and a whole contract apart here,
/// which is why the table carries both.
pub fn attention_flashinfer_prefill_planless(
    q: &Val,
    kv: &Kv,
    window_left: i32,
) -> Option<Val> {
    attn_at(q, kv, "attn::attention_flashinfer_prefill", window_left)
}

/// The paged attention for a fire CLASS: decode's dispatch, or the
/// plan-free prefill.
///
/// A plain function, deliberately, and this is the same argument
/// [`crate::attention_landing`] makes for itself one screen
/// away: the alternative is nine families each hand-writing the same
/// two-arm match, and the ORDER of the arms is the contract.
///
/// **Why it is worth existing at all.** Seven of eleven declared
/// families serve Decode ONLY, and they do not refuse — they
/// `panic!("<family> states no {class:?} class yet")` on the first
/// prefill. Counting the class-dependent sites in the two families
/// that DO serve both says why: `gpt_oss` has four and all four
/// select an attention op; `gemma_4` has six and five do. The classes
/// differ in almost nothing else, so Prefill was missing from seven
/// families because each would have had to hand-write this match —
/// not because their bodies diverge.
///
/// **What it does not decide.** Which prefill a family fires is a
/// contract difference and not a spelling:
/// [`attention_flashinfer_prefill`] names the dispatch alone and its
/// caller owes it a plan, while [`attention_flashinfer_prefill_planless`]
/// owes nothing and cannot be given a row window. This helper takes
/// the planless one, which is what a family with no prefill plan of
/// its own can state; a family that owes a plan calls the other
/// directly and always did.
pub fn attention_for(
    class: model_ir::trace::FireClass,
    q: &Val,
    kv: &Kv,
    window_left: i32,
) -> Option<Val> {
    match class {
        model_ir::trace::FireClass::Decode => attention_flashinfer_decode(q, kv, window_left),
        _ => attention_flashinfer_prefill_planless(q, kv, window_left),
    }
}

/// [`attention_for`], asked for its LSE — the sink families' form.
///
/// Its own function rather than an `Option` return on the one above,
/// because the LSE spellings return `(Val, Val)` and not
/// `Option<Val>`: a dispatch asked for two values cannot decline the
/// second half. Two functions with two return types is the honest
/// shape; one function returning a tuple of options would make every
/// caller unwrap a case that cannot happen.
pub fn attention_for_lse(
    class: model_ir::trace::FireClass,
    q: &Val,
    kv: &Kv,
    q_heads: u32,
) -> (Val, Val) {
    match class {
        model_ir::trace::FireClass::Decode => attention_flashinfer_decode_lse(q, kv, q_heads),
        _ => attention_flashinfer_prefill_lse(q, kv, q_heads),
    }
}

/// `kernels::attn::dispatch_attention_flashinfer_decode` asked for its LSE.
///    /// The SAME symbol as [`attention_flashinfer_decode`] and a
/// different call: `lse_out` is the last positional argument of
/// every flashinfer entry point, and the driver passes it only on
/// layers that carry attention sinks (`layer.attn_sinks != nullptr`,
/// a load-time per-layer answer). Supplying it costs a per-layer
/// write, which is why plain Mixtral layers do not — so whether this
/// statement or the one-value one runs is a FACT, not a branch.
///
/// Produces `(o, lse)`. The LSE is fp32 `[Tokens, q_heads]`, and it
/// exists so [`attention_sink_rescale`] can apply the
/// softmax-denominator extension flashinfer's DefaultAttention does
/// not emit natively.
pub fn attention_flashinfer_decode_lse(q: &Val, kv: &Kv, q_heads: u32) -> (Val, Val) {
    let shape = q.t.inner.borrow().value_shape(q.id);
    let ids = q.t.with(Some(kv.l), |b| {
        b.launch(
            "attn::dispatch_attention_flashinfer_decode",
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

/// `kernels::rope::rope_yarn_original_bf16`: the YaRN-paper rope —
/// a dim-index ramp between interpolated and extrapolated
/// frequencies, plus an `attention_factor` magnitude scale.
///
/// A different KERNEL from the plain rope, not a parameterisation:
/// which one a deployment fires is decided by its config at load and
/// erases here. The semantic [`super::rope`] carries a `RopeKind`
/// the lowering refuses for anything but Standard, so a family that
/// scales says so by naming the launcher.
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
///
/// The semantic [`super::rope`] carries a `RopeKind` and a rotary
/// width, and the driver's arm asked whether the width was zero to
/// decide between two launchers. That is a KERNEL CHOICE — a full
/// rotation and a partial one are different arithmetic — so it
/// belongs in the statement, and the pair below is that statement.
pub fn rope(q: &Val, k: &Val) -> (Val, Val) {
    rope_launch(q, k, "rope::rope_bf16", vec![0])
}

/// `kernels::rope::rope_partial_bf16`: only the first `rotary_dim`
/// channels of each head rotate.
///
/// `rotary_dim` rides the statement's PARAMS
/// ([`model_ir::trace::OpKind::Launch`]), not the executor's config.
///
/// The THETA does not, yet, and the reason is worth writing down
/// rather than leaving as an omission: gemma-4 alternates it per
/// layer between its local and global attention, so a driver
/// reading the single `cfg.rope_theta` reads the wrong one for half
/// that model's layers — the fact belongs here. What blocks it is
/// the emission fixtures, which would have to state each target's
/// real theta, and inventing those numbers is worse than a driver
/// reading a config value that is uniform for every family but one.
/// It is a property of this rotation and no operand shape spells it
/// — the operands are full-width q and k either way — which is
/// exactly what that channel is for.
pub fn rope_partial(q: &Val, k: &Val, rotary_dim: u32) -> (Val, Val) {
    assert!(
        rotary_dim > 0,
        "a partial rotation with no channels is the full one; state \
         `cuda::rope`"
    );
    rope_launch(q, k, "rope::rope_partial_bf16", vec![rotary_dim])
}

/// The shape both rotations share: two operands in, two results out,
/// each landing where its operand lies (the `kernel!` rows alias
/// both pairs).
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

/// `kernels::gemm::act_x_wt_bias_bf16`: a projection whose BIAS RIDES IN
/// THE EPILOGUE.
///
/// Not a `matmul` plus an [`add_bias`](crate::add_bias). At decode this routes
/// to the warp-per-row GEMV, whose epilogue absorbs the bias for
/// free — so the folded form is one launch where the split form is
/// two, and the two do not accumulate in the same order. A family
/// whose driver folds must say so: mixtral folds q/k/v and the
/// router, and adds `o_bias` separately, which is why gpt-oss's text
/// uses both spellings and neither by default.
pub fn gemm_bias(x: &Val, w: &MatW, bias: &MatW) -> Val {
    record(
        &x.t,
        w.layer,
        "gemm::act_x_wt_bias_bf16",
        vec![w.name.clone(), bias.name.clone()],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(w.width)]), DType::BF16)),
    )
    .expect("a biased projection produces its value")
}

/// `kernels::attn::attention_flashinfer_prefill` asked for its LSE —
/// the prefill twin of [`attention_flashinfer_decode_lse`], and the
/// same argument makes the difference.
pub fn attention_flashinfer_prefill_lse(q: &Val, kv: &Kv, q_heads: u32) -> (Val, Val) {
    let shape = q.t.inner.borrow().value_shape(q.id);
    let ids = q.t.with(Some(kv.l), |b| {
        b.launch(
            "attn::attention_flashinfer_prefill",
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

/// `kernels::attn::attention_sink_rescale_bf16`: `o *= sigmoid(lse
/// - sink_h)`, in place, per (token, head).
///
/// gpt-oss learns a per-head SINK logit that participates in the
/// softmax denominator without contributing a value — so the whole
/// effect is a rescale of the attention output by how much
/// probability mass the sink would have taken. The sink weight is
/// `[q_heads]`, which is why it rides in the weight slot.
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

/// `kernels::quant::bf16_to_fp16`: the activation cast the MXFP4
/// routed GEMVs want on their input.
///
/// A statement rather than an implementation detail of the GEMV
/// because it is its own launch over its own extent — and because
/// the routed leg casts TWICE, once on the block input and once on
/// the post-activation routes, over different extents.
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

/// `kernels::quant::mxfp4_moe_gate_up_decode_bf16`: BOTH routed
/// projections of gpt-oss's fused decode leg, in one launch,
/// reading the packed 4-bit nibbles straight out of HBM.
///
/// The weight slot names the layer's per-expert POINTER BANK, not a
/// tensor: the kernel indexes experts through a device array of
/// pointers plus a parallel scale array. That indirection is a
/// BINDING — the executor resolves the name to whatever the layer
/// holds, exactly as [`moe_fused_cutlass`] resolves its two banks —
/// so it is not the obstacle it looks like. What would be an
/// obstacle is the host-routed walk this leg replaces: its launch
/// count depends on which experts the router picked, and no
/// rectangle spells that.
///
/// Produces `(gate, up)`, each `[Tokens, k, intermediate]` — the
/// routed extent as a third dim, [`moe_gate_up_gemv`]'s convention.
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

/// `kernels::quant::mxfp4_moe_down_decode_bf16`: the routed down
/// projection, the same bank convention as
/// [`mxfp4_moe_gate_up_decode`].
pub fn mxfp4_moe_down_decode(
    x: &Val,
    experts: &Val,
    bank: &MatW,
    top_k: u32,
    hidden: u32,
) -> Val {
    record(
        &x.t,
        bank.layer,
        "quant::mxfp4_moe_down_decode_bf16",
        vec![bank.name.clone()],
        None,
        vec![experts.id, x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(hidden)]),
            DType::BF16,
        )),
    )
    .expect("the routed down projection produces its value")
}

/// `kernels::mlp::gpt_oss_glu_bf16`: SwiGLU with gpt-oss's CLAMP.
///
/// A different kernel from [`swiglu`], not a parameterisation of it:
/// `swiglu_limit` is a config constant, so which of the two runs is
/// decided at load and erases here. Reading it as a runtime scalar
/// would put a branch in every fire for an answer that never
/// changes.
/// Its extent is the ROUTED one — `[Tokens, k, intermediate]`, the
/// shape of the operands it consumes, not `[Tokens, intermediate]`.
/// Declaring the collapsed shape made the two `bf16_to_fp16` sites
/// indistinguishable to anything reading the trace, and the second
/// one re-cast the block input while the routed activations were
/// never written — a live defect the ledger, the golden and the
/// registry all passed.
/// `limit` is the deployment's `swiglu_limit`, and it rides the
/// param channel for the reason [`scalar_mul`](crate::cuda::scalar_mul)'s scale does:
/// it is a load-time number the host has, and the executor was
/// reaching into a config struct for it.
pub fn gpt_oss_glu(gate: &Val, up: &Val, top_k: u32, intermediate: u32, limit: f32) -> Val {
    record_with_params(
        &gate.t,
        gate.layer,
        "mlp::gpt_oss_glu_bf16",
        vec![],
        None,
        vec![limit.to_bits()],
        vec![gate.id, up.id],
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(top_k),
                Dim::Const(intermediate),
            ]),
            DType::BF16,
        )),
    )
    .expect("the clamped GLU produces its value")
}

/// `kernels::attn::attention_naive_paged` — the fallback prefill for a
/// head dim flashinfer's TC prefill template rejects.
///
/// gemma-4's FULL-attention layers run at head_dim 512, and
/// flashinfer 0.6.x refuses to instantiate a prefill at
/// `NUM_MMA_D_QK=32`. So the deployment states a naive paged kernel
/// on exactly those layers — a per-layer HEAD DIM fact, erased at
/// trace time, not a runtime fallback the executor discovers.
pub fn attention_naive_paged(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(q, kv, "attn::attention_naive_paged", window_left)
}

/// `kernels::attn::write_kv_explicit_bf16`: the explicit-descriptor
/// KV write (graph-replay steering; N cells, one per query token).
/// Stated inside the `HasWriteDesc` guard's then-region.
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

/// `kernels::attn::write_kv_to_pages`: the page-derived append
/// (position re-derived from the page table). The `HasWriteDesc`
/// guard's else-region.
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

/// `kernels::ssm::causal_conv1d_update_batched_bf16`: the
/// slot-indirected decode conv update (+ fused SiLU) against the
/// layer's per-request conv slab. Shape-preserving, like the
/// semantic [`causal_conv1d`](crate::causal_conv1d) it lowers.
pub fn gdn_conv_update_batched(x: &Val, w: &ConvW, rs: &Rs) -> Val {
    gdn_conv(x, w, rs, "ssm::causal_conv1d_update_batched_bf16")
}

/// `kernels::ssm::causal_conv1d_prefill_batched_bf16`: the batched
/// prefill conv walk (each request walking its qo_indptr window and
/// persisting the trailing K-window into the slab).
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

/// `kernels::ssm::recurrent_gated_delta_step_batched[_gqa][_state_bf16]`:
/// the one-token decode recurrence step against the layer's
/// per-request recurrent state. `gqa` states the compact-K_h-indexing
/// GQA variant (value heads != key heads); `state_bf16` the store
/// dtype. Output = the semantic [`gated_delta`](crate::gated_delta)'s: the core keeps v's
/// `[Tokens, Vh, Vd]` f32 shape.
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

/// `kernels::ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa[_state_bf16]`:
/// the warp-tiled small-N prefill recurrence. NOT a value producer:
/// the three prefill recurrence signatures record launches with NO
/// outputs, because each runs inside a value-producing guard chain
/// ([`guarded_value`](crate::guarded_value)) whose output IS the recurrence core — the
/// region launches bind the guard's buffer and add no SSA values of
/// their own.
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
    // ONE arm per state dtype, and nothing else to choose. The GQA
    // kernel's `repeat` is 1 when `K_h == V_h`, its `qk_h` is `h`, and
    // its index reduces to the non-GQA one exactly — so that pair was
    // a second copy of the same arithmetic and upstream deleted it.
    // Keeping a statement for a symbol the driver no longer exports
    // would be a declaration that cannot load.
    let kernel = if state_bf16 {
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
    } else {
        "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
    };
    gdn_prefill(q, k, v, g, beta, rs, kernel);
}

/// `kernels::ssm::chunk_gated_delta_prefill_batched_cached[_state_bf16]`:
/// the env-gated cached prefill recurrence. No `_gqa` variant exists —
/// this family indexes the REPEATED `[Vh]`-head layout, which is why
/// its guard arm materializes [`repeat_interleave_heads`] first.
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

/// `kernels::ssm::chunk_gated_delta_prefill_batched[_state_bf16]`:
/// the batched GQA-aware FLA prefill recurrence — the fallback arm
/// (it indexes the compact K_h layout directly, so no repeats).
/// Guard-region launch, output-less like the warp-tiled form.
pub fn gdn_prefill_fla(
    q: &Val,
    k: &Val,
    v: &Val,
    g: &Val,
    beta: &Val,
    rs: &Rs,
    state_bf16: bool,
) {
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

/// `kernels::ssm::repeat_interleave_heads_fp32`: materialize the
/// K_h → V_h head repeat of a compact per-head f32 value. Stated
/// only inside the cached arm, because only that kernel family
/// consumes the repeated layout (the decode-GQA step, warp-tiled and
/// FLA kernels all index the compact layout directly).
///
/// It DECLARES its result, which it did not use to. Output-less, the
/// stance was "where that buffer lives is the driver's binding, not
/// dataflow" — and the cost of that stance was paid twice over: the
/// driver kept a `repeat_next_is_k` toggle to decide which of two
/// workspace fields a launch meant, the emitter kept the SAME toggle
/// to decide it statically, and the recurrence below could not name
/// its own q/k operands because the value between them had no id.
/// A repeat is dataflow; it took a value to say so.
///
/// `[Tokens, value_heads, key_dim]` f32 — the compact `[Tokens,
/// key_heads, key_dim]` operand with each key head repeated to fill
/// the value-head count.
pub fn repeat_interleave_heads(x: &Val, value_heads: u32, key_dim: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "ssm::repeat_interleave_heads_fp32",
        vec![],
        None,
        vec![x.id],
        Some((
            Shape(vec![
                Dim::Tokens,
                Dim::Const(value_heads),
                Dim::Const(key_dim),
            ]),
            DType::F32,
        )),
    )
    .expect("the head repeat produces its value")
}

/// `"ssm::verify_stash_load"`: replay the layer's stashed in-proj
/// outputs — `[mixed_qkv | a | b]` from the verify hidden stash slab
/// into the workspace buffers the following conv/prep read.
///
/// **It was `qwen35_verify_stash_load`, and the rename is the point of
/// this paragraph.** Every other symbol in this file is `family::kernel`,
/// where the family is a `csrc/src/` directory — and `kernels-cuda`
/// DERIVES that prefix from the module path, so the Rust path and the
/// trace symbol are one string that cannot drift apart. These two named
/// a MODEL instead, which breaks the equation twice over: a bare symbol
/// has no family to derive from, and a symbol that names a deployment
/// makes the table grow once per model. `qwen3.6` would have wanted its
/// own pair, for an operation that is not qwen-anything.
///
/// `ssm`, because that is what it is: the in-proj triple of a LINEAR
/// layer, stashed and replayed around the recurrent state. It sits four
/// lines below `ssm::repeat_interleave_heads_fp32` and describes the
/// same layer. The Rust builders were called `verify_stash_load` and
/// `verify_stash_store` all along — only the wire name carried the
/// deployment, which is the shape of a name nobody re-read.
///
/// A PSEUDO-SYMBOL, the first: it names an OPERATION the driver
/// implements as a `cudaMemcpyAsync` trio, not a `__global__` entry
/// point. The contract stands regardless — a launcher may be three
/// API calls; the symbol names the operation, and the driver's
/// name→launcher registry resolves it like any other. No inputs
/// (the stash is the layer's per-request state, marked below);
/// THREE outputs, the in-proj triple the GEMMs would have produced —
/// mixed_qkv `[Tokens, conv_dim]`, a `[Tokens, value_heads]`, b
/// `[Tokens, value_heads]`, all bf16 — so the CommitAdvance pass's
/// dataflow into `gdn_prep` stays complete. WHERE those buffers
/// live is the driver's binding, [`repeat_interleave_heads`]-style.
pub fn verify_stash_load(
    t: &Trace,
    rs: &Rs,
    conv_dim: u32,
    value_heads: u32,
) -> (Val, Val, Val) {
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

/// `"ssm::verify_stash_store"`: persist a linear layer's in-proj
/// triple `[qkv, a, b]` into the layer's verify hidden stash slab —
/// [`verify_stash_load`]'s writing half, the same pseudo-symbol
/// contract (a memcpy trio behind one name). Output-less: the stash
/// is the layer's per-request state, not dataflow.
///
/// No class this rung states it: its consumer is the future
/// frozen-verify class (the `write_state=false` verify pass that
/// fills the stash the commit pass replays — semantic this rung).
/// The pair is declared together because the load's contract is only
/// meaningful against the store's layout.
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

/// `kernels::attn::dispatch_attention_flashinfer_decode_capture`: the
/// score-capturing decode dispatch (the OnAttn sideband's producer;
/// its contract includes the capture publish against the possibly
/// page-mask-compacted CSR). Region launch of the WantsAttnScore
/// guard — output-less; the guard owns the attention output.
pub fn attention_flashinfer_decode_capture(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_decode_capture",
        window_left,
    )
}

/// `kernels::attn::dispatch_attention_flashinfer_prefill_capture_bf16` — the
/// prefill counterpart, same guard-region contract.
pub fn attention_flashinfer_prefill_capture(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_capture_bf16",
        window_left,
    )
}

/// Output-less [`qkv_decode_qk_norm_rope_write_kv`] for the Peel's
/// prefix region (A3): the peel owns q; this launch binds its
/// `[0, fast_rows)` rows. Same operands, same aux contract.
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

/// `kernels::attn::dispatch_attention_flashinfer_prefill_custom`: the
/// custom-mask prefill dispatch — a genuinely distinct launcher, so
/// no pseudo-symbol is needed. The mask data (BRLE bytes + indptr)
/// crosses as runtime args of the stated kernel, commit_lens's peer.
/// Since A1 (the class-collapse amendment) it is stated inside the
/// `HasCustomMask` guard arm of the Decode/Prefill traces.
pub fn attention_flashinfer_prefill_custom(q: &Val, kv: &Kv, window_left: i32) -> Option<Val> {
    attn_at(
        q,
        kv,
        "attn::dispatch_attention_flashinfer_prefill_custom",
        window_left,
    )
}

/// `"gemm::lora_qkv_correction"`: the §5.1 adapter correction — every
/// usable lora lane's `x·Aᵀ·Bᵀ` delta landed on the materialized q/v
/// projections, before anything consumes them (bias, norms, rope,
/// KV append). A launcher may be many calls; the symbol names the
/// operation. Output-less and in place on q/v; stated inside the
/// `HasLora` guard's then-region (the else is empty — a fire with no
/// adapters launches nothing, which is the truth).
///
/// **It was `pie_lora_qkv_correction`, and was called a PSEUDO-SYMBOL on
/// the strength of that name.** (Unquoted deliberately:
/// `model/tests/kernels_table.rs` reads this file's string literals to
/// find the symbols a wrapper records, and a quoted dead name in a doc
/// comment is a symbol it would go looking for a row for.) It is not one: it names three passes
/// of batched cuBLAS, which is exactly what `gemm::mla_absorb_*` names,
/// and those two are ordinary routines. What the old name really carried
/// was a `pie_` from a C ABI this tree no longer has, and the cost of
/// carrying it was that no family could offer the symbol — a CUDA family
/// namespace is its module path, and a bare name has no module. The body
/// is `kernels_cuda::gemm::lora`'s and the row derives from it.
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

/// `kernels::attn::dequant_kv_cache_layer_to_bf16_active`: the
/// staging launch a quantized cache needs before a prefill-shaped
/// dispatch. Its OWN statement — see
/// [`attention_flashinfer_prefill`] for why it is not folded into
/// any attention wrapper.
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

/// ONE attention statement, whatever position it is written in
/// (`.wiki/tart/dsl.md` ②, migration step 2).
///
/// A dispatch inside a value-producing guard or peel region binds
/// that construct's output and records no SSA output of its own; the
/// same dispatch written as a plain statement produces its own
/// value. That is a property of the STATEMENT'S POSITION, which the
/// tape knows ([`model_ir::trace::TraceBuilder::inside_value_region`]),
/// so it stops being spelled in the wrapper's name — the `_region`
/// half of every attention wrapper is deleted by this one function.
///
/// The output shape is q's own: these kernels are width-preserving
/// on the query, which is what the retired `q_width` parameter was
/// re-stating at each call site.
/// Every FlashInfer/XQA dispatch's shape: one query in, the layer's
/// cache as state, and the attention output — or none, inside a
/// value-producing guard region, where the guard owns the value.
///
/// `window_left` is the SLIDING WINDOW this layer attends over,
/// `-1` for none. It is a load-time fact (a config's
/// `sliding_window`, or its per-layer list where the architecture
/// alternates), and it used to be derived inside every executor:
/// eleven copies of the same three lines across four families,
/// reaching into `fwd_cfg.per_layer_window_left` — a per-layer array
/// no statement mentioned.
///
/// It rides the statement's PARAMS because no operand shape gives
/// it. What is NOT closed by this is the per-FIRE override
/// (`runtime_window_left`), which is a runtime input and wants a
/// guard predicate; `DeclineReason::SlidingWindow` still names it.
fn attn_at(q: &Val, kv: &Kv, kernel: &str, window_left: i32) -> Option<Val> {
    let out = q.t.inner.borrow().inside_value_region();
    let shape = (!out).then(|| q.t.inner.borrow().value_shape(q.id));
    record_with_params(
        &q.t,
        Some(kv.l),
        kernel,
        vec![],
        kv_state(kv),
        vec![window_left as u32],
        vec![q.id],
        shape.map(|s| (s, DType::BF16)),
    )
}
