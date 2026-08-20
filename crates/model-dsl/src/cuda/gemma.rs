//! GEMMA — 3n's AltUp, and gemma-4.

use super::*;

// Statements here are kept in algorithm order; `builder!` only covers
// records whose result shape can be stated from dims.
// AltUp carries rank-K residual streams, so its explicit shapes stay.
builder! {
    /// `norm::altup_predict_bf16`: predicted K-stream post-layer state.
    /// `coefs` stays fp32 for the K-sum.
    pub fn altup_predict(streams: &Val, coefs: &Val, k: u32, hidden: u32) -> Val {
        symbol: "norm::altup_predict_bf16",
        on: streams,
        inputs: [streams, coefs],
        out: [Dim::Const(k), Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the prediction produces its value",
    }


    /// `norm::altup_correct_bf16`: correct the K-1 inactive streams.
    /// The `+1` is already folded into `correction_coefs`.
    pub fn altup_correct(
        predictions: &Val,
        activated: &Val,
        correction_coefs: &Val,
        k: u32,
        hidden: u32,
    ) -> Val {
        symbol: "norm::altup_correct_bf16",
        on: predictions,
        inputs: [predictions, activated, correction_coefs],
        out: [Dim::Const(k), Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the correction produces its value",
    }


    /// `norm::altup_unpack_predict_coefs`: `[T, K*K]` bf16 to `[T, K, K]` fp32,
    /// including HF's transpose.
    pub fn altup_unpack_predict_coefs(packed: &Val, k: u32) -> Val {
        symbol: "norm::altup_unpack_predict_coefs",
        on: packed,
        inputs: [packed],
        out: [Dim::Tokens, Dim::Const(k), Dim::Const(k)] as F32,
        made: "the unpack produces its value",
    }


    /// `norm::altup_unpack_correct_coefs`: `[T, K]` fp32 with HF's `+1` folded in.
    pub fn altup_unpack_correct_coefs(packed: &Val, k: u32) -> Val {
        symbol: "norm::altup_unpack_correct_coefs",
        on: packed,
        inputs: [packed],
        out: [Dim::Tokens, Dim::Const(k)] as F32,
        made: "the unpack produces its value",
    }


    /// `norm::mean_streams_bf16`: average K residual streams into one.
    pub fn mean_streams(streams: &Val, hidden: u32) -> Val {
        symbol: "norm::mean_streams_bf16",
        on: streams,
        inputs: [streams],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the mean produces its value",
    }


    /// `norm::compute_rms_bf16`: row RMS measurement for [`magnitude_rescale`].
    pub fn compute_rms(x: &Val) -> Val {
        symbol: "norm::compute_rms_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens] as F32,
        made: "the measurement produces its value",
    }


    /// `norm::magnitude_rescale_bf16`: scale each row of `x` to `target_rms`.
    /// The kernel is in-place; the trace still records the produced value.
    pub fn magnitude_rescale(x: &Val, target_rms: &Val, hidden: u32) -> Val {
        symbol: "norm::magnitude_rescale_bf16",
        on: x,
        inputs: [x, target_rms],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the rescale produces its value",
    }
}

/// `norm::tanh_bf16` on AltUp's modality-router output.
/// The output shape is copied from the operand; respelling it once disagreed
/// with a selected stream slice, and the in-place row exposed the alias.
pub fn tanh(x: &Val) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record(
        &x.t,
        x.layer,
        "norm::tanh_bf16",
        vec![],
        None,
        vec![x.id],
        Some(out),
    )
    .expect("the activation produces its value")
}

builder! {
    /// `mlp::gaussian_topk_bf16`: zero values below `mean + m·std` per row.
    /// `params[0]` is `std_multiplier` as `f32` bits; no driver table supplies it.
    pub fn gaussian_topk(x: &Val, width: u32, std_multiplier: f32) -> Val {
        symbol: "mlp::gaussian_topk_bf16",
        on: x,
        params: [std_multiplier.to_bits()],
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the sparsifier produces its value",
    }
}

// ── gemma-4 ────────────────────────────────────────────────────
// Hand-written cases either read result shapes from the tape, use `Trace::with`,
// or choose a symbol from facts/arguments.
pub fn geglu_tanh(x: &Val, intermediate: u32, packed: bool) -> Val {
    record(
        &x.t,
        x.layer,
        if packed {
            "mlp::chunked_geglu_tanh_bf16"
        } else {
            "mlp::geglu_tanh_bf16"
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

builder! {
    /// `mlp::geglu_tanh_bf16` pair form: gate and up are separate operands.
    /// Used when PLE supplies `up` from a per-layer table, not the packed bank.
    pub fn geglu_tanh_pair(gate: &Val, up: &Val, width: u32) -> Val {
        symbol: "mlp::geglu_tanh_bf16",
        on: gate,
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the activation produces its value",
    }
}

/// `rope::rope_partial_q_bf16`: rotate Q only; shared K was cached at its source.
/// Separate symbol: operand count must not encode the operation.
pub fn rope_partial_q_only(q: &Val, rotary_dim: u32) -> Val {
    let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
    record_with_params(
        &q.t,
        q.layer,
        "rope::rope_partial_q_bf16",
        vec![],
        None,
        vec![rotary_dim],
        vec![q.id],
        Some(out),
    )
    .expect("the rotation produces its value")
}

/// [`qk_rmsnorm_rope_rounded`] with K absent, as one fused launch.
/// The driver passes `k_norm = nullptr` and `num_kv_heads = 0`; separate symbol, shared C++.
pub fn qk_rmsnorm_rope_rounded_q_only(q: &Val, q_norm: &NormW) -> Val {
    let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
    record(
        &q.t,
        q_norm.layer,
        "rope::q_rmsnorm_rope_bf16_rounded",
        vec![q_norm.name.clone()],
        None,
        vec![q.id],
        Some(out),
    )
    .expect("the fused pair produces q")
}

/// `norm::rmsnorm_no_scale_bf16`: per-head `v / rms(v)` with no learnable weight.
pub fn rmsnorm_no_scale(x: &Val) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record(
        &x.t,
        x.layer,
        "norm::rmsnorm_no_scale_bf16",
        vec![],
        None,
        vec![x.id],
        Some(out),
    )
    .expect("the norm produces its value")
}

/// `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`: norm, residual add, scale, next norm.
/// Returns `(hidden, norm_out)` in that order.
///
/// `scale` is the routine's one `Const`, and the caller states it: the
/// unfused landing (`norm_residual_add`) applies none, so a family that folds
/// the next norm in must say what it scales by rather than leave a parameter
/// nothing supplies.
pub fn norm_residual_scale_norm(
    x: &Val,
    y: &Val,
    w: &NormW,
    next: &NormW,
    hidden: u32,
    scale: f32,
) -> (Val, Val) {
    let shape = (Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16);
    let ids = x.t.with(w.layer, |b| {
        b.launch_with_params(
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16",
            vec![w.name.clone(), next.name.clone()],
            None,
            vec![scale.to_bits()],
            // `y` is the old stream: read, accumulated into, and returned.
            vec![x.id, y.id],
            vec![shape.clone(), shape],
        )
    });
    let mk = |id| Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

builder! {
    /// `norm::rmsnorm_residual_add_bf16`: post-FFN norm plus residual add.
    pub fn norm_residual_add(x: &Val, y: &Val, w: &NormW, hidden: u32) -> Val {
        symbol: "norm::rmsnorm_residual_add_bf16",
        on: x,
        weights: [w.name],
        layer: w.layer,
        // `y` is the residual stream; naming it preserves the SSA edge.
        inputs: [x, y],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the fused norm+residual produces its value",
    }
}

/// `norm::scalar_mul_bf16`: multiply by a named load-time constant.
/// The weight name is `scale.<name>`; if present, `params[0]` is `by` as `f32` bits.
/// `None` means this family has not derived the number and falls through to a handwritten arm.
pub fn scalar_mul(x: &Val, scale: &str, by: Option<f32>) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record_with_params(
        &x.t,
        x.layer,
        "norm::scalar_mul_bf16",
        vec![format!("scale.{scale}")],
        None,
        by.map(f32::to_bits).into_iter().collect(),
        vec![x.id],
        Some(out),
    )
    .expect("the scale produces its value")
}

builder! {
    /// `attn::logit_softcap_bf16`: `cap * tanh(x / cap)`, present only when traced.
    pub fn logit_softcap(x: &Val, vocab: u32) -> Val {
        symbol: "attn::logit_softcap_bf16",
        on: x,
        layer: None,
        inputs: [x],
        out: [Dim::Requests, Dim::Const(vocab)] as BF16,
        made: "the softcap produces its value",
    }


    /// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`: split packed qkv, norm q/k/v,
    /// rope q/k, write k/v to cache, and return q.
    pub fn qkv_packed_post(
        packed: &Val,
        q_norm: &NormW,
        k_norm: &NormW,
        kv: &Kv,
        q_width: u32,
    ) -> Val {
        symbol: "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        on: packed,
        weights: [q_norm.name, k_norm.name],
        layer: q_norm.layer,
        state: kv_state(kv),
        inputs: [packed],
        out: [Dim::Tokens, Dim::Const(q_width)] as BF16,
        made: "the fused post produces q",
    }
}

/// `rope::qk_rmsnorm_rope_bf16_rounded`: rounded per-head q/k norm + rope.
/// In-place on q and k; SSA records two fresh values.
pub fn qk_rmsnorm_rope_rounded(q: &Val, k: &Val, q_norm: &NormW, k_norm: &NormW) -> (Val, Val) {
    let shapes = {
        let b = q.t.inner.borrow();
        vec![
            (b.value_shape(q.id), DType::BF16),
            (b.value_shape(k.id), DType::BF16),
        ]
    };
    let ids = q.t.with(q_norm.layer, |b| {
        b.launch(
            "rope::qk_rmsnorm_rope_bf16_rounded",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            None,
            vec![q.id, k.id],
            shapes,
        )
    });
    let mk = |id| Val {
        t: q.t.clone(),
        id,
        layer: q_norm.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

builder! {
    /// `layout::transpose_bf16_nld_to_lnd`: `[N, L, D]` to `[L, N, D]` for contiguous layer slices.
    pub fn transpose_nld_to_lnd(x: &Val, layers: u32, dim: u32) -> Val {
        symbol: "layout::transpose_bf16_nld_to_lnd",
        on: x,
        layer: None,
        inputs: [x],
        out: [Dim::Const(layers), Dim::Tokens, Dim::Const(dim)] as BF16,
        made: "the relay produces its value",
    }
}

/// `moe::topk_softmax_bf16`: router top-k, softmax, renorm.
/// Returns `(expert_indices: [Tokens,k] i32, routing_weights: [Tokens,k] f32)`.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let ids = logits.t.with(logits.layer, |b| {
        b.launch(
            "moe::topk_softmax_bf16",
            vec![],
            None,
            vec![logits.id],
            vec![
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::I32),
                (Shape(vec![Dim::Tokens, Dim::Const(k)]), DType::F32),
            ],
        )
    });
    let mk = |id| Val {
        t: logits.t.clone(),
        id,
        layer: logits.layer,
    };
    (mk(ids[0]), mk(ids[1]))
}

/// Decode routed GEMVs over `N * k` routes.
/// Inputs are `[experts, x]`; output is `[Tokens, k, width]`.
pub fn moe_gate_up_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    moe_routed_gemv("moe::moe_gate_up_decode_gemv_bf16", x, w, experts, top_k)
}

pub fn moe_down_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    moe_routed_gemv("moe::moe_down_decode_gemv_bf16", x, w, experts, top_k)
}

fn moe_routed_gemv(kernel: &str, x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    record(
        &x.t,
        w.layer,
        kernel,
        vec![w.name.clone()],
        None,
        vec![experts.id, x.id],
        Some((
            Shape(vec![Dim::Tokens, Dim::Const(top_k), Dim::Const(w.width)]),
            DType::BF16,
        )),
    )
    .expect("a routed projection produces its value")
}

builder! {
    /// `moe::flashinfer_cutlass_moe_bf16`: fused decode MoE block.
    /// Consumes `(x, experts, weights)` and both expert banks; output is `[Tokens, hidden]`.
    pub fn moe_fused_cutlass(
        x: &Val,
        experts: &Val,
        weights: &Val,
        gate_up: &MatW,
        down: &MatW,
        hidden: u32,
    ) -> Val {
        symbol: "moe::flashinfer_cutlass_moe_bf16",
        on: x,
        weights: [gate_up.name, down.name],
        layer: gate_up.layer,
        inputs: [x, experts, weights],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the fused MoE produces its value",
    }


    /// `norm::residual_add_bf16`: explicit stream add when the producer wrote scratch.
    pub fn residual_add(x: &Val, residual: &Val, hidden: u32) -> Val {
        symbol: "norm::residual_add_bf16",
        on: x,
        inputs: [x, residual],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the residual add produces its value",
    }
}

/// Row RMSNorm variant selected by [`NormW`].
/// Per-head falls through to the semantic path because `head_dim` has no launch slot here.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    let id = x.t.with(w.layer, |b| match w.per_head {
        // Per-head is handle-selected; callers need not branch.
        Some(head_dim) => b.rmsnorm_per_head(x.id, &w.name, head_dim, w.variant),
        None => {
            let symbol = match w.variant {
                NormVariant::Gemma => "norm::rmsnorm_gemma_bf16",
                _ => "norm::rmsnorm_bf16",
            };
            let shape = b.value_shape(x.id);
            b.launch(
                symbol,
                vec![w.name.clone()],
                None,
                vec![x.id],
                vec![(shape, DType::BF16)],
            )[0]
        }
    });
    Val {
        t: x.t.clone(),
        id,
        layer: w.layer,
    }
}
