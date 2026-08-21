//! GEMMA — 3n's AltUp, and gemma-4.

use super::*;

// Statements here are kept in algorithm order; `builder!` only covers
// records whose result shape can be stated from dims.
// AltUp carries rank-K residual streams, so its explicit shapes stay.
builder! {
    /// `norm::altup_predict`: predicted K-stream post-layer state.
    /// `coefs` stays fp32 for the K-sum.
    pub fn altup_predict(streams: &Val, coefs: &Val, k: u32, hidden: u32) -> Val {
        symbol: "norm::altup_predict",
        on: streams,
        inputs: [streams, coefs],
        out: [Dim::Const(k), Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the prediction produces its value",
    }


    /// `norm::altup_correct`: correct the K-1 inactive streams.
    /// The `+1` is already folded into `correction_coefs`; `params[0]` says
    /// which stream is the active one.
    pub fn altup_correct(
        predictions: &Val,
        activated: &Val,
        correction_coefs: &Val,
        k: u32,
        hidden: u32,
        active_idx: u32,
    ) -> Val {
        symbol: "norm::altup_correct",
        on: predictions,
        params: [active_idx],
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


    /// `norm::mean_streams`: average K residual streams into one.
    pub fn mean_streams(streams: &Val, hidden: u32) -> Val {
        symbol: "norm::mean_streams",
        on: streams,
        inputs: [streams],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the mean produces its value",
    }


    /// `norm::compute_rms`: row RMS measurement for [`magnitude_rescale`].
    pub fn compute_rms(x: &Val) -> Val {
        symbol: "norm::compute_rms",
        on: x,
        inputs: [x],
        out: [Dim::Tokens] as F32,
        made: "the measurement produces its value",
    }


    /// `norm::magnitude_rescale`: scale each row of `x` to `target_rms`.
    /// The kernel is in-place; the trace still records the produced value.
    pub fn magnitude_rescale(x: &Val, target_rms: &Val, hidden: u32) -> Val {
        symbol: "norm::magnitude_rescale",
        on: x,
        inputs: [x, target_rms],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the rescale produces its value",
    }
}

/// `norm::tanh` on AltUp's modality-router output.
/// The output shape is copied from the operand; respelling it once disagreed
/// with a selected stream slice, and the in-place row exposed the alias.
pub fn tanh(x: &Val) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record(
        &x.t,
        x.layer,
        "norm::tanh",
        vec![],
        None,
        vec![x.id],
        Some(out),
    )
    .expect("the activation produces its value")
}

builder! {
    /// `mlp::gaussian_topk`: zero values below `mean + m·std` per row.
    /// `params[0]` is `std_multiplier` as `f32` bits; no driver table supplies it.
    pub fn gaussian_topk(x: &Val, width: u32, std_multiplier: f32) -> Val {
        symbol: "mlp::gaussian_topk",
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
            "mlp::chunked_geglu_tanh"
        } else {
            "mlp::geglu_tanh"
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
    /// `mlp::geglu_tanh` pair form: gate and up are separate operands.
    /// Used when PLE supplies `up` from a per-layer table, not the packed bank.
    pub fn geglu_tanh_pair(gate: &Val, up: &Val, width: u32) -> Val {
        symbol: "mlp::geglu_tanh",
        on: gate,
        inputs: [gate, up],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the activation produces its value",
    }
}

/// `rope::rope_partial_q_bf16`: rotate Q only; shared K was cached at its source.
/// Separate symbol: operand count must not encode the operation.
/// `[rotary_dim, head_dim, theta]` is the run; positions minted by name.
pub fn rope_partial_q_only(q: &Val, rotary_dim: u32, head_dim: u32, theta: f32) -> Val {
    let positions = rt_tokens(&q.t, "positions");
    let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
    record_with_params(
        &q.t,
        q.layer,
        "rope::rope_partial_q_bf16",
        vec![],
        None,
        vec![rotary_dim, head_dim, theta.to_bits()],
        vec![q.id, positions],
        Some(out),
    )
    .expect("the rotation produces its value")
}

/// [`qk_rmsnorm_rope_rounded`] with K absent, as one fused launch.
/// The driver passes `k_norm = nullptr` and `num_kv_heads = 0`; separate symbol, shared C++.
/// `[head_dim, theta, eps]` is the run; the handle carries dim and epsilon.
pub fn qk_rmsnorm_rope_rounded_q_only(q: &Val, q_norm: &NormW, theta: f32) -> Val {
    let positions = rt_tokens(&q.t, "positions");
    let head_dim = q_norm
        .per_head
        .expect("a per-head q norm carries its head dim");
    let out = (q.t.inner.borrow().value_shape(q.id), DType::BF16);
    record_with_params(
        &q.t,
        q_norm.layer,
        "rope::q_rmsnorm_rope_bf16_rounded",
        vec![q_norm.name.clone()],
        None,
        vec![head_dim, theta.to_bits(), q_norm.eps.to_bits()],
        vec![q.id, positions],
        Some(out),
    )
    .expect("the fused pair produces q")
}

/// `norm::rmsnorm_no_scale`: per-head `v / rms(v)` with no learnable weight.
/// `[per_head_dim, eps]` is the run — no handle carries them here, so the
/// caller does.
pub fn rmsnorm_no_scale(x: &Val, per_head_dim: u32, eps: f32) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record_with_params(
        &x.t,
        x.layer,
        "norm::rmsnorm_no_scale",
        vec![],
        None,
        vec![per_head_dim, eps.to_bits()],
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
            // `[scale, eps]` — the epsilon rides the norm handle.
            vec![scale.to_bits(), w.eps.to_bits()],
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
    /// `norm::rmsnorm_residual_add`: post-FFN norm plus residual add.
    /// The epsilon rides the handle and the params run.
    pub fn norm_residual_add(x: &Val, y: &Val, w: &NormW, hidden: u32) -> Val {
        symbol: "norm::rmsnorm_residual_add",
        on: x,
        weights: [w.name],
        layer: w.layer,
        params: [w.eps.to_bits()],
        // `y` is the residual stream; naming it preserves the SSA edge.
        inputs: [x, y],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the fused norm+residual produces its value",
    }
}

/// `norm::scalar_mul`: multiply by a named load-time constant.
/// The weight name is `scale.<name>`; if present, `params[0]` is `by` as `f32` bits.
/// `None` means this family has not derived the number and falls through to a handwritten arm.
pub fn scalar_mul(x: &Val, scale: &str, by: Option<f32>) -> Val {
    let out = (x.t.inner.borrow().value_shape(x.id), DType::BF16);
    record_with_params(
        &x.t,
        x.layer,
        "norm::scalar_mul",
        vec![format!("scale.{scale}")],
        None,
        by.map(f32::to_bits).into_iter().collect(),
        vec![x.id],
        Some(out),
    )
    .expect("the scale produces its value")
}

builder! {
    /// `attn::logit_softcap`: `cap * tanh(x / cap)`, present only when traced.
    /// The cap is the checkpoint's and rides the run.
    pub fn logit_softcap(x: &Val, vocab: u32, cap: f32) -> Val {
        symbol: "attn::logit_softcap",
        on: x,
        layer: None,
        params: [cap.to_bits()],
        inputs: [x],
        out: [Dim::Requests, Dim::Const(vocab)] as BF16,
        made: "the softcap produces its value",
    }


}

/// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`: split packed qkv, norm q/k/v,
/// rope q/k, write k/v to cache, and return q. The KV view, positions and
/// row validity are operands; `[num_kv_heads, head_dim, theta, eps]` the run.
pub fn qkv_packed_post(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    q_width: u32,
    num_kv_heads: u32,
    theta: f32,
) -> Val {
    let kvc = rt_object(&packed.t, "kv_cache", Some(kv.l));
    let positions = rt_tokens(&packed.t, "positions");
    let row_valid = rt_tokens(&packed.t, "row_valid");
    let head_dim = q_norm
        .per_head
        .expect("a per-head q norm carries its head dim");
    record_with_params(
        &packed.t,
        q_norm.layer,
        "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
        vec![
            num_kv_heads,
            head_dim,
            theta.to_bits(),
            q_norm.eps.to_bits(),
        ],
        vec![packed.id, kvc, positions, row_valid],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("the fused post produces q")
}

/// `rope::qk_rmsnorm_rope_bf16_rounded`: rounded per-head q/k norm + rope.
/// In-place on q and k; SSA records two fresh values.
/// `[head_dim, theta, eps]` is the run; positions minted by name.
pub fn qk_rmsnorm_rope_rounded(
    q: &Val,
    k: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    theta: f32,
) -> (Val, Val) {
    let head_dim = q_norm
        .per_head
        .expect("a per-head q norm carries its head dim");
    let shapes = {
        let b = q.t.inner.borrow();
        vec![
            (b.value_shape(q.id), DType::BF16),
            (b.value_shape(k.id), DType::BF16),
        ]
    };
    let ids = q.t.with(q_norm.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        b.launch_with_params(
            "rope::qk_rmsnorm_rope_bf16_rounded",
            vec![q_norm.name.clone(), k_norm.name.clone()],
            None,
            vec![head_dim, theta.to_bits(), q_norm.eps.to_bits()],
            vec![q.id, k.id, positions],
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
        params: [dim],
        inputs: [x],
        out: [Dim::Const(layers), Dim::Tokens, Dim::Const(dim)] as BF16,
        made: "the relay produces its value",
    }
}

/// `moe::topk_softmax`: router top-k, softmax, renorm.
/// Returns `(expert_indices: [Tokens,k] i32, routing_weights: [Tokens,k] f32)`.
pub fn topk(logits: &Val, k: u32) -> (Val, Val) {
    let ids = logits.t.with(logits.layer, |b| {
        b.launch(
            "moe::topk_softmax",
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
    moe_routed_gemv("moe::moe_gate_up_decode_gemv", x, w, experts, top_k)
}

pub fn moe_down_gemv(x: &Val, w: &MatW, experts: &Val, top_k: u32) -> Val {
    moe_routed_gemv("moe::moe_down_decode_gemv", x, w, experts, top_k)
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


    /// `norm::residual_add`: explicit stream add when the producer wrote scratch.
    pub fn residual_add(x: &Val, residual: &Val, hidden: u32) -> Val {
        symbol: "norm::residual_add",
        on: x,
        inputs: [x, residual],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the residual add produces its value",
    }
}

/// Row RMSNorm variant selected by [`NormW`]. One statement either way:
/// params `[per_head_dim, eps]` per the swept signature, `0` for whole-row.
pub fn rmsnorm(x: &Val, w: &NormW) -> Val {
    crate::rmsnorm(x, w)
}
