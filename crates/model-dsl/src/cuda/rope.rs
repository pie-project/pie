//! Rope variants and small adjacent CUDA shapes.

use super::*;


/// `kernels::rope::rope_yarn_bf16`.
/// `outputs[0]` = q, `outputs[1]` = k; shapes are read from inputs.
#[must_use]
pub fn rope_yarn(q: &Val, k: &Val) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch(
            "rope::rope_yarn_bf16",
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

/// `kernels::rope::qk_rmsnorm_mrope_bf16`.
/// MRoPE positions are `(t, h, w)`; `outputs[0]` = q, `outputs[1]` = k.
#[must_use]
pub fn qk_rmsnorm_mrope(q: &Val, k: &Val, q_weight: &str, k_weight: &str) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch(
            "rope::qk_rmsnorm_mrope_bf16",
            vec![q_weight.to_string(), k_weight.to_string()],
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

builder! {
    /// `kernels::quant::scale_rows_bf16`: scale each row by its own factor.
    pub fn scale_rows(x: &Val, scale: &Val, width: u32) -> Val {
        symbol: "quant::scale_rows_bf16",
        on: x,
        inputs: [x, scale],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the scale produces its value",
    }


    /// `kernels::quant::cast_fp32_to_bf16`: narrow.
    pub fn cast_f32_to_bf16(x: &Val, width: u32) -> Val {
        symbol: "quant::cast_fp32_to_bf16",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the cast produces its value",
    }


    /// `kernels::moe::apply_per_expert_scale_bf16`.
    pub fn apply_per_expert_scale(topk_idx: &Val, topk_w: &Val, scale: &str, top_k: u32) -> Val {
        symbol: "moe::apply_per_expert_scale_bf16",
        on: topk_w,
        weights: [scale],
        inputs: [topk_idx, topk_w],
        out: [Dim::Tokens, Dim::Const(top_k)] as F32,
        made: "the scale produces its value",
    }
}
