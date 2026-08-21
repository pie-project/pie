//! Rope variants and small adjacent CUDA shapes.

use super::*;


/// `kernels::rope::rope_yarn_bf16`.
/// `outputs[0]` = q, `outputs[1]` = k; shapes are read from inputs.
///
/// The llama-3 ramp's numbers ride the params run in the routine's order:
/// `[factor, low_freq_factor, high_freq_factor, original_max_position,
/// num_q_heads, num_kv_heads, head_dim, theta]`; positions minted by name.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn(
    q: &Val,
    k: &Val,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    theta: f32,
) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch_with_params(
            "rope::rope_yarn_bf16",
            vec![],
            None,
            vec![
                factor.to_bits(),
                low_freq_factor.to_bits(),
                high_freq_factor.to_bits(),
                original_max_position,
                num_q_heads,
                num_kv_heads,
                head_dim,
                theta.to_bits(),
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

/// `kernels::rope::qk_rmsnorm_mrope_bf16`.
/// MRoPE positions are `(t, h, w)`; `outputs[0]` = q, `outputs[1]` = k.
///
/// The run, in the routine's order: `[mrope_section_t, mrope_section_h,
/// mrope_section_w, num_q_heads, num_kv_heads, head_dim, theta, eps]`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn qk_rmsnorm_mrope(
    q: &Val,
    k: &Val,
    q_weight: &str,
    k_weight: &str,
    mrope_section_t: u32,
    mrope_section_h: u32,
    mrope_section_w: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    theta: f32,
    eps: f32,
) -> (Val, Val) {
    let ids = q.t.with(q.layer, |b| {
        let positions =
            b.runtime_tensor("positions", None, Shape(vec![Dim::Tokens]), DType::I32);
        let q_sh = b.value_shape(q.id);
        let k_sh = b.value_shape(k.id);
        b.launch_with_params(
            "rope::qk_rmsnorm_mrope_bf16",
            vec![q_weight.to_string(), k_weight.to_string()],
            None,
            vec![
                mrope_section_t,
                mrope_section_h,
                mrope_section_w,
                num_q_heads,
                num_kv_heads,
                head_dim,
                theta.to_bits(),
                eps.to_bits(),
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

builder! {
    /// `kernels::quant::scale_rows`: scale each row by its own factor.
    pub fn scale_rows(x: &Val, scale: &Val, width: u32) -> Val {
        symbol: "quant::scale_rows",
        on: x,
        inputs: [x, scale],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the scale produces its value",
    }


    /// `kernels::quant::cast_fp32_to`: narrow.
    pub fn cast_f32_to_bf16(x: &Val, width: u32) -> Val {
        symbol: "quant::cast_fp32_to",
        on: x,
        inputs: [x],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the cast produces its value",
    }


    /// `kernels::moe::apply_per_expert_scale`.
    pub fn apply_per_expert_scale(topk_idx: &Val, topk_w: &Val, scale: &str, top_k: u32) -> Val {
        symbol: "moe::apply_per_expert_scale",
        on: topk_w,
        weights: [scale],
        inputs: [topk_idx, topk_w],
        out: [Dim::Tokens, Dim::Const(top_k)] as F32,
        made: "the scale produces its value",
    }
}
