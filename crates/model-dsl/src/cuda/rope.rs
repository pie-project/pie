//! THE ROPE VARIANTS, and three small shapes that travel with them.

use super::*;

// ── the rope variants, and three small shapes ──────────────────

/// `kernels::rope::rope_yarn_bf16`: YaRN-scaled rope.
///
/// A different statement from [`rope_yarn_original`](crate::cuda::rope_yarn_original), not a
/// parameterization of it: the two interpolate frequencies differently,
/// and which a checkpoint wants is a load-time fact.
pub fn rope_yarn(q: &Val, k: &Val, q_width: u32) -> Val {
    record(
        &q.t,
        q.layer,
        "rope::rope_yarn_bf16",
        vec![],
        None,
        vec![q.id, k.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("the rope produces its value")
}

/// `kernels::rope::qk_rmsnorm_mrope_bf16`: per-head q/k norms and MROPE.
///
/// MROPE takes `[num_tokens, 3]` positions — a `(t, h, w)` triple rather
/// than one index — because a vision model's tokens sit in a grid. That
/// is why it cannot be the plain `qk_rmsnorm_rope` with a different
/// theta.
pub fn qk_rmsnorm_mrope(q: &Val, k: &Val, q_weight: &str, k_weight: &str, q_width: u32) -> Val {
    record(
        &q.t,
        q.layer,
        "rope::qk_rmsnorm_mrope_bf16",
        vec![q_weight.to_string(), k_weight.to_string()],
        None,
        vec![q.id, k.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("the norm+rope produces its value")
}

/// `kernels::quant::scale_rows_bf16`: scale each row by its own factor.
pub fn scale_rows(x: &Val, scale: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "quant::scale_rows_bf16",
        vec![],
        None,
        vec![x.id, scale.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the scale produces its value")
}

/// `kernels::quant::cast_fp32_to_bf16`: narrow.
pub fn cast_f32_to_bf16(x: &Val, width: u32) -> Val {
    record(
        &x.t,
        x.layer,
        "quant::cast_fp32_to_bf16",
        vec![],
        None,
        vec![x.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the cast produces its value")
}

/// `kernels::moe::apply_per_expert_scale_bf16`: multiply each route's
/// weight by its expert's scale, in place.
pub fn apply_per_expert_scale(topk_idx: &Val, topk_w: &Val, scale: &str, top_k: u32) -> Val {
    record(
        &topk_w.t,
        topk_w.layer,
        "moe::apply_per_expert_scale_bf16",
        vec![scale.to_string()],
        None,
        vec![topk_idx.id, topk_w.id],
        Some((Shape(vec![Dim::Tokens, Dim::Const(top_k)]), DType::F32)),
    )
    .expect("the scale produces its value")
}
