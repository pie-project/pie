//! Mixture paths: WNA16, MXFP4, nemotron_h dispatch, and aligned dispatch.
use super::*;

// ── mixtral / gpt-oss: MXFP4 MoE ──────────────────────────────
// Some statements are weight-shaped launches with no token extent.

// ── MoE: aligned dispatch path ────────────────────────────────
// Routes are bucketed and padded globally; whole statements use route order from the full fire.

/// `moe::gather_moe_aligned_inputs`: gather block-major operands.
/// `params = [top_k, tokens]`: `top_k` because no operand/result carries k,
/// `tokens` because the launch's rows are the STACK's — the fire's token
/// count is spliced as an extent.
#[must_use]
pub fn gather_moe_aligned_inputs(
    x: &Val,
    sorted_route_ids: &Val,
    aligned: Dim,
    hidden: u32,
    top_k: u32,
) -> Val {
    record_with_extents(
        &x.t,
        x.layer,
        "moe::gather_moe_aligned_inputs",
        vec![],
        None,
        vec![top_k, 0],
        vec![tokens_extent(1)],
        vec![x.id, sorted_route_ids.id],
        Some((Shape(vec![aligned, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the gather produces its value")
}

/// `moe::build_moe_ptrs_aligned_bf16`: declare staging buffers and build
/// grouped-GEMM pointers — untraced (no generated twin). Outputs
/// `(gate_up, act, out)` as `[aligned, 2I]`, `[aligned, I]`, `[aligned, H]`.
/// Pointer arrays remain driver-owned; this vocabulary has no dtype for them.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn build_moe_ptrs_aligned(
    expert_ids: &Val,
    aligned_in: &Val,
    l: u32,
    gate_up_bank: &str,
    down_bank: &str,
    aligned: Dim,
    hidden: u32,
    moe_intermediate: u32,
) -> (Val, Val, Val) {
    let outs = record_many(
        &expert_ids.t,
        Some(l),
        "moe::build_moe_ptrs_aligned_bf16",
        vec![gate_up_bank.to_string(), down_bank.to_string()],
        vec![expert_ids.id, aligned_in.id],
        vec![
            (
                Shape(vec![aligned, Dim::Const(2 * moe_intermediate)]),
                DType::BF16,
            ),
            (
                Shape(vec![aligned, Dim::Const(moe_intermediate)]),
                DType::BF16,
            ),
            (Shape(vec![aligned, Dim::Const(hidden)]), DType::BF16),
        ],
    );
    let mut it = outs.into_iter();
    let mut next = || it.next().expect("the ptr build states three stages");
    let gate_up = next();
    let act = next();
    let out = next();
    (gate_up, act, out)
}
