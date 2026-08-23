//! Qwen 3.5-only CUDA statements.

use super::*;

/// `kernels::moe::moe_grouped_gemm` — a `driver` routine (no generated
/// twin): the driver reaches it through `matmul_select`, and this is the
/// one statement form a text states for it.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn moe_grouped_gemm(
    act: &Val,
    expert_ids: &Val,
    stage: &Val,
    aligned: Dim,
    width: u32,
    bank: &str,
    block_size: u32,
    max_blocks: u32,
) -> Val {
    record_with_params(
        &act.t,
        act.layer,
        "moe::moe_grouped_gemm",
        vec![bank.to_string()],
        None,
        // params[0] = max_blocks, params[1] = m (the rows per block), which
        // is the routine's order — it was stated reversed.
        vec![max_blocks, block_size],
        // inputs[1] is per-block expert id; inputs[2] is the in-place
        // destination. Block-major rows, not tokens.
        vec![act.id, expert_ids.id, stage.id],
        Some((Shape(vec![aligned, Dim::Const(width)]), DType::BF16)),
    )
    .expect("the gemm produces its value")
}
