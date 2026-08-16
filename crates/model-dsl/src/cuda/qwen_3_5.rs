//! Qwen 3.5-only CUDA statements.

use super::*;




builder! {
    /// `kernels::attn::mtp_shift_hidden_bf16`.
    pub fn mtp_shift_hidden(target: &Val, pending: &Val, hidden: u32) -> Val {
        symbol: "attn::mtp_shift_hidden_bf16",
        on: target,
        inputs: [target, pending],
        out: [Dim::Tokens, Dim::Const(hidden)] as BF16,
        made: "the shift produces its value",
    }
}

/// `kernels::attn::mtp_update_pending_hidden_bf16`.
pub fn mtp_update_pending_hidden(target: &Val, l: u32) {
    record(
        &target.t,
        Some(l),
        "attn::mtp_update_pending_hidden_bf16",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![target.id],
        None,
    );
}



builder! {
    /// `kernels::norm::rmsnorm_gated_bf16`.
    pub fn rmsnorm_gated_launch(x: &Val, gate: &Val, weight: &str, width: u32) -> Val {
        symbol: "norm::rmsnorm_gated_bf16",
        on: x,
        weights: [weight],
        inputs: [x, gate],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the norm produces its value",
    }


    /// Emits token ids directly, not logits.
    pub fn lm_head_gemv_argmax_int8(x: &Val, weight: &str, scale: &str) -> Val {
        symbol: "sample::lm_head_gemv_argmax_int8",
        on: x,
        weights: [weight, scale],
        layer: None,
        inputs: [x],
        out: [Dim::Requests] as I32,
        made: "the readout produces its value",
    }


    /// `kernels::moe::moe_grouped_gemm_bf16`.
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
        symbol: "moe::moe_grouped_gemm_bf16",
        on: act,
        weights: [bank],
        // params[0] = block_size, params[1] = max_blocks.
        params: [block_size, max_blocks],
        // inputs[1] is per-block expert id; inputs[2] is the in-place destination.
        inputs: [act, expert_ids, stage],
        // Block-major rows, not tokens.
        out: [aligned, Dim::Const(width)] as BF16,
        made: "the gemm produces its value",
    }
}
