//! Qwen 3.5-only CUDA statements.

use super::*;




/// `kernels::attn::mtp_shift_hidden`.
/// The qo CSR and the recurrent view are operands after the hidden pair;
/// the request count is the CSR operand's own row count, so the run is empty.
#[must_use]
pub fn mtp_shift_hidden(target: &Val, pending: &Val, hidden: u32, l: u32) -> Val {
    let qo_indptr = rt_requests(&target.t, "qo_indptr");
    let rsv = rt_object(&target.t, "recurrent_state", Some(l));
    record_with_extents(
        &target.t,
        Some(l),
        "attn::mtp_shift_hidden",
        vec![],
        None,
        vec![],
        vec![],
        vec![target.id, pending.id, qo_indptr, rsv],
        Some((Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)),
    )
    .expect("the shift produces its value")
}

/// `kernels::attn::mtp_update_pending_hidden`.
/// Operands: the target hidden, the qo CSR, the recurrent view and the
/// MTP pending-hidden slab; the request count is the CSR operand's own
/// row count, so the run is empty.
pub fn mtp_update_pending_hidden(target: &Val, l: u32) {
    let qo_indptr = rt_requests(&target.t, "qo_indptr");
    let rsv = rt_object(&target.t, "recurrent_state", Some(l));
    let pending = rt_object(&target.t, "mtp.pending_hidden", None);
    record_with_extents(
        &target.t,
        Some(l),
        "attn::mtp_update_pending_hidden",
        vec![],
        Some(StateRef {
            store: StateStore::RecurrentState,
            layer: l,
        }),
        vec![],
        vec![],
        vec![target.id, qo_indptr, rsv, pending],
        None,
    );
}



builder! {
    /// `kernels::norm::rmsnorm_gated`; `[per_head_dim, eps]` is the run.
    pub fn rmsnorm_gated_launch(
        x: &Val,
        gate: &Val,
        weight: &str,
        width: u32,
        per_head_dim: u32,
        eps: f32,
    ) -> Val {
        symbol: "norm::rmsnorm_gated",
        on: x,
        weights: [weight],
        params: [per_head_dim, eps.to_bits()],
        inputs: [x, gate],
        out: [Dim::Tokens, Dim::Const(width)] as BF16,
        made: "the norm produces its value",
    }


    /// Emits token ids directly, not logits; the vocab rides the run.
    pub fn lm_head_gemv_argmax_int8(x: &Val, weight: &str, scale: &str, vocab: u32) -> Val {
        symbol: "sample::lm_head_gemv_argmax_int8",
        on: x,
        weights: [weight, scale],
        layer: None,
        params: [vocab],
        inputs: [x],
        out: [Dim::Requests] as I32,
        made: "the readout produces its value",
    }


    /// `kernels::moe::moe_grouped_gemm`.
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
        symbol: "moe::moe_grouped_gemm",
        on: act,
        weights: [bank],
        // params[0] = max_blocks, params[1] = m (the rows per block), which
        // is the routine's order — it was stated reversed.
        params: [max_blocks, block_size],
        // inputs[1] is per-block expert id; inputs[2] is the in-place destination.
        inputs: [act, expert_ids, stage],
        // Block-major rows, not tokens.
        out: [aligned, Dim::Const(width)] as BF16,
        made: "the gemm produces its value",
    }
}
