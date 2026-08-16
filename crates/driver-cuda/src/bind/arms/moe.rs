//! What a trace that states one of `moe`'s symbols binds to.
//!
//! A column that resolves at every position can still bind the WRONG buffer,
//! and nothing detects it: an arity check counts pointers and both spellings
//! are one pointer. Hence the rows below that STATE their operand indices.

use super::Bound;

/// Every symbol this family binds.
pub static ARMS: &[Bound] = &[
    // Stated indices: derivation would put `topk_w` (scaled in place) at
    // `Out(0)` and `per_expert_scale` at `In(1)`, which is `topk_w`'s own bytes.
    Bound::derived("moe::apply_per_expert_scale_bf16"),
    // `Or(Weight(0), Lit(Null))`: this bias is the POSITIONAL run `Cx::weight(0)`
    // indexes, not the model text's named bank.
    Bound::derived("moe::topk_sqrtsoftplus_bf16"),
    // Its bias is REQUIRED: a null here is a fault, not an absence.
    Bound::derived("moe::hash_route_lookup"),
    // One fact spelled `normalize` here and three other ways across this file,
    // so it needs a `Source` rather than an entry keyed on the parameter name.
    Bound::derived("moe::topk_sigmoid_bias_fp32"),
    // `num_tokens_past_padded` is `Lit(Null)`: `moe_dispatch.cuh` guards every
    // write to that pointer with `!= nullptr`.
    Bound::derived("moe::moe_align_decode"),
    // `rows.count` is the ALIGNED row count here, so the fan-out is in the
    // RESULT: `rows * top_k` is `OutElements(0) / InWidth(0)`.
    // `DispatchCtx::experts_per_token` is structurally zero on a CUDA plan.
    Bound::derived("moe::reorder_moe_aligned_output_bf16"),
    // The `Or` is load-bearing: `dsl::cuda::topk_sigmoid` states no `weights:`,
    // so `Weight(0)` refuses on every real fire and the null binds.
    Bound::derived("moe::topk_sigmoid_bf16"),
    Bound::derived("moe::topk_softmax_bf16"),
    Bound::derived("moe::moe_gate_up_decode_gemv_bf16"),
    Bound::derived("moe::moe_down_decode_gemv_bf16"),
    Bound::derived("moe::token_batched_weighted_sum_bf16"),
    // Its `in_place = &[(0, 2)]` moves nothing: a stated index is not something
    // a remap may touch.
    Bound::derived("moe::token_batched_weighted_sum_add_bf16"),
    // The pitch is `Param<0, i32>` because no `Source` reaches a pitch, and
    // `bias` is `Bank<0, T>` because this statement's input 0 is `x` -- an
    // `In<0, T>` would hand the kernel the activation rectangle.
    Bound::derived("moe::add_moe_route_bias_bf16"),
    Bound {
        symbol: "moe::transpose_expert_scales_u8",
        arm: None,
        unbound: Some(
            "a caller in the weight loader: this rewrites checkpoint planes before any fire exists",
        ),
    },
    // The inverse map is not optional: its store has no null guard where the
    // padded twin's does.
    Bound::derived("moe::moe_bucket_exact"),
];
