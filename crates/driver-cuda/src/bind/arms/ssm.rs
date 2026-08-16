//! What happens when a trace states one of `ssm`'s symbols.
//!
//! No arms: every row derives. The three that held longest were released by
//! the mamba aux slab becoming operands, and by the conv bias turning out to
//! be [`keys::NamedWeight2`](kernels::keys::NamedWeight2) — `spec.weight2`,
//! the statement's second name — rather than the `keys::WeightBias` suffix it
//! had been filed under.

use super::Bound;

/// Every symbol this family accounts for.
pub static ARMS: &[Bound] = &[
    Bound::derived("ssm::qwen_gdn_post_conv_prep_bf16"),
    Bound::derived("ssm::nemotron_mamba_split_bf16"),
    Bound::derived("ssm::nemotron_prepare_mamba_params"),
    Bound::derived("ssm::nemotron_prepare_mamba_dt_da"),
    Bound::derived("ssm::nemotron_mamba_ssm_batched_bf16"),
    Bound::derived("ssm::kda_gate_beta_bf16"),
    Bound::derived("ssm::kda_o_norm_gated_bf16"),
    Bound::derived("ssm::causal_conv1d_update_batched_bf16"),
    Bound::derived("ssm::causal_conv1d_prefill_batched_bf16"),
    Bound::derived("ssm::recurrent_gated_delta_step_batched"),
    Bound::derived("ssm::recurrent_gated_delta_step_batched_gqa"),
    Bound::derived("ssm::recurrent_gated_delta_step_batched_state_bf16"),
    Bound::derived("ssm::recurrent_gated_delta_step_batched_gqa_state_bf16"),
    Bound::derived("ssm::chunk_gated_delta_prefill_batched"),
    Bound::derived("ssm::chunk_gated_delta_prefill_batched_state_bf16"),
    Bound::derived("ssm::chunk_gated_delta_prefill_batched_cached"),
    Bound::derived("ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"),
    Bound::derived("ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"),
    Bound::derived("ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"),
    Bound::derived("ssm::repeat_interleave_heads_fp32"),
    Bound::derived("ssm::l2norm_scale_bf16_to_fp32"),
    Bound::derived("ssm::bf16_to_fp32"),
    Bound::derived("ssm::fp32_to_bf16"),
    Bound::derived("ssm::zamba_rmsnorm_gated_bf16"),
    Bound::derived("ssm::kda_recurrent_step_batched"),
    Bound::derived("ssm::kda_prefill_batched"),
    // A counting sort between statements builds these pointer arrays, so no
    // checkpoint carries them and no `Deployment` states them. The undeclared
    // results are the symptom, not a separate arity fix.
    Bound {
        symbol: "ssm::build_nemotron_moe_ptrs_aligned_bf16",
        arm: None,
        unbound: Some("the batched-GEMM pointer arrays, which are built between statements"),
    },
    Bound {
        symbol: "ssm::build_nemotron_moe_ptrs_decode_batched_bf16",
        arm: None,
        unbound: Some("the batched-GEMM pointer arrays, which are built between statements"),
    },
];
