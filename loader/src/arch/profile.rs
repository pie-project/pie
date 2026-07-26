//! Per-architecture facts table: which model families need which
//! detectors, source prefixes, TP quirks and runtime-quant support.

use super::policy::{dsv4_shard_axis, llama_like_shard_axis};
use super::*;

/// Declarative per-architecture facts. Centralizes the `model_type` knowledge
/// that was previously scattered across the ABI builder (detector gating,
/// source-name prefix, TP shard quirks, runtime-quant eligibility) so adding a
/// model is a single table edit instead of new `matches!(model_type, …)` checks
/// in half a dozen places. `arch_profile()` returns the matching profile, or a
/// generic-dense default (all flags off).
#[derive(Clone, Copy, Debug)]
pub(super) struct ArchProfile {
    /// Source tensors live under this prefix; stripped from output names.
    pub(super) source_prefix: Option<&'static str>,
    /// MLA attention: fuse q_a + kv_a and shared gate+up (DeepSeek/Kimi).
    pub(super) mla_fused_joins: bool,
    /// Phi-3 fused qkv / gate_up splits.
    pub(super) phi3_fused_splits: bool,
    /// Nemotron-H packed-expert views.
    pub(super) nemotron_packed_experts: bool,
    /// GPT-OSS native MXFP4 expert groups.
    pub(super) gpt_oss_mxfp4_groups: bool,
    /// Shard `embed_tokens` on axis 0 under tensor parallelism.
    pub(super) shard_embed_tokens: bool,
    /// Keep `lm_head` replicated under TP (don't shard).
    pub(super) replicate_lm_head: bool,
    /// FP4/MXFP4 runtime quant of routed experts is wired for this arch.
    pub(super) mxfp4_runtime_quant: bool,
    /// BF16 -> FP8/INT8 runtime quant is supported for this arch.
    pub(super) bf16_runtime_quant: bool,
    /// Skip the dense fused-QKV projection join. Qwen3-MoE / Qwen3.5-MoE bind
    /// attention q/k/v as separate tensors and never read a fused qkv, so the
    /// generic join would consume q_proj/k_proj/v_proj and leave the bind path
    /// with a `missing weight 'self_attn.q_proj.weight'` error.
    pub(super) skip_dense_qkv_fusion: bool,
    /// Stack per-expert MoE weights (`experts.{i}.{gate,up,down}_proj`) into the
    /// fused 3-D `experts.gate_up_proj` / `experts.down_proj` tensors the
    /// qwen3_5_moe forward expects. Plain Qwen3-MoE (Qwen3-30B-A3B) ships
    /// per-expert weights; qwen3_5_moe checkpoints ship pre-fused (then a no-op).
    pub(super) stack_per_expert_moe: bool,
    /// The canonical Metal Qwen3.5/GDN storage schema is defined for this arch.
    pub(super) metal_qwen35: bool,
    /// Tensor-parallel shard-axis strategy, keyed by tensor name. Defaults to the
    /// llama-family rules; archs with a different expert/attention layout (e.g.
    /// DeepSeek-V4's native `.ffn.experts.w*`) register their own function.
    pub(super) shard_axis_fn: fn(&str) -> Option<Axis>,
}

pub(super) const GENERIC_ARCH: ArchProfile = ArchProfile {
    source_prefix: None,
    mla_fused_joins: false,
    phi3_fused_splits: false,
    nemotron_packed_experts: false,
    gpt_oss_mxfp4_groups: false,
    shard_embed_tokens: false,
    replicate_lm_head: false,
    mxfp4_runtime_quant: false,
    bf16_runtime_quant: false,
    skip_dense_qkv_fusion: false,
    stack_per_expert_moe: false,
    metal_qwen35: false,
    shard_axis_fn: llama_like_shard_axis,
};

/// (matching model_type strings, profile). Matched case-insensitively. The
/// generic-dense fallback (`GENERIC_ARCH`) covers llama-family models that need
/// no special handling beyond the name-pattern detectors that always run.
pub(super) const ARCH_PROFILES: &[(&[&str], ArchProfile)] = &[
    (
        &["kimi_k2", "kimi_k25"],
        ArchProfile {
            source_prefix: Some("language_model."),
            mla_fused_joins: true,
            shard_embed_tokens: true,
            replicate_lm_head: true,
            ..GENERIC_ARCH
        },
    ),
    (
        &["deepseek_v2", "deepseek_v3"],
        ArchProfile {
            mla_fused_joins: true,
            ..GENERIC_ARCH
        },
    ),
    (
        // Native `.ffn.experts.w*` naming + per-expert intermediate sharding.
        &["deepseek_v4"],
        ArchProfile {
            shard_axis_fn: dsv4_shard_axis,
            ..GENERIC_ARCH
        },
    ),
    (
        &["phi3"],
        ArchProfile {
            phi3_fused_splits: true,
            ..GENERIC_ARCH
        },
    ),
    (
        &["nemotron_h"],
        ArchProfile {
            nemotron_packed_experts: true,
            ..GENERIC_ARCH
        },
    ),
    (
        // GPT-OSS binds attention q/k/v separately (its CUDA builder reads
        // `self_attn.q_proj.weight` / `k_proj` / `v_proj`, never a fused qkv), so
        // opt out of the dense projection join like Qwen3-MoE — otherwise the
        // join consumes q/k/v into `qkv_proj.fused.weight` and the bind path
        // fails with `missing weight 'self_attn.q_proj.weight'`.
        &["gpt_oss", "gpt-oss", "gptoss"],
        ArchProfile {
            gpt_oss_mxfp4_groups: true,
            skip_dense_qkv_fusion: true,
            ..GENERIC_ARCH
        },
    ),
    (
        &["glm_moe_dsa"],
        ArchProfile {
            shard_embed_tokens: true,
            mxfp4_runtime_quant: true,
            bf16_runtime_quant: true,
            ..GENERIC_ARCH
        },
    ),
    (
        // Plain Qwen3-MoE (Qwen3-30B-A3B) and Qwen3.5-MoE: their bind path reads
        // attention q/k/v separately and never a fused qkv, so opt out of the
        // dense projection join that would otherwise consume q_proj/k_proj/v_proj.
        &["qwen3_moe", "qwen3_5_moe", "qwen3_5_moe_text"],
        ArchProfile {
            bf16_runtime_quant: true,
            skip_dense_qkv_fusion: true,
            stack_per_expert_moe: true,
            ..GENERIC_ARCH
        },
    ),
    (
        &[
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_next",
            "qwen3_next_text",
            "qwen3_6",
        ],
        ArchProfile {
            bf16_runtime_quant: true,
            metal_qwen35: true,
            ..GENERIC_ARCH
        },
    ),
    (
        &["qwen3", "qwen2", "llama", "llama3", "mistral"],
        ArchProfile {
            bf16_runtime_quant: true,
            ..GENERIC_ARCH
        },
    ),
];

pub(super) fn arch_profile(model_type: &str) -> ArchProfile {
    for (names, profile) in ARCH_PROFILES {
        if names.iter().any(|n| model_type.eq_ignore_ascii_case(n)) {
            return *profile;
        }
    }
    GENERIC_ARCH
}
