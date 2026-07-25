//! Name-pattern policy: tensor-parallel shard axes and runtime-quant
//! eligibility rules keyed off checkpoint tensor names.

use super::arch::arch_profile;
use super::*;

pub(super) fn llama_like_shard_axis(name: &str) -> Option<Axis> {
    if name.contains(".mlp.experts.") {
        if ends_with_any(
            name,
            &[
                ".gate_proj.weight_packed",
                ".gate_proj.weight_scale",
                ".up_proj.weight_packed",
                ".up_proj.weight_scale",
            ],
        ) {
            return Some(Axis(0));
        }
        if ends_with_any(
            name,
            &[".down_proj.weight_packed", ".down_proj.weight_scale"],
        ) {
            return Some(Axis(1));
        }
    }
    if ends_with_any(
        name,
        &[
            ".q_proj.weight",
            ".q_proj.bias",
            ".k_proj.weight",
            ".k_proj.bias",
            ".v_proj.weight",
            ".v_proj.bias",
            ".gate_proj.weight",
            ".up_proj.weight",
            ".sinks",
            ".w1.weight",
            ".w3.weight",
            ".w1.bias",
            ".w3.bias",
            ".q_proj.weight_scale",
            ".q_proj.weight_scale_inv",
            ".k_proj.weight_scale",
            ".k_proj.weight_scale_inv",
            ".v_proj.weight_scale",
            ".v_proj.weight_scale_inv",
            ".gate_proj.weight_scale",
            ".gate_proj.weight_scale_inv",
            ".up_proj.weight_scale",
            ".up_proj.weight_scale_inv",
            ".linear_attn.in_proj_z.weight",
            ".linear_attn.in_proj_b.weight",
            ".linear_attn.in_proj_a.weight",
            ".linear_attn.dt_bias",
            ".linear_attn.A_log",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_b_proj.weight",
        ],
    ) {
        Some(Axis(0))
    } else if ends_with_any(
        name,
        &[
            ".o_proj.weight",
            ".down_proj.weight",
            ".w2.weight",
            ".linear_attn.out_proj.weight",
        ],
    ) {
        Some(Axis(1))
    } else if name.ends_with(".experts.down_proj") || name.ends_with(".mlp.experts.down_proj") {
        Some(Axis(2))
    } else {
        None
    }
}

pub(super) fn runtime_quant_model_supported(model_type: &str, scheme: QuantScheme) -> bool {
    // Eligibility now lives in the ArchProfile registry (single source of truth).
    let profile = arch_profile(model_type);
    match scheme {
        QuantScheme::Mxfp4E2M1E8M0 => profile.mxfp4_runtime_quant,
        _ => profile.bf16_runtime_quant,
    }
}

pub(super) fn runtime_quantizable_name(name: &str, scheme: QuantScheme) -> bool {
    if scheme == QuantScheme::Mxfp4E2M1E8M0 {
        // For FP4 we only touch GLM-5.1's routed/shared experts. Attention
        // projections stay as FP8+scale (block-scaled GEMM) because there's
        // no FP4 GEMM path for them on this hardware.
        return is_glm_expert_weight(name);
    }
    ends_with_any(
        name,
        &[
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.o_proj.weight",
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_a_proj_with_mqa.weight",
            ".self_attn.kv_b_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
        ],
    ) || is_glm_expert_weight(name)
}

pub(super) fn is_glm_expert_weight(name: &str) -> bool {
    (name.contains(".mlp.experts.") || name.contains(".mlp.shared_experts."))
        && ends_with_any(
            name,
            &[".gate_proj.weight", ".up_proj.weight", ".down_proj.weight"],
        )
}

pub(super) fn dsv4_shard_axis(name: &str) -> Option<Axis> {
    // Routed experts: shard intermediate dim within each expert.
    // w1/w3 on axis 0 (gate/up out dim), w2 on axis 1 (down in dim).
    // Each rank computes a partial expert output; combined via all-reduce.
    if name.contains(".ffn.experts.") {
        if ends_with_any(
            name,
            &[".w1.weight", ".w1.scale", ".w3.weight", ".w3.scale"],
        ) {
            return Some(Axis(0));
        }
        if ends_with_any(name, &[".w2.weight", ".w2.scale"]) {
            return Some(Axis(1));
        }
    }
    // Shared experts: same column/row parallelism.
    if ends_with_any(
        name,
        &[
            ".shared_experts.w1.weight",
            ".shared_experts.w1.scale",
            ".shared_experts.w3.weight",
            ".shared_experts.w3.scale",
        ],
    ) {
        return Some(Axis(0));
    }
    if ends_with_any(
        name,
        &[".shared_experts.w2.weight", ".shared_experts.w2.scale"],
    ) {
        return Some(Axis(1));
    }
    // Everything else replicated (avoids TP communication in main path).
    None
}

pub(super) fn ends_with_any(value: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| value.ends_with(suffix))
}
