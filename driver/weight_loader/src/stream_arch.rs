//! Arch-specific [`StreamArchDesc`] plugins for SSD expert streaming.
//!
//! Generic plan construction lives in [`crate::stream`]. This module owns
//! checkpoint naming, section order, and binding-grid collectors, registered
//! on each model's `ArchProfile` in [`crate::abi`].

use crate::error::CompileError;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::StreamBinding;
use crate::stream::{StreamArchDesc, collect_bindings_from_named_tensors};

/// Fixed DSv4 section order — must match `dsv4_expert_sections.hpp` in the
/// CUDA driver (`w1/w2/w3` × weight/scale).
pub const DSV4_EXPERT_SECTIONS: &[&str] = &[
    "w1.weight",
    "w1.scale",
    "w2.weight",
    "w2.scale",
    "w3.weight",
    "w3.scale",
];

/// DeepSeek-V4 main-stack routed experts:
/// `layers.{L}.ffn.experts.{E}.{w1,w2,w3}.{weight,scale}`.
///
/// Shared experts (`.ffn.shared_experts.`), routers (`ffn.gate.*`), and MTP
/// modules (`mtp.*.ffn.experts.*`) are **not** streamable — only the primary
/// layer MoE bank is paged by the expert stream cache today.
pub(crate) fn is_dsv4_routed_expert_tensor(name: &str) -> bool {
    // Require the main-stack prefix so MTP / other modules stay resident.
    let Some(rest) = name.strip_prefix("layers.") else {
        return false;
    };
    let Some((_, rest)) = rest.split_once('.') else {
        return false;
    };
    rest.starts_with("ffn.experts.")
        && ends_with_any(
            name,
            &[
                ".w1.weight",
                ".w1.scale",
                ".w2.weight",
                ".w2.scale",
                ".w3.weight",
                ".w3.scale",
            ],
        )
}

/// Parse `layers.{L}.ffn.experts.{E}.{section}` → (layer, expert, section_idx).
pub(crate) fn parse_dsv4_expert_section(name: &str) -> Option<(u32, u32, usize)> {
    let rest = name.strip_prefix("layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("ffn.experts.")?;
    let (expert_str, section) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;
    let expert: u32 = expert_str.parse().ok()?;
    let section_idx = DSV4_EXPERT_SECTIONS.iter().position(|s| *s == section)?;
    Some((layer, expert, section_idx))
}

fn dsv4_collect_bindings(
    metadata: &CheckpointMetadata,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    collect_bindings_from_named_tensors(
        metadata,
        num_layers,
        num_experts,
        DSV4_EXPERT_SECTIONS.len(),
        is_dsv4_routed_expert_tensor,
        parse_dsv4_expert_section,
    )
}

pub(crate) const DSV4_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: DSV4_EXPERT_SECTIONS,
    is_streamed: is_dsv4_routed_expert_tensor,
    collect_bindings: dsv4_collect_bindings,
};

/// Fixed GPT-OSS section order — must match `gpt_oss_expert_sections.hpp`.
/// Biases stay resident and are not part of the stream plan.
pub const GPT_OSS_EXPERT_SECTIONS: &[&str] = &[
    "gate_up.weight",
    "gate_up.scale",
    "down.weight",
    "down.scale",
];

/// GPT-OSS fused MXFP4 expert packs/scales (not biases).
/// Checkpoint names: `…mlp.experts.{gate_up,down}_proj_{blocks,scales}`.
pub(crate) fn is_gpt_oss_streamed_expert_tensor(name: &str) -> bool {
    let Some(rest) = name.split_once("mlp.experts.").map(|(_, r)| r) else {
        return false;
    };
    matches!(
        rest,
        "gate_up_proj_blocks"
            | "gate_up_proj_scales"
            | "down_proj_blocks"
            | "down_proj_scales"
    )
}

fn gpt_oss_find_tensor<'a>(
    metadata: &'a CheckpointMetadata,
    name: &str,
) -> Result<&'a RawTensor, CompileError> {
    metadata
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "stream_routed_experts: missing GPT-OSS expert tensor '{name}'"
            ))
        })
}

fn gpt_oss_collect_bindings(
    metadata: &CheckpointMetadata,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    let e = num_experts as u64;
    if e == 0 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS num_experts must be > 0".to_string(),
        ));
    }
    let mut bindings = Vec::with_capacity(
        (num_layers as usize) * (num_experts as usize) * GPT_OSS_EXPERT_SECTIONS.len(),
    );
    for layer in 0..num_layers {
        let prefix = format!("model.layers.{layer}.mlp.experts.");
        let gate_up_w = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_blocks"))?;
        let gate_up_s = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_scales"))?;
        let down_w = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_blocks"))?;
        let down_s = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_scales"))?;

        let gu_w_span = gate_up_w.span_bytes / e;
        let gu_s_span = gate_up_s.span_bytes / e;
        let dn_w_span = down_w.span_bytes / e;
        let dn_s_span = down_s.span_bytes / e;
        if gu_w_span * e != gate_up_w.span_bytes
            || gu_s_span * e != gate_up_s.span_bytes
            || dn_w_span * e != down_w.span_bytes
            || dn_s_span * e != down_s.span_bytes
        {
            return Err(CompileError::InvalidInput(format!(
                "stream_routed_experts: GPT-OSS fused expert spans at layer \
                 {layer} are not divisible by num_experts={num_experts}"
            )));
        }

        for expert in 0..num_experts as u64 {
            bindings.push(StreamBinding {
                file_id: gate_up_w.file_id,
                file_offset: gate_up_w.file_offset + expert * gu_w_span,
                span_bytes: gu_w_span,
            });
            bindings.push(StreamBinding {
                file_id: gate_up_s.file_id,
                file_offset: gate_up_s.file_offset + expert * gu_s_span,
                span_bytes: gu_s_span,
            });
            bindings.push(StreamBinding {
                file_id: down_w.file_id,
                file_offset: down_w.file_offset + expert * dn_w_span,
                span_bytes: dn_w_span,
            });
            bindings.push(StreamBinding {
                file_id: down_s.file_id,
                file_offset: down_s.file_offset + expert * dn_s_span,
                span_bytes: dn_s_span,
            });
        }
    }
    Ok(bindings)
}

pub(crate) const GPT_OSS_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: GPT_OSS_EXPERT_SECTIONS,
    is_streamed: is_gpt_oss_streamed_expert_tensor,
    collect_bindings: gpt_oss_collect_bindings,
};

/// Fixed Mixtral section order — must match `mixtral_expert_sections.hpp`.
/// HF layout: w1=gate, w2=down, w3=up (BF16, no scales).
pub const MIXTRAL_EXPERT_SECTIONS: &[&str] = &["w1.weight", "w2.weight", "w3.weight"];

/// Mixtral routed experts:
/// `model.layers.{L}.block_sparse_moe.experts.{E}.w{1,2,3}.weight`.
///
/// The router (`…block_sparse_moe.gate.weight`) stays resident.
pub(crate) fn is_mixtral_routed_expert_tensor(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((_, rest)) = rest.split_once('.') else {
        return false;
    };
    rest.starts_with("block_sparse_moe.experts.")
        && ends_with_any(name, &[".w1.weight", ".w2.weight", ".w3.weight"])
}

/// Parse `model.layers.{L}.block_sparse_moe.experts.{E}.{section}`.
pub(crate) fn parse_mixtral_expert_section(name: &str) -> Option<(u32, u32, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("block_sparse_moe.experts.")?;
    let (expert_str, section) = rest.split_once('.')?;
    let layer: u32 = layer_str.parse().ok()?;
    let expert: u32 = expert_str.parse().ok()?;
    let section_idx = MIXTRAL_EXPERT_SECTIONS.iter().position(|s| *s == section)?;
    Some((layer, expert, section_idx))
}

fn mixtral_collect_bindings(
    metadata: &CheckpointMetadata,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    collect_bindings_from_named_tensors(
        metadata,
        num_layers,
        num_experts,
        MIXTRAL_EXPERT_SECTIONS.len(),
        is_mixtral_routed_expert_tensor,
        parse_mixtral_expert_section,
    )
}

pub(crate) const MIXTRAL_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: MIXTRAL_EXPERT_SECTIONS,
    is_streamed: is_mixtral_routed_expert_tensor,
    collect_bindings: mixtral_collect_bindings,
};

fn ends_with_any(value: &str, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|suffix| value.ends_with(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dsv4_names() {
        assert_eq!(
            parse_dsv4_expert_section("layers.3.ffn.experts.12.w2.scale"),
            Some((3, 12, 3))
        );
        assert!(parse_dsv4_expert_section("layers.0.ffn.shared_experts.w1.weight").is_none());
        assert!(parse_dsv4_expert_section("layers.0.ffn.gate.weight").is_none());
        assert!(parse_dsv4_expert_section("mtp.0.ffn.experts.0.w1.scale").is_none());
    }

    #[test]
    fn gpt_oss_streamed_matcher() {
        assert!(is_gpt_oss_streamed_expert_tensor(
            "model.layers.0.mlp.experts.gate_up_proj_blocks"
        ));
        assert!(is_gpt_oss_streamed_expert_tensor(
            "model.layers.1.mlp.experts.down_proj_scales"
        ));
        assert!(!is_gpt_oss_streamed_expert_tensor(
            "model.layers.0.mlp.experts.gate_up_proj_bias"
        ));
        assert!(!is_gpt_oss_streamed_expert_tensor(
            "model.layers.0.mlp.experts.down_proj_bias"
        ));
    }

    #[test]
    fn parse_mixtral_names() {
        assert_eq!(
            parse_mixtral_expert_section(
                "model.layers.3.block_sparse_moe.experts.7.w2.weight"
            ),
            Some((3, 7, 1))
        );
        assert!(is_mixtral_routed_expert_tensor(
            "model.layers.0.block_sparse_moe.experts.0.w1.weight"
        ));
        assert!(!is_mixtral_routed_expert_tensor(
            "model.layers.0.block_sparse_moe.gate.weight"
        ));
        assert!(parse_mixtral_expert_section(
            "model.layers.0.block_sparse_moe.gate.weight"
        )
        .is_none());
    }
}
