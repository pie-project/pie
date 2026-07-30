//! Arch-specific [`StreamArchDesc`] plugins for SSD expert streaming.
//!
//! Generic plan construction lives in [`crate::stream`]. This module owns
//! checkpoint naming, section order, and binding-grid collectors. Each
//! streaming-capable arch registers a `select_*` resolver on `ArchProfile`
//! in [`crate::abi`]; policy-dependent recipes (e.g. GPT-OSS native vs
//! routed) live in those selectors — not as special cases in `abi`.

use crate::error::CompileError;
use crate::source::{CheckpointMetadata, RawTensor};
use crate::storage::{ExpertPackKind, StorageTarget, StreamBinding};
use crate::stream::{StreamArchDesc, collect_bindings_from_named_tensors};
use crate::types::Mxfp4MoePolicy;

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
    _target: &StorageTarget,
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
    pack_kind: ExpertPackKind::None,
};

pub(crate) fn select_dsv4(_target: &StorageTarget) -> Option<StreamArchDesc> {
    Some(DSV4_STREAM_ARCH)
}

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
    _target: &StorageTarget,
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
    pack_kind: ExpertPackKind::None,
};

/// Native Marlin stream sections — must match `gpt_oss_expert_sections.hpp`.
/// Built offline into an expert pack; biases stay resident.
pub const GPT_OSS_NATIVE_EXPERT_SECTIONS: &[&str] = &[
    "gate.weight",
    "gate.scale",
    "up.weight",
    "up.scale",
    "down.weight",
    "down.scale",
];

fn align_up_u64(v: u64, a: u64) -> u64 {
    (v + a - 1) / a * a
}

fn tp_local_range(full: i64, target: &StorageTarget) -> Result<(i64, i64), CompileError> {
    let world = i64::from(target.tp_size.max(1));
    let rank = i64::from(target.tp_rank);
    if full % world != 0 {
        return Err(CompileError::InvalidInput(format!(
            "dimension {full} is not divisible by tp_size {}",
            target.tp_size
        )));
    }
    let local = full / world;
    Ok((rank * local, local))
}

/// Marlin per-expert section byte sizes from GPT-OSS fused HF bank shapes.
/// Uses TP-local intermediate (`I_full / tp_size`), then pads to 128 for Marlin.
pub(crate) fn gpt_oss_native_section_bytes(
    gate_up_blocks: &RawTensor,
    down_blocks: &RawTensor,
    target: &StorageTarget,
) -> Result<[u64; 6], CompileError> {
    if gate_up_blocks.shape.len() != 4 || down_blocks.shape.len() != 4 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS native stream expects 4-D \
             gate_up/down _blocks tensors"
                .to_string(),
        ));
    }
    let fused_rows = gate_up_blocks.shape[1] as u64;
    let gu_groups = gate_up_blocks.shape[2] as u64;
    let gu_lanes = gate_up_blocks.shape[3] as u64;
    if fused_rows % 2 != 0 || gu_lanes != 16 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS native gate_up expected \
             [E, 2I, H/32, 16], got {:?}",
            gate_up_blocks.shape
        )));
    }
    let full_intermediate = (fused_rows / 2) as i64;
    let (_local_start, local_intermediate) =
        tp_local_range(full_intermediate, target)?;
    if local_intermediate % 32 != 0 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS native TP shard I_local={local_intermediate} \
             must be divisible by 32"
        )));
    }
    let intermediate_native = align_up_u64(local_intermediate as u64, 128);
    let hidden = gu_groups * 32;
    let down_hidden = down_blocks.shape[1] as u64;
    let down_groups = down_blocks.shape[2] as u64;
    let down_lanes = down_blocks.shape[3] as u64;
    if down_lanes != 16 || down_hidden != hidden || down_groups * 32 != full_intermediate as u64 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS native down shape mismatch \
             (gate_up {:?}, down {:?})",
            gate_up_blocks.shape, down_blocks.shape
        )));
    }
    // MXFP4 nibble packing: rows * cols / 2 bytes. Scales: rows * groups.
    let gate_w = intermediate_native * hidden / 2;
    let gate_s = intermediate_native * gu_groups;
    let down_w = hidden * intermediate_native / 2;
    let down_s = hidden * (intermediate_native / 32);
    Ok([gate_w, gate_s, gate_w, gate_s, down_w, down_s])
}

fn gpt_oss_native_collect_bindings(
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    let e = num_experts as u64;
    if e == 0 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS num_experts must be > 0".to_string(),
        ));
    }
    // Pack layout is a single virtual file; C++ remaps path after pack build.
    // Offsets are deterministic: (layer * E + expert) * slot + section_offset.
    const SECTION_ALIGN: u64 = 256;
    let mut bindings = Vec::with_capacity(
        (num_layers as usize) * (num_experts as usize) * GPT_OSS_NATIVE_EXPERT_SECTIONS.len(),
    );
    let mut section_bytes: Option<[u64; 6]> = None;
    let mut section_offsets = [0u64; 6];
    let mut slot_bytes = 0u64;
    for layer in 0..num_layers {
        let prefix = format!("model.layers.{layer}.mlp.experts.");
        let gate_up_w = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_blocks"))?;
        let down_w = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_blocks"))?;
        // Still require scales to exist (consumed / validated by ABI skip).
        let _ = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_scales"))?;
        let _ = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_scales"))?;
        let bytes = gpt_oss_native_section_bytes(gate_up_w, down_w, target)?;
        if let Some(prev) = section_bytes {
            if prev != bytes {
                return Err(CompileError::InvalidInput(format!(
                    "stream_routed_experts: GPT-OSS native section sizes differ \
                     across layers (layer {layer})"
                )));
            }
        } else {
            let mut off = 0u64;
            for s in 0..6 {
                section_offsets[s] = off;
                off = align_up_u64(off + bytes[s], SECTION_ALIGN);
            }
            slot_bytes = off;
            section_bytes = Some(bytes);
        }
        for expert in 0..e {
            let base = (layer as u64 * e + expert) * slot_bytes;
            for s in 0..6 {
                bindings.push(StreamBinding {
                    file_id: crate::types::FileId(0),
                    file_offset: base + section_offsets[s],
                    span_bytes: bytes[s],
                });
            }
        }
    }
    Ok(bindings)
}

pub(crate) const GPT_OSS_NATIVE_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: GPT_OSS_NATIVE_EXPERT_SECTIONS,
    // Same HF tensors are excluded from the resident schedule.
    is_streamed: is_gpt_oss_streamed_expert_tensor,
    collect_bindings: gpt_oss_native_collect_bindings,
    pack_kind: ExpertPackKind::GptOssNativeMarlin,
};

/// GPT-OSS stream recipe depends on `mxfp4_moe`: HF packs (routed_dequant),
/// Marlin pack (native), or offline BF16 pack (eager_bf16). Under TP,
/// RoutedDecode switches to a per-rank MXFP4 pack (strided down shards).
pub(crate) fn select_gpt_oss(target: &StorageTarget) -> Option<StreamArchDesc> {
    match target.mxfp4_moe {
        Mxfp4MoePolicy::NativeGemm => Some(GPT_OSS_NATIVE_STREAM_ARCH),
        Mxfp4MoePolicy::EagerBf16 => Some(GPT_OSS_EAGER_BF16_STREAM_ARCH),
        Mxfp4MoePolicy::RoutedDecode => {
            if target.tp_size > 1 {
                Some(GPT_OSS_ROUTED_TP_STREAM_ARCH)
            } else {
                Some(GPT_OSS_STREAM_ARCH)
            }
        }
    }
}

/// TP-local HF MXFP4 section sizes for RoutedDecode packs.
pub(crate) fn gpt_oss_routed_tp_section_bytes(
    gate_up_blocks: &RawTensor,
    down_blocks: &RawTensor,
    target: &StorageTarget,
) -> Result<[u64; 4], CompileError> {
    if gate_up_blocks.shape.len() != 4 || down_blocks.shape.len() != 4 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS RoutedDecode TP pack expects 4-D \
             gate_up/down _blocks tensors"
                .to_string(),
        ));
    }
    let fused_rows = gate_up_blocks.shape[1] as u64;
    let gu_groups = gate_up_blocks.shape[2] as u64;
    let gu_lanes = gate_up_blocks.shape[3] as u64;
    if fused_rows % 2 != 0 || gu_lanes != 16 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS RoutedDecode gate_up expected \
             [E, 2I, H/32, 16], got {:?}",
            gate_up_blocks.shape
        )));
    }
    let full_intermediate = (fused_rows / 2) as i64;
    let (_local_start, local_intermediate) = tp_local_range(full_intermediate, target)?;
    if local_intermediate % 32 != 0 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS RoutedDecode TP shard \
             I_local={local_intermediate} must be divisible by 32"
        )));
    }
    let i_local = local_intermediate as u64;
    let hidden = gu_groups * 32;
    let down_hidden = down_blocks.shape[1] as u64;
    let down_groups = down_blocks.shape[2] as u64;
    let down_lanes = down_blocks.shape[3] as u64;
    if down_lanes != 16 || down_hidden != hidden || down_groups * 32 != full_intermediate as u64 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS RoutedDecode down shape mismatch \
             (gate_up {:?}, down {:?})",
            gate_up_blocks.shape, down_blocks.shape
        )));
    }
    // Dense local HF MXFP4: gate_up rows 2*I_local; down groups I_local/32.
    let gu_w = i_local * hidden; // 2*I_local * (H/32) * 16
    let gu_s = 2 * i_local * (hidden / 32);
    let dn_w = hidden * i_local / 2; // H * (I_local/32) * 16
    let dn_s = hidden * (i_local / 32);
    Ok([gu_w, gu_s, dn_w, dn_s])
}

fn gpt_oss_routed_tp_collect_bindings(
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    let e = num_experts as u64;
    if e == 0 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS num_experts must be > 0".to_string(),
        ));
    }
    const SECTION_ALIGN: u64 = 256;
    let mut bindings = Vec::with_capacity(
        (num_layers as usize) * (num_experts as usize) * GPT_OSS_EXPERT_SECTIONS.len(),
    );
    let mut section_bytes: Option<[u64; 4]> = None;
    let mut section_offsets = [0u64; 4];
    let mut slot_bytes = 0u64;
    for layer in 0..num_layers {
        let prefix = format!("model.layers.{layer}.mlp.experts.");
        let gate_up_w = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_blocks"))?;
        let down_w = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_blocks"))?;
        let _ = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_scales"))?;
        let _ = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_scales"))?;
        let bytes = gpt_oss_routed_tp_section_bytes(gate_up_w, down_w, target)?;
        if let Some(prev) = section_bytes {
            if prev != bytes {
                return Err(CompileError::InvalidInput(format!(
                    "stream_routed_experts: GPT-OSS RoutedDecode TP section \
                     sizes differ across layers (layer {layer})"
                )));
            }
        } else {
            let mut off = 0u64;
            for s in 0..4 {
                section_offsets[s] = off;
                off = align_up_u64(off + bytes[s], SECTION_ALIGN);
            }
            slot_bytes = off;
            section_bytes = Some(bytes);
        }
        for expert in 0..e {
            let base = (layer as u64 * e + expert) * slot_bytes;
            for s in 0..4 {
                bindings.push(StreamBinding {
                    file_id: crate::types::FileId(0),
                    file_offset: base + section_offsets[s],
                    span_bytes: bytes[s],
                });
            }
        }
    }
    Ok(bindings)
}

pub(crate) const GPT_OSS_ROUTED_TP_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: GPT_OSS_EXPERT_SECTIONS,
    is_streamed: is_gpt_oss_streamed_expert_tensor,
    collect_bindings: gpt_oss_routed_tp_collect_bindings,
    pack_kind: ExpertPackKind::GptOssRoutedMxfp4,
};

/// Eager-BF16 stream sections — offline dequant pack; biases stay resident.
/// Must match `gpt_oss_expert_sections.hpp` BF16 helpers.
pub const GPT_OSS_EAGER_BF16_EXPERT_SECTIONS: &[&str] =
    &["gate.weight", "up.weight", "down.weight"];

fn gpt_oss_eager_bf16_section_bytes(
    gate_up_blocks: &RawTensor,
    down_blocks: &RawTensor,
    target: &StorageTarget,
) -> Result<[u64; 3], CompileError> {
    if gate_up_blocks.shape.len() != 4 || down_blocks.shape.len() != 4 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS eager BF16 stream expects 4-D \
             gate_up/down _blocks tensors"
                .to_string(),
        ));
    }
    let fused_rows = gate_up_blocks.shape[1] as u64;
    let gu_groups = gate_up_blocks.shape[2] as u64;
    let gu_lanes = gate_up_blocks.shape[3] as u64;
    if fused_rows % 2 != 0 || gu_lanes != 16 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS eager BF16 gate_up expected \
             [E, 2I, H/32, 16], got {:?}",
            gate_up_blocks.shape
        )));
    }
    let full_intermediate = (fused_rows / 2) as i64;
    let (_local_start, local_intermediate) = tp_local_range(full_intermediate, target)?;
    let intermediate = local_intermediate as u64;
    let hidden = gu_groups * 32;
    let down_hidden = down_blocks.shape[1] as u64;
    let down_groups = down_blocks.shape[2] as u64;
    let down_lanes = down_blocks.shape[3] as u64;
    if down_lanes != 16 || down_hidden != hidden || down_groups * 32 != full_intermediate as u64 {
        return Err(CompileError::InvalidInput(format!(
            "stream_routed_experts: GPT-OSS eager BF16 down shape mismatch \
             (gate_up {:?}, down {:?})",
            gate_up_blocks.shape, down_blocks.shape
        )));
    }
    // BF16: rows * cols * 2 (TP-local intermediate).
    let gate = intermediate * hidden * 2;
    let down = hidden * intermediate * 2;
    Ok([gate, gate, down])
}

fn gpt_oss_eager_bf16_collect_bindings(
    metadata: &CheckpointMetadata,
    target: &StorageTarget,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    let e = num_experts as u64;
    if e == 0 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: GPT-OSS num_experts must be > 0".to_string(),
        ));
    }
    const SECTION_ALIGN: u64 = 256;
    let mut bindings = Vec::with_capacity(
        (num_layers as usize)
            * (num_experts as usize)
            * GPT_OSS_EAGER_BF16_EXPERT_SECTIONS.len(),
    );
    let mut section_bytes: Option<[u64; 3]> = None;
    let mut section_offsets = [0u64; 3];
    let mut slot_bytes = 0u64;
    for layer in 0..num_layers {
        let prefix = format!("model.layers.{layer}.mlp.experts.");
        let gate_up_w = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_blocks"))?;
        let down_w = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_blocks"))?;
        let _ = gpt_oss_find_tensor(metadata, &format!("{prefix}gate_up_proj_scales"))?;
        let _ = gpt_oss_find_tensor(metadata, &format!("{prefix}down_proj_scales"))?;
        let bytes = gpt_oss_eager_bf16_section_bytes(gate_up_w, down_w, target)?;
        if let Some(prev) = section_bytes {
            if prev != bytes {
                return Err(CompileError::InvalidInput(format!(
                    "stream_routed_experts: GPT-OSS eager BF16 section sizes \
                     differ across layers (layer {layer})"
                )));
            }
        } else {
            let mut off = 0u64;
            for s in 0..3 {
                section_offsets[s] = off;
                off = align_up_u64(off + bytes[s], SECTION_ALIGN);
            }
            slot_bytes = off;
            section_bytes = Some(bytes);
        }
        for expert in 0..e {
            let base = (layer as u64 * e + expert) * slot_bytes;
            for s in 0..3 {
                bindings.push(StreamBinding {
                    file_id: crate::types::FileId(0),
                    file_offset: base + section_offsets[s],
                    span_bytes: bytes[s],
                });
            }
        }
    }
    Ok(bindings)
}

pub(crate) const GPT_OSS_EAGER_BF16_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: GPT_OSS_EAGER_BF16_EXPERT_SECTIONS,
    is_streamed: is_gpt_oss_streamed_expert_tensor,
    collect_bindings: gpt_oss_eager_bf16_collect_bindings,
    pack_kind: ExpertPackKind::GptOssEagerBf16,
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
    _target: &StorageTarget,
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
    pack_kind: ExpertPackKind::None,
};

pub(crate) fn select_mixtral(_target: &StorageTarget) -> Option<StreamArchDesc> {
    Some(MIXTRAL_STREAM_ARCH)
}

/// Plain Qwen3-MoE per-expert BF16 weights — must match `qwen_moe_expert_sections.hpp`.
pub const QWEN3_MOE_EXPERT_SECTIONS: &[&str] = &[
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
];

/// Qwen3-MoE routed experts:
/// `model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight`.
pub(crate) fn is_qwen3_moe_routed_expert_tensor(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return false;
    };
    let Some((_, rest)) = rest.split_once('.') else {
        return false;
    };
    rest.starts_with("mlp.experts.")
        && ends_with_any(
            name,
            &[
                ".gate_proj.weight",
                ".up_proj.weight",
                ".down_proj.weight",
            ],
        )
        // Exclude fused bank names used by Qwen3.5-MoE.
        && !name.ends_with("mlp.experts.gate_up_proj")
        && !name.ends_with("mlp.experts.down_proj")
}

/// Parse `model.layers.{L}.mlp.experts.{E}.{section}`.
pub(crate) fn parse_qwen3_moe_expert_section(name: &str) -> Option<(u32, u32, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let (layer_str, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("mlp.experts.")?;
    let (expert_str, section) = rest.split_once('.')?;
    // Reject fused bank names (no expert index).
    if expert_str == "gate_up_proj" || expert_str == "down_proj" {
        return None;
    }
    let layer: u32 = layer_str.parse().ok()?;
    let expert: u32 = expert_str.parse().ok()?;
    let section_idx = QWEN3_MOE_EXPERT_SECTIONS.iter().position(|s| *s == section)?;
    Some((layer, expert, section_idx))
}

fn qwen3_moe_collect_bindings(
    metadata: &CheckpointMetadata,
    _target: &StorageTarget,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    collect_bindings_from_named_tensors(
        metadata,
        num_layers,
        num_experts,
        QWEN3_MOE_EXPERT_SECTIONS.len(),
        is_qwen3_moe_routed_expert_tensor,
        parse_qwen3_moe_expert_section,
    )
}

pub(crate) const QWEN3_MOE_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: QWEN3_MOE_EXPERT_SECTIONS,
    is_streamed: is_qwen3_moe_routed_expert_tensor,
    collect_bindings: qwen3_moe_collect_bindings,
    pack_kind: ExpertPackKind::None,
};

pub(crate) fn select_qwen3_moe(_target: &StorageTarget) -> Option<StreamArchDesc> {
    Some(QWEN3_MOE_STREAM_ARCH)
}

/// Qwen3.5/3.6-MoE fused BF16 banks — must match `qwen_moe_expert_sections.hpp`.
pub const QWEN35_MOE_EXPERT_SECTIONS: &[&str] = &["gate_up.weight", "down.weight"];

fn qwen35_moe_layers_prefix(name: &str) -> Option<&str> {
    name.strip_prefix("model.language_model.layers.")
        .or_else(|| name.strip_prefix("model.layers."))
}

/// Qwen3.5-MoE fused expert packs (not shared expert / router).
pub(crate) fn is_qwen35_moe_streamed_expert_tensor(name: &str) -> bool {
    let Some(rest) = qwen35_moe_layers_prefix(name) else {
        return false;
    };
    let Some((_, rest)) = rest.split_once('.') else {
        return false;
    };
    matches!(
        rest,
        "mlp.experts.gate_up_proj" | "mlp.experts.down_proj"
    )
}

fn qwen35_moe_find_tensor<'a>(
    metadata: &'a CheckpointMetadata,
    name: &str,
) -> Result<&'a RawTensor, CompileError> {
    metadata
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| {
            CompileError::InvalidInput(format!(
                "stream_routed_experts: missing Qwen3.5-MoE expert tensor '{name}'"
            ))
        })
}

fn qwen35_moe_collect_bindings(
    metadata: &CheckpointMetadata,
    _target: &StorageTarget,
    num_layers: u32,
    num_experts: u32,
) -> Result<Vec<StreamBinding>, CompileError> {
    let e = num_experts as u64;
    if e == 0 {
        return Err(CompileError::InvalidInput(
            "stream_routed_experts: Qwen3.5-MoE num_experts must be > 0".to_string(),
        ));
    }
    // Prefer language_model prefix when present (multimodal checkpoints).
    let prefix_root = if metadata
        .tensors
        .iter()
        .any(|t| t.name.starts_with("model.language_model.layers."))
    {
        "model.language_model.layers."
    } else {
        "model.layers."
    };
    let mut bindings = Vec::with_capacity(
        (num_layers as usize) * (num_experts as usize) * QWEN35_MOE_EXPERT_SECTIONS.len(),
    );
    for layer in 0..num_layers {
        let prefix = format!("{prefix_root}{layer}.mlp.experts.");
        let gate_up = qwen35_moe_find_tensor(metadata, &format!("{prefix}gate_up_proj"))?;
        let down = qwen35_moe_find_tensor(metadata, &format!("{prefix}down_proj"))?;
        let gu_span = gate_up.span_bytes / e;
        let dn_span = down.span_bytes / e;
        if gu_span * e != gate_up.span_bytes || dn_span * e != down.span_bytes {
            return Err(CompileError::InvalidInput(format!(
                "stream_routed_experts: Qwen3.5-MoE fused expert spans at layer \
                 {layer} are not divisible by num_experts={num_experts}"
            )));
        }
        for expert in 0..e {
            bindings.push(StreamBinding {
                file_id: gate_up.file_id,
                file_offset: gate_up.file_offset + expert * gu_span,
                span_bytes: gu_span,
            });
            bindings.push(StreamBinding {
                file_id: down.file_id,
                file_offset: down.file_offset + expert * dn_span,
                span_bytes: dn_span,
            });
        }
    }
    Ok(bindings)
}

pub(crate) const QWEN35_MOE_STREAM_ARCH: StreamArchDesc = StreamArchDesc {
    sections: QWEN35_MOE_EXPERT_SECTIONS,
    is_streamed: is_qwen35_moe_streamed_expert_tensor,
    collect_bindings: qwen35_moe_collect_bindings,
    pack_kind: ExpertPackKind::None,
};

pub(crate) fn select_qwen35_moe(_target: &StorageTarget) -> Option<StreamArchDesc> {
    Some(QWEN35_MOE_STREAM_ARCH)
}

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

    #[test]
    fn parse_qwen3_moe_names() {
        assert_eq!(
            parse_qwen3_moe_expert_section(
                "model.layers.2.mlp.experts.5.up_proj.weight"
            ),
            Some((2, 5, 1))
        );
        assert!(is_qwen3_moe_routed_expert_tensor(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ));
        assert!(!is_qwen3_moe_routed_expert_tensor(
            "model.layers.0.mlp.experts.gate_up_proj"
        ));
        assert!(parse_qwen3_moe_expert_section(
            "model.layers.0.mlp.experts.gate_up_proj"
        )
        .is_none());
    }

    #[test]
    fn qwen35_moe_fused_matcher() {
        assert!(is_qwen35_moe_streamed_expert_tensor(
            "model.layers.0.mlp.experts.gate_up_proj"
        ));
        assert!(is_qwen35_moe_streamed_expert_tensor(
            "model.language_model.layers.1.mlp.experts.down_proj"
        ));
        assert!(!is_qwen35_moe_streamed_expert_tensor(
            "model.layers.0.mlp.shared_expert.gate_proj.weight"
        ));
        assert!(!is_qwen35_moe_streamed_expert_tensor(
            "model.layers.0.mlp.gate.weight"
        ));
        assert!(!is_qwen35_moe_streamed_expert_tensor(
            "model.layers.0.mlp.experts.0.gate_proj.weight"
        ));
    }
}
