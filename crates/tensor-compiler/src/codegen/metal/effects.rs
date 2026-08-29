//! The readiness and commit kernels: the guard that decides a lane may run,
//! and the cursor advance that retires it.
//!
//! Single-lane (`M1`) forms specialise the channel effects into the emitted
//! text; the grouped (`M3`) forms are generic and read the same decisions out
//! of a per-channel `M3ChannelMeta` flag word at run time.

use crate::codegen::error::{EmitError, EmitterKind};
use alloc::string::String;
use core::fmt::Write as _;

use alloc::vec::Vec;

use crate::codegen::fault::{
    LANE_HEADER_MISMATCH, M1_HEAD_STALE, M1_NOT_EMPTY, M1_NOT_FULL, M1_PUT_BLOCKED,
    M1_RING_CORRUPT, M3_NOT_READY, M3_RING_CORRUPT,
};
use crate::codegen::layout;
use tensor_ir::op::ChannelUse;
use tensor_ir::validate::{BoundTrace, Direction};

use super::preamble::{common_effect_preamble, emit_word_arguments, grouped_preamble};
use super::{M1ChannelEffect, METAL_M1_MAX_CHANNELS};

/// The lane-table ABI revision the emitted readiness kernel refuses to run
/// against anything else (`PTIR_LANE_TABLE_ABI_VERSION`).
const LANE_TABLE_ABI_VERSION: u32 = crate::plan::LANE_TABLE_ABI_VERSION;

/// `emit_readiness_msl` — one lane, channel effects baked in.
///
/// Rejects a channel count past [`METAL_M1_MAX_CHANNELS`] rather than emitting
/// it. The constant states where buffer 30 falls; before anything compiled the
/// emitter's output it stated only that, and a program one channel wider
/// produced a `[[buffer(31)]]` Metal rejects — a raw shader diagnostic at PSO
/// build time, from a count a guest container chooses.
pub fn emit_readiness(
    function_name: &str,
    channels: &[M1ChannelEffect],
) -> Result<String, EmitError> {
    if channels.len() > METAL_M1_MAX_CHANNELS {
        return Err(EmitError::ChannelLimitExceeded {
            emitter: EmitterKind::MetalReadiness,
            limit: METAL_M1_MAX_CHANNELS,
        });
    }
    let mut source = String::new();
    source.push_str(common_effect_preamble());
    let _ = write!(
        source,
        "kernel void {function_name}(device M1Status* status [[buffer(0)]], \
         const device uchar* lane_bytes [[buffer(1)]]"
    );
    emit_word_arguments(&mut source, channels.len());
    source.push_str(", uint gid [[thread_position_in_grid]]) {\n");
    source.push_str("  if (gid != 0) return;\n");
    source.push_str(
        "  const device M1LaneHeader* header = \
         reinterpret_cast<const device M1LaneHeader*>(lane_bytes);\n",
    );
    source.push_str(
        "  const device M1LaneChannelSlot* slots = \
         reinterpret_cast<const device M1LaneChannelSlot*>(lane_bytes + \
         sizeof(M1LaneHeader) + sizeof(M1LaneRecord));\n",
    );
    let _ = writeln!(
        source,
        "  if (header->abi_version != {LANE_TABLE_ABI_VERSION} || header->lane_count != 1 || \
         header->channel_count != {}) {{ status->state = 3; status->fault = {LANE_HEADER_MISMATCH:#x}; \
         status->reserved0 = header->abi_version; \
         status->reserved1 = (header->lane_count << 16) | header->channel_count; return; }}",
        channels.len()
    );
    source.push_str("  status->state = 0; status->fault = 0;\n");
    for (channel, effect) in channels.iter().enumerate() {
        source.push_str("  {\n");
        let _ = writeln!(source, "    const ulong head = words_{channel}[0];");
        let _ = writeln!(source, "    const ulong tail = words_{channel}[1];");
        let _ = writeln!(
            source,
            "    if (words_{channel}[2] != 0ul || words_{channel}[3] != 0ul || tail < head) \
             {{ status->state = 3; status->fault = {}; return; }}",
            M1_RING_CORRUPT + channel as u32
        );
        let _ = writeln!(
            source,
            "    const ulong expected_head = slots[{channel}].expected_head;"
        );
        let _ = writeln!(
            source,
            "    const ulong expected_tail = slots[{channel}].expected_tail;"
        );
        let _ = writeln!(
            source,
            "    if (expected_head != ~0ul && head != expected_head) \
             {{ status->state = 2; status->fault = {}; return; }}",
            M1_HEAD_STALE + channel as u32
        );
        if effect.requires_full {
            let _ = writeln!(
                source,
                "    if (tail <= head) {{ status->state = 2; status->fault = {}; return; }}",
                M1_NOT_FULL + channel as u32
            );
        }
        if effect.requires_empty {
            let _ = writeln!(
                source,
                "    if (tail - head >= {}ul) {{ status->state = 2; status->fault = {}; return; }}",
                effect.capacity,
                M1_NOT_EMPTY + channel as u32
            );
        }
        if effect.put {
            let credit = u32::from(effect.take && effect.requires_full);
            let _ = writeln!(
                source,
                "    if (expected_tail == ~0ul || tail != expected_tail || \
                 tail - head >= {}ul + {credit}ul) \
                 {{ status->state = 2; status->fault = {}; return; }}",
                effect.capacity,
                M1_PUT_BLOCKED + channel as u32
            );
        }
        source.push_str("  }\n");
    }
    source.push_str("  status->state = 1;\n}\n");
    Ok(source)
}

/// `emit_commit_msl` — one lane; advance head for takes, tail for puts.
pub fn emit_commit(function_name: &str, channels: &[M1ChannelEffect]) -> Result<String, EmitError> {
    if channels.len() > METAL_M1_MAX_CHANNELS {
        return Err(EmitError::ChannelLimitExceeded {
            emitter: EmitterKind::MetalCommit,
            limit: METAL_M1_MAX_CHANNELS,
        });
    }
    let mut source = String::new();
    source.push_str(common_effect_preamble());
    let _ = write!(
        source,
        "kernel void {function_name}(device M1Status* status [[buffer(0)]], \
         const device uchar* lane_bytes [[buffer(1)]]"
    );
    emit_word_arguments(&mut source, channels.len());
    source.push_str(", uint gid [[thread_position_in_grid]]) {\n");
    source.push_str("  if (gid != 0 || status->state != 1) return;\n");
    source.push_str(
        "  const device M1LaneChannelSlot* slots = \
         reinterpret_cast<const device M1LaneChannelSlot*>(lane_bytes + \
         sizeof(M1LaneHeader) + sizeof(M1LaneRecord));\n",
    );
    for (channel, effect) in channels.iter().enumerate() {
        source.push_str("  {\n");
        let _ = writeln!(source, "    const ulong old_head = words_{channel}[0];");
        let _ = writeln!(source, "    const ulong old_tail = words_{channel}[1];");
        if effect.take {
            let _ = writeln!(
                source,
                "    if (old_tail > old_head) words_{channel}[0] = old_head + 1ul;"
            );
        }
        if effect.put {
            let _ = writeln!(source, "    words_{channel}[1] = old_tail + 1ul;");
        }
        source.push_str("  }\n");
    }
    source.push_str("  status->state = 4;\n}\n");
    Ok(source)
}

/// `emit_grouped_readiness_msl` — one thread per lane, effects read from
/// `M3ChannelMeta::flags` instead of baked in.
pub fn emit_grouped_readiness(function_name: &str) -> String {
    let mut source = String::new();
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(&layout::STATUS.emit_msl("M1"));
    source.push_str(grouped_preamble());
    let _ = writeln!(
        source,
        "kernel void {function_name}(const device uchar* lane_bytes [[buffer(0)]], \
         const device M3ChannelMeta* meta [[buffer(1)]], \
         uint lane_index [[thread_position_in_grid]]) {{"
    );
    source.push_str(
        "  const device M3LaneHeader* header = reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n",
    );
    source.push_str("  if (lane_index >= header->lane_count) return;\n");
    source.push_str(
        "  const device M3LaneRecord* lanes = reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n",
    );
    source.push_str(
        "  const device M3LaneChannelSlot* slots = reinterpret_cast<const device M3LaneChannelSlot*>(lane_bytes + sizeof(M3LaneHeader) + header->lane_count * sizeof(M3LaneRecord));\n",
    );
    source.push_str("  const M3LaneRecord lane = lanes[lane_index];\n");
    source.push_str(
        "  device M1Status* status = reinterpret_cast<device M1Status*>(lane.commit_slot);\n",
    );
    source.push_str("  status->state = 0; status->fault = 0;\n");
    source.push_str("  for (uint channel = 0; channel < header->channel_count; ++channel) {\n");
    source.push_str("    const M3ChannelMeta m = meta[lane.channel_slot_offset + channel];\n");
    source.push_str("    if ((m.flags & 1u) == 0u) continue;\n");
    source.push_str(
        "    const device ulong* words = reinterpret_cast<const device ulong*>(m.words);\n",
    );
    source.push_str("    const ulong head = words[0], tail = words[1];\n");
    source.push_str(
        "    const M3LaneChannelSlot slot = slots[lane.channel_slot_offset + channel];\n",
    );
    let _ = writeln!(
        source,
        "    if (words[2] != 0ul || words[3] != 0ul || tail < head) \
         {{ status->state = 3; status->fault = {M3_RING_CORRUPT:#x}u + channel; return; }}"
    );
    source.push_str("    bool retry = slot.expected_head != ~0ul && head != slot.expected_head;\n");
    source.push_str("    retry = retry || (((m.flags & 2u) != 0u) && tail <= head);\n");
    source.push_str(
        "    retry = retry || (((m.flags & 4u) != 0u) && tail - head >= ulong(m.capacity));\n",
    );
    source.push_str("    if ((m.flags & 16u) != 0u) {\n");
    source.push_str("      const ulong credit = ((m.flags & 10u) == 10u) ? 1ul : 0ul;\n");
    source.push_str(
        "      retry = retry || slot.expected_tail == ~0ul || tail != slot.expected_tail || \
         tail - head >= ulong(m.capacity) + credit;\n",
    );
    source.push_str("    }\n");
    let _ = writeln!(
        source,
        "    if (retry) {{ status->state = (m.flags & 32u) != 0u ? 3u : 2u; \
         status->fault = {M3_NOT_READY:#x}u + channel; return; }}"
    );
    source.push_str("  }\n");
    source.push_str("  status->state = 1;\n}\n");
    source
}

/// `emit_grouped_commit_msl` — the grouped counterpart of [`emit_commit`].
pub fn emit_grouped_commit(function_name: &str) -> String {
    let mut source = String::new();
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(&layout::STATUS.emit_msl("M1"));
    source.push_str(grouped_preamble());
    let _ = writeln!(
        source,
        "kernel void {function_name}(const device uchar* lane_bytes [[buffer(0)]], \
         const device M3ChannelMeta* meta [[buffer(1)]], \
         uint lane_index [[thread_position_in_grid]]) {{"
    );
    source.push_str(
        "  const device M3LaneHeader* header = reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n",
    );
    source.push_str("  if (lane_index >= header->lane_count) return;\n");
    source.push_str(
        "  const device M3LaneRecord* lanes = reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n",
    );
    source.push_str("  const M3LaneRecord lane = lanes[lane_index];\n");
    source.push_str(
        "  device M1Status* status = reinterpret_cast<device M1Status*>(lane.commit_slot);\n",
    );
    source.push_str("  if (status->state != 1) return;\n");
    source.push_str("  for (uint channel = 0; channel < header->channel_count; ++channel) {\n");
    source.push_str("    const M3ChannelMeta m = meta[lane.channel_slot_offset + channel];\n");
    source.push_str("    if ((m.flags & 1u) == 0u) continue;\n");
    source.push_str("    device ulong* words = reinterpret_cast<device ulong*>(m.words);\n");
    source.push_str("    const ulong old_head = words[0], old_tail = words[1];\n");
    source.push_str(
        "    if ((m.flags & 8u) != 0u && old_tail > old_head) words[0] = old_head + 1ul;\n",
    );
    source.push_str("    if ((m.flags & 16u) != 0u) words[1] = old_tail + 1ul;\n");
    source.push_str("  }\n");
    source.push_str("  status->state = 4;\n}\n");
    source
}

/// The per-program channel effect table the M1/M2 readiness and commit kernels
/// are baked against.
///
/// Capacity and the take/put flags come from the container; the two readiness
/// bits come from the bound trace's first-op direction table.
///
/// The table is emitted here rather than derived on the far side so that a
/// engine never needs the decoded plan to build it. That keeps the effect
/// kernels the M1 and M2 launch paths bind host-emitted like every other
/// kernel, instead of leaving one kernel family that each engine assembles for
/// itself and can get subtly different.
pub fn channel_effects(bound: &BoundTrace) -> Vec<M1ChannelEffect> {
    let channels = &bound.container.channels;
    let mut effects = Vec::with_capacity(channels.len());
    for (index, channel) in channels.iter().enumerate() {
        let chan = index as u32;
        let mut effect = M1ChannelEffect {
            requires_full: false,
            requires_empty: false,
            take: false,
            put: false,
            capacity: channel.capacity,
        };
        for program in &bound.container.stages {
            for op in &program.ops {
                let Some((use_, used)) = op.channel_use() else {
                    continue;
                };
                if used != chan {
                    continue;
                }
                // The engine advances the ring off these two flags, so a
                // channel op that is not classified here leaves the ring
                // un-advanced rather than raising anything.
                match use_ {
                    ChannelUse::Take => effect.take = true,
                    ChannelUse::Put => effect.put = true,
                    // A peek neither drains nor fills; `requires_full` below
                    // is what keeps it ordered.
                    ChannelUse::Read => {}
                }
            }
        }
        if let Some(entry) = bound.readiness.iter().find(|entry| entry.chan == chan) {
            effect.requires_full = entry.dir == Direction::NeedsFull;
            effect.requires_empty = entry.dir == Direction::NeedsEmpty;
        }
        effects.push(effect);
    }
    effects
}
