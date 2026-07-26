//! The readiness and commit kernels: the guard that decides a lane may run,
//! and the cursor advance that retires it.
//!
//! Single-lane (`M1`) forms specialise the channel effects into the emitted
//! text; the grouped (`M3`) forms are generic and read the same decisions out
//! of a per-channel `M3ChannelMeta` flag word at run time.

use alloc::string::String;
use core::fmt::Write as _;

use super::M1ChannelEffect;
use super::preamble::{common_effect_preamble, emit_word_arguments, grouped_preamble};

/// The lane-table ABI revision the emitted readiness kernel refuses to run
/// against anything else (`PTIR_LANE_TABLE_ABI_VERSION`).
const LANE_TABLE_ABI_VERSION: u32 = pie_plan::LANE_TABLE_ABI_VERSION;

/// `emit_readiness_msl` — one lane, channel effects baked in.
pub fn emit_readiness(function_name: &str, channels: &[M1ChannelEffect]) -> String {
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
    let _ = write!(
        source,
        "  if (header->abi_version != {LANE_TABLE_ABI_VERSION} || header->lane_count != 1 || \
         header->channel_count != {}) {{ status->state = 3; status->fault = 0x100; \
         status->reserved0 = header->abi_version; \
         status->reserved1 = (header->lane_count << 16) | header->channel_count; return; }}\n",
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
            0x200 + channel
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
            0x300 + channel
        );
        if effect.requires_full {
            let _ = writeln!(
                source,
                "    if (tail <= head) {{ status->state = 2; status->fault = {}; return; }}",
                0x400 + channel
            );
        }
        if effect.requires_empty {
            let _ = writeln!(
                source,
                "    if (tail - head >= {}ul) {{ status->state = 2; status->fault = {}; return; }}",
                effect.capacity,
                0x480 + channel
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
                0x500 + channel
            );
        }
        source.push_str("  }\n");
    }
    source.push_str("  status->state = 1;\n}\n");
    source
}

/// `emit_commit_msl` — one lane; advance head for takes, tail for puts.
pub fn emit_commit(function_name: &str, channels: &[M1ChannelEffect]) -> String {
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
    source
}

/// `emit_grouped_readiness_msl` — one thread per lane, effects read from
/// `M3ChannelMeta::flags` instead of baked in.
pub fn emit_grouped_readiness(function_name: &str) -> String {
    let mut source = String::new();
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    source
        .push_str("struct M1Status { uint state; uint fault; uint reserved0; uint reserved1; };\n");
    source.push_str(grouped_preamble());
    let _ = write!(
        source,
        "kernel void {function_name}(const device uchar* lane_bytes [[buffer(0)]], \
         const device M3ChannelMeta* meta [[buffer(1)]], \
         uint lane_index [[thread_position_in_grid]]) {{\n"
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
    source.push_str(
        "    if (words[2] != 0ul || words[3] != 0ul || tail < head) \
         { status->state = 3; status->fault = 0x700u + channel; return; }\n",
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
    source.push_str(
        "    if (retry) { status->state = (m.flags & 32u) != 0u ? 3u : 2u; \
         status->fault = 0x780u + channel; return; }\n",
    );
    source.push_str("  }\n");
    source.push_str("  status->state = 1;\n}\n");
    source
}

/// `emit_grouped_commit_msl` — the grouped counterpart of [`emit_commit`].
pub fn emit_grouped_commit(function_name: &str) -> String {
    let mut source = String::new();
    source.push_str("#include <metal_stdlib>\nusing namespace metal;\n");
    source
        .push_str("struct M1Status { uint state; uint fault; uint reserved0; uint reserved1; };\n");
    source.push_str(grouped_preamble());
    let _ = write!(
        source,
        "kernel void {function_name}(const device uchar* lane_bytes [[buffer(0)]], \
         const device M3ChannelMeta* meta [[buffer(1)]], \
         uint lane_index [[thread_position_in_grid]]) {{\n"
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
