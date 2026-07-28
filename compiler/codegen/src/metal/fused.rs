//! `emit_fused_region_msl` and `emit_grouped_fused_region_msl` — one kernel
//! for a whole region, calling the runtime's op switch per node instead of
//! per dispatch.
//!
//! The single-lane form binds each channel's committed/pending cells directly
//! as buffers (hence the 12-channel cap); the grouped form reads them out of
//! the lane table so one kernel serves every lane in the group, and it grows
//! two inline expansions the single-lane form does not have — the MTP-draft
//! argmax and the logits gather.

use pie_ir::op::{intrinsic_tags, tags};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use pie_plan::{CompiledStage, Region};

use super::METAL_M2_MAX_FUSED_CHANNELS;
use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{intrinsics_bindable, library_region_valid, used_channel_slots};
use crate::op_view::{OpView, result_bases};

fn value_ptr(value: u32) -> String {
    format!("scratch + offsets[{value}]")
}

/// The five operand slots `ptir_m1_execute` takes, defaulted to `scratch` the
/// way the C++ does before the per-tag overrides.
struct Slots {
    a0: String,
    a1: String,
    a2: String,
    o0: String,
    o1: String,
}

fn slots_for(op: &OpView, base: u32) -> Slots {
    let mut slots = Slots {
        a0: "scratch".to_string(),
        a1: "scratch".to_string(),
        a2: "scratch".to_string(),
        o0: "scratch".to_string(),
        o1: "scratch".to_string(),
    };
    if !op.args.is_empty() {
        slots.a0 = value_ptr(op.args[0]);
    }
    if op.args.len() > 1 {
        slots.a1 = value_ptr(op.args[1]);
    }
    if op.args.len() > 2 {
        slots.a2 = value_ptr(op.args[2]);
    }
    if op.tag == tags::PIVOT_THRESHOLD {
        slots.a1 = value_ptr(op.pred_payload);
    }
    if op.results > 0 {
        slots.o0 = value_ptr(base);
    }
    if op.results > 1 {
        slots.o1 = value_ptr(base + 1);
    }
    slots
}

/// `emit_fused_region_msl` — single lane, channels bound directly.
pub fn emit_fused_region(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, String> {
    if !library_region_valid(stage, region) {
        return Err("library region ABI is invalid".to_string());
    }
    let channel_bindings = &stage.normalized.channel_bindings;
    if channel_bindings.len() > METAL_M2_MAX_FUSED_CHANNELS {
        return Err("fused region exceeds the 12-channel direct-binding limit".to_string());
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    intrinsics_bindable(&ops, region)?;
    let bases = result_bases(&ops);

    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    let _ = writeln!(source, "kernel void {function_name}(");
    source.push_str("    device M1Status* status [[buffer(0)]],\n");
    source.push_str("    const device M1ValueDesc* descriptors [[buffer(1)]],\n");
    source.push_str("    const device M1OpParams* params [[buffer(2)]],\n");
    source.push_str("    const device uint* offsets [[buffer(3)]],\n");
    source.push_str("    device uchar* scratch [[buffer(4)]],\n");
    source.push_str("    device uchar* temporary [[buffer(5)]],\n");
    source.push_str("    const device uchar* logits [[buffer(6)]]");
    for channel in 0..channel_bindings.len() {
        let _ = write!(
            source,
            ",\n    const device uchar* committed_{channel} [[buffer({})]],\n\
             \x20   device uchar* pending_{channel} [[buffer({})]]",
            7 + channel * 2,
            8 + channel * 2
        );
    }
    source.push_str(",\n    uint gid [[thread_position_in_grid]]) {\n");
    source.push_str("  if (gid != 0 || status->state != 1) return;\n");
    for channel in 0..channel_bindings.len() {
        let _ = writeln!(
            source,
            "  const device uchar* current_{channel} = committed_{channel};"
        );
    }
    for &node in &region.nodes {
        let Some(op) = ops.get(node as usize) else {
            return Err("fused region node out of range".to_string());
        };
        let mut slots = slots_for(op, bases[node as usize]);
        if op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ {
            if op.chan < 0 || op.chan as usize >= channel_bindings.len() {
                return Err("fused channel root binding out of range".to_string());
            }
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == tags::CHAN_PUT {
            if op.chan < 0 || op.chan as usize >= channel_bindings.len() {
                return Err("fused channel sink binding out of range".to_string());
            }
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == tags::INTRINSIC_VAL {
            slots.a0 = "logits".to_string();
        }
        let _ = writeln!(
            source,
            "  ptir_m1_execute({}u, status, descriptors, params + {node}, {}, {}, {}, {}, {}, temporary);",
            op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
        );
        source.push_str("  if (status->state != 1) return;\n");
        if op.tag == tags::CHAN_PUT {
            let _ = writeln!(source, "  current_{} = pending_{};", op.chan, op.chan);
        }
    }
    source.push_str("}\n");
    Ok(source)
}

/// `emit_grouped_fused_region_msl` — one kernel, one thread per lane.
pub fn emit_grouped_fused_region(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, String> {
    if !library_region_valid(stage, region) {
        return Err("grouped library region ABI is invalid".to_string());
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    intrinsics_bindable(&ops, region)?;
    let bases = result_bases(&ops);
    let channel_count = used_channel_slots(&ops);

    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    source.push_str(grouped_preamble());
    let _ = writeln!(source, "kernel void {function_name}(");
    source.push_str("    const device uchar* lane_bytes [[buffer(0)]],\n");
    source.push_str("    const device M1ValueDesc* all_descriptors [[buffer(1)]],\n");
    source.push_str("    const device M1OpParams* params [[buffer(2)]],\n");
    source.push_str("    const device uint* offsets [[buffer(3)]],\n");
    source.push_str("    device uchar* all_scratch [[buffer(4)]],\n");
    source.push_str("    const device M3GroupLayout* layout [[buffer(5)]],\n");
    source.push_str("    const device uint* channel_bindings [[buffer(6)]],\n");
    source.push_str("    device uchar* pending_flags [[buffer(7)]],\n");
    source.push_str("    const device uint* lane_indices [[buffer(8)]],\n");
    source.push_str("    const device M3RowMeta* all_row_meta [[buffer(9)]],\n");
    source.push_str("    const device uint* row_indices [[buffer(10)]],\n");
    source.push_str("    uint dispatch_lane [[thread_position_in_grid]]) {\n");
    source.push_str("  if (dispatch_lane >= layout->lane_count) return;\n");
    source.push_str("  const uint lane_index = lane_indices[dispatch_lane];\n");
    source.push_str(
        "  const device M3LaneHeader* header = \
         reinterpret_cast<const device M3LaneHeader*>(lane_bytes);\n",
    );
    source.push_str(
        "  const device M3LaneRecord* lanes = \
         reinterpret_cast<const device M3LaneRecord*>(lane_bytes + sizeof(M3LaneHeader));\n",
    );
    source.push_str(
        "  const device M3LaneChannelSlot* slots = \
         reinterpret_cast<const device M3LaneChannelSlot*>(lane_bytes + \
         sizeof(M3LaneHeader) + header->lane_count * sizeof(M3LaneRecord));\n",
    );
    source.push_str("  const M3LaneRecord lane = lanes[lane_index];\n");
    source.push_str("  const M3RowMeta row_meta = all_row_meta[lane_index];\n");
    source.push_str(
        "  device M1Status* status = \
         reinterpret_cast<device M1Status*>(lane.commit_slot);\n",
    );
    source.push_str("  if (status->state != 1) return;\n");
    source.push_str(
        "  const device M1ValueDesc* descriptors = all_descriptors + \
         dispatch_lane * layout->value_count;\n",
    );
    source.push_str(
        "  const device M1OpParams* lane_params = params + \
         dispatch_lane * layout->reserved2;\n",
    );
    source.push_str(
        "  device uchar* scratch = all_scratch + dispatch_lane * layout->scratch_stride;\n",
    );
    source.push_str("  device uchar* temporary = scratch + layout->temporary_offset;\n");
    source.push_str(
        "  const device bfloat* logits = \
         reinterpret_cast<const device bfloat*>(lane.logits_base);\n",
    );
    for channel in 0..channel_count {
        let _ = writeln!(
            source,
            "  const uint dense_{channel} = channel_bindings[dispatch_lane * layout->reserved0 + {channel}];"
        );
        let _ = writeln!(
            source,
            "  const M3LaneChannelSlot channel_{channel} = slots[lane.channel_slot_offset + dense_{channel}];"
        );
        let _ = writeln!(
            source,
            "  const uint pending_index_{channel} = lane.channel_slot_offset + dense_{channel};"
        );
        let _ = writeln!(
            source,
            "  const device uchar* current_{channel} = reinterpret_cast<const device uchar*>(\
             pending_flags[pending_index_{channel}] != 0 ? channel_{channel}.pending_cell : \
             channel_{channel}.committed_cell);"
        );
        let _ = writeln!(
            source,
            "  device uchar* pending_{channel} = reinterpret_cast<device uchar*>(\
             channel_{channel}.pending_cell);"
        );
    }
    for &node in &region.nodes {
        let Some(op) = ops.get(node as usize) else {
            return Err("grouped fused region node out of range".to_string());
        };
        let base = bases[node as usize];
        let mut slots = slots_for(op, base);
        if op.tag == tags::INTRINSIC_VAL && op.intr == intrinsic_tags::MTP_DRAFTS {
            emit_mtp_drafts(&mut source, base, &slots.o0);
            continue;
        }
        if op.tag == tags::INTRINSIC_VAL && (op.intr == intrinsic_tags::LOGITS || op.intr == intrinsic_tags::MTP_LOGITS) {
            emit_logits_gather(&mut source, base, op.intr == intrinsic_tags::MTP_LOGITS, &slots.o0);
            continue;
        }
        if op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ {
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == tags::CHAN_PUT {
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == tags::INTRINSIC_VAL {
            slots.a0 = "logits".to_string();
        }
        let _ = writeln!(
            source,
            "  ptir_m1_execute({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary);",
            op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
        );
        source.push_str("  if (status->state != 1) return;\n");
        if op.tag == tags::CHAN_PUT {
            let _ = writeln!(source, "  current_{} = pending_{};", op.chan, op.chan);
            let _ = writeln!(source, "  pending_flags[pending_index_{}] = 1;", op.chan);
        }
    }
    source.push_str("}\n");
    Ok(source)
}

/// The `mtp_drafts` intrinsic: a per-draft argmax over the lane's logits.
fn emit_mtp_drafts(source: &mut String, base: u32, o0: &str) {
    source.push_str("  {\n");
    let _ = writeln!(
        source,
        "    const M1ValueDesc draft_desc = descriptors[{base}];"
    );
    source.push_str(
        "    if (layout->vocab == 0u || row_meta.mtp_offset > row_meta.count || \
         draft_desc.len > row_meta.count - row_meta.mtp_offset) \
         { m1_fault(status, 0xA0u); return; }\n",
    );
    let _ = writeln!(
        source,
        "    device int* draft_out = reinterpret_cast<device int*>({o0});"
    );
    source.push_str("    for (uint draft = 0; draft < draft_desc.len; ++draft) {\n");
    source.push_str(
        "      const uint source_row = \
         row_indices[row_meta.offset + row_meta.mtp_offset + draft];\n",
    );
    source.push_str("      float best_value = -INFINITY;\n");
    source.push_str("      uint best_index = 0u;\n");
    source.push_str("      bool have = false;\n");
    source.push_str("      for (uint column = 0; column < layout->vocab; ++column) {\n");
    source.push_str(
        "        const float value = float(logits[ulong(source_row) * layout->vocab + column]);\n",
    );
    source.push_str(
        "        if (!isnan(value) && (!have || value > best_value || \
         (value == best_value && column < best_index))) { \
         best_value = value; best_index = column; have = true; }\n",
    );
    source.push_str("      }\n");
    source.push_str("      draft_out[draft] = int(have ? best_index : 0u);\n");
    source.push_str("    }\n");
    source.push_str("  }\n");
}

/// The `logits` / `mtp_logits` intrinsics: a strided gather out of the lane's
/// logits buffer, rebased for MTP rows.
fn emit_logits_gather(source: &mut String, base: u32, mtp: bool, o0: &str) {
    source.push_str("  {\n");
    let _ = writeln!(
        source,
        "    const M1ValueDesc intrinsic_desc = descriptors[{base}];"
    );
    let _ = writeln!(
        source,
        "    const uint intrinsic_row_base = {};",
        if mtp { "row_meta.mtp_offset" } else { "0u" }
    );
    source.push_str(
        "    if (layout->vocab == 0u || intrinsic_desc.len % layout->vocab != 0u || \
         intrinsic_row_base > row_meta.count || \
         intrinsic_desc.len / layout->vocab > row_meta.count - intrinsic_row_base) \
         { m1_fault(status, 0xA0u); return; }\n",
    );
    let _ = writeln!(
        source,
        "    device float* intrinsic_out = reinterpret_cast<device float*>({o0});"
    );
    source.push_str("    for (uint element = 0; element < intrinsic_desc.len; ++element) {\n");
    source.push_str("      const uint logical_row = element / layout->vocab;\n");
    source.push_str("      const uint column = element % layout->vocab;\n");
    source.push_str(
        "      const uint source_row = \
         row_indices[row_meta.offset + intrinsic_row_base + logical_row];\n",
    );
    source.push_str(
        "      intrinsic_out[element] = \
         float(logits[ulong(source_row) * layout->vocab + column]);\n",
    );
    source.push_str("    }\n");
    source.push_str("  }\n");
}
