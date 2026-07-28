//! `emit_fused_region_msl` and `emit_grouped_fused_region_msl` — one kernel
//! for a whole region, calling the runtime's op switch per node instead of
//! per dispatch.
//!
//! The single-lane form binds each channel's committed/pending cells directly
//! as buffers (hence the 12-channel cap); the grouped form reads them out of
//! the lane table so one kernel serves every lane in the group, and it grows
//! two inline expansions the single-lane form does not have — the MTP-draft
//! argmax and the logits gather.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use pie_plan::{CompiledStage, Dimension, Region};

use super::METAL_M2_MAX_FUSED_CHANNELS;
use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{library_region_valid, used_channel_slots};
use crate::op_view::{OpView, result_bases};

const OP_PIVOT_THRESHOLD: u8 = 0x58;
const OP_CHAN_TAKE: u8 = 0x90;
const OP_CHAN_READ: u8 = 0x91;
const OP_CHAN_PUT: u8 = 0x92;
const OP_INTRINSIC_VAL: u8 = 0xA0;
const OP_RESHAPE: u8 = 0x39;

const INTR_LOGITS: u16 = 0;
const INTR_MTP_LOGITS: u16 = 1;
const INTR_MTP_DRAFTS: u16 = 6;

fn value_ptr(value: u32) -> String {
    format!("scratch + offsets[{value}]")
}

/// Follow a chain of elided reshapes to the value that actually holds the bytes.
fn resolve_alias(alias: &BTreeMap<u32, u32>, mut value: u32) -> u32 {
    // The chain is acyclic (SSA), but bound the walk anyway.
    for _ in 0..64 {
        match alias.get(&value) {
            Some(&source) => value = source,
            None => break,
        }
    }
    value
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
    slots_for_aliased(op, base, &BTreeMap::new())
}

fn slots_for_aliased(
    op: &OpView,
    base: u32,
    alias: &BTreeMap<u32, u32>,
) -> Slots {
    let value_ptr = |value: u32| value_ptr(resolve_alias(alias, value));
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
    if op.tag == OP_PIVOT_THRESHOLD {
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

/// Threads a grouped region's threadgroup gets per lane. The emitted kernel
/// sizes its threadgroup reduction buffer to this, and `m1_runtime.cpp` must
/// launch exactly this many.
pub const METAL_M3_REGION_THREADS: u32 = 256;

/// Device+threadgroup barrier between two ops of a region.
const BARRIER: &str =
    "  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n";

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
        if op.tag == OP_CHAN_TAKE || op.tag == OP_CHAN_READ {
            if op.chan < 0 || op.chan as usize >= channel_bindings.len() {
                return Err("fused channel root binding out of range".to_string());
            }
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == OP_CHAN_PUT {
            if op.chan < 0 || op.chan as usize >= channel_bindings.len() {
                return Err("fused channel sink binding out of range".to_string());
            }
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == OP_INTRINSIC_VAL {
            slots.a0 = "logits".to_string();
        }
        let _ = writeln!(
            source,
            "  ptir_m1_execute({}u, status, descriptors, params + {node}, {}, {}, {}, {}, {}, temporary);",
            op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
        );
        source.push_str("  if (status->state != 1) return;\n");
        if op.tag == OP_CHAN_PUT {
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
    source.push_str("    uint dispatch_lane [[threadgroup_position_in_grid]],\n");
    source.push_str("    uint m3_tid [[thread_position_in_threadgroup]],\n");
    source.push_str("    uint m3_threads [[threads_per_threadgroup]]) {\n");
    // A lane owns a threadgroup rather than a thread. Everything the region does
    // still happens once per lane -- ops that cannot be partitioned run on thread
    // 0 -- but the ones that walk the whole vocabulary split across it. The
    // driver picks the actual width from the pipeline's own limit and passes it
    // in `m3_threads`; this array only has to be big enough for the largest it
    // will ever ask for.
    let _ = writeln!(
        source,
        "  threadgroup M1ArgmaxCandidate m3_tgbuf[{METAL_M3_REGION_THREADS}];"
    );
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
    // The threadgroup buffer above is sized for exactly this width, and the
    // driver launches it (see `m1_runtime.cpp`). A mismatch is a wiring bug, so
    // say so rather than reading past the buffer.
    let _ = writeln!(
        source,
        "  if (m3_threads > {METAL_M3_REGION_THREADS}u) {{ \
         if (m3_tid == 0) m1_fault(status, 0xB3u); return; }}"
    );
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
    // A reshape that does not change the element count or dtype is a view, but
    // the runtime still executes it as a byte-for-byte copy of the whole value.
    // In the sampler's graph that is a vocabulary-wide round trip through device
    // memory for nothing. Elide it and point its consumers at the source
    // instead, as long as the result stays inside this region -- a region output
    // or sink is read through its own offset by someone else, so those keep the
    // copy.
    let escapes: BTreeSet<u32> = region
        .outputs
        .iter()
        .copied()
        .chain(region.sinks.iter().map(|sink| sink.value))
        .collect();
    let mut alias: BTreeMap<u32, u32> = BTreeMap::new();
    let value_types = &stage.normalized.value_types;
    // Element count, and only when every extent is static: a symbolic dim could
    // resolve differently for the two values and the copy would not be a view.
    let static_elements = |value: u32| -> Option<u64> {
        let ty = value_types.get(value as usize)?;
        let mut total: u64 = 1;
        for dim in &ty.dims {
            match dim {
                Dimension::Static(extent) => total *= u64::from(*extent),
                Dimension::Symbolic(_) => return None,
            }
        }
        Some(total)
    };
    let same_footprint = |a: u32, b: u32| {
        match (value_types.get(a as usize), value_types.get(b as usize)) {
            (Some(x), Some(y)) if x.dtype == y.dtype => {
                matches!((static_elements(a), static_elements(b)),
                         (Some(m), Some(n)) if m == n)
            }
            _ => false,
        }
    };

    for &node in &region.nodes {
        let Some(op) = ops.get(node as usize) else {
            return Err("grouped fused region node out of range".to_string());
        };
        let base = bases[node as usize];
        if op.tag == OP_RESHAPE
            && op.results == 1
            && op.args.len() == 1
            && !escapes.contains(&base)
            && same_footprint(op.args[0], base)
        {
            alias.insert(base, resolve_alias(&alias, op.args[0]));
            continue;
        }
        let mut slots = slots_for_aliased(op, base, &alias);
        if op.tag == OP_INTRINSIC_VAL && op.intr == INTR_MTP_DRAFTS {
            emit_mtp_drafts(&mut source, base, &slots.o0);
            source.push_str(BARRIER);
            continue;
        }
        if op.tag == OP_INTRINSIC_VAL && (op.intr == INTR_LOGITS || op.intr == INTR_MTP_LOGITS) {
            emit_logits_gather(&mut source, base, op.intr == INTR_MTP_LOGITS, &slots.o0);
            source.push_str(BARRIER);
            continue;
        }
        if op.tag == OP_CHAN_TAKE || op.tag == OP_CHAN_READ {
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == OP_CHAN_PUT {
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == OP_INTRINSIC_VAL {
            // `logits` is a `const device bfloat*` here because the gather and
            // the draft argmax above index it as one; `ptir_m1_execute` takes a
            // `const device uchar*`, and MSL will not convert between them. The
            // singleton emitter has no cast because there `logits` arrives as a
            // kernel parameter already typed `uchar*`.
            slots.a0 = "reinterpret_cast<const device uchar*>(logits)".to_string();
        }
        let _ = writeln!(
            source,
            "  ptir_m1_execute_mt({}u, status, descriptors, lane_params + {node}, {}, {}, {}, {}, {}, temporary, m3_tid, m3_threads, m3_tgbuf);",
            op.tag, slots.a0, slots.a1, slots.a2, slots.o0, slots.o1
        );
        // The next op reads what this one wrote, and `status` is how a fault
        // reaches the other threads, so both need to be visible before either
        // is read. The status test is uniform across the threadgroup, so the
        // return below never strands a thread at a later barrier.
        source.push_str(BARRIER);
        source.push_str("  if (status->state != 1) return;\n");
        if op.tag == OP_CHAN_PUT {
            let _ = writeln!(source, "  current_{} = pending_{};", op.chan, op.chan);
            let _ = writeln!(
                source,
                "  if (m3_tid == 0) pending_flags[pending_index_{}] = 1;",
                op.chan
            );
            source.push_str(BARRIER);
        }
    }
    source.push_str("}\n");
    Ok(source)
}

/// The `mtp_drafts` intrinsic: a per-draft argmax over the lane's logits.
fn emit_mtp_drafts(source: &mut String, base: u32, o0: &str) {
    source.push_str("  {\n");
    // Per-draft argmax over the vocabulary. Drafts are few, so partition by
    // draft rather than by column; a lane with one draft keeps one thread busy,
    // which is what the serial emitter did anyway.
    source.push_str("    const uint draft_begin = m3_tid;\n");
    source.push_str("    const uint draft_step = m3_threads;\n");
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
    source.push_str(
        "    for (uint draft = draft_begin; draft < draft_desc.len; \
         draft += draft_step) {\n",
    );
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
    // This walks the whole vocabulary. In the grouped region every thread of
    // the lane's threadgroup reaches it, so it must be split, not repeated.
    source.push_str("    const uint gather_begin = m3_tid;\n");
    source.push_str("    const uint gather_step = m3_threads;\n");
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
    // Row-major, not a flat walk with a divide. `layout` is device memory and
    // `intrinsic_out` is a device store the compiler cannot prove disjoint from
    // it, so a flat loop reloaded `layout->vocab` three times per element and
    // paid a runtime div and mod on top -- a vocabulary-sized row cost ~85ms of
    // an ~89ms decode step. Hoisting the extent turns it into a coalesced copy.
    source.push_str("    const uint gather_vocab = layout->vocab;\n");
    source.push_str(
        "    const uint gather_rows = intrinsic_desc.len / gather_vocab;\n",
    );
    source.push_str(
        "    const uint gather_row_base = row_meta.offset + intrinsic_row_base;\n",
    );
    source.push_str("    for (uint gr = 0u; gr < gather_rows; ++gr) {\n");
    source.push_str(
        "      const uint source_row = row_indices[gather_row_base + gr];\n",
    );
    source.push_str(
        "      const device bfloat* gather_src = logits + \
         ulong(source_row) * gather_vocab;\n",
    );
    source.push_str(
        "      device float* gather_dst = intrinsic_out + ulong(gr) * gather_vocab;\n",
    );
    source.push_str(
        "      for (uint column = gather_begin; column < gather_vocab; \
         column += gather_step) {\n",
    );
    source.push_str(
        "        gather_dst[column] = float(gather_src[column]);\n",
    );
    source.push_str("      }\n");
    source.push_str("    }\n");
    source.push_str("  }\n");
}
