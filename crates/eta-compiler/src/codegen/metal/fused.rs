//! `emit_fused_region_msl` and `emit_grouped_fused_region_msl` — one kernel
//! for a whole region, calling the runtime's op switch per node instead of
//! per dispatch.
//!
//! The single-lane form binds each channel's committed/pending cells directly
//! as buffers (hence the 12-channel cap — eleven for a region that also reads
//! a second intrinsic rectangle, which binds down from the top of the same
//! index space; see [`super::intrinsics`]); the grouped form reads them out of
//! the lane table so one kernel serves every lane in the group, and it grows
//! two inline expansions the single-lane form does not have — the MTP-draft
//! argmax and the logits gather.

use crate::codegen::error::{EmitError, EmitterKind, RegionForm};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::{intrinsic_tags, tags};

use crate::plan::{CompiledStage, Region};

use super::METAL_M2_MAX_FUSED_CHANNELS;
use super::intrinsics::{M2_LOGITS_BUFFER, fused_channel_ceiling, m2_intrinsic_buffer};
use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{
    grouped_intrinsics_bindable, intrinsics_bindable, library_region_valid, used_channel_slots,
};
use crate::codegen::fault::{FUSED_GEOMETRY_MISMATCH, M3_THREADS_EXCEEDED};
use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::slots::Slots;

fn value_ptr(value: u32) -> String {
    format!("scratch + offsets[{value}]")
}

/// The kernel argument holding intrinsic `intr`'s rectangle.
///
/// The trunk keeps the name it has always had — `logits` reads as itself in
/// the emitted source and in every golden — and the others are named by id
/// rather than by spelling, so a new intrinsic is a row in
/// [`m2_intrinsic_buffer`] and not a name this file has to learn.
fn intrinsic_slot_name(intr: u16) -> String {
    if m2_intrinsic_buffer(intr) == Some(M2_LOGITS_BUFFER) {
        return "logits".to_string();
    }
    format!("intrinsic_{intr}")
}

/// The intrinsic ids `region` reads that need a rectangle of their own, in
/// ascending id order.
///
/// Scoped to the REGION rather than the stage, exactly as
/// [`intrinsics_bindable`] is: a sibling region in the same stage may read a
/// different set, and each region is its own kernel with its own signature.
/// The trunk's `logits` is not in here — its index sits below the channels
/// and is written unconditionally.
fn extra_intrinsics(ops: &[OpView], region: &Region) -> Vec<u16> {
    let mut used = BTreeSet::new();
    for &node in &region.nodes {
        let Some(op) = ops.get(node.index()) else {
            continue;
        };
        if op.tag == tags::INTRINSIC_VAL
            && m2_intrinsic_buffer(op.intr).is_some_and(|at| at != M2_LOGITS_BUFFER)
        {
            used.insert(op.intr);
        }
    }
    used.into_iter().collect()
}

/// Threads a grouped region's threadgroup gets per lane. The emitted kernel
/// sizes its threadgroup reduction buffer to this; the engine launches the
/// narrower of it and the pipeline's own maxTotalThreadsPerThreadgroup, and
/// the kernel faults `0xB3` on a wider launch rather than reading past the
/// buffer.
///
/// It must not be transcribed on the engine side: a hand-kept copy carrying a
/// "must equal" comment has nothing comparing the two, and the failure mode is
/// a threadgroup sized for one count reducing over a buffer built for another.
/// This was published as `PTIR_METAL_M3_REGION_THREADS` in the generated
/// `ptir_abi.h` while the driver was C++; the Rust `engine-metal` reads this
/// constant itself, and `grouped.rs` asserts its mirror against it.
///
/// 512 measured against 256 with the model DAG truncated away, interleaved to
/// cancel thermal drift: 0.951ms vs 1.557ms for the sampler region, reproduced
/// twice. 1024 is not better (0.963ms) and costs twice the threadgroup memory.
pub const METAL_M3_REGION_THREADS: u32 = 512;

/// Device+threadgroup barrier between two ops of a region.
const BARRIER: &str =
    "  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);\n";

/// `emit_fused_region_msl` — single lane, channels bound directly.
pub fn emit_fused_region(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !library_region_valid(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::Unnamed));
    }
    let channel_bindings = &stage.normalized.channel_bindings;
    if channel_bindings.len() > METAL_M2_MAX_FUSED_CHANNELS {
        return Err(EmitError::ChannelLimitExceeded {
            emitter: EmitterKind::MetalFused,
            limit: METAL_M2_MAX_FUSED_CHANNELS,
        });
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    intrinsics_bindable(&ops, region)?;
    let bases = result_bases(&ops);

    // ── **THE SLOT TABLE** (`super::intrinsics`). Every intrinsic this
    //    region reads other than the trunk's `logits` gets a rectangle of its
    //    own, at a buffer index fixed per intrinsic id and taken from the TOP
    //    of Metal's argument space. Emitted only for the ids the region
    //    actually reads, so a stage that reads nothing but `logits` — which is
    //    every stage this emitter served before the table existed — emits the
    //    bytes it always did.
    let extra = extra_intrinsics(&ops, region);
    let ceiling = fused_channel_ceiling(&extra);
    if channel_bindings.len() > ceiling {
        // The channels grow up from 7 and these grow down from 30; a region
        // that let them meet would bind a rectangle at the index a channel
        // cell already holds, and a take would answer with logits.
        return Err(EmitError::ChannelLimitExceeded {
            emitter: EmitterKind::MetalFused,
            limit: ceiling,
        });
    }

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
    for &intr in &extra {
        let at = m2_intrinsic_buffer(intr).expect("`extra_intrinsics` filters to bindable ids");
        let _ = write!(
            source,
            ",\n    const device uchar* {} [[buffer({at})]]",
            intrinsic_slot_name(intr)
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
        let node = node.index();
        let Some(op) = ops.get(node) else {
            return Err(EmitError::RegionNodeOutOfRange(RegionForm::Fused));
        };
        let mut slots = Slots::of(op, bases[node], value_ptr);
        if op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ {
            if op.chan < 0 || op.chan as usize >= channel_bindings.len() {
                return Err(EmitError::ChannelRootBindingOutOfRange);
            }
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == tags::CHAN_PUT {
            if op.chan < 0 || op.chan as usize >= channel_bindings.len() {
                return Err(EmitError::ChannelSinkBindingOutOfRange);
            }
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == tags::INTRINSIC_VAL {
            // **THE RECTANGLE IS PICKED BY ID, AND IT USED TO BE PICKED BY
            // NOTHING.** Every `INTRINSIC_VAL` op took `logits` here, so a
            // stage reading `mtp_logits` beside `logits` read one rectangle
            // twice — no fault, no bounds violation, the draft column's rows
            // silently answered by the trunk's. That is the mis-binding
            // `metal_intrinsic_supported` was written to keep out of the
            // emitter; now there is somewhere for the second one to go.
            slots.a0 = intrinsic_slot_name(op.intr);
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
) -> Result<String, EmitError> {
    if !library_region_valid(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedFused));
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    // **THE GROUPED PATH KEEPS THE NARROWER LIST**, and the id it still
    // refuses is `attn_score`: a grouped kernel binds no per-intrinsic buffer
    // at all — every `INTRINSIC_VAL` op is handed `lane.logits_base` out of
    // the lane record — and the score slab is not that allocation. See
    // `grouped_intrinsics_bindable`.
    grouped_intrinsics_bindable(&ops, region)?;
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
    // engine picks the actual width from the pipeline's own limit and passes it
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
    // The threadgroup buffer above is sized for this width and the engine asks
    // for no more than it, so a wider launch would read past the buffer. Say so
    // rather than doing it.
    let _ = writeln!(
        source,
        "  if (m3_threads > {METAL_M3_REGION_THREADS}u) {{ \
         if (m3_tid == 0) m1_fault(status, {M3_THREADS_EXCEEDED:#X}u); return; }}"
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
    let escapes = crate::codegen::alias::escaping_values(region);
    let value_types = &stage.normalized.value_types;
    let covers =
        |source: u32, result: u32| crate::codegen::alias::covers(value_types, source, result);

    // Decided before anything is emitted, because the gather/argmax fusion below
    // has to see through these aliases to recognise its pattern.
    let is_view_reshape = |node: usize| -> bool {
        let Some(op) = ops.get(node) else {
            return false;
        };
        op.tag == tags::RESHAPE
            && op.results == 1
            && op.args.len() == 1
            && !escapes.contains(&bases[node])
            && covers(op.args[0], bases[node])
    };
    let mut alias = crate::codegen::alias::AliasTable::new();
    for &node in &region.nodes {
        let node = node.index();
        if is_view_reshape(node) {
            let arg = ops[node].args[0];
            alias.elide(bases[node], arg);
        }
    }

    // A logits gather whose only consumer is an argmax does not need to exist:
    // the pair fuses into one pass over the bf16 row. Decided up front because
    // the gather is emitted before the argmax is reached.
    let mut consumers: BTreeMap<u32, usize> = BTreeMap::new();
    for &node in &region.nodes {
        if is_view_reshape(node.index()) {
            continue;
        }
        if let Some(op) = ops.get(node.index()) {
            for &arg in &op.args {
                *consumers.entry(alias.resolve(arg)).or_insert(0) += 1;
            }
        }
    }
    let mut fused_argmax: BTreeMap<usize, usize> = BTreeMap::new();
    let mut elided_gather: BTreeSet<usize> = BTreeSet::new();
    for &node in &region.nodes {
        let node = node.index();
        let Some(op) = ops.get(node) else { continue };
        if op.tag != tags::REDUCE_ARGMAX || op.args.len() != 1 {
            continue;
        }
        let source_value = alias.resolve(op.args[0]);
        if consumers.get(&source_value).copied().unwrap_or(0) != 1
            || escapes.contains(&source_value)
        {
            continue;
        }
        let producer = region.nodes.iter().map(|n| n.index()).find(|&n| {
            bases[n] == source_value
                && ops.get(n).is_some_and(|p| {
                    p.tag == tags::INTRINSIC_VAL
                        && (p.intr == intrinsic_tags::LOGITS
                            || p.intr == intrinsic_tags::MTP_LOGITS)
                })
        });
        if let Some(producer) = producer {
            fused_argmax.insert(node, producer);
            elided_gather.insert(producer);
        }
    }

    for &node in &region.nodes {
        let node = node.index();
        let Some(op) = ops.get(node) else {
            return Err(EmitError::RegionNodeOutOfRange(RegionForm::GroupedFused));
        };
        let base = bases[node];
        if elided_gather.contains(&node) {
            continue;
        }
        if let Some(&producer) = fused_argmax.get(&node) {
            let slots = Slots::of(op, base, |value| value_ptr(alias.resolve(value)));
            emit_logits_argmax(
                &mut source,
                bases[producer],
                ops[producer].intr == intrinsic_tags::MTP_LOGITS,
                &slots.o0,
            );
            source.push_str(BARRIER);
            source.push_str("  if (status->state != 1) return;\n");
            continue;
        }
        if is_view_reshape(node) {
            continue;
        }
        let mut slots = Slots::of(op, base, |value| value_ptr(alias.resolve(value)));
        if op.tag == tags::INTRINSIC_VAL && op.intr == intrinsic_tags::MTP_DRAFTS {
            emit_mtp_drafts(&mut source, base, &slots.o0);
            source.push_str(BARRIER);
            continue;
        }
        if op.tag == tags::INTRINSIC_VAL
            && (op.intr == intrinsic_tags::LOGITS || op.intr == intrinsic_tags::MTP_LOGITS)
        {
            emit_logits_gather(
                &mut source,
                base,
                op.intr == intrinsic_tags::MTP_LOGITS,
                &slots.o0,
            );
            source.push_str(BARRIER);
            continue;
        }
        if op.tag == tags::CHAN_TAKE || op.tag == tags::CHAN_READ {
            slots.a0 = format!("current_{}", op.chan);
        } else if op.tag == tags::CHAN_PUT {
            slots.o0 = format!("pending_{}", op.chan);
        } else if op.tag == tags::INTRINSIC_VAL {
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
        if op.tag == tags::CHAN_PUT {
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
    let _ = writeln!(
        source,
        "    if (layout->vocab == 0u || row_meta.mtp_offset > row_meta.count || \
         draft_desc.len > row_meta.count - row_meta.mtp_offset) \
         {{ m1_fault(status, {FUSED_GEOMETRY_MISMATCH:#X}u); return; }}"
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

/// `argmax(logits)` without materializing the logits.
///
/// The gather's only job in this shape is to widen bf16 to f32 so the generic
/// reduction can read it, and bf16 -> f32 is exact, so the argmax over the
/// stored halves has the same value and the same index. Fusing the two removes a
/// vocabulary-wide f32 write and the read back of it, which was the sampler's
/// whole remaining traffic for a graph that only wants one integer.
fn emit_logits_argmax(source: &mut String, in_base: u32, mtp: bool, o0: &str) {
    source.push_str("  {\n");
    let _ = writeln!(
        source,
        "    const M1ValueDesc am_in = descriptors[{in_base}];"
    );
    let _ = writeln!(
        source,
        "    const uint am_row_base = row_meta.offset + {};",
        if mtp { "row_meta.mtp_offset" } else { "0u" }
    );
    source.push_str("    const uint am_vocab = layout->vocab;\n");
    let _ = writeln!(
        source,
        "    if (am_vocab == 0u || am_in.last != am_vocab || am_in.rows > row_meta.count) \
         {{ m1_fault(status, {FUSED_GEOMETRY_MISMATCH:#X}u); return; }}"
    );
    let _ = writeln!(
        source,
        "    device int* am_out = reinterpret_cast<device int*>({o0});"
    );
    source.push_str("    for (uint am_r = 0u; am_r < am_in.rows; ++am_r) {\n");
    source.push_str("      const uint am_src_row = row_indices[am_row_base + am_r];\n");
    source.push_str("      const device bfloat* am_src = logits + ulong(am_src_row) * am_vocab;\n");
    source.push_str("      M1ArgmaxCandidate am_best = {-INFINITY, 0u, 0u, 0u};\n");
    // Four independent accumulators. The combine is a strict total order, so
    // splitting the fold changes nothing, and it breaks the dependency chain
    // that otherwise serialises one device load per iteration in each thread.
    source.push_str("      M1ArgmaxCandidate am_b1 = am_best, am_b2 = am_best, am_b3 = am_best;\n");
    let _ = writeln!(
        source,
        "      constexpr uint am_w = {METAL_M3_REGION_THREADS}u;"
    );
    source.push_str("      uint am_c = m3_tid;\n");
    source.push_str("      for (; am_c + 3u * am_w < am_vocab; am_c += 4u * am_w) {\n");
    source.push_str("        const float v0 = float(am_src[am_c]);\n");
    source.push_str("        const float v1 = float(am_src[am_c + am_w]);\n");
    source.push_str("        const float v2 = float(am_src[am_c + 2u * am_w]);\n");
    source.push_str("        const float v3 = float(am_src[am_c + 3u * am_w]);\n");
    source.push_str("        am_best = m1_argmax_combine(am_best, M1ArgmaxCandidate{v0, am_c, isnan(v0) ? 0u : 1u, 0u});\n");
    source.push_str("        am_b1 = m1_argmax_combine(am_b1, M1ArgmaxCandidate{v1, am_c + am_w, isnan(v1) ? 0u : 1u, 0u});\n");
    source.push_str("        am_b2 = m1_argmax_combine(am_b2, M1ArgmaxCandidate{v2, am_c + 2u * am_w, isnan(v2) ? 0u : 1u, 0u});\n");
    source.push_str("        am_b3 = m1_argmax_combine(am_b3, M1ArgmaxCandidate{v3, am_c + 3u * am_w, isnan(v3) ? 0u : 1u, 0u});\n");
    source.push_str("      }\n");
    source.push_str("      for (; am_c < am_vocab; am_c += am_w) {\n");
    source.push_str("        const float am_v = float(am_src[am_c]);\n");
    source.push_str("        am_best = m1_argmax_combine(am_best, M1ArgmaxCandidate{am_v, am_c, isnan(am_v) ? 0u : 1u, 0u});\n");
    source.push_str("      }\n");
    source.push_str("      am_best = m1_argmax_combine(m1_argmax_combine(am_best, am_b1), m1_argmax_combine(am_b2, am_b3));\n");
    source.push_str("      m3_tgbuf[m3_tid] = am_best;\n");
    source.push_str("      threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("      for (uint am_s = 1u; am_s < m3_threads; am_s <<= 1) {\n");
    source.push_str(
        "        if ((m3_tid % (2u * am_s)) == 0u && m3_tid + am_s < m3_threads) m3_tgbuf[m3_tid] = m1_argmax_combine(m3_tgbuf[m3_tid], m3_tgbuf[m3_tid + am_s]);\n",
    );
    source.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    source.push_str("      }\n");
    source.push_str("      if (m3_tid == 0) am_out[am_r] = int(m3_tgbuf[0].index);\n");
    source.push_str("      threadgroup_barrier(mem_flags::mem_threadgroup);\n");
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
    let _ = writeln!(
        source,
        "    if (layout->vocab == 0u || intrinsic_desc.len % layout->vocab != 0u || \
         intrinsic_row_base > row_meta.count || \
         intrinsic_desc.len / layout->vocab > row_meta.count - intrinsic_row_base) \
         {{ m1_fault(status, {FUSED_GEOMETRY_MISMATCH:#X}u); return; }}"
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
    source.push_str("    const uint gather_rows = intrinsic_desc.len / gather_vocab;\n");
    source.push_str("    const uint gather_row_base = row_meta.offset + intrinsic_row_base;\n");
    source.push_str("    for (uint gr = 0u; gr < gather_rows; ++gr) {\n");
    source.push_str("      const uint source_row = row_indices[gather_row_base + gr];\n");
    source.push_str(
        "      const device bfloat* gather_src = logits + \
         ulong(source_row) * gather_vocab;\n",
    );
    source.push_str("      device float* gather_dst = intrinsic_out + ulong(gr) * gather_vocab;\n");
    source.push_str(
        "      for (uint column = gather_begin; column < gather_vocab; \
         column += gather_step) {\n",
    );
    source.push_str("        gather_dst[column] = float(gather_src[column]);\n");
    source.push_str("      }\n");
    source.push_str("    }\n");
    source.push_str("  }\n");
}
