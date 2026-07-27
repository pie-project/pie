//! `emit_fused_region_cuda` — one kernel for a whole generated region.
//!
//! The region's ops are emitted inline, in plan order, into a single
//! `__global__` body: each op resolves its operands to scratch offsets and
//! calls the runtime's block-parallel helper for its family, falling back to
//! the single-thread `ptir_m1_execute` switch for the ops that have no
//! parallel form. Between ops the block syncs and bails out if the status word
//! moved off `1`, which is what keeps a fused region pass-atomic like the
//! one-op-per-launch path.
//!
//! Two analyses run before emission:
//!
//! * **reshape aliasing** — a `reshape` whose result never leaves the region is
//!   not emitted at all; its result id is aliased to its input's, so the copy
//!   never exists rather than being generated and skipped.
//! * **direct argmax** ([`analyze_direct_argmax`]) — an `argmax` fed by a
//!   logits intrinsic through nothing but reshapes reads the intrinsic's device
//!   buffer straight, which makes both the intrinsic materialisation and the
//!   reshapes redundant.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;

use pie_plan::{CompiledStage, Dimension, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use crate::op_view::{OpView, result_bases};

use super::runtime::singleton_runtime_source;
use super::singleton::valid_identifier;
use super::validate::validate_generated_region;

const PROLOGUE: &str = include_str!("../../runtime/cuda/fused_block0.cuh");
const SIGNATURE: &str = include_str!("../../runtime/cuda/fused_block1.cuh");
const PREAMBLE: &str = include_str!("../../runtime/cuda/fused_block2.cuh");

/// `kPtirIntrinsicSlots` — per-lane intrinsic descriptor slots.
pub const PTIR_INTRINSIC_SLOTS: u32 = 8;

const OP_EXP: u8 = 0x01;
const OP_CAST: u8 = 0x07;
const OP_ADD: u8 = 0x10;
const OP_SELECT: u8 = 0x20;
const OP_CONST: u8 = 0x81;
const OP_IOTA: u8 = 0x64;
const OP_BROADCAST: u8 = 0x38;
const OP_RESHAPE: u8 = 0x39;
const OP_TRANSPOSE: u8 = 0x3A;
const OP_REDUCE_SUM: u8 = 0x30;
const OP_REDUCE_MAX: u8 = 0x31;
const OP_REDUCE_MIN: u8 = 0x32;
const OP_REDUCE_ARGMAX: u8 = 0x33;
const OP_PIVOT_THRESHOLD: u8 = 0x58;
const OP_GATHER: u8 = 0x60;
const OP_GATHER_ROW: u8 = 0x61;
const OP_SCATTER_ADD: u8 = 0x62;
const OP_SCATTER_SET: u8 = 0x63;
const OP_MASK_APPLY_PACKED: u8 = 0x65;
const OP_CAUSAL_MASK: u8 = 0x66;
const OP_SLIDING_WINDOW_MASK: u8 = 0x67;
const OP_SINK_WINDOW_MASK: u8 = 0x68;
const OP_RNG: u8 = 0x70;
const OP_RNG_KEYED: u8 = 0x71;
const OP_CHAN_TAKE: u8 = 0x90;
const OP_CHAN_READ: u8 = 0x91;
const OP_CHAN_PUT: u8 = 0x92;
const OP_INTRINSIC_VAL: u8 = 0xA0;

const INTR_LOGITS: u16 = 0;
const INTR_MTP_LOGITS: u16 = 1;
const INTR_LAYER: u16 = 5;
const INTR_MTP_DRAFTS: u16 = 6;

const DT_F32: u8 = 0;

/// The ops the runtime has a block-parallel elementwise helper for.
fn parallel_elementwise(tag: u8) -> bool {
    (OP_EXP..=OP_CAST).contains(&tag)
        || (OP_ADD..=OP_SELECT).contains(&tag)
        || matches!(
            tag,
            OP_IOTA
                | OP_MASK_APPLY_PACKED
                | OP_CAUSAL_MASK
                | OP_SLIDING_WINDOW_MASK
                | OP_SINK_WINDOW_MASK
                | OP_RNG
                | OP_RNG_KEYED
        )
}

/// A value's row decomposition: everything but the trailing dim is rows.
#[derive(Clone, Copy, PartialEq, Eq)]
struct RowShape {
    fixed_rows: u64,
    row_extent: u32,
    width: u32,
}

fn row_shape(dims: &[Dimension]) -> Option<RowShape> {
    let mut shape = RowShape {
        fixed_rows: 1,
        row_extent: u32::MAX,
        width: 1,
    };
    if dims.len() >= 2 {
        for dimension in &dims[..dims.len() - 1] {
            match dimension {
                Dimension::Symbolic(role) => {
                    if shape.row_extent != u32::MAX {
                        return None;
                    }
                    shape.row_extent = *role as u32;
                }
                Dimension::Static(value) => {
                    if *value == 0 || shape.fixed_rows > u64::MAX / *value as u64 {
                        return None;
                    }
                    shape.fixed_rows *= *value as u64;
                }
            }
        }
    }
    if let Some(last) = dims.last() {
        let Dimension::Static(width) = last else {
            return None;
        };
        if *width == 0 {
            return None;
        }
        shape.width = *width;
    }
    Some(shape)
}

/// Which `argmax` nodes may read a logits intrinsic's buffer directly, and
/// which nodes that makes redundant.
struct DirectArgmax {
    intrinsic: Vec<u16>,
    skipped: Vec<u8>,
}

fn analyze_direct_argmax(stage: &CompiledStage, region: &Region, bases: &[u32]) -> DirectArgmax {
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let value_count = stage.normalized.value_types.len();
    let mut producers = vec![u32::MAX; value_count];
    let mut consumers: Vec<Vec<u32>> = vec![Vec::new(); value_count];
    for (node, op) in ops.iter().enumerate() {
        for result in 0..op.results {
            producers[(bases[node] + result) as usize] = node as u32;
        }
        for argument in &op.args {
            consumers[*argument as usize].push(node as u32);
        }
    }
    let mut analysis = DirectArgmax {
        intrinsic: vec![u16::MAX; ops.len()],
        skipped: vec![0; ops.len()],
    };

    for &node in &region.nodes {
        let node = node as usize;
        let reduction = &ops[node];
        if reduction.tag != OP_REDUCE_ARGMAX || reduction.args.is_empty() {
            continue;
        }
        let mut value = reduction.args[0];
        let mut expected_consumer = node as u32;
        let mut chain: Vec<u32> = Vec::new();
        while (value as usize) < producers.len()
            && producers[value as usize] != u32::MAX
            && consumers[value as usize].len() == 1
            && consumers[value as usize][0] == expected_consumer
        {
            let producer = producers[value as usize];
            let op = &ops[producer as usize];
            chain.push(producer);
            if op.tag == OP_RESHAPE && !op.args.is_empty() {
                expected_consumer = producer;
                value = op.args[0];
                continue;
            }
            if op.tag != OP_INTRINSIC_VAL || (op.intr != INTR_LOGITS && op.intr != INTR_MTP_LOGITS)
            {
                break;
            }
            let source_shape =
                row_shape(&stage.normalized.value_types[bases[producer as usize] as usize].dims);
            let reduction_shape =
                row_shape(&stage.normalized.value_types[reduction.args[0] as usize].dims);
            let exact_shape = source_shape.is_some() && reduction_shape == source_shape;
            let runtime_single_row = match (source_shape, reduction_shape) {
                (Some(source), Some(target)) => {
                    source.width == target.width
                        && source.fixed_rows == 1
                        && target.fixed_rows == 1
                        && source.row_extent != u32::MAX
                        && target.row_extent == u32::MAX
                }
                _ => false,
            };
            if exact_shape || runtime_single_row {
                analysis.intrinsic[node] = op.intr;
                for &skipped in &chain {
                    analysis.skipped[skipped as usize] = 1;
                }
            }
            break;
        }
    }
    analysis
}

fn resolve_alias(aliases: &[u32], mut value: u32) -> u32 {
    while aliases[value as usize] != value {
        value = aliases[value as usize];
    }
    value
}

/// `emit_fused_region_cuda`.
pub fn emit_fused_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, String> {
    if !valid_identifier(entry_name) {
        return Err("CUDA fused entry name is not a C identifier".to_string());
    }
    validate_generated_region(stage, region)?;

    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let next_value: u32 = ops.iter().map(|op| op.results).sum();
    if next_value as usize != stage.normalized.value_types.len() {
        return Err("fused stage value layout does not match normalized ops".to_string());
    }

    let mut aliases: Vec<u32> = (0..next_value).collect();
    let direct = analyze_direct_argmax(stage, region, &bases);
    let mut skipped = direct.skipped;

    // A nucleus library region elsewhere in the stage consumes the logits and
    // the sampler state directly, so whatever produced them here is redundant.
    for candidate in &stage.fused.regions {
        if !matches!(
            candidate.kind,
            RegionKind::Library(LibraryOp::NucleusSample)
        ) || candidate.inputs.len() != 5
        {
            continue;
        }
        for start in [candidate.inputs[0], candidate.inputs[2]] {
            let mut value = start;
            let mut depth = 0;
            while depth < 2 {
                let mut found = false;
                for &node in &region.nodes {
                    let node = node as usize;
                    let op = &ops[node];
                    if value < bases[node] || value >= bases[node] + op.results {
                        continue;
                    }
                    found = true;
                    skipped[node] = 1;
                    if op.tag == OP_RESHAPE && !op.args.is_empty() {
                        value = op.args[0];
                    } else {
                        depth = 2;
                    }
                    break;
                }
                if !found {
                    break;
                }
                depth += 1;
            }
        }
    }

    let mut source = singleton_runtime_source();
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);

    for &node in &region.nodes {
        let node = node as usize;
        let op = &ops[node];
        let base = bases[node];
        if skipped[node] != 0 && op.tag != OP_RESHAPE {
            continue;
        }
        if op.tag == OP_RESHAPE && !region.outputs.contains(&base) {
            aliases[base as usize] = resolve_alias(&aliases, op.args[0]);
            continue;
        }

        let mut a0 = "scratch".to_string();
        let mut a1 = "scratch".to_string();
        let mut a2 = "scratch".to_string();
        let mut o0 = "scratch".to_string();
        let mut o1 = "scratch".to_string();
        let pointer = |value: u32, aliases: &[u32]| {
            format!("scratch + offsets[{}]", resolve_alias(aliases, value))
        };
        if !op.args.is_empty() {
            a0 = pointer(op.args[0], &aliases);
        }
        if op.args.len() > 1 {
            a1 = pointer(op.args[1], &aliases);
        }
        if op.args.len() > 2 {
            a2 = pointer(op.args[2], &aliases);
        }
        if op.tag == OP_PIVOT_THRESHOLD {
            a1 = pointer(op.pred_payload, &aliases);
        }
        if op.results > 0 {
            o0 = pointer(base, &aliases);
        }
        if op.results > 1 {
            o1 = pointer(base + 1, &aliases);
        }

        source.push_str("  {\n");
        let _ = writeln!(source, "    M1OpParams p = params[{node}u];");
        source.push_str("    p.rng_seed = 0u;\n");

        if matches!(op.tag, OP_CHAN_TAKE | OP_CHAN_READ | OP_CHAN_PUT) {
            let _ = writeln!(
                source,
                "    const m1_u32 channel_index = lane.channel_slot_offset + {}u;",
                op.chan as u32
            );
            source.push_str("    const PtirLaneChannelSlot channel = channels[channel_index];\n");
            if op.tag == OP_CHAN_PUT {
                o0 = "reinterpret_cast<m1_u8*>(channel.pending_cell)".to_string();
            } else {
                a0 = "reinterpret_cast<const m1_u8*>(pending_flags[channel_index] != 0u ? \
                      channel.pending_cell : channel.committed_cell)"
                    .to_string();
            }
        } else if op.tag == OP_INTRINSIC_VAL {
            let _ = writeln!(
                source,
                "    const m1_u32 intrinsic_index = dispatch_lane * {PTIR_INTRINSIC_SLOTS}u + p.intr;"
            );
            source.push_str("    p.intrinsic_dtype = intrinsic_modes[intrinsic_index];\n");
            source.push_str("    p.imm = intrinsic_widths[intrinsic_index];\n");
            source.push_str("    p.intrinsic_row_stride = intrinsic_strides[intrinsic_index];\n");
            source.push_str("    p.intrinsic_row_offset = intrinsic_offsets[intrinsic_index];\n");
            a0 = "reinterpret_cast<const m1_u8*>(intrinsic_bases[intrinsic_index])".to_string();
        }

        emit_body(
            &mut source,
            stage,
            op,
            node,
            &direct.intrinsic,
            &a0,
            &a1,
            &a2,
            &o0,
            &o1,
        );

        source.push_str("    __syncthreads();\n");
        source.push_str("    if (status.state != 1u) {\n");
        source.push_str("      if (threadIdx.x == 0u) *commit = 0u;\n");
        source.push_str("      return;\n");
        source.push_str("    }\n");
        if op.tag == OP_CHAN_PUT {
            source.push_str("    if (threadIdx.x == 0u) pending_flags[channel_index] = 1u;\n");
            source.push_str("    __syncthreads();\n");
        }
        source.push_str("  }\n");
    }
    source.push_str("}\n");
    Ok(source)
}

/// The per-op body: one runtime helper call, or the single-thread fallback.
#[allow(clippy::too_many_arguments)]
fn emit_body(
    source: &mut String,
    stage: &CompiledStage,
    op: &OpView,
    node: usize,
    direct_intrinsic: &[u16],
    a0: &str,
    a1: &str,
    a2: &str,
    o0: &str,
    o1: &str,
) {
    let tag = op.tag;
    let fallback = |source: &mut String| {
        let _ = writeln!(
            source,
            "    if (threadIdx.x == 0u) ptir_m1_execute({tag}u, &status, descriptors, &p, {a0}, {a1}, {a2}, {o0}, {o1}, temporary);"
        );
    };

    if tag == OP_CONST {
        source.push_str("    const M1ValueDesc out = descriptors[p.o0];\n");
        source.push_str("    for (m1_u32 i = threadIdx.x; i < out.len; i += blockDim.x) {\n");
        let _ = writeln!(
            source,
            "      if (p.lit_dtype == 0u) m1_store_f({o0}, i, m1_bits_f32(p.lit_bits));"
        );
        let _ = writeln!(
            source,
            "      else if (p.lit_dtype == 1u) m1_store_i({o0}, i, m1_bits_i32(p.lit_bits));"
        );
        let _ = writeln!(
            source,
            "      else if (p.lit_dtype == 2u) m1_store_u({o0}, i, p.lit_bits);"
        );
        let _ = writeln!(source, "      else m1_store_b({o0}, i, p.lit_bits != 0u);");
        source.push_str("    }\n");
    } else if matches!(tag, OP_CHAN_TAKE | OP_CHAN_READ) {
        source.push_str("    const M1ValueDesc out = descriptors[p.o0];\n");
        let _ = writeln!(
            source,
            "    ptir_parallel_copy({a0}, {o0}, out.len, out.dtype);"
        );
    } else if tag == OP_CHAN_PUT {
        emit_chan_put(source, op, a0, o0);
    } else if tag == OP_INTRINSIC_VAL {
        if op.intr == INTR_LAYER || op.intr == INTR_MTP_DRAFTS {
            fallback(source);
        } else {
            let _ = writeln!(
                source,
                "    ptir_parallel_intrinsic({a0}, {o0}, descriptors[p.o0], p);"
            );
        }
    } else if tag == OP_BROADCAST {
        let _ = writeln!(
            source,
            "    ptir_parallel_broadcast({a0}, {o0}, descriptors[p.a0], descriptors[p.o0]);"
        );
    } else if tag == OP_RESHAPE {
        source.push_str("    const M1ValueDesc out = descriptors[p.o0];\n");
        let _ = writeln!(
            source,
            "    ptir_parallel_copy({a0}, {o0}, out.len, out.dtype);"
        );
    } else if tag == OP_TRANSPOSE {
        let _ = writeln!(
            source,
            "    ptir_parallel_transpose(&status, {a0}, {o0}, descriptors[p.a0], descriptors[p.o0]);"
        );
    } else if matches!(tag, OP_REDUCE_SUM | OP_REDUCE_MAX | OP_REDUCE_MIN) {
        if stage.normalized.value_types[op.args[0] as usize].dtype as u8 == DT_F32 {
            let _ = writeln!(
                source,
                "    ptir_parallel_reduce_f32({tag}u, {a0}, {o0}, temporary, descriptors[p.a0]);"
            );
        } else {
            fallback(source);
        }
    } else if tag == OP_REDUCE_ARGMAX {
        if direct_intrinsic[node] != u16::MAX {
            let _ = writeln!(
                source,
                "    const m1_u32 direct_intrinsic_index = dispatch_lane * {PTIR_INTRINSIC_SLOTS}u + {}u;",
                direct_intrinsic[node]
            );
            source.push_str("    ptir_fast_argmax_intrinsic(\n");
            source.push_str(
                "        reinterpret_cast<const m1_u8*>(intrinsic_bases[direct_intrinsic_index]),\n",
            );
            let _ = writeln!(source, "        {o0},");
            source.push_str("        descriptors[p.a0],\n");
            source.push_str("        intrinsic_modes[direct_intrinsic_index],\n");
            source.push_str("        intrinsic_strides[direct_intrinsic_index],\n");
            source.push_str("        intrinsic_offsets[direct_intrinsic_index]);\n");
        } else {
            let _ = writeln!(
                source,
                "    ptir_fast_argmax({a0}, {o0}, descriptors[p.a0]);"
            );
        }
    } else if matches!(tag, OP_GATHER | OP_GATHER_ROW) {
        let _ = writeln!(
            source,
            "    ptir_parallel_gather({tag}u, {a0}, {a1}, {o0}, descriptors[p.a0], descriptors[p.a1], descriptors[p.o0]);"
        );
    } else if matches!(tag, OP_SCATTER_ADD | OP_SCATTER_SET) {
        let _ = writeln!(
            source,
            "    ptir_parallel_copy({a0}, {o0}, descriptors[p.a0].len, descriptors[p.a0].dtype);"
        );
        source.push_str("    __syncthreads();\n");
        let _ = writeln!(
            source,
            "    if (threadIdx.x == 0u) ptir_scatter_updates({tag}u, {a1}, {a2}, {o0}, descriptors[p.a0], descriptors[p.a1], descriptors[p.a2]);"
        );
    } else if tag == OP_PIVOT_THRESHOLD && op.pred_tag == 1 {
        let _ = writeln!(
            source,
            "    ptir_parallel_pivot_cummass({a0}, {a1}, {o0}, descriptors[p.a0], descriptors[p.a1]);"
        );
    } else if tag == OP_PIVOT_THRESHOLD {
        let _ = writeln!(
            source,
            "    ptir_parallel_pivot({a0}, {a1}, {o0}, descriptors[p.a0], descriptors[p.a1], p);"
        );
    } else if parallel_elementwise(tag) {
        let _ = writeln!(
            source,
            "    ptir_parallel_elementwise({tag}u, &status, descriptors, p, {a0}, {a1}, {a2}, {o0});"
        );
    } else {
        fallback(source);
    }
}

/// `chan_put` is the only op that reconciles the lane's row validity against
/// the committed cell, so it gets its own emitter.
fn emit_chan_put(source: &mut String, op: &OpView, a0: &str, o0: &str) {
    let tag = op.tag;
    source.push_str("    const M1ValueDesc input = descriptors[p.a0];\n");
    source.push_str(
        "    const m1_u32 logical_bytes = input.dtype == 3u ? input.len : input.len * 4u;\n",
    );
    source.push_str("    if (logical_bytes > p.sink_bytes) {\n");
    let _ = writeln!(
        source,
        "      if (threadIdx.x == 0u) m1_fault(&status, {tag}u);"
    );
    source.push_str("    } else {\n");
    let _ = writeln!(
        source,
        "      const bool sample_output = (lane.sample_output_channel_mask & (1ull << {}u)) != 0ull;",
        op.chan as u32
    );
    source.push_str(
        "      const m1_u8* committed = reinterpret_cast<const m1_u8*>(channel.committed_cell);\n",
    );
    source.push_str("      const m1_u32 element_bytes = input.dtype == 3u ? 1u : 4u;\n");
    source.push_str("      m1_u32 elements_per_validity_row = 0u;\n");
    source.push_str("      if (lane.token_count != 0u) {\n");
    source.push_str(
        "        if (input.rows == lane.token_count) elements_per_validity_row = input.last;\n",
    );
    source.push_str(
        "        else if (input.len == lane.token_count) elements_per_validity_row = 1u;\n",
    );
    source.push_str("      }\n");
    source.push_str(
        "      for (m1_u32 byte = threadIdx.x; byte < p.sink_bytes; byte += blockDim.x) {\n",
    );
    source.push_str("        if (byte >= logical_bytes) {\n");
    let _ = writeln!(source, "          {o0}[byte] = 0u;");
    source.push_str("          continue;\n");
    source.push_str("        }\n");
    source.push_str("        bool row_active = lane_active;\n");
    source
        .push_str("        if (lane_row_valid != nullptr && elements_per_validity_row != 0u) {\n");
    source.push_str("          const m1_u32 element = byte / element_bytes;\n");
    source.push_str("          const m1_u32 row = element / elements_per_validity_row;\n");
    source.push_str(
        "          if (row < lane.token_count) row_active = lane_row_valid[lane.row_valid_offset + row] != 0u;\n",
    );
    source.push_str("        }\n");
    let _ = writeln!(source, "        if (row_active) {o0}[byte] = ({a0})[byte];");
    let _ = writeln!(
        source,
        "        else if (sample_output) {o0}[byte] = 0xffu;"
    );
    let _ = writeln!(
        source,
        "        else if (committed != nullptr) {o0}[byte] = committed[byte];"
    );
    let _ = writeln!(
        source,
        "        else if (threadIdx.x == 0u) m1_fault(&status, {tag}u);"
    );
    source.push_str("      }\n");
    source.push_str("    }\n");
}
