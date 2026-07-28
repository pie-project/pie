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
use pie_ir::op::{intrinsic_tags, tags};

use pie_plan::{
    CompiledStage, Dimension, LANE_TABLE_ABI_VERSION, LibraryOp, NodeIndex, Region, RegionKind,
};

use crate::op_view::{OpView, result_bases};
use crate::slots::Slots;

use super::runtime::singleton_runtime_source;
use super::singleton::valid_identifier;
use super::validate::validate_generated_region;

const PROLOGUE: &str = include_str!("../../runtime/cuda/fused_block0.cuh");
const SIGNATURE: &str = include_str!("../../runtime/cuda/fused_block1.cuh");
const PREAMBLE: &str = include_str!("../../runtime/cuda/fused_block2.cuh");

/// `kPtirIntrinsicSlots` — per-lane intrinsic descriptor slots.
pub const PTIR_INTRINSIC_SLOTS: u32 = 8;

const DT_F32: u8 = 0;

/// The ops `ptir_parallel_elementwise` in `fused_block0.cuh` has an arm for.
///
/// Answering `true` for a tag that helper does not handle is not a fallback —
/// the helper's dispatch chain simply runs off the end of the loop body,
/// writing nothing, and the region goes on with an untouched output buffer and
/// a clean status word. The single-thread path this competes with ends in
/// `m1_fault(status, p.tag)`, so being wrong in the other direction is loud.
///
/// Hence the explicit list (it used to be `EXP..=CAST` and `ADD..=SELECT`,
/// which would absorb any tag allocated into a gap) and hence
/// `parallel_elementwise_matches_the_cuda_runtime`, which reads the arms back
/// out of the `.cuh` and fails if the two sides disagree either way.
fn parallel_elementwise(tag: u8) -> bool {
    matches!(
        tag,
        tags::EXP
            | tags::LOG
            | tags::NEG
            | tags::RECIP
            | tags::ABS
            | tags::SIGN
            | tags::CAST
            | tags::ADD
            | tags::SUB
            | tags::MUL
            | tags::DIV
            | tags::MAX_ELEM
            | tags::MIN_ELEM
            | tags::GT
            | tags::GE
            | tags::EQ
            | tags::NE
            | tags::LT
            | tags::LE
            | tags::AND
            | tags::OR
            | tags::NOT
            | tags::REM
            | tags::SELECT
            | tags::IOTA
            | tags::MASK_APPLY_PACKED
            | tags::CAUSAL_MASK
            | tags::SLIDING_WINDOW_MASK
            | tags::SINK_WINDOW_MASK
            | tags::RNG
            | tags::RNG_KEYED
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
///
/// `source_value` and `requires_single_row` are not used by emission — the
/// emitted kernel only needs to know *that* the rewrite applies. They exist
/// because the driver's launch packer needs them, and shipping them from here
/// is what keeps this the only implementation of the analysis
/// (`ptir-refactor.md` §4.2).
pub(crate) struct DirectArgmax {
    pub(crate) intrinsic: Vec<u16>,
    pub(crate) skipped: Vec<u8>,
    pub(crate) source_value: Vec<u32>,
    pub(crate) requires_single_row: Vec<u8>,
}

pub(crate) fn analyze_direct_argmax(
    stage: &CompiledStage,
    region: &Region,
    bases: &[u32],
) -> DirectArgmax {
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
        source_value: vec![u32::MAX; ops.len()],
        requires_single_row: vec![0; ops.len()],
    };

    for &node in &region.nodes {
        let node = node.index();
        let reduction = &ops[node];
        if reduction.tag != tags::REDUCE_ARGMAX || reduction.args.is_empty() {
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
            if op.tag == tags::RESHAPE && !op.args.is_empty() {
                expected_consumer = producer;
                value = op.args[0];
                continue;
            }
            if op.tag != tags::INTRINSIC_VAL
                || (op.intr != intrinsic_tags::LOGITS && op.intr != intrinsic_tags::MTP_LOGITS)
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
                analysis.source_value[node] = bases[producer as usize];
                analysis.requires_single_row[node] = u8::from(runtime_single_row);
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

    // A nucleus library region elsewhere in the stage reads the logits and the
    // sampler state straight out of the lane's bf16 rows, so the nodes that
    // would have materialized them here are redundant -- but only under the
    // same two conditions the driver applies when it shrinks those value slots
    // to four bytes (`fused_runtime.cuh` `resolves_to_logits_intrinsic` /
    // `lone_generated_node`). Skipping a node the driver still budgets a full
    // slot for leaves that slot unwritten: whatever the region emits next reads
    // uninitialized scratch and publishes it. No fault, no failing test, just a
    // wrong sample. The sibling analysis `analyze_direct_argmax` above has
    // always checked its chain; this one did not.
    let producer_of = |value: u32| -> Option<usize> {
        region
            .nodes
            .iter()
            .map(|node| node.index())
            .chain(0..ops.len())
            .find(|&node| value >= bases[node] && value < bases[node] + ops[node].results)
    };
    let produces_logits_intrinsic = |value: u32| -> bool {
        producer_of(value).is_some_and(|node| {
            let op = &ops[node];
            op.tag == tags::INTRINSIC_VAL
                && (op.intr == intrinsic_tags::LOGITS || op.intr == intrinsic_tags::MTP_LOGITS)
        })
    };
    let resolves_to_logits_intrinsic = |value: u32| -> bool {
        if produces_logits_intrinsic(value) {
            return true;
        }
        producer_of(value).is_some_and(|node| {
            let op = &ops[node];
            op.tag == tags::RESHAPE && !op.args.is_empty() && produces_logits_intrinsic(op.args[0])
        })
    };
    // The scale divide is folded into the nucleus kernel only when it is a
    // region of its own; fused into a larger generated region it still runs and
    // still writes a full tensor.
    let mut lone_generated_node = vec![false; ops.len()];
    for candidate in &stage.fused.regions {
        if !matches!(candidate.kind, RegionKind::Library(_)) && candidate.nodes.len() == 1 {
            lone_generated_node[candidate.nodes[0].index()] = true;
        }
    }
    let in_region = |node: usize| region.nodes.contains(&NodeIndex(node as u32));

    for candidate in &stage.fused.regions {
        if !matches!(
            candidate.kind,
            RegionKind::Library(LibraryOp::NucleusSample)
        ) || candidate.inputs.len() != 5
        {
            continue;
        }
        if resolves_to_logits_intrinsic(candidate.inputs[0])
            && let Some(node) = producer_of(candidate.inputs[0])
            && in_region(node)
        {
            skipped[node] = 1;
        }
        let Some(node) = producer_of(candidate.inputs[2]) else {
            continue;
        };
        if !lone_generated_node[node] {
            continue;
        }
        if in_region(node) {
            skipped[node] = 1;
        }
        let op = &ops[node];
        if op.tag == tags::DIV
            && !op.args.is_empty()
            && resolves_to_logits_intrinsic(op.args[0])
            && let Some(reshape) = producer_of(op.args[0])
            && ops[reshape].tag == tags::RESHAPE
            && in_region(reshape)
        {
            skipped[reshape] = 1;
        }
    }

    let mut source = singleton_runtime_source();
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);

    for &node in &region.nodes {
        let node = node.index();
        let op = &ops[node];
        let base = bases[node];
        if skipped[node] != 0 && op.tag != tags::RESHAPE {
            continue;
        }
        if op.tag == tags::RESHAPE && !region.outputs.contains(&base) {
            aliases[base as usize] = resolve_alias(&aliases, op.args[0]);
            continue;
        }

        // CUDA resolves reshape aliases before indexing the offsets table;
        // which value lands in which slot is shared with Metal.
        let Slots {
            mut a0,
            a1,
            a2,
            mut o0,
            o1,
        } = Slots::of(op, base, |value| {
            format!("scratch + offsets[{}]", resolve_alias(&aliases, value))
        });

        source.push_str("  {\n");
        let _ = writeln!(source, "    M1OpParams p = params[{node}u];");
        source.push_str("    p.rng_seed = 0u;\n");

        if matches!(op.tag, tags::CHAN_TAKE | tags::CHAN_READ | tags::CHAN_PUT) {
            let _ = writeln!(
                source,
                "    const m1_u32 channel_index = lane.channel_slot_offset + {}u;",
                op.chan as u32
            );
            source.push_str("    const PtirLaneChannelSlot channel = channels[channel_index];\n");
            if op.tag == tags::CHAN_PUT {
                o0 = "reinterpret_cast<m1_u8*>(channel.pending_cell)".to_string();
            } else {
                a0 = "reinterpret_cast<const m1_u8*>(pending_flags[channel_index] != 0u ? \
                      channel.pending_cell : channel.committed_cell)"
                    .to_string();
            }
        } else if op.tag == tags::INTRINSIC_VAL {
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
        if op.tag == tags::CHAN_PUT {
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

    if tag == tags::CONST {
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
    } else if matches!(tag, tags::CHAN_TAKE | tags::CHAN_READ) {
        source.push_str("    const M1ValueDesc out = descriptors[p.o0];\n");
        let _ = writeln!(
            source,
            "    ptir_parallel_copy({a0}, {o0}, out.len, out.dtype);"
        );
    } else if tag == tags::CHAN_PUT {
        emit_chan_put(source, op, a0, o0);
    } else if tag == tags::INTRINSIC_VAL {
        if op.intr == intrinsic_tags::LAYER || op.intr == intrinsic_tags::MTP_DRAFTS {
            fallback(source);
        } else {
            let _ = writeln!(
                source,
                "    ptir_parallel_intrinsic({a0}, {o0}, descriptors[p.o0], p);"
            );
        }
    } else if tag == tags::BROADCAST {
        let _ = writeln!(
            source,
            "    ptir_parallel_broadcast({a0}, {o0}, descriptors[p.a0], descriptors[p.o0]);"
        );
    } else if tag == tags::RESHAPE {
        source.push_str("    const M1ValueDesc out = descriptors[p.o0];\n");
        let _ = writeln!(
            source,
            "    ptir_parallel_copy({a0}, {o0}, out.len, out.dtype);"
        );
    } else if tag == tags::TRANSPOSE {
        let _ = writeln!(
            source,
            "    ptir_parallel_transpose(&status, {a0}, {o0}, descriptors[p.a0], descriptors[p.o0]);"
        );
    } else if matches!(tag, tags::REDUCE_SUM | tags::REDUCE_MAX | tags::REDUCE_MIN) {
        if stage.normalized.value_types[op.args[0] as usize].dtype as u8 == DT_F32 {
            let _ = writeln!(
                source,
                "    ptir_parallel_reduce_f32({tag}u, {a0}, {o0}, temporary, descriptors[p.a0]);"
            );
        } else {
            fallback(source);
        }
    } else if tag == tags::REDUCE_ARGMAX {
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
    } else if matches!(tag, tags::GATHER | tags::GATHER_ROW) {
        let _ = writeln!(
            source,
            "    ptir_parallel_gather({tag}u, {a0}, {a1}, {o0}, descriptors[p.a0], descriptors[p.a1], descriptors[p.o0]);"
        );
    } else if matches!(tag, tags::SCATTER_ADD | tags::SCATTER_SET) {
        let _ = writeln!(
            source,
            "    ptir_parallel_copy({a0}, {o0}, descriptors[p.a0].len, descriptors[p.a0].dtype);"
        );
        source.push_str("    __syncthreads();\n");
        let _ = writeln!(
            source,
            "    if (threadIdx.x == 0u) ptir_scatter_updates({tag}u, {a1}, {a2}, {o0}, descriptors[p.a0], descriptors[p.a1], descriptors[p.a2]);"
        );
    } else if tag == tags::PIVOT_THRESHOLD && op.pred_tag == 1 {
        let _ = writeln!(
            source,
            "    ptir_parallel_pivot_cummass({a0}, {a1}, {o0}, descriptors[p.a0], descriptors[p.a1]);"
        );
    } else if tag == tags::PIVOT_THRESHOLD {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_scan::{function_body, tags_compared_in};
    use alloc::collections::BTreeSet;

    /// `parallel_elementwise` is a claim about C++ that the C++ cannot check:
    /// the helper's dispatch runs off the end for a tag it does not know,
    /// leaving the output buffer untouched and the status word clean. So the
    /// two lists are compared here instead.
    #[test]
    fn parallel_elementwise_matches_the_cuda_runtime() {
        let handled = tags_compared_in(function_body(PROLOGUE, "ptir_parallel_elementwise"));
        let claimed: BTreeSet<u8> = pie_ir::op::OP_TABLE
            .iter()
            .map(|spec| spec.tag)
            .filter(|tag| parallel_elementwise(*tag))
            .collect();
        let name = |tag: &u8| pie_ir::op::spec(*tag).map_or("?", |spec| spec.name);
        let unhandled: Vec<&str> = claimed.difference(&handled).map(name).collect();
        let unclaimed: Vec<&str> = handled.difference(&claimed).map(name).collect();
        assert!(
            unhandled.is_empty(),
            "emitted to ptir_parallel_elementwise but not handled there \
             (these write nothing, silently): {unhandled:?}"
        );
        assert!(
            unclaimed.is_empty(),
            "handled by ptir_parallel_elementwise but never emitted to it \
             (dead runtime code, or a missing tag in parallel_elementwise): {unclaimed:?}"
        );
    }
}
