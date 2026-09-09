use crate::codegen::error::{EmitError, EmitterKind, ValueLayoutSite};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::Dtype;
use eta_ir::op::{IntrinsicId, intrinsic_tags, tags};

use crate::plan::{CompiledStage, Dimension, LANE_TABLE_ABI_VERSION, Region};

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::slots::Slots;

use super::runtime::singleton_runtime_source;
use super::singleton::valid_identifier;
use super::validate::validate_generated_region;

pub(super) const PROLOGUE: &str = include_str!("../../../runtime/cuda/fused_block0.cuh");
pub(super) const SIGNATURE: &str = include_str!("../../../runtime/cuda/fused_block1.cuh");
pub(super) const PREAMBLE: &str = include_str!("../../../runtime/cuda/fused_block2.cuh");

pub const PTIR_INTRINSIC_SLOTS: u32 = IntrinsicId::SLOTS;

fn parallel_elementwise(tag: u8) -> bool {
    matches!(
        tag,
        tags::EXP
            | tags::LOG
            | tags::NEG
            | tags::RECIP
            | tags::SIN
            | tags::COS
            | tags::SQRT
            | tags::RSQRT
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

pub(crate) struct ArgmaxScan {
    pub(crate) intrinsic: Vec<u16>,
    pub(crate) skipped: Vec<u8>,
    pub(crate) source_value: Vec<u32>,
    pub(crate) requires_single_row: Vec<u8>,
    pub(crate) gumbel: Vec<Option<GumbelChain>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GumbelChain {
    pub(crate) intrinsic: u16,
    pub(crate) state: u32,
    pub(crate) divisor: Option<u32>,
    pub(crate) noise: u32,
}

pub static GUMBEL_DIRECT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(true);

fn gumbel_direct_enabled() -> bool {
    GUMBEL_DIRECT.load(core::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn analyze_direct_argmax(
    stage: &CompiledStage,
    region: &Region,
    bases: &[u32],
) -> ArgmaxScan {
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
    let mut analysis = ArgmaxScan {
        intrinsic: vec![u16::MAX; ops.len()],
        skipped: vec![0; ops.len()],
        source_value: vec![u32::MAX; ops.len()],
        requires_single_row: vec![0; ops.len()],
        gumbel: vec![None; ops.len()],
    };
    let single_consumer = |value: u32, consumer: u32| -> bool {
        (value as usize) < producers.len()
            && producers[value as usize] != u32::MAX
            && consumers[value as usize].len() == 1
            && consumers[value as usize][0] == consumer
    };
    let one_element = |value: u32| -> bool {
        stage
            .normalized
            .value_types
            .get(value as usize)
            .is_some_and(|ty| {
                ty.dims
                    .iter()
                    .all(|dim| matches!(dim, Dimension::Static(1)))
            })
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
        let mut gumbel: Option<(u32, Option<u32>, u32)> = None;
        if gumbel_direct_enabled()
            && single_consumer(value, node as u32)
            && ops[producers[value as usize] as usize].tag == tags::ADD
        {
            let add = producers[value as usize];
            let add_op = &ops[add as usize];
            if add_op.args.len() == 2 {
                let is_gumbel = |v: u32| {
                    single_consumer(v, add)
                        && ops[producers[v as usize] as usize].tag == tags::RNG_KEYED
                        && ops[producers[v as usize] as usize].kind == 1
                        && !ops[producers[v as usize] as usize].args.is_empty()
                };
                let (logits, noise) = if is_gumbel(add_op.args[1]) {
                    (add_op.args[0], Some(add_op.args[1]))
                } else if is_gumbel(add_op.args[0]) {
                    (add_op.args[1], Some(add_op.args[0]))
                } else {
                    (add_op.args[0], None)
                };
                if let Some(noise) = noise {
                    let rng = producers[noise as usize];
                    let state = ops[rng as usize].args[0];
                    let mut head = vec![add, rng];
                    let mut divisor = None;
                    let mut source = logits;
                    let mut consumer = add;
                    if single_consumer(logits, add) {
                        let scale = producers[logits as usize];
                        let scale_op = &ops[scale as usize];
                        if scale_op.tag == tags::DIV
                            && scale_op.args.len() == 2
                            && one_element(scale_op.args[1])
                        {
                            divisor = Some(scale_op.args[1]);
                            source = scale_op.args[0];
                            consumer = scale;
                            head.push(scale);
                        }
                    }
                    if single_consumer(source, consumer) {
                        gumbel = Some((state, divisor, noise));
                        chain.extend(head);
                        value = source;
                        expected_consumer = consumer;
                    }
                }
            }
        }
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
            if let Some((state, divisor, noise)) = gumbel {
                if exact_shape || runtime_single_row {
                    analysis.gumbel[node] = Some(GumbelChain {
                        intrinsic: op.intr,
                        state,
                        divisor,
                        noise,
                    });
                    for &skipped in &chain {
                        analysis.skipped[skipped as usize] = 1;
                    }
                }
                break;
            }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TopKDirect {
    pub(crate) intrinsic: u16,
    pub(crate) node: u32,
    pub(crate) divisor: Option<u32>,
}

pub(crate) fn analyze_direct_topk(stage: &CompiledStage) -> Vec<Option<TopKDirect>> {
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let index = crate::plan::StageIndex::of(&stage.normalized);
    (0..ops.len())
        .map(|node| {
            let direct = crate::plan::direct_topk(
                &stage.normalized,
                &index,
                crate::plan::NodeIndex(node as u32),
            )?;
            let producer = direct.intrinsic.index();
            Some(TopKDirect {
                intrinsic: ops[producer].intr,
                node: producer as u32,
                divisor: direct.divisor,
            })
        })
        .collect()
}

pub(crate) fn row_geometry(stage: &CompiledStage, region: &Region) -> Option<(u64, u32)> {
    region
        .row_value
        .and_then(|witness| stage.normalized.value_types.get(witness as usize))
        .and_then(|ty| crate::plan::value_rows(&ty.dims))
        .filter(|&(fixed, extent)| !(fixed == 1 && extent == u32::MAX))
}

pub(crate) fn row_kinds(stage: &CompiledStage, region: &Region, geometry: (u64, u32)) -> Vec<u8> {
    let (fixed, extent) = geometry;
    stage
        .normalized
        .value_types
        .iter()
        .map(|ty| {
            let alias = region.row_alias;
            if ty.dims.len() >= 2
                && crate::plan::value_rows(&ty.dims)
                    .is_some_and(|shape| crate::plan::same_rows(shape, (fixed, extent), alias))
            {
                1
            } else if crate::plan::is_row_vector(&ty.dims, fixed, extent, alias) {
                2
            } else {
                0
            }
        })
        .collect()
}

pub fn emit_fused_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(EmitterKind::CudaFused));
    }
    validate_generated_region(stage, region)?;

    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let next_value: u32 = ops.iter().map(|op| op.results).sum();
    if next_value as usize != stage.normalized.value_types.len() {
        return Err(EmitError::NormalizedValueLayoutMismatch(
            ValueLayoutSite::CudaFusedStage,
        ));
    }

    let mut aliases = crate::codegen::alias::AliasTable::new();
    let direct = analyze_direct_argmax(stage, region, &bases);
    let skipped = direct.skipped;

    let mut source = singleton_runtime_source();
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);

    let row_geometry = row_geometry(stage, region);
    let row_parallel = row_geometry.is_some();
    let row_kinds: Vec<u8> =
        row_geometry.map_or_else(Vec::new, |geometry| row_kinds(stage, region, geometry));
    if row_geometry.is_some() {
        let kinds: Vec<String> = row_kinds.iter().map(|k| format!("{k}u")).collect();
        let count = kinds.len().max(1);
        let _ = writeln!(
            source,
            "  const m1_u8 ptir_rowkind[{count}u] = {{{}}};",
            kinds.join(", ")
        );
        let _ = writeln!(source, "  __shared__ M1ValueDesc ptir_rowdesc[{count}u];");
        let _ = writeln!(source, "  __shared__ m1_u64 ptir_rowshift[{count}u];");
        let _ = writeln!(
            source,
            "  for (m1_u32 v = threadIdx.x; v < value_count && v < {count}u; v += blockDim.x) {{"
        );
        source.push_str(
            "    M1ValueDesc d = descriptors[v];
",
        );
        source.push_str(
            "    m1_u64 shift = 0u;
",
        );
        source.push_str(
            "    const m1_u64 elem = d.dtype == 3u ? 1u : 4u;
",
        );
        source.push_str(
            "    if (ptir_rowkind[v] == 1u) {
",
        );
        source.push_str(
            "      shift = (m1_u64)lane_row * (m1_u64)d.last * elem;
",
        );
        source.push_str(
            "      d.len = d.last; d.rows = 1u; d.rank = 1u; d.dims[0] = d.last;
",
        );
        let _ = writeln!(
            source,
            "      for (m1_u32 k = 1u; k < {}u; ++k) d.dims[k] = 0u;",
            eta_ir::types::MAX_RANK
        );
        source.push_str(
            "    } else if (ptir_rowkind[v] == 2u) {
",
        );
        source.push_str(
            "      shift = (m1_u64)lane_row * elem;
",
        );
        source.push_str(
            "      d.len = 1u; d.rows = 1u; d.last = 1u; d.dims[0] = 1u;
",
        );
        source.push_str(
            "    }
",
        );
        source.push_str(
            "    ptir_rowdesc[v] = d;
",
        );
        source.push_str(
            "    ptir_rowshift[v] = shift;
",
        );
        source.push_str(
            "  }
",
        );
        source.push_str(
            "  __syncthreads();
",
        );
        source.push_str(
            "  descriptors = ptir_rowdesc;
",
        );
    }

    const TAIL: &str = "    __syncthreads();\n    if (status.state != 1u) {\n      if (threadIdx.x == 0u) *commit = 0u;\n      return;\n    }\n";
    let direct_topk = analyze_direct_topk(stage);
    let streams = row_parallel.then(|| {
        super::stream::Streams::new(
            stage,
            region,
            &ops,
            &bases,
            &row_kinds,
            &direct.intrinsic,
            &skipped,
            &direct_topk,
        )
    });

    let order: Vec<usize> = match &streams {
        Some(streams) => streams.order.clone(),
        None => region.nodes.iter().map(|n| n.index()).collect(),
    };
    let mut at = 0usize;
    while at < order.len() {
        let node = order[at];
        at += 1;
        let op = &ops[node];
        let base = bases[node];
        if skipped[node] != 0 && op.tag != tags::RESHAPE {
            continue;
        }
        if let Some(streams) = &streams {
            let mut pointer = |value: u32| {
                let value = aliases.resolve(value);
                format!("scratch + offsets[{value}] + ptir_rowshift[{value}]")
            };
            if let Some(covered) =
                super::stream::emit_stream(&mut source, streams, at - 1, &mut pointer, TAIL)
            {
                at += covered - 1;
                continue;
            }
        }
        if op.tag == tags::RESHAPE
            && !op.args.is_empty()
            && !region.outputs.contains(&base)
            && crate::codegen::alias::covers(&stage.normalized.value_types, op.args[0], base)
        {
            aliases.elide(base, op.args[0]);
            continue;
        }

        let resolve = |value: u32| {
            let value = aliases.resolve(value);
            if row_parallel {
                format!("scratch + offsets[{value}] + ptir_rowshift[{value}]")
            } else {
                format!("scratch + offsets[{value}]")
            }
        };
        let mut slots = Slots::of(op, base, resolve);
        let gumbel = direct.gumbel[node].map(|chain| GumbelSlots {
            intrinsic: chain.intrinsic,
            state: resolve(chain.state),
            state_desc: aliases.resolve(chain.state),
            divisor: chain.divisor.map(resolve),
            divisor_desc: chain.divisor.map(|d| aliases.resolve(d)),
            noise_desc: aliases.resolve(chain.noise),
        });

        source.push_str("  {\n");
        let _ = writeln!(source, "    M1OpParams p = params[{node}u];");
        source.push_str("    p.rng_seed = 0u;\n");
        if matches!(op.tag, tags::RNG | tags::RNG_KEYED) {
            source.push_str("    p.imm3 = lane_row * descriptors[p.o0].len;\n");
        }

        if matches!(op.tag, tags::CHAN_TAKE | tags::CHAN_READ | tags::CHAN_PUT) {
            let _ = writeln!(
                source,
                "    const m1_u32 channel_index = lane.channel_slot_offset + {}u;",
                op.chan as u32
            );
            source.push_str("    const PtirLaneChannelSlot channel = channels[channel_index];\n");
            if op.tag == tags::CHAN_PUT {
                slots.o0 = "reinterpret_cast<m1_u8*>(channel.pending_cell)".to_string();
            } else {
                slots.a0 = "reinterpret_cast<const m1_u8*>(pending_flags[channel_index] != 0u ? \
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
            source.push_str("    p.intrinsic_row_offset += lane_row;\n");
            slots.a0 =
                "reinterpret_cast<const m1_u8*>(intrinsic_bases[intrinsic_index])".to_string();
        }

        emit_body(
            &mut source,
            stage,
            op,
            node,
            &direct.intrinsic,
            &slots,
            gumbel.as_ref(),
        );

        source.push_str(TAIL);
        if op.tag == tags::CHAN_PUT {
            source.push_str("    if (threadIdx.x == 0u) pending_flags[channel_index] = 1u;\n");
            source.push_str("    __syncthreads();\n");
        }
        source.push_str("  }\n");
    }
    source.push_str("}\n");
    Ok(source)
}

struct GumbelSlots {
    intrinsic: u16,
    state: String,
    state_desc: u32,
    divisor: Option<String>,
    divisor_desc: Option<u32>,
    noise_desc: u32,
}

fn emit_body(
    source: &mut String,
    stage: &CompiledStage,
    op: &OpView,
    node: usize,
    direct_intrinsic: &[u16],
    slots: &Slots,
    gumbel: Option<&GumbelSlots>,
) {
    let Slots { a0, a1, a2, o0, o1 } = slots;
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
        if stage.normalized.value_types[op.args[0] as usize].dtype == Dtype::F32 {
            let _ = writeln!(
                source,
                "    ptir_parallel_reduce_f32({tag}u, {a0}, {o0}, temporary, descriptors[p.a0]);"
            );
        } else {
            fallback(source);
        }
    } else if tag == tags::REDUCE_ARGMAX {
        if let Some(chain) = gumbel {
            let _ = writeln!(
                source,
                "    const m1_u32 gumbel_intrinsic_index = dispatch_lane * {PTIR_INTRINSIC_SLOTS}u + {}u;",
                chain.intrinsic
            );
            source.push_str(
                "    ptir_fast_gumbel_argmax_intrinsic(
",
            );
            source.push_str(
                "        reinterpret_cast<const m1_u8*>(intrinsic_bases[gumbel_intrinsic_index]),
",
            );
            let _ = writeln!(source, "        {o0},");
            source.push_str(
                "        descriptors[p.a0],
",
            );
            source.push_str(
                "        intrinsic_modes[gumbel_intrinsic_index],
",
            );
            source.push_str(
                "        intrinsic_strides[gumbel_intrinsic_index],
",
            );
            source.push_str(
                "        intrinsic_offsets[gumbel_intrinsic_index] + lane_row,
",
            );
            let _ = writeln!(source, "        {},", chain.state);
            let _ = writeln!(source, "        descriptors[{}u],", chain.state_desc);
            match (&chain.divisor, chain.divisor_desc) {
                (Some(divisor), Some(desc)) => {
                    let _ = writeln!(source, "        {divisor},");
                    let _ = writeln!(source, "        descriptors[{desc}u],");
                }
                _ => {
                    source.push_str(
                        "        nullptr,
",
                    );
                    source.push_str(
                        "        descriptors[p.a0],
",
                    );
                }
            }
            let _ = writeln!(
                source,
                "        lane_row * descriptors[{}u].len);",
                chain.noise_desc
            );
        } else if direct_intrinsic[node] != u16::MAX {
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
            source.push_str("        intrinsic_offsets[direct_intrinsic_index] + lane_row);\n");
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
    use crate::codegen::layout::{HOST_SHARED, LANE_CHANNEL_SLOT, LANE_RECORD, LANE_TABLE_HEADER};

    #[test]
    fn cuda_runtime_lane_table_matches_layout() {
        for declared in HOST_SHARED {
            let expected = declared.emit_cuda();
            assert!(
                PROLOGUE.contains(&expected),
                "fused_block0.cuh has drifted from layout.rs for {}; expected:\n{expected}",
                declared.c_name
            );
        }
        for (declared, note) in [
            (LANE_TABLE_HEADER, "lane header ABI"),
            (LANE_RECORD, "lane record ABI"),
            (LANE_CHANNEL_SLOT, "lane channel ABI"),
        ] {
            let expected = declared.emit_cuda_size_assert(note);
            assert!(
                PROLOGUE.contains(&expected),
                "fused_block0.cuh size assert has drifted; expected:\n{expected}"
            );
        }
    }
}
