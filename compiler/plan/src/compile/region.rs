//! Partitioning the normalized op DAG into regions.
//!
//! Two partitions come out of every stage: a singleton partition (one region
//! per op, the always-correct fallback) and a fused partition that groups ops
//! sharing a schedule and lifts recognized dataflows -- nucleus sampling,
//! top-k, sort, scan, matmul -- into library calls.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use pie_ir::op::Op;
use pie_ir::types::{DType, Literal, Predicate, RngKind};

use super::normalize::{NormalizedStage, result_layout};
use super::symbolic::{Dimension, symbolic_dims_match_expected, symbolic_shape_matches_static};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ScheduleTemplate {
    Effects = 0,
    OneCtaPerRow = 1,
    HierarchicalRow = 2,
    Library = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LibraryOp {
    NucleusSample = 0,
    TopK = 1,
    Sort = 2,
    Scan = 3,
    MatMul = 4,
    SecondParty = 5,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Generated,
    Library(LibraryOp),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSink {
    pub channel_slot: u32,
    pub value: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub kind: RegionKind,
    pub schedule: ScheduleTemplate,
    pub nodes: Vec<u32>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub sinks: Vec<ChannelSink>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LibraryMatch {
    library: LibraryOp,
    nodes: Vec<u32>,
    inputs: Vec<u32>,
    outputs: Vec<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PartitionKind {
    Singleton = 0,
    Fused = 1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionPartition {
    pub kind: PartitionKind,
    pub regions: Vec<Region>,
    /// Legacy wire bit retained for decoder compatibility. Revision 6 plans
    /// never request whole-stage fallback.
    pub whole_stage_fallback: bool,
}

pub(crate) fn singleton_partition(stage: &NormalizedStage) -> RegionPartition {
    let index = StageIndex::of(stage);
    let regions = (0..stage.ops.len())
        .map(|node| {
            build_region(
                stage,
                &index,
                vec![node as u32],
                region_kind_for_node(stage, node),
            )
        })
        .collect();
    RegionPartition {
        kind: PartitionKind::Singleton,
        regions,
        whole_stage_fallback: false,
    }
}

pub(crate) fn recognize_library_dataflows(stage: &NormalizedStage) -> Vec<LibraryMatch> {
    let (bases, producer) = result_layout(&stage.ops);
    let mut consumers = vec![Vec::new(); stage.value_types.len()];
    for (node, op) in stage.ops.iter().enumerate() {
        for operand in op.operands() {
            consumers[operand as usize].push(node as u32);
        }
    }

    let mut claimed = BTreeSet::new();
    let mut matches = Vec::new();
    for final_node in 0..stage.ops.len() {
        let Some(candidate) =
            match_nucleus_dataflow(stage, final_node, &bases, &producer, &consumers)
        else {
            continue;
        };
        if candidate.nodes.iter().any(|node| claimed.contains(node)) {
            continue;
        }
        claimed.extend(candidate.nodes.iter().copied());
        matches.push(candidate);
    }
    matches
}

pub(crate) fn match_nucleus_dataflow(
    stage: &NormalizedStage,
    final_node: usize,
    bases: &[u32],
    producer: &[usize],
    consumers: &[Vec<u32>],
) -> Option<LibraryMatch> {
    let Op::ReduceArgmax(perturbed) = stage.ops.get(final_node)? else {
        return None;
    };
    let add_node = *producer.get(*perturbed as usize)?;
    let Op::Add(left, right) = stage.ops.get(add_node)? else {
        return None;
    };

    match_nucleus_add_order(
        stage, final_node, add_node, *left, *right, bases, producer, consumers,
    )
    .or_else(|| {
        match_nucleus_add_order(
            stage, final_node, add_node, *right, *left, bases, producer, consumers,
        )
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn match_nucleus_add_order(
    stage: &NormalizedStage,
    final_node: usize,
    add_node: usize,
    masked: u32,
    noise: u32,
    bases: &[u32],
    producer: &[usize],
    consumers: &[Vec<u32>],
) -> Option<LibraryMatch> {
    let select_node = *producer.get(masked as usize)?;
    let Op::Select {
        cond: keep,
        a: logits,
        b: negative_infinity,
    } = stage.ops.get(select_node)?
    else {
        return None;
    };
    let (keep, logits, negative_infinity) = (*keep, *logits, *negative_infinity);

    let rng_node = *producer.get(noise as usize)?;
    let Op::RngKeyed {
        state: rng_state,
        shape: rng_shape,
        kind: RngKind::Gumbel,
    } = stage.ops.get(rng_node)?
    else {
        return None;
    };
    let rng_state = *rng_state;
    let rng_shape = *rng_shape;

    let negative_infinity_node = *producer.get(negative_infinity as usize)?;
    let Op::Const(Literal::F32(value)) = stage.ops.get(negative_infinity_node)? else {
        return None;
    };
    if value.to_bits() != f32::NEG_INFINITY.to_bits() {
        return None;
    }

    let pivot_node = *producer.get(keep as usize)?;
    let Op::PivotThreshold {
        input: probabilities,
        predicate: Predicate::CummassLe(top_p),
    } = stage.ops.get(pivot_node)?
    else {
        return None;
    };
    let (probabilities, top_p) = (*probabilities, *top_p);

    let div_node = *producer.get(probabilities as usize)?;
    let Op::Div(exponentials, sum_broadcast) = stage.ops.get(div_node)? else {
        return None;
    };
    let (exponentials, sum_broadcast) = (*exponentials, *sum_broadcast);

    let exponential_node = *producer.get(exponentials as usize)?;
    let Op::Exp(centered) = stage.ops.get(exponential_node)? else {
        return None;
    };
    let centered = *centered;

    let centered_node = *producer.get(centered as usize)?;
    let Op::Sub(centered_logits, maximum_broadcast) = stage.ops.get(centered_node)? else {
        return None;
    };
    if *centered_logits != logits {
        return None;
    }
    let maximum_broadcast = *maximum_broadcast;

    let maximum_broadcast_node = *producer.get(maximum_broadcast as usize)?;
    let Op::Broadcast {
        value: maximum,
        shape: maximum_shape,
    } = stage.ops.get(maximum_broadcast_node)?
    else {
        return None;
    };
    let (maximum, maximum_shape) = (*maximum, *maximum_shape);

    let maximum_node = *producer.get(maximum as usize)?;
    let Op::ReduceMax(maximum_logits) = stage.ops.get(maximum_node)? else {
        return None;
    };
    if *maximum_logits != logits {
        return None;
    }

    let sum_broadcast_node = *producer.get(sum_broadcast as usize)?;
    let Op::Broadcast {
        value: sum,
        shape: sum_shape,
    } = stage.ops.get(sum_broadcast_node)?
    else {
        return None;
    };
    let (sum, sum_shape) = (*sum, *sum_shape);

    let sum_node = *producer.get(sum as usize)?;
    let Op::ReduceSum(sum_exponentials) = stage.ops.get(sum_node)? else {
        return None;
    };
    if *sum_exponentials != exponentials || maximum_shape != sum_shape || maximum_shape != rng_shape
    {
        return None;
    }

    let token = *bases.get(final_node)?;
    let nodes = [
        maximum_node,
        maximum_broadcast_node,
        centered_node,
        exponential_node,
        sum_node,
        sum_broadcast_node,
        div_node,
        pivot_node,
        negative_infinity_node,
        select_node,
        rng_node,
        add_node,
        final_node,
    ];
    let mut ordered_nodes = nodes.map(|node| node as u32).to_vec();
    let mut library_inputs = vec![logits, top_p, rng_state];
    let mut scaled_input = None;
    if let Some(&scale_node) = producer.get(logits as usize)
        && let Some(Op::Div(raw_logits, divisor)) = stage.ops.get(scale_node)
    {
        let mut actual = consumers.get(logits as usize)?.clone();
        actual.sort_unstable();
        let mut expected = vec![
            maximum_node as u32,
            centered_node as u32,
            select_node as u32,
        ];
        expected.sort_unstable();
        if actual == expected {
            let mut library_logits = *raw_logits;
            if let Some(&reshape_node) = producer.get(*raw_logits as usize)
                && let Some(Op::Reshape { value, .. }) = stage.ops.get(reshape_node)
            {
                library_logits = *value;
            }
            library_inputs = vec![library_logits, *divisor, logits, top_p, rng_state];
            scaled_input = Some((library_logits, *divisor));
        }
    }
    ordered_nodes.sort_unstable();
    ordered_nodes.dedup();
    if ordered_nodes.len() != nodes.len() {
        return None;
    }
    let node_set: BTreeSet<u32> = ordered_nodes.iter().copied().collect();
    if library_inputs.iter().copied().any(|input| {
        producer
            .get(input as usize)
            .is_some_and(|node| node_set.contains(&(*node as u32)))
    }) {
        return None;
    }

    let exact_consumers = [
        (maximum, vec![maximum_broadcast_node as u32]),
        (maximum_broadcast, vec![centered_node as u32]),
        (centered, vec![exponential_node as u32]),
        (exponentials, vec![sum_node as u32, div_node as u32]),
        (sum, vec![sum_broadcast_node as u32]),
        (sum_broadcast, vec![div_node as u32]),
        (probabilities, vec![pivot_node as u32]),
        (keep, vec![select_node as u32]),
        (negative_infinity, vec![select_node as u32]),
        (masked, vec![add_node as u32]),
        (noise, vec![add_node as u32]),
        (
            *stage.ops[final_node].operands().first()?,
            vec![final_node as u32],
        ),
    ];
    for (value, mut expected) in exact_consumers {
        let mut actual = consumers.get(value as usize)?.clone();
        actual.sort_unstable();
        expected.sort_unstable();
        if actual != expected {
            return None;
        }
    }
    if consumers
        .get(token as usize)?
        .iter()
        .all(|consumer| node_set.contains(consumer))
    {
        return None;
    }

    let value_type = |value: u32| stage.value_types.get(value as usize);
    let logits_type = value_type(logits)?;
    if logits_type.dtype != DType::F32
        || !(1..=2).contains(&logits_type.rank())
        || !symbolic_shape_matches_static(logits_type, maximum_shape)
    {
        return None;
    }
    let row_dims = &logits_type.dims[..logits_type.dims.len() - 1];
    if let Some((raw_logits, divisor)) = scaled_input {
        let raw_type = value_type(raw_logits)?;
        if raw_type.dtype != DType::F32
            || !(1..=2).contains(&raw_type.rank())
            || raw_type.dims.last() != logits_type.dims.last()
        {
            return None;
        }
        let divisor_type = value_type(divisor)?;
        if divisor_type.dtype != DType::F32
            || (!divisor_type.is_scalar() && divisor_type.dims.as_slice() != row_dims)
        {
            return None;
        }
    }
    let top_p_type = value_type(top_p)?;
    if top_p_type.dtype != DType::F32
        || (!top_p_type.is_scalar()
            && !symbolic_dims_match_expected(
                &top_p_type.dims,
                row_dims,
                &maximum_shape.dims()[..maximum_shape.rank() - 1],
            ))
    {
        return None;
    }
    let rng_state_type = value_type(rng_state)?;
    if rng_state_type.dtype != DType::U32
        || rng_state_type.dims.as_slice() != [Dimension::Static(2)]
    {
        return None;
    }
    let token_type = value_type(token)?;
    if token_type.dtype != DType::I32 || token_type.dims.as_slice() != row_dims {
        return None;
    }

    for value in [
        maximum_broadcast,
        centered,
        exponentials,
        sum_broadcast,
        probabilities,
        masked,
        *stage.ops[final_node].operands().first()?,
    ] {
        if value_type(value)? != logits_type {
            return None;
        }
    }
    let noise_type = value_type(noise)?;
    if noise_type.dtype != DType::F32 || !symbolic_shape_matches_static(noise_type, rng_shape) {
        return None;
    }
    for value in [maximum, sum] {
        let ty = value_type(value)?;
        if ty.dtype != DType::F32 || ty.dims.as_slice() != row_dims {
            return None;
        }
    }
    let keep_type = value_type(keep)?;
    if keep_type.dtype != DType::Bool || keep_type.dims != logits_type.dims {
        return None;
    }
    let negative_infinity_type = value_type(negative_infinity)?;
    if negative_infinity_type.dtype != DType::F32 || !negative_infinity_type.dims.is_empty() {
        return None;
    }

    Some(LibraryMatch {
        library: LibraryOp::NucleusSample,
        nodes: ordered_nodes,
        inputs: library_inputs,
        outputs: vec![token],
    })
}

pub(crate) fn fused_partition(
    stage: &NormalizedStage,
    library_matches: &[LibraryMatch],
) -> RegionPartition {
    let index = StageIndex::of(stage);
    let matched_nodes: BTreeSet<u32> = library_matches
        .iter()
        .flat_map(|candidate| candidate.nodes.iter().copied())
        .collect();
    let matches_by_end: BTreeMap<u32, &LibraryMatch> = library_matches
        .iter()
        .map(|candidate| {
            (
                *candidate.nodes.last().expect("library match has nodes"),
                candidate,
            )
        })
        .collect();
    let mut regions = Vec::new();
    let mut generated = Vec::new();
    for node in 0..stage.ops.len() as u32 {
        if matched_nodes.contains(&node) {
            flush_generated_region(stage, &index, &mut regions, &mut generated);
            if let Some(candidate) = matches_by_end.get(&node) {
                regions.push(build_library_match_region(stage, &index, candidate));
            }
            continue;
        }

        let kind = region_kind_for_node(stage, node as usize);
        if matches!(kind, RegionKind::Library(_)) {
            flush_generated_region(stage, &index, &mut regions, &mut generated);
            regions.push(build_region(stage, &index, vec![node], kind));
            continue;
        }

        if generated.first().is_some_and(|first| {
            !compatible_schedule(&stage.ops[*first as usize], &stage.ops[node as usize])
        }) {
            flush_generated_region(stage, &index, &mut regions, &mut generated);
        }
        generated.push(node);
    }
    flush_generated_region(stage, &index, &mut regions, &mut generated);
    RegionPartition {
        kind: PartitionKind::Fused,
        regions,
        whole_stage_fallback: false,
    }
}

pub(crate) fn flush_generated_region(
    stage: &NormalizedStage,
    index: &StageIndex,
    regions: &mut Vec<Region>,
    nodes: &mut Vec<u32>,
) {
    if !nodes.is_empty() {
        regions.push(build_region(
            stage,
            index,
            core::mem::take(nodes),
            RegionKind::Generated,
        ));
    }
}

pub(crate) fn build_library_match_region(
    stage: &NormalizedStage,
    index: &StageIndex,
    candidate: &LibraryMatch,
) -> Region {
    let mut region = build_region(
        stage,
        index,
        candidate.nodes.clone(),
        RegionKind::Library(candidate.library),
    );
    region.inputs = candidate.inputs.clone();
    region.outputs = candidate.outputs.clone();
    region
}

/// The library kernel a wire tag is routed to, or `None` when the fused
/// generated kernel emits it inline.
///
/// Keyed on the tag rather than the `Op` variant so the classification can be
/// enumerated against [`pie_ir::op::OP_TABLE`]: `every_op_is_classified` in
/// `compiler/tests` partitions the whole table into this set and the generated
/// set, which is what makes a new op fail the build until someone says which
/// side it is on. The `_ => None` arm alone would answer "emit it inline" for
/// a new library op, and inline is not a kernel that exists.
///
/// `Family` cannot drive this: `Order` holds `sort_desc` and `top_k` (library)
/// alongside `pivot_threshold` (generated), and `ReduceScan` holds `cumsum`
/// and `cumprod` (library) alongside the four reductions (generated).
pub fn library_op_for_tag(tag: u8) -> Option<LibraryOp> {
    use pie_ir::op::tags;
    match tag {
        tags::TOP_K => Some(LibraryOp::TopK),
        tags::SORT_DESC => Some(LibraryOp::Sort),
        tags::CUMSUM | tags::CUMPROD => Some(LibraryOp::Scan),
        tags::MATMUL => Some(LibraryOp::MatMul),
        tags::KERNEL_CALL | tags::SINK_CALL => Some(LibraryOp::SecondParty),
        _ => None,
    }
}

pub(crate) fn region_kind_for_node(stage: &NormalizedStage, node: usize) -> RegionKind {
    match library_op_for_tag(stage.ops[node].tag()) {
        Some(library) => RegionKind::Library(library),
        None => RegionKind::Generated,
    }
}

pub(crate) fn compatible_schedule(first: &Op, next: &Op) -> bool {
    !matches!(
        (first, next),
        (
            Op::CumSum(_) | Op::CumProd(_) | Op::SortDesc(_) | Op::TopK { .. } | Op::MatMul(_, _),
            _
        ) | (
            _,
            Op::CumSum(_) | Op::CumProd(_) | Op::SortDesc(_) | Op::TopK { .. } | Op::MatMul(_, _)
        )
    )
}

/// A stage's SSA layout and consumer map, computed once.
///
/// `build_region` used to recompute `result_layout` and rebuild the whole
/// consumer map on every call, and `singleton_partition` calls it once per op
/// — so partitioning an N-op stage did N passes over N ops. That is a clean
/// quadratic: 128 ops planned in 1.2 ms, 1024 in 65 ms, 4096 in 1.1 s, with
/// `singleton_partition` accounting for >95% of it. Since a stage body is
/// bounded only by the container length, and the container is guest-supplied,
/// the curve was reachable from untrusted input. Hoisting the two tables out
/// of the loop makes partitioning linear.
pub(crate) struct StageIndex {
    /// First SSA id each op defines.
    bases: Vec<u32>,
    /// Op index that defines each SSA id.
    producer: Vec<usize>,
    /// Op indices reading each SSA id.
    consumers: Vec<Vec<u32>>,
}

impl StageIndex {
    fn of(stage: &NormalizedStage) -> Self {
        let (bases, producer) = result_layout(&stage.ops);
        let mut consumers: Vec<Vec<u32>> = vec![Vec::new(); stage.value_types.len()];
        for (node, op) in stage.ops.iter().enumerate() {
            for operand in op.operands() {
                consumers[operand as usize].push(node as u32);
            }
        }
        Self {
            bases,
            producer,
            consumers,
        }
    }
}

pub(crate) fn build_region(
    stage: &NormalizedStage,
    index: &StageIndex,
    nodes: Vec<u32>,
    kind: RegionKind,
) -> Region {
    let node_set: BTreeSet<u32> = nodes.iter().copied().collect();
    let StageIndex {
        bases,
        producer,
        consumers,
    } = index;

    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut sinks = Vec::new();
    for &node in &nodes {
        let op = &stage.ops[node as usize];
        for operand in op.operands() {
            if !node_set.contains(&(producer[operand as usize] as u32)) {
                inputs.insert(operand);
            }
        }
        if let Op::ChanPut { chan, value } = *op {
            sinks.push(ChannelSink {
                channel_slot: chan,
                value,
            });
        }
        let base = bases[node as usize];
        for result in 0..op.result_count() {
            let value = base + result;
            if consumers[value as usize]
                .iter()
                .any(|consumer| !node_set.contains(consumer))
            {
                outputs.insert(value);
            }
        }
    }

    let schedule = match kind {
        RegionKind::Library(_) => ScheduleTemplate::Library,
        RegionKind::Generated => {
            let has_compute = nodes.iter().any(|node| {
                !matches!(
                    stage.ops[*node as usize],
                    Op::ChanTake(_) | Op::ChanRead(_) | Op::ChanPut { .. } | Op::SinkCall { .. }
                )
            });
            let hierarchical = nodes.iter().any(|node| {
                let op = &stage.ops[*node as usize];
                if !matches!(
                    op,
                    Op::ReduceSum(_) | Op::ReduceMax(_) | Op::ReduceMin(_) | Op::ReduceArgmax(_)
                ) {
                    return false;
                }
                op.operands()
                    .first()
                    .and_then(|value| stage.value_types.get(*value as usize))
                    .and_then(|value_type| value_type.dims.last())
                    .is_some_and(|dimension| {
                        matches!(dimension, Dimension::Static(length) if *length > 32_768)
                    })
            });
            if !has_compute {
                ScheduleTemplate::Effects
            } else if hierarchical {
                ScheduleTemplate::HierarchicalRow
            } else {
                ScheduleTemplate::OneCtaPerRow
            }
        }
    };

    Region {
        kind,
        schedule,
        nodes,
        inputs: inputs.into_iter().collect(),
        outputs: outputs.into_iter().collect(),
        sinks,
    }
}
