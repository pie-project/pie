//! Partitioning the normalized op DAG into regions.
//!
//! Two partitions come out of every stage: a singleton partition (one region
//! per op, the always-correct fallback) and a fused partition that groups ops
//! sharing a schedule and lifts recognized dataflows -- nucleus sampling,
//! top-k, sort, scan, matmul -- into library calls.
//!
//! Recognizing those dataflows is a separate question with its own file. A
//! single-op lift is a tag lookup ([`library_op_for_tag`]); the multi-op ones
//! are pattern matches over the DAG, and they live in [`super::nucleus`].
//! This file only asks what to do with a match once it has one.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use pie_ir::op::{ChannelIndex, Family, Op};
use pie_ir::types::ValueId;

use super::normalize::{NodeIndex, NormalizedStage, result_layout};
use super::nucleus::LibraryMatch;
use super::symbolic::Dimension;

pie_ir::declare_tagged_enum! {
    /// How a region is scheduled on a device.
    ///
    /// Enumerated into the C header as `PtirScheduleTemplate`.
    pub enum ScheduleTemplate {
        Effects = 0, "effects";
        OneCtaPerRow = 1, "one_cta_per_row";
        HierarchicalRow = 2, "hierarchical_row";
        Library = 3, "library";
    }
}

pie_ir::declare_tagged_enum! {
    /// A region a backend implements with a library kernel rather than
    /// generated code.
    ///
    /// Enumerated into the C header as `PtirLibraryOp`.
    pub enum LibraryOp {
        NucleusSample = 0, "nucleus_sample";
        TopK = 1, "top_k";
        Sort = 2, "sort";
        Scan = 3, "scan";
        MatMul = 4, "matmul";
        SecondParty = 5, "second_party";
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionKind {
    Generated,
    Library(LibraryOp),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSink {
    pub channel_slot: ChannelIndex,
    pub value: ValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub kind: RegionKind,
    pub schedule: ScheduleTemplate,
    /// Positions in the stage's op list.
    pub nodes: Vec<NodeIndex>,
    /// Values the region reads from outside itself.
    pub inputs: Vec<ValueId>,
    /// Values the region defines that something outside it reads.
    pub outputs: Vec<ValueId>,
    pub sinks: Vec<ChannelSink>,
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

pub(crate) fn singleton_partition(stage: &NormalizedStage, index: &StageIndex) -> RegionPartition {
    let regions = (0..stage.ops.len() as u32)
        .map(NodeIndex)
        .map(|node| build_region(stage, index, vec![node], region_kind_for_node(stage, node)))
        .collect();
    RegionPartition {
        kind: PartitionKind::Singleton,
        regions,
        whole_stage_fallback: false,
    }
}

pub(crate) fn fused_partition(
    stage: &NormalizedStage,
    index: &StageIndex,
    library_matches: &[LibraryMatch],
) -> RegionPartition {
    let matched_nodes: BTreeSet<NodeIndex> = library_matches
        .iter()
        .flat_map(|candidate| candidate.nodes.iter().copied())
        .collect();
    let matches_by_end: BTreeMap<NodeIndex, &LibraryMatch> = library_matches
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
    for node in (0..stage.ops.len() as u32).map(NodeIndex) {
        if matched_nodes.contains(&node) {
            flush_generated_region(stage, index, &mut regions, &mut generated);
            if let Some(candidate) = matches_by_end.get(&node) {
                regions.push(build_library_match_region(stage, index, candidate));
            }
            continue;
        }

        let kind = region_kind_for_node(stage, node);
        if matches!(kind, RegionKind::Library(_)) {
            flush_generated_region(stage, index, &mut regions, &mut generated);
            regions.push(build_region(stage, index, vec![node], kind));
            continue;
        }

        // Nothing else can break the run. `compatible_schedule` used to sit
        // here, refusing to fuse cumsum/cumprod/sort_desc/top_k/matmul with a
        // neighbour -- but every one of those five is a `library_op_for_tag`
        // op, so the branch above took them and `continue`d before this line
        // could see one. It was provably `true`, and read as a scheduling
        // policy the planner did not have. If a *generated* op ever needs its
        // own run, the check comes back here, against a table, not a list.
        generated.push(node);
    }
    flush_generated_region(stage, index, &mut regions, &mut generated);
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
    nodes: &mut Vec<NodeIndex>,
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

pub(crate) fn region_kind_for_node(stage: &NormalizedStage, node: NodeIndex) -> RegionKind {
    match library_op_for_tag(stage.ops[node.index()].tag()) {
        Some(library) => RegionKind::Library(library),
        None => RegionKind::Generated,
    }
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
///
/// It is also the only place the node space and the value space meet: every
/// table here is keyed by one and yields the other, and the accessors are
/// what make that direction checkable. `recognize_library_dataflows` used to
/// rebuild all three by hand and pass them down as three bare slices, so the
/// nucleus matcher took `bases`, `producer` and `consumers` next to each
/// other and had to cast on every use.
pub(crate) struct StageIndex {
    /// First SSA id each op defines.
    bases: Vec<ValueId>,
    /// Node that defines each SSA id.
    producer: Vec<NodeIndex>,
    /// Nodes reading each SSA id.
    consumers: Vec<Vec<NodeIndex>>,
}

impl StageIndex {
    pub(crate) fn of(stage: &NormalizedStage) -> Self {
        let (bases, producer) = result_layout(&stage.ops);
        let mut consumers: Vec<Vec<NodeIndex>> = vec![Vec::new(); stage.value_types.len()];
        for (node, op) in stage.ops.iter().enumerate() {
            for operand in op.operands() {
                consumers[operand as usize].push(NodeIndex(node as u32));
            }
        }
        Self {
            bases,
            producer,
            consumers,
        }
    }

    /// The node that defines `value`.
    pub(crate) fn producer(&self, value: ValueId) -> Option<NodeIndex> {
        self.producer.get(value as usize).copied()
    }

    /// The first SSA id `node` defines.
    pub(crate) fn base(&self, node: NodeIndex) -> Option<ValueId> {
        self.bases.get(node.index()).copied()
    }

    /// The nodes that read `value`.
    pub(crate) fn consumers(&self, value: ValueId) -> Option<&[NodeIndex]> {
        self.consumers.get(value as usize).map(Vec::as_slice)
    }
}

pub(crate) fn build_region(
    stage: &NormalizedStage,
    index: &StageIndex,
    nodes: Vec<NodeIndex>,
    kind: RegionKind,
) -> Region {
    let node_set: BTreeSet<NodeIndex> = nodes.iter().copied().collect();

    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut sinks = Vec::new();
    for &node in &nodes {
        let op = &stage.ops[node.index()];
        for operand in op.operands() {
            if !index
                .producer(operand)
                .is_some_and(|producer| node_set.contains(&producer))
            {
                inputs.insert(operand);
            }
        }
        if let Op::ChanPut { chan, value } = *op {
            sinks.push(ChannelSink {
                channel_slot: chan,
                value,
            });
        }
        let base = index.base(node).unwrap_or_default();
        for result in 0..op.result_count() {
            let value = base + result;
            if index
                .consumers(value)
                .is_some_and(|consumers| consumers.iter().any(|c| !node_set.contains(c)))
            {
                outputs.insert(value);
            }
        }
    }

    let schedule = match kind {
        RegionKind::Library(_) => ScheduleTemplate::Library,
        RegionKind::Generated => {
            // A generated region that is nothing but channel traffic gets
            // the effects-only schedule. `Family::Channel` is the whole
            // answer here: `library_op_for_tag` already routed `kernel_call`
            // and `sink_call` to `RegionKind::Library`, so the only ops that
            // can reach this arm and emit no arithmetic are the channel ops.
            // The variant list this replaces named `sink_call` (unreachable)
            // and omitted `kernel_call` (also unreachable) — two mistakes
            // that cancelled, and would not have next time.
            let has_compute = nodes
                .iter()
                .any(|node| stage.ops[node.index()].family() != Family::Channel);
            let hierarchical = nodes.iter().any(|node| {
                let op = &stage.ops[node.index()];
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
