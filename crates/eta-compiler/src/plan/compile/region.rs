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

use eta_ir::op::{Family, Op};
use eta_ir::types::ValueId;

use super::normalize::{ChannelSlot, NodeIndex, NormalizedStage, result_layout};
use super::nucleus::LibraryMatch;
use super::symbolic::Dimension;

eta_ir::declare_tagged_enum! {
    /// How a region is scheduled on a device.
    ///
    /// Enumerated into the C header as `PtirScheduleTemplate`.
    pub enum ScheduleTemplate {
        /// The region only moves channel traffic and emits no arithmetic.
        Effects = 0, "effects";
        /// One cooperative thread array per row — the default for compute.
        OneCtaPerRow = 1, "one_cta_per_row";
        /// A row reduction wide enough (last dim > 32768) to split
        /// hierarchically across blocks.
        HierarchicalRow = 2, "hierarchical_row";
        /// The region is a library call, scheduled by that kernel.
        Library = 3, "library";
    }
}

eta_ir::declare_tagged_enum! {
    /// A region a backend implements with a library kernel rather than
    /// generated code. Enumerated into the C header as `PtirLibraryOp`; the
    /// wire numbering is the `= 0` ... `= 5` written below, so tags are
    /// explicit rather than left to declaration order.
    #[derive(serde::Serialize, serde::Deserialize)]
    pub enum LibraryOp {
        /// Fused nucleus (top-p) sampling: softmax, top-p mask, Gumbel noise,
        /// then argmax.
        NucleusSample = 0, "nucleus_sample";
        /// Top-k selection ([`Op::TopK`]).
        TopK = 1, "top_k";
        /// Descending sort ([`Op::SortDesc`]).
        Sort = 2, "sort";
        /// A prefix scan — [`Op::CumSum`] or [`Op::CumProd`].
        Scan = 3, "scan";
        /// Matrix multiply ([`Op::MatMul`]).
        MatMul = 4, "matmul";
        /// A second-party kernel or sink call ([`Op::KernelCall`] /
        /// [`Op::SinkCall`]).
        SecondParty = 5, "second_party";
    }
}

/// Whether a region is emitted as generated code or dispatched to a library.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum RegionKind {
    /// Emitted as generated device code.
    #[default]
    Generated,
    /// Dispatched to the named [`LibraryOp`] kernel.
    Library(LibraryOp),
}

/// A channel write performed by a region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSink {
    /// The stage-local channel slot written, indexing
    /// [`NormalizedStage::channel_bindings`].
    pub channel_slot: ChannelSlot,
    /// The value whose contents are put into the channel.
    pub value: ValueId,
}

/// One schedulable unit of a partition: a set of ops, its device schedule, and
/// the values crossing its boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Whether the region is generated or a library call.
    pub kind: RegionKind,
    /// The device schedule chosen for the region.
    pub schedule: ScheduleTemplate,
    /// Positions in the stage's op list.
    pub nodes: Vec<NodeIndex>,
    /// Values the region reads from outside itself.
    pub inputs: Vec<ValueId>,
    /// Values the region defines that something outside it reads.
    pub outputs: Vec<ValueId>,
    /// Channel writes ([`Op::ChanPut`]) performed inside the region.
    pub sinks: Vec<ChannelSink>,
    /// A multi-row value naming the region's row geometry, when every op
    /// in it works row by row (see [`node_geometry`]): a backend may then
    /// launch one block per row, each block seeing only its row of every
    /// multi-row value. For a `top_k`/`sort_desc` library region it names
    /// the sorted input, for the same purpose. `None`: one block per lane.
    pub row_value: Option<ValueId>,
    /// When the geometry's rows are symbolic, the STATIC row count the
    /// program's own arithmetic equates with it (`add([rows, v],
    /// [256, v])`): a value with that many static rows is of the geometry
    /// too. See [`row_alias`].
    pub row_alias: Option<u64>,
}

/// Which of a stage's two partitions a [`RegionPartition`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PartitionKind {
    /// One region per op; always correct, never fused.
    Singleton = 0,
    /// Ops grouped by schedule with library dataflows lifted out.
    Fused = 1,
}

/// A stage partitioned into regions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionPartition {
    /// Which partition this is.
    pub kind: PartitionKind,
    /// The regions, in stage op order.
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
            // Every pattern names at least one node (an empty match would
            // describe a library call over no ops).
            (
                *candidate.nodes.last().expect("library match has nodes"),
                candidate,
            )
        })
        .collect();
    // Row geometries some arithmetic op works in. A multi-row value only an
    // intrinsic copy or a constant touches (the one-row sampler's
    // `[rows, vocab]` logits, reshaped to a vector at once) is not one:
    // its ops stay in the one-block run with their consumers, where the
    // direct-argmax path finds them.
    let alias = row_alias(stage, index);
    let geometries: Vec<(Geometry, Option<ValueId>)> = (0..stage.ops.len() as u32)
        .map(NodeIndex)
        .map(|node| node_geometry(stage, index, node, alias))
        .collect();
    let arithmetic: BTreeSet<(u64, u32)> = geometries
        .iter()
        .zip(&stage.ops)
        .filter_map(|((geometry, witness), op)| match geometry {
            Geometry::Rows { fixed, extent }
                if witness.is_some()
                    && !matches!(op.tag(), eta_ir::op::tags::INTRINSIC_VAL | eta_ir::op::tags::CONST) =>
            {
                Some((*fixed, *extent))
            }
            _ => None,
        })
        .collect();
    let class = |node: NodeIndex| -> (Geometry, Option<ValueId>) {
        match geometries[node.index()] {
            (Geometry::Rows { fixed, extent }, witness) if arithmetic.contains(&(fixed, extent)) => {
                (Geometry::Rows { fixed, extent }, witness)
            }
            // One block per lane, fused with its neighbours as always.
            _ => (Geometry::Single, None),
        }
    };
    // A scalar constant joins whatever run it sits in: every block of a row
    // run writes the same word.
    let joins_any = |node: NodeIndex| stage.ops[node.index()].tag() == eta_ir::op::tags::CONST
        && geometries[node.index()].0 == Geometry::Single;

    let mut regions = Vec::new();
    let mut generated = Vec::new();
    let mut run: Option<(Geometry, Option<ValueId>)> = None;
    for node in (0..stage.ops.len() as u32).map(NodeIndex) {
        if matched_nodes.contains(&node) {
            flush_generated_run(stage, index, &mut regions, &mut generated, &mut run, alias);
            if let Some(candidate) = matches_by_end.get(&node) {
                regions.push(build_library_match_region(stage, index, candidate));
            }
            continue;
        }

        let kind = region_kind_for_node(stage, node);
        if matches!(kind, RegionKind::Library(_)) {
            flush_generated_run(stage, index, &mut regions, &mut generated, &mut run, alias);
            let mut region = build_region(stage, index, vec![node], kind);
            if matches!(kind, RegionKind::Library(LibraryOp::TopK | LibraryOp::Sort)) {
                region.row_value = stage.ops[node.index()].operands().first().copied();
            }
            regions.push(region);
            continue;
        }

        // Generated ops fuse while they share a class: one row geometry the
        // launch splits by rows, or the one-block run.
        if joins_any(node) && run.is_some() {
            generated.push(node);
            continue;
        }
        let (geometry, witness) = class(node);
        if run.is_some_and(|(seen, _)| seen != geometry) {
            flush_generated_run(stage, index, &mut regions, &mut generated, &mut run, alias);
        }
        match run {
            None => run = Some((geometry, witness)),
            // A run opened by vector-only ops learns its witness from the
            // first multi-row op.
            Some((seen, None)) if witness.is_some() => run = Some((seen, witness)),
            _ => {}
        }

        // The only ops worth refusing to fuse are all `library_op_for_tag`
        // ops, already routed above, so nothing else can break the run here.
        generated.push(node);
    }
    flush_generated_run(stage, index, &mut regions, &mut generated, &mut run, alias);
    RegionPartition {
        kind: PartitionKind::Fused,
        regions,
        whole_stage_fallback: false,
    }
}

/// [`flush_generated_region`], stamping the run's row geometry witness.
fn flush_generated_run(
    stage: &NormalizedStage,
    index: &StageIndex,
    regions: &mut Vec<Region>,
    nodes: &mut Vec<NodeIndex>,
    run: &mut Option<(Geometry, Option<ValueId>)>,
    alias: Option<(u32, u64)>,
) {
    let (witness, row_alias) = match run.take() {
        Some((Geometry::Rows { fixed, extent }, witness)) => (
            witness,
            alias.filter(|&(role, _)| fixed == 1 && role == extent).map(|(_, n)| n),
        ),
        _ => (None, None),
    };
    let before = regions.len();
    flush_generated_region(stage, index, regions, nodes);
    if regions.len() > before
        && let Some(region) = regions.last_mut()
    {
        region.row_value = witness;
        region.row_alias = row_alias;
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
/// generated kernel emits it inline. Keyed on the tag, not the `Op` variant
/// or `Family` (a `Family` can mix library and generated ops), so a new op
/// can be enumerated against [`eta_ir::op::OP_TABLE`] and classified rather
/// than silently emitted inline.
pub fn library_op_for_tag(tag: u8) -> Option<LibraryOp> {
    use eta_ir::op::tags;
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

/// A stage's SSA layout and consumer map, computed once per stage and
/// shared: `build_region` runs once per op, so recomputing this per call
/// would be quadratic in stage size, and a stage body is guest-supplied
/// (untrusted) length. Also the only place the node space and value space
/// meet, keyed one way with typed accessors rather than three bare parallel
/// slices.
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
            // `chan` is already stage-local: `localize_stage` rewrote it
            // before regions were formed.
            sinks.push(ChannelSink {
                channel_slot: ChannelSlot(chan),
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
            // Channel-only traffic gets the effects-only schedule;
            // `kernel_call`/`sink_call` are already routed to
            // `RegionKind::Library`, so no other op here emits no arithmetic
            // except channel ops.
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
        row_value: None,
        row_alias: None,
    }
}

/// The row geometry of one node, for the row-parallel launch of a generated
/// region: the `(fixed rows, symbolic row extent)` every multi-row tensor
/// it touches shares (`Rows`), no multi-row tensor at all (`Single` — the
/// one-row sampler's every op, fused as before), or a node that touches a
/// multi-row tensor and may not be split by rows (`Mixed`): one that mixes
/// geometries, cannot say its row width, or reaches across rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Geometry {
    Single,
    Rows { fixed: u64, extent: u32 },
    Mixed,
}

/// The rows of a value: the product of its leading dims, one of which may
/// be symbolic. Rank 0 and rank 1 are one row. `None` when a rank-2+ value
/// has a symbolic trailing dim (its row width is unknown) or two symbolic
/// leading dims.
pub fn value_rows(dims: &[Dimension]) -> Option<(u64, u32)> {
    let Some((last, leading)) = dims.split_last() else {
        return Some((1, u32::MAX));
    };
    if leading.is_empty() {
        return Some((1, u32::MAX));
    }
    if !matches!(last, Dimension::Static(width) if *width > 0) {
        return None;
    }
    let mut fixed = 1u64;
    let mut extent = u32::MAX;
    for dimension in leading {
        match dimension {
            Dimension::Symbolic(role) => {
                if extent != u32::MAX {
                    return None;
                }
                extent = *role as u32;
            }
            Dimension::Static(value) => {
                if *value == 0 {
                    return None;
                }
                fixed = fixed.checked_mul(*value as u64)?;
            }
        }
    }
    Some((fixed, extent))
}

/// Whether `dims` is the rank-1 vector of a geometry's rows — the shape a
/// row reduction writes one element of per row. `alias` is the static row
/// count that stands for a symbolic geometry ([`row_alias`]).
pub fn is_row_vector(dims: &[Dimension], fixed: u64, extent: u32, alias: Option<u64>) -> bool {
    match dims {
        [Dimension::Static(n)] => {
            (extent == u32::MAX && *n as u64 == fixed) || (fixed == 1 && alias == Some(*n as u64))
        }
        [Dimension::Symbolic(role)] => fixed == 1 && *role as u32 == extent,
        _ => false,
    }
}

/// Whether a multi-row `shape` (from [`value_rows`]) is `geometry`, given
/// `alias` — the static row count standing for the geometry's symbolic rows.
pub fn same_rows(shape: (u64, u32), geometry: (u64, u32), alias: Option<u64>) -> bool {
    shape == geometry || (geometry.0 == 1 && alias.is_some_and(|n| shape == (n, u32::MAX)))
}

/// The one static row count a stage's arithmetic equates with a symbolic
/// row extent: an op whose multi-row tensors are exactly `[role, w]` and
/// `[n, w]` (the trace's own shape check let it through, so `role` was `n`
/// when the trace ran) says `(role, n)`. `None` when no op does, or two
/// disagree.
pub(crate) fn row_alias(stage: &NormalizedStage, index: &StageIndex) -> Option<(u32, u64)> {
    let mut found: Option<(u32, u64)> = None;
    for node in (0..stage.ops.len() as u32).map(NodeIndex) {
        let op = &stage.ops[node.index()];
        let base = index.base(node).unwrap_or_default();
        let mut symbolic: Option<u32> = None;
        let mut fixed: Option<u64> = None;
        let values = op
            .operands()
            .into_iter()
            .chain((0..op.result_count()).map(|result| base + result));
        for value in values {
            let Some(ty) = stage.value_types.get(value as usize) else { continue };
            match value_rows(&ty.dims) {
                Some((1, u32::MAX)) | None => {}
                Some((1, role)) => symbolic = Some(role),
                Some((n, u32::MAX)) => fixed = Some(n),
                Some(_) => {}
            }
        }
        if let (Some(role), Some(n)) = (symbolic, fixed) {
            match found {
                None => found = Some((role, n)),
                Some(seen) if seen == (role, n) => {}
                Some(_) => return None,
            }
        }
    }
    found
}

/// The ops a row-parallel block may run on its row alone: elementwise maps
/// (an operand of one element broadcasts), row reductions, broadcasts,
/// constants, the intrinsic copy, the keyed RNG (its key is the element's
/// position, which the backend offsets by the row) and `gather_row`. Not:
/// `iota` and the masks (they generate by absolute position), reshape
/// (it re-rows), gathers/scatters/transposes/pivots (they reach across
/// rows), and channel traffic (a copy of the whole cell).
fn row_parallel_tag(tag: u8) -> bool {
    use eta_ir::op::tags;
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
            | tags::RNG
            | tags::RNG_KEYED
            | tags::REDUCE_SUM
            | tags::REDUCE_MAX
            | tags::REDUCE_MIN
            | tags::REDUCE_ARGMAX
            | tags::BROADCAST
            | tags::CONST
            | tags::INTRINSIC_VAL
            | tags::GATHER_ROW
    )
}

/// [`Geometry`] of `node`, with a multi-row value witnessing it.
pub(crate) fn node_geometry(
    stage: &NormalizedStage,
    index: &StageIndex,
    node: NodeIndex,
    alias: Option<(u32, u64)>,
) -> (Geometry, Option<ValueId>) {
    use eta_ir::op::tags;
    let op = &stage.ops[node.index()];
    let base = index.base(node).unwrap_or_default();
    // A static row count the stage equates with a symbolic one reads as
    // the symbolic geometry.
    let canonical = |shape: (u64, u32)| match (shape, alias) {
        ((n, u32::MAX), Some((role, m))) if n == m => (1, role),
        _ => shape,
    };
    let static_rows = alias.map(|(_, n)| n);
    let operands = op.operands();
    let values = operands
        .iter()
        .copied()
        .chain((0..op.result_count()).map(|result| base + result));
    let mut rows: Option<(u64, u32)> = None;
    let mut witness = None;
    let mut vectors: Vec<ValueId> = Vec::new();
    for value in values {
        let Some(ty) = stage.value_types.get(value as usize) else {
            return (Geometry::Mixed, None);
        };
        let Some(shape) = value_rows(&ty.dims) else {
            return (Geometry::Mixed, None);
        };
        let shape = canonical(shape);
        if shape == (1, u32::MAX) {
            if ty.dims.len() == 1 {
                vectors.push(value);
            }
            continue;
        }
        match rows {
            None => {
                rows = Some(shape);
                witness = Some(value);
            }
            Some(seen) if seen == shape => {}
            Some(_) => return (Geometry::Mixed, None),
        }
    }
    let whitelisted = row_parallel_tag(op.tag());
    let Some((fixed, extent)) = rows else {
        // No multi-row tensor. A row block can still own one element of a
        // per-row VECTOR (what a row reduction writes) in an elementwise
        // map or a keyed draw — that is what keeps `-h`, `select(accept,
        // ..)` and the like inside the row run. The geometry is only known
        // once some multi-row op names it, so this is a candidate that
        // `fused_partition` confirms against the arithmetic geometries; a
        // reduction of a vector reads across the rows and stays one block.
        // Everything else is one row's work, fused as it always was.
        if whitelisted
            && !matches!(
                op.tag(),
                tags::REDUCE_SUM
                    | tags::REDUCE_MAX
                    | tags::REDUCE_MIN
                    | tags::REDUCE_ARGMAX
                    | tags::GATHER_ROW
                    | tags::INTRINSIC_VAL
                    | tags::CONST
            )
            && !vectors.is_empty()
            && let Some(&first) = vectors.first()
            && let Some(ty) = stage.value_types.get(first as usize)
            && let [dim] = ty.dims.as_slice()
            && vectors
                .iter()
                .all(|&v| stage.value_types.get(v as usize).map(|t| t.dims.as_slice()) == Some(&[*dim]))
        {
            return match dim {
                Dimension::Static(n) => {
                    let (fixed, extent) = canonical((*n as u64, u32::MAX));
                    (Geometry::Rows { fixed, extent }, None)
                }
                Dimension::Symbolic(role) => (Geometry::Rows { fixed: 1, extent: *role as u32 }, None),
            };
        }
        return (Geometry::Single, None);
    };
    if !whitelisted {
        return (Geometry::Mixed, None);
    }
    // A reduction's per-row result is a vector of the rows; any other rank-1
    // operand or result must be that vector too (the broadcast source, the
    // `gather_row` index), or the op reads across rows — except a one-element
    // vector, and a draw's `[key, counter]` state, which every block reads
    // whole.
    let state = matches!(op.tag(), tags::RNG | tags::RNG_KEYED);
    if vectors.iter().any(|&v| {
        stage.value_types.get(v as usize).is_none_or(|t| {
            !is_row_vector(&t.dims, fixed, extent, static_rows)
                && !matches!(t.dims.as_slice(), [Dimension::Static(1)])
                && !(state && operands.contains(&v))
        })
    }) {
        return (Geometry::Mixed, None);
    }
    (Geometry::Rows { fixed, extent }, witness)
}
