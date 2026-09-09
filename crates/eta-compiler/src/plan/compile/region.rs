use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use eta_ir::op::{Family, IntrinsicId, Op};
use eta_ir::types::ValueId;

use super::normalize::{ChannelSlot, NodeIndex, NormalizedStage, result_layout};
use super::nucleus::LibraryMatch;
use super::symbolic::Dimension;

eta_ir::declare_tagged_enum! {
    pub enum ScheduleTemplate {
        Effects = 0, "effects";
        OneCtaPerRow = 1, "one_cta_per_row";
        HierarchicalRow = 2, "hierarchical_row";
        Library = 3, "library";
    }
}

eta_ir::declare_tagged_enum! {
    #[derive(serde::Serialize, serde::Deserialize)]
    pub enum LibraryOp {
        NucleusSample = 0, "nucleus_sample";
        TopK = 1, "top_k";
        Sort = 2, "sort";
        Scan = 3, "scan";
        MatMul = 4, "matmul";
        SecondParty = 5, "second_party";
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum RegionKind {
    #[default]
    Generated,
    Library(LibraryOp),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelSink {
    pub channel_slot: ChannelSlot,
    pub value: ValueId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    pub kind: RegionKind,
    pub schedule: ScheduleTemplate,
    pub nodes: Vec<NodeIndex>,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub sinks: Vec<ChannelSink>,
    pub row_value: Option<ValueId>,
    pub row_alias: Option<u64>,
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
                    && !matches!(
                        op.tag(),
                        eta_ir::op::tags::INTRINSIC_VAL | eta_ir::op::tags::CONST
                    ) =>
            {
                Some((*fixed, *extent))
            }
            _ => None,
        })
        .collect();
    let class = |node: NodeIndex| -> (Geometry, Option<ValueId>) {
        match geometries[node.index()] {
            (Geometry::Rows { fixed, extent }, witness)
                if arithmetic.contains(&(fixed, extent)) =>
            {
                (Geometry::Rows { fixed, extent }, witness)
            }
            _ => (Geometry::Single, None),
        }
    };
    let joins_any = |node: NodeIndex| {
        stage.ops[node.index()].tag() == eta_ir::op::tags::CONST
            && geometries[node.index()].0 == Geometry::Single
    };

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
            Some((seen, None)) if witness.is_some() => run = Some((seen, witness)),
            _ => {}
        }

        generated.push(node);
    }
    flush_generated_run(stage, index, &mut regions, &mut generated, &mut run, alias);
    RegionPartition {
        kind: PartitionKind::Fused,
        regions,
        whole_stage_fallback: false,
    }
}

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
            alias
                .filter(|&(role, _)| fixed == 1 && role == extent)
                .map(|(_, n)| n),
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

pub(crate) struct StageIndex {
    bases: Vec<ValueId>,
    producer: Vec<NodeIndex>,
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

    pub(crate) fn producer(&self, value: ValueId) -> Option<NodeIndex> {
        self.producer.get(value as usize).copied()
    }

    pub(crate) fn base(&self, node: NodeIndex) -> Option<ValueId> {
        self.bases.get(node.index()).copied()
    }

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
    let produced_here = |value: ValueId| {
        index
            .producer(value)
            .is_some_and(|producer| node_set.contains(&producer))
    };

    let mut inputs = BTreeSet::new();
    let mut outputs = BTreeSet::new();
    let mut sinks = Vec::new();
    for (other, _) in stage.ops.iter().enumerate() {
        let other = NodeIndex(other as u32);
        let Some(DirectTopK {
            divisor: Some(divisor),
            ..
        }) = direct_topk(stage, index, other)
        else {
            continue;
        };
        match (node_set.contains(&other), produced_here(divisor)) {
            (true, false) => {
                inputs.insert(divisor);
            }
            (false, true) => {
                outputs.insert(divisor);
            }
            _ => {}
        }
    }
    for &node in &nodes {
        let op = &stage.ops[node.index()];
        for operand in op.operands() {
            if !produced_here(operand) {
                inputs.insert(operand);
            }
        }
        if let Op::ChanPut { chan, value } = *op {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Geometry {
    Single,
    Rows { fixed: u64, extent: u32 },
    Mixed,
}

pub(crate) struct DirectTopK {
    pub(crate) intrinsic: NodeIndex,
    pub(crate) divisor: Option<ValueId>,
}

pub(crate) fn direct_topk(
    stage: &NormalizedStage,
    index: &StageIndex,
    node: NodeIndex,
) -> Option<DirectTopK> {
    let Op::TopK { input, .. } = stage.ops.get(node.index())? else {
        return None;
    };
    let input = *input;
    let through_reshapes = |mut value: ValueId| -> ValueId {
        while let Some(producer) = index.producer(value) {
            match stage.ops[producer.index()] {
                Op::Reshape { value: inner, .. } => value = inner,
                _ => break,
            }
        }
        value
    };
    let one_element = |value: ValueId| -> bool {
        stage.value_types.get(value as usize).is_some_and(|ty| {
            ty.dims
                .iter()
                .all(|dim| matches!(dim, Dimension::Static(1)))
        })
    };
    let mut value = through_reshapes(input);
    let mut divisor = None;
    if let Some(producer) = index.producer(value)
        && let Op::Div(numerator, element) = stage.ops[producer.index()]
        && one_element(element)
    {
        divisor = Some(element);
        value = through_reshapes(numerator);
    }
    let intrinsic = index.producer(value)?;
    let Op::IntrinsicVal { intr, .. } = stage.ops[intrinsic.index()] else {
        return None;
    };
    if !matches!(intr, IntrinsicId::Logits | IntrinsicId::MtpLogits) {
        return None;
    }
    let plane = &stage.value_types.get(index.base(intrinsic)? as usize)?.dims;
    let ranked = &stage.value_types.get(input as usize)?.dims;
    let same_rows = value_rows(plane).is_some() && value_rows(plane) == value_rows(ranked);
    (same_rows && plane.last() == ranked.last()).then_some(DirectTopK { intrinsic, divisor })
}

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

pub fn is_row_vector(dims: &[Dimension], fixed: u64, extent: u32, alias: Option<u64>) -> bool {
    match dims {
        [Dimension::Static(n)] => {
            (extent == u32::MAX && *n as u64 == fixed) || (fixed == 1 && alias == Some(*n as u64))
        }
        [Dimension::Symbolic(role)] => fixed == 1 && *role as u32 == extent,
        _ => false,
    }
}

pub fn same_rows(shape: (u64, u32), geometry: (u64, u32), alias: Option<u64>) -> bool {
    shape == geometry || (geometry.0 == 1 && alias.is_some_and(|n| shape == (n, u32::MAX)))
}

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
            let Some(ty) = stage.value_types.get(value as usize) else {
                continue;
            };
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

fn row_parallel_tag(tag: u8) -> bool {
    use eta_ir::op::tags;
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

pub(crate) fn node_geometry(
    stage: &NormalizedStage,
    index: &StageIndex,
    node: NodeIndex,
    alias: Option<(u32, u64)>,
) -> (Geometry, Option<ValueId>) {
    use eta_ir::op::tags;
    let op = &stage.ops[node.index()];
    let base = index.base(node).unwrap_or_default();
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
            && vectors.iter().all(|&v| {
                stage.value_types.get(v as usize).map(|t| t.dims.as_slice()) == Some(&[*dim])
            })
        {
            return match dim {
                Dimension::Static(n) => {
                    let (fixed, extent) = canonical((*n as u64, u32::MAX));
                    (Geometry::Rows { fixed, extent }, None)
                }
                Dimension::Symbolic(role) => (
                    Geometry::Rows {
                        fixed: 1,
                        extent: *role as u32,
                    },
                    None,
                ),
            };
        }
        return (Geometry::Single, None);
    };
    if !whitelisted {
        return (Geometry::Mixed, None);
    }
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
