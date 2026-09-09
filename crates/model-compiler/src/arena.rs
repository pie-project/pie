use model_ir::{
    ClassSet, ClassTable, Def, Dim, Dtype, Operands, RowAxis, RuntimeInput, StructKind, Trace, Ty,
    ValueId,
};

use crate::budget::Budgets;
use crate::compiled::Region;
use crate::error::{Error, Share, Unrectangled};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readers {
    EveryClass,
    ItsOwnClasses,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Export {
    seam: &'static str,
    read_by: Readers,
}

const EXPORTS: [Export; 7] = [
    Export {
        seam: "out",
        read_by: Readers::EveryClass,
    },
    Export {
        seam: "mtp",
        read_by: Readers::ItsOwnClasses,
    },
    Export {
        seam: "attn.scores",
        read_by: Readers::ItsOwnClasses,
    },
    Export {
        seam: "mtp.drafts",
        read_by: Readers::ItsOwnClasses,
    },
    Export {
        seam: "velocity",
        read_by: Readers::ItsOwnClasses,
    },
    Export {
        seam: "hidden",
        read_by: Readers::ItsOwnClasses,
    },
    Export {
        seam: "pixels",
        read_by: Readers::ItsOwnClasses,
    },
];

pub const EXPORT_SEAMS: [&str; 7] = [
    EXPORTS[0].seam,
    EXPORTS[1].seam,
    EXPORTS[2].seam,
    EXPORTS[3].seam,
    EXPORTS[4].seam,
    EXPORTS[5].seam,
    EXPORTS[6].seam,
];

pub const FLOAT_READOUT_SEAMS: [&str; 3] = [EXPORTS[4].seam, EXPORTS[5].seam, EXPORTS[6].seam];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowExpr {
    Const(u64),
    Tokens,
    TokensTimes(u32),
    Lanes,
    LanesPlus(u32),
    Readouts,
    Patches,
    Images,
    ImagesPlus(u32),
    Voxels,
    VoxelsTimes(u32),
    Clips,
    ClipsPlus(u32),
}

impl RowExpr {
    #[must_use]
    pub fn of(dim: Dim) -> RowExpr {
        match dim {
            Dim::Const(n) => RowExpr::Const(n),
            Dim::Tokens => RowExpr::Tokens,
            Dim::TokensTimes(k) => RowExpr::TokensTimes(k),
            Dim::Lanes => RowExpr::Lanes,
            Dim::LanesPlus(k) => RowExpr::LanesPlus(k),
            Dim::Readouts => RowExpr::Readouts,
            Dim::Patches => RowExpr::Patches,
            Dim::Images => RowExpr::Images,
            Dim::ImagesPlus(k) => RowExpr::ImagesPlus(k),
            Dim::Voxels => RowExpr::Voxels,
            Dim::VoxelsTimes(k) => RowExpr::VoxelsTimes(k),
            Dim::Clips => RowExpr::Clips,
            Dim::ClipsPlus(k) => RowExpr::ClipsPlus(k),
        }
    }

    #[must_use]
    pub fn axis(self) -> Option<RowAxis> {
        match self {
            RowExpr::Const(_) => None,
            RowExpr::Tokens
            | RowExpr::TokensTimes(_)
            | RowExpr::Lanes
            | RowExpr::LanesPlus(_)
            | RowExpr::Readouts => Some(RowAxis::Tokens),
            RowExpr::Patches | RowExpr::Images | RowExpr::ImagesPlus(_) => Some(RowAxis::Patches),
            RowExpr::Voxels | RowExpr::VoxelsTimes(_) | RowExpr::Clips | RowExpr::ClipsPlus(_) => {
                Some(RowAxis::Voxels)
            }
        }
    }

    #[must_use]
    pub fn max(self, budgets: &Budgets) -> u64 {
        self.at(FireRows::ceilings(budgets))
    }

    #[must_use]
    pub fn cut_per_class(self) -> bool {
        match self {
            RowExpr::Tokens
            | RowExpr::TokensTimes(_)
            | RowExpr::Patches
            | RowExpr::Images
            | RowExpr::Voxels
            | RowExpr::VoxelsTimes(_)
            | RowExpr::Clips => true,
            RowExpr::Lanes
            | RowExpr::Const(_)
            | RowExpr::LanesPlus(_)
            | RowExpr::ImagesPlus(_)
            | RowExpr::ClipsPlus(_)
            | RowExpr::Readouts => false,
        }
    }

    #[must_use]
    pub fn at(self, fire: FireRows) -> u64 {
        match self {
            RowExpr::Const(n) => n,
            RowExpr::Tokens => fire.tokens,
            RowExpr::TokensTimes(k) => fire.tokens.saturating_mul(u64::from(k)),
            RowExpr::Lanes => fire.lanes,
            RowExpr::LanesPlus(k) => fire.lanes.saturating_add(u64::from(k)),
            RowExpr::Readouts => fire.readouts,
            RowExpr::Patches => fire.patches,
            RowExpr::Images => fire.images,
            RowExpr::ImagesPlus(k) => fire.images.saturating_add(u64::from(k)),
            RowExpr::Voxels => fire.voxels,
            RowExpr::VoxelsTimes(k) => fire.voxels.saturating_mul(u64::from(k)),
            RowExpr::Clips => fire.clips,
            RowExpr::ClipsPlus(k) => fire.clips.saturating_add(u64::from(k)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FireRows {
    pub tokens: u64,
    pub lanes: u64,
    pub patches: u64,
    pub images: u64,
    pub voxels: u64,
    pub clips: u64,
    pub readouts: u64,
}

impl FireRows {
    #[must_use]
    pub fn text_only(tokens: u64, lanes: u64) -> FireRows {
        FireRows {
            tokens,
            lanes,
            patches: 0,
            images: 0,
            voxels: 0,
            clips: 0,
            readouts: lanes,
        }
    }

    #[must_use]
    pub fn ceilings(budgets: &Budgets) -> FireRows {
        FireRows {
            tokens: u64::from(budgets.tokens.max_tokens),
            lanes: u64::from(budgets.tokens.max_lanes),
            patches: u64::from(budgets.max_patches()),
            images: u64::from(budgets.max_images()),
            voxels: u64::from(budgets.max_voxels()),
            clips: u64::from(budgets.max_clips()),
            readouts: u64::from(budgets.tokens.max_tokens),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub first: u32,
    pub last: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Concurrency {
    region_of: Vec<u32>,
    pairs: Vec<(u32, u32)>,
    paired: Vec<((u32, u32), (u32, u32))>,
}

impl Concurrency {
    #[must_use]
    pub fn sequential(regions: &[Region], nodes: usize) -> Concurrency {
        Concurrency {
            region_of: map_regions(regions, nodes),
            pairs: Vec::new(),
            paired: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_pairs(
        regions: &[Region],
        nodes: usize,
        pairs: impl IntoIterator<Item = (u32, u32)>,
    ) -> Concurrency {
        let mut pairs: Vec<(u32, u32)> = pairs
            .into_iter()
            .filter(|(a, b)| a != b)
            .map(|(a, b)| if a < b { (a, b) } else { (b, a) })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        let bounds = |r: u32| {
            let region = regions
                .get(r as usize)
                .expect("a fork pair names a region of this template");
            (region.nodes.start, region.nodes.end)
        };
        let paired = pairs.iter().map(|&(a, b)| (bounds(a), bounds(b))).collect();
        Concurrency {
            region_of: map_regions(regions, nodes),
            pairs,
            paired,
        }
    }

    #[must_use]
    pub fn region_of(&self, node: u32) -> u32 {
        self.region_of
            .get(node as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    #[must_use]
    pub fn pairs(&self) -> &[(u32, u32)] {
        &self.pairs
    }

    #[must_use]
    pub fn overlap(&self, a: Span, b: Span) -> bool {
        if a.first <= b.last && b.first <= a.last {
            return true;
        }
        self.paired
            .iter()
            .any(|&(one, two)| (meets(one, a) && meets(two, b)) || (meets(one, b) && meets(two, a)))
    }
}

fn meets((start, end): (u32, u32), span: Span) -> bool {
    start <= span.last && span.first < end
}

fn map_regions(regions: &[Region], nodes: usize) -> Vec<u32> {
    let mut map = vec![regions.len() as u32; nodes + 1];
    for (r, region) in regions.iter().enumerate() {
        for node in region.nodes.clone() {
            if let Some(slot) = map.get_mut(node as usize) {
                *slot = r as u32;
            }
        }
    }
    map
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    Runtime(RuntimeInput),
    Param(u32),
    Cache(u32),
    Arena {
        offset: u64,
        bytes: u64,
        rows: RowExpr,
        width: u64,
        dtype: Dtype,
    },
    Alias(ValueId),
    Struct(StructKind),
}

impl Placement {
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            Placement::Arena { bytes, .. } => *bytes,
            Placement::Runtime(_)
            | Placement::Param(_)
            | Placement::Cache(_)
            | Placement::Alias(_)
            | Placement::Struct(_) => 0,
        }
    }

    fn is_arena(&self) -> bool {
        matches!(self, Placement::Arena { .. })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArenaMap {
    pub placements: Vec<Placement>,
    pub spans: Vec<Option<Span>>,
    pub live_in: Vec<ClassSet>,
    pub bytes: u64,
}

impl ArenaMap {
    #[must_use]
    pub fn root(&self, value: ValueId) -> ValueId {
        root(&self.placements, value)
    }

    #[must_use]
    pub fn window(&self, value: ValueId, fire: FireRows) -> Option<Extent> {
        let root = self.root(value);
        match self.placements.get(root.0 as usize)? {
            Placement::Arena {
                offset,
                rows,
                width,
                dtype,
                ..
            } => Some(Extent {
                offset: *offset,
                bytes: rows
                    .at(fire)
                    .saturating_mul(*width)
                    .saturating_mul(elem_bytes(*dtype).unwrap_or(0)),
            }),
            _ => None,
        }
    }

    #[must_use]
    pub fn co_tenants(&self, a: ValueId, b: ValueId) -> bool {
        let (Some(x), Some(y)) = (
            self.placements.get(a.0 as usize),
            self.placements.get(b.0 as usize),
        ) else {
            return false;
        };
        let (
            Placement::Arena {
                offset: a_at,
                bytes: a_bytes,
                rows: a_rows,
                width: a_width,
                dtype: a_dtype,
            },
            Placement::Arena {
                offset: b_at,
                bytes: b_bytes,
                rows: b_rows,
                width: b_width,
                dtype: b_dtype,
            },
        ) = (x, y)
        else {
            return false;
        };
        if !a_rows.cut_per_class()
            || (a_at, a_bytes, a_rows, a_width, a_dtype)
                != (b_at, b_bytes, b_rows, b_width, b_dtype)
        {
            return false;
        }
        let (Some(a_in), Some(b_in)) = (
            self.live_in.get(a.0 as usize),
            self.live_in.get(b.0 as usize),
        ) else {
            return false;
        };
        !a_in.is_empty() && !b_in.is_empty() && a_in.disjoint(b_in)
    }

    #[must_use]
    pub fn clashes(&self, conc: &Concurrency) -> Vec<(ValueId, ValueId)> {
        self.clashes_blind(conc)
            .into_iter()
            .filter(|(a, b)| !self.co_tenants(*a, *b))
            .collect()
    }

    #[must_use]
    pub fn clashes_blind(&self, conc: &Concurrency) -> Vec<(ValueId, ValueId)> {
        let live = self.live();
        let mut found = Vec::new();
        for (i, (a, a_span, a_at, a_bytes)) in live.iter().enumerate() {
            for (b, b_span, b_at, b_bytes) in &live[i + 1..] {
                if conc.overlap(*a_span, *b_span)
                    && *a_at < b_at + b_bytes
                    && *b_at < a_at + a_bytes
                {
                    found.push((*a, *b));
                }
            }
        }
        found
    }

    #[must_use]
    pub fn live_bound(&self) -> u64 {
        let mut live = self.live();
        live.sort_by_key(|(_, _, offset, _)| *offset);
        let end = live.iter().map(|(_, s, _, _)| s.last).max().unwrap_or(0);
        let mut most = 0u64;
        for at in 0..=end {
            let mut total = 0u64;
            let mut column: Option<(u64, u64)> = None;
            for (_, span, offset, bytes) in &live {
                if span.first > at || at > span.last {
                    continue;
                }
                column = match column {
                    Some((held_at, held)) if held_at == *offset => {
                        Some((held_at, held.max(align(*bytes))))
                    }
                    Some((_, held)) => {
                        total += held;
                        Some((*offset, align(*bytes)))
                    }
                    None => Some((*offset, align(*bytes))),
                };
            }
            total += column.map_or(0, |(_, held)| held);
            most = most.max(total);
        }
        most
    }

    fn live(&self) -> Vec<(ValueId, Span, u64, u64)> {
        self.placements
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| match slot {
                Placement::Arena { offset, bytes, .. } => Some((
                    ValueId(id as u32),
                    self.spans[id].expect("the carve spans every arena slot"),
                    *offset,
                    *bytes,
                )),
                _ => None,
            })
            .collect()
    }
}

pub(crate) fn carve(
    trace: &Trace,
    budgets: &Budgets,
    classes: &ClassTable,
    conc: &Concurrency,
) -> Result<ArenaMap, Error> {
    let mut placements = rectangles(trace, budgets)?;
    fold_in_place(trace, &mut placements)?;
    fold_merges(trace, &mut placements)?;
    flatten(&mut placements);
    let (mut spans, live_in) = lives(trace, &placements, classes);
    outlive_the_region(trace, conc, &mut spans);

    let mut conservative = placements.clone();
    let blind = place(&mut conservative, &spans, &live_in, conc, Columns::PerValue);
    let shared = place(&mut placements, &spans, &live_in, conc, Columns::Shared);
    let (placements, bytes) = if shared < blind {
        (placements, shared)
    } else {
        (conservative, blind)
    };

    Ok(ArenaMap {
        placements,
        spans,
        live_in,
        bytes,
    })
}

fn rectangles(trace: &Trace, budgets: &Budgets) -> Result<Vec<Placement>, Error> {
    trace
        .values
        .iter()
        .enumerate()
        .map(|(id, decl)| {
            let value = ValueId(id as u32);
            match &decl.def {
                Def::Input(which) => Ok(Placement::Runtime(*which)),
                Def::Weight(i) => Ok(Placement::Param(*i)),
                Def::Cache(i) => Ok(Placement::Cache(*i)),
                Def::Op(_) | Def::Merge(_) => match &decl.ty {
                    Ty::Struct(kind) => Ok(Placement::Struct(*kind)),
                    Ty::Tensor { shape, dtype } => {
                        let (rows, width) =
                            rect(shape).map_err(|why| Error::Unrectangled { value, why })?;
                        let elem = elem_bytes(*dtype).ok_or(Error::Unrectangled {
                            value,
                            why: Unrectangled::PackedElement,
                        })?;
                        Ok(Placement::Arena {
                            offset: 0,
                            bytes: rows
                                .max(budgets)
                                .checked_mul(width)
                                .and_then(|bytes| bytes.checked_mul(elem))
                                .ok_or(Error::Unrectangled {
                                    value,
                                    why: Unrectangled::Oversize,
                                })?,
                            rows,
                            width,
                            dtype: *dtype,
                        })
                    }
                },
            }
        })
        .collect()
}

fn rect(shape: &[Dim]) -> Result<(RowExpr, u64), Unrectangled> {
    let rows = shape
        .first()
        .copied()
        .map_or(RowExpr::Const(1), RowExpr::of);
    let mut width = 1u64;
    for dim in shape.iter().skip(1) {
        match dim {
            Dim::Const(n) => width = width.checked_mul(*n).ok_or(Unrectangled::Oversize)?,
            Dim::Tokens
            | Dim::TokensTimes(_)
            | Dim::Lanes
            | Dim::LanesPlus(_)
            | Dim::Patches
            | Dim::Images
            | Dim::ImagesPlus(_)
            | Dim::Voxels
            | Dim::VoxelsTimes(_)
            | Dim::Clips
            | Dim::ClipsPlus(_)
            | Dim::Readouts => {
                return Err(Unrectangled::SymbolicWidth);
            }
        }
    }
    Ok((rows, width))
}

#[must_use]
pub fn elem_bytes(dtype: Dtype) -> Option<u64> {
    match dtype {
        Dtype::Bf16 | Dtype::F16 | Dtype::I16 | Dtype::U16 => Some(2),
        Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::I64 | Dtype::U64 => Some(8),
        Dtype::U8 | Dtype::I8 | Dtype::E4m3 | Dtype::E5m2 | Dtype::E8m0 | Dtype::Bool => Some(1),
        Dtype::E2m1
        | Dtype::Mxfp4
        | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::Nvfp4
        | Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k
        | Dtype::E4m3row
        | Dtype::E4m3tile128
        | Dtype::U2g128 => None,
    }
}

fn fold_in_place(trace: &Trace, placements: &mut [Placement]) -> Result<(), Error> {
    let mut pairs: Vec<(ValueId, ValueId)> = Vec::new();
    for node in &trace.nodes {
        pairs.clear();
        node.op.aliases(&mut pairs);
        for (out, overwritten) in &pairs {
            share(placements, Share::InPlace, *overwritten, *out)?;
        }
    }
    Ok(())
}

fn fold_merges(trace: &Trace, placements: &mut [Placement]) -> Result<(), Error> {
    for (id, decl) in trace.values.iter().enumerate() {
        let Def::Merge(arms) = &decl.def else {
            continue;
        };
        let merge = ValueId(id as u32);
        for (arm, _) in arms {
            share(placements, Share::MergeArm, merge, *arm)?;
        }
    }
    Ok(())
}

fn share(
    placements: &mut [Placement],
    kind: Share,
    holds: ValueId,
    shares: ValueId,
) -> Result<(), Error> {
    let (h, s) = (root(placements, holds), root(placements, shares));
    if h == s {
        return Ok(());
    }
    let (Some(a), Some(b)) = (placements.get(h.0 as usize), placements.get(s.0 as usize)) else {
        return Err(Error::AliasOutside { holds, shares });
    };
    if !a.is_arena() || !b.is_arena() {
        return match kind {
            Share::InPlace => Err(Error::AliasOutside { holds, shares }),
            Share::MergeArm => Ok(()),
        };
    }
    let same = match (a, b) {
        (
            Placement::Arena {
                bytes: ab,
                rows: ar,
                width: aw,
                dtype: ad,
                ..
            },
            Placement::Arena {
                bytes: bb,
                rows: br,
                width: bw,
                dtype: bd,
                ..
            },
        ) => (ab, ar, aw, ad) == (bb, br, bw, bd),
        _ => false,
    };
    if !same {
        return Err(Error::Mismatch {
            kind,
            holds: h,
            shares: s,
        });
    }
    placements[s.0 as usize] = Placement::Alias(h);
    Ok(())
}

fn root(placements: &[Placement], mut value: ValueId) -> ValueId {
    for _ in 0..=placements.len() {
        match placements.get(value.0 as usize) {
            Some(Placement::Alias(to)) => value = *to,
            _ => return value,
        }
    }
    panic!("a cycle of aliases through v{}", value.0)
}

fn flatten(placements: &mut [Placement]) {
    for id in 0..placements.len() {
        if let Placement::Alias(to) = placements[id] {
            placements[id] = Placement::Alias(root(placements, to));
        }
    }
}

fn outlive_the_region(trace: &Trace, conc: &Concurrency, spans: &mut [Option<Span>]) {
    let end = trace.nodes.len() as u32;
    let mut region_end: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for at in 0..end {
        let slot = region_end.entry(conc.region_of(at)).or_insert(at);
        *slot = (*slot).max(at);
    }
    let mut passed: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        if crate::region::is_router(node) {
            passed.insert(conc.region_of(at as u32) + 1);
        }
    }
    for span in spans.iter_mut().flatten() {
        if span.last >= end {
            continue;
        }
        let (born, read) = (conc.region_of(span.first), conc.region_of(span.last));
        if born != read {
            if let Some(&last) = region_end.get(&read) {
                span.last = span.last.max(last);
            }
        } else if passed.contains(&born)
            && let Some(&last) = region_end.get(&born)
        {
            span.last = span.last.max(last);
        }
    }
}

fn lives(
    trace: &Trace,
    placements: &[Placement],
    classes: &ClassTable,
) -> (Vec<Option<Span>>, Vec<ClassSet>) {
    let end = trace.nodes.len() as u32;
    let everywhere = ClassSet::of(0..classes.classes.len());
    let mut spans: Vec<Option<Span>> = vec![None; placements.len()];
    let mut live_in: Vec<ClassSet> = vec![ClassSet::default(); placements.len()];
    let mut touched: Vec<ValueId> = Vec::new();

    for (at, node) in trace.nodes.iter().enumerate() {
        let at = at as u32;
        let mask = &classes.node_mask[at as usize];
        touched.clear();
        node.op.inputs(&mut touched);
        node.op.outputs(&mut touched);
        for value in &touched {
            touch(placements, &mut spans, &mut live_in, *value, at, mask);
        }
    }

    for export in EXPORTS {
        for seam in trace.seams.iter().filter(|s| s.seam == export.seam) {
            for value in &seam.values {
                let root = root(placements, *value);
                if !placements
                    .get(root.0 as usize)
                    .is_some_and(Placement::is_arena)
                {
                    continue;
                }
                spans[root.0 as usize]
                    .get_or_insert(Span {
                        first: 0,
                        last: end,
                    })
                    .last = end;
                if export.read_by == Readers::EveryClass {
                    widen(&mut live_in[root.0 as usize], &everywhere);
                }
            }
        }
    }

    for (id, decl) in trace.values.iter().enumerate() {
        let Def::Merge(arms) = &decl.def else {
            continue;
        };
        let fed = arms.iter().any(|(arm, _)| {
            matches!(
                trace.values.get(arm.0 as usize).map(|decl| &decl.def),
                Some(Def::Input(_))
            )
        });
        if !fed {
            continue;
        }
        let root = root(placements, ValueId(id as u32));
        if !placements
            .get(root.0 as usize)
            .is_some_and(Placement::is_arena)
        {
            continue;
        }
        spans[root.0 as usize]
            .get_or_insert(Span {
                first: 0,
                last: end,
            })
            .first = 0;
    }

    for (id, slot) in placements.iter().enumerate() {
        if slot.is_arena() {
            spans[id].get_or_insert(Span {
                first: 0,
                last: end,
            });
            if live_in[id].is_empty() {
                widen(&mut live_in[id], &everywhere);
            }
        }
    }
    (spans, live_in)
}

fn touch(
    placements: &[Placement],
    spans: &mut [Option<Span>],
    live_in: &mut [ClassSet],
    value: ValueId,
    at: u32,
    mask: &ClassSet,
) {
    let root = root(placements, value);
    if !placements
        .get(root.0 as usize)
        .is_some_and(Placement::is_arena)
    {
        return;
    }
    match &mut spans[root.0 as usize] {
        Some(span) => {
            span.first = span.first.min(at);
            span.last = span.last.max(at);
        }
        None => {
            spans[root.0 as usize] = Some(Span {
                first: at,
                last: at,
            })
        }
    }
    widen(&mut live_in[root.0 as usize], mask);
}

fn widen(set: &mut ClassSet, by: &ClassSet) {
    for class in by.iter() {
        set.insert(class);
    }
}

const BIND_ALIGN: u64 = 256;

fn align(bytes: u64) -> u64 {
    bytes.div_ceil(BIND_ALIGN) * BIND_ALIGN
}

fn place(
    placements: &mut [Placement],
    spans: &[Option<Span>],
    live_in: &[ClassSet],
    conc: &Concurrency,
    mode: Columns,
) -> u64 {
    let mut order: Vec<(u64, Span, usize)> = placements
        .iter()
        .enumerate()
        .filter(|(_, slot)| slot.is_arena())
        .map(|(id, slot)| {
            let span = spans[id].expect("`spans` answers every arena slot");
            (align(slot.bytes()), span, id)
        })
        .collect();
    order.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then(a.1.first.cmp(&b.1.first))
            .then(a.2.cmp(&b.2))
    });

    let columns = gather(placements, live_in, &order, mode);

    let mut placed: Vec<(u64, u64, Span)> = Vec::with_capacity(columns.len());
    let mut blockers: Vec<(u64, u64)> = Vec::new();
    let mut bytes = 0u64;
    for column in &columns {
        blockers.clear();
        blockers.extend(
            placed
                .iter()
                .filter(|(_, _, live)| conc.overlap(*live, column.span))
                .map(|(at, size, _)| (*at, at + size)),
        );
        blockers.sort_unstable();
        let mut at = 0u64;
        for (from, to) in &blockers {
            if *from >= at + column.size {
                break;
            }
            at = at.max(*to);
        }
        for id in &column.members {
            let Placement::Arena { offset, .. } = &mut placements[*id] else {
                unreachable!("only arena placements are gathered into columns")
            };
            *offset = at;
        }
        placed.push((at, column.size, column.span));
        bytes = bytes.max(at + column.size);
    }
    bytes
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Columns {
    PerValue,
    Shared,
}

struct Column {
    size: u64,
    span: Span,
    classes: ClassSet,
    pitch: Option<(RowExpr, u64, Dtype)>,
    members: Vec<usize>,
}

fn gather(
    placements: &[Placement],
    live_in: &[ClassSet],
    order: &[(u64, Span, usize)],
    mode: Columns,
) -> Vec<Column> {
    let mut columns: Vec<Column> = Vec::with_capacity(order.len());
    for (size, span, id) in order {
        let pitch = pitch_of(&placements[*id]);
        let classes = &live_in[*id];
        let joined = match (mode, pitch) {
            (Columns::Shared, Some(pitch)) if !classes.is_empty() => {
                columns.iter_mut().find(|column| {
                    column.pitch == Some(pitch)
                        && column.size == *size
                        && touching(column.span, *span)
                        && column.classes.disjoint(classes)
                })
            }
            _ => None,
        };
        match joined {
            Some(column) => {
                column.span.first = column.span.first.min(span.first);
                column.span.last = column.span.last.max(span.last);
                widen(&mut column.classes, classes);
                column.members.push(*id);
            }
            None => columns.push(Column {
                size: *size,
                span: *span,
                classes: classes.clone(),
                pitch,
                members: vec![*id],
            }),
        }
    }
    columns
}

fn pitch_of(slot: &Placement) -> Option<(RowExpr, u64, Dtype)> {
    match slot {
        Placement::Arena {
            rows, width, dtype, ..
        } if rows.cut_per_class() => Some((*rows, *width, *dtype)),
        _ => None,
    }
}

fn touching(a: Span, b: Span) -> bool {
    a.first <= b.last && b.first <= a.last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, fact};
    use crate::region;
    use model_ir::{Guard, resolve_classes};

    fn budget() -> Budgets {
        Budgets::of(crate::Budget::new(4, 16))
    }

    #[test]
    fn arena_every_case() {
        the_row_algebra_sizes_every_dim_at_its_ceiling();
        a_symbolic_width_is_refused_and_names_the_value();
        a_merge_of_two_sizes_is_refused_rather_than_carved();
    }

    fn the_row_algebra_sizes_every_dim_at_its_ceiling() {
        let b = Budgets::of(crate::Budget::new(4, 16));
        assert_eq!(RowExpr::of(Dim::Tokens).max(&b), 16);
        assert_eq!(RowExpr::of(Dim::TokensTimes(3)).max(&b), 48);
        assert_eq!(RowExpr::of(Dim::Lanes).max(&b), 4);
        assert_eq!(RowExpr::of(Dim::LanesPlus(1)).max(&b), 5);
        assert_eq!(RowExpr::of(Dim::Const(7)).max(&b), 7);
        assert_eq!(RowExpr::of(Dim::Tokens).at(FireRows::text_only(3, 2)), 3);
        assert_eq!(
            RowExpr::of(Dim::LanesPlus(1)).at(FireRows::text_only(3, 2)),
            3
        );
    }

    fn a_symbolic_width_is_refused_and_names_the_value() {
        let mut b = Build::new();
        let x = b.input(8);
        let node = b.trace.nodes.len() as u32;
        let y = b.value(
            Def::Op(node),
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Lanes],
                dtype: Dtype::Bf16,
            },
        );
        b.trace.nodes.push(model_ir::Node {
            op: model_ir::ops::Elementwise::RmsnormNoScale {
                x,
                head_dim: 1,
                eps: 1e-6,
                y,
            }
            .into(),
            guard: Guard::Always,
            layer: None,
        });
        b.out(y);

        let classes = resolve_classes(&b.trace).expect("resolves");
        let regions = region::coalesce(&b.trace, &classes).expect("the fixture coalesces");
        let conc = Concurrency::sequential(&regions, b.trace.nodes.len());
        assert_eq!(
            carve(&b.trace, &budget(), &classes, &conc),
            Err(Error::Unrectangled {
                value: y,
                why: Unrectangled::SymbolicWidth,
            }),
        );
    }

    fn a_merge_of_two_sizes_is_refused_rather_than_carved() {
        let mut b = Build::new();
        let x = b.input(8);
        let d = b.op(x, 8, fact(0));
        let p = b.op(x, 4, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 8);
        b.out(o);

        let classes = resolve_classes(&b.trace).expect("resolves");
        let regions = region::coalesce(&b.trace, &classes).expect("the fixture coalesces");
        let conc = Concurrency::sequential(&regions, b.trace.nodes.len());
        assert_eq!(
            carve(&b.trace, &budget(), &classes, &conc),
            Err(Error::Mismatch {
                kind: Share::MergeArm,
                holds: o,
                shares: p,
            }),
        );
    }
}
