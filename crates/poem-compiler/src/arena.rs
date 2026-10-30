use poem_ir::{
    ClassSet, ClassTable, Def, Dim, Dtype, Operands, RowAxis, RuntimeInput, StructKind, Trace, Ty,
    ValueId,
};

use crate::budget::Budgets;
use crate::compiled::Region;
use crate::error::{Error, Share, Unrectangled};

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
    /// The fire's widest voxel rows, scaled by `num/den`.
    Voxels(u32, u32),
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
            Dim::Voxels => RowExpr::Voxels(1, 1),
            Dim::VoxelsScaled(n, d) => RowExpr::Voxels(n, d),
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
            RowExpr::Voxels(..) | RowExpr::Clips | RowExpr::ClipsPlus(_) => Some(RowAxis::Voxels),
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
            | RowExpr::Voxels(..)
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
            RowExpr::Voxels(n, d) => fire
                .voxels
                .saturating_mul(u64::from(n))
                .div_ceil(u64::from(d)),
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

/// A budget's logit rows a lane where nothing sizes them to a trace.
pub const READOUTS_PER_LANE: u32 = 16;

/// The fewest logit rows a lane reads out under [`readouts_per_lane`]: room
/// beside a batch for a lone request reading a few rows of its own.
pub const READOUTS_FLOOR: u32 = 4;

/// The logit rows a lane may read out in one fire under `trace`: a block
/// drafter's verified block (and its anchor where proposals start at row 0),
/// or the tokens an output of ids drafts past each readout row and the row
/// they follow, never under [`READOUTS_FLOOR`]. On a vocab-wide head each row
/// a lane is the vocab's width again in every readout plane (two 1.9 GiB
/// planes at 16 rows a lane on Qwen3.6-27B's 248k vocab, 256 lanes).
#[must_use]
pub fn readouts_per_lane(trace: &Trace) -> u32 {
    // A block drafter's verify reads its block, and the anchor before it
    // where its proposals start at row 0.
    let block = trace.drafter.as_ref().map_or(0, |drafter| {
        drafter.rows + u32::from(drafter.proposals_from == 0)
    });
    let drafted = trace
        .outputs
        .iter()
        .filter_map(
            |output| match &trace.values.get(output.value.0 as usize)?.ty {
                Ty::Tensor {
                    shape,
                    dtype: Dtype::I32,
                } if shape.first() == Some(&Dim::Readouts) => {
                    shape.iter().skip(1).try_fold(1u32, |width, dim| match dim {
                        Dim::Const(n) => width.checked_mul(u32::try_from(*n).ok()?),
                        _ => None,
                    })
                }
                _ => None,
            },
        )
        .max()
        .map_or(0, |drafts| drafts.saturating_add(1));
    READOUTS_FLOOR.max(block).max(drafted)
}

/// A readout plane whose row is at most this wide (a velocity or hidden
/// export) is carved for every token; a wider one (vocab-wide logits) for
/// the budget's `readouts_per_lane` rows a lane.
pub const NARROW_READOUT_ROW_BYTES: u64 = 64 << 10;

#[must_use]
pub fn readouts_ceiling(budget: &crate::Budget) -> u64 {
    (u64::from(budget.max_lanes) * u64::from(budget.readouts_per_lane.max(1)))
        .min(u64::from(budget.max_tokens))
}

#[must_use]
pub fn readout_rows(width: u64, elem: u64, budget: &crate::Budget) -> u64 {
    if width.saturating_mul(elem) <= NARROW_READOUT_ROW_BYTES {
        u64::from(budget.max_tokens)
    } else {
        readouts_ceiling(budget)
    }
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
            readouts: readouts_ceiling(&budgets.tokens),
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

    /// Which fork pairs' two sides `span` meets, as bitsets: two spans
    /// overlap when they touch, or when one meets a pair's side and the other
    /// meets the other. Computed once per span, so a placement's every
    /// overlap test is a few words of AND rather than a walk of the pairs.
    #[must_use]
    pub fn reach(&self, span: Span) -> Reach {
        let words = self.paired.len().div_ceil(64);
        let (mut one, mut two) = (vec![0u64; words], vec![0u64; words]);
        for (i, &(a, b)) in self.paired.iter().enumerate() {
            let bit = 1u64 << (i % 64);
            if meets(a, span) {
                one[i / 64] |= bit;
            }
            if meets(b, span) {
                two[i / 64] |= bit;
            }
        }
        Reach { span, one, two }
    }
}

/// A span with the fork-pair sides it meets ([`Concurrency::reach`]).
#[derive(Debug, Clone)]
pub struct Reach {
    span: Span,
    one: Vec<u64>,
    two: Vec<u64>,
}

impl Reach {
    #[must_use]
    pub fn overlaps(&self, other: &Reach) -> bool {
        let across = |a: &[u64], b: &[u64]| a.iter().zip(b).any(|(x, y)| x & y != 0);
        touching(self.span, other.span)
            || across(&self.one, &other.two)
            || across(&self.two, &other.one)
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

/// An arena slot: the value it holds, the nodes it is live over, and where.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveSlot {
    pub value: ValueId,
    pub span: Span,
    pub offset: u64,
    pub bytes: u64,
}

/// The node where the most arena bytes are live at once, and the slots live
/// there, largest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Peak {
    pub at: u32,
    pub slots: Vec<LiveSlot>,
}

impl ArenaMap {
    #[must_use]
    pub fn root(&self, value: ValueId) -> ValueId {
        root(&self.placements, value)
    }

    /// Where the `attn.scores` planes start: every other slot lies under it.
    #[must_use]
    pub fn scores_from(&self, trace: &Trace) -> u64 {
        trace
            .outputs
            .iter()
            .filter(|output| output.name == SCORES)
            .filter_map(
                |output| match self.placements.get(self.root(output.value).0 as usize) {
                    Some(Placement::Arena { offset, .. }) => Some(*offset),
                    _ => None,
                },
            )
            .min()
            .unwrap_or(self.bytes)
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
        let live: Vec<_> = self
            .live()
            .into_iter()
            .map(|slot| (slot, conc.reach(slot.span)))
            .collect();
        let mut found = Vec::new();
        for (i, (a, a_reach)) in live.iter().enumerate() {
            for (b, b_reach) in &live[i + 1..] {
                if a.offset < b.offset + b.bytes
                    && b.offset < a.offset + a.bytes
                    && a_reach.overlaps(b_reach)
                {
                    found.push((a.value, b.value));
                }
            }
        }
        found
    }

    #[must_use]
    pub fn live_bound(&self) -> u64 {
        let mut live = self.live();
        live.sort_by_key(|slot| slot.offset);
        let end = live.iter().map(|slot| slot.span.last).max().unwrap_or(0);
        let mut most = 0u64;
        for at in 0..=end {
            let mut total = 0u64;
            let mut column: Option<(u64, u64)> = None;
            for LiveSlot {
                span,
                offset,
                bytes,
                ..
            } in &live
            {
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

    /// The rows one fire may read out through `value`'s plane.
    #[must_use]
    pub fn readout_ceiling_of(&self, value: ValueId) -> Option<u64> {
        match self.placements.get(self.root(value).0 as usize)? {
            Placement::Arena {
                bytes,
                rows: RowExpr::Readouts,
                width,
                dtype,
                ..
            } => Some(bytes / width.saturating_mul(elem_bytes(*dtype)?).max(1)),
            _ => None,
        }
    }

    #[must_use]
    pub fn prefix_for(&self, readouts: u64) -> u64 {
        self.placements
            .iter()
            .map(|slot| match slot {
                Placement::Arena {
                    offset,
                    bytes,
                    rows: RowExpr::Readouts,
                    width,
                    dtype,
                } => {
                    let used = readouts
                        .saturating_mul(*width)
                        .saturating_mul(elem_bytes(*dtype).unwrap_or(0));
                    offset + used.min(*bytes)
                }
                Placement::Arena { offset, bytes, .. } => offset + bytes,
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    #[must_use]
    pub fn peak(&self) -> Peak {
        let live = self.live();
        let end = live.iter().map(|slot| slot.span.last).max().unwrap_or(0);
        let mut best = Peak {
            at: 0,
            slots: Vec::new(),
        };
        let mut best_total = 0;
        for at in 0..=end {
            let here: Vec<LiveSlot> = live
                .iter()
                .filter(|slot| slot.span.first <= at && at <= slot.span.last)
                .copied()
                .collect();
            let total: u64 = here.iter().map(|slot| align(slot.bytes)).sum();
            if total > best_total {
                best_total = total;
                best = Peak { at, slots: here };
            }
        }
        best.slots
            .sort_by_key(|slot| core::cmp::Reverse(slot.bytes));
        best
    }

    fn live(&self) -> Vec<LiveSlot> {
        self.placements
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| match slot {
                Placement::Arena { offset, bytes, .. } => Some(LiveSlot {
                    value: ValueId(id as u32),
                    span: self.spans[id].expect("the carve spans every arena slot"),
                    offset: *offset,
                    bytes: *bytes,
                }),
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
    let mut placements = rectangles(trace, budgets, classes)?;
    fold_in_place(trace, &mut placements)?;
    fold_merges(trace, &mut placements)?;
    flatten(&mut placements);
    let (mut spans, live_in) = lives(trace, &placements, classes);
    outlive_the_region(trace, conc, &mut spans);

    let stacks = stacks(trace, &placements);
    let mut conservative = placements.clone();
    let blind = place(
        &mut conservative,
        &spans,
        &live_in,
        &stacks,
        conc,
        Columns::PerValue,
    );
    let shared = place(
        &mut placements,
        &spans,
        &live_in,
        &stacks,
        conc,
        Columns::Shared,
    );
    let (mut placements, bytes) = if shared < blind {
        (placements, shared)
    } else {
        (conservative, blind)
    };
    let bytes = lift_scores(trace, &mut placements, bytes);

    Ok(ArenaMap {
        placements,
        spans,
        live_in,
        bytes,
    })
}

fn rectangles(
    trace: &Trace,
    budgets: &Budgets,
    classes: &ClassTable,
) -> Result<Vec<Placement>, Error> {
    let carved = voxel_carve(trace, classes);
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
                        let rows = match (rows, carved[id]) {
                            (RowExpr::Voxels(..), Some((n, d))) => RowExpr::Voxels(n, d),
                            _ => rows,
                        };
                        let elem = elem_bytes(*dtype).ok_or(Error::Unrectangled {
                            value,
                            why: Unrectangled::PackedElement,
                        })?;
                        let carved = match rows {
                            RowExpr::Readouts => readout_rows(width, elem, &budgets.tokens),
                            _ => rows.max(budgets),
                        };
                        Ok(Placement::Arena {
                            offset: 0,
                            bytes: carved
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
            | Dim::VoxelsScaled(..)
            | Dim::Clips
            | Dim::ClipsPlus(_)
            | Dim::Readouts => {
                return Err(Unrectangled::SymbolicWidth);
            }
        }
    }
    Ok((rows, width))
}

/// The classes each value is made in: its node's, or its arms' for a merge.
fn made_in(trace: &Trace, classes: &ClassTable) -> Vec<ClassSet> {
    let mut sets: Vec<ClassSet> = Vec::with_capacity(trace.values.len());
    for decl in &trace.values {
        let set = match &decl.def {
            Def::Op(node) => classes
                .node_mask
                .get(*node as usize)
                .cloned()
                .unwrap_or_default(),
            Def::Merge(arms) => ClassSet::of(
                arms.iter()
                    .filter_map(|(arm, _)| sets.get(arm.0 as usize))
                    .flat_map(ClassSet::iter)
                    .collect::<Vec<_>>(),
            ),
            _ => ClassSet::default(),
        };
        sets.push(set);
    }
    sets
}

fn scale_of(decl: &poem_ir::ValueDecl) -> Option<(u32, u32)> {
    match &decl.ty {
        Ty::Tensor { shape, .. } => shape.first()?.voxel_scale(),
        Ty::Struct(_) => None,
    }
}

fn wider(a: (u32, u32), b: (u32, u32)) -> bool {
    u64::from(a.0) * u64::from(b.1) > u64::from(b.0) * u64::from(a.1)
}

fn reduced(n: u64, d: u64) -> (u32, u32) {
    let g = {
        let (mut a, mut b) = (n, d);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a.max(1)
    };
    (
        u32::try_from(n / g).unwrap_or(u32::MAX),
        u32::try_from(d / g).unwrap_or(u32::MAX),
    )
}

/// Each class's growth: its widest voxel value over the voxel input it reads.
#[must_use]
pub fn voxel_growth(trace: &Trace, classes: &ClassTable) -> Vec<(u32, u32)> {
    let sets = made_in(trace, classes);
    let mut growth = vec![(1, 1); classes.classes.len()];
    for (decl, set) in trace.values.iter().zip(&sets) {
        let Some(scale) = scale_of(decl) else {
            continue;
        };
        for class in set.iter() {
            if let Some(g) = growth.get_mut(class)
                && wider(scale, *g)
            {
                *g = scale;
            }
        }
    }
    growth
}

/// Each voxel value's rows over the fire's widest voxel rows: its scale over
/// the growth of the class it is made in, the widest over its classes.
fn voxel_carve(trace: &Trace, classes: &ClassTable) -> Vec<Option<(u32, u32)>> {
    let growth = voxel_growth(trace, classes);
    let sets = made_in(trace, classes);
    trace
        .values
        .iter()
        .zip(&sets)
        .map(|(decl, set)| {
            let (n, d) = scale_of(decl)?;
            set.iter()
                .filter_map(|class| growth.get(class))
                .map(|&(gn, gd)| {
                    reduced(u64::from(n) * u64::from(gd), u64::from(d) * u64::from(gn))
                })
                .reduce(|a, b| if wider(b, a) { b } else { a })
                .or(Some((n, d)))
        })
        .collect()
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
        | Dtype::E4m3g128
        | Dtype::E4m3tensor
        | Dtype::Fp8b128
        | Dtype::Fp8b32
        | Dtype::Mxfp8
        | Dtype::Ptq1_0
        // C2b packed KV codec: a block-quantized format with an inline scale
        // plane, so it has no per-element byte size — callers that must size a
        // packed row do so through `Dtype::row_bytes()`, not this function.
        | Dtype::KvU4
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

    for output in &trace.outputs {
        for value in core::iter::once(output.value).chain(output.grid) {
            let root = root(placements, value);
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

/// Each layer's `attn.scores` plane on top of every other slot, so a load no
/// program reads scores from backs the arena only up to [`ArenaMap::scores_from`].
fn lift_scores(trace: &Trace, placements: &mut [Placement], bytes: u64) -> u64 {
    let mut lifted: Vec<usize> = Vec::new();
    for output in trace.outputs.iter().filter(|output| output.name == SCORES) {
        let id = root(placements, output.value).0 as usize;
        if placements.get(id).is_some_and(Placement::is_arena) && !lifted.contains(&id) {
            lifted.push(id);
        }
    }
    if lifted.is_empty() {
        return bytes;
    }
    let mut at = placements
        .iter()
        .enumerate()
        .filter(|(id, _)| !lifted.contains(id))
        .filter_map(|(_, slot)| match slot {
            Placement::Arena { offset, bytes, .. } => Some(offset + align(*bytes)),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    for id in lifted {
        if let Placement::Arena { offset, bytes, .. } = &mut placements[id] {
            *offset = at;
            at += align(*bytes);
        }
    }
    at
}

const SCORES: &str = "attn.scores";

fn align(bytes: u64) -> u64 {
    bytes.div_ceil(BIND_ALIGN) * BIND_ALIGN
}

/// The slots of each output written at several layers, in the order it was
/// written: one plane, a layer after another. An output whose layers differ
/// in shape, or share a slot, is left to its slots.
fn stacks(trace: &Trace, placements: &[Placement]) -> Vec<Vec<usize>> {
    let mut names: Vec<&str> = Vec::new();
    for output in &trace.outputs {
        if output.layer.is_some() && !names.contains(&output.name.as_str()) {
            names.push(&output.name);
        }
    }
    names
        .into_iter()
        .filter_map(|name| {
            let slots: Vec<usize> = trace
                .outputs
                .iter()
                .filter(|output| output.name == name && output.layer.is_some())
                .map(|output| root(placements, output.value).0 as usize)
                .collect();
            let first = placements.get(*slots.first()?)?;
            let alike = slots.iter().enumerate().all(|(at, slot)| {
                placements.get(*slot).is_some_and(|placed| {
                    placed.is_arena() && pitch_and_bytes(placed) == pitch_and_bytes(first)
                }) && !slots[..at].contains(slot)
            });
            (slots.len() > 1 && alike).then_some(slots)
        })
        .collect()
}

fn pitch_and_bytes(slot: &Placement) -> Option<(RowExpr, u64, Dtype, u64)> {
    match slot {
        Placement::Arena {
            rows,
            width,
            dtype,
            bytes,
            ..
        } => Some((*rows, *width, *dtype, *bytes)),
        _ => None,
    }
}

fn place(
    placements: &mut [Placement],
    spans: &[Option<Span>],
    live_in: &[ClassSet],
    stacks: &[Vec<usize>],
    conc: &Concurrency,
    mode: Columns,
) -> u64 {
    let stacked = |id: usize| stacks.iter().any(|stack| stack.contains(&id));
    let mut order: Vec<(u64, Span, usize)> = placements
        .iter()
        .enumerate()
        .filter(|(id, slot)| slot.is_arena() && !stacked(*id))
        .map(|(id, slot)| {
            let span = spans[id].expect("`spans` answers every arena slot");
            (align(slot.bytes()), span, id)
        })
        .collect();
    let reads_out = |id: usize| {
        matches!(
            placements[id],
            Placement::Arena {
                rows: RowExpr::Readouts,
                ..
            }
        )
    };
    order.sort_by(|a, b| {
        reads_out(a.2)
            .cmp(&reads_out(b.2))
            .then(b.0.cmp(&a.0))
            .then(a.1.first.cmp(&b.1.first))
            .then(a.2.cmp(&b.2))
    });

    let mut columns: Vec<Column> = stacks
        .iter()
        .map(|stack| {
            let slot = align(placements[stack[0]].bytes());
            let mut column = Column {
                size: slot * stack.len() as u64,
                span: spans[stack[0]].expect("`spans` answers every arena slot"),
                classes: ClassSet::default(),
                pitch: None,
                members: Vec::with_capacity(stack.len()),
            };
            for (layer, id) in stack.iter().enumerate() {
                let span = spans[*id].expect("`spans` answers every arena slot");
                column.span.first = column.span.first.min(span.first);
                column.span.last = column.span.last.max(span.last);
                widen(&mut column.classes, &live_in[*id]);
                column.members.push((*id, slot * layer as u64));
            }
            column
        })
        .collect();
    columns.extend(gather(placements, live_in, &order, mode));

    let mut placed: Vec<(u64, u64, Reach)> = Vec::with_capacity(columns.len());
    let mut blockers: Vec<(u64, u64)> = Vec::new();
    let mut bytes = 0u64;
    for column in &columns {
        let reach = conc.reach(column.span);
        blockers.clear();
        blockers.extend(
            placed
                .iter()
                .filter(|(_, _, live)| live.overlaps(&reach))
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
        for (id, within) in &column.members {
            let Placement::Arena { offset, .. } = &mut placements[*id] else {
                unreachable!("only arena placements are gathered into columns")
            };
            *offset = at + within;
        }
        placed.push((at, column.size, reach));
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
    /// Each slot, and where it sits inside the column.
    members: Vec<(usize, u64)>,
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
                column.members.push((*id, 0));
            }
            None => columns.push(Column {
                size: *size,
                span: *span,
                classes: classes.clone(),
                pitch,
                members: vec![(*id, 0)],
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
    use poem_ir::{Guard, resolve_classes};
    use poem_ir::{NormAxis, NormWeight};

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
        b.trace.nodes.push(poem_ir::Node {
            op: poem_ir::ops::Elemwise::Rmsnorm {
                x,
                axis: NormAxis::Head(1),
                eps: 1e-6,
                y,
                weight: NormWeight::None,
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
        let p = b.op(x, 4, !fact(0));
        let o = b.merge(&[(d, fact(0)), (p, !fact(0))], 8);
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
