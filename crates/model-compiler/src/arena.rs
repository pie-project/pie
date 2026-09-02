//! The carve: where every value lives, and how few bytes that takes.
//!
//! A value's rectangle spans the node that writes it to the last node that
//! reads it ([`Span`]); two values whose spans never coincide share bytes,
//! since a wrongly shared slab computes silently rather than faulting
//! ([`ArenaMap::clashes`] guards this). Offsets are static, decided once;
//! row counts stay symbolic ([`RowExpr`]), each column reserved at its
//! maximum under [`Budgets`]. Class-disjoint values may share one column as
//! row windows ([`ArenaMap::co_tenants`] and [`Concurrency`]).

use model_ir::{
    ClassSet, ClassTable, Def, Dim, Dtype, Operands, RowAxis, Trace, RuntimeInput, StructKind, Ty,
    ValueId,
};

use crate::compiled::Region;
use crate::budget::Budgets;
use crate::error::{Error, Share, Unrectangled};

/// Who reads an export after the graph has run — whose rows stay spoken for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readers {
    /// Every class — e.g. the sampler, nobody's co-tenant.
    EveryClass,
    /// Only the classes that ran the export, e.g. a draft readout.
    ItsOwnClasses,
}

/// One declared export: a seam a reader touches after the graph has run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Export {
    seam: &'static str,
    read_by: Readers,
}

/// The export set: names `model_dsl::seam` states for values materializing
/// outside the graph. Republished as [`crate::EXPORT_SEAMS`].
const EXPORTS: [Export; 4] = [
    // Trunk logits, into the runtime's sampler.
    Export {
        seam: "out",
        read_by: Readers::EveryClass,
    },
    // Draft head logits, into the same sampler.
    Export {
        seam: "mtp",
        read_by: Readers::ItsOwnClasses,
    },
    // Per-query normalizing mass, one column per layer.
    Export {
        seam: "attn.scores",
        read_by: Readers::ItsOwnClasses,
    },
    // The draft head's token ids, `[rows, depth]` i32, into `mtp_drafts`.
    Export {
        seam: "mtp.drafts",
        read_by: Readers::ItsOwnClasses,
    },
];

/// The export seam names, in the order [`EXPORTS`] states them.
pub const EXPORT_SEAMS: [&str; 4] =
    [EXPORTS[0].seam, EXPORTS[1].seam, EXPORTS[2].seam, EXPORTS[3].seam];

/// How many rows a value has, in the terms the carve evaluates rather than
/// `model_ir::Dim`'s serializable form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowExpr {
    /// A fixed count, whatever the fire's size.
    Const(u64),
    /// One row per token this fire carries.
    Tokens,
    /// `tokens * k` — one row per route.
    TokensTimes(u32),
    /// One row per lane — the geometry vectors.
    Lanes,
    /// `lanes + k` — indptr-shaped.
    LanesPlus(u32),
    /// One row per patch this fire carries — the second row axis
    /// (`RowAxis::Patches`), never a co-tenant of a token rectangle.
    Patches,
    /// One row per image — [`Lanes`](RowExpr::Lanes) for the patch axis.
    Images,
    /// `images + k` — the patch axis's indptr shape.
    ImagesPlus(u32),
}

impl RowExpr {
    /// The `Dim` this reads, as a row count.
    #[must_use]
    pub fn of(dim: Dim) -> RowExpr {
        match dim {
            Dim::Const(n) => RowExpr::Const(n),
            Dim::Tokens => RowExpr::Tokens,
            Dim::TokensTimes(k) => RowExpr::TokensTimes(k),
            Dim::Lanes => RowExpr::Lanes,
            Dim::LanesPlus(k) => RowExpr::LanesPlus(k),
            Dim::Patches => RowExpr::Patches,
            Dim::Images => RowExpr::Images,
            Dim::ImagesPlus(k) => RowExpr::ImagesPlus(k),
        }
    }

    /// Which row space this is measured in. `None` for [`Const`](RowExpr::Const).
    #[must_use]
    pub fn axis(self) -> Option<RowAxis> {
        match self {
            RowExpr::Const(_) => None,
            RowExpr::Tokens | RowExpr::TokensTimes(_) | RowExpr::Lanes | RowExpr::LanesPlus(_) => {
                Some(RowAxis::Tokens)
            }
            RowExpr::Patches | RowExpr::Images | RowExpr::ImagesPlus(_) => Some(RowAxis::Patches),
        }
    }

    /// The most rows this can be under `budgets`.
    #[must_use]
    pub fn max(self, budgets: &Budgets) -> u64 {
        self.at(FireRows::ceilings(budgets))
    }

    /// Does a windowed reader see only its own classes' rows? `Const` and
    /// `*Plus` variants reach past their own class, so cannot share.
    #[must_use]
    pub fn cut_per_class(self) -> bool {
        match self {
            // `Patches` cuts for `Tokens`' reason, one axis over.
            RowExpr::Tokens
            | RowExpr::TokensTimes(_)
            | RowExpr::Lanes
            | RowExpr::Patches
            | RowExpr::Images => true,
            RowExpr::Const(_) | RowExpr::LanesPlus(_) | RowExpr::ImagesPlus(_) => false,
        }
    }

    /// The rows this expression has in a fire of these counts.
    #[must_use]
    pub fn at(self, fire: FireRows) -> u64 {
        match self {
            RowExpr::Const(n) => n,
            RowExpr::Tokens => fire.tokens,
            RowExpr::TokensTimes(k) => fire.tokens.saturating_mul(u64::from(k)),
            RowExpr::Lanes => fire.lanes,
            RowExpr::LanesPlus(k) => fire.lanes.saturating_add(u64::from(k)),
            RowExpr::Patches => fire.patches,
            RowExpr::Images => fire.images,
            RowExpr::ImagesPlus(k) => fire.images.saturating_add(u64::from(k)),
        }
    }
}

/// What one fire is, as far as a rectangle's length is concerned: a row
/// count and a lane count per row axis; a lane may carry no image or three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FireRows {
    /// Token rows this fire carries.
    pub tokens: u64,
    /// Lanes — requests — this fire carries.
    pub lanes: u64,
    /// Patch rows this fire carries.
    pub patches: u64,
    /// Images this fire carries.
    pub images: u64,
}

impl FireRows {
    /// A fire on the token axis alone: no images, no patch rows.
    #[must_use]
    pub fn text_only(tokens: u64, lanes: u64) -> FireRows {
        FireRows {
            tokens,
            lanes,
            patches: 0,
            images: 0,
        }
    }

    /// The largest fire these budgets admit, on both axes at once.
    #[must_use]
    pub fn ceilings(budgets: &Budgets) -> FireRows {
        FireRows {
            tokens: u64::from(budgets.tokens.max_tokens),
            lanes: u64::from(budgets.tokens.max_lanes),
            patches: u64::from(budgets.max_patches()),
            images: u64::from(budgets.max_images()),
        }
    }
}

/// The nodes one value must survive, in program order, both ends
/// inclusive. `last == trace.nodes.len()` is the instant after the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// The node that writes the value.
    pub first: u32,
    /// The last node that reads it, or `trace.nodes.len()` for a seam.
    pub last: u32,
}

/// Which regions the engine may have in flight at once. A no-op hook
/// today: with one stream, [`overlap`](Concurrency::overlap) is the
/// interval test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Concurrency {
    /// Node index -> region index, plus one entry for the post-graph instant.
    region_of: Vec<u32>,
    /// Region pairs that may be in flight together, `(lo, hi)`, sorted.
    pairs: Vec<(u32, u32)>,
    /// [`pairs`](Concurrency::pairs) again, as node ranges rather than
    /// region indices, so the overlap walk stays linear in pairs found.
    paired: Vec<((u32, u32), (u32, u32))>,
}

impl Concurrency {
    /// One stream: the regions run one after another.
    #[must_use]
    pub fn sequential(regions: &[Region], nodes: usize) -> Concurrency {
        Concurrency {
            region_of: map_regions(regions, nodes),
            pairs: Vec::new(),
            paired: Vec::new(),
        }
    }

    /// The same map, plus the pairs that may overlap, normalized and sorted.
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

    /// The region a node belongs to, or the sentinel for after the graph.
    #[must_use]
    pub fn region_of(&self, node: u32) -> u32 {
        self.region_of
            .get(node as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// The region pairs that may be in flight together.
    #[must_use]
    pub fn pairs(&self) -> &[(u32, u32)] {
        &self.pairs
    }

    /// May these two values be live at the same instant?
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

/// Does a region's half-open node run meet a span's inclusive one?
fn meets((start, end): (u32, u32), span: Span) -> bool {
    start <= span.last && span.first < end
}

fn map_regions(regions: &[Region], nodes: usize) -> Vec<u32> {
    // `regions.len()` is the sentinel for the instant after the graph.
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

/// Where one value lives at the fire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Bound by the engine each fire. Outside the arena: the engine owns it.
    Runtime(RuntimeInput),
    /// A loader-resident weight table, by index into `Trace::params`.
    Param(u32),
    /// A cache space, by index into `Trace::caches`; outlives the fire.
    Cache(u32),
    /// A rectangle of the arena: `rows x width` elements of `dtype` at a
    /// static `offset`. Non-overlapping values share the same bytes on purpose.
    Arena {
        /// Byte offset into the arena. Static, and the same in every bucket.
        offset: u64,
        /// Bytes the largest admissible fire touches, before alignment.
        bytes: u64,
        /// How many rows, symbolically.
        rows: RowExpr,
        /// Elements per row.
        width: u64,
        /// The element.
        dtype: Dtype,
    },
    /// This value IS another value's rectangle: a `Def::Merge`'s arms, or
    /// `Operands::aliases`' in-place overwrite.
    Alias(ValueId),
    /// A host-owned plan object: opaque, sized at plan-build time.
    Struct(StructKind),
}

impl Placement {
    /// The bytes a kernel touches — zero outside the arena.
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

/// The byte range one value occupies in one fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    /// Byte offset into the arena.
    pub offset: u64,
    /// Bytes THIS fire touches, which is at most the slot's reservation.
    pub bytes: u64,
}

/// The arena, carved: one slot per plan value, and what it all adds up to.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaMap {
    /// Indexed by [`ValueId`], parallel to `Trace::values`.
    pub placements: Vec<Placement>,
    /// Each value's life, indexed the same way; `None` for no rectangle.
    pub spans: Vec<Option<Span>>,
    /// The classes each value's life falls in. Empty co-tenants with nobody.
    pub live_in: Vec<ClassSet>,
    /// Bytes one fire needs, at the budget's ceiling — the busiest instant,
    /// not the sum.
    pub bytes: u64,
}

impl ArenaMap {
    /// The rectangle a value finally names, following aliases.
    #[must_use]
    pub fn root(&self, value: ValueId) -> ValueId {
        root(&self.placements, value)
    }

    /// Where a value sits in a fire, or `None` if not in the arena.
    /// `fire`'s counts are not optional; see [`FireRows::text_only`].
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

    /// Are these two values two row windows of one column? Requires one
    /// offset/pitch, a cut window, and disjoint known classes.
    #[must_use]
    pub fn co_tenants(&self, a: ValueId, b: ValueId) -> bool {
        let (Some(x), Some(y)) = (self.placements.get(a.0 as usize), self.placements.get(b.0 as usize))
        else {
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

    /// Every pair of values that may be live at one instant yet share a
    /// byte they both write. Empty on every map [`carve`] builds.
    #[must_use]
    pub fn clashes(&self, conc: &Concurrency) -> Vec<(ValueId, ValueId)> {
        self.clashes_blind(conc)
            .into_iter()
            .filter(|(a, b)| !self.co_tenants(*a, *b))
            .collect()
    }

    /// The v1 predicate: overlapping lives and bytes, no shared columns.
    /// Kept as the oracle for maps with no co-tenants.
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

    /// The busiest instant: the most bytes live at any one node, aligned,
    /// each column counted once — the floor no layout can beat.
    #[must_use]
    pub fn live_bound(&self) -> u64 {
        let mut live = self.live();
        // By offset, so a column's members stand together and one pass
        // charges each column once.
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

    /// `(value, span, offset, bytes)` for every rectangle of the arena.
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

/// Cut the arena: a slot per value, aliases folded, spans measured, offsets placed.
///
/// # Errors
///
/// [`Error::Unrectangled`] for a declared type the row algebra cannot size,
/// [`Error::Mismatch`] for two values the IR says share one column and
/// declares at different sizes.
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

    // Both walks run, and the smaller wins: placement is greedy and not
    // monotone in general, so the tightened walk is not guaranteed to win.
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

/// One slot per value, sized at the budget's ceiling, offset still zero.
fn rectangles(trace: &Trace, budgets: &Budgets) -> Result<Vec<Placement>, Error> {
    trace.values
        .iter()
        .enumerate()
        .map(|(id, decl)| {
            let value = ValueId(id as u32);
            match &decl.def {
                Def::Input(which) => Ok(Placement::Runtime(*which)),
                Def::Weight(i) => Ok(Placement::Param(*i)),
                Def::Cache(i) => Ok(Placement::Cache(*i)),
                // A merge is given the column; its arms are folded onto it
                // below since they write disjoint row windows of one buffer.
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

/// A shape, read as `rows x width`: the leading dim is the row count, the
/// rest the width.
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
            | Dim::ImagesPlus(_) => {
                return Err(Unrectangled::SymbolicWidth);
            }
        }
    }
    Ok((rows, width))
}

/// The bytes one element occupies, or `None` for a packed storage plane
/// (e.g. `Mxfp4`), not a rectangle of this arena.
#[must_use]
pub fn elem_bytes(dtype: Dtype) -> Option<u64> {
    match dtype {
        Dtype::Bf16 | Dtype::F16 | Dtype::I16 | Dtype::U16 => Some(2),
        Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::I64 | Dtype::U64 => Some(8),
        Dtype::U8
        | Dtype::I8
        | Dtype::E4m3
        | Dtype::E5m2
        | Dtype::E8m0
        | Dtype::Bool => Some(1),
        // `U8g64` is byte-wide but still packed: it names a weight bank's
        // affine codes, meaningful only beside its group's scale and offset.
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

/// Fold every in-place result onto the operand it overwrites; not folding
/// would mint a second rectangle nothing ever writes.
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

/// Fold every merge's arms onto the merged column: each arm's guard admits
/// a disjoint set of lanes, so the merge costs zero instructions.
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

/// Put `shares` into `holds`'s column, or refuse if declared at two sizes.
fn share(placements: &mut [Placement], kind: Share, holds: ValueId, shares: ValueId) -> Result<(), Error> {
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

/// The rectangle an alias finally names.
fn root(placements: &[Placement], mut value: ValueId) -> ValueId {
    for _ in 0..=placements.len() {
        match placements.get(value.0 as usize) {
            Some(Placement::Alias(to)) => value = *to,
            _ => return value,
        }
    }
    panic!("a cycle of aliases through v{}", value.0)
}

/// Collapse alias chains to one hop, so every reader sees the rectangle.
fn flatten(placements: &mut [Placement]) {
    for id in 0..placements.len() {
        if let Placement::Alias(to) = placements[id] {
            placements[id] = Placement::Alias(root(placements, to));
        }
    }
}

/// A region the engine walks in pieces (a streamed load's routed segment,
/// `FireDescriptor::run_caps`) runs its nodes once per piece, so a value
/// written BEFORE the region and read inside it is read by every piece: its
/// life runs to the region's last node, not to its last reader's. Otherwise
/// a later node of the same region may be seated on its bytes and the next
/// piece reads that node's output as, say, a routing vector. Stated for
/// every region, since the carve does not know which are capped; the cost
/// is a few in-region aliasings forgone.
fn outlive_the_region(trace: &Trace, conc: &Concurrency, spans: &mut [Option<Span>]) {
    let end = trace.nodes.len() as u32;
    let mut region_end: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for at in 0..end {
        let slot = region_end.entry(conc.region_of(at)).or_insert(at);
        *slot = (*slot).max(at);
    }
    // The regions a streamed load may walk in expert-major PASSES: the one
    // after each router's. A pass recomputes the region's routed values for
    // one group of experts and leaves the other groups' rows as earlier
    // passes wrote them, so every value born inside such a region must hold
    // its bytes to the region's end — a later node's output may not be
    // seated on a dead-looking `packed` whose rows the next pass still owes.
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
        } else if passed.contains(&born) {
            if let Some(&last) = region_end.get(&born) {
                span.last = span.last.max(last);
            }
        }
    }
}

/// Each value's life: the node indices it spans, and the classes it spans
/// them in. Spans are the plan's, not a class's — a mixed fire runs every
/// class's nodes at one wall clock, so a per-class span would call a value
/// dead while a kernel is still reading it.
fn lives(trace: &Trace, placements: &[Placement], classes: &ClassTable) -> (Vec<Option<Span>>, Vec<ClassSet>) {
    let end = trace.nodes.len() as u32;
    // The reader with no class: the runtime, over every row.
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

    // Every declared export is read after the last node, by a reader no
    // node occupies. `everywhere` only for `"out"`, else a per-layer
    // capture column would price at the whole fire's height.
    for export in EXPORTS {
        for seam in trace.seams.iter().filter(|s| s.seam == export.seam) {
            for value in &seam.values {
                let root = root(placements, *value);
                if !placements.get(root.0 as usize).is_some_and(Placement::is_arena) {
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

    // Every rectangle ends up with a span.
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

/// Extend a value's life to cover instant `at` in `mask`'s classes, through any alias.
fn touch(
    placements: &[Placement],
    spans: &mut [Option<Span>],
    live_in: &mut [ClassSet],
    value: ValueId,
    at: u32,
    mask: &ClassSet,
) {
    // A merge is its column's life, not its own: reading a merged value
    // reads the column its arms wrote.
    let root = root(placements, value);
    if !placements.get(root.0 as usize).is_some_and(Placement::is_arena) {
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

/// Add every class of `by` to `set`.
fn widen(set: &mut ClassSet, by: &ClassSet) {
    for class in by.iter() {
        set.insert(class);
    }
}

/// The alignment every reservation and offset is rounded up to — the
/// largest offset alignment a conformant device may demand.
const BIND_ALIGN: u64 = 256;

/// A reservation, rounded up to [`BIND_ALIGN`].
fn align(bytes: u64) -> u64 {
    bytes.div_ceil(BIND_ALIGN) * BIND_ALIGN
}

/// Give every rectangle an offset, sharing bytes between values never live
/// together, and answer what the arena adds up to. Greedy-by-size and
/// deterministic (bytes descending, then birth node, then value id).
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
        // The lowest offset no column live beside this one already holds.
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

/// Whether [`place`] may put two values in one column, or must give each its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Columns {
    /// The v1 walk: one column per rectangle, classes unread.
    PerValue,
    /// The tightening: class-disjoint rectangles of one pitch may share.
    Shared,
}

/// One reservation of the arena and the values that live in it — one
/// member under [`Columns::PerValue`], class-disjoint row windows under
/// [`Columns::Shared`].
struct Column {
    /// The rounded reservation — one number, since members agree about pitch.
    size: u64,
    /// The union of the members' lives, an interval the members cover.
    span: Span,
    /// Every class any member is touched in.
    classes: ClassSet,
    /// `(rows, width, dtype)` a member must match; `None` admits no second member.
    pitch: Option<(RowExpr, u64, Dtype)>,
    /// Placement indices, in the order they joined.
    members: Vec<usize>,
}

/// Gather the placement order into columns.
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
        // Every clause is load-bearing: matching pitch/size, a cut window,
        // disjoint known classes, and spans that already touch.
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

/// What a value must agree about to share a column, or `None` if a window
/// hands it over whole ([`RowExpr::cut_per_class`]).
fn pitch_of(slot: &Placement) -> Option<(RowExpr, u64, Dtype)> {
    match slot {
        Placement::Arena {
            rows, width, dtype, ..
        } if rows.cut_per_class() => Some((*rows, *width, *dtype)),
        _ => None,
    }
}

/// Do these two lives share a node index? The interval test alone, not
/// [`Concurrency::overlap`]: the join needs the union of two spans to be one
/// interval the two of them cover.
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
    fn the_row_algebra_sizes_every_dim_at_its_ceiling() {
        let b = Budgets::of(crate::Budget::new(4, 16));
        assert_eq!(RowExpr::of(Dim::Tokens).max(&b), 16);
        assert_eq!(RowExpr::of(Dim::TokensTimes(3)).max(&b), 48);
        assert_eq!(RowExpr::of(Dim::Lanes).max(&b), 4);
        assert_eq!(RowExpr::of(Dim::LanesPlus(1)).max(&b), 5);
        assert_eq!(RowExpr::of(Dim::Const(7)).max(&b), 7);
        // …and a fire smaller than the ceiling uses fewer.
        assert_eq!(RowExpr::of(Dim::Tokens).at(FireRows::text_only(3, 2)), 3);
        assert_eq!(RowExpr::of(Dim::LanesPlus(1)).at(FireRows::text_only(3, 2)), 3);
    }

    #[test]
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

    #[test]
    fn a_merge_of_two_sizes_is_refused_rather_than_carved() {
        // The IR says the arms write one column. Two sizes means one of them
        // writes past the other's end, and the carve cannot paper over it.
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
