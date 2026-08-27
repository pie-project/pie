//! P7 — the carve. Where every value lives, and how few bytes that takes.
//!
//! A value's rectangle is needed from the node that writes it to the last node
//! that reads it ([`Span`]), and two values whose spans never coincide are
//! given the SAME BYTES. That is the difference between an arena that is the
//! sum of everything a plan ever mints and one that is the plan's busiest
//! instant: on the rewrite's identical walk, gemma's row went from 21.8 MiB to
//! 1 MiB and qwen's from 2.45 MiB to 487 KiB, both landing exactly on the
//! bound. [`ArenaMap::clashes`] is the invariant's guard, because a slab
//! shared with the wrong value does not fault — it computes.
//!
//! # What is static and what is symbolic
//!
//! **Offsets are static bytes.** A slot's offset is decided here, once, and
//! every shape bucket reads the same number. **Row counts stay symbolic**
//! ([`RowExpr`]): a value knows it is `Dim::Tokens` rows or `Dim::Lanes + 1`
//! rows, and what a particular fire does with that is
//! [`ArenaMap::window`]'s arithmetic, not a recapture.
//!
//! The two meet at the budget. A column is RESERVED at its maximum under
//! [`Budgets`] — `Tokens` at `max_tokens`, `TokensTimes(k)` at
//! `max_tokens * k`, `Lanes` at `max_lanes`, `LanesPlus(k)` at
//! `max_lanes + k` — and a smaller fire writes its first rows and leaves the
//! tail alone. The arena buffer has to be that big anyway, since it is
//! resident and shared by every bucket, so reserving at the ceiling costs
//! nothing that a per-bucket layout would have saved.
//!
//! THIS IS WHERE THE REWRITE'S ARITHMETIC WAS DROPPED. There, a slot's byte
//! base was `offset * fire_rows` — a per-row pitch scaled at the fire, which
//! works only when every value's row count is the SAME symbol. This IR has
//! two: `Tokens` and `Lanes` are different numbers, and an indptr's
//! `LanesPlus(1)` is not `Tokens`-shaped at any scale. Scaling everything by
//! the token count would over-reserve the lane-shaped vectors and still get
//! `LanesPlus` wrong in an all-decode bucket where lanes equal tokens. Static
//! bytes handle the mixture exactly, and the packing is otherwise the same
//! walk.
//!
//! # Classes do not buy a shorter life. They buy a shared COLUMN.
//!
//! The tempting tightening is to measure a life PER CLASS — a value written
//! and read inside the decode window would then not be held live across the
//! prefill nodes standing between them, since no single class runs both — and
//! for this arena it is WRONG. A fire is MIXED. `WindowTable` gives every
//! class present its own interval of rows and the walk runs every region over
//! its own mask's interval, so a fire's decode nodes and its prefill nodes are
//! alive at the same wall clock. Two values whose classes never coincide are
//! nevertheless both resident, and bytes handed to both are bytes two live
//! kernels write.
//!
//! What class-disjointness does buy is exact, and it is the mechanism §0's φ
//! already uses. `Run::cut` slices a value's column at the ASKING NODE'S
//! window — a `Dim::Tokens` column by the mask's rows, a `Dim::Lanes` one by
//! its lanes — so a node that runs only in class A only ever touches class
//! A's rows of the column. Give two class-disjoint values the SAME OFFSET at
//! the SAME PITCH and they are two row windows of one column, disjoint by the
//! seriation, exactly the way a merge's arms are. That is
//! [`ArenaMap::co_tenants`], and it is the whole of the per-class tightening:
//! not a shorter span — a shared column.
//!
//! **The pitch has to match and the offset has to be equal.** A row window is
//! `offset + row * pitch`, so two columns at different offsets, or at one
//! offset with two pitches, lay class A's rows of the one across class B's
//! rows of the other. And a cut has to exist at all: `Dim::Const` is handed
//! over whole and `Dim::LanesPlus(k)` takes `lanes + k` entries from the
//! window's first lane, which reaches into the class beside it. Neither is a
//! per-class window, so neither may co-tenant ([`RowExpr::cut_per_class`]).
//!
//! **A column is only shared where the conservative carve was already stuck.**
//! Two values join one column only if their spans OVERLAP — which is to say
//! only where the v1 rule had to give them different bytes anyway. That keeps
//! a column's span an interval its own members cover, so the column blocks
//! exactly what its members blocked and costs one reservation instead of
//! several; sharing can never lengthen a conflict. And because a greedy
//! placement is not monotone in the number of blocks it is handed, [`carve`]
//! runs BOTH walks and keeps the smaller, so a tightened arena is never a byte
//! larger than the conservative one — and is bit-identical to it wherever the
//! tightening buys nothing.
//!
//! # What is deliberately conservative in v1
//!
//! **The busiest instant is a step, not an interval of wall clock.** Two
//! values may share bytes only if no instant holds both, and today "instant"
//! means "node index", which is exact because the walk is a straight line
//! (one stream). [`Concurrency`] is the seam where P6 widens that: when
//! regions can run beside each other, two values in concurrently-live regions
//! overlap even if their node ranges do not.

use model_ir::{
    ClassSet, Classes, Def, Dim, Dtype, Operands, Plan, RuntimeInput, StructKind, Ty, ValueId,
};

use crate::baked::Region;
use crate::budget::Budgets;
use crate::refusal::{Refusal, Share, Unrectangled};

/// Who reads an export after the graph has run — which is the only thing the
/// delivery tail below needs to know about one.
///
/// **THE TAIL IS ABOUT THE READER, NOT ABOUT THE WRITER.** Both variants hold
/// the column open past the last node; they differ in whose ROWS of it are
/// spoken for, and that is a fact about who comes to collect rather than about
/// which classes happened to write it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readers {
    /// Every class. The sampler takes the logits of every lane in the fire,
    /// whichever behavior the lane is, so this column's rows are all of them
    /// and it is nobody's co-tenant — even where the node that wrote it stood
    /// in one window.
    EveryClass,
    /// Only the classes that ran the export. A draft readout is collected for
    /// the lanes that drafted and for no others; a lane that asked for no
    /// draft has none, and claiming its rows would price the column at the
    /// whole fire's height for a reader that never looks at them.
    ItsOwnClasses,
}

/// One declared export: a seam whose value a reader touches after the graph
/// has run, and who that reader is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Export {
    seam: &'static str,
    read_by: Readers,
}

/// **THE EXPORT SET** — the names `model_dsl::seam` states for values that
/// materialize outside the graph (design §9), and the whole of what this
/// module knows about the authoring vocabulary.
///
/// STRINGS, BECAUSE THE VOCABULARY IS NOT IN THE IR. `model-dsl` owns the seam
/// names — "the new IR keeps only the `Seam` rows a plan carries" — and this
/// crate does not depend on the authoring surface. The coupling is this table,
/// republished as [`crate::EXPORT_SEAMS`] so that a shell resolving the same
/// names reads them from here instead of keeping a second copy of the literals
/// (`driver_cuda::serve` did, until this landed), and
/// `tests/every_sku_carves_an_arena.rs` is what notices if they ever stop
/// matching.
///
/// **A SET RATHER THAN ONE NAME, AND THAT IS palo C3b/C4b's WHOLE SEAT.** Until
/// this wave the tail belonged to `"out"` alone and the draft column was safe
/// only because the model text stated it LAST — nothing runs after the last
/// node, so nothing can be carved on top of it. That is a true argument about
/// one statement order and not a property of the artifact; a text that stated
/// its second readout anywhere else got a column the following GEMM was free to
/// re-use, which is a wrong answer that computes (build log 25). The order is
/// still right and the model text still keeps it; what changed is that it is no
/// longer load-bearing.
///
/// The seams NOT here — `attn.q`, `attn.out`, `attn.qv`, `recurrent`, `in` —
/// are trace-time attach points, planted once per layer, and nothing reads them
/// after the fire. Pinning them would hold two or three activations per layer
/// live across the whole stack, which is sixty layers of an arena that exists
/// to be the busiest instant.
const EXPORTS: [Export; 3] = [
    // `model_dsl::seam::OUT` — the trunk's logits, into the engine's sampler.
    Export {
        seam: "out",
        read_by: Readers::EveryClass,
    },
    // `model_dsl::seam::MTP` — the draft head's logits over the draft window,
    // into the same sampler through `driver::program`'s `MtpLogits` intrinsic
    // at `mtp_draft_row` (palo C3b).
    Export {
        seam: "mtp",
        read_by: Readers::ItsOwnClasses,
    },
    // `model_dsl::seam::SCORES` — the attention's per-query normalizing mass,
    // one column per attention layer, into `LaneReadout::scores` (palo C4b).
    Export {
        seam: "attn.scores",
        read_by: Readers::ItsOwnClasses,
    },
];

/// The export seam names, in the order [`EXPORTS`] states them — for a shell
/// that has to find the same values in a `Plan` it was handed.
///
/// Published because the alternative is what was here before: the same literals
/// spelled a second time in `driver-cuda`, with a comment in each place saying
/// the other one exists.
pub const EXPORT_SEAMS: [&str; 3] = [EXPORTS[0].seam, EXPORTS[1].seam, EXPORTS[2].seam];

/// How many rows a value has, in the only terms a plan can state.
///
/// COMPILER-INTERNAL, AND NOT AN EXTENSION OF `Dim` (design open item 3).
/// `model_ir::Dim` is what a model text writes; this is what the carve
/// evaluates, and the two want different things — `Dim` wants to be small and
/// serializable, this wants to be asked "how many, at most" and "how many, in
/// this bucket". Every `Dim` maps in; nothing maps back out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowExpr {
    /// A fixed count, whatever the fire's size.
    Const(u64),
    /// One row per token this fire carries.
    Tokens,
    /// `tokens * k` — one row per ROUTE, `k` the router's `top_k`. The MoE
    /// fan-out, and the reason a row count is an expression rather than a flag.
    TokensTimes(u32),
    /// One row per lane — the geometry vectors.
    Lanes,
    /// `lanes + k` — indptr-shaped, the bounds vector that needs its closing
    /// entry.
    LanesPlus(u32),
}

impl RowExpr {
    /// The `Dim` this reads, as a row count. Total: every `Dim` is a row
    /// count when it stands first in a shape.
    #[must_use]
    pub fn of(dim: Dim) -> RowExpr {
        match dim {
            Dim::Const(n) => RowExpr::Const(n),
            Dim::Tokens => RowExpr::Tokens,
            Dim::TokensTimes(k) => RowExpr::TokensTimes(k),
            Dim::Lanes => RowExpr::Lanes,
            Dim::LanesPlus(k) => RowExpr::LanesPlus(k),
        }
    }

    /// The most rows this can be under `budgets` — what the column reserves.
    #[must_use]
    pub fn max(self, budgets: &Budgets) -> u64 {
        self.at(u64::from(budgets.max_tokens), u64::from(budgets.max_lanes))
    }

    /// Does a windowed reader see only its OWN classes' rows of this column?
    ///
    /// THE PRECONDITION OF A SHARED COLUMN, AND IT IS THE SHELL'S RULE READ
    /// BACK. `driver_cuda::run::Run::cut` slices a rectangle at the asking
    /// node's window by the value's leading `Dim`: `Tokens` at
    /// `(row_offset, rows)`, `TokensTimes(k)` at `(row_offset * k, rows * k)`,
    /// `Lanes` at `(lane_offset, lanes)` — three cuts that land inside the
    /// mask's own interval and nowhere else. The other two do not:
    /// `Const(n)` is handed over WHOLE, because a bias is not fire-aligned,
    /// and `LanesPlus(k)` takes `lanes + k` entries from the window's first
    /// lane, so the closing entry of one class's indptr sits on the first
    /// entry of the class beside it. A rectangle those two shapes name is a
    /// column of exactly one value.
    #[must_use]
    pub fn cut_per_class(self) -> bool {
        match self {
            RowExpr::Tokens | RowExpr::TokensTimes(_) | RowExpr::Lanes => true,
            RowExpr::Const(_) | RowExpr::LanesPlus(_) => false,
        }
    }

    /// The rows a fire of `tokens` rows over `lanes` requests actually has.
    #[must_use]
    pub fn at(self, tokens: u64, lanes: u64) -> u64 {
        match self {
            RowExpr::Const(n) => n,
            RowExpr::Tokens => tokens,
            RowExpr::TokensTimes(k) => tokens.saturating_mul(u64::from(k)),
            RowExpr::Lanes => lanes,
            RowExpr::LanesPlus(k) => lanes.saturating_add(u64::from(k)),
        }
    }
}

/// The nodes one value must survive, in program order.
///
/// `first` is the node that writes it and `last` the last node that reads it,
/// both INCLUSIVE — so a value with no reader still spans the one node that
/// minted it, because the launch writes through the pointer either way.
///
/// INCLUSIVE IS THE WHOLE SAFETY ARGUMENT, and it is what makes the in-place
/// question answer itself. A node at `j` reads its inputs and writes its
/// outputs in the same launch: the input's span ends at `j`, the output's
/// begins at `j`, and the two touch — so the carve can never hand one the
/// other's bytes by accident. A plan that WANTS the aliasing says so the way
/// the IR already can, through `Operands::aliases`, which allocates nothing
/// and shares by construction.
///
/// `last == plan.nodes.len()` is the instant AFTER the graph: the `"out"`
/// seam's value is read there, by a reader no node occupies. Giving that time
/// an index rather than a special case is what keeps the rule one sentence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// The node that writes the value.
    pub first: u32,
    /// The last node that reads it, or `plan.nodes.len()` for a seam export.
    pub last: u32,
}

/// Which regions the driver may have in flight at once — P6's answer, and the
/// argument the carve's overlap predicate takes.
///
/// A NO-OP HOOK IN v1, DELIBERATELY SHAPED. With one stream the walk is a
/// straight line and [`Concurrency::sequential`] holds no pairs, so
/// [`overlap`](Concurrency::overlap) is exactly the interval test the rewrite
/// used. The reason the argument exists now rather than later: when P6 forks a
/// dep DAG into streams, two values whose node ranges are disjoint may still
/// be live at the same wall-clock instant, and a carve that shared their bytes
/// would produce a race that computes. Threading the relation through from the
/// start means P6 is a pass that fills a table, not a pass that also has to
/// find every place the old assumption was baked in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Concurrency {
    /// Node index -> region index, with one extra entry for the post-graph
    /// instant, which belongs to no region.
    region_of: Vec<u32>,
    /// Region pairs that may be in flight together, `(lo, hi)` with `lo < hi`,
    /// sorted. Empty means "one stream": nothing runs beside anything.
    pairs: Vec<(u32, u32)>,
}

impl Concurrency {
    /// One stream: the regions run one after another, in the order they stand.
    #[must_use]
    pub fn sequential(regions: &[Region], nodes: usize) -> Concurrency {
        Concurrency {
            region_of: map_regions(regions, nodes),
            pairs: Vec::new(),
        }
    }

    /// The same map, plus the pairs P6 says may overlap. Pairs are normalized
    /// and sorted, so a caller may hand them in either order and any order.
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
        Concurrency {
            region_of: map_regions(regions, nodes),
            pairs,
        }
    }

    /// The region a node belongs to, or the sentinel past the last region for
    /// the instant after the graph.
    #[must_use]
    pub fn region_of(&self, node: u32) -> u32 {
        self.region_of
            .get(node as usize)
            .copied()
            .unwrap_or(u32::MAX)
    }

    /// The region pairs that may be in flight together. Empty under one
    /// stream.
    #[must_use]
    pub fn pairs(&self) -> &[(u32, u32)] {
        &self.pairs
    }

    /// May these two values be live at the same instant?
    ///
    /// The interval test first, because that is the answer for almost every
    /// pair and the only answer there is under one stream. Then, only if P6
    /// has said some regions run together, the wider question: does a's life
    /// touch a region that runs beside one b's life touches?
    #[must_use]
    pub fn overlap(&self, a: Span, b: Span) -> bool {
        if a.first <= b.last && b.first <= a.last {
            return true;
        }
        if self.pairs.is_empty() {
            return false;
        }
        for ra in self.spanned(a) {
            for rb in self.spanned(b) {
                let key = if ra < rb { (ra, rb) } else { (rb, ra) };
                if ra != rb && self.pairs.binary_search(&key).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// The regions a span passes through, ascending and deduplicated. Regions
    /// are runs of adjacent nodes, so this is a contiguous range.
    fn spanned(&self, span: Span) -> impl Iterator<Item = u32> + '_ {
        let lo = self.region_of(span.first);
        let hi = self.region_of(span.last);
        // An index the map does not hold answers `u32::MAX`, and a range that
        // started there would be a walk over four billion regions. It cannot
        // happen — spans run over `0..=nodes.len()` and the map is that long —
        // and answering nothing is the reading that stays cheap if it ever did.
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        let empty = hi == u32::MAX;
        (lo..=hi).take(if empty { 0 } else { usize::MAX })
    }
}

fn map_regions(regions: &[Region], nodes: usize) -> Vec<u32> {
    // The sentinel is `regions.len()` rather than a magic number: it is the
    // region index the instant after the graph would have, if that instant
    // were a region.
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
pub enum Slot {
    /// Bound by the driver each fire — tokens, positions, a mask, a geometry
    /// vector. Outside the arena: the driver owns the buffer.
    Runtime(RuntimeInput),
    /// A loader-resident weight table, by index into `Plan::params`. Outside
    /// the arena: it is written once at residency and read forever.
    Param(u32),
    /// A cache space, by index into `Plan::caches`. Outside the arena: it
    /// outlives the fire, which is what makes it a cache.
    Cache(u32),
    /// A rectangle of the arena: `rows x width` elements of `dtype`, at a
    /// static byte `offset`.
    ///
    /// AN OFFSET IS NOT PRIVATE. Values whose lives do not overlap are given
    /// the same bytes ON PURPOSE; what a slot owns is its offset for the nodes
    /// its [`Span`] covers, and nothing outside those nodes may read it. The
    /// one reader past the last node is the `"out"` seam, and the spans hold
    /// it open to fire end for exactly that reason.
    Arena {
        /// Byte offset into the arena. Static, and the same in every bucket.
        offset: u64,
        /// Bytes the largest admissible fire touches — `rows.max(budgets) *
        /// width * dtype`. What a kernel may write, BEFORE the alignment the
        /// carve rounds a reservation up to.
        bytes: u64,
        /// How many rows, symbolically.
        rows: RowExpr,
        /// Elements per row.
        width: u64,
        /// The element.
        dtype: Dtype,
    },
    /// This value IS another value's rectangle.
    ///
    /// TWO RULES PRODUCE THESE, and both are stated by the IR rather than
    /// guessed here. A `Def::Merge` owns one column and its arms write
    /// disjoint row windows of it — phi lowers to zero instructions (design
    /// §0), so the arms alias the merge. And `Operands::aliases` names the
    /// input an op overwrites in place, so the result aliases that input.
    Alias(ValueId),
    /// A host-owned plan object: opaque, sized at plan-build time, outside the
    /// arena. What makes its defining node a `Phase::Prepare`.
    Struct(StructKind),
}

impl Slot {
    /// The bytes a kernel touches — zero for anything that is not a rectangle
    /// of the arena.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            Slot::Arena { bytes, .. } => *bytes,
            Slot::Runtime(_)
            | Slot::Param(_)
            | Slot::Cache(_)
            | Slot::Alias(_)
            | Slot::Struct(_) => 0,
        }
    }

    fn is_arena(&self) -> bool {
        matches!(self, Slot::Arena { .. })
    }
}

/// The byte range one value occupies in one fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    /// Byte offset into the arena — the slot's static offset.
    pub offset: u64,
    /// Bytes THIS fire touches, which is at most the slot's reservation.
    pub bytes: u64,
}

/// The arena, carved: one slot per plan value, and what it all adds up to.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaMap {
    /// Indexed by [`ValueId`], parallel to `Plan::values`.
    pub slots: Vec<Slot>,
    /// Each value's life, indexed the same way. `None` for a value with no
    /// rectangle of its own.
    ///
    /// KEPT RATHER THAN RE-DERIVED. [`clashes`](ArenaMap::clashes) and
    /// [`live_bound`](ArenaMap::live_bound) are both questions about spans,
    /// and re-deriving them means walking the plan again with the same
    /// aliasing rules — which is to say, having a second copy of the rules.
    pub spans: Vec<Option<Span>>,
    /// The classes each value's life falls in, indexed the same way — the
    /// union of `Classes::node_mask` over every node that touches it, through
    /// aliases, and every class for the `"out"` seam, which is read after the
    /// fire by a reader that is not in any one class.
    ///
    /// EMPTY MEANS "NO CLASS RUNS IT", AND THAT IS NOT A LICENCE. A rectangle
    /// only dead nodes touch is one `Classes::dead` already reports and the
    /// compiler is free to drop; while it is kept, the reading that cannot be
    /// wrong is that its classes are unknown, so an empty set co-tenants with
    /// nobody.
    pub live_in: Vec<ClassSet>,
    /// Bytes one fire needs, at the budget's ceiling.
    ///
    /// THE BUSIEST INSTANT, NOT THE SUM. Values whose spans never coincide
    /// share bytes, so this is the most the arena ever holds live at once
    /// (plus what the greedy placement leaves in holes) and not the total of
    /// everything the plan mints. [`live_bound`](ArenaMap::live_bound) is the
    /// floor it is measured against.
    pub bytes: u64,
}

impl ArenaMap {
    /// The rectangle a value finally names, following aliases.
    #[must_use]
    pub fn root(&self, value: ValueId) -> ValueId {
        root(&self.slots, value)
    }

    /// Where a value sits in a fire of `tokens` rows over `lanes` requests, or
    /// `None` if it is not in the arena.
    ///
    /// THE DRIVER'S WHOLE ARITHMETIC, and it is this short because the offset
    /// is static: only the length moves with the bucket.
    #[must_use]
    pub fn window(&self, value: ValueId, tokens: u64, lanes: u64) -> Option<Window> {
        let root = self.root(value);
        match self.slots.get(root.0 as usize)? {
            Slot::Arena {
                offset,
                rows,
                width,
                dtype,
                ..
            } => Some(Window {
                offset: *offset,
                bytes: rows
                    .at(tokens, lanes)
                    .saturating_mul(*width)
                    .saturating_mul(elem_bytes(*dtype).unwrap_or(0)),
            }),
            _ => None,
        }
    }

    /// Are these two values two row windows of ONE column?
    ///
    /// THE REFINED RULE, IN ONE PLACE. Everything the safety argument needs is
    /// here and none of it is a guess:
    ///
    /// - **one offset, one pitch** — same `offset`, `rows`, `width` and
    ///   `dtype`, so row `r` of the one is byte-for-byte row `r` of the other;
    /// - **the window cuts it** ([`RowExpr::cut_per_class`]), so a node only
    ///   ever touches the rows of the classes it runs in;
    /// - **the classes are disjoint and both are known**, so the rows the one
    ///   is touched over and the rows the other is touched over are two
    ///   different intervals of the fire's seriated order.
    ///
    /// Then no instant of any fire, mixed or pure, has a kernel writing one's
    /// bytes while the other's are wanted — not because they are never live
    /// together (they usually are) but because "their bytes" are different
    /// bytes. It is the same argument design §0 makes for a merge's arms,
    /// which is why an arm is not a co-tenant: [`fold_merges`] made the arms
    /// ONE rectangle, and this is what a plan gets when it wants the sharing
    /// without having said `merge`.
    ///
    /// A `Slot::Alias`, a runtime binding, a param, a cache and a struct all
    /// answer `false`: they are not rectangles of this arena, so there is no
    /// column to share.
    #[must_use]
    pub fn co_tenants(&self, a: ValueId, b: ValueId) -> bool {
        let (Some(x), Some(y)) = (self.slots.get(a.0 as usize), self.slots.get(b.0 as usize))
        else {
            return false;
        };
        let (
            Slot::Arena {
                offset: a_at,
                bytes: a_bytes,
                rows: a_rows,
                width: a_width,
                dtype: a_dtype,
            },
            Slot::Arena {
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
        !a_in.is_empty() && !b_in.is_empty() && disjoint(a_in, b_in)
    }

    /// Every pair of values that may be live at one instant and whose
    /// rectangles nevertheless share a byte they both write.
    ///
    /// THE ARENA'S WHOLE INVARIANT, and cheap enough to keep as its guard. A
    /// reused slab does not fault when it is wrong — the addresses stay inside
    /// the buffer and every launch succeeds — so the only thing that catches a
    /// layout mistake is arithmetic, either this or a checkpoint's logits.
    /// Empty on every map [`carve`] builds.
    ///
    /// The bytes compared are what a kernel TOUCHES, not the rounded
    /// reservation: a pair that shared only padding would be a carve this walk
    /// never produces, and reporting it would name a clash no launch can see.
    ///
    /// **[`co_tenants`](ArenaMap::co_tenants) IS THE ONE EXEMPTION, AND IT HAD
    /// TO BE ADDED HERE TOO.** A predicate that only knew about node indices
    /// would report every shared column as a clash and reject the layouts the
    /// tightening exists to produce. [`clashes_blind`](ArenaMap::clashes_blind)
    /// is that older predicate, kept as the oracle the refinement is pinned
    /// against.
    #[must_use]
    pub fn clashes(&self, conc: &Concurrency) -> Vec<(ValueId, ValueId)> {
        self.clashes_blind(conc)
            .into_iter()
            .filter(|(a, b)| !self.co_tenants(*a, *b))
            .collect()
    }

    /// The v1 predicate: overlapping lives and overlapping bytes, and no
    /// notion of a column two classes share.
    ///
    /// KEPT AS THE ORACLE, NOT AS A SECOND ANSWER. Wherever a map has no
    /// co-tenants — every catalog row today — this and
    /// [`clashes`](ArenaMap::clashes) return exactly the same list, and that
    /// agreement is what the tests assert: the refinement is a strict
    /// weakening of a predicate that was already correct, and the only pairs
    /// it drops are the pairs [`co_tenants`](ArenaMap::co_tenants) can name a
    /// reason for.
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

    /// The busiest instant: the most bytes this plan holds live at any one
    /// node, each rounded to the alignment an offset has to sit on, and each
    /// COLUMN counted once.
    ///
    /// THE FLOOR NO LAYOUT CAN BEAT, and the number the reuse is measured
    /// against — [`bytes`](ArenaMap::bytes) sitting ON it is the whole claim.
    /// A greedy placement can exceed it by leaving holes; the rewrite's
    /// identical walk sat on the bound for every catalog row, which is why
    /// nothing more elaborate has earned its way in.
    ///
    /// COLUMNS, NOT RECTANGLES, IS WHAT MOVED. Two rectangles live at one node
    /// and sharing an offset are co-tenants — the seriation gives them
    /// different rows of one reservation — or else they are a clash
    /// [`clashes`](ArenaMap::clashes) is standing right there to report. So
    /// the honest floor charges one reservation per distinct offset, and a map
    /// with no shared column answers exactly the number the v1 walk did.
    #[must_use]
    pub fn live_bound(&self) -> u64 {
        let mut live = self.live();
        // By offset, so that a column's members stand together and one pass
        // charges each column once. The sum does not care about the order and
        // the maximum over nodes does not either, so this reorder is free.
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
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| match slot {
                Slot::Arena { offset, bytes, .. } => Some((
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

/// Cut the arena: a slot per value, aliases folded, spans measured, offsets
/// placed.
///
/// # Errors
///
/// [`Refusal::Unrectangled`] for a declared type the row algebra cannot size,
/// [`Refusal::Mismatch`] for two values the IR says share one column and
/// declares at different sizes.
pub(crate) fn carve(
    plan: &Plan,
    budgets: &Budgets,
    classes: &Classes,
    conc: &Concurrency,
) -> Result<ArenaMap, Refusal> {
    let mut slots = rectangles(plan, budgets)?;
    fold_in_place(plan, &mut slots)?;
    fold_merges(plan, &mut slots)?;
    flatten(&mut slots);
    let (spans, live_in) = lives(plan, &slots, classes);

    // BOTH WALKS, AND THE SMALLER WINS. The tightened walk can only ever merge
    // blocks the conservative one had to keep apart, so on the argument it
    // cannot lose — but the placement it feeds is greedy, and a greedy
    // first-fit handed a different multiset of blocks is not monotone in
    // general. Running the conservative walk too and keeping its answer on a
    // tie is what turns "should not regress" into "cannot", and it is what
    // makes a plan the tightening buys nothing on lay out BYTE-IDENTICALLY to
    // the way it laid out before this pass existed.
    let mut conservative = slots.clone();
    let blind = place(&mut conservative, &spans, &live_in, conc, Columns::PerValue);
    let shared = place(&mut slots, &spans, &live_in, conc, Columns::Shared);
    let (slots, bytes) = if shared < blind {
        (slots, shared)
    } else {
        (conservative, blind)
    };

    Ok(ArenaMap {
        slots,
        spans,
        live_in,
        bytes,
    })
}

/// One slot per value, every rectangle sized at the budget's ceiling and every
/// offset still zero.
fn rectangles(plan: &Plan, budgets: &Budgets) -> Result<Vec<Slot>, Refusal> {
    plan.values
        .iter()
        .enumerate()
        .map(|(id, decl)| {
            let value = ValueId(id as u32);
            match &decl.def {
                Def::Input(which) => Ok(Slot::Runtime(*which)),
                Def::Weight(i) => Ok(Slot::Param(*i)),
                Def::Cache(i) => Ok(Slot::Cache(*i)),
                // A MERGE IS GIVEN THE COLUMN, and its arms are folded onto it
                // below. That is the direction design §0 states: the arms
                // write disjoint row windows of ONE buffer, so the buffer is
                // the merge's and the arms are the writers.
                //
                // AND AN OP OUTPUT NO CLASS DEMANDS STILL GETS ONE.
                // `Classes::dead` reports those and the compiler is free to
                // drop them; not dropping them is the conservative reading,
                // and a shipped plan has none for it to cost anything on.
                Def::Op(_) | Def::Merge(_) => match &decl.ty {
                    Ty::Struct(kind) => Ok(Slot::Struct(*kind)),
                    Ty::Tensor { shape, dtype } => {
                        let (rows, width) =
                            rect(shape).map_err(|why| Refusal::Unrectangled { value, why })?;
                        let elem = elem_bytes(*dtype).ok_or(Refusal::Unrectangled {
                            value,
                            why: Unrectangled::PackedElement,
                        })?;
                        Ok(Slot::Arena {
                            offset: 0,
                            bytes: rows.max(budgets).saturating_mul(width).saturating_mul(elem),
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

/// A shape, read as `rows x width`.
///
/// THE LEADING DIM IS THE ROW COUNT AND THE REST IS THE WIDTH, and that is the
/// IR's own rule rather than this pass's convention: `check` faults a
/// `SymbolicAxis` for any symbolic dim past axis 0. The rule generalizes it
/// rather than assuming it — a shape with no dims is one row of one element,
/// and a shape whose leading dim is `Const` is a fixed block — and the refusal
/// below is the defensive echo a front door owes a plan that reached it
/// unvalidated.
fn rect(shape: &[Dim]) -> Result<(RowExpr, u64), Unrectangled> {
    let rows = shape
        .first()
        .copied()
        .map_or(RowExpr::Const(1), RowExpr::of);
    let mut width = 1u64;
    for dim in shape.iter().skip(1) {
        match dim {
            Dim::Const(n) => width = width.saturating_mul(*n),
            Dim::Tokens | Dim::TokensTimes(_) | Dim::Lanes | Dim::LanesPlus(_) => {
                return Err(Unrectangled::SymbolicWidth);
            }
        }
    }
    Ok((rows, width))
}

/// The bytes one element occupies, or `None` for a packed storage plane that
/// has none.
///
/// A PACKED PLANE IS NOT AN ACTIVATION. `Mxfp4` is 32 codes to 16 bytes and
/// `Fp4` is half a byte; both name how a WEIGHT is stored or how a kv page is
/// quantized, neither of which is a rectangle of this arena. Reaching one here
/// means an op declared its output in a storage element, and the honest answer
/// is a refusal naming the value.
#[must_use]
pub fn elem_bytes(dtype: Dtype) -> Option<u64> {
    match dtype {
        Dtype::Bf16 | Dtype::F16 => Some(2),
        Dtype::F32 | Dtype::I32 | Dtype::U32 => Some(4),
        Dtype::U8 | Dtype::I8 | Dtype::Fp8E4m3 | Dtype::E8m0 => Some(1),
        Dtype::Fp4 | Dtype::Mxfp4 => None,
    }
}

/// Fold every in-place result onto the operand it overwrites.
///
/// `Operands::aliases` states the pair — "`(out, the in it overwrites)`" — and
/// its own doc names this crate as what folds it: *"the compiler folds each
/// pair onto one arena slot, keeping InOut ops SSA at the IR level"*. The
/// declaration is not a hint about what the kernel MIGHT do; the kernel writes
/// through the operand's pointer, so the operand's bytes are gone whether the
/// carve folds them or not. Not folding would mint a second rectangle that
/// nothing ever writes.
///
/// AN ALIAS THAT REACHES OUTSIDE THE ARENA IS NOT THIS PASS'S BUSINESS. A
/// runtime binding and a host struct have no bytes here to fold, so the pair
/// is left alone rather than refused: the driver owns that buffer and the
/// in-place write lands in it.
fn fold_in_place(plan: &Plan, slots: &mut [Slot]) -> Result<(), Refusal> {
    let mut pairs: Vec<(ValueId, ValueId)> = Vec::new();
    for node in &plan.nodes {
        pairs.clear();
        node.op.aliases(&mut pairs);
        for (out, overwritten) in &pairs {
            share(slots, Share::InPlace, *overwritten, *out)?;
        }
    }
    Ok(())
}

/// Fold every merge's arms onto the merged column.
///
/// PHI OWNS THE BYTES AND THE ARMS ARE ITS WRITERS (design §0). Each arm's
/// guard admits a disjoint set of lanes, so the arms write disjoint ROW
/// WINDOWS of one buffer and the merge costs zero instructions. A merge
/// lowering to a copy — or to a second buffer nothing reads — is exactly the
/// cost this design exists to not pay.
///
/// Value order does the nesting for free: the recorder pushes a merge after
/// its arms, so an inner merge has already claimed its arms by the time the
/// outer one claims it, and [`share`] unions through the root.
fn fold_merges(plan: &Plan, slots: &mut [Slot]) -> Result<(), Refusal> {
    for (id, decl) in plan.values.iter().enumerate() {
        let Def::Merge(arms) = &decl.def else {
            continue;
        };
        let merge = ValueId(id as u32);
        for (arm, _) in arms {
            share(slots, Share::MergeArm, merge, *arm)?;
        }
    }
    Ok(())
}

/// Put `shares` into `holds`'s column, or refuse if the IR says they are one
/// column and declares them at two sizes.
fn share(slots: &mut [Slot], kind: Share, holds: ValueId, shares: ValueId) -> Result<(), Refusal> {
    let (h, s) = (root(slots, holds), root(slots, shares));
    if h == s {
        return Ok(());
    }
    let (Some(a), Some(b)) = (slots.get(h.0 as usize), slots.get(s.0 as usize)) else {
        // An id that indexes nothing is the validator's fault to name, not
        // this pass's to panic on.
        return Ok(());
    };
    if !a.is_arena() || !b.is_arena() {
        return Ok(());
    }
    let same = match (a, b) {
        (
            Slot::Arena {
                bytes: ab,
                rows: ar,
                width: aw,
                dtype: ad,
                ..
            },
            Slot::Arena {
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
        return Err(Refusal::Mismatch {
            kind,
            holds: h,
            shares: s,
        });
    }
    slots[s.0 as usize] = Slot::Alias(h);
    Ok(())
}

/// The rectangle an alias finally names.
fn root(slots: &[Slot], mut value: ValueId) -> ValueId {
    for _ in 0..=slots.len() {
        match slots.get(value.0 as usize) {
            Some(Slot::Alias(to)) => value = *to,
            _ => return value,
        }
    }
    panic!("a cycle of aliases through v{}", value.0)
}

/// Collapse alias chains to one hop, so that every reader sees the rectangle
/// and not another alias.
fn flatten(slots: &mut [Slot]) {
    for id in 0..slots.len() {
        if let Slot::Alias(to) = slots[id] {
            slots[id] = Slot::Alias(root(slots, to));
        }
    }
}

/// Each value's life: the node indices it spans, and the classes it spans them
/// in.
///
/// THE SPANS ARE STILL THE PLAN'S, NOT A CLASS'S, and the module head says why
/// they have to be: a mixed fire runs every class's nodes at one wall clock,
/// so a per-class span would call a value dead while a kernel is still reading
/// it. What the walk collects ALONGSIDE the span is the class mask of every
/// node that touched the value, and that is the tightening's whole input — not
/// a shorter life, but the answer to "whose rows of this column are ever
/// touched".
fn lives(plan: &Plan, slots: &[Slot], classes: &Classes) -> (Vec<Option<Span>>, Vec<ClassSet>) {
    let end = plan.nodes.len() as u32;
    // The reader with no class: the engine, past the last node, over every row
    // the fire carried. Both pins below widen to it.
    let everywhere = ClassSet::of(0..classes.classes.len());
    let nowhere = ClassSet::default();
    let mut spans: Vec<Option<Span>> = vec![None; slots.len()];
    let mut live_in: Vec<ClassSet> = vec![ClassSet::default(); slots.len()];
    let mut touched: Vec<ValueId> = Vec::new();

    for (at, node) in plan.nodes.iter().enumerate() {
        let at = at as u32;
        // Parallel to `plan.nodes`, and a plan whose sweep and node list
        // disagree gets the empty mask — the same conservative reading
        // `region::coalesce` takes at the same index.
        let mask = classes.node_mask.get(at as usize).unwrap_or(&nowhere);
        touched.clear();
        node.op.inputs(&mut touched);
        node.op.outputs(&mut touched);
        for value in &touched {
            touch(slots, &mut spans, &mut live_in, *value, at, mask);
        }
    }

    // THE DELIVERY TAIL, AND EVERY DECLARED EXPORT TAKES IT ([`EXPORTS`]).
    // These values are read after the last node has run — the trunk's logits
    // into the engine's sampler, a draft column into the same sampler through
    // its intrinsic, a capture column into the lane's readout — by a reader no
    // node occupies, so a value sharing their bytes would clobber them between
    // the launch and the read.
    //
    // **THE TAIL IS TWO SEPARATE PINS AND ONLY ONE OF THEM IS THE SAME FOR
    // EVERY EXPORT.** The SPAN runs to `end` in all three cases: the reader is
    // past the last node whichever export it came for. The CLASS MASK is the
    // reader's own — `everywhere` for `"out"`, whose sampler takes every lane
    // of the fire, and the classes the export actually ran in for the other
    // two, whose readers came for the drafting lanes' rows or the capturing
    // lanes' rows and for nobody else's. Widening those to `everywhere` would
    // be a claim about rows no reader will look at, and it would price a
    // per-layer capture column at the whole fire's height in every SKU that
    // declares one.
    for export in EXPORTS {
        for seam in plan.seams.iter().filter(|s| s.seam == export.seam) {
            for value in &seam.values {
                let root = root(slots, *value);
                if !slots.get(root.0 as usize).is_some_and(Slot::is_arena) {
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

    // EVERY RECTANGLE ENDS UP WITH A SPAN, so nothing downstream has to decide
    // what an absent one means. One can only be absent for a value no node
    // names at all, and holding that open across the whole plan is the reading
    // that cannot be wrong.
    for (id, slot) in slots.iter().enumerate() {
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

/// Extend a value's life to cover instant `at` in the classes `mask` runs,
/// through any alias.
fn touch(
    slots: &[Slot],
    spans: &mut [Option<Span>],
    live_in: &mut [ClassSet],
    value: ValueId,
    at: u32,
    mask: &ClassSet,
) {
    // A MERGE IS ITS COLUMN'S LIFE, NOT ITS OWN. Reading a merged value reads
    // the column its arms wrote, so the read lands on the column — which is
    // how an alias extends the life of what it points at. The classes go the
    // same way, and that is exactly why a merge column co-tenants with nobody:
    // it collects the union of its arms' masks, which is every class the
    // merge resolves in.
    let root = root(slots, value);
    if !slots.get(root.0 as usize).is_some_and(Slot::is_arena) {
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

/// Do these two sets of classes hold no class in common?
fn disjoint(a: &ClassSet, b: &ClassSet) -> bool {
    !a.iter().any(|class| b.contains(class))
}

/// The alignment every reservation is rounded up to, and therefore every
/// offset sits on.
///
/// TWO HUNDRED AND FIFTY-SIX, AND THE REWRITE EXPLAINS WHY. Rounding the
/// SIZES is what keeps a freed hole aligned too, so every offset lands aligned
/// with no separate padding pass; the number is the largest storage-buffer
/// offset alignment a conformant device may demand
/// (Vulkan's `minStorageBufferOffsetAlignment`), and `Platform::Vulkan` is in
/// the IR. It started at 16 there and was raised after a plan this compiler laid
/// out turned out to be one such a device would refuse to bind — silently, on
/// the adapter that tree tested on, which reports 16.
///
/// MEASURED BEFORE IT WAS CHANGED, over every catalog row: the pitch is
/// IDENTICAL at 16 and at 256 for every SKU but one, and gpt-oss paid 128
/// bytes on a 407,936-byte row — 0.03%. A slot count in the hundreds and sizes
/// already far above 256 is why the rounding almost never has anything to
/// round.
const BIND_ALIGN: u64 = 256;

/// A reservation, rounded up to [`BIND_ALIGN`].
fn align(bytes: u64) -> u64 {
    bytes.div_ceil(BIND_ALIGN) * BIND_ALIGN
}

/// Give every rectangle an offset, sharing bytes between values that are never
/// live together, and answer what the arena adds up to.
///
/// # Why greedy-by-size, and how close it gets
///
/// The lower bound is the busiest instant: the total bytes live at whichever
/// node holds the most ([`ArenaMap::live_bound`]). Reaching it exactly is
/// dynamic storage allocation, which is NP-hard in general — but these
/// intervals are a transformer's, which is to say a few long-lived residuals
/// crossing a long chain of short-lived scratch, and placing the big blocks
/// first leaves gaps the small ones drop into. The rewrite's identical walk
/// sat ON the bound for every catalog row, so nothing more elaborate has
/// earned its way in.
///
/// # Deterministic, and that is a requirement
///
/// The same plan must lay out the same way on every host: a `Baked` is cached,
/// compared and fired by offsets. So the order is a TOTAL one — bytes
/// descending, then birth node, then value id — and never a hash's. The
/// columns are gathered in that same order and a value joins the FIRST column
/// that will have it, so which values end up sharing one is a function of the
/// plan and of nothing else.
fn place(
    slots: &mut [Slot],
    spans: &[Option<Span>],
    live_in: &[ClassSet],
    conc: &Concurrency,
    mode: Columns,
) -> u64 {
    let mut order: Vec<(u64, Span, usize)> = slots
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

    let columns = gather(slots, live_in, &order, mode);

    let mut placed: Vec<(u64, u64, Span)> = Vec::with_capacity(columns.len());
    let mut blockers: Vec<(u64, u64)> = Vec::new();
    let mut bytes = 0u64;
    for column in &columns {
        // The lowest offset no column live beside this one already holds: walk
        // the blockers in address order, stepping past each one that starts
        // before the gap under consideration closes.
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
            let Slot::Arena { offset, .. } = &mut slots[*id] else {
                unreachable!("only arena slots are gathered into columns")
            };
            *offset = at;
        }
        placed.push((at, column.size, column.span));
        bytes = bytes.max(at + column.size);
    }
    bytes
}

/// Whether [`place`] may put two values in one column, or has to give every
/// rectangle its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Columns {
    /// The v1 walk: one column per rectangle, classes unread.
    PerValue,
    /// The tightening: class-disjoint rectangles of one pitch may share.
    Shared,
}

/// One reservation of the arena, and the values that live in it.
///
/// A column is what actually gets an offset. Under [`Columns::PerValue`] it
/// has exactly one member and this is the v1 walk with one indirection added;
/// under [`Columns::Shared`] its members are class-disjoint row windows of the
/// one rectangle.
struct Column {
    /// The rounded reservation — one number for every member, since they agree
    /// about pitch.
    size: u64,
    /// The union of the members' lives, which is an interval the members
    /// COVER: a value only joins a column whose span its own already touches,
    /// so the column blocks exactly what its members blocked.
    span: Span,
    /// Every class any member is touched in.
    classes: ClassSet,
    /// `(rows, width, dtype)` — what a member has to match — and `None` for a
    /// rectangle no window cuts, which admits no second member.
    pitch: Option<(RowExpr, u64, Dtype)>,
    /// Slot indices, in the order they joined.
    members: Vec<usize>,
}

/// Gather the placement order into columns.
fn gather(
    slots: &[Slot],
    live_in: &[ClassSet],
    order: &[(u64, Span, usize)],
    mode: Columns,
) -> Vec<Column> {
    let mut columns: Vec<Column> = Vec::with_capacity(order.len());
    for (size, span, id) in order {
        let pitch = pitch_of(&slots[*id]);
        let classes = &live_in[*id];
        // EVERY CLAUSE IS LOAD-BEARING. One pitch and one reservation, or the
        // rows do not line up; a cut, or the reader never sees a window at
        // all; disjoint and KNOWN classes, or two live kernels want the same
        // rows; and spans that ALREADY TOUCH, so the column's life stays the
        // interval its members cover and sharing can only ever drop a
        // reservation, never lengthen a conflict.
        let joined = match (mode, pitch) {
            (Columns::Shared, Some(pitch)) if !classes.is_empty() => {
                columns.iter_mut().find(|column| {
                    column.pitch == Some(pitch)
                        && column.size == *size
                        && touching(column.span, *span)
                        && disjoint(&column.classes, classes)
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

/// What a value has to agree about to share a column, or `None` for a
/// rectangle a window hands over whole ([`RowExpr::cut_per_class`]).
fn pitch_of(slot: &Slot) -> Option<(RowExpr, u64, Dtype)> {
    match slot {
        Slot::Arena {
            rows, width, dtype, ..
        } if rows.cut_per_class() => Some((*rows, *width, *dtype)),
        _ => None,
    }
}

/// Do these two lives share a node index?
///
/// THE INTERVAL TEST ALONE, DELIBERATELY, and not [`Concurrency::overlap`].
/// What the join needs is that the union of two spans is one interval the two
/// of them cover, which a P6 pair does not give. Two values a pair forces
/// apart are already forbidden from sharing bytes, so asking the narrower
/// question here loses nothing.
fn touching(a: Span, b: Span) -> bool {
    a.first <= b.last && b.first <= a.last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Build, block, fact};
    use crate::region;
    use model_ir::{Cond, resolve_classes};

    /// Carve a fixture plan the way `compile` would.
    fn carved(b: &Build, budgets: &Budgets) -> (ArenaMap, Concurrency) {
        let classes = resolve_classes(&b.plan).expect("the fixture plans resolve");
        let regions = region::coalesce(&b.plan, &classes);
        let conc = Concurrency::sequential(&regions, b.plan.nodes.len());
        let arena = carve(&b.plan, budgets, &classes, &conc).expect("the fixture plans carve");
        (arena, conc)
    }

    /// The v1 walk, whole, as the oracle the tightening is measured against:
    /// the same folds and the same spans, placed one column per rectangle.
    ///
    /// KEPT AS A SECOND CALLER OF THE SAME PIECES rather than as a second
    /// implementation — everything up to [`place`] is shared, and the only
    /// thing that differs is the [`Columns`] mode, which is exactly the
    /// difference under test.
    fn conservative(b: &Build, budgets: &Budgets) -> u64 {
        let classes = resolve_classes(&b.plan).expect("the fixture plans resolve");
        let regions = region::coalesce(&b.plan, &classes);
        let conc = Concurrency::sequential(&regions, b.plan.nodes.len());
        let mut slots = rectangles(&b.plan, budgets).expect("the fixture plans carve");
        fold_in_place(&b.plan, &mut slots).expect("the fixture plans carve");
        fold_merges(&b.plan, &mut slots).expect("the fixture plans carve");
        flatten(&mut slots);
        let (spans, live_in) = lives(&b.plan, &slots, &classes);
        place(&mut slots, &spans, &live_in, &conc, Columns::PerValue)
    }

    /// A value's offset, or a panic naming the value that has none.
    fn at(arena: &ArenaMap, v: ValueId) -> u64 {
        match arena.slots[v.0 as usize] {
            Slot::Arena { offset, .. } => offset,
            _ => panic!("v{} is not a rectangle", v.0),
        }
    }

    /// THE SHAPE THE TIGHTENING IS ABOUT: two guarded runs INTERLEAVED in
    /// program order, so that each one's scratch is still live when the
    /// other's is born. `d1` lives over nodes 1..3 and `p1` over 2..4, and no
    /// class holds both.
    fn interleaved(width: u64) -> (Build, ValueId, ValueId) {
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always); // node 0 — everywhere
        let d1 = b.op(q, 8, fact(0)); // node 1 — one class
        let p1 = b.op(q, width, Cond::not(fact(0))); // node 2 — the other
        let d2 = b.op(d1, 8, fact(0)); // node 3 — d1 dies here
        let p2 = b.op(p1, 8, Cond::not(fact(0))); // node 4 — p1 dies here
        let o = b.merge(&[(d2, fact(0)), (p2, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always); // node 5
        b.out(y);
        (b, d1, p1)
    }

    fn budgets() -> Budgets {
        Budgets::new(4, 16)
    }

    #[test]
    fn the_row_algebra_sizes_every_dim_at_its_ceiling() {
        let b = Budgets::new(4, 16);
        assert_eq!(RowExpr::of(Dim::Tokens).max(&b), 16);
        assert_eq!(RowExpr::of(Dim::TokensTimes(3)).max(&b), 48);
        assert_eq!(RowExpr::of(Dim::Lanes).max(&b), 4);
        assert_eq!(RowExpr::of(Dim::LanesPlus(1)).max(&b), 5);
        assert_eq!(RowExpr::of(Dim::Const(7)).max(&b), 7);
        // …and a fire smaller than the ceiling uses fewer.
        assert_eq!(RowExpr::of(Dim::Tokens).at(3, 2), 3);
        assert_eq!(RowExpr::of(Dim::LanesPlus(1)).at(3, 2), 3);
    }

    #[test]
    fn a_chain_of_scratch_reuses_one_pair_of_columns() {
        // x -> a -> c -> d, each read once by the next. Only two are ever live
        // at a time, so the arena is two columns wide however long the chain
        // gets. The sum of the four rectangles would be four.
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let c = b.op(a, 8, Cond::Always);
        let d = b.op(c, 8, Cond::Always);
        let e = b.op(d, 8, Cond::Always);
        b.out(e);

        let (arena, conc) = carved(&b, &budgets());
        // One rectangle is 16 rows x 8 bf16 = 256 bytes, which is exactly the
        // alignment, so the arithmetic here is unrounded.
        let one = 16 * 8 * 2;
        assert_eq!(arena.bytes, 2 * one, "two columns, not four");
        assert_eq!(arena.bytes, arena.live_bound(), "on the floor");
        assert!(arena.clashes(&conc).is_empty());
        // `a` and `c` are never live together, so they are the same bytes.
        assert_ne!(at(&arena, a), at(&arena, c), "adjacent links do overlap");
        assert_eq!(at(&arena, a), at(&arena, d), "two links apart, they do not");
    }

    #[test]
    fn a_stack_of_layers_costs_a_handful_of_columns_and_not_a_stack_of_them() {
        // THE CLAIM, AT SOMETHING LIKE THE SHAPE IT IS CLAIMED ABOUT. Twenty-
        // four blocks, each minting a norm, an activation and a residual
        // update, is 72 rectangles — and only a few are ever live at once,
        // because the residual ledger writes through itself and every scratch
        // dies at its one reader. An arena that grew with the depth would be
        // the sum, which is the thing the liveness walk exists to not be.
        let layers = 24u64;
        let mut b = Build::new();
        let x = b.input(8);
        let mut resid = b.op(x, 8, Cond::Always);
        for _ in 0..layers {
            let norm = b.op(resid, 8, Cond::Always);
            let attn = b.op(norm, 8, Cond::Always);
            resid = b.residual_add(attn, resid, 8, Cond::Always);
        }
        b.out(resid);

        let (arena, conc) = carved(&b, &budgets());
        assert!(arena.clashes(&conc).is_empty());
        assert_eq!(arena.bytes, arena.live_bound(), "on the floor");
        let one = 16 * 8 * 2;
        assert!(
            arena.bytes <= 4 * one,
            "{layers} layers took {} columns",
            arena.bytes / one,
        );
        // …and the residual really is ONE column, however many times it is
        // added into.
        let columns = arena.slots.iter().filter(|slot| slot.is_arena()).count();
        assert!(
            columns as u64 <= 2 * layers + 2,
            "{columns} rectangles for {layers} layers — the in-place folds did not fold",
        );
    }

    #[test]
    fn a_merge_owns_the_column_and_its_arms_write_into_it() {
        // phi costs nothing: the arms are windows of one buffer.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.out(y);

        let (arena, conc) = carved(&b, &budgets());
        assert_eq!(arena.slots[d.0 as usize], Slot::Alias(o));
        assert_eq!(arena.slots[p.0 as usize], Slot::Alias(o));
        assert!(matches!(arena.slots[o.0 as usize], Slot::Arena { .. }));
        assert_eq!(arena.root(d), o);
        assert!(arena.clashes(&conc).is_empty());
        // The two arms are the same bytes ON PURPOSE — that is what makes
        // a merge free — and the clash guard must not call it a clash.
        assert_eq!(arena.window(d, 4, 2), arena.window(p, 4, 2));
    }

    #[test]
    fn an_in_place_op_writes_through_its_operand() {
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let c = b.in_place(a, 8, Cond::Always);
        b.out(c);

        let (arena, _) = carved(&b, &budgets());
        assert_eq!(arena.slots[c.0 as usize], Slot::Alias(a));
        assert_eq!(arena.bytes, 16 * 8 * 2, "one column, written twice");
    }

    #[test]
    fn the_out_seam_is_live_past_the_last_node() {
        // `y` is written by the last node and read by nobody in the plan. If
        // its span ended there, the scratch before it could take its bytes and
        // the driver would read whatever the last launch left.
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let c = b.op(a, 8, Cond::Always);
        b.out(c);

        let (arena, conc) = carved(&b, &budgets());
        let end = b.plan.nodes.len() as u32;
        assert_eq!(arena.spans[c.0 as usize].unwrap().last, end);
        assert!(arena.clashes(&conc).is_empty());
    }

    #[test]
    fn a_lane_shaped_vector_reserves_lanes_and_not_tokens() {
        // The mixture the rewrite's per-row pitch could not express: an
        // indptr is `lanes + 1` rows and an activation is `tokens` rows, and
        // 16 tokens over 4 lanes means the two reserve different amounts.
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let indptr = b.value(
            Def::Op(b.plan.nodes.len() as u32),
            Ty::Tensor {
                shape: vec![Dim::LanesPlus(1), Dim::Const(1)],
                dtype: Dtype::I32,
            },
        );
        b.plan.nodes.push(model_ir::Node {
            op: model_ir::ops::Elementwise::RmsnormNoScale {
                x: a,
                head_dim: 1,
                eps: 1e-6,
                y: indptr,
            }
            .into(),
            cond: Cond::Always,
            layer: None,
        });
        b.out(indptr);

        let (arena, _) = carved(&b, &budgets());
        let Slot::Arena { bytes, rows, .. } = arena.slots[indptr.0 as usize] else {
            panic!("the indptr is a rectangle")
        };
        assert_eq!(rows, RowExpr::LanesPlus(1));
        // 5 rows (lanes + 1), one element wide, 4 bytes each.
        assert_eq!(bytes, 5 * 4, "lanes + 1 rows of i32, not tokens");
        // …and in a fire of 2 lanes it touches three rows.
        assert_eq!(arena.window(indptr, 7, 2).unwrap().bytes, 3 * 4);
    }

    #[test]
    fn a_concurrency_relation_stops_two_regions_sharing_bytes() {
        // The P6 hook, exercised by hand: two values whose node ranges are
        // disjoint DO share bytes under one stream, and must not once the two
        // regions they live in may run at once.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0)); // node 1, its own region
        let p = b.op(q, 8, Cond::not(fact(0))); // node 2, its own region
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.out(y);

        let classes = resolve_classes(&b.plan).expect("resolves");
        let regions = region::coalesce(&b.plan, &classes);
        let nodes = b.plan.nodes.len();
        let sequential = Concurrency::sequential(&regions, nodes);
        let shared = carve(&b.plan, &budgets(), &classes, &sequential).expect("carves");
        assert!(shared.clashes(&sequential).is_empty());

        // Now say regions 1 and 2 may be in flight together. The carve must
        // grow, and the sequential map must now REPORT a clash under the wider
        // relation — which is the whole reason the predicate takes it.
        let forked = Concurrency::with_pairs(&regions, nodes, [(1, 2)]);
        let apart = carve(&b.plan, &budgets(), &classes, &forked).expect("carves");
        assert!(apart.clashes(&forked).is_empty());
        assert!(
            apart.bytes >= shared.bytes,
            "running regions together cannot need fewer bytes",
        );
    }

    #[test]
    fn two_values_no_class_holds_at_once_share_one_column() {
        // THE TIGHTENING, AT THE SMALLEST SHAPE THAT SHOWS IT. `d1` and `p1`
        // are live over overlapping runs of nodes — the v1 walk has to give
        // them different bytes — but one is touched only where `qo_one` holds
        // and the other only where it does not, so they are two row windows of
        // ONE column: `Run::cut` hands each guarded node its own class's rows
        // and the two intervals never meet.
        let (b, d1, p1) = interleaved(8);
        let (arena, conc) = carved(&b, &budgets());

        assert_eq!(at(&arena, d1), at(&arena, p1), "one column, two windows");
        assert!(arena.co_tenants(d1, p1));
        assert!(
            arena.clashes(&conc).is_empty(),
            "the refined guard is happy"
        );

        // …and the v1 predicate is NOT, which is the whole reason it had to be
        // refined alongside the carve: blind to classes, it reads a shared
        // column as a shared byte.
        assert_eq!(arena.clashes_blind(&conc), vec![(d1, p1)]);

        // The saving is one whole column, and the carve lands on the floor.
        let one = 16 * 8 * 2;
        assert_eq!(conservative(&b, &budgets()), 3 * one);
        assert_eq!(arena.bytes, 2 * one);
        assert_eq!(arena.bytes, arena.live_bound(), "on the floor");
    }

    #[test]
    fn two_values_one_class_holds_at_once_do_not_share() {
        // The other half of the rule. `q` is written by an unconditional node,
        // so it is live in EVERY class, and `d1` is live in one of them — the
        // masks meet, and a fire of that class has both resident at node 1.
        let (b, d1, _) = interleaved(8);
        let (arena, conc) = carved(&b, &budgets());
        let q = ValueId(1);

        assert!(!arena.co_tenants(q, d1));
        assert_ne!(at(&arena, q), at(&arena, d1));
        assert!(arena.clashes(&conc).is_empty());
    }

    #[test]
    fn a_column_is_not_shared_across_two_pitches() {
        // Same lives, same classes, HALF the width. A row window is
        // `offset + row * pitch`, so class 1's rows of the narrow column would
        // land inside class 0's rows of the wide one — and the rounding hides
        // it, because 16x4 bf16 and 16x8 bf16 reserve the same 256 bytes.
        let (b, d1, p1) = interleaved(4);
        let (arena, conc) = carved(&b, &budgets());

        assert!(!arena.co_tenants(d1, p1));
        assert_ne!(at(&arena, d1), at(&arena, p1));
        assert_eq!(arena.bytes, conservative(&b, &budgets()), "nothing shared");
        assert!(arena.clashes(&conc).is_empty());
    }

    #[test]
    fn a_fixed_block_never_shares_a_column_however_disjoint_its_classes() {
        // `Dim::Const` is handed over WHOLE — `Run::cut` does not slice a
        // rectangle that is not fire-aligned — so both guarded nodes would
        // write all three rows of the same block. Class-disjointness buys
        // nothing here, and the rule has to know it.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always); // node 0
        let d1 = b.shaped(q, block(3, 8), fact(0)); // node 1
        let p1 = b.shaped(q, block(3, 8), Cond::not(fact(0))); // node 2
        let d2 = b.op(d1, 8, fact(0)); // node 3
        let p2 = b.op(p1, 8, Cond::not(fact(0))); // node 4
        let o = b.merge(&[(d2, fact(0)), (p2, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always); // node 5
        b.out(y);

        let (arena, conc) = carved(&b, &budgets());
        assert!(disjoint(
            &arena.live_in[d1.0 as usize],
            &arena.live_in[p1.0 as usize],
        ));
        assert!(!arena.co_tenants(d1, p1), "a const block has no window");
        assert_ne!(at(&arena, d1), at(&arena, p1));
        assert!(arena.clashes(&conc).is_empty());
    }

    #[test]
    fn a_merge_arm_is_not_a_co_tenant_because_it_is_not_a_second_rectangle() {
        // The mechanism this generalizes, kept distinct from it. An arm is
        // folded onto the merge by `fold_merges`, so there is one rectangle
        // and the arms are `Slot::Alias` — which has no column to share and
        // therefore no co-tenancy to report.
        let mut b = Build::new();
        let x = b.input(8);
        let q = b.op(x, 8, Cond::Always);
        let d = b.op(q, 8, fact(0));
        let p = b.op(q, 8, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        let y = b.op(o, 8, Cond::Always);
        b.out(y);

        let (arena, conc) = carved(&b, &budgets());
        assert_eq!(arena.slots[d.0 as usize], Slot::Alias(o));
        assert!(!arena.co_tenants(d, p), "they are one rectangle already");
        assert_eq!(arena.window(d, 4, 2), arena.window(p, 4, 2));
        // The column collects both arms' classes, so IT co-tenants with
        // nobody — which is right: an unguarded reader takes all of its rows.
        assert_eq!(arena.live_in[o.0 as usize].len(), 2);
        assert!(arena.clashes(&conc).is_empty());
    }

    #[test]
    fn the_out_seam_is_read_in_every_class_and_so_shares_with_nobody() {
        // `s` is touched only where `qo_one` is false and `a` only where it is
        // true, and their spans meet at node 1 — every clause of the join but
        // one. The one is the seam: `a` is the plan's output, and the sampler
        // that reads it past the last node reads EVERY lane's rows, whichever
        // class the lane is. Without that pin the two would be given one
        // column and the delivery would hand back somebody else's rows.
        let mut b = Build::new();
        let x = b.input(8);
        let s = b.op(x, 8, Cond::not(fact(0))); // node 0
        let a = b.op(s, 8, fact(0)); // node 1 — the "out" seam
        b.out(a);

        let (arena, conc) = carved(&b, &budgets());
        assert_eq!(arena.live_in[s.0 as usize].len(), 1, "one class writes it");
        assert_eq!(
            arena.live_in[a.0 as usize].len(),
            2,
            "the seam is read in both"
        );
        assert!(!arena.co_tenants(s, a));
        assert_ne!(at(&arena, s), at(&arena, a));
        assert!(arena.clashes(&conc).is_empty());
    }

    #[test]
    fn a_shared_column_is_not_undone_by_a_concurrency_pair() {
        // P6 IS ABOUT TIME AND CO-TENANCY IS ABOUT BYTES, so the two do not
        // argue. Two regions running side by side write the rows of their own
        // classes, and rows of one column that never meet do not race however
        // concurrently they are written — which is why the join asks
        // `touching` and not `Concurrency::overlap`.
        let (b, d1, p1) = interleaved(8);
        let classes = resolve_classes(&b.plan).expect("resolves");
        let regions = region::coalesce(&b.plan, &classes);
        let nodes = b.plan.nodes.len();

        let forked = Concurrency::with_pairs(&regions, nodes, [(1, 2), (3, 4)]);
        let arena = carve(&b.plan, &budgets(), &classes, &forked).expect("carves");
        assert_eq!(at(&arena, d1), at(&arena, p1), "still one column");
        assert!(arena.co_tenants(d1, p1));
        assert!(arena.clashes(&forked).is_empty());
    }

    #[test]
    fn a_symbolic_width_is_refused_and_names_the_value() {
        let mut b = Build::new();
        let x = b.input(8);
        let node = b.plan.nodes.len() as u32;
        let y = b.value(
            Def::Op(node),
            Ty::Tensor {
                shape: vec![Dim::Tokens, Dim::Lanes],
                dtype: Dtype::Bf16,
            },
        );
        b.plan.nodes.push(model_ir::Node {
            op: model_ir::ops::Elementwise::RmsnormNoScale {
                x,
                head_dim: 1,
                eps: 1e-6,
                y,
            }
            .into(),
            cond: Cond::Always,
            layer: None,
        });
        b.out(y);

        let classes = resolve_classes(&b.plan).expect("resolves");
        let regions = region::coalesce(&b.plan, &classes);
        let conc = Concurrency::sequential(&regions, b.plan.nodes.len());
        assert_eq!(
            carve(&b.plan, &budgets(), &classes, &conc),
            Err(Refusal::Unrectangled {
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
        let p = b.op(x, 4, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 8);
        b.out(o);

        let classes = resolve_classes(&b.plan).expect("resolves");
        let regions = region::coalesce(&b.plan, &classes);
        let conc = Concurrency::sequential(&regions, b.plan.nodes.len());
        assert_eq!(
            carve(&b.plan, &budgets(), &classes, &conc),
            Err(Refusal::Mismatch {
                kind: Share::MergeArm,
                holds: o,
                shares: p,
            }),
        );
    }

    #[test]
    fn a_deliberately_wrong_offset_is_caught_by_the_clash_guard() {
        // The guard's own test: break a carve on purpose and check that the
        // invariant notices. A reused slab does not fault when it is wrong.
        let mut b = Build::new();
        let x = b.input(8);
        let a = b.op(x, 8, Cond::Always);
        let c = b.op(a, 8, Cond::Always);
        b.out(c);

        let (mut arena, conc) = carved(&b, &budgets());
        assert!(arena.clashes(&conc).is_empty());
        if let Slot::Arena { offset, .. } = &mut arena.slots[c.0 as usize] {
            *offset = 0;
        }
        assert_eq!(arena.clashes(&conc), vec![(a, c)]);
    }
}
