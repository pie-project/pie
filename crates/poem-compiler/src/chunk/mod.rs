//! How a model's states pack into chunks: a pure pass over a trace that
//! picks the chunk size `C` for a device and lays each kind of state into a
//! slot shape a chunk holds many of.
//!
//! Three slot shapes. A **list** slot holds `T` tokens of every
//! full-retention kv row, layer-major (each row a contiguous run of
//! `T x row bytes`). A **ring** slot holds `T` tokens of the window rows
//! sharing one window; a working set holds `cap_slots` of them. A **fixed**
//! slot holds a working set's record of every fixed state (recurrent and
//! conv states, frame caches, n-gram ids, the rs buffer as `[P, width]`
//! rows), whole when it fits in a chunk, else in slots of a piece's size: a
//! row no larger stays whole beside others, a row larger is cut on the atoms
//! its readers let a piece cut into equal pieces, each at one offset in its
//! own slot ([`FixedRun`]). The rs rows lie together in the same slots, whole
//! or cut alike on their tokens, whichever wastes less ([`RsLayout`]), as
//! an engine reads a lane's rs buffer through one page table.
//!
//! Beside the states, a **weight** slot holds read-only weights a device may
//! stream ([`weights`]): a routed expert of an MoE layer, or a block of
//! token-id-gathered table rows; the chunk size weighs how whole experts
//! fill it too.
//!
//! The pass reads today's IR ([`poem_ir::CacheRow`]): spaces, windows,
//! planes and slabs from the rows, layers, compression ratios, atoms and the
//! rs buffer from the ops that read them.

pub mod census;
mod fixed;
pub mod profile;
pub mod weights;

use std::collections::BTreeMap;
use std::ops::Range;

use poem_ir::{Shard, Trace};

pub use census::{Atom, Contract, MOST_RANKS, RsPlane, RsRow, UNFOLDED_ROWS};
pub use profile::ChunkProfile;
pub use weights::{WeightBank, WeightKind, WeightSlot};

use crate::{Budgets, DeviceProfile};
use census::{FixedRow, TokenRow};

/// The least bytes a piece of a split fixed record holds: a row no larger is
/// never cut, and no piece of a split record is smaller.
pub const MIN_PIECE: u64 = 1 << 20;

/// The least contiguous run a layer's rows make in a token slot.
pub const MIN_RUN: u64 = 32 << 10;

/// Every token slot holds a multiple of this many tokens.
pub const TOKEN_GRAIN: u32 = 16;

/// The fixed records a working set holds, resting or running: the one its
/// fires read.
pub const FIXED_SETS_PER_WS: u32 = 1;

/// The copy-on-write fixed records a lane that runs in a fire takes beside its
/// working set's (a fork, or a snapshot restored, within the fire). The pool
/// keeps one reserve of these for its most lanes, not one a working set.
pub const COW_SETS_PER_LANE: u32 = 1;

/// The copy-on-write fixed records a pool reserves under `budgets`:
/// [`COW_SETS_PER_LANE`] for each of the most lanes a fire runs. Derived, not
/// configured.
#[must_use]
pub fn cow_sets(budgets: &Budgets) -> u32 {
    budgets
        .tokens
        .max_lanes
        .max(1)
        .saturating_mul(COW_SETS_PER_LANE)
}

/// The least bytes a fixed slot takes: a record of a few bytes shares a chunk
/// with a thousand-odd others.
pub const FIXED_SLOT_FLOOR: u64 = 1 << 10;

/// How much more of the chunks the weight slots fill may hold nothing than
/// at the best-filling size, for the states to choose among the sizes.
const WEIGHT_TAIL_SLACK: f64 = 0.01;

/// The slots of a token of every rs row the largest chunk weighed holds.
const RS_SHARES: u64 = 4;

/// The most ways the fixed slot sizes weighed cut a row.
const MOST_CUTS: u64 = 64;

/// The most chunk sizes the plan weighs.
const MOST_CANDIDATES: u64 = 128;

/// A row a slot holds: a cache row of the trace, or a row of the rs buffer:
/// its ordinal among the deferred-commit recurrences in node order, then
/// among the n-gram id windows ([`ChunkRecipe::rs`] says which).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RowId {
    Cache(u32),
    Rs(u32),
}

/// Where a per-token row runs inside a token slot: `offset`, then
/// `tokens x per_token` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenRun {
    pub row: RowId,
    pub layer: Option<u32>,
    pub offset: u64,
    pub per_token: u64,
}

/// What a ring holds past its window: the drafted rows a lane verifies and
/// the rows one fire lands before the oldest retire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingCap {
    pub window: u32,
    pub draft: u32,
    pub fire_rows: u32,
}

impl RingCap {
    /// The tokens a working set's ring must hold at once.
    #[must_use]
    pub fn tokens(&self) -> u32 {
        self.window + self.draft + self.fire_rows
    }

    /// `ceil((W + draft + fire rows) / T) + 1`: the slot the oldest token
    /// is retiring from stays held while the newest fills.
    #[must_use]
    pub fn slots(&self, tokens: u32) -> u32 {
        self.tokens().div_ceil(tokens.max(1)) + 1
    }
}

/// Where a fixed row lies in a working set's record, as an engine addresses
/// it (CUDA `StateRows`, Metal `state_map`): `pieces` equal pieces of
/// `atoms_per_piece` atoms along the row's atom axis, the last padded where
/// the atoms do not divide; piece `p` at `offset` bytes into the record's
/// slot `slots[p]`. A whole row is one piece of all its atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRun {
    pub row: RowId,
    pub pieces: u32,
    pub atoms_per_piece: u64,
    pub offset: u64,
    /// The record's slots the pieces lie in, `0..pieces_per_ws`, one a piece.
    pub slots: Vec<u32>,
}

impl FixedRun {
    /// The atoms piece `p` holds of a row of `extent` atoms; empty past them.
    #[must_use]
    pub fn span(&self, p: u32, extent: u64) -> Range<u64> {
        let start = (u64::from(p) * self.atoms_per_piece).min(extent);
        start..(start + self.atoms_per_piece).min(extent)
    }

    /// A piece's bytes, padding counted, of a row of `atom`s.
    #[must_use]
    pub fn piece_bytes(&self, atom: &Atom) -> u64 {
        self.atoms_per_piece * atom.bytes
    }

    /// Whether the row is cut.
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.pieces > 1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotLayout {
    List {
        tokens: u32,
        bytes: u64,
        rows: Vec<TokenRun>,
    },
    /// A window's rows: a working set holds `cap_slots` slots of `tokens`
    /// tokens each, reused in turn. Position `t` lies in the working set's
    /// ring slot `(t / tokens) % cap_slots`, a row's token at its run's
    /// `offset + (t % tokens) x per_token`; `cap.window` is the retention.
    Ring {
        tokens: u32,
        bytes: u64,
        cap_slots: u32,
        cap: RingCap,
        rows: Vec<TokenRun>,
    },
    Fixed {
        /// One piece's slot.
        bytes: u64,
        /// The record's bytes, its rows aligned.
        record: u64,
        /// The slots a working set's record takes.
        pieces_per_ws: u32,
        /// Every fixed row's run, in the record's row order.
        rows: Vec<FixedRun>,
        /// Window rings small enough to hold whole in the record, and the
        /// tokens each holds: a fixed row of that many token atoms,
        /// position `t` at atom `t % tokens`.
        folded: Vec<(RowId, u32)>,
    },
}

impl SlotLayout {
    /// A slot's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            SlotLayout::List { bytes, .. }
            | SlotLayout::Ring { bytes, .. }
            | SlotLayout::Fixed { bytes, .. } => *bytes,
        }
    }

    /// The slots of this shape a working set holds at `context` tokens.
    #[must_use]
    pub fn slots_per_ws(&self, context: u32) -> u64 {
        match self {
            SlotLayout::List { tokens, .. } => u64::from(context.div_ceil(*tokens)),
            SlotLayout::Ring { cap_slots, .. } => u64::from(*cap_slots),
            SlotLayout::Fixed { pieces_per_ws, .. } => {
                u64::from(FIXED_SETS_PER_WS) * u64::from(*pieces_per_ws)
            }
        }
    }

    /// The slots of this shape a chunk of `chunk` bytes holds.
    #[must_use]
    pub fn per_chunk(&self, chunk: u64) -> u64 {
        chunk / self.bytes().max(1)
    }

    fn runs(&self) -> &[TokenRun] {
        match self {
            SlotLayout::List { rows, .. } | SlotLayout::Ring { rows, .. } => rows,
            SlotLayout::Fixed { .. } => &[],
        }
    }
}

/// A working set's bytes at the profile's context, and what of them holds
/// nothing, by cause.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Waste {
    /// What the states hold: list tokens at the context, the rings' slots,
    /// the working set's fixed record and its share of the pool's
    /// copy-on-write records.
    pub state: u64,
    /// The list's last slot past the context, and alignment inside slots.
    pub list_tail: u64,
    /// Fixed pieces' room past the record: the working set's, and its share
    /// of the copy-on-write reserve's.
    pub fixed_pad: u64,
    /// The working set's share of each chunk's room past its last whole slot.
    pub chunk_tail: u64,
    /// The working set's share of the one partly filled chunk each shape
    /// keeps open, half a chunk each, over the profile's concurrency.
    pub open_chunks: u64,
    /// The working set's share of each segment's room past its last chunk.
    pub segment_tail: u64,
    /// Not a working set's: the room past the last whole weight slot of
    /// every chunk the weight slots fill, all of them held.
    pub weight_tail: u64,
}

impl Waste {
    #[must_use]
    pub fn total(&self) -> u64 {
        self.list_tail + self.fixed_pad + self.chunk_tail + self.open_chunks + self.segment_tail
    }

    /// Waste over everything a working set takes, in percent.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn percent(&self) -> f64 {
        let all = self.state + self.total();
        if all == 0 {
            0.0
        } else {
            100.0 * self.total() as f64 / all as f64
        }
    }
}

/// How a model's states pack into chunks on one device.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkRecipe {
    pub chunk_bytes: u64,
    pub slots: Vec<SlotLayout>,
    /// Each fixed row's bytes and cuttable axis.
    pub atoms: BTreeMap<RowId, Atom>,
    /// How the ranks cut each row; per-rank sizes divide by it.
    pub shards: BTreeMap<RowId, Shard>,
    /// Layers of a token slot whose run is under [`MIN_RUN`] because a slot
    /// long enough would not fit a chunk.
    pub short_runs: Vec<RowId>,
    /// The copy-on-write fixed records the pool reserves once, beside every
    /// working set's own: [`cow_sets`] of the budgets planned for.
    pub cow_sets: u32,
    pub waste: Waste,
    pub profile: ChunkProfile,
    /// The weight slot shapes: routed experts and gathered table rows the
    /// pool may hold read-only, one shape each kind of atom.
    pub weights: Vec<WeightSlot>,
    /// How the rs rows lie in the fixed record, whole or cut, and what each
    /// buffers.
    pub rs: RsLayout,
}

/// The rs buffer in a working set's fixed record. Every rs row lies in the
/// same record slots, cut alike: piece `p` of every row (its tokens
/// [`RsLayout::tokens`]) in record slot `slots[p]`, each row at its
/// [`FixedRun::offset`] there; one piece (slot 0, all 16 tokens) where whole.
/// An engine reads a lane's rs buffer through one page table: the lane's
/// pieces of these slots.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RsLayout {
    /// The pieces every rs row is cut into; 1 where whole, 0 where there
    /// are no rs rows.
    pub pieces: u32,
    /// The tokens a piece holds; the last piece holds what is left.
    pub tokens_per_piece: u64,
    /// The record slots the pieces lie in, one a piece.
    pub slots: Vec<u32>,
    /// What each rs row buffers: the state it commits into and its planes,
    /// in the order they lie in the row.
    pub rows: BTreeMap<RowId, RsRow>,
}

impl RsLayout {
    /// Whether the rs rows are cut into pieces.
    #[must_use]
    pub fn is_cut(&self) -> bool {
        self.pieces > 1
    }

    /// The tokens piece `p` holds, of [`UNFOLDED_ROWS`].
    #[must_use]
    pub fn tokens(&self, p: u32) -> Range<u64> {
        let extent = u64::from(UNFOLDED_ROWS);
        let start = (u64::from(p) * self.tokens_per_piece).min(extent);
        start..(start + self.tokens_per_piece).min(extent)
    }
}

impl ChunkRecipe {
    /// The fixed records a working set holds: [`FIXED_SETS_PER_WS`].
    #[must_use]
    pub fn fixed_sets_per_ws(&self) -> u32 {
        FIXED_SETS_PER_WS
    }

    /// The copy-on-write fixed records the pool reserves once.
    #[must_use]
    pub fn cow_sets(&self) -> u32 {
        self.cow_sets
    }

    /// The fixed slots the copy-on-write reserve takes in the pool.
    #[must_use]
    pub fn cow_slots(&self) -> u64 {
        match self.fixed() {
            Some(SlotLayout::Fixed { pieces_per_ws, .. }) => {
                u64::from(self.cow_sets) * u64::from(*pieces_per_ws)
            }
            _ => 0,
        }
    }

    /// The chunks the copy-on-write reserve takes in the pool, counted once.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cow_chunks(&self) -> f64 {
        self.fixed().map_or(0.0, |fixed| {
            self.cow_slots() as f64 / fixed.per_chunk(self.chunk_bytes).max(1) as f64
        })
    }

    #[must_use]
    pub fn list(&self) -> Option<&SlotLayout> {
        self.slots
            .iter()
            .find(|s| matches!(s, SlotLayout::List { .. }))
    }

    pub fn rings(&self) -> impl Iterator<Item = &SlotLayout> {
        self.slots
            .iter()
            .filter(|s| matches!(s, SlotLayout::Ring { .. }))
    }

    #[must_use]
    pub fn fixed(&self) -> Option<&SlotLayout> {
        self.slots
            .iter()
            .find(|s| matches!(s, SlotLayout::Fixed { .. }))
    }

    /// The bytes the weight floor holds: each MoE shape's top-k experts of
    /// two layers, each table group's rows of one fire.
    #[must_use]
    pub fn weight_floor_bytes(&self) -> u64 {
        self.weights.iter().map(WeightSlot::floor_bytes).sum()
    }

    /// The chunks the weight floor takes.
    #[must_use]
    pub fn weight_floor_chunks(&self) -> u64 {
        self.weights
            .iter()
            .map(|w| w.floor_chunks(self.chunk_bytes))
            .sum()
    }

    /// The chunks every streamable weight takes, held whole.
    #[must_use]
    pub fn weight_chunks(&self) -> u64 {
        self.weights
            .iter()
            .map(|w| w.chunks(self.chunk_bytes))
            .sum()
    }

    /// The weight shapes in a line ([`WeightSlot::shape`]).
    #[must_use]
    pub fn weight_shapes(&self) -> String {
        self.weights
            .iter()
            .map(WeightSlot::shape)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The chunks a working set takes at the profile's context, its share of
    /// chunks it shares with others counted in part; the copy-on-write
    /// reserve is the pool's ([`ChunkRecipe::cow_chunks`]), not counted here.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn chunks_per_ws(&self) -> f64 {
        self.slots
            .iter()
            .map(|s| {
                s.slots_per_ws(self.profile.context) as f64
                    / s.per_chunk(self.chunk_bytes).max(1) as f64
            })
            .sum()
    }

    /// The recipe as one of `tp` ranks holds it: every row cut between the
    /// ranks shrinks by its parts, each shape keeps its slots per chunk, so
    /// every rank holds as many chunks; the chunk is the largest any shape's
    /// slots then take.
    #[must_use]
    pub fn per_rank(&self, tp: u32) -> RankRecipe {
        let align = self.profile.align();
        let parts = |row: &RowId| self.shards.get(row).map_or(1, |shard| parts(shard, tp));
        let mut slots = Vec::with_capacity(self.slots.len());
        let mut fixed = Vec::new();
        for shape in &self.slots {
            let bytes = match shape {
                SlotLayout::List { tokens, rows, .. } | SlotLayout::Ring { tokens, rows, .. } => {
                    align_up(
                        rows.iter()
                            .map(|r| {
                                align_up(
                                    u64::from(*tokens) * r.per_token.div_ceil(parts(&r.row)),
                                    align,
                                )
                            })
                            .sum(),
                        align,
                    )
                }
                SlotLayout::Fixed { rows, .. } => {
                    let (used, runs) = self.rank_runs(rows, tp);
                    fixed = runs;
                    align_up(used.max(FIXED_SLOT_FLOOR), align)
                }
            };
            slots.push((bytes, shape.per_chunk(self.chunk_bytes)));
        }
        let chunk = slots
            .iter()
            .map(|(bytes, per)| bytes * per)
            .max()
            .unwrap_or(self.chunk_bytes / u64::from(tp.max(1)));
        RankRecipe {
            tp,
            chunk_bytes: align_up(chunk, align),
            slots,
            fixed,
        }
    }

    /// The fixed `rows` on one of `tp` ranks: each keeps its pieces and
    /// slots, its pieces shrink by the parts the ranks cut it into (atoms
    /// where they cut its atom axis, an atom's bytes where they cut another),
    /// and the offsets close up, a cut row's still common to its slots. The
    /// most bytes a slot then holds, and the runs.
    fn rank_runs(&self, rows: &[FixedRun], tp: u32) -> (u64, Vec<RankRun>) {
        let align = self.profile.align();
        let mut order: Vec<(usize, &FixedRun)> = rows.iter().enumerate().collect();
        order.sort_by_key(|(_, r)| (r.offset, r.slots.first().copied()));
        let mut used: BTreeMap<u32, u64> = BTreeMap::new();
        let mut runs = vec![None; rows.len()];
        for (at, run) in order {
            let atom = self.atoms[&run.row];
            let shard = self.shards.get(&run.row);
            let parts = shard.map_or(1, |shard| parts(shard, tp));
            let on_atoms = parts > 1
                && match shard {
                    Some(Shard::Cut { axis, .. }) => *axis == atom.axis,
                    Some(Shard::Inferred) => atom.axis == 0,
                    _ => false,
                }
                && run.atoms_per_piece % parts == 0;
            let (atoms_per_piece, atom_bytes) = if on_atoms {
                (run.atoms_per_piece / parts, atom.bytes)
            } else {
                (run.atoms_per_piece, atom.bytes.div_ceil(parts))
            };
            let piece_bytes = atoms_per_piece * atom_bytes;
            let offset = run
                .slots
                .iter()
                .map(|s| align_up(used.get(s).copied().unwrap_or(0), align))
                .max()
                .unwrap_or(0);
            for s in &run.slots {
                used.insert(*s, offset + piece_bytes);
            }
            runs[at] = Some(RankRun {
                row: run.row,
                offset,
                atoms_per_piece,
                piece_bytes,
            });
        }
        (
            used.values().copied().max().unwrap_or(0),
            runs.into_iter().flatten().collect(),
        )
    }

    /// The shapes in a line: `L16x1.25M` (list: tokens x slot), `R1024:16x12.5M*67`
    /// (ring: window: tokens x slot * slots a working set), `F3x16M/47.3M`
    /// (fixed: pieces x slot / record).
    #[must_use]
    pub fn shapes(&self) -> String {
        self.slots
            .iter()
            .map(|s| match s {
                SlotLayout::List { tokens, bytes, .. } => format!("L{tokens}x{}", human(*bytes)),
                SlotLayout::Ring {
                    tokens,
                    bytes,
                    cap_slots,
                    cap,
                    ..
                } => format!("R{}:{tokens}x{}*{cap_slots}", cap.window, human(*bytes)),
                SlotLayout::Fixed {
                    bytes,
                    record,
                    pieces_per_ws,
                    ..
                } => format!("F{pieces_per_ws}x{}/{}", human(*bytes), human(*record)),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// [`ChunkRecipe`] on one of `tp` ranks: each shape's slot bytes and slots a
/// chunk, as the unsplit recipe's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankRecipe {
    pub tp: u32,
    pub chunk_bytes: u64,
    pub slots: Vec<(u64, u64)>,
    /// Each fixed row's run on the rank, in the recipe's row order: the
    /// unsplit run's pieces and slots, at these offsets and sizes.
    pub fixed: Vec<RankRun>,
}

/// A fixed row's run on one rank ([`ChunkRecipe::per_rank`]): its pieces
/// and slots are the unsplit [`FixedRun`]'s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankRun {
    pub row: RowId,
    pub offset: u64,
    pub atoms_per_piece: u64,
    pub piece_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChunkError {
    #[error("cache `{row}` is stored as {dtype:?}, which has no element size")]
    Unsized { row: String, dtype: poem_ir::Dtype },
    #[error("a deferred-commit recurrence replays a plane of no constant width")]
    UnsizedRs,
    #[error("{row:?} is {bytes} bytes that no piece may cut, past the {most}-byte binding")]
    Uncuttable { row: RowId, bytes: u64, most: u64 },
    #[error("an expert of {bytes} bytes fits no chunk of at most {most} bytes, whole or cut")]
    Expert { bytes: u64, most: u64 },
}

/// Decides how `trace`'s states pack into chunks on the device `profile`
/// describes: a pure function of the unsplit trace.
pub fn plan(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Result<ChunkRecipe, ChunkError> {
    let device = profile.chunk;
    let align = device.align();
    let (lo, hi) = device.range();
    let census = census::census(trace, device.state_f32)?;
    let cow = cow_sets(budgets);
    let weight_groups = weights::census(trace, budgets.tokens.max_tokens);
    // Where experts are not cut, no chunk is smaller than one.
    let expert = weight_groups
        .iter()
        .filter(|g| g.is_expert())
        .map(|g| g.whole(align))
        .max()
        .filter(|_| !device.cut_experts);
    if let Some(bytes) = expert
        && bytes > device.most()
    {
        return Err(ChunkError::Expert {
            bytes,
            most: device.most(),
        });
    }

    let draft = crate::arena::readouts_per_lane(trace)
        .min(budgets.tokens.readouts_per_lane.max(1))
        .saturating_sub(1);
    let fire_rows = UNFOLDED_ROWS.min(budgets.tokens.max_tokens.max(1));

    let mut shards: BTreeMap<RowId, Shard> = BTreeMap::new();
    let mut short_runs = Vec::new();
    let mut token_shapes = Vec::new();
    let mut fixed_rows: Vec<FixedRow> = census.fixed;

    // One list shape: every full-retention row, of every space.
    let full: Vec<&TokenRow> = census
        .tokens
        .iter()
        .filter(|r| r.window.is_none())
        .collect();
    if !full.is_empty() {
        let base = full
            .iter()
            .fold(TOKEN_GRAIN, |base, r| census::lcm(base, r.ratio));
        let (tokens, bytes, rows, short) = token_layout(&full, base, device.most(), align);
        short_runs.extend(short);
        token_shapes.push(SlotLayout::List {
            tokens,
            bytes,
            rows,
        });
    }

    // A ring shape a window, unless its whole cap is smaller than a piece.
    let mut windows: BTreeMap<u32, Vec<&TokenRow>> = BTreeMap::new();
    for row in census.tokens.iter().filter(|r| r.window.is_some()) {
        windows
            .entry(row.window.unwrap_or(0))
            .or_default()
            .push(row);
    }
    let mut folded = Vec::new();
    for (window, rows) in windows {
        let cap = RingCap {
            window,
            draft,
            fire_rows,
        };
        let per_token: u64 = rows.iter().map(|r| r.per_token()).sum();
        if u64::from(cap.tokens()) * per_token <= MIN_PIECE {
            for row in rows {
                let extent = u64::from(cap.tokens());
                folded.push((row.id, cap.tokens()));
                fixed_rows.push(FixedRow {
                    id: row.id,
                    layer: row.layer,
                    bytes: extent * row.per_token(),
                    atom: Atom {
                        contract: Contract::Tokens,
                        axis: 0,
                        extent,
                        bytes: row.per_token(),
                        grain: 1,
                    },
                    shard: row.shard.clone(),
                    ring_tokens: Some(cap.tokens()),
                });
            }
            continue;
        }
        let base = rows
            .iter()
            .fold(TOKEN_GRAIN, |base, r| census::lcm(base, r.ratio));
        let (tokens, bytes, runs, short) = token_layout(&rows, base, device.most(), align);
        short_runs.extend(short);
        token_shapes.push(SlotLayout::Ring {
            tokens,
            bytes,
            cap_slots: cap.slots(tokens),
            cap,
            rows: runs,
        });
    }
    for row in &census.tokens {
        shards.insert(row.id, row.shard.clone());
    }

    fixed_rows.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    let mut atoms = BTreeMap::new();
    for row in &fixed_rows {
        atoms.insert(row.id, row.atom);
        shards.insert(row.id, row.shard.clone());
    }
    let record = fixed_rows
        .iter()
        .fold(0, |at, r| align_up(at, align) + r.bytes);
    // The rs rows lie together: whole, or cut alike on their tokens where
    // that wastes less and they are no smaller than a piece may be.
    let rs: Vec<&FixedRow> = fixed_rows.iter().filter(|r| fixed::is_rs(r)).collect();
    let rs_bytes = fixed::rs_piece(&rs, 1, align).unwrap_or(0);
    let rs_cut = rs_bytes >= MIN_PIECE;
    let rs_whole = rs
        .first()
        .filter(|_| !rs_cut)
        .map(|first| (first.id, rs_bytes));
    let whole = fixed_rows
        .iter()
        .filter(|r| !fixed::is_rs(r) && !fixed::cuttable(r))
        .map(|r| (r.id, r.bytes))
        .chain(rs_whole)
        .max_by_key(|(_, bytes)| *bytes);

    // The chunk sizes weighed: multiples of the list slot (or of the
    // alignment where there is none), none smaller than a token slot, an
    // uncuttable row, or a token of every rs row (the least share of them a
    // slot holds).
    let unit = token_shapes
        .first()
        .filter(|s| matches!(s, SlotLayout::List { .. }))
        .map_or(align_up(64 << 10, align), SlotLayout::bytes);
    let least = token_shapes
        .iter()
        .map(SlotLayout::bytes)
        .chain(whole.map(|(_, bytes)| bytes))
        .chain(fixed::rs_piece(&rs, u64::from(UNFOLDED_ROWS), align))
        .chain(expert)
        .fold(lo, u64::max);
    if let Some((row, bytes)) = whole
        && bytes > device.max_binding_bytes
    {
        return Err(ChunkError::Uncuttable {
            row,
            bytes,
            most: device.max_binding_bytes,
        });
    }
    let first = least.div_ceil(unit).max(1);
    // A chunk holds a few slots of a token of every rs row cut, where that
    // is more than the device's range allows.
    let rs_share = fixed::rs_piece(&rs, u64::from(UNFOLDED_ROWS), align)
        .filter(|_| rs_cut)
        .unwrap_or(0);
    let hi = hi.max(RS_SHARES * rs_share).min(device.most());
    let last = (hi.max(least) / unit).max(first);
    let step = (last - first + 1).div_ceil(MOST_CANDIDATES).max(1);
    let candidates: Vec<u64> = (first..=last)
        .step_by(usize::try_from(step).unwrap_or(1))
        .chain(core::iter::once(last))
        .map(|m| m * unit)
        .collect();

    // Weight slots first: of the sizes whose chunks the weights fill within
    // a point of the best, the one the states waste least in.
    let mut weighed = Vec::new();
    for chunk in candidates {
        let pick = if fixed_rows.is_empty() {
            None
        } else {
            match pick_fixed(&fixed_rows, record, chunk, align, rs_cut) {
                Some(pick) => Some(pick),
                None => continue,
            }
        };
        let Some(slots) = weights::layout(&weight_groups, chunk, align, device.cut_experts) else {
            continue;
        };
        weighed.push((chunk, pick, weights::tail(&slots, chunk)));
    }
    let fill = weighed
        .iter()
        .map(|(_, _, tail)| *tail)
        .fold(f64::INFINITY, f64::min);
    let mut best: Option<(u64, Waste, Option<FixedPick>)> = None;
    for (chunk, pick, tail) in weighed {
        if tail > fill + WEIGHT_TAIL_SLACK {
            continue;
        }
        let waste = weigh(&token_shapes, pick.as_ref(), record, cow, chunk, &device);
        let better = match &best {
            None => true,
            Some((was, worse, _)) => {
                let slack = (waste.state / 10_000).max(4 << 10);
                let (now, then) = (waste.total(), worse.total());
                now + slack < then
                    || (now <= then + slack
                        && distance(chunk, device.target_chunk_bytes)
                            < distance(*was, device.target_chunk_bytes))
            }
        };
        if better {
            best = Some((chunk, waste, pick));
        }
    }
    let (chunk_bytes, mut waste, pick) = best.unwrap_or((least, Waste::default(), None));
    let weights = weights::layout(&weight_groups, chunk_bytes, align, device.cut_experts)
        .ok_or_else(|| ChunkError::Expert {
            bytes: weight_groups
                .iter()
                .map(|g| g.whole(align))
                .max()
                .unwrap_or(0),
            most: chunk_bytes,
        })?;
    waste.weight_tail = weights
        .iter()
        .map(|w| w.chunks(chunk_bytes) * chunk_bytes - w.total_bytes())
        .sum();

    let mut slots = token_shapes;
    let mut rs = RsLayout {
        rows: census.rs,
        ..RsLayout::default()
    };
    if let Some(pick) = pick {
        let (pieces_per_ws, rows) =
            fixed::pack(&fixed_rows, pick.packing(align, rs_cut), true).unwrap_or((0, Vec::new()));
        if let Some(run) = rows.iter().find(|r| matches!(r.row, RowId::Rs(_))) {
            rs.pieces = run.pieces;
            rs.tokens_per_piece = run.atoms_per_piece;
            rs.slots.clone_from(&run.slots);
        }
        slots.push(SlotLayout::Fixed {
            bytes: pick.slot,
            record,
            pieces_per_ws,
            rows,
            folded,
        });
    }
    Ok(ChunkRecipe {
        chunk_bytes,
        slots,
        atoms,
        shards,
        short_runs,
        cow_sets: cow,
        waste,
        profile: device,
        weights,
        rs,
    })
}

/// A token slot of `rows`, layer-major: the fewest tokens, a multiple of
/// `base`, at which every layer's rows run [`MIN_RUN`] bytes, unless the slot
/// would outgrow `most`; then the most tokens that fit, and the layers left
/// short.
fn token_layout(
    rows: &[&TokenRow],
    base: u32,
    most: u64,
    align: u64,
) -> (u32, u64, Vec<TokenRun>, Vec<RowId>) {
    let mut rows: Vec<&TokenRow> = rows.to_vec();
    rows.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    // A layer's bytes a token; a row no layer reads is its own.
    let mut layers: BTreeMap<(Option<u32>, Option<RowId>), (u64, RowId)> = BTreeMap::new();
    for row in &rows {
        let key = (row.layer, row.layer.is_none().then_some(row.id));
        let entry = layers.entry(key).or_insert((0, row.id));
        entry.0 += row.per_token();
    }
    let thinnest = layers.values().map(|(b, _)| *b).min().unwrap_or(1).max(1);
    let slot = |tokens: u32| -> u64 {
        align_up(
            rows.iter()
                .map(|r| align_up(u64::from(tokens) * r.per_token(), align))
                .sum(),
            align,
        )
    };
    let base = base.max(1);
    let need = u32::try_from(MIN_RUN.div_ceil(thinnest))
        .unwrap_or(u32::MAX)
        .div_ceil(base)
        .max(1)
        * base;
    let mut tokens = need;
    if slot(tokens) > most {
        tokens = base;
        while tokens + base < need && slot(tokens + base) <= most {
            tokens += base;
        }
    }
    let mut at = 0u64;
    let runs = rows
        .iter()
        .map(|r| {
            let offset = at;
            at = align_up(at + u64::from(tokens) * r.per_token(), align);
            TokenRun {
                row: r.id,
                layer: r.layer,
                offset,
                per_token: r.per_token(),
            }
        })
        .collect();
    let short = layers
        .values()
        .filter(|(bytes, _)| u64::from(tokens) * bytes < MIN_RUN)
        .map(|(_, first)| *first)
        .collect();
    (tokens, slot(tokens), runs, short)
}

/// A fixed slot size and the pieces a working set's record takes in it.
#[derive(Debug, Clone, Copy)]
struct FixedPick {
    slot: u64,
    pieces: u32,
    /// The most bytes a cut piece of a row other than the rs rows holds.
    cap: u64,
    /// The least bytes a cut piece holds.
    least: u64,
}

impl FixedPick {
    fn packing(&self, align: u64, rs_cut: bool) -> fixed::Packing {
        fixed::Packing {
            slot: self.slot,
            cap: self.cap,
            align,
            least: self.least,
            rs_cut,
        }
    }
}

/// The fixed slot for chunks of `chunk` bytes: the record whole where it
/// fits; else, of the piece sizes that fill a chunk (`chunk / k`), end a slot
/// on a row boundary, or hold `k` equal pieces of a cut row, no smaller than
/// [`MIN_PIECE`], the one that wastes least in the slots (padding counted)
/// and in the chunks they fill, the larger on a near tie. Cut pieces hold
/// [`MIN_PIECE`] at least where any size lets them, else as few as a size
/// takes. A cut piece fills its slot, or the room the rs rows leave in
/// theirs, whichever wastes less.
fn pick_fixed(
    rows: &[FixedRow],
    record: u64,
    chunk: u64,
    align: u64,
    rs_cut: bool,
) -> Option<FixedPick> {
    if record <= chunk {
        let slot = align_up(record.max(FIXED_SLOT_FLOOR), align);
        return Some(FixedPick {
            slot,
            pieces: 1,
            cap: slot,
            least: 0,
        });
    }
    let least = MIN_PIECE.min(chunk);
    let mut sizes: Vec<u64> = (1..=(chunk / least).max(1))
        .map(|k| align_down(chunk / k, align))
        .collect();
    // Slots ending on a row boundary, the rs rows first as they pack.
    let rs: Vec<&FixedRow> = rows.iter().filter(|r| fixed::is_rs(r)).collect();
    let mut at = 0;
    for row in rs
        .iter()
        .copied()
        .chain(rows.iter().filter(|r| !fixed::is_rs(r)))
    {
        at = align_up(at, align) + row.bytes;
        sizes.push(align_up(at, align));
    }
    // A slot of `k` pieces of the rs rows cut `n` ways together.
    if rs_cut {
        for n in 2..=u64::from(UNFOLDED_ROWS) {
            if let Some(piece) = fixed::rs_piece(&rs, n, align) {
                sizes.extend((1..=chunk / piece).map(|k| k * piece));
            }
        }
    }
    // A slot of `k` equal pieces of a row cut `n` ways.
    let mut atoms: Vec<Atom> = rows
        .iter()
        .filter(|r| !fixed::is_rs(r) && fixed::cuttable(r) && r.bytes > least)
        .map(|r| r.atom)
        .collect();
    atoms.sort_by_key(|a| (a.extent, a.bytes, a.grain));
    atoms.dedup();
    for atom in atoms {
        let units = atom.extent.div_ceil(atom.grain);
        for n in 2..=units.min(MOST_CUTS) {
            let piece = align_up(units.div_ceil(n) * atom.grain * atom.bytes, align);
            if piece < least {
                break;
            }
            sizes.extend((1..=chunk / piece).map(|k| k * piece));
        }
    }
    sizes.retain(|&slot| slot >= least && slot <= chunk);
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes.dedup();
    let near = record / 1000;
    for least in [least, 0] {
        let mut best: Option<(u64, FixedPick)> = None;
        for &slot in &sizes {
            let room =
                fixed::rs_room(rows, slot, align, rs_cut).filter(|&r| r >= least && r < slot);
            for cap in core::iter::once(slot).chain(room) {
                let pick = FixedPick {
                    slot,
                    pieces: 0,
                    cap,
                    least,
                };
                let Some((pieces, _)) = fixed::pack(rows, pick.packing(align, rs_cut), false)
                else {
                    continue;
                };
                let per = chunk / slot;
                let held = u64::from(pieces) * slot;
                let cost =
                    held.saturating_sub(record) + u64::from(pieces) * (chunk - per * slot) / per;
                if best.as_ref().is_none_or(|(was, _)| cost + near < *was) {
                    best = Some((cost, FixedPick { pieces, ..pick }));
                }
            }
        }
        if best.is_some() {
            return best.map(|(_, pick)| pick);
        }
    }
    None
}

/// What a working set takes at the profile's context with chunks of `chunk`
/// bytes, and what of that holds nothing: its own fixed record, and its share
/// of the pool's `cow` copy-on-write records over the profile's concurrency.
fn weigh(
    tokens: &[SlotLayout],
    fixed: Option<&FixedPick>,
    record: u64,
    cow: u32,
    chunk: u64,
    device: &ChunkProfile,
) -> Waste {
    let mut waste = Waste::default();
    let context = device.context;
    let tail = |slots: u64, slot: u64| {
        let per = (chunk / slot.max(1)).max(1);
        slots * (chunk - per * slot) / per
    };
    let mut shapes = 0u64;
    for shape in tokens {
        shapes += 1;
        let slots = shape.slots_per_ws(context);
        let per_token: u64 = shape.runs().iter().map(|r| r.per_token).sum();
        match shape {
            SlotLayout::List { .. } => {
                let held = u64::from(context) * per_token;
                waste.state += held;
                waste.list_tail += slots * shape.bytes() - held;
            }
            _ => waste.state += slots * shape.bytes(),
        }
        waste.chunk_tail += tail(slots, shape.bytes());
    }
    if let Some(pick) = fixed {
        shapes += 1;
        // The pool's fixed records: a set a working set, and the reserve once.
        let working = u64::from(device.concurrency.max(1));
        let sets = working * u64::from(FIXED_SETS_PER_WS) + u64::from(cow);
        let held = u64::from(pick.pieces) * pick.slot;
        waste.state += sets * record / working;
        waste.fixed_pad += sets * held.saturating_sub(record) / working;
        waste.chunk_tail += tail(sets * u64::from(pick.pieces), pick.slot) / working;
    }
    waste.open_chunks = shapes * chunk / 2 / u64::from(device.concurrency.max(1));
    let (segment, chunks) = device.segment(chunk);
    waste.segment_tail = (segment - chunks * chunk) * (waste.state + waste.total()) / segment;
    waste
}

/// The parts `tp` ranks cut a row into.
fn parts(shard: &Shard, tp: u32) -> u64 {
    let world = u64::from(tp.max(1));
    match shard {
        Shard::Replicated => 1,
        Shard::Inferred => world,
        Shard::Cut { heads, .. } => heads
            .iter()
            .map(|h| Shard::parts(Some(*h), world).unwrap_or(1))
            .min()
            .unwrap_or(world),
    }
}

#[allow(clippy::cast_precision_loss)]
fn distance(chunk: u64, target: u64) -> f64 {
    (chunk as f64 / target.max(1) as f64).ln().abs()
}

#[must_use]
pub fn align_up(bytes: u64, align: u64) -> u64 {
    bytes.div_ceil(align.max(1)) * align.max(1)
}

#[must_use]
pub fn align_down(bytes: u64, align: u64) -> u64 {
    bytes / align.max(1) * align.max(1)
}

/// Bytes in the shortest of B, K, M, G with up to one decimal.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn human(bytes: u64) -> String {
    let units = [(1u64 << 30, "G"), (1 << 20, "M"), (1 << 10, "K")];
    for (unit, name) in units {
        if bytes >= unit {
            let value = bytes as f64 / unit as f64;
            return if (value - value.round()).abs() < 0.05 {
                format!("{value:.0}{name}")
            } else {
                format!("{value:.1}{name}")
            };
        }
    }
    format!("{bytes}B")
}

#[cfg(test)]
mod tests;
