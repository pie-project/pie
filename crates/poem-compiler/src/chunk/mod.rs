//! How a model's states pack into chunks: a pure pass over a trace that
//! picks the chunk size `C` for a device and lays each kind of state into a
//! slot shape a chunk holds many of.
//!
//! Three slot shapes. A **list** slot holds `T` tokens of every
//! full-retention kv row, layer-major (each row a contiguous run of
//! `T x row bytes`). A **window list** is a list of the window rows sharing
//! one window, one a distinct window, its slots of `T_w` tokens of its own
//! (a power of two no larger than the kv page `T`, chosen by waste): a
//! working set holds slots in token order like the kv list and drops each
//! once it lies wholly behind the window, so it holds about
//! `ceil(window / T_w) + 1` of them. An **rs list** slot holds `T_rs` tokens
//! of the rs buffer (a deferred-commit recurrence's unfolded inputs, an
//! n-gram id window's unfolded ids), every row's planes, layer-major like
//! the list ([`RsLayout`]); a lane holds as many as its unfolded tokens
//! fill, none when everything is folded. A **fixed** slot holds a working
//! set's record of every fixed state (recurrent and conv states, frame
//! caches, n-gram ids), whole when it fits
//! in a chunk, else in slots of a piece's size: a row no larger stays whole
//! beside others, a row larger is cut on the atoms its readers let a piece
//! cut into equal pieces, each at one offset in its own slot ([`FixedRun`]).
//!
//! Beside the states, a **weight** slot holds read-only weights a device may
//! stream ([`weights`]): a routed expert of an MoE layer, or a block of
//! token-id-gathered table rows; the chunk size weighs how whole experts
//! fill it too.
//!
//! The pass reads the IR's state rows ([`poem_ir::CacheRow`]): windows,
//! planes and slabs from the rows, layers, compression ratios, atoms and the
//! rs buffer from the ops that read them.

pub mod census;
mod fixed;
pub mod profile;
pub mod weights;

use std::collections::BTreeMap;
use std::ops::Range;

use poem_ir::{Shard, Trace};

pub use census::{Atom, Contract, MOST_RANKS, RsPlane, RsRow};
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

/// The least bytes a fixed slot takes: a record of a few bytes shares a chunk
/// with a thousand-odd others.
pub const FIXED_SLOT_FLOOR: u64 = 1 << 10;

/// How much more of the chunks the weight slots fill may hold nothing than
/// at the best-filling size, for the states to choose among the sizes.
const WEIGHT_TAIL_SLACK: f64 = 0.01;

/// The most tokens an rs-list slot holds; it holds a power of two of them.
pub const MOST_RS_TOKENS: u32 = 16;

/// The unfolded tokens a working set is taken to carry when the plan weighs
/// the rs list's last slot (a fire's few speculative rows before they
/// fold). A weight only: the rs list holds as many as a lane leaves
/// unfolded.
pub const EXPECTED_UNFOLDED: u32 = 4;

/// The rs-list slots of one token the largest chunk weighed holds, where
/// one token of every rs row outgrows the device's range.
const RS_SHARES: u64 = 4;

/// The most ways the fixed slot sizes weighed cut a row.
const MOST_CUTS: u64 = 64;

/// The most chunk sizes the plan weighs.
const MOST_CANDIDATES: u64 = 128;

/// A row a slot holds: a cache row of the trace, or a row of the rs list:
/// its ordinal among the deferred-commit recurrences in node order, then
/// among the n-gram id windows ([`ChunkRecipe::rs`] says which).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RowId {
    Cache(u32),
    Rs(u32),
    /// The compressor's state for the kv row a pool gather keys by its
    /// pages (that cache row): a kv row's projection and its score, each
    /// kept a token until its group closes.
    Pool(u32),
    /// The windowed layers' ring region ([`WindowedRecipe`]): one a record,
    /// laid inside as the engine's ring hook says ([`RingLayout`]).
    Ring,
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
    /// `tokens` tokens of each row, layer-major. With a `window`, a window
    /// list: its rows retain `window` tokens, and a working set's slots
    /// wholly behind them are dropped (the table keeps its token-order
    /// length, a dropped slot the null ref); `tokens` is its own `T_w`.
    List {
        tokens: u32,
        bytes: u64,
        rows: Vec<TokenRun>,
        window: Option<u32>,
    },
    /// The rs buffer's rows, `tokens` (`T_rs`) of each, layer-major:
    /// [`ChunkRecipe::rs`] says how a buffered token lies.
    RsList {
        tokens: u32,
        bytes: u64,
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
    },
}

impl SlotLayout {
    /// A slot's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        match self {
            SlotLayout::List { bytes, .. }
            | SlotLayout::RsList { bytes, .. }
            | SlotLayout::Fixed { bytes, .. } => *bytes,
        }
    }

    /// The slots of this shape a working set holds at `context` tokens: of
    /// a window list, no more than `ceil(window / T) + 1` (the slot the
    /// oldest row retires from stays while the newest fills); of the rs
    /// list, at [`EXPECTED_UNFOLDED`] tokens.
    #[must_use]
    pub fn slots_per_ws(&self, context: u32) -> u64 {
        match self {
            SlotLayout::List { tokens, window, .. } => {
                let tokens = (*tokens).max(1);
                let all = context.div_ceil(tokens);
                u64::from(window.map_or(all, |w| all.min(w.div_ceil(tokens) + 1)))
            }
            SlotLayout::RsList { tokens, .. } => {
                u64::from(EXPECTED_UNFOLDED.div_ceil((*tokens).max(1)))
            }
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

    /// The rows a window list retains; `None` for every other shape.
    #[must_use]
    pub fn window(&self) -> Option<u32> {
        match self {
            SlotLayout::List { window, .. } => *window,
            _ => None,
        }
    }

    fn runs(&self) -> &[TokenRun] {
        match self {
            SlotLayout::List { rows, .. } | SlotLayout::RsList { rows, .. } => rows,
            SlotLayout::Fixed { .. } => &[],
        }
    }
}

/// A working set's bytes at the profile's context, and what of them holds
/// nothing, by cause.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Waste {
    /// What the states hold: kv list tokens at the context, each window
    /// list's tokens at the context or its window, rs-list tokens at
    /// [`EXPECTED_UNFOLDED`], the working set's fixed record and its share
    /// of the pool's copy-on-write records.
    pub state: u64,
    /// The kv and window lists' slots past the tokens they hold (the last
    /// slot's room, a window list's slot the oldest row retires from), and
    /// alignment inside slots.
    pub list_tail: u64,
    /// The rs list's last slot past [`EXPECTED_UNFOLDED`] tokens, and
    /// alignment inside slots.
    pub rs_tail: u64,
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
        self.list_tail
            + self.rs_tail
            + self.fixed_pad
            + self.chunk_tail
            + self.open_chunks
            + self.segment_tail
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
    pub waste: Waste,
    pub profile: ChunkProfile,
    /// The weight slot shapes: routed experts and gathered table rows the
    /// pool may hold read-only, one shape each kind of atom.
    pub weights: Vec<WeightSlot>,
    /// How the rs list's slot lays its rows, and what each buffers.
    pub rs: RsLayout,
    /// Windowed layers as ring + buffer (CONTRACT-BC 19, 21), where the
    /// engine asked for them ([`plan_windowed`]); `None` from [`plan`],
    /// whose window lists hold them.
    pub windowed: Option<WindowedRecipe>,
}

/// A windowed (sliding-window attention) layer as the ring + buffer recipe
/// states it to the engine's ring hook ([`RingLayout`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowedRow {
    /// The model's layer; `None` for a row no layer reads (its own).
    pub layer: Option<u32>,
    /// The layer's kv cache rows, in the order the buffer slot lays them.
    pub rows: Vec<RowId>,
    /// W: the rows the layer attends back over, its own included.
    pub window: u32,
    /// The layer's K and V bytes of one row, every row's planes, unsplit.
    pub per_token: u64,
    /// How the ranks cut the layer's rows (its kv heads): each rank holds
    /// the layer's ring at its own per-rank bytes
    /// ([`WindowedRecipe::rank_layer_bytes`]).
    pub shard: Shard,
}

/// Where the ring region lies, as the engine's ring hook answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingPlacement {
    /// Inside the state record beside the state rows (the record's first
    /// row), at a multiple of this many bytes into its slot.
    Aligned(u64),
    /// Its own pieces, each starting at a chunk boundary: a fixed shape of
    /// one slot a chunk ([`WindowedRecipe::ring_shape`]), which the engine
    /// states as `SlotRole::Ring` before its state shapes. The record names
    /// the ring's pieces first and its state pieces after them.
    ChunkStart,
}

/// The engine's answer to the ring hook: one ring region a record, whose
/// layout inside is the engine's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingRegion {
    /// The region's bytes: what the record counts for it.
    pub bytes: u64,
    /// Pieces cut the region only between atoms of this many bytes; `bytes`
    /// is a whole number of them (`bytes` itself: never cut).
    pub atom: u64,
    pub placement: RingPlacement,
    /// Each windowed layer's share of `bytes`, in the order of the layers
    /// asked about, summing to `bytes`: the bytes the engine states for the
    /// layer in its facts (`engine::chunk::WindowedLayer::bytes`).
    pub layer_bytes: Vec<u64>,
}

/// What the ring hook is told of the window buffer: its slot of `tokens`
/// rows of every windowed layer, layer-major.
#[derive(Debug, Clone, Copy)]
pub struct WindowBufferSlot<'a> {
    pub tokens: u32,
    pub bytes: u64,
    pub rows: &'a [TokenRun],
}

/// The engine's ring hook: how big its ring region is, and where it lies.
/// Required of an engine that asks for the windowed recipe; an engine that
/// indexes per-layer rings takes [`plain_rings`]' answer.
pub trait RingLayout {
    fn ring_region(&self, layers: &[WindowedRow], buffer: &WindowBufferSlot<'_>) -> RingRegion;
}

/// The plain per-layer form: each layer's ring of `W` rows of its K and V,
/// one after another, each at a multiple of `align`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlainRings {
    pub region: RingRegion,
    /// Each layer's ring's offset inside the region, in `layers` order.
    pub offsets: Vec<u64>,
}

/// The plain per-layer ring region ([`PlainRings`]): placed inside the
/// record at `align`, cut only between whole layers' rings of one size (the
/// largest size every layer's ring is a whole number of).
#[must_use]
pub fn plain_rings(layers: &[WindowedRow], align: u64) -> PlainRings {
    let align = align.max(1);
    let mut offsets = Vec::with_capacity(layers.len());
    let mut layer_bytes = Vec::with_capacity(layers.len());
    let mut at = 0u64;
    for layer in layers {
        let ring = align_up(u64::from(layer.window) * layer.per_token, align);
        offsets.push(at);
        layer_bytes.push(ring);
        at += ring;
    }
    let atom = layer_bytes.iter().copied().fold(0, gcd).max(1);
    PlainRings {
        region: RingRegion {
            bytes: at,
            atom: if at == 0 { 1 } else { atom },
            placement: RingPlacement::Aligned(align),
            layer_bytes,
        },
        offsets,
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The ring + buffer part of a recipe ([`plan_windowed`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowedRecipe {
    /// The windowed layers, in the order the hook was asked about them.
    pub layers: Vec<WindowedRow>,
    /// The engine's answer.
    pub region: RingRegion,
    /// The window buffer's place in [`ChunkRecipe::slots`]: a
    /// `SlotLayout::List { window: None }` of `tokens` rows of every
    /// windowed layer, layer-major, each layer's run in its rows.
    pub buffer: usize,
    /// The ring shape's place in [`ChunkRecipe::slots`] where the region
    /// lies at chunk starts ([`RingPlacement::ChunkStart`]); `None` where
    /// it lies inside the state record.
    pub ring_shape: Option<usize>,
}

impl WindowedRecipe {
    /// The ring's run: in the state record's rows, or the ring shape's.
    #[must_use]
    pub fn ring_run<'a>(&self, recipe: &'a ChunkRecipe) -> Option<&'a FixedRun> {
        let shape = match self.ring_shape {
            Some(at) => recipe.slots.get(at),
            None => recipe.fixed(),
        }?;
        match shape {
            SlotLayout::Fixed { rows, .. } => rows.iter().find(|r| r.row == RowId::Ring),
            _ => None,
        }
    }

    /// The ring shape, where the region lies at chunk starts.
    #[must_use]
    pub fn ring_shape<'a>(&self, recipe: &'a ChunkRecipe) -> Option<&'a SlotLayout> {
        self.ring_shape.and_then(|at| recipe.slots.get(at))
    }

    /// The window buffer's slot shape.
    #[must_use]
    pub fn buffer<'a>(&self, recipe: &'a ChunkRecipe) -> Option<&'a SlotLayout> {
        recipe.slots.get(self.buffer)
    }

    /// Each windowed layer's share of the ring region on one of `tp` ranks:
    /// its share cut by the parts the ranks cut that layer's own rows into
    /// (the fewest, where its rows are cut differently), aligned as the
    /// recipe's rows are.
    #[must_use]
    pub fn rank_layer_bytes(&self, recipe: &ChunkRecipe, tp: u32) -> Vec<u64> {
        let align = recipe.profile.align();
        self.layers
            .iter()
            .zip(&self.region.layer_bytes)
            .map(|(layer, bytes)| {
                let cut = layer
                    .rows
                    .iter()
                    .map(|row| recipe.shards.get(row).map_or(1, |shard| parts(shard, tp)))
                    .min()
                    .unwrap_or(1);
                align_up(bytes.div_ceil(cut.max(1)), align)
            })
            .collect()
    }

    /// Where layer `at`'s rows run in the window buffer's slot: each of its
    /// cache rows' run.
    #[must_use]
    pub fn buffer_runs<'a>(&self, recipe: &'a ChunkRecipe, at: usize) -> Vec<&'a TokenRun> {
        let (Some(layer), Some(shape)) = (self.layers.get(at), self.buffer(recipe)) else {
            return Vec::new();
        };
        shape
            .runs()
            .iter()
            .filter(|run| layer.rows.contains(&run.row))
            .collect()
    }
}

/// The rs list: a lane's unfolded rows of every rs row, `tokens` (`T_rs`)
/// a slot, in slots refcounted like kv pages (forks share their common
/// unfolded prefix). Buffered token `s` of a lane whose page table starts at
/// token `base` (a multiple of `tokens`) lies in its page
/// `(s - base) / tokens`, at token `(s - base) % tokens` of each row's run
/// ([`RsLayout::at`]).
///
/// Inside a slot the rows lie layer-major, each a contiguous run of
/// `tokens x` its bytes a token, planes one after another
/// ([`RsRow::plane_at`]). No run need reach [`MIN_RUN`]: an engine writes and
/// replays a buffered token at a time, a fire's few; only the alignment
/// every run starts at matters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RsLayout {
    /// The tokens a slot holds; 0 where there are no rs rows.
    pub tokens: u32,
    /// A slot's bytes.
    pub bytes: u64,
    /// Each rs row's run in the slot, layer-major (the rs-list shape's).
    pub runs: Vec<TokenRun>,
    /// What each rs row buffers: the state it commits into and its planes,
    /// in the order they lie in the row's run.
    pub rows: BTreeMap<RowId, RsRow>,
}

impl RsLayout {
    /// Whether there is no rs list.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// The run of `row` in a slot.
    #[must_use]
    pub fn run(&self, row: RowId) -> Option<&TokenRun> {
        self.runs.iter().find(|r| r.row == row)
    }

    /// Where buffered token `s` of `row`'s plane `plane` lies, for a page
    /// table starting at token `base`: the page, and the bytes into its slot.
    #[must_use]
    pub fn at(&self, row: RowId, plane: usize, s: u64, base: u64) -> Option<(u64, u64)> {
        let run = self.run(row)?;
        let what = self.rows.get(&row)?;
        let tokens = u64::from(self.tokens.max(1));
        let into = s.checked_sub(base)?;
        Some((
            into / tokens,
            run.offset
                + what.plane_at(plane, tokens)
                + (into % tokens) * what.planes.get(plane)?.bytes,
        ))
    }
}

impl ChunkRecipe {
    /// The kv list shape (never the window buffer).
    #[must_use]
    pub fn list(&self) -> Option<&SlotLayout> {
        let buffer = self.windowed.as_ref().map(|w| w.buffer);
        self.slots
            .iter()
            .enumerate()
            .find(|(at, s)| {
                Some(*at) != buffer && matches!(s, SlotLayout::List { window: None, .. })
            })
            .map(|(_, s)| s)
    }

    /// The window buffer's slot shape, in a windowed recipe.
    #[must_use]
    pub fn window_buffer(&self) -> Option<&SlotLayout> {
        self.windowed.as_ref().and_then(|w| w.buffer(self))
    }

    /// The rs list shape.
    #[must_use]
    pub fn rs_list(&self) -> Option<&SlotLayout> {
        self.slots
            .iter()
            .find(|s| matches!(s, SlotLayout::RsList { .. }))
    }

    /// The window lists, one a distinct window, narrowest first.
    pub fn windows(&self) -> impl Iterator<Item = &SlotLayout> {
        self.slots.iter().filter(|s| {
            matches!(
                s,
                SlotLayout::List {
                    window: Some(_),
                    ..
                }
            )
        })
    }

    /// The state record's fixed shape (never the ring shape).
    #[must_use]
    pub fn fixed(&self) -> Option<&SlotLayout> {
        let ring = self.windowed.as_ref().and_then(|w| w.ring_shape);
        self.slots
            .iter()
            .enumerate()
            .find(|(at, s)| Some(*at) != ring && matches!(s, SlotLayout::Fixed { .. }))
            .map(|(_, s)| s)
    }

    /// The chunks every streamable weight takes, held whole.
    #[must_use]
    pub fn weight_chunks(&self) -> u64 {
        self.weights
            .iter()
            .map(|w| w.chunks(self.chunk_bytes))
            .sum()
    }

    /// The chunks a working set takes at the profile's context, its share of
    /// chunks it shares with others counted in part. A fork's copy-on-write
    /// record is taken from the pool when the fork writes, as a seat's is.
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
                SlotLayout::List { tokens, rows, .. } | SlotLayout::RsList { tokens, rows, .. } => {
                    skew(
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
                    // The ring shape's run is not the state record's.
                    if fixed.is_empty() || rows.iter().any(|r| r.row != RowId::Ring) {
                        fixed = runs;
                    }
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
            let (atoms_per_piece, atom_bytes) = if run.row == RowId::Ring {
                // The ring shrinks layer by layer, each by its own rows'
                // parts: its atoms keep their count, each its share of the
                // ring's bytes on the rank.
                let rank: u64 = self
                    .windowed
                    .as_ref()
                    .map_or(0, |w| w.rank_layer_bytes(self, tp).iter().sum());
                (run.atoms_per_piece, rank.div_ceil(atom.extent.max(1)))
            } else if on_atoms {
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

    /// The shapes in a line: `L16x1.25M` (list: tokens x slot), `rs4x880K`
    /// (rs list: tokens x slot), `W1024:16x12.5M` (window list: window:
    /// tokens x slot), `F3x16M/47.3M` (fixed: pieces x slot / record).
    #[must_use]
    pub fn shapes(&self) -> String {
        self.slots
            .iter()
            .map(|s| match s {
                SlotLayout::List {
                    tokens,
                    bytes,
                    window: None,
                    ..
                } => format!("L{tokens}x{}", human(*bytes)),
                SlotLayout::List {
                    tokens,
                    bytes,
                    window: Some(window),
                    ..
                } => format!("W{window}:{tokens}x{}", human(*bytes)),
                SlotLayout::RsList { tokens, bytes, .. } => {
                    format!("rs{tokens}x{}", human(*bytes))
                }
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
    #[error("the engine's ring region does not hang together: {0}")]
    Ring(String),
}

/// Decides how `trace`'s states pack into chunks on the device `profile`
/// describes: a pure function of the unsplit trace. Windowed rows pack as
/// window lists (removed by the windowed-state change; an engine stating
/// `windowed_state` asks [`plan_windowed`]).
pub fn plan(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
) -> Result<ChunkRecipe, ChunkError> {
    plan_with(trace, budgets, profile, None)
}

/// [`plan`] for an engine that serves windowed layers as ring + buffer
/// (CONTRACT-BC 19, 21): every windowed row's unfolded rows in one window
/// buffer (a list of the kv page's `T` rows of every windowed layer,
/// layer-major), and one ring region a record, sized and placed as `ring`
/// answers ([`RingLayout`]); no window lists. A model with windowed rows
/// gets a record though it holds no other state.
pub fn plan_windowed(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
    ring: &dyn RingLayout,
) -> Result<ChunkRecipe, ChunkError> {
    plan_with(trace, budgets, profile, Some(ring))
}

/// The windowed layers of `rows`, by layer (a row no layer reads its own),
/// ascending.
fn windowed_layers(rows: &[&TokenRow]) -> Vec<WindowedRow> {
    let mut sorted: Vec<&TokenRow> = rows.to_vec();
    sorted.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    let mut out: Vec<WindowedRow> = Vec::new();
    for row in sorted {
        let window = row.window.unwrap_or(0);
        match out.last_mut() {
            Some(last) if row.layer.is_some() && last.layer == row.layer => {
                last.rows.push(row.id);
                last.window = last.window.max(window);
                last.per_token += row.per_token();
            }
            _ => out.push(WindowedRow {
                layer: row.layer,
                rows: vec![row.id],
                window,
                per_token: row.per_token(),
                shard: row.shard.clone(),
            }),
        }
    }
    out
}

/// Checks the engine's ring answer against the layers asked about.
fn check_ring(region: &RingRegion, layers: &[WindowedRow]) -> Result<(), ChunkError> {
    let wrong = |what: String| Err(ChunkError::Ring(what));
    if region.layer_bytes.len() != layers.len() {
        return wrong(format!(
            "{} layer shares for {} windowed layers",
            region.layer_bytes.len(),
            layers.len()
        ));
    }
    let sum: u64 = region.layer_bytes.iter().sum();
    if sum != region.bytes {
        return wrong(format!(
            "layer shares of {sum} bytes for a region of {}",
            region.bytes
        ));
    }
    if let Some((at, _)) = region
        .layer_bytes
        .iter()
        .zip(layers)
        .enumerate()
        .find(|(_, (bytes, layer))| **bytes < u64::from(layer.window) * layer.per_token)
    {
        return wrong(format!("layer {at}'s share holds fewer than its W rows"));
    }
    if region.atom == 0 || !region.bytes.is_multiple_of(region.atom) {
        return wrong(format!(
            "{} bytes are no whole number of {}-byte atoms",
            region.bytes, region.atom
        ));
    }
    if region.placement == RingPlacement::Aligned(0) {
        return wrong("an alignment of 0".into());
    }
    Ok(())
}

fn plan_with(
    trace: &Trace,
    budgets: &Budgets,
    profile: &DeviceProfile,
    ring: Option<&dyn RingLayout>,
) -> Result<ChunkRecipe, ChunkError> {
    let device = profile.chunk;
    let align = device.align();
    let (lo, hi) = device.range();
    let census = census::census(trace, device.state_f32, device.rs_mamba2)?;
    let weight_groups = weights::census(trace, budgets.tokens.max_tokens, device.tma_rows);
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

    let mut shards: BTreeMap<RowId, Shard> = BTreeMap::new();
    let mut short_runs = Vec::new();
    let mut token_shapes = Vec::new();
    let mut fixed_rows: Vec<FixedRow> = census.fixed;

    // The kv list: every full-retention row, `T` tokens a
    // slot (the kv page size).
    // A window that holds a whole working set is none.
    let mut token_rows = census.tokens;
    for row in &mut token_rows {
        if row
            .window
            .is_some_and(|w| device.full_from.is_some_and(|full| w >= full))
        {
            row.window = None;
        }
    }
    if device.one_window
        && ring.is_none()
        && let Some(widest) = token_rows.iter().filter_map(|r| r.window).max()
    {
        for row in token_rows.iter_mut().filter(|r| r.window.is_some()) {
            row.window = Some(widest);
        }
    }
    let full: Vec<&TokenRow> = token_rows.iter().filter(|r| r.window.is_none()).collect();
    // Windowed rows as ring + buffer ([`plan_windowed`]) leave no window
    // list.
    let windowed: Vec<&TokenRow> = token_rows
        .iter()
        .filter(|r| ring.is_some() && r.window.is_some())
        .collect();
    let mut windows: BTreeMap<u32, Vec<&TokenRow>> = BTreeMap::new();
    for row in token_rows.iter().filter(|_| ring.is_none()) {
        if let Some(window) = row.window {
            windows.entry(window).or_default().push(row);
        }
    }
    let base = token_rows
        .iter()
        .fold(TOKEN_GRAIN, |base, r| census::lcm(base, r.ratio));
    // The kv page; with no kv list, the page every window list's rows would
    // run [`MIN_RUN`] at, the most tokens a window slot is weighed at.
    // A backend's own page is taken as it is stated.
    let page = if let Some(stated) = device.token_slot.filter(|&tokens| tokens > 0) {
        stated
    } else if full.is_empty() && !windowed.is_empty() {
        page_tokens(
            core::iter::once(windowed.as_slice()),
            base,
            device.most(),
            align,
        )
    } else if full.is_empty() {
        page_tokens(
            windows.values().map(Vec::as_slice),
            base,
            device.most(),
            align,
        )
    } else {
        page_tokens(
            core::iter::once(full.as_slice()),
            base,
            device.most(),
            align,
        )
    };
    if !full.is_empty() {
        let (bytes, runs, short) = token_layout(&full, page, align);
        short_runs.extend(short);
        token_shapes.push(SlotLayout::List {
            tokens: page,
            bytes,
            rows: runs,
            window: None,
        });
    }
    // A window list a distinct window, narrowest first, each slot `T_w`
    // tokens of its own: the slot sizes weighed are the powers of two from
    // [`TOKEN_GRAIN`] (a multiple of its rows' compression) to the kv page,
    // chosen per chunk by what a trimmed list holds past its window
    // ([`pick_window`]). A short window's runs may stay under
    // [`MIN_RUN`]: a fire reads a window's few slots.
    let window_shapes: Vec<Vec<SlotLayout>> = windows
        .iter()
        .map(|(&window, rows)| {
            let grain = rows
                .iter()
                .fold(TOKEN_GRAIN, |base, r| census::lcm(base, r.ratio));
            // A backend that pages its windowed rows by one table pages them
            // by its stated page; one with a table a window list takes the
            // window's own slot.
            let least = if device.token_slot.is_some() && device.one_window {
                page
            } else {
                grain.min(page)
            };
            core::iter::successors(Some(least), |t| t.checked_mul(2))
                .take_while(|&t| t <= page)
                .chain(core::iter::once(page))
                .filter(|t| t % grain == 0 || *t == page)
                .collect::<std::collections::BTreeSet<u32>>()
                .into_iter()
                .map(|tokens| {
                    let (bytes, runs, _) = token_layout(rows, tokens, align);
                    SlotLayout::List {
                        tokens,
                        bytes,
                        rows: runs,
                        window: Some(window),
                    }
                })
                .collect()
        })
        .collect();
    for row in token_rows.iter().chain(&census.rs_rows) {
        shards.insert(row.id, row.shard.clone());
    }

    // The window buffer, `T` (the kv page's) rows of every windowed layer,
    // and the engine's ring region beside the windowed layers.
    let windowing = match ring {
        Some(hook) if !windowed.is_empty() => {
            let (bytes, runs, _) = token_layout(&windowed, page, align);
            let layers = windowed_layers(&windowed);
            let region = hook.ring_region(
                &layers,
                &WindowBufferSlot {
                    tokens: page,
                    bytes,
                    rows: &runs,
                },
            );
            check_ring(&region, &layers)?;
            Some((
                SlotLayout::List {
                    tokens: page,
                    bytes,
                    rows: runs,
                    window: None,
                },
                layers,
                region,
            ))
        }
        _ => None,
    };
    // The ring's atom, as the record (or the ring shape) cuts it.
    let ring_atom = windowing.as_ref().map(|(_, _, region)| Atom {
        contract: census::Contract::Tokens,
        axis: 0,
        extent: region.bytes / region.atom.max(1),
        bytes: region.atom,
        grain: 1,
    });
    let in_record = windowing
        .as_ref()
        .and_then(|(_, _, region)| match region.placement {
            RingPlacement::Aligned(at) => Some((region.bytes, at)),
            RingPlacement::ChunkStart => None,
        });
    // The record's rows align as the ring asks where it lies inside.
    let fixed_align = in_record.map_or(align, |(_, at)| align_up(at.max(align), align));

    // The rs list's slots: a power of two of tokens of every rs row, the
    // least a token of each, which no chunk is smaller than.
    let rs_shapes: Vec<RsShape> = core::iter::successors(Some(1u32), |t| t.checked_mul(2))
        .take_while(|&t| t <= MOST_RS_TOKENS)
        .filter(|_| !census.rs_rows.is_empty())
        .map(|tokens| RsShape::of(&census.rs_rows, tokens, align))
        .collect();
    let rs_token = rs_shapes.first().map(|s| s.bytes);
    if let Some(bytes) = rs_token
        && bytes > device.most()
    {
        return Err(ChunkError::Uncuttable {
            row: census.rs_rows[0].id,
            bytes,
            most: device.most(),
        });
    }

    fixed_rows.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    // The ring inside the record lies first, at its slot's start.
    if let (Some((bytes, _)), Some(atom)) = (in_record, ring_atom)
        && bytes > 0
    {
        fixed_rows.insert(
            0,
            FixedRow {
                id: RowId::Ring,
                layer: None,
                bytes,
                atom,
                shard: Shard::Replicated,
            },
        );
    }
    let mut atoms = BTreeMap::new();
    for row in &fixed_rows {
        atoms.insert(row.id, row.atom);
        shards.insert(row.id, row.shard.clone());
    }
    let record = fixed_rows
        .iter()
        .fold(0, |at, r| align_up(at, fixed_align) + r.bytes);
    let whole = fixed_rows
        .iter()
        .filter(|r| !fixed::cuttable(r))
        .map(|r| (r.id, r.bytes))
        .max_by_key(|(_, bytes)| *bytes);

    // The chunk sizes weighed: multiples of the list slot (or of the
    // alignment where there is none), none smaller than a token slot, an
    // uncuttable row, a token of every rs row, or an expert.
    let unit = token_shapes
        .first()
        .filter(|s| matches!(s, SlotLayout::List { window: None, .. }))
        .map_or(align_up(64 << 10, align), SlotLayout::bytes);
    let least = token_shapes
        .iter()
        .map(SlotLayout::bytes)
        .chain(
            window_shapes
                .iter()
                .filter_map(|c| c.iter().map(SlotLayout::bytes).min()),
        )
        .chain(whole.map(|(_, bytes)| bytes))
        .chain(rs_token)
        .chain(windowing.as_ref().map(|(buffer, ..)| buffer.bytes()))
        .chain(
            windowing
                .as_ref()
                .filter(|(.., region)| region.placement == RingPlacement::ChunkStart)
                .map(|(.., region)| region.atom),
        )
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
    // A chunk holds a few rs-list slots of a token, where that is more than
    // the device's range allows.
    let hi = hi.max(RS_SHARES * rs_token.unwrap_or(0)).min(device.most());
    let last = (hi.max(least) / unit).max(first);
    let step = (last - first + 1).div_ceil(MOST_CANDIDATES).max(1);
    // And the whole multiples of each uncut expert slot in that range: an
    // expert's planes are whole pages of the list plus their scales, so no
    // multiple of the list slot holds whole experts without leaving most of
    // a slot a chunk (4.5% of Qwen3-30B-A3B-FP8's 29 GB on CUDA's range),
    // where a chunk of whole experts leaves the list the scales' share of a
    // page.
    let slot_sizes: std::collections::BTreeSet<u64> = if device.cut_experts {
        std::collections::BTreeSet::new()
    } else {
        weight_groups
            .iter()
            .filter(|g| g.is_expert())
            .map(|g| g.whole(align))
            .filter(|&bytes| bytes > 0)
            .collect()
    };
    let candidates: Vec<u64> =
        (first..=last)
            .step_by(usize::try_from(step).unwrap_or(1))
            .chain(core::iter::once(last))
            .map(|m| m * unit)
            .chain(slot_sizes.iter().flat_map(|&slot| {
                (least.div_ceil(slot)..=hi.max(least) / slot).map(move |n| n * slot)
            }))
            .collect::<std::collections::BTreeSet<u64>>()
            .into_iter()
            .collect();
    // Where slots lay their planes on TMA rows, a chunk a whole number of
    // every expert slot's boundary, so a slot of any chunk lies on them past
    // the pool's base; none such, the sizes as they are (the engine then
    // finds its entries off those rows and maps no window).
    let edge = weight_groups
        .iter()
        .filter(|g| g.is_expert())
        .map(weights::Group::boundary)
        .fold(1, weights::lcm);
    let on_edge: Vec<u64> = candidates
        .iter()
        .copied()
        .filter(|c| c % edge == 0)
        .collect();
    let candidates = if on_edge.is_empty() {
        candidates
    } else {
        on_edge
    };

    // A windowed recipe's window buffer and ring shape (at chunk starts),
    // weighed beside the others; nothing where the recipe has neither.
    let windowed_weighed = windowing.as_ref().map(|(buffer, _, region)| Windowed {
        buffer,
        ring: ring_atom.filter(|_| region.placement == RingPlacement::ChunkStart),
    });
    // Weight slots first: of the sizes whose chunks the weights fill within
    // a point of the best, the one the states (kv list, window lists, rs
    // list and fixed records together) waste least in.
    let mut weighed = Vec::new();
    for chunk in candidates {
        let pick = if fixed_rows.is_empty() {
            None
        } else {
            match pick_fixed(&fixed_rows, record, chunk, fixed_align) {
                Some(pick) => Some(pick),
                None => continue,
            }
        };
        let rs = pick_rs(&rs_shapes, chunk);
        if rs.is_none() && !rs_shapes.is_empty() {
            continue;
        }
        let Some(lists) = window_shapes
            .iter()
            .map(|c| pick_window(c, chunk, device.context))
            .collect::<Option<Vec<SlotLayout>>>()
        else {
            continue;
        };
        let Some(slots) = weights::layout(&weight_groups, chunk, align, device.cut_experts) else {
            continue;
        };
        weighed.push((chunk, pick, rs, lists, weights::tail(&slots, chunk)));
    }
    let fill = weighed
        .iter()
        .map(|(.., tail)| *tail)
        .fold(f64::INFINITY, f64::min);
    type Best = (
        u64,
        Waste,
        Option<FixedPick>,
        Option<RsShape>,
        Vec<SlotLayout>,
    );
    let mut best: Option<Best> = None;
    for (chunk, pick, rs, lists, tail) in weighed {
        if tail > fill + WEIGHT_TAIL_SLACK {
            continue;
        }
        let shapes: Vec<SlotLayout> = token_shapes.iter().chain(&lists).cloned().collect();
        let waste = weigh(
            &shapes,
            rs.as_ref(),
            pick.as_ref(),
            record,
            chunk,
            &device,
            windowed_weighed,
        );
        let better = match &best {
            None => true,
            Some((was, worse, ..)) => {
                let slack = (waste.state / 10_000).max(4 << 10);
                let (now, then) = (waste.total(), worse.total());
                now + slack < then
                    || (now <= then + slack
                        && distance(chunk, device.target_chunk_bytes)
                            < distance(*was, device.target_chunk_bytes))
            }
        };
        if better {
            best = Some((chunk, waste, pick, rs, lists));
        }
    }
    let (chunk_bytes, mut waste, pick, rs_pick, lists) =
        best.unwrap_or((least, Waste::default(), None, None, Vec::new()));
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

    // The shapes, as engines take them: kv list, rs list, window lists,
    // fixed.
    let mut slots = token_shapes;
    let mut rs = RsLayout {
        rows: census.rs,
        ..RsLayout::default()
    };
    if let Some(shape) = rs_pick {
        let runs = rs_runs(&census.rs_rows, shape.tokens, align);
        rs.tokens = shape.tokens;
        rs.bytes = shape.bytes;
        rs.runs.clone_from(&runs);
        let at = usize::from(matches!(
            slots.first(),
            Some(SlotLayout::List { window: None, .. })
        ));
        slots.insert(
            at,
            SlotLayout::RsList {
                tokens: shape.tokens,
                bytes: shape.bytes,
                rows: runs,
            },
        );
    }
    slots.extend(lists);
    let mut windowed_recipe = None;
    if let Some((buffer, layers, region)) = windowing {
        windowed_recipe = Some(WindowedRecipe {
            layers,
            region,
            buffer: slots.len(),
            ring_shape: None,
        });
        slots.push(buffer);
    }
    // A ring at chunk starts: a fixed shape of one slot a chunk, its pieces
    // whole atoms, equal, the last padded; before the state record's shape,
    // as the engine states them (`SlotRole::Ring` first).
    if let Some(windowed) = windowed_recipe.as_mut()
        && windowed.region.placement == RingPlacement::ChunkStart
        && let Some(atom) = ring_atom
    {
        let per_piece = (chunk_bytes / atom.bytes.max(1)).max(1);
        let pieces = atom.extent.div_ceil(per_piece).max(1);
        let atoms_per_piece = atom.extent.div_ceil(pieces);
        let pieces = u32::try_from(pieces).unwrap_or(u32::MAX);
        atoms.insert(RowId::Ring, atom);
        windowed.ring_shape = Some(slots.len());
        slots.push(SlotLayout::Fixed {
            bytes: chunk_bytes,
            record: windowed.region.bytes,
            pieces_per_ws: pieces,
            rows: vec![FixedRun {
                row: RowId::Ring,
                pieces,
                atoms_per_piece,
                offset: 0,
                slots: (0..pieces).collect(),
            }],
        });
    }
    if let Some(pick) = pick {
        let (pieces_per_ws, rows) =
            fixed::pack(&fixed_rows, pick.packing(fixed_align), true).unwrap_or((0, Vec::new()));
        slots.push(SlotLayout::Fixed {
            bytes: pick.slot,
            record,
            pieces_per_ws,
            rows,
        });
    }
    Ok(ChunkRecipe {
        chunk_bytes,
        slots,
        atoms,
        shards,
        short_runs,
        waste,
        profile: device,
        weights,
        rs,
        windowed: windowed_recipe,
    })
}

/// A layer's bytes a token of `rows`, the thinnest; a row no layer reads is
/// its own.
fn thinnest(rows: &[&TokenRow]) -> u64 {
    layers(rows)
        .values()
        .map(|(b, _)| *b)
        .min()
        .unwrap_or(1)
        .max(1)
}

/// Each layer's bytes a token of `rows` and its first row; a row no layer
/// reads is its own layer.
fn layers(rows: &[&TokenRow]) -> BTreeMap<(Option<u32>, Option<RowId>), (u64, RowId)> {
    let mut rows: Vec<&TokenRow> = rows.to_vec();
    rows.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    let mut layers: BTreeMap<(Option<u32>, Option<RowId>), (u64, RowId)> = BTreeMap::new();
    for row in &rows {
        let key = (row.layer, row.layer.is_none().then_some(row.id));
        let entry = layers.entry(key).or_insert((0, row.id));
        entry.0 += row.per_token();
    }
    layers
}

/// A slot's bytes of `tokens` of each of `rows`, each run aligned.
fn slot_bytes(rows: &[&TokenRow], tokens: u32, align: u64) -> u64 {
    skew(
        rows.iter()
            .map(|r| align_up(u64::from(tokens) * r.per_token(), align))
            .sum(),
        align,
    )
}

/// The step a token slot's stride is an odd number of: a paged kernel reads
/// one layer's run of every slot it names, all at one offset into slots a
/// stride apart, and a stride of many whole steps lands those reads on a
/// few of the memory's channels.
pub const SLOT_SKEW: u64 = 1 << 10;

/// A token slot's stride: `bytes` aligned, and, where it is two skew steps
/// or more, an odd number of them. L40S, one layer's kv runs of 4400 pages
/// read a launch a layer, against the same bytes laid layer-major: slots of
/// 3 to 36 layers of 4 to 32 KiB pages read 1.12 to 1.49 slower where the
/// stride is three times a power of two (a 12-channel bus), and at most
/// 1.02 slower at an odd number of KiB (2, 4 and 8 KiB steps: 1.06, 1.07,
/// 1.14). A smaller slot's stride spreads its reads already.
#[must_use]
pub fn skew(bytes: u64, align: u64) -> u64 {
    let step = SLOT_SKEW.max(align.max(1));
    let aligned = align_up(bytes, align);
    if aligned < 2 * step {
        return aligned;
    }
    let stepped = align_up(aligned, step);
    if (stepped / step).is_multiple_of(2) {
        stepped + step
    } else {
        stepped
    }
}

/// The kv page size `T` of `lists` (the kv list's; with no kv list, every
/// window list's, the most a window slot is weighed at): the fewest tokens, a multiple of `base`, at which every layer of
/// every list runs [`MIN_RUN`] bytes, unless a list's slot would outgrow
/// `most`; then the most tokens at which every list's slot fits.
fn page_tokens<'a>(
    lists: impl Iterator<Item = &'a [&'a TokenRow]>,
    base: u32,
    most: u64,
    align: u64,
) -> u32 {
    let lists: Vec<&[&TokenRow]> = lists.collect();
    let base = base.max(1);
    let need = lists
        .iter()
        .map(|rows| {
            u32::try_from(MIN_RUN.div_ceil(thinnest(rows)))
                .unwrap_or(u32::MAX)
                .div_ceil(base)
                .max(1)
                * base
        })
        .max()
        .unwrap_or(base);
    let fits = |tokens: u32| {
        lists
            .iter()
            .all(|rows| slot_bytes(rows, tokens, align) <= most)
    };
    if fits(need) {
        return need;
    }
    let mut tokens = base;
    while tokens + base < need && fits(tokens + base) {
        tokens += base;
    }
    tokens
}

/// A token slot of `tokens` of each of `rows`, layer-major: its bytes, the
/// runs, and the layers whose run is under [`MIN_RUN`].
fn token_layout(rows: &[&TokenRow], tokens: u32, align: u64) -> (u64, Vec<TokenRun>, Vec<RowId>) {
    let mut sorted: Vec<&TokenRow> = rows.to_vec();
    sorted.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    let mut at = 0u64;
    let runs = sorted
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
    let short = layers(rows)
        .values()
        .filter(|(bytes, _)| u64::from(tokens) * bytes < MIN_RUN)
        .map(|(_, first)| *first)
        .collect();
    (slot_bytes(rows, tokens, align), runs, short)
}

/// What a working set's slots of window list `shape` hold past the rows
/// its window retains at `context` (a trimmed list holds about
/// `ceil(window / T_w) + 1` slots), and its share of the room past the last
/// whole slot of chunks of `chunk` bytes.
fn window_waste(shape: &SlotLayout, chunk: u64, context: u32) -> u64 {
    let bytes = shape.bytes().max(1);
    let per = (chunk / bytes).max(1);
    let slots = shape.slots_per_ws(context);
    let per_token: u64 = shape.runs().iter().map(|r| r.per_token).sum();
    let held = u64::from(shape.window().unwrap_or(context).min(context)) * per_token;
    (slots * bytes).saturating_sub(held) + slots * (chunk.saturating_sub(per * bytes)) / per
}

/// The window list's slot for chunks of `chunk` bytes: of the `shapes` a
/// chunk holds, the one that wastes least ([`window_waste`]), the more
/// tokens on a tie (a shorter table).
fn pick_window(shapes: &[SlotLayout], chunk: u64, context: u32) -> Option<SlotLayout> {
    shapes
        .iter()
        .filter(|s| s.bytes() <= chunk)
        .min_by_key(|s| {
            let tokens = match s {
                SlotLayout::List { tokens, .. } => *tokens,
                _ => 0,
            };
            (window_waste(s, chunk, context), core::cmp::Reverse(tokens))
        })
        .cloned()
}

/// An rs-list slot of `tokens` tokens: its bytes, and a token's bytes of
/// every rs row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RsShape {
    tokens: u32,
    bytes: u64,
    per_token: u64,
}

impl RsShape {
    fn of(rows: &[TokenRow], tokens: u32, align: u64) -> RsShape {
        RsShape {
            tokens,
            bytes: skew(
                rows.iter()
                    .map(|r| align_up(u64::from(tokens) * r.per_token(), align))
                    .sum(),
                align,
            ),
            per_token: rows.iter().map(TokenRow::per_token).sum(),
        }
    }

    /// The slots a working set holds at [`EXPECTED_UNFOLDED`] tokens.
    fn slots(&self) -> u64 {
        u64::from(EXPECTED_UNFOLDED.div_ceil(self.tokens.max(1)))
    }

    /// What a working set's slots hold past [`EXPECTED_UNFOLDED`] tokens,
    /// and its share of the room past the chunk's last whole slot.
    fn waste(&self, chunk: u64) -> u64 {
        let per = (chunk / self.bytes.max(1)).max(1);
        let slots = self.slots();
        slots * self.bytes - u64::from(EXPECTED_UNFOLDED) * self.per_token
            + slots * (chunk - per * self.bytes) / per
    }
}

/// The rs rows' runs in a slot of `tokens`, layer-major, each aligned.
fn rs_runs(rows: &[TokenRow], tokens: u32, align: u64) -> Vec<TokenRun> {
    let mut rows: Vec<&TokenRow> = rows.iter().collect();
    rows.sort_by_key(|r| (r.layer.unwrap_or(u32::MAX), r.id));
    let mut at = 0u64;
    rows.iter()
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
        .collect()
}

/// The rs-list slot for chunks of `chunk` bytes: of the `shapes` a chunk
/// holds, the one that wastes least past the expected unfolded tokens and
/// in the chunks it fills, the fewer tokens on a tie (a lane holds no more
/// than it buffers); more only where they fill better.
fn pick_rs(shapes: &[RsShape], chunk: u64) -> Option<RsShape> {
    shapes
        .iter()
        .filter(|s| s.bytes <= chunk)
        .min_by_key(|s| (s.waste(chunk), s.tokens))
        .copied()
}

/// A fixed slot size and the pieces a working set's record takes in it.
#[derive(Debug, Clone, Copy)]
struct FixedPick {
    slot: u64,
    pieces: u32,
    /// The least bytes a cut piece holds.
    least: u64,
}

impl FixedPick {
    fn packing(&self, align: u64) -> fixed::Packing {
        fixed::Packing {
            slot: self.slot,
            align,
            least: self.least,
        }
    }
}

/// The fixed slot for chunks of `chunk` bytes: the record whole where it
/// fits; else, of the piece sizes that fill a chunk (`chunk / k`), end a slot
/// on a row boundary, or hold `k` equal pieces of a cut row, no smaller than
/// [`MIN_PIECE`], the one that wastes least in the slots (padding counted)
/// and in the chunks they fill, the larger on a near tie. Cut pieces hold
/// [`MIN_PIECE`] at least where any size lets them, else as few as a size
/// takes.
fn pick_fixed(rows: &[FixedRow], record: u64, chunk: u64, align: u64) -> Option<FixedPick> {
    if record <= chunk {
        let slot = align_up(record.max(FIXED_SLOT_FLOOR), align);
        return Some(FixedPick {
            slot,
            pieces: 1,
            least: 0,
        });
    }
    let least = MIN_PIECE.min(chunk);
    let mut sizes: Vec<u64> = (1..=(chunk / least).max(1))
        .map(|k| align_down(chunk / k, align))
        .collect();
    // Slots ending on a row boundary.
    let mut at = 0;
    for row in rows {
        at = align_up(at, align) + row.bytes;
        sizes.push(align_up(at, align));
    }
    // A slot of `k` equal pieces of a row cut `n` ways.
    let mut atoms: Vec<Atom> = rows
        .iter()
        .filter(|r| fixed::cuttable(r) && r.bytes > least)
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
            let pick = FixedPick {
                slot,
                pieces: 0,
                least,
            };
            let Some((pieces, _)) = fixed::pack(rows, pick.packing(align), false) else {
                continue;
            };
            let per = chunk / slot;
            let held = u64::from(pieces) * slot;
            let cost = held.saturating_sub(record) + u64::from(pieces) * (chunk - per * slot) / per;
            if best.as_ref().is_none_or(|(was, _)| cost + near < *was) {
                best = Some((cost, FixedPick { pieces, ..pick }));
            }
        }
        if best.is_some() {
            return best.map(|(_, pick)| pick);
        }
    }
    None
}

/// What a working set takes at the profile's context with chunks of `chunk`
/// bytes, and what of that holds nothing: its kv and window lists, its rs list
/// at [`EXPECTED_UNFOLDED`] tokens, and its own fixed record.
/// A windowed recipe's shapes as [`weigh`] weighs them: the window buffer,
/// and the ring's atoms where it lies at chunk starts.
#[derive(Clone, Copy)]
struct Windowed<'a> {
    buffer: &'a SlotLayout,
    ring: Option<Atom>,
}

#[allow(clippy::too_many_arguments)]
fn weigh(
    tokens: &[SlotLayout],
    rs: Option<&RsShape>,
    fixed: Option<&FixedPick>,
    record: u64,
    chunk: u64,
    device: &ChunkProfile,
    windowed: Option<Windowed<'_>>,
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
            SlotLayout::List { window, .. } => {
                let tokens = window.map_or(context, |w| w.min(context));
                let held = u64::from(tokens) * per_token;
                waste.state += held;
                waste.list_tail += (slots * shape.bytes()).saturating_sub(held);
            }
            _ => waste.state += slots * shape.bytes(),
        }
        waste.chunk_tail += tail(slots, shape.bytes());
    }
    if let Some(rs) = rs {
        shapes += 1;
        let held = u64::from(EXPECTED_UNFOLDED) * rs.per_token;
        waste.state += held;
        waste.rs_tail += rs.slots() * rs.bytes - held;
        waste.chunk_tail += tail(rs.slots(), rs.bytes);
    }
    if let Some(pick) = fixed {
        shapes += 1;
        // The pool's fixed records: a set a working set.
        let working = u64::from(device.concurrency.max(1));
        let sets = working * u64::from(FIXED_SETS_PER_WS);
        let held = u64::from(pick.pieces) * pick.slot;
        waste.state += sets * record / working;
        waste.fixed_pad += sets * held.saturating_sub(record) / working;
        waste.chunk_tail += tail(sets * u64::from(pick.pieces), pick.slot) / working;
    }
    if let Some(Windowed { buffer, ring }) = windowed {
        // The window buffer holds a fire's unfolded tail, as the rs list
        // does: [`EXPECTED_UNFOLDED`] rows.
        shapes += 1;
        let tokens = match buffer {
            SlotLayout::List { tokens, .. } => (*tokens).max(1),
            _ => 1,
        };
        let slots = u64::from(EXPECTED_UNFOLDED.div_ceil(tokens));
        let per_token: u64 = buffer.runs().iter().map(|r| r.per_token).sum();
        let held = u64::from(EXPECTED_UNFOLDED.min(tokens)) * per_token;
        waste.state += held;
        waste.list_tail += (slots * buffer.bytes()).saturating_sub(held);
        waste.chunk_tail += tail(slots, buffer.bytes());
        // A ring at chunk starts: whole chunks a record, as a fixed shape.
        if let Some(atom) = ring {
            shapes += 1;
            let per_piece = (chunk / atom.bytes.max(1)).max(1);
            let pieces = atom.extent.div_ceil(per_piece).max(1);
            let ring_record = atom.extent * atom.bytes;
            let working = u64::from(device.concurrency.max(1));
            let sets = working * u64::from(FIXED_SETS_PER_WS);
            waste.state += sets * ring_record / working;
            waste.fixed_pad += sets * (pieces * chunk).saturating_sub(ring_record) / working;
        }
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
