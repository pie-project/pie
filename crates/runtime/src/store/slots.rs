//! One pool of device chunks for every kind of state.
//!
//! A **slot** is what a holder takes and refcounts; a **chunk** holds slots of
//! one slot shape and is shared by every working set that has a slot in it;
//! a **segment** (see [`super::segments`]) is what the backend maps and
//! unmaps. The pool hands out *logical* chunk ids, which holders and lanes
//! keep for as long as they hold a slot; the engine maps them to physical
//! positions, and an evacuation changes only that mapping.
//!
//! Chunks are opened lowest physical address first, so what is held packs
//! toward the bottom and the top segments empty out. A shape fills its
//! fullest partly filled chunk first, so few chunks stay partly filled.
//!
//! A slot whose last holder lets go is not handed out again until the fire
//! that last touched it retires (as [`super::pool::Pool::recycle_after_epoch`]
//! and [`super::pool::Pool::retire_through`]); a chunk whose slots are all
//! back is free again.
//!
//! Budget: `reserved + state chunks + act_chunks(rung) <= committed <= total`.
//!
//! Position (two-ended pools): activation takes the top `act_chunks(rung)`
//! physical chunks of the range, so state chunks lie only below
//! `total - act_chunks(rung)` ([`SlotPool::state_limit`]). The segments under
//! the largest rung's activation are committed from the load and never
//! evacuated or released. Raising the rung first moves whatever state sits
//! in the new activation span down ([`SlotPool::plan_rung`],
//! [`SlotPool::apply_rung`]); opening lowest first keeps the top free.
//!
//! Logical ids mirror the engine's `ChunkTable`: logical and physical ids are
//! a permutation of `0..total`, the identity at load, and only an evacuation
//! changes it, the way `ChunkTable::apply` does (the moved chunk takes the
//! target's place, the free logical id there takes the moved one's old
//! place). Opening physical chunk `p` hands out the logical id the table maps
//! to `p`, so the runtime's slot refs and the engine's table always agree.
//!
//! The local types stand in for package E1's `engine::chunk` until it lands:
//! [`SlotShape`] is its `SlotShape` without the per-kind geometry (pieces,
//! tokens, window, rows: the pool needs only the slot count and size),
//! [`SlotPoolFacts`] is the part of `ChunkFacts` the pool keeps (chunk_bytes,
//! total_chunks, segment_chunks, committed segments, the reserved chunk
//! count, shapes, act_chunks), and [`Evacuation`] carries E1's
//! `moves: Vec<(logical, new_physical)>` plus the old position.
//!
//! Placement by class: a chunk holds one class while it is open, set as it
//! opens ([`ChunkMeta::class`]). Single slots ([`SlotPool::acquire`]) lie
//! only in single-slot chunks; a run of `n` adjacent slots
//! ([`SlotPool::acquire_run`]) only among runs of exactly `n`, its class the
//! run's own length, with no table of classes; a chunk taken whole
//! ([`SlotPool::acquire_chunk_runs`]) holds nothing else. A run's hole is
//! then one the next run of its length takes exactly, no single slot ever
//! breaks a run's room, and a chunk whose items all leave is free whole
//! again for any class. Its cost is at most one partly filled chunk per
//! class in use.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use state::key::Hash256;

pub use super::segments::{ChunkMove, Evacuation, SegmentError, SegmentState, Segments};

mod claims;
mod sizes;
mod rung;
mod compaction;
mod census;
mod holdings;
mod acquire;
mod release;
mod chunks;
mod span;
mod walk;

/// A slot shape's id: its index in [`SlotPoolFacts::shapes`].
pub type ShapeId = u16;

/// A piece of a parked guest state its restore retakes, as
/// [`SlotPool::chunk_runs_beside`] places it on its view: `pieces` runs of
/// `run` whole chunks, or one run of `slots` slots of `shape` in a chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnPiece {
    Chunks { run: u32, pieces: u32 },
    Slots { shape: ShapeId, slots: u32 },
    /// `slots` single slots of `shape`, as [`SlotPool::acquire`] takes them:
    /// the shape's partly filled chunks first, then free chunks opened
    /// lowest first ([`SlotPool::opens_for`]).
    Take { shape: ShapeId, slots: u64 },
}

/// Who holds what [`SlotPool::place`] takes on its scratch copy.
const PLACING: Holder = Holder::WorkingSet(u64::MAX - 1);

/// Where [`SlotPool::place`] put a list of pieces, on a scratch copy of
/// the pool the takes themselves ran on.
#[derive(Clone, Debug)]
pub struct Placement {
    /// The free physical chunks the placed pieces opened.
    pub opened: std::collections::BTreeSet<u32>,
    /// The budget's headroom left once they were placed.
    pub room: u32,
    /// The pieces placed, in order, before the first the takes refused.
    pub placed: usize,
    asked: usize,
    scratch: SlotPool,
}

impl Placement {
    /// Whether every piece was placed.
    #[must_use]
    pub fn stands(&self) -> bool {
        self.placed == self.asked
    }

    /// The free runs of `run` adjacent chunks the pool holds once the
    /// placed pieces are taken ([`SlotPool::free_chunk_runs`] on the copy).
    #[must_use]
    pub fn free_runs(&self, run: u32) -> u32 {
        self.scratch.free_chunk_runs(run)
    }

    /// Whether `runs` (`(run, pieces)` of whole chunks) are placed by the
    /// takes themselves once the placed pieces are taken, within the
    /// budget's headroom: the one reading a restore's reservation
    /// ([`super::roles::Roles::reserve_restore`]) holds its bind's runs by.
    #[must_use]
    pub fn places_runs(&self, runs: &[(u32, u32)]) -> bool {
        let mut scratch = self.scratch.scratch();
        runs.iter()
            .all(|&(run, pieces)| scratch.acquire_chunk_runs(PLACING, 0, run, pieces).is_ok())
    }
}

/// Fixed sets each working set holds: the one its fires read. Derived, not
/// configured.
pub const FIXED_SETS_PER_WORKING_SET: u32 = 1;

const NONE: u32 = u32::MAX;
const NO_SHAPE: ShapeId = ShapeId::MAX;
const WORD: u32 = u64::BITS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    /// Whole-sequence state (recurrent, conv, rs buffer, snapshot ids).
    Fixed,
    /// Per-token state, in token order: the kv list, a window list (its
    /// slots behind the window trimmed), the rs list.
    List,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotShape {
    pub kind: ShapeKind,
    pub slot_bytes: u64,
    pub slots_per_chunk: u32,
}

impl SlotShape {
    /// Slot sets each working set holds of this shape; a fork's write takes
    /// one more of a fixed one from the pool while a sibling still reads it.
    pub fn sets_per_working_set(&self) -> u32 {
        match self.kind {
            ShapeKind::Fixed => FIXED_SETS_PER_WORKING_SET,
            ShapeKind::List => 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SlotPoolFacts {
    pub chunk_bytes: u64,
    pub total_chunks: u32,
    pub segment_chunks: u32,
    /// The lowest segments committed at load.
    pub committed_segments: u32,
    /// The lowest chunks, never handed out: the null chunk, a sink, chunks
    /// for lanes the runtime does not seat. Logical = physical for them.
    pub reserved_chunks: u32,
    pub shapes: Vec<SlotShape>,
    /// `(rung tokens, activation chunks)`, ascending.
    pub act_chunks: Vec<(u32, u32)>,
    /// The span under the largest rung's activation is backed on demand
    /// (`ChunkFacts::span_backed_on_demand`): not committed at load; the
    /// engine backs its own activation carve, and a span segment is
    /// committed here ([`SlotPool::commit_span`]) before state opens a
    /// chunk in it.
    pub span_on_demand: bool,
    /// The chunks a rung's activation may take as the engine stands
    /// (`ChunkFacts::act_span_chunks`); `None`: nothing of the engine's
    /// own bounds it.
    pub act_span: Option<u32>,
}

/// One slot: a logical chunk and the slot's index in it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Slot {
    pub chunk: u32,
    pub index: u32,
}

/// Who holds a reference on a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Holder {
    WorkingSet(u64),
    CacheEntry(Hash256),
    Snapshot(u64),
    /// The weight cache's slots of weight shape `shape` (see
    /// [`super::weights`]): `pinned`, its staging floor (and, not
    /// streaming, every atom), held whatever the pressure; else the
    /// resident atoms above the floor, which go first when the pool is
    /// short.
    Weights {
        shape: ShapeId,
        pinned: bool,
    },
}

impl Holder {
    /// Working sets, snapshots and the weights' staging floor keep their
    /// slots live; a cache entry's slots, and resident weights above the
    /// floor, are wanted but may go.
    pub fn is_live(&self) -> bool {
        match self {
            Holder::CacheEntry(_) => false,
            Holder::Weights { pinned, .. } => *pinned,
            Holder::WorkingSet(_) | Holder::Snapshot(_) => true,
        }
    }
}

/// Demand on the pool, counted slot by slot, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Demand {
    /// Slots a working set or a snapshot holds.
    pub live: u64,
    /// Live, plus slots only cache entries hold, plus what waiters lack.
    pub wanted: u64,
}

/// One shape's demand, in slots.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeDemand {
    pub live: u64,
    pub cached: u64,
    pub waiting: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SlotError {
    #[error("no slot shape {0}")]
    UnknownShape(ShapeId),
    #[error("shape {shape} is short {slots} slots ({chunks} chunks)")]
    Short {
        shape: ShapeId,
        slots: u64,
        chunks: u32,
    },
    #[error("{needed} chunks do not fit in {committed} committed")]
    Budget { needed: u32, committed: u32 },
    #[error("rung {rung}'s activation takes {chunks} chunks state holds; evacuate them first")]
    Occupied { rung: u32, chunks: u32 },
    #[error("no free chunk below rung {rung}'s activation for {chunks} chunks in its way")]
    NoRoom { rung: u32, chunks: u32 },
    #[error("a rung evacuation no longer matches the pool")]
    StaleRung,
    #[error("a compaction no longer matches the pool")]
    StaleCompaction,
    /// Moves that would leave a run of chunks split: one of its chunks
    /// moved without the rest, or not to the next physical chunk after its
    /// first's destination. Refused whoever planned them.
    #[error("a run of {run} chunk(s) from physical chunk {first} would lie across a segment's end")]
    RunAcrossSegment { first: u32, run: u32 },
    #[error("chunk moves would split the run of {run} chunk(s) from logical chunk {head} at its chunk {at}")]
    SplitsRun { head: u32, run: u32, at: u32 },
    /// A claim named a chunk that cannot be taken whole as it stands.
    #[error("chunk {chunk} cannot be claimed: {why}")]
    Unclaimable { chunk: u32, why: &'static str },
    /// A run of adjacent slots longer than a chunk holds.
    #[error("a run of {slots} slots of shape {shape} is past a chunk's {per_chunk}")]
    RunPastChunk {
        shape: ShapeId,
        slots: u64,
        per_chunk: u32,
    },
    /// A run of adjacent chunks longer than a segment holds: chunks lie at
    /// adjacent addresses only inside one segment.
    #[error("a run of {run} adjacent chunks is past a segment's {segment}")]
    RunPastSegment { run: u32, segment: u32 },
    /// Fewer free runs of `run` adjacent chunks inside one segment than
    /// `pieces`: what was free, as runs of that length and as chunks.
    #[error(
        "{pieces} runs of {run} adjacent chunks inside one segment asked, and {free_runs} \
         such runs are free ({free_chunks} free chunks)"
    )]
    NoChunkRun {
        run: u32,
        pieces: u32,
        free_runs: u32,
        free_chunks: u32,
    },
}

/// The state chunks in the way of a rung's activation, each moved to a free
/// chunk below it; logical ids stand. The engine copies `from` -> `to` and
/// rewrites its chunk table before the first fire at the rung.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RungEvacuation {
    pub rung: u32,
    /// The rung the activation was set aside for when it was planned.
    pub from: u32,
    /// The activation chunks the rung takes.
    pub act: u32,
    pub moves: Vec<ChunkMove>,
}

/// A word bitmap with a count and a hint below which no bit is set.
#[derive(Clone, Debug)]
struct Bits {
    words: Vec<u64>,
    lowest: usize,
    count: u32,
}

impl Bits {
    fn new(len: u32, set: bool) -> Self {
        let mut words = vec![if set { u64::MAX } else { 0 }; len.div_ceil(WORD) as usize];
        if set
            && let Some(last) = words.last_mut()
            && !len.is_multiple_of(WORD)
        {
            *last = (1 << (len % WORD)) - 1;
        }
        Self {
            words,
            lowest: 0,
            count: if set { len } else { 0 },
        }
    }

    fn locate(at: u32) -> (usize, u64) {
        ((at / WORD) as usize, 1 << (at % WORD))
    }

    fn get(&self, at: u32) -> bool {
        let (word, bit) = Self::locate(at);
        self.words[word] & bit != 0
    }

    fn set(&mut self, at: u32) {
        let (word, bit) = Self::locate(at);
        if self.words[word] & bit == 0 {
            self.words[word] |= bit;
            self.count += 1;
            self.lowest = self.lowest.min(word);
        }
    }

    fn clear(&mut self, at: u32) {
        let (word, bit) = Self::locate(at);
        if self.words[word] & bit != 0 {
            self.words[word] &= !bit;
            self.count -= 1;
        }
    }

    /// The lowest set bit.
    fn first(&mut self) -> Option<u32> {
        if self.count == 0 {
            return None;
        }
        while self.words[self.lowest] == 0 {
            self.lowest += 1;
        }
        let word = self.lowest;
        Some(word as u32 * WORD + self.words[word].trailing_zeros())
    }

    /// The set bits in `from..to`, ascending.
    fn iter_in(&self, from: u32, to: u32) -> impl Iterator<Item = u32> + '_ {
        let mut at = from;
        std::iter::from_fn(move || {
            while at < to {
                let (word, _) = Self::locate(at);
                let rest = self.words[word] >> (at % WORD);
                if rest == 0 {
                    at = (word as u32 + 1) * WORD;
                    continue;
                }
                let found = at + rest.trailing_zeros();
                if found >= to {
                    return None;
                }
                at = found + 1;
                return Some(found);
            }
            None
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SlotRefs {
    live: u32,
    cache: u32,
    /// The last fire that touched the slot.
    touched: u64,
}

#[derive(Clone, Copy, Debug)]
struct ChunkMeta {
    shape: ShapeId,
    /// The chunk's frame in its shape's slot arena.
    frame: u32,
    /// Slots held or waiting on an epoch.
    used: u32,
    /// What the chunk holds while open ([`SINGLE`], a run length, or
    /// [`WHOLE`]): set as it opens, fixed until it is free again.
    class: u32,
}

/// A chunk of a shape's single slots, as [`SlotPool::single_chunks`] reads
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SingleChunk {
    pub logical: u32,
    /// The slots it holds or keeps for an epoch.
    pub used: u32,
    /// No chunk move or residency copy holds it.
    pub settled: bool,
}

/// The class of a chunk of single slots ([`SlotPool::acquire`]): a run of
/// one is a single slot too.
pub const SINGLE: u32 = 1;

/// The class of a chunk taken whole ([`SlotPool::acquire_chunk_runs`]):
/// no single slot or run is placed in it.
pub const WHOLE: u32 = u32::MAX;

const FREE_CHUNK: ChunkMeta = ChunkMeta {
    shape: NO_SHAPE,
    frame: NONE,
    used: 0,
    class: SINGLE,
};

/// One shape's slot arena: each open chunk owns a frame of
/// `slots_per_chunk` slots, and frames are reused, so opening a chunk
/// allocates nothing once the arena has grown.
#[derive(Clone, Debug)]
struct ShapeState {
    shape: SlotShape,
    words: usize,
    /// Per frame: a set bit is a slot free to hand out.
    free: Vec<u64>,
    refs: Vec<SlotRefs>,
    spare: Vec<u32>,
    frames: u32,
    /// Partly filled chunks by class ([`ChunkMeta::class`]), each class's
    /// fullest first, then lowest physical address:
    /// `(Reverse(used), physical, logical)`.
    partial: BTreeMap<u32, BTreeSet<(Reverse<u32>, u32, u32)>>,
    /// Free slots in single-slot chunks' `partial`: what a take of single
    /// slots finds without opening a chunk.
    partial_free: u64,
    /// Slots some working set or snapshot holds.
    live: u64,
    /// Slots only cache entries hold.
    cached: u64,
    waiting: u64,
}

impl ShapeState {
    fn new(shape: SlotShape) -> Self {
        assert!(shape.slots_per_chunk > 0, "a chunk holds at least one slot");
        Self {
            shape,
            words: shape.slots_per_chunk.div_ceil(WORD) as usize,
            free: Vec::new(),
            refs: Vec::new(),
            spare: Vec::new(),
            frames: 0,
            partial: BTreeMap::new(),
            partial_free: 0,
            live: 0,
            cached: 0,
            waiting: 0,
        }
    }

    fn spc(&self) -> u32 {
        self.shape.slots_per_chunk
    }

    fn open_frame(&mut self) -> u32 {
        let frame = self.spare.pop().unwrap_or_else(|| {
            let frame = self.frames;
            self.frames += 1;
            self.free.resize(self.free.len() + self.words, 0);
            self.refs
                .resize(self.refs.len() + self.spc() as usize, SlotRefs::default());
            frame
        });
        let spc = self.spc();
        let words = &mut self.free[frame as usize * self.words..][..self.words];
        words.fill(u64::MAX);
        if !spc.is_multiple_of(WORD) {
            words[self.words - 1] = (1 << (spc % WORD)) - 1;
        }
        frame
    }

    /// The lowest slot of frame `frame` that starts `n` free slots in a
    /// row, if one does.
    /// The slots `held` reads as held count as taken too: a view of what
    /// runs placed on it would leave ([`SlotPool::chunk_runs_beside`]).
    fn free_run_beside(&self, frame: u32, n: u32, held: impl Fn(u32) -> bool) -> Option<u32> {
        let base = frame as usize * self.words;
        let free = |index: u32| {
            let (word, bit) = Bits::locate(index);
            self.free[base + word] & bit != 0 && !held(index)
        };
        let mut start = 0;
        let mut length = 0;
        for index in 0..self.spc() {
            if free(index) {
                if length == 0 {
                    start = index;
                }
                length += 1;
                if length == n {
                    return Some(start);
                }
            } else {
                length = 0;
            }
        }
        None
    }

    fn ref_at(&mut self, frame: u32, index: u32) -> &mut SlotRefs {
        let at = (frame * self.spc() + index) as usize;
        &mut self.refs[at]
    }

    /// Moves a chunk's place among its class's partly filled as its count
    /// goes from `was` to `now`.
    fn reposition(&mut self, physical: u32, logical: u32, class: u32, was: u32, now: u32) {
        let spc = self.spc();
        if was > 0 && was < spc {
            let emptied = self.partial.get_mut(&class).is_some_and(|chunks| {
                chunks.remove(&(Reverse(was), physical, logical));
                chunks.is_empty()
            });
            if emptied {
                self.partial.remove(&class);
            }
            if class == SINGLE {
                self.partial_free -= u64::from(spc - was);
            }
        }
        if now > 0 && now < spc {
            self.partial
                .entry(class)
                .or_default()
                .insert((Reverse(now), physical, logical));
            if class == SINGLE {
                self.partial_free += u64::from(spc - now);
            }
        }
    }

    /// The fullest partly filled chunk of `class`, its logical id.
    fn fullest(&self, class: u32) -> Option<u32> {
        self.partial
            .get(&class)
            .and_then(|chunks| chunks.first())
            .map(|&(_, _, logical)| logical)
    }

    /// Counts a slot's move between live, cached-only and unheld.
    fn reclassify(&mut self, was: SlotRefs, now: SlotRefs) {
        let class = |refs: SlotRefs| (refs.live > 0, refs.live == 0 && refs.cache > 0);
        let (was_live, was_cached) = class(was);
        let (now_live, now_cached) = class(now);
        self.live = self.live + u64::from(now_live) - u64::from(was_live);
        self.cached = self.cached + u64::from(now_cached) - u64::from(was_cached);
    }
}

/// Test builds: the queue a pool's given-back slots go to
/// ([`SlotPool::poison_into`]). A clone of the pool (a planner's or a test's
/// copy to try a placement on) taps nothing: what it gives back it gives
/// back in its copy alone, and the slots are still the live pool's.
#[cfg(feature = "test-slot-fill")]
#[derive(Default)]
struct PoisonTap(Option<::engine::chunk::PoisonQueue>);

#[cfg(feature = "test-slot-fill")]
impl Clone for PoisonTap {
    fn clone(&self) -> Self {
        PoisonTap(None)
    }
}

#[cfg(feature = "test-slot-fill")]
impl std::fmt::Debug for PoisonTap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "PoisonTap(tapped)" } else { "PoisonTap(none)" })
    }
}

/// See the module docs.
#[derive(Clone, Debug)]
pub struct SlotPool {
    /// Test builds: where a slot given back goes to be poisoned
    /// ([`PoisonTap`]).
    #[cfg(feature = "test-slot-fill")]
    poison: PoisonTap,
    chunk_bytes: u64,
    total: u32,
    reserved: u32,
    shapes: Vec<ShapeState>,
    act_chunks: Vec<(u32, u32)>,
    act: u32,
    /// The activation chunks the engine may still back for fires: where
    /// the span is backed on demand, the widest activation set aside since
    /// the engine last narrowed its carve ([`Self::engine_narrowed`]). No
    /// state opens there: a lower the engine was not told (the rung
    /// standing, a guest term given back) leaves its carve as wide as
    /// before, and state lent into it would be refused at its first copy
    /// (an eviction's copy inside the engine's carve, failing the
    /// process). Elsewhere it
    /// is the activation itself.
    carved: u32,
    /// Physical chunks a residency move's copy (an eviction, parked state
    /// moving down) reads from while it is in flight, each with how many
    /// such moves read it: a raise does not take them back for its
    /// activation until the copy has landed ([`Self::plan_rung`]).
    moving_out: BTreeMap<u32, u32>,
    /// What a fire takes for the guest programs it carries, on top of
    /// every rung's activation ([`Self::act_chunks`]): the term of the
    /// pairs that can be in a fire ([`::engine::GuestTerm`]).
    fire: ::engine::GuestTerm,
    /// The engine's standing guest bytes, beside the fire term below every
    /// rung's carve ([`Self::set_standing`]).
    standing: u64,
    /// The readout term ([`Self::set_readouts`]): the bytes a row a fire
    /// reads out takes beside the model's activation (0: inside the span),
    /// and the most rows one fire reads out.
    readout_row_bytes: u64,
    readout_rows: u32,
    /// The most lanes a fire carries (the engine's `FireLimits::max_lanes`):
    /// a rung of `r` tokens allows `min(r, max_lanes)`.
    max_lanes: u32,
    /// The largest rung's activation chunks, the top of the range.
    act_max: u32,
    /// The first segment under the largest rung's activation: it and those
    /// above are committed from the load and never evacuated or released,
    /// unless the span is backed on demand ([`SlotPoolFacts::span_on_demand`]).
    top_segment: u32,
    /// See [`SlotPoolFacts::span_on_demand`].
    span_on_demand: bool,
    /// What the engine states its activation span covers as it stands
    /// ([`Self::set_act_span`]): the statement itself, nothing derived.
    act_span: Option<u32>,
    chunks: Vec<ChunkMeta>,
    /// A permutation of `0..total`, as the engine's chunk table.
    logical_to_physical: Vec<u32>,
    physical_to_logical: Vec<u32>,
    /// Free physical chunks.
    free: Bits,
    /// Free physical chunks of committed segments below the activation, not
    /// fenced: what may be opened.
    open: Bits,
    /// Physical chunks a rung evacuation moved state off, kept closed until
    /// the fire that carries the copy retires, by that fire's epoch.
    fences: BTreeMap<u64, Vec<u32>>,
    fenced: Bits,
    /// No chunk at or past this physical id is opened: the engine binds
    /// none there yet ([`Self::set_ceiling`]).
    ceiling: u32,
    segments: Segments,
    /// Chunks opened for a shape.
    held: u32,
    pending: BTreeMap<u64, Vec<Slot>>,
    retired: u64,
    holders: HashMap<Holder, u64>,
    waiters: HashMap<u64, (ShapeId, u64)>,
    /// Runs of adjacent chunks taken whole ([`Self::acquire_chunk_runs`])
    /// and longer than one: each run's first logical chunk and its length.
    /// The chunks after the first lie at the next physical chunks, and an
    /// evacuation moves the run whole.
    runs: HashMap<u32, u32>,
    /// The chunks the last compaction moved off ([`Self::apply_compaction`]),
    /// fenced until its copy's fire retires.
    compacted: Vec<u32>,
    /// Told each time a chunk comes free for state ([`Self::room_watch`]).
    given: std::sync::Arc<RoomGiven>,
    /// Slots taken and given back since the pool was made: moved, the
    /// placement of what is held has changed ([`Self::room_changes`]).
    changes: u64,
    /// Free physical chunks held for a served restore by its ticket
    /// ([`Self::reserve_chunks`]): closed to every take but its own
    /// ([`Self::with_claim`]) until its unpark and bind land.
    held_for: BTreeMap<u64, BTreeSet<u32>>,
    next_ticket: u64,
}

/// Room given back: a chunk opened for state, by a slot's release, a fence
/// retired, a lower or a commit. A count of the gives, and their event.
#[derive(Debug, Default)]
pub struct RoomGiven {
    gives: std::sync::atomic::AtomicU64,
    notify: tokio::sync::Notify,
    /// Told each time slots come free, of any shape, in an open chunk or
    /// not (a release, a fire's epoch retiring), with the pool's counts
    /// after: the one place every return of room passes, so the waiter it
    /// can serve is woken whichever path gave it back
    /// ([`SlotPool::set_room_signal`]). Its tier, and the shape whose free
    /// slots it counts.
    signal: std::sync::OnceLock<(RoomSignal, RoomTier, Option<ShapeId>)>,
}

/// Which tier a [`RoomCount`] is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoomTier {
    Device,
    /// The host or the disk tier: where a victim's move goes.
    Swap,
}

/// A pool's room as slots came free: its free chunks and the free slots of
/// the shape it counts (the kv list's), read under its own lock and handed
/// to the signal, which takes none.
#[derive(Clone, Copy, Debug)]
pub struct RoomCount {
    pub tier: RoomTier,
    pub chunks: u64,
    pub slots: u64,
}

/// What [`RoomGiven::signal`] calls: lock-free, for a release may run under
/// the pool's lock and its caller's.
#[derive(Clone)]
pub struct RoomSignal(pub std::sync::Arc<dyn Fn(RoomCount) + Send + Sync>);

impl std::fmt::Debug for RoomSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RoomSignal")
    }
}

impl RoomGiven {
    fn give(&self) {
        self.gives.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.notify.notify_waiters();
    }
}

/// A watch on room given back from when it was taken ([`SlotPool::room_watch`]).
#[derive(Debug, Clone)]
pub struct RoomWatch {
    given: std::sync::Arc<RoomGiven>,
    seen: u64,
}

impl RoomWatch {
    /// Ends once a chunk has come free since the watch was taken: at once
    /// where one already has.
    pub async fn given(self) {
        use std::sync::atomic::Ordering;
        loop {
            let notified = self.given.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.given.gives.load(Ordering::Acquire) != self.seen {
                return;
            }
            notified.await;
        }
    }
}

impl SlotPool {
    pub fn new(facts: SlotPoolFacts) -> Self {
        let total = facts.total_chunks;
        assert!(facts.reserved_chunks <= total);
        assert!(facts.shapes.len() < NO_SHAPE as usize);
        let mut act_chunks = facts.act_chunks;
        act_chunks.sort_unstable();
        let act_max = act_chunks.last().map_or(0, |&(_, chunks)| chunks);
        assert!(
            facts.reserved_chunks + act_max <= total,
            "the reserved chunks and the largest rung's activation outgrow the pool"
        );
        let mut segments = Segments::new(total, facts.segment_chunks, facts.committed_segments);
        // The largest rung's activation is backed from the load, unless the
        // span is backed on demand.
        // The span starts where the engine pins it: the segment its widest
        // carve with its guests' terms starts in (`act_span_chunks`), when
        // that is wider than the largest rung's activation. Read from the
        // rungs alone, a segment inside the span would count as a bottom
        // segment: the runtime would open its chunks for state that Metal
        // backs as span, and restores wanting them be refused ("the load
        // reserved ...").
        let span = act_max.max(facts.act_span.unwrap_or(0)).min(total);
        let top_segment = if span == 0 {
            segments.count()
        } else {
            segments.of(total - span)
        };
        for segment in top_segment..segments.count() {
            if !facts.span_on_demand && segments.state(segment) == SegmentState::Uncommitted {
                segments.set_state(segment, SegmentState::Committed);
            }
        }
        let mut free = Bits::new(total, true);
        for chunk in 0..facts.reserved_chunks {
            free.clear(chunk);
            segments.hold(chunk);
        }
        let mut pool = Self {
            chunk_bytes: facts.chunk_bytes,
            total,
            reserved: facts.reserved_chunks,
            shapes: facts.shapes.into_iter().map(ShapeState::new).collect(),
            act_chunks,
            act: 0,
            carved: 0,
            moving_out: BTreeMap::new(),
            fire: ::engine::GuestTerm::default(),
            standing: 0,
            readout_row_bytes: 0,
            readout_rows: u32::MAX,
            max_lanes: u32::MAX,
            act_max,
            top_segment,
            span_on_demand: facts.span_on_demand,
            act_span: facts.act_span,
            chunks: vec![FREE_CHUNK; total as usize],
            logical_to_physical: (0..total).collect(),
            physical_to_logical: (0..total).collect(),
            free,
            open: Bits::new(total, false),
            fences: BTreeMap::new(),
            compacted: Vec::new(),
            fenced: Bits::new(total, false),
            ceiling: total,
            segments,
            held: 0,
            pending: BTreeMap::new(),
            retired: 0,
            holders: HashMap::new(),
            waiters: HashMap::new(),
            runs: HashMap::new(),
            given: std::sync::Arc::default(),
            changes: 0,
            held_for: BTreeMap::new(),
            next_ticket: 0,
            #[cfg(feature = "test-slot-fill")]
            poison: PoisonTap::default(),
        };
        pool.act = pool.act_chunks(0);
        assert!(
            pool.reserved + pool.act <= pool.committed_chunks(),
            "the reserved chunks and the smallest rung's activation outgrow the committed segments"
        );
        for chunk in 0..total {
            pool.reopen(chunk);
        }
        pool
    }
}

#[cfg(test)]
mod tests;
