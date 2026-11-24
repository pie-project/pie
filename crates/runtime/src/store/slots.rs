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

    /// Whether physical chunk `chunk` may be opened: free, in a committed
    /// segment, below the activation and not fenced.
    fn openable(&self, chunk: u32) -> bool {
        self.free.get(chunk)
            && !self.fenced.get(chunk)
            && chunk < self.state_limit()
            && chunk < self.ceiling
            && self.segments.state(self.segments.of(chunk)) == SegmentState::Committed
            && !self.held_for.values().any(|held| held.contains(&chunk))
    }

    /// No chunk at or past physical id `ceiling` is opened from now on, the
    /// engine binding none there yet (while it holds back part of the range
    /// it stated, [`::engine::chunk::ChunkFacts::held_bytes`]); chunks below it open as
    /// before. A chunk already open past it stays its holder's.
    pub fn set_ceiling(&mut self, ceiling: u32) {
        let ceiling = ceiling.min(self.total);
        let (from, to) = (ceiling.min(self.ceiling), ceiling.max(self.ceiling));
        self.ceiling = ceiling;
        for chunk in from..to {
            self.reopen(chunk);
        }
    }

    /// See [`Self::set_ceiling`].
    pub fn ceiling(&self) -> u32 {
        self.ceiling
    }

    /// Whether `segment` may be committed: it lies wholly under the
    /// ceiling (all of it, with no ceiling). One that straddles it would map
    /// chunks the engine holds back, past the pool's hard limit.
    pub fn backs(&self, segment: u32) -> bool {
        self.ceiling >= self.total || self.segments.range(segment).end <= self.ceiling
    }

    /// Brings `chunk`'s place in [`Self::open`] up to date.
    fn reopen(&mut self, chunk: u32) {
        if self.openable(chunk) {
            if !self.open.get(chunk) {
                self.open.set(chunk);
                self.given.give();
            }
        } else {
            self.open.clear(chunk);
        }
    }

    /// How many times room was given back (a chunk opened for state) since
    /// the pool was made.
    #[must_use]
    pub fn room_gives(&self) -> u64 {
        self.given.gives.load(std::sync::atomic::Ordering::Acquire)
    }

    /// How many times slots were taken or given back since the pool was
    /// made: unmoved, every placement read on the pool reads the same.
    #[must_use]
    pub fn room_changes(&self) -> u64 {
        self.changes
    }

    /// Holds free physical `chunks` for one claimant: closed to every
    /// other take until [`Self::release_reservation`], opened to its own
    /// inside [`Self::with_claim`]. Chunks not free now are left out. Its
    /// ticket.
    pub fn reserve_chunks(&mut self, chunks: impl IntoIterator<Item = u32>) -> u64 {
        self.next_ticket += 1;
        let ticket = self.next_ticket;
        let held: BTreeSet<u32> = chunks
            .into_iter()
            .filter(|&chunk| chunk < self.total && self.open.get(chunk))
            .collect();
        for &chunk in &held {
            self.open.clear(chunk);
        }
        self.changes += 1;
        self.held_for.insert(ticket, held);
        ticket
    }

    /// Whether a reservation holds a chunk in `start..end`.
    fn reserved_in(&self, start: u32, end: u32) -> bool {
        self.held_for
            .values()
            .any(|held| held.range(start..end).next().is_some())
    }

    /// The chunks `ticket` still holds.
    #[must_use]
    pub fn reserved(&self, ticket: u64) -> Vec<u32> {
        self.held_for
            .get(&ticket)
            .map(|held| held.iter().copied().collect())
            .unwrap_or_default()
    }

    /// `ticket`'s chunks still free go back to every take.
    pub fn release_reservation(&mut self, ticket: u64) {
        let Some(held) = self.held_for.remove(&ticket) else {
            return;
        };
        for chunk in held {
            self.reopen(chunk);
        }
        self.changes += 1;
        self.slots_freed();
    }

    /// `f` run with `ticket`'s held chunks open to it: what it takes of
    /// them leaves the reservation, what it leaves stays held.
    pub fn with_claim<R>(&mut self, ticket: Option<u64>, f: impl FnOnce(&mut Self) -> R) -> R {
        let opened = self.claim_open(ticket);
        let out = f(self);
        self.claim_close(opened);
        out
    }

    /// Opens `ticket`'s held chunks to the takes that follow, until
    /// [`Self::claim_close`]: what [`Self::with_claim`] does around its
    /// takes, for a caller whose takes go through another owner's methods.
    pub fn claim_open(&mut self, ticket: Option<u64>) -> Option<(u64, BTreeSet<u32>)> {
        let ticket = ticket?;
        let held = self.held_for.remove(&ticket)?;
        for &chunk in &held {
            if self.openable(chunk) {
                self.open.set(chunk);
            }
        }
        Some((ticket, held))
    }

    /// Closes what [`Self::claim_open`] opened: the chunks taken leave the
    /// reservation, the rest stay held.
    pub fn claim_close(&mut self, opened: Option<(u64, BTreeSet<u32>)>) {
        let Some((ticket, held)) = opened else {
            return;
        };
        let mut kept = BTreeSet::new();
        for chunk in held {
            if self.open.get(chunk) {
                self.open.clear(chunk);
                kept.insert(chunk);
            }
        }
        self.held_for.insert(ticket, kept);
    }

    /// A watch on room given back from now on: what a rise short of free
    /// room waits on.
    #[must_use]
    pub fn room_watch(&self) -> RoomWatch {
        RoomWatch {
            seen: self.given.gives.load(std::sync::atomic::Ordering::Acquire),
            given: std::sync::Arc::clone(&self.given),
        }
    }

    /// The span's chunks under `rung`'s activation that hold state: what a
    /// rise to `rung` must move out ([`Self::plan_rung`]).
    #[must_use]
    pub fn held_in_span(&self, rung: u32) -> u32 {
        let limit = self.total - self.act_chunks(rung);
        u32::try_from(self.held_in(limit, self.state_limit()).len()).unwrap_or(u32::MAX)
    }

    // ---- accounting ----

    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }

    pub fn total_chunks(&self) -> u32 {
        self.total
    }

    pub fn reserved_chunks(&self) -> u32 {
        self.reserved
    }

    /// Chunks opened for a slot shape, waiting slots included.
    pub fn state_chunks(&self) -> u32 {
        self.held
    }

    /// Chunks with memory under them for the pool's use: the committed
    /// segments' and, where the span is backed on demand, the activation's
    /// in span segments not committed here (the engine backs its carve).
    pub fn committed_chunks(&self) -> u32 {
        self.committed_for(self.act)
    }

    /// [`Self::committed_chunks`] with an activation of `act` chunks.
    fn committed_for(&self, act: u32) -> u32 {
        let committed = self.segments.committed_chunks();
        if !self.span_on_demand {
            return committed;
        }
        let from = self.total - act;
        let carve: u32 = (self.top_segment..self.segments.count())
            .filter(|&segment| self.segments.state(segment) == SegmentState::Uncommitted)
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.saturating_sub(range.start.max(from))
            })
            .sum();
        committed + carve
    }

    /// Activation chunks a fire at `rung` tokens takes: the smallest rung at
    /// or above it, else the largest; with the chunks the guest term takes
    /// at that rung's lanes ([`Self::set_fire`]) and the engine's standing
    /// guest bytes ([`Self::set_standing`]), one ceil over the two.
    pub fn act_chunks(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(0, |&(tokens, chunks)| {
                chunks.saturating_add(self.fire_chunks(tokens))
            })
    }

    /// The chunks the guests take at a rung of `tokens`: what the worst
    /// fire of `min(tokens, max_lanes)` lanes lays for the pairs open
    /// ([`::engine::GuestTerm::bytes`]), beside it what the engine keeps
    /// standing for every live instance whatever the fire, and the rows
    /// such a fire reads out where they land beside the activation
    /// ([`Self::readout_bytes`]).
    fn fire_chunks(&self, tokens: u32) -> u32 {
        let bytes = self
            .fire
            .bytes(tokens.min(self.max_lanes))
            .saturating_add(self.standing)
            .saturating_add(self.readout_bytes(tokens));
        u32::try_from(bytes.div_ceil(self.chunk_bytes.max(1))).unwrap_or(u32::MAX)
    }

    /// The readout term at a rung of `tokens`: a fire there reads out no
    /// more rows than its tokens, nor than the most one fire reads
    /// ([`Self::set_readouts`]), each taking the bytes a row the engine
    /// states beside the activation; 0 where they land inside the span.
    #[must_use]
    pub fn readout_bytes(&self, tokens: u32) -> u64 {
        u64::from(tokens.min(self.readout_rows)).saturating_mul(self.readout_row_bytes)
    }

    /// The readout term from now on, of the engine's
    /// [`::engine::ReadoutFacts`]: `row_bytes` a row beside the activation
    /// (0: inside the span, no term) for up to `rows` rows a fire (0: no
    /// bound but the fire's tokens). Nothing moves here, as with
    /// [`Self::set_fire`].
    pub fn set_readouts(&mut self, row_bytes: u64, rows: u32) {
        self.readout_row_bytes = row_bytes;
        self.readout_rows = if rows == 0 { u32::MAX } else { rows };
    }

    /// The most rows one fire reads out ([`Self::set_readouts`]):
    /// `u32::MAX` where nothing but its tokens bounds them.
    #[must_use]
    pub fn readout_rows(&self) -> u32 {
        self.readout_rows
    }

    /// The bytes the guest term takes at the rung fires of `rung` tokens
    /// take, before they are rounded to chunks.
    #[must_use]
    pub fn fire_bytes(&self, rung: u32) -> u64 {
        self.fire.bytes(self.rung_of(rung).min(self.max_lanes))
    }

    /// The guest term of the pairs that can be in a fire from now on,
    /// built for fires of up to [`Self::max_lanes`] lanes. Nothing moves
    /// here: a term that raises the rung's need is taken by a raise
    /// ([`Self::plan_rung`], [`Self::apply_rung`]), and one that drops
    /// gives the room back at the next lower ([`Self::set_rung`]).
    /// The term it replaces is handed back.
    pub fn set_fire(&mut self, fire: ::engine::GuestTerm) -> ::engine::GuestTerm {
        std::mem::replace(&mut self.fire, fire)
    }

    /// Makes the guest term the one of `pairs` in place
    /// ([`::engine::GuestTerm::fit`]): whether it changed.
    pub fn fit_fire(&mut self, pairs: &[::engine::FirePair]) -> bool {
        self.fire.fit(pairs, self.max_lanes)
    }

    /// See [`Self::set_fire`].
    pub fn fire(&self) -> &::engine::GuestTerm {
        &self.fire
    }

    /// The engine's standing guest bytes from now on
    /// ([`::engine::Engine::standing_after`]): the span bytes it keeps for
    /// the live instances of registered programs beside every fire, with
    /// what the binds under way may add. Set aside below the carve at
    /// every rung ([`Self::act_chunks`]); nothing moves here, as with
    /// [`Self::set_fire`]. The bytes it replaces are handed back.
    pub fn set_standing(&mut self, bytes: u64) -> u64 {
        std::mem::replace(&mut self.standing, bytes)
    }

    /// See [`Self::set_standing`].
    #[must_use]
    pub fn standing(&self) -> u64 {
        self.standing
    }

    /// Whether the guests take nothing of the span at any rung: no fire
    /// term, nothing standing, and no readout row beside the activation.
    #[must_use]
    pub fn guests_zero(&self) -> bool {
        self.fire.is_zero() && self.standing == 0 && self.readout_row_bytes == 0
    }

    /// The most lanes a fire carries: the engine's `FireLimits::max_lanes`.
    pub fn set_max_lanes(&mut self, lanes: u32) {
        self.max_lanes = lanes.max(1);
    }

    /// See [`Self::set_max_lanes`].
    #[must_use]
    pub fn max_lanes(&self) -> u32 {
        self.max_lanes
    }

    /// The engine states its activation span covers `chunks` from the
    /// range's end as it stands (`ChunkFacts::act_span_chunks`), `None`
    /// where nothing of its own bounds it: the statement in force from
    /// here, read where a rung is fitted ([`Self::act_span`]). Nothing is
    /// kept that an earlier statement decided.
    pub fn set_act_span(&mut self, chunks: Option<u32>) {
        self.act_span = chunks;
    }

    /// See [`Self::set_act_span`].
    #[must_use]
    pub fn act_span(&self) -> Option<u32> {
        self.act_span
    }

    /// The rung fires of `rung` tokens take: the smallest at or above it,
    /// else the largest (`rung` itself where no rung is stated).
    #[must_use]
    pub fn rung_of(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(rung, |&(tokens, _)| tokens)
    }

    /// The activation chunks set aside now.
    pub fn act_reserved(&self) -> u32 {
        self.act
    }

    /// Every rung's tokens, ascending.
    pub fn rungs(&self) -> Vec<u32> {
        self.act_chunks.iter().map(|&(tokens, _)| tokens).collect()
    }

    /// The rung the activation is set aside for now: the largest whose
    /// chunks it holds, the fire term's included (0 where no rung is
    /// stated, or where a fire term registered since leaves it holding
    /// none until a raise).
    pub fn rung(&self) -> u32 {
        self.act_chunks
            .iter()
            .map(|&(tokens, _)| tokens)
            .filter(|&tokens| self.act_chunks(tokens) <= self.act)
            .max()
            .unwrap_or(0)
    }

    /// The largest rung's activation chunks: the top span, committed from
    /// the load.
    pub fn act_max(&self) -> u32 {
        self.act_max
    }

    /// The most disjoint runs of `run` adjacent chunks the pool could ever
    /// hold for state at the smallest rung, each inside one segment past the
    /// reserved chunks: what a need of runs is fitting against alone.
    #[must_use]
    pub fn runs_most(&self, run: u32) -> u32 {
        let run = run.max(1);
        let limit = self.total.saturating_sub(self.act_chunks(0));
        (0..self.segments.count())
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(limit).saturating_sub(range.start.max(self.reserved)) / run
            })
            .sum()
    }

    /// The longest run of adjacent chunks any arrangement of the pool lays
    /// for state: inside one segment, past the reserved chunks, below the
    /// narrowest rung's activation. A run longer is never laid, whatever
    /// moves.
    #[must_use]
    pub fn longest_run_most(&self) -> u32 {
        let limit = self.total.saturating_sub(self.act_chunks(0));
        (0..self.segments.count())
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(limit).saturating_sub(range.start.max(self.reserved))
            })
            .max()
            .unwrap_or(0)
    }

    /// Held state chunks lying where the engine's state pool does not
    /// hold them under the rung in force, `(logical, physical)`: at or past
    /// the current rung's activation (the carve the activation writes), or
    /// in a segment with no memory under it (a span segment backed on
    /// demand among them). Empty
    /// where every holder's refs read memory the engine backs for state.
    #[must_use]
    pub fn held_unbacked(&self) -> Vec<(u32, u32)> {
        let limit = self.state_limit();
        (0..self.total)
            .filter(|&logical| self.chunks[logical as usize].shape != NO_SHAPE)
            .map(|logical| (logical, self.logical_to_physical[logical as usize]))
            .filter(|&(_, physical)| {
                let segment = self.segments.of(physical);
                physical >= limit || self.segments.state(segment) == SegmentState::Uncommitted
            })
            .collect()
    }

    /// Every chunk opened for a shape, `(logical, physical)`.
    #[must_use]
    pub fn held_physical(&self) -> Vec<(u32, u32)> {
        (0..self.total)
            .filter(|&logical| self.chunks[logical as usize].shape != NO_SHAPE)
            .map(|logical| (logical, self.logical_to_physical[logical as usize]))
            .collect()
    }

    /// Every free chunk a take could open now, `(logical, physical)`: what
    /// the engine must resolve to backed memory at the same physical.
    #[must_use]
    pub fn takeable_physical(&self) -> Vec<(u32, u32)> {
        self.open
            .iter_in(0, self.total)
            .map(|physical| (self.physical_to_logical[physical as usize], physical))
            .collect()
    }

    /// The first segment under the largest rung's activation (the segment
    /// count where activation lives apart).
    pub fn top_segment(&self) -> u32 {
        self.top_segment
    }

    /// State chunks lie at physical ids below this: the activation of the
    /// current rung takes the rest.
    pub fn state_limit(&self) -> u32 {
        self.total - self.act.max(self.carved)
    }

    /// Committed segments below the top span: what `commit_segments(n)`
    /// backs as a prefix, `0..n`.
    pub fn prefix_segments(&self) -> u32 {
        (0..self.top_segment)
            .take_while(|&segment| self.segments.state(segment) != SegmentState::Uncommitted)
            .count() as u32
    }

    fn rung_budget(&self, act: u32) -> Result<(), SlotError> {
        let needed = self.reserved + self.held + act;
        let committed = self.committed_for(act);
        if needed > committed {
            return Err(SlotError::Budget { needed, committed });
        }
        Ok(())
    }

    /// State chunks at physical ids `from..to`.
    fn held_in(&self, from: u32, to: u32) -> Vec<u32> {
        (from..to)
            .filter(|&chunk| !self.free.get(chunk) && chunk >= self.reserved)
            .collect()
    }

    /// Sets the activation aside for fires up to `rung` tokens; refused when
    /// the state already held leaves too little, or when state sits in the
    /// span the rung's activation takes (raise it with [`Self::plan_rung`]
    /// and [`Self::apply_rung`] then).
    pub fn set_rung(&mut self, rung: u32) -> Result<(), SlotError> {
        let act = self.act_chunks(rung);
        self.rung_budget(act)?;
        let in_way = self.held_in(self.total - act, self.state_limit()).len() as u32;
        if in_way > 0 {
            return Err(SlotError::Occupied {
                rung,
                chunks: in_way,
            });
        }
        self.set_act(act);
        Ok(())
    }

    fn set_act(&mut self, act: u32) {
        self.changes += 1;
        let was = self.state_limit();
        self.act = act;
        self.carved = if self.span_on_demand {
            self.carved.max(act)
        } else {
            act
        };
        let now = self.state_limit();
        for chunk in was.min(now)..was.max(now) {
            self.reopen(chunk);
        }
    }

    /// The activation is `act` again, the engine's carve with it, after a
    /// raise whose copy landed and whose carve the engine refused: the
    /// state the copy moved stays where it went.
    pub fn restore_act(&mut self, act: u32) {
        self.set_act(act);
        self.engine_narrowed();
    }

    /// A residency move's copy reads `chunks` (physical) from here until
    /// [`Self::moved_out`]: no raise takes them back meanwhile.
    pub fn moving_out(&mut self, chunks: &[u32]) {
        for &chunk in chunks {
            *self.moving_out.entry(chunk).or_default() += 1;
        }
    }

    /// The copy [`Self::moving_out`] named has landed, or will not.
    pub fn moved_out(&mut self, chunks: &[u32]) {
        for chunk in chunks {
            if let Some(count) = self.moving_out.get_mut(chunk) {
                *count -= 1;
                if *count == 0 {
                    self.moving_out.remove(chunk);
                }
            }
        }
    }

    /// The engine narrowed its carve to the activation set aside now (a
    /// lower's notice landed, `Engine::set_rung`): the span above it opens
    /// for state ([`Self::carved`]).
    pub fn engine_narrowed(&mut self) {
        let was = self.state_limit();
        self.carved = self.act;
        let now = self.state_limit();
        for chunk in was.min(now)..was.max(now) {
            self.reopen(chunk);
        }
    }

    /// The moves that clear rung `rung`'s activation span of state: each
    /// state chunk in it to the lowest free chunk below it. Empty when
    /// nothing is in the way (or the rung takes no more than now).
    pub fn plan_rung(&self, rung: u32) -> Result<RungEvacuation, SlotError> {
        let act = self.act_chunks(rung);
        self.rung_budget(act)?;
        let limit = self.total - act;
        // A move's copy still reads chunks the raise would take back: it
        // waits for the copy to land (its end asks the walk again), never
        // laying its activation over state a copy is reading.
        // Chunks a served restore's reservation keeps there are as taken:
        // laid under the activation, its bind's runs were gone.
        let read = self.moving_out.range(limit..).count()
            + usize::from(self.reserved_in(limit, self.total));
        if read > 0 {
            return Err(SlotError::Occupied {
                rung,
                chunks: u32::try_from(read).unwrap_or(u32::MAX),
            });
        }
        let sources = self.held_in(limit, self.state_limit());
        // A run of chunks moves whole, to a free run below (its first chunk
        // and every chunk after it at the next physical ones): moved chunk
        // by chunk, a run in the span came back split, its first logical
        // chunk's run ending past a gap ("a guest item ... lies in a run of
        // 2 chunk(s) ... ends past a gap", and corrupted reads).
        let targets: BTreeSet<u32> = self.open.iter_in(0, limit).collect();
        let Some(moves) = self.moves_keeping_runs(&sources, targets, false) else {
            return Err(SlotError::NoRoom {
                rung,
                chunks: sources.len() as u32,
            });
        };
        Ok(RungEvacuation {
            rung,
            from: self.rung(),
            act,
            moves,
        })
    }

    /// The chunks raising the activation to `rung` lacks now: past what is
    /// committed, and in free chunks below its span for the state in it.
    /// 0 when the raise fits.
    /// Where `rung`'s activation and the reserved chunks could never fit
    /// the pool, even with all state moved off it: the chunks they take
    /// and the pool's total.
    #[must_use]
    pub fn rung_past(&self, rung: u32, own: u32) -> Option<(u32, u32)> {
        let need = self
            .reserved
            .saturating_add(self.act_chunks(rung))
            .saturating_add(own);
        (need > self.total).then_some((need, self.total))
    }

    pub fn rung_short(&self, rung: u32) -> u32 {
        let act = self.act_chunks(rung);
        let budget = (self.reserved + self.held + act).saturating_sub(self.committed_for(act));
        let limit = self.total - act;
        let sources = self.held_in(limit, self.state_limit()).len() as u32;
        let targets = self.open.iter_in(0, limit).take(sources as usize).count() as u32;
        let short = budget.max(sources - targets);
        // Runs in the way move whole, each to a free run below: free chunks
        // enough, scattered, open none. A compaction below the raise's
        // limit opens them where it can (the walk makes it,
        // [`super::shared::SharedPool::compaction_for_raise`]); where none
        // can, what the runs take is short, for the planner to free (the
        // simulation's field seed 603: read as fitting, the raise waited
        // with nothing funded and nothing to move).
        match self.raise_runs_short(rung) {
            Some((run, pieces))
                if !self.compacting() && self.plan_compaction_below(run, pieces, limit).is_none() =>
            {
                let open = self.open.iter_in(0, limit).count() as u32;
                short.max((run * pieces + sources).saturating_sub(open).max(1))
            }
            _ => short,
        }
    }

    /// Carries out `evacuation` and takes its rung; the fire at `epoch`
    /// carries the copy, and the chunks moved off stay closed until it
    /// retires. All or nothing.
    pub fn apply_rung(&mut self, evacuation: &RungEvacuation, epoch: u64) -> Result<(), SlotError> {
        let act = self.act_chunks(evacuation.rung);
        self.rung_budget(act)?;
        let limit = self.total - act;
        if act != evacuation.act {
            return Err(SlotError::StaleRung);
        }
        if let Some((head, run, at)) = self.splits_run(&evacuation.moves) {
            return Err(SlotError::SplitsRun { head, run, at });
        }
        let mut targets = BTreeSet::new();
        let froms: BTreeSet<u32> = evacuation.moves.iter().map(|step| step.from).collect();
        if !self.held_in(limit, self.state_limit()).iter().all(|chunk| froms.contains(chunk)) {
            return Err(SlotError::StaleRung);
        }
        for step in &evacuation.moves {
            // A run's chunks below the limit move with it (`plan_rung`).
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && step.to < limit
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(SlotError::StaleRung);
            }
        }
        for step in &evacuation.moves {
            // Fenced before the move reopens it, as a compaction's: a run's
            // member below the span (moved whole with the one in it) was
            // left open while its copy still read it, another owner's to
            // take mid-copy.
            self.fenced.set(step.from);
            self.move_chunk(*step);
            // Its destination too, until the copy into it has landed: let
            // go meanwhile, it would reopen and be handed to another owner
            // whose data the copy then overwrites ("the kernel wrote no
            // status at all").
            self.fenced.set(step.to);
        }
        if !evacuation.moves.is_empty() {
            self.fences
                .entry(epoch)
                .or_default()
                .extend(evacuation.moves.iter().flat_map(|step| [step.from, step.to]));
        }
        self.set_act(act);
        Ok(())
    }

    /// The fewest chunk moves that stand one free run of `run` adjacent
    /// chunks inside one committed segment, where free chunks are enough
    /// but scattered (a guest bind's run: pages and slots of many
    /// owners interleave across chunks). The window of `run` chunks whose
    /// held chunks are fewest is cleared: each held chunk in it moves to a
    /// free chunk outside it, the lowest first. A chunk moves on its
    /// logical id, so its holders keep their refs and the engine copies its
    /// bytes ([`Engine::evacuate`]). `None` where a run stands already,
    /// where a move this pool made is still fenced (one at a time), or
    /// where no window can be cleared: no free chunks outside it for what
    /// it holds, or something in it that does not move (a reserved chunk,
    /// a run taken whole, a fenced one).
    ///
    /// [`Engine::evacuate`]: ::engine::Engine::evacuate
    #[must_use]
    pub fn plan_compaction(&self, run: u32) -> Option<Vec<ChunkMove>> {
        self.plan_compaction_for(run, 1)
    }

    /// [`Self::plan_compaction`] for a need of `pieces` runs: while fewer
    /// than that stand free, one more is cleared beside those that do,
    /// neither its window nor its moves' destinations in them. Planned only
    /// where no run stands, a head short of a second run with free and
    /// emptiable chunks enough would wait with no move.
    pub fn plan_compaction_for(&self, run: u32, pieces: u32) -> Option<Vec<ChunkMove>> {
        self.plan_compaction_below(run, pieces, self.state_limit())
    }

    /// The runs a raise to `rung` finds no free run below its activation
    /// for, as `(run, pieces)` of the longest: each run of chunks in the
    /// raise's way moves whole to a free run below ([`Self::plan_rung`]),
    /// and where free chunks there stand scattered, none does. `None` where
    /// the raise's moves plan as the pool stands, or its way holds no run.
    #[must_use]
    pub fn raise_runs_short(&self, rung: u32) -> Option<(u32, u32)> {
        let act = self.act_chunks(rung);
        let limit = self.total - act;
        let sources = self.held_in(limit, self.state_limit());
        if sources.is_empty() {
            return None;
        }
        let targets: BTreeSet<u32> = self.open.iter_in(0, limit).collect();
        if self.moves_keeping_runs(&sources, targets.clone(), false).is_some() {
            return None;
        }
        // The runs in the way, by length: the longest that finds no room is
        // the one a compaction opens first.
        let mut lengths: BTreeMap<u32, u32> = BTreeMap::new();
        for (&head, &run) in &self.runs {
            let at = self.logical_to_physical[head as usize];
            if sources.contains(&at) {
                *lengths.entry(run).or_default() += 1;
            }
        }
        let (&run, &pieces) = lengths.iter().next_back()?;
        let standing = self.runs_below(run, u32::MAX, limit, |chunk| targets.contains(&chunk)).len();
        (standing < pieces as usize).then_some((run, pieces))
    }

    /// [`Self::plan_compaction_for`] below physical chunk `below`: a raise's
    /// activation limit, where the runs in its way go
    /// ([`Self::raise_runs_short`]).
    pub fn plan_compaction_below(&self, run: u32, pieces: u32, below: u32) -> Option<Vec<ChunkMove>> {
        let run = run.max(1);
        if run > self.segments.segment_chunks() || self.compacting() {
            return None;
        }
        let below = below.min(self.state_limit());
        let standing = self.runs_below(run, u32::MAX, below, |chunk| self.open.get(chunk));
        if standing.len() >= pieces.max(1) as usize {
            return None;
        }
        let kept: HashSet<u32> = standing.iter().flat_map(|&first| first..first + run).collect();
        let free_here = |chunk: u32| self.open.get(chunk) && !kept.contains(&chunk);
        // Each run's chunks by physical chunk, its first and its length: a
        // run moves whole, to a free run of its length outside the window
        // ([`Self::moves_keeping_runs`]). Held as unmovable, a pool of runs
        // laid side by side would open no window for another: a restore of
        // three runs of 57 chunks with 99 chunks free would wait for good,
        // its planner finding no step.
        let mut member_of: HashMap<u32, (u32, u32)> = HashMap::new();
        for (&head, &len) in &self.runs {
            let at = self.logical_to_physical[head as usize];
            for k in 0..len {
                member_of.insert(at + k, (at, len));
            }
        }
        // Never a chunk a residency move's copy still reads (an eviction's,
        // parked state's moving down): moved under it, the copy read a chunk
        // handed on. Reserved chunks are free and closed: neither a source
        // (not held) nor a destination (not open).
        // Nor one a move before still copies into or out of (fenced until
        // the fire carrying that copy retires): moved again, the first
        // move's retire unfenced the chunk the second still reads, and it
        // was handed on under that copy.
        let still = |chunk: u32| self.moving_out.contains_key(&chunk) || self.fenced.get(chunk);
        let movable = |chunk: u32| {
            chunk >= self.reserved
                && !self.free.get(chunk)
                && !still(chunk)
                && self.chunks[self.physical_to_logical[chunk as usize] as usize].shape != NO_SHAPE
                && member_of
                    .get(&chunk)
                    .is_none_or(|&(at, len)| (at..at + len).all(|member| !still(member)))
        };
        let limit = below.min(self.ceiling);
        let mut windows: Vec<(u32, u32)> = Vec::new();
        for segment in 0..self.segments.count() {
            if self.segments.state(segment) != SegmentState::Committed {
                continue;
            }
            let range = self.segments.range(segment);
            let end = range.end.min(limit);
            let mut start = range.start;
            while start + run <= end {
                let mut held = 0u32;
                let mut blocked = None;
                for chunk in start..start + run {
                    if kept.contains(&chunk) {
                        blocked = Some(chunk);
                        break;
                    }
                    if self.open.get(chunk) {
                        continue;
                    }
                    if movable(chunk) {
                        held += 1;
                    } else {
                        blocked = Some(chunk);
                        break;
                    }
                }
                if let Some(chunk) = blocked {
                    start = chunk + 1;
                    continue;
                }
                windows.push((held, start));
                start += 1;
            }
        }
        // The window that moves fewest first, each tried in turn: the first
        // whose holders (each run among them whole) find room outside it.
        windows.sort_unstable();
        windows.into_iter().find_map(|(_, start)| {
            let window = start..start + run;
            let mut sources: BTreeSet<u32> = BTreeSet::new();
            for chunk in window.clone().filter(|&chunk| !self.open.get(chunk)) {
                match member_of.get(&chunk) {
                    Some(&(at, len)) => sources.extend(at..at + len),
                    None => {
                        sources.insert(chunk);
                    }
                }
            }
            let targets: BTreeSet<u32> = self
                .open
                .iter_in(0, limit)
                .filter(|&chunk| free_here(chunk) && !window.contains(&chunk))
                .collect();
            let sources: Vec<u32> = sources.into_iter().collect();
            self.moves_keeping_runs(&sources, targets, false)
        })
    }

    /// Carries out `moves` ([`Self::plan_compaction`]'s), which the fire at
    /// `epoch` copies: each chunk moved off stays closed until that fire
    /// retires, and then the run it cleared stands free. All or nothing.
    pub fn apply_compaction(&mut self, moves: &[ChunkMove], epoch: u64) -> Result<(), SlotError> {
        if let Some((head, run, at)) = self.splits_run(moves) {
            return Err(SlotError::SplitsRun { head, run, at });
        }
        let mut targets = BTreeSet::new();
        for step in moves {
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && !self.free.get(step.from)
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(SlotError::StaleCompaction);
            }
        }
        for step in moves {
            // Fenced before the move reopens it: closed until the copy.
            self.fenced.set(step.from);
            self.move_chunk(*step);
            // Its destination too, until the copy into it has landed (see
            // `apply_rung`): never handed to another owner meanwhile.
            self.fenced.set(step.to);
        }
        if !moves.is_empty() {
            self.fences
                .entry(epoch)
                .or_default()
                .extend(moves.iter().flat_map(|step| [step.from, step.to]));
        }
        self.compacted = moves.iter().map(|step| step.from).collect();
        Ok(())
    }

    /// Whether logical chunk `logical` is open for a shape: it holds state.
    #[must_use]
    pub fn holds_chunk(&self, logical: u32) -> bool {
        self.chunks
            .get(logical as usize)
            .is_some_and(|meta| meta.shape != NO_SHAPE)
    }

    /// The moves that take the state in logical chunks `logicals` down to
    /// the lowest free chunks below the state limit: chunks the engine
    /// refused a fire for, its state lying where it will not read or write
    /// it (`engine::Error::Placement`). Chunks no longer held, or already
    /// below the limit and below every chunk free there, move nothing.
    #[must_use]
    pub fn plan_moves_of(&self, logicals: &[u32]) -> Vec<ChunkMove> {
        let limit = self.state_limit().min(self.ceiling);
        let sources: Vec<u32> = logicals
            .iter()
            .filter_map(|&logical| {
                let from = *self.logical_to_physical.get(logical as usize)?;
                (self.chunks[logical as usize].shape != NO_SHAPE
                    && !self.free.get(from)
                    && !self.fenced.get(from))
                .then_some(from)
            })
            .collect();
        let targets: BTreeSet<u32> = self.open.iter_in(0, limit).collect();
        self.moves_keeping_runs(&sources, targets, true).unwrap_or_default()
    }

    /// Whether `moves` would split a run of chunks ([`Self::runs`]), and
    /// how, by name: a run lies at its first chunk's physical chunk and the
    /// next ones, in one segment, and the engine addresses it so (a piece
    /// from its first chunk's first byte); moved, every chunk of it moves,
    /// each to the chunk after its predecessor's destination, in one
    /// segment. Read on the moves alone, whoever planned them: the check
    /// the movers' own placement does not stand in for (runs split by a
    /// raise moving them chunk by chunk).
    #[must_use]
    pub fn splits_run(&self, moves: &[ChunkMove]) -> Option<(u32, u32, u32)> {
        let to_of: HashMap<u32, u32> = moves.iter().map(|step| (step.from, step.to)).collect();
        for (&head, &run) in &self.runs {
            let Some(&start) = self.logical_to_physical.get(head as usize) else {
                continue;
            };
            let moved: Vec<Option<u32>> = (start..start + run).map(|at| to_of.get(&at).copied()).collect();
            if moved.iter().all(Option::is_none) {
                continue;
            }
            let Some(first) = moved[0] else {
                let at = moved.iter().position(Option::is_some).unwrap_or(0);
                return Some((head, run, u32::try_from(at).unwrap_or(u32::MAX)));
            };
            for (k, to) in (0..run).zip(&moved) {
                let whole = *to == Some(first + k)
                    && self.segments.of(first + k) == self.segments.of(first);
                if !whole {
                    return Some((head, run, k));
                }
            }
        }
        None
    }

    /// Moves of the chunks at physical `sources` to `targets` (free
    /// chunks), each run of chunks ([`Self::runs`]) moved whole to a free
    /// run of its length, its first chunk first: a member named moves its
    /// whole run with it, never alone. `partial`: what finds no room stays
    /// and the rest moves; else `None` unless all fit.
    fn moves_keeping_runs(
        &self,
        sources: &[u32],
        mut targets: BTreeSet<u32>,
        partial: bool,
    ) -> Option<Vec<ChunkMove>> {
        // Each run's chunks by physical chunk: its first and its length.
        let mut member_of: HashMap<u32, (u32, u32)> = HashMap::new();
        for (&head, &run) in &self.runs {
            let Some(&start) = self.logical_to_physical.get(head as usize) else {
                continue;
            };
            for k in 0..run {
                member_of.insert(start + k, (start, run));
            }
        }
        let mut runs: BTreeSet<(u32, u32)> = BTreeSet::new();
        let mut singles: Vec<u32> = Vec::new();
        for &from in sources {
            match member_of.get(&from) {
                Some(&run) => {
                    runs.insert(run);
                }
                None => singles.push(from),
            }
        }
        let mut moves = Vec::new();
        for (start, run) in runs {
            let Some(to) = self.free_run_in(&targets, run) else {
                if partial {
                    continue;
                }
                return None;
            };
            for k in 0..run {
                targets.remove(&(to + k));
                moves.push(ChunkMove {
                    logical: self.physical_to_logical[(start + k) as usize],
                    from: start + k,
                    to: to + k,
                });
            }
        }
        for from in singles {
            let Some(to) = targets.pop_first() else {
                if partial {
                    break;
                }
                return None;
            };
            moves.push(ChunkMove {
                logical: self.physical_to_logical[from as usize],
                from,
                to,
            });
        }
        Some(moves)
    }

    /// Takes back `moves`, applied with [`Self::apply_compaction`] and
    /// refused by the engine before its copy: each chunk goes back to where
    /// it lay, and its fence lifts.
    pub fn undo_compaction(&mut self, moves: &[ChunkMove]) {
        for step in moves.iter().rev() {
            for chunks in self.fences.values_mut() {
                chunks.retain(|&chunk| chunk != step.from && chunk != step.to);
            }
            self.fenced.clear(step.from);
            self.fenced.clear(step.to);
            self.move_chunk(ChunkMove {
                logical: step.logical,
                from: step.to,
                to: step.from,
            });
        }
        self.fences.retain(|_, chunks| !chunks.is_empty());
        self.compacted.clear();
    }

    /// The state budget a guest's room is read against, in one line: what
    /// `free_chunks` (the open chunks within the headroom) and
    /// `committable_runs` (segments still to commit within the budget)
    /// stop at. The committed chunks, less the reserved, the held and the
    /// activation's, are the headroom (bounded by the ceiling where it sits
    /// below the pool); the uncommitted segments below the top span, and
    /// what the pool could still grow to, bound what commits.
    #[must_use]
    pub fn budget_line(&self) -> String {
        let uncommitted: Vec<u32> = self
            .segments
            .uncommitted()
            .filter(|&segment| segment < self.top_segment)
            .collect();
        let backed = uncommitted.iter().filter(|&&segment| self.backs(segment)).count();
        let lent = self.span_lent();
        format!(
            "budget: total={} reserved={} held={} act={} committed={} headroom={} \
             ceiling={} state_limit={} open={} free_chunks={} segments={} top_segment={} \
             uncommitted_below_top={} of_them_backed={} growable={} committable_one_chunk={} \
             span_lent={} span_lent_backed={} span_reclaim_at_top_rung={}",
            self.total,
            self.reserved,
            self.held,
            self.act,
            self.committed_chunks(),
            self.headroom(),
            self.ceiling,
            self.state_limit(),
            self.open.count,
            self.free_chunks(),
            self.segments.count(),
            self.top_segment,
            uncommitted.len(),
            backed,
            self.growable_chunks(),
            self.committable_runs(1),
            lent.0,
            lent.1,
            lent.2,
        )
    }

    /// What stands between free chunks and a run of `run` adjacent ones,
    /// said in one line for a head short of a run (the soaks' "free chunks
    /// and no run" stalls): the budget's terms and its headroom, the free
    /// and open chunks, chunks opened for a shape and holding no slot,
    /// fenced chunks, the segment's length, each committed segment's
    /// longest open stretch (the five longest), the open runs of `run`,
    /// and what a compaction would move.
    #[must_use]
    pub fn census(&self, run: u32) -> String {
        let limit = self.state_limit().min(self.ceiling);
        let free = (0..self.total).filter(|&chunk| self.free.get(chunk)).count();
        let past_limit = (limit..self.total).filter(|&chunk| self.free.get(chunk)).count();
        let open = self.open.iter_in(0, self.total).count();
        let fenced = (0..self.total).filter(|&chunk| self.fenced.get(chunk)).count();
        let empty_opened = self
            .chunks
            .iter()
            .filter(|meta| meta.shape != NO_SHAPE && meta.used == 0)
            .count();
        let mut by_shape: BTreeMap<ShapeId, (u32, u32)> = BTreeMap::new();
        for meta in self.chunks.iter().filter(|meta| meta.shape != NO_SHAPE) {
            let entry = by_shape.entry(meta.shape).or_default();
            entry.0 += 1;
            if meta.used == 0 {
                entry.1 += 1;
            }
        }
        let mut stretches: Vec<(u32, u32)> = Vec::new();
        for segment in 0..self.segments.count() {
            if self.segments.state(segment) != SegmentState::Committed {
                continue;
            }
            let range = self.segments.range(segment);
            let (mut longest, mut stretch) = (0u32, 0u32);
            for chunk in range.start..range.end.min(limit) {
                stretch = if self.open.get(chunk) { stretch + 1 } else { 0 };
                longest = longest.max(stretch);
            }
            stretches.push((longest, segment));
        }
        stretches.sort_unstable_by(|a, b| b.cmp(a));
        stretches.truncate(5);
        let compaction = self.plan_compaction(run).map(|moves| moves.len());
        format!(
            "pool (scope: every chunk): run={run} committed={} reserved={} opened={} act={} \
             headroom={} free_any={free} free_open_for_state={open} \
             free_past_state_limit={past_limit} fenced={fenced} opened_empty={empty_opened} \
             shapes(opened,empty)={by_shape:?} segment_chunks={} state_limit={} ceiling={} \
             longest_open_by_segment={stretches:?} open_runs={} compaction_moves={compaction:?} \
             compacting={}",
            self.committed_chunks(),
            self.reserved,
            self.held,
            self.act,
            self.headroom(),
            self.segments.segment_chunks(),
            self.state_limit(),
            self.ceiling,
            self.free_chunk_runs(run),
            self.compacting(),
        )
    }

    /// Whether the chunks the last compaction moved off are still fenced:
    /// the run it clears is on its way. A rung raise's fence is not one.
    #[must_use]
    pub fn compacting(&self) -> bool {
        self.compacted.iter().any(|&chunk| self.fenced.get(chunk))
    }

    /// Re-fences the chunks a compaction fenced until `from` to `to`: the
    /// engine took the copy and waits it out ([`Self::drains_through`]'s
    /// counterpart for chunk moves).
    pub fn compaction_copied(&mut self, from: u64, to: u64) {
        if from > to {
            self.refence(from, to);
        }
    }

    /// Takes back `evacuation`, applied with [`Self::apply_rung`] and
    /// refused by the engine before its copy: each moved chunk goes back to
    /// where it lay (the engine's table never changed), its fence lifts, and
    /// the activation is `act` again, the rung in hand.
    pub fn undo_rung(&mut self, evacuation: &RungEvacuation, act: u32) {
        for step in evacuation.moves.iter().rev() {
            for chunks in self.fences.values_mut() {
                chunks.retain(|&chunk| chunk != step.from && chunk != step.to);
            }
            self.fenced.clear(step.from);
            self.fenced.clear(step.to);
            self.move_chunk(ChunkMove {
                logical: step.logical,
                from: step.to,
                to: step.from,
            });
        }
        self.fences.retain(|_, chunks| !chunks.is_empty());
        self.set_act(act);
        // The engine refused it: its carve is the one it held.
        self.engine_narrowed();
    }

    /// Moves a held chunk onto free physical chunk `step.to`, as the
    /// engine's `ChunkTable::apply` does: the free logical id there takes the
    /// moved one's old place.
    #[track_caller]
    fn move_chunk(&mut self, step: ChunkMove) {
        self.changes += 1;
        let meta = self.chunks[step.logical as usize];
        let displaced = self.physical_to_logical[step.to as usize];
        // A chunk let go while its move was under way (its process ended,
        // or was torn down, before the engine took or refused the copy):
        // its logical id goes back as the engine's table has it, both
        // physical chunks free, no shape to keep count of. Read as a shape,
        // its free id (`NO_SHAPE`) would index past the shapes and panic
        // (on a SIGTERM's teardown).
        if meta.shape == NO_SHAPE {
            self.logical_to_physical[step.logical as usize] = step.to;
            self.logical_to_physical[displaced as usize] = step.from;
            self.physical_to_logical[step.to as usize] = step.logical;
            self.physical_to_logical[step.from as usize] = displaced;
            self.reopen(step.from);
            self.reopen(step.to);
            return;
        }
        self.assert_backed("a move", step.logical, step.to);
        self.open.clear(step.to);
        self.free.clear(step.to);
        self.free.set(step.from);
        self.logical_to_physical[step.logical as usize] = step.to;
        self.logical_to_physical[displaced as usize] = step.from;
        self.physical_to_logical[step.to as usize] = step.logical;
        self.physical_to_logical[step.from as usize] = displaced;
        self.segments.let_go(step.from);
        self.segments.hold(step.to);
        let state = &mut self.shapes[meta.shape as usize];
        let spc = state.spc();
        if meta.used > 0 && meta.used < spc {
            let chunks = state.partial.entry(meta.class).or_default();
            chunks.remove(&(Reverse(meta.used), step.from, step.logical));
            chunks.insert((Reverse(meta.used), step.to, step.logical));
        }
        self.reopen(step.from);
    }

    /// Committed chunks neither held nor set aside for activation, and
    /// below the ceiling ([`Self::set_ceiling`]): a committed chunk the
    /// engine binds none of yet is no room, so the walker commits ahead
    /// of a waiter rather than count it.
    pub fn headroom(&self) -> u32 {
        let all = self
            .committed_chunks()
            .saturating_sub(self.reserved + self.held + self.act);
        if self.ceiling >= self.total {
            return all;
        }
        let below: u32 = (0..self.segments.count())
            .filter(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(self.ceiling).saturating_sub(range.start)
            })
            .sum();
        all.min(below.saturating_sub(self.reserved + self.held))
    }

    pub fn shape(&self, shape: ShapeId) -> Option<&SlotShape> {
        self.shapes.get(shape as usize).map(|state| &state.shape)
    }

    /// The engine's slot reference: `chunk * slots_per_chunk + index`.
    pub fn slot_ref(&self, slot: Slot) -> u32 {
        let shape = self.chunks[slot.chunk as usize].shape;
        slot.chunk * self.shapes[shape as usize].spc() + slot.index
    }

    /// Where logical chunk `chunk` lies now, when it is reserved or open.
    pub fn physical(&self, chunk: u32) -> Option<u32> {
        (chunk < self.reserved || self.chunks[chunk as usize].shape != NO_SHAPE)
            .then(|| self.logical_to_physical[chunk as usize])
    }

    /// The slots chunk `chunk` (logical) holds or keeps for an epoch: 0 for
    /// a free chunk.
    #[must_use]
    pub fn chunk_used(&self, chunk: u32) -> u32 {
        self.chunks.get(chunk as usize).map_or(0, |meta| meta.used)
    }

    /// The physical chunks state may open now ([`Self::free_chunks`]'s).
    #[must_use]
    pub fn open_chunks(&self) -> Vec<u32> {
        self.open.iter_in(0, self.total).collect()
    }

    /// The open chunks a guest's state may take ([`Self::chunk_runs_for`]'s):
    /// below the activation the rung lends and the ceiling.
    #[must_use]
    pub fn state_open_chunks(&self) -> Vec<u32> {
        self.open.iter_in(0, self.state_limit().min(self.ceiling)).collect()
    }

    /// The logical -> physical table the engine keeps: a permutation, free
    /// logical ids included.
    pub fn chunk_table(&self) -> &[u32] {
        &self.logical_to_physical
    }

    /// Whether `slot` of `shape` is handed out: its chunk open for the shape
    /// and the slot not free in it (held, or waiting on a fire's epoch).
    pub fn is_held(&self, shape: ShapeId, slot: Slot) -> bool {
        let Some(meta) = self.chunks.get(slot.chunk as usize) else {
            return false;
        };
        let Some(state) = self.shapes.get(shape as usize) else {
            return false;
        };
        if meta.shape != shape || slot.index >= state.spc() {
            return false;
        }
        let word = meta.frame as usize * state.words + (slot.index / WORD) as usize;
        state.free[word] & (1 << (slot.index % WORD)) == 0
    }

    /// The shape logical chunk `chunk` is open for.
    pub fn chunk_shape(&self, chunk: u32) -> Option<ShapeId> {
        Some(self.chunks[chunk as usize].shape).filter(|&shape| shape != NO_SHAPE)
    }

    /// The bytes the chunks opened now for any of `shapes` hold free: their
    /// unused slots, room no other shape takes until a chunk empties.
    #[must_use]
    pub fn slack_of(&self, shapes: &[ShapeId]) -> u64 {
        self.chunks
            .iter()
            .filter(|meta| meta.shape != NO_SHAPE && shapes.contains(&meta.shape))
            .map(|meta| {
                let shape = &self.shapes[meta.shape as usize].shape;
                u64::from(shape.slots_per_chunk.saturating_sub(meta.used)) * shape.slot_bytes
            })
            .sum()
    }

    /// Slots `holder` holds references on.
    pub fn held_by(&self, holder: &Holder) -> u64 {
        self.holders.get(holder).copied().unwrap_or(0)
    }

    /// References on `slot`: `(working sets and snapshots, cache entries)`.
    pub fn refs(&self, slot: Slot) -> (u32, u32) {
        let meta = self.chunks[slot.chunk as usize];
        if meta.shape == NO_SHAPE {
            return (0, 0);
        }
        let shape = &self.shapes[meta.shape as usize];
        let refs = shape.refs[(meta.frame * shape.spc() + slot.index) as usize];
        (refs.live, refs.cache)
    }

    /// Free chunks a shape could open now without breaking the budget.
    pub fn free_chunks(&self) -> u32 {
        self.open.count.min(self.headroom())
    }

    /// Free slots `shape` could take now without breaking the budget.
    /// Free slots of `shape` in its partly filled chunks: what a take of
    /// its slots fills before it opens a chunk.
    #[must_use]
    pub fn partial_free(&self, shape: ShapeId) -> u64 {
        self.shapes
            .get(shape as usize)
            .map_or(0, |state| state.partial_free)
    }

    pub fn available(&self, shape: ShapeId) -> u64 {
        let Some(state) = self.shapes.get(shape as usize) else {
            return 0;
        };
        let chunks = self.free_chunks();
        state.partial_free + u64::from(chunks) * u64::from(state.spc())
    }

    /// Whether every `(shape, slots)` of `needs` could be taken at once:
    /// each shape's partly filled chunks first, then free chunks, which the
    /// shapes share.
    pub fn fits(&self, needs: &[(ShapeId, u64)]) -> bool {
        let room = u64::from(self.free_chunks());
        let mut chunks = 0u64;
        for &(shape, slots) in needs {
            let Some(state) = self.shapes.get(shape as usize) else {
                return slots == 0;
            };
            chunks += slots
                .saturating_sub(state.partial_free)
                .div_ceil(u64::from(state.spc()));
        }
        chunks <= room
    }

    /// Free chunks a shape could open once every segment below the
    /// activation the facts allow were committed: what the pool can grow
    /// to, short of the reserved chunks, the activation and what is held.
    pub fn growable_chunks(&self) -> u32 {
        let free = (0..self.state_limit())
            .filter(|&chunk| {
                self.free.get(chunk)
                    && !self.fenced.get(chunk)
                    && !matches!(
                        self.segments.state(self.segments.of(chunk)),
                        SegmentState::Draining { .. }
                    )
            })
            .count() as u32;
        free.min(
            self.total
                .saturating_sub(self.reserved + self.held + self.act),
        )
    }

    /// [`Self::fits`] against what the pool can grow to
    /// ([`Self::growable_chunks`]), committed or not.
    pub fn fits_grown(&self, needs: &[(ShapeId, u64)]) -> bool {
        self.chunks_for(needs)
            .is_some_and(|chunks| chunks <= u64::from(self.growable_chunks()))
    }

    fn chunks_for(&self, needs: &[(ShapeId, u64)]) -> Option<u64> {
        let mut chunks = 0u64;
        for &(shape, slots) in needs {
            let Some(state) = self.shapes.get(shape as usize) else {
                if slots == 0 {
                    continue;
                }
                return None;
            };
            chunks += slots
                .saturating_sub(state.partial_free)
                .div_ceil(u64::from(state.spc()));
        }
        Some(chunks)
    }

    /// Commits the lowest uncommitted segments below the activation and
    /// the ceiling, a prefix, until `needs` fit the committed chunks and every chunk below
    /// physical chunk `through` lies in a committed segment. The segments
    /// committed, lowest first: the engine backs them (`commit_segments`
    /// of the prefix they end) before anything lands in them. Stops short
    /// where no segment is left; the take that follows says so.
    pub fn commit_for(&mut self, needs: &[(ShapeId, u64)], through: u32) -> Vec<u32> {
        let mut committed = Vec::new();
        loop {
            let covered = (0..through.min(self.total)).all(|chunk| {
                self.segments.state(self.segments.of(chunk)) != SegmentState::Uncommitted
            });
            if covered && self.fits(needs) {
                return committed;
            }
            // Never into what the engine holds back (the ceiling): a
            // segment there is no room the engine binds, and backing it
            // takes the memory the engine keeps for its bodies.
            let Some(segment) = self
                .segments
                .uncommitted()
                .find(|&segment| segment < self.top_segment && self.backs(segment))
            else {
                return committed;
            };
            self.commit_segment(segment)
                .expect("an uncommitted segment commits");
            committed.push(segment);
        }
    }

    pub fn shape_demand(&self, shape: ShapeId) -> ShapeDemand {
        self.shapes
            .get(shape as usize)
            .map_or_else(ShapeDemand::default, |state| ShapeDemand {
                live: state.live,
                cached: state.cached,
                waiting: state.waiting,
            })
    }

    /// Demand counted slot by slot: what is live, and what is wanted.
    pub fn demand(&self) -> Demand {
        self.shapes.iter().fold(Demand::default(), |demand, state| {
            let bytes = state.shape.slot_bytes;
            Demand {
                live: demand.live + state.live * bytes,
                wanted: demand.wanted + (state.live + state.cached + state.waiting) * bytes,
            }
        })
    }

    /// Who holds the pool's slots, by kind of holder (the store's own, as
    /// guest state and role slots are taken, apart), and the chunks free
    /// and opened: what a stall is read by.
    #[must_use]
    pub fn held_census(&self) -> String {
        let (mut sets, mut cache, mut snapshots, mut weights, mut store) = (0, 0, 0, 0, 0);
        for (holder, &slots) in &self.holders {
            match holder {
                h if *h == super::chunks::STORE => store += slots,
                Holder::WorkingSet(_) => sets += slots,
                Holder::CacheEntry(_) => cache += slots,
                Holder::Snapshot(_) => snapshots += slots,
                Holder::Weights { .. } => weights += slots,
            }
        }
        format!(
            "slots: working_sets={sets} cache={cache} snapshots={snapshots} weights={weights} \
             store={store} ({} holders); chunks: opened={} free={} committed={} total={} \
             pending={}",
            self.holders.len(),
            self.held,
            self.free_chunks(),
            self.committed_chunks(),
            self.total,
            self.pending.values().map(Vec::len).sum::<usize>(),
        )
    }

    /// Records that `waiter` is parked short `slots` of `shape`, replacing
    /// what it was short before.
    pub fn wait(&mut self, waiter: u64, shape: ShapeId, slots: u64) {
        self.unwait(waiter);
        if let Some(state) = self.shapes.get_mut(shape as usize) {
            state.waiting += slots;
            self.waiters.insert(waiter, (shape, slots));
        }
    }

    /// Whether no holder holds a slot of the pool: no working set, cache
    /// entry, snapshot, store table or weight lies in any chunk.
    pub fn holds_nothing(&self) -> bool {
        self.holders.is_empty()
    }

    /// Whether any waiter is parked short of a slot.
    pub fn waiting(&self) -> bool {
        self.shapes.iter().any(|state| state.waiting > 0)
    }

    pub fn unwait(&mut self, waiter: u64) {
        if let Some((shape, slots)) = self.waiters.remove(&waiter) {
            self.shapes[shape as usize].waiting -= slots;
        }
    }

    // ---- slots ----

    /// `n` fresh slots of `shape` for `holder`, from the fullest partly
    /// filled chunks first, then from chunks opened lowest address first.
    /// All or nothing.
    pub fn acquire(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        n: u64,
    ) -> Result<Vec<Slot>, SlotError> {
        let state = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?;
        let spc = u64::from(state.spc());
        let fresh = n.saturating_sub(state.partial_free).div_ceil(spc);
        let room = u64::from(self.free_chunks());
        if fresh > room {
            let short = fresh - room;
            return Err(SlotError::Short {
                shape,
                slots: n - state.partial_free - room * spc,
                chunks: short as u32,
            });
        }
        let mut taken = Vec::with_capacity(n as usize);
        while (taken.len() as u64) < n {
            let chunk = match self.shapes[shape as usize].fullest(SINGLE) {
                Some(logical) => logical,
                None => self.open_chunk(shape, SINGLE),
            };
            self.fill(chunk, holder.is_live(), n - taken.len() as u64, &mut taken);
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Ok(taken)
    }

    /// The chunks of `shape`'s single slots ([`SINGLE`]), each with the
    /// slots it holds or keeps for an epoch: what a list compaction reads
    /// its sources and destinations from. `settled` is false for one a
    /// chunk move or a residency copy holds (fenced, or moving out): its
    /// slots stay where they lie meanwhile, neither a source nor a
    /// destination.
    #[must_use]
    pub fn single_chunks(&self, shape: ShapeId) -> Vec<SingleChunk> {
        self.chunks
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.shape == shape && meta.class == SINGLE)
            .map(|(logical, meta)| {
                let physical = self.logical_to_physical[logical];
                SingleChunk {
                    logical: logical as u32,
                    used: meta.used,
                    settled: !self.fenced.get(physical)
                        && !self.moving_out.contains_key(&physical),
                }
            })
            .collect()
    }

    /// `n` fresh slots of `shape` for `holder` from the partly filled
    /// chunks of its single slots outside `outside` (logical), fullest
    /// first, opening none: a list compaction's destinations. All or
    /// nothing.
    pub fn acquire_outside(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        n: u64,
        outside: &BTreeSet<u32>,
    ) -> Option<Vec<Slot>> {
        let state = self.shapes.get(shape as usize)?;
        let order: Vec<u32> = state
            .partial
            .get(&SINGLE)
            .into_iter()
            .flatten()
            .map(|&(_, _, logical)| logical)
            .filter(|logical| !outside.contains(logical))
            .collect();
        let spc = u64::from(state.spc());
        let room: u64 = order
            .iter()
            .map(|&logical| spc - u64::from(self.chunks[logical as usize].used))
            .sum();
        if room < n {
            return None;
        }
        let mut taken = Vec::with_capacity(n as usize);
        for logical in order {
            if taken.len() as u64 == n {
                break;
            }
            self.fill(logical, holder.is_live(), n - taken.len() as u64, &mut taken);
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Some(taken)
    }

    /// `n` adjacent slots of `shape` for `holder`, in index order, in one
    /// chunk: the lowest run of `n` free slots in the fullest partly filled
    /// chunk that has one, else the first `n` slots of a chunk opened for
    /// the shape. A run is never split across chunks: where no chunk has
    /// room for it whole, nothing is taken.
    ///
    /// # Errors
    ///
    /// [`SlotError::RunPastChunk`] where `n` is past a chunk's slots;
    /// [`SlotError::Short`] where no chunk has room for the run.
    pub fn acquire_run(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        n: u64,
    ) -> Result<Vec<Slot>, SlotError> {
        let state = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?;
        let per_chunk = state.spc();
        if n > u64::from(per_chunk) {
            return Err(SlotError::RunPastChunk {
                shape,
                slots: n,
                per_chunk,
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let n = n as u32;
        let found = self.partial_run(shape, n, |_, _| false);
        let (chunk, first) = match found {
            Some(found) => found,
            None if self.free_chunks() > 0 => (self.open_chunk(shape, n), 0),
            None => {
                return Err(SlotError::Short {
                    shape,
                    slots: u64::from(n),
                    chunks: 1,
                });
            }
        };
        let mut taken = Vec::with_capacity(n as usize);
        self.fill_run(chunk, holder.is_live(), first, n, &mut taken);
        *self.holders.entry(holder).or_default() += u64::from(n);
        Ok(taken)
    }

    /// Where a run of `n` slots of `shape` lies in a chunk already open for
    /// runs of its length, the slots `held(logical, index)` reads as held
    /// taken too: its logical chunk and first slot. A run lies only among
    /// runs of its own length: a run's hole left behind is one the next run
    /// of that length takes exactly, and a chunk whose runs all leave is
    /// free whole again. What [`Self::acquire_run`] takes, and what
    /// [`Self::chunk_runs_beside`] places on its view.
    fn partial_run(&self, shape: ShapeId, n: u32, held: impl Fn(u32, u32) -> bool) -> Option<(u32, u32)> {
        let state = self.shapes.get(shape as usize)?;
        state.partial.get(&n).and_then(|chunks| {
            chunks.iter().find_map(|&(_, _, logical)| {
                let frame = self.chunks[logical as usize].frame;
                state
                    .free_run_beside(frame, n, |index| held(logical, index))
                    .map(|first| (logical, first))
            })
        })
    }

    /// The free runs of `run` adjacent chunks state may open now, each
    /// inside one committed segment, below the current rung's activation
    /// and the ceiling ([`Self::state_limit`], [`Self::set_ceiling`]): each
    /// run's first physical chunk, lowest first, no two sharing a chunk,
    /// and no more than the budget's room holds ([`Self::headroom`]).
    fn chunk_runs_for(&self, run: u32) -> Vec<u32> {
        self.runs_where(run, self.headroom(), |chunk| self.open.get(chunk))
    }

    /// [`Self::chunk_runs_for`] over the chunks `free` reads as free and
    /// `room` chunks of the budget's headroom.
    fn runs_where(&self, run: u32, room: u32, free: impl Fn(u32) -> bool) -> Vec<u32> {
        self.runs_below(run, room, self.state_limit(), free)
    }

    /// [`Self::runs_where`] below physical chunk `below` (and the ceiling).
    fn runs_below(&self, run: u32, room: u32, below: u32, free: impl Fn(u32) -> bool) -> Vec<u32> {
        let run = run.max(1);
        if run > self.segments.segment_chunks() {
            return Vec::new();
        }
        let limit = below.min(self.ceiling);
        let most = (room / run) as usize;
        let mut firsts = Vec::new();
        for segment in 0..self.segments.count() {
            let range = self.segments.range(segment);
            let mut stretch = 0u32;
            for chunk in range.start..range.end.min(limit) {
                if !free(chunk) {
                    stretch = 0;
                    continue;
                }
                stretch += 1;
                if stretch == run {
                    firsts.push(chunk + 1 - run);
                    stretch = 0;
                }
            }
        }
        firsts.truncate(most);
        firsts
    }

    /// Whether `pieces` free runs of `run` adjacent chunks would stand
    /// once `own` has been placed first, in order, exactly where the takes
    /// would put it, read only: a [`OwnPiece::Chunks`] as
    /// [`Self::acquire_chunk_runs`] places it (the lowest free runs), a
    /// [`OwnPiece::Slots`] as [`Self::acquire_run`] does (a chunk open for
    /// runs of its length first, else the lowest free chunk, which runs of
    /// its length placed after it share). Nothing is taken: the pool is the
    /// same after.
    #[must_use]
    pub fn chunk_runs_beside(&self, own: &[OwnPiece], run: u32, pieces: u32) -> bool {
        let placement = self.place(own);
        placement.stands() && placement.free_runs(run) >= pieces
    }

    /// The free chunks `takes` (single slots by shape, as
    /// [`Self::acquire`] places them) would open, lowest first, the shapes'
    /// partly filled chunks filled first: read only. The one placement the
    /// planner's map and need read a restore's state by, so they see the
    /// room its unpark sees once its serve has taken that state. `None`
    /// where they do not fit.
    #[must_use]
    pub fn opens_for(&self, takes: &[(ShapeId, u64)]) -> Option<Vec<u32>> {
        let own: Vec<OwnPiece> = takes
            .iter()
            .map(|&(shape, slots)| OwnPiece::Take { shape, slots })
            .collect();
        self.free_beside(&own)
            .map(|(opened, _)| opened.into_iter().collect())
    }

    /// Where `own` (placed in order as the takes would place it) fits at
    /// all, the free chunks it opens: the runs of whole chunks, then the
    /// chunks its runs and takes of slots open beyond the partly filled
    /// ones. `None` where it does not fit: the one placement a restore's
    /// need and its standing are read by ([`super::roles::Roles::unpark_need`]).
    #[must_use]
    pub fn opens_beside(&self, own: &[OwnPiece]) -> Option<u32> {
        self.free_beside(own)
            .map(|(opened, _)| u32::try_from(opened.len()).unwrap_or(u32::MAX))
    }

    /// The chunks `own` opens placed in order as the takes would place it,
    /// and the headroom it leaves; `None` where it does not fit.
    /// The chunks `own` opens placed alone, on a pool holding nothing
    /// else, by the store's own placement: whole runs as they are, single
    /// slots of a shape packed into its chunks, runs of slots of one length
    /// and shape sharing chunks of their class (a chunk holds `spc / len`
    /// of them). What a never-fits check reads: [`Self::opens_beside`] is
    /// the room now, and a part placed alone in a chunk of its own each was
    /// no placement (a beam's runs of slots each read as a whole chunk,
    /// past the pool, refusing beams the budget serves).
    #[must_use]
    pub fn opens_alone(&self, own: &[OwnPiece]) -> u32 {
        use std::collections::BTreeMap;
        let mut chunks = 0u64;
        let mut takes: BTreeMap<ShapeId, u64> = BTreeMap::new();
        let mut classes: BTreeMap<(ShapeId, u32), u64> = BTreeMap::new();
        for piece in own {
            match *piece {
                OwnPiece::Chunks { run, pieces } => {
                    chunks += u64::from(run.max(1)) * u64::from(pieces);
                }
                OwnPiece::Take { shape, slots } => *takes.entry(shape).or_default() += slots,
                OwnPiece::Slots { shape, slots } if slots > 0 => {
                    *classes.entry((shape, slots)).or_default() += 1;
                }
                OwnPiece::Slots { .. } => {}
            }
        }
        let spc = |shape: ShapeId| self.shapes.get(shape as usize).map_or(1, |state| u64::from(state.spc().max(1)));
        for (shape, slots) in takes {
            chunks += slots.div_ceil(spc(shape));
        }
        for ((shape, len), runs) in classes {
            let per = (spc(shape) / u64::from(len)).max(1);
            chunks += runs.div_ceil(per);
        }
        u32::try_from(chunks).unwrap_or(u32::MAX)
    }

    fn free_beside(&self, own: &[OwnPiece]) -> Option<(std::collections::BTreeSet<u32>, u32)> {
        let placement = self.place(own);
        placement
            .stands()
            .then(|| (placement.opened.clone(), placement.room))
    }

    /// `own` placed in order by the takes themselves ([`Self::acquire`],
    /// [`Self::acquire_run`], [`Self::acquire_chunk_runs`]) on a scratch
    /// copy of the pool: the one placement every check reads, so what
    /// stands here is what the takes find. A view that mirrored the takes
    /// counted a shape's partly filled slots once per take and stood a
    /// pending bind's runs the restore's own takes then broke
    /// (`PendingBindShort`, its unpark asking no run).
    /// It stops at the first piece the takes refuse, as a restore does.
    #[must_use]
    pub fn place(&self, own: &[OwnPiece]) -> Placement {
        let mut scratch = self.scratch();
        let mut placed = 0usize;
        for piece in own {
            let took = match *piece {
                OwnPiece::Take { shape, slots } => {
                    scratch.acquire(PLACING, shape, slots).map(drop)
                }
                OwnPiece::Slots { shape, slots } => {
                    scratch.acquire_run(PLACING, shape, u64::from(slots)).map(drop)
                }
                // Whole chunks are placed alike whatever shape opens
                // them (no slot of another class shares one): the first.
                OwnPiece::Chunks { run, pieces } => {
                    scratch.acquire_chunk_runs(PLACING, 0, run, pieces).map(drop)
                }
            };
            if took.is_err() {
                break;
            }
            placed += 1;
        }
        let opened = self
            .open
            .iter_in(0, self.total)
            .filter(|&chunk| !scratch.open.get(chunk))
            .collect();
        let room = scratch.headroom();
        Placement {
            opened,
            room,
            placed,
            asked: own.len(),
            scratch,
        }
    }

    /// A copy of the pool for [`Self::place`]: every field a take reads
    /// and writes, none of the bookkeeping no take reads (the slots
    /// waiting on a fire, the holders' counts, the waiters, the fences'
    /// epochs), so a placement costs the chunk tables' copy alone.
    fn scratch(&self) -> SlotPool {
        SlotPool {
            chunk_bytes: self.chunk_bytes,
            total: self.total,
            reserved: self.reserved,
            shapes: self.shapes.clone(),
            act_chunks: self.act_chunks.clone(),
            act: self.act,
            carved: self.carved,
            moving_out: BTreeMap::new(),
            fire: self.fire.clone(),
            standing: self.standing,
            readout_row_bytes: self.readout_row_bytes,
            readout_rows: self.readout_rows,
            max_lanes: self.max_lanes,
            act_max: self.act_max,
            top_segment: self.top_segment,
            span_on_demand: self.span_on_demand,
            act_span: self.act_span,
            chunks: self.chunks.clone(),
            logical_to_physical: self.logical_to_physical.clone(),
            physical_to_logical: self.physical_to_logical.clone(),
            free: self.free.clone(),
            open: self.open.clone(),
            fences: BTreeMap::new(),
            fenced: self.fenced.clone(),
            ceiling: self.ceiling,
            segments: self.segments.clone(),
            held: self.held,
            pending: BTreeMap::new(),
            retired: self.retired,
            holders: HashMap::new(),
            waiters: HashMap::new(),
            runs: HashMap::new(),
            compacted: Vec::new(),
            given: std::sync::Arc::default(),
            changes: 0,
            held_for: self.held_for.clone(),
            next_ticket: self.next_ticket,
            // A placement's scratch gives nothing back for real.
            #[cfg(feature = "test-slot-fill")]
            poison: PoisonTap::default(),
        }
    }

    /// Free runs of `run` adjacent chunks inside one segment live state may
    /// take now ([`Self::acquire_chunk_runs`]), counted disjoint.
    #[must_use]
    pub fn free_chunk_runs(&self, run: u32) -> u32 {
        u32::try_from(self.chunk_runs_for(run).len()).unwrap_or(u32::MAX)
    }

    /// Runs of `run` adjacent chunks the segments still to commit hold
    /// below the activation: uncommitted segments under the top rung's
    /// span that the engine binds ([`Self::backs`]), the room
    /// [`super::shared::SharedPool::grow_step`] commits into, counted no
    /// further than what the pool may grow to ([`Self::growable_chunks`]).
    /// What a guest bind short of a free run waits on a commit for; none
    /// where no segment is left to commit within the budget.
    #[must_use]
    pub fn committable_runs(&self, run: u32) -> u32 {
        let run = run.max(1);
        if run > self.segments.segment_chunks() {
            return 0;
        }
        let limit = self.state_limit().min(self.ceiling);
        let prefix: u32 = self
            .segments
            .uncommitted()
            .filter(|&segment| segment < self.top_segment && self.backs(segment))
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(limit).saturating_sub(range.start) / run
            })
            .sum();
        let lent: u32 = self.lent_unbacked().iter().map(|&(_, chunks)| chunks / run).sum();
        (prefix + lent).min(self.growable_chunks() / run)
    }

    /// `pieces` runs of `run` adjacent free chunks for `holder`, each run
    /// inside one segment, the lowest first, each chunk opened for `shape`
    /// and all its slots taken: refs in order, a run's chunks in physical
    /// order. A run longer than one is kept so (an evacuation moves it
    /// whole) until its first chunk is free again. All or nothing.
    ///
    /// # Errors
    ///
    /// [`SlotError::RunPastSegment`] where `run` is past a segment's
    /// chunks; [`SlotError::NoChunkRun`] where fewer than `pieces` such
    /// runs are free, naming what was.
    pub fn acquire_chunk_runs(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        run: u32,
        pieces: u32,
    ) -> Result<Vec<Slot>, SlotError> {
        let spc = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?
            .spc();
        let run = run.max(1);
        let segment = self.segments.segment_chunks();
        if run > segment {
            return Err(SlotError::RunPastSegment { run, segment });
        }
        let live = holder.is_live();
        let firsts = self.chunk_runs_for(run);
        if firsts.len() < pieces as usize {
            return Err(SlotError::NoChunkRun {
                run,
                pieces,
                free_runs: u32::try_from(firsts.len()).unwrap_or(u32::MAX),
                free_chunks: self.free_chunks(),
            });
        }
        // Each piece inside one segment, whatever found it: on Metal a
        // segment is a buffer of its own, and a piece across its end reads
        // past it ("a guest piece ... lies across the end of its
        // segment").
        if let Some(&first) = firsts[..pieces as usize]
            .iter()
            .find(|&&first| self.segments.of(first) != self.segments.of(first + run - 1))
        {
            return Err(SlotError::RunAcrossSegment { first, run });
        }
        let mut taken = Vec::with_capacity((pieces * run * spc) as usize);
        for &first in &firsts[..pieces as usize] {
            let head = self.physical_to_logical[first as usize];
            for physical in first..first + run {
                let chunk = self.open_at(physical, shape, WHOLE);
                self.fill(chunk, live, u64::from(spc), &mut taken);
            }
            if run > 1 {
                self.runs.insert(head, run);
            }
        }
        if !taken.is_empty() {
            *self.holders.entry(holder).or_default() += taken.len() as u64;
        }
        Ok(taken)
    }

    /// Exactly slot refs `first .. first + n` of `shape` for `holder`, in
    /// ref order: the chunks they lie in, `first` its chunk's slot 0, each
    /// opened for the shape where it is named rather than lowest first (a
    /// last chunk's slots past `n` stay free for the shape). What a load's
    /// own placement is adopted by, its segments committed first (the
    /// load's `commit_for`, which the boot backs): a claim never commits a
    /// segment itself, so nothing opens that the engine was not told to
    /// back. All or nothing.
    ///
    /// # Errors
    ///
    /// [`SlotError::Unclaimable`] where `first` is not a slot 0 or a chunk
    /// is past the pool, reserved, open already, not free, in the
    /// activation span, draining or in a segment not committed; [`SlotError::Budget`] where the chunks
    /// outgrow what is committed.
    pub fn claim(
        &mut self,
        holder: Holder,
        shape: ShapeId,
        first: u32,
        n: u64,
    ) -> Result<Vec<Slot>, SlotError> {
        let range = self.claimable(shape, first, n)?;
        let count = range.end - range.start;
        let needed = self.reserved + self.held + self.act + count;
        let committed = self.committed_chunks();
        if needed > committed {
            return Err(SlotError::Budget { needed, committed });
        }
        let mut taken = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
        for chunk in range {
            let physical = self.logical_to_physical[chunk as usize];
            let opened = self.open_at(physical, shape, SINGLE);
            debug_assert_eq!(opened, chunk);
            self.fill(chunk, holder.is_live(), n - taken.len() as u64, &mut taken);
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Ok(taken)
    }

    /// Whether [`Self::claim`] of the same slots could take their chunks as
    /// the pool stands, budget aside: the logical chunks they lie in.
    ///
    /// # Errors
    ///
    /// As [`Self::claim`]'s [`SlotError::Unclaimable`].
    pub fn claimable(
        &self,
        shape: ShapeId,
        first: u32,
        n: u64,
    ) -> Result<std::ops::Range<u32>, SlotError> {
        let spc = self
            .shapes
            .get(shape as usize)
            .ok_or(SlotError::UnknownShape(shape))?
            .spc();
        let start = first / spc;
        if !first.is_multiple_of(spc) {
            return Err(SlotError::Unclaimable {
                chunk: start,
                why: "the first slot is not its chunk's slot 0",
            });
        }
        let count = u32::try_from(n.div_ceil(u64::from(spc))).unwrap_or(u32::MAX);
        let range = start..start.saturating_add(count);
        for chunk in range.clone() {
            if let Some(why) = self.unclaimable(chunk) {
                return Err(SlotError::Unclaimable { chunk, why });
            }
        }
        Ok(range)
    }

    /// Why logical chunk `chunk` cannot be claimed now, if it cannot.
    fn unclaimable(&self, chunk: u32) -> Option<&'static str> {
        if chunk >= self.total {
            return Some("past the pool");
        }
        if chunk < self.reserved {
            return Some("reserved");
        }
        if self.chunks[chunk as usize].shape != NO_SHAPE {
            return Some("open for a shape already");
        }
        let physical = self.logical_to_physical[chunk as usize];
        if !self.free.get(physical) || self.fenced.get(physical) {
            Some("not free")
        } else if physical >= self.state_limit() {
            Some("in the activation span")
        } else if matches!(
            self.segments.state(self.segments.of(physical)),
            SegmentState::Draining { .. }
        ) {
            Some("its segment is draining")
        } else if self.segments.state(self.segments.of(physical)) == SegmentState::Uncommitted {
            // Committed here only as the engine backs it: a claim that
            // committed a segment itself opened chunks the engine was never
            // told to back, and a fire on them was refused (Metal's admit:
            // "committed chunks of the state pool").
            Some(if self.segments.of(physical) >= self.top_segment {
                "its span segment is not backed yet (commit_span)"
            } else {
                "its segment is not backed yet (commit_segments)"
            })
        } else {
            None
        }
    }

    /// Adds `holder`'s reference to slots someone already holds.
    #[track_caller]
    pub fn share(&mut self, holder: Holder, slots: &[Slot]) {
        let live = holder.is_live();
        for &slot in slots {
            let (state, at) = self.locate(slot);
            let was = state.refs[at];
            assert!(was.live + was.cache > 0, "{slot:?} is not held");
            let mut now = was;
            if live {
                now.live += 1;
            } else {
                now.cache += 1;
            }
            state.refs[at] = now;
            state.reclassify(was, now);
        }
        if !slots.is_empty() {
            *self.holders.entry(holder).or_default() += slots.len() as u64;
        }
    }

    /// Notes that a fire at `epoch` touches `slots`: once let go, they wait
    /// for it.
    pub fn touch(&mut self, slots: &[Slot], epoch: u64) {
        for &slot in slots {
            let (state, at) = self.locate(slot);
            let refs = &mut state.refs[at];
            refs.touched = refs.touched.max(epoch);
        }
    }

    /// Drops `holder`'s reference on each of `slots`. A slot no one holds
    /// any more is handed out again once the later of `epoch` and the last
    /// fire that touched it retires.
    #[track_caller]
    pub fn release(&mut self, holder: Holder, slots: &[Slot], epoch: u64) {
        let live = holder.is_live();
        let mut immediate = Vec::new();
        // Let go slots by the epoch they wait on; few epochs per call.
        let mut waiting: Vec<(u64, Vec<Slot>)> = Vec::new();
        for &slot in slots {
            let (state, at) = self.locate(slot);
            let was = state.refs[at];
            let mut now = was;
            if live {
                assert!(now.live > 0, "{slot:?} has no live reference of {holder:?} to drop");
                now.live -= 1;
            } else {
                assert!(now.cache > 0, "{slot:?} has no cache reference to drop");
                now.cache -= 1;
            }
            state.refs[at] = now;
            state.reclassify(was, now);
            if now.live + now.cache == 0 {
                let after = now.touched.max(epoch);
                if after <= self.retired {
                    immediate.push(slot);
                } else if let Some((_, batch)) = waiting.iter_mut().find(|(at, _)| *at == after) {
                    batch.push(slot);
                } else {
                    waiting.push((after, vec![slot]));
                }
            }
        }
        if !slots.is_empty() {
            let count = self
                .holders
                .get_mut(&holder)
                .expect("a holder releases only what it holds");
            *count -= slots.len() as u64;
            if *count == 0 {
                self.holders.remove(&holder);
            }
        }
        for (after, batch) in waiting {
            match self.pending.entry(after) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(batch);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    entry.get_mut().extend(batch);
                }
            }
        }
        let freed = !immediate.is_empty();
        for slot in immediate {
            self.give_back(slot);
        }
        if freed {
            self.slots_freed();
        }
    }

    /// Calls `signal` with this pool's counts each time slots come free
    /// ([`RoomGiven::signal`]), once a release or a retire however many
    /// came: set once, by the planner of the engine this pool backs, its
    /// tier and the shape whose free slots it counts.
    pub fn set_room_signal(&self, signal: RoomSignal, tier: RoomTier, counts: Option<ShapeId>) {
        let _ = self.given.signal.set((signal, tier, counts));
    }

    /// Test builds: each slot given back from here on is pushed to `queue`
    /// as its physical byte range, for the engine that backs this pool to
    /// poison before its next verb ([`::engine::chunk::PoisonQueue`]).
    #[cfg(feature = "test-slot-fill")]
    pub fn poison_into(&mut self, queue: ::engine::chunk::PoisonQueue) {
        self.poison = PoisonTap(Some(queue));
    }

    fn slots_freed(&self) {
        if let Some((signal, tier, counts)) = self.given.signal.get() {
            (signal.0)(RoomCount {
                tier: *tier,
                chunks: u64::from(self.free_chunks()),
                slots: counts.map_or(0, |shape| self.available(shape)),
            });
        }
    }

    /// Every fire through `epoch` has retired: what waited on them is free.
    /// The chunks fenced until `from` stay fenced until `to` instead.
    pub fn refence(&mut self, from: u64, to: u64) {
        if let Some(chunks) = self.fences.remove(&from) {
            self.fences.entry(to).or_default().extend(chunks);
        }
    }

    pub fn retire_through(&mut self, epoch: u64) {
        self.retired = self.retired.max(epoch);
        let later = self.fences.split_off(&(epoch.saturating_add(1)));
        let mut due = std::mem::replace(&mut self.fences, later);
        if epoch == u64::MAX {
            due.append(&mut self.fences);
        }
        for chunk in due.into_values().flatten() {
            self.fenced.clear(chunk);
            self.reopen(chunk);
        }
        let later = self.pending.split_off(&(epoch.saturating_add(1)));
        let due = std::mem::replace(&mut self.pending, later);
        let mut freed = false;
        if epoch == u64::MAX {
            // split_off(MAX) leaves MAX itself behind.
            let rest = std::mem::take(&mut self.pending);
            for slot in rest.into_values().flatten() {
                self.give_back(slot);
                freed = true;
            }
        }
        for slot in due.into_values().flatten() {
            self.give_back(slot);
            freed = true;
        }
        if freed {
            self.slots_freed();
        }
    }

    /// Per class of `shape` ([`ChunkMeta::class`]), its partly filled
    /// chunks and their free slots: what segregating by class holds back.
    pub fn partial_classes(&self, shape: ShapeId) -> Vec<(u32, usize, u64)> {
        let Some(state) = self.shapes.get(shape as usize) else {
            return Vec::new();
        };
        state
            .partial
            .iter()
            .map(|(&class, chunks)| {
                let free = chunks.iter().map(|&(std::cmp::Reverse(used), _, _)| u64::from(state.spc() - used)).sum();
                (class, chunks.len(), free)
            })
            .collect()
    }

    /// Slots let go and waiting on a fire.
    pub fn pending_slots(&self) -> usize {
        self.pending.values().map(Vec::len).sum()
    }

    /// A chunk taken or moved onto lies in memory the engine backs for
    /// state: a committed segment (the bottom's, or a span segment the
    /// walk committed). Checked at every take and
    /// every move's destination in debug builds, naming the logical chunk,
    /// its physical and the segment: met, a fire on it reads or writes
    /// memory the engine gave back (a chunk of a segment just released).
    #[track_caller]
    fn assert_backed(&self, what: &str, logical: u32, physical: u32) {
        if cfg!(debug_assertions) {
            let segment = self.segments.of(physical);
            let state = self.segments.state(segment);
            assert!(
                physical >= self.reserved && state == SegmentState::Committed,
                "{what} of logical chunk {logical} on physical {physical} in segment {segment} \
                 ({state:?}), which the engine does not back for state"
            );
        }
    }

    #[track_caller]
    fn locate(&mut self, slot: Slot) -> (&mut ShapeState, usize) {
        let meta = self.chunks[slot.chunk as usize];
        assert!(meta.shape != NO_SHAPE, "{slot:?} lies in a free chunk");
        let state = &mut self.shapes[meta.shape as usize];
        assert!(slot.index < state.spc(), "{slot:?} is past its chunk");
        let at = (meta.frame * state.spc() + slot.index) as usize;
        (state, at)
    }

    /// Opens the lowest free committed chunk for `shape`. The caller has
    /// checked the room.
    fn open_chunk(&mut self, shape: ShapeId, class: u32) -> u32 {
        let physical = self.open.first().expect("room was checked");
        self.open_at(physical, shape, class)
    }

    /// Opens free physical chunk `physical` for `shape`, to hold `class`
    /// ([`ChunkMeta::class`]); its logical id.
    #[track_caller]
    fn open_at(&mut self, physical: u32, shape: ShapeId, class: u32) -> u32 {
        let logical = self.physical_to_logical[physical as usize];
        self.assert_backed("a take", logical, physical);
        self.open.clear(physical);
        self.free.clear(physical);
        debug_assert_eq!(self.chunks[logical as usize].shape, NO_SHAPE);
        self.segments.hold(physical);
        self.held += 1;
        let frame = self.shapes[shape as usize].open_frame();
        self.chunks[logical as usize] = ChunkMeta {
            shape,
            frame,
            used: 0,
            class,
        };
        logical
    }

    /// Takes up to `want` of `chunk`'s lowest free slots.
    fn fill(&mut self, chunk: u32, live: bool, want: u64, taken: &mut Vec<Slot>) {
        self.changes += 1;
        let meta = self.chunks[chunk as usize];
        let physical = self.logical_to_physical[chunk as usize];
        let state = &mut self.shapes[meta.shape as usize];
        let base = meta.frame as usize * state.words;
        let mut got = 0u32;
        for word in 0..state.words {
            while state.free[base + word] != 0 && u64::from(got) < want {
                let bits = &mut state.free[base + word];
                let index = word as u32 * WORD + bits.trailing_zeros();
                *bits &= *bits - 1;
                let refs = state.ref_at(meta.frame, index);
                let was = *refs;
                *refs = SlotRefs {
                    live: u32::from(live),
                    cache: u32::from(!live),
                    touched: 0,
                };
                let now = *refs;
                state.reclassify(was, now);
                taken.push(Slot { chunk, index });
                got += 1;
            }
            if u64::from(got) == want {
                break;
            }
        }
        let now = meta.used + got;
        state.reposition(physical, chunk, meta.class, meta.used, now);
        self.chunks[chunk as usize].used = now;
    }

    /// Takes slots `first .. first + n` of `chunk`, every one free.
    fn fill_run(&mut self, chunk: u32, live: bool, first: u32, n: u32, taken: &mut Vec<Slot>) {
        self.changes += 1;
        let meta = self.chunks[chunk as usize];
        let physical = self.logical_to_physical[chunk as usize];
        let state = &mut self.shapes[meta.shape as usize];
        let base = meta.frame as usize * state.words;
        for index in first..first + n {
            let (word, bit) = Bits::locate(index);
            let bits = &mut state.free[base + word];
            debug_assert_ne!(*bits & bit, 0, "slot {index} of chunk {chunk} is taken");
            *bits &= !bit;
            let refs = state.ref_at(meta.frame, index);
            let was = *refs;
            *refs = SlotRefs {
                live: u32::from(live),
                cache: u32::from(!live),
                touched: 0,
            };
            let now = *refs;
            state.reclassify(was, now);
            taken.push(Slot { chunk, index });
        }
        let now = meta.used + n;
        state.reposition(physical, chunk, meta.class, meta.used, now);
        self.chunks[chunk as usize].used = now;
    }

    fn give_back(&mut self, slot: Slot) {
        self.changes += 1;
        let meta = self.chunks[slot.chunk as usize];
        let physical = self.logical_to_physical[slot.chunk as usize];
        #[cfg(feature = "test-slot-fill")]
        if let Some(queue) = &self.poison.0 {
            let bytes = self.shapes[meta.shape as usize].shape.slot_bytes;
            let offset = u64::from(physical) * self.chunk_bytes + u64::from(slot.index) * bytes;
            queue
                .lock()
                .expect("the poison queue's lock is never poisoned")
                .push((offset, bytes));
        }
        let state = &mut self.shapes[meta.shape as usize];
        let (word, bit) = Bits::locate(slot.index);
        let at = meta.frame as usize * state.words + word;
        debug_assert_eq!(state.free[at] & bit, 0, "{slot:?} freed twice");
        state.free[at] |= bit;
        let now = meta.used - 1;
        state.reposition(physical, slot.chunk, meta.class, meta.used, now);
        self.chunks[slot.chunk as usize].used = now;
        if now == 0 {
            state.spare.push(meta.frame);
            self.close_chunk(slot.chunk, physical);
        }
    }

    fn close_chunk(&mut self, logical: u32, physical: u32) {
        self.runs.remove(&logical);
        self.chunks[logical as usize] = FREE_CHUNK;
        self.free.set(physical);
        self.segments.let_go(physical);
        self.held -= 1;
        self.reopen(physical);
    }

    // ---- the span backed on demand ----

    /// Whether the span is backed on demand ([`SlotPoolFacts::span_on_demand`]).
    pub fn span_on_demand(&self) -> bool {
        self.span_on_demand
    }

    /// The span segments the rung lends and no segment backs yet, each with
    /// the chunks of it the rung lends (below [`Self::state_limit`]),
    /// highest first, the order [`Self::span_to_commit`] backs them. Empty
    /// where the span is backed from the load or the engine holds part of
    /// the range back.
    ///
    /// Span lending. The pool is one range: the top `act_max` chunks are
    /// the largest rung's activation, and a fire at a smaller rung takes
    /// only its own `act_chunks(rung)` of them. The chunks between, below
    /// the state limit, are lent to state (kv pages, records, rs lists,
    /// guest parts alike): they open as any chunk does once their segment
    /// is backed, and a grow for a waiter short of room backs them after
    /// the prefix ([`super::shared::SharedPool::grow_step`]). Lending
    /// allocates nothing past load: the span's segments are in the range
    /// reserved at load, backed on demand within it. A fire whose rung
    /// needs lent chunks takes them back first: a raise moves the state
    /// lying there down into free chunks below ([`Self::plan_rung`]); where
    /// too few stand free, [`Self::rung_short`] says how many, and the
    /// planner frees them as it does any head's room (its occupants
    /// evicted or moved down, under the key rule), the raise waiting for
    /// that with no timer. The budget line states the span lent, how much
    /// of it is backed, and the state chunks a raise to the largest rung
    /// takes back.
    fn lent_unbacked(&self) -> Vec<(u32, u32)> {
        if !self.span_on_demand || self.ceiling < self.total {
            return Vec::new();
        }
        let limit = self.state_limit();
        (self.top_segment..self.segments.count())
            .rev()
            .filter(|&segment| self.segments.state(segment) == SegmentState::Uncommitted)
            .filter_map(|segment| {
                let range = self.segments.range(segment);
                let lent = range.end.min(limit).saturating_sub(range.start);
                (lent > 0).then_some((segment, lent))
            })
            .collect()
    }

    /// The span the rung lends to state now (`act_max` less the rung's
    /// activation), how much of it is backed, and the state chunks a raise
    /// to the largest rung must take back ([`Self::held_in_span`]).
    #[must_use]
    pub fn span_lent(&self) -> (u32, u32, u32) {
        let lent = self.act_max.saturating_sub(self.act);
        let unbacked: u32 = self.lent_unbacked().iter().map(|&(_, chunks)| chunks).sum();
        let top = self.act_chunks.last().map_or(0, |&(tokens, _)| tokens);
        (lent, lent.saturating_sub(unbacked), self.held_in_span(top))
    }

    /// Span segments committed here, counted from the top of the pool: the
    /// `n` of `commit_span(n)` (0 where the span is backed from the load).
    pub fn span_committed(&self) -> u32 {
        if !self.span_on_demand {
            return 0;
        }
        (self.top_segment..self.segments.count())
            .rev()
            .take_while(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .count() as u32
    }

    /// The span segment counted `n`th from the top (`n` from 1).
    fn span_segment(&self, n: u32) -> u32 {
        self.segments.count() - n
    }

    /// The `n` of the next `commit_span(n)` that backs a span chunk the
    /// lowered rung lends to state and no segment backs yet: the highest
    /// such segment's, so the span is backed from the top down. `None`
    /// where every lent span chunk is backed, nothing is lent, or the
    /// engine holds back part of the range (no state opens past it then).
    pub fn span_to_commit(&self) -> Option<u32> {
        self.lent_unbacked()
            .first()
            .map(|&(segment, _)| self.segments.count() - segment)
    }

    /// The engine backed the top `n` span segments (`commit_span(n)`):
    /// their chunks the rung lends open for state.
    pub fn commit_span(&mut self, n: u32) {
        let n = n.min(self.segments.count() - self.top_segment);
        for k in 1..=n {
            let segment = self.span_segment(k);
            if self.segments.state(segment) == SegmentState::Uncommitted {
                self.segments.set_state(segment, SegmentState::Committed);
                for chunk in self.segments.range(segment) {
                    self.reopen(chunk);
                }
            }
        }
    }

    /// The `n` of the next `release_span(n)`: the lowest span segments
    /// committed here that the rung no longer lends (wholly in the
    /// activation, the rung raised past them), hold no state chunk and no
    /// chunk a raise's copy still reads, given back from the bottom up.
    /// `None` where none is.
    pub fn span_to_release(&self) -> Option<u32> {
        let committed = self.span_committed();
        let limit = self.state_limit();
        let mut keep = committed;
        while keep > 0 {
            let segment = self.span_segment(keep);
            let range = self.segments.range(segment);
            let idle = range.start >= limit
                && self.segments.used(segment) == 0
                && range.clone().all(|chunk| !self.fenced.get(chunk));
            if !idle {
                break;
            }
            keep -= 1;
        }
        (keep < committed).then_some(keep)
    }

    /// Gives back the span segments below the top `n` committed here
    /// ([`Self::span_to_release`] says which may go), ahead of the
    /// engine's `release_span(n)`.
    pub fn release_span(&mut self, n: u32) {
        for k in n + 1..=self.span_committed() {
            let segment = self.span_segment(k);
            debug_assert_eq!(self.segments.used(segment), 0);
            self.segments.set_state(segment, SegmentState::Uncommitted);
            for chunk in self.segments.range(segment) {
                self.reopen(chunk);
            }
        }
    }

    // ---- segments ----

    pub fn segments(&self) -> &Segments {
        &self.segments
    }

    /// The committed segment below the top span holding the fewest chunks,
    /// the highest among equals.
    pub fn emptiest_segment(&self) -> Option<u32> {
        (0..self.top_segment)
            .rev()
            .filter(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .min_by_key(|&segment| self.segments.used(segment))
    }

    /// Moves every held chunk of `segment` into free committed chunks of
    /// lower segments, lowest first, a run of adjacent chunks
    /// ([`Self::acquire_chunk_runs`]) whole into as many adjacent free
    /// chunks of one segment; none when they do not fit there, when the
    /// segment holds reserved chunks, or when the budget would not survive
    /// losing the segment.
    pub fn plan_evacuation(&self, segment: u32) -> Option<Evacuation> {
        if segment >= self.top_segment || self.segments.state(segment) != SegmentState::Committed {
            return None;
        }
        let range = self.segments.range(segment);
        if range.start < self.reserved {
            return None;
        }
        let needed = self.reserved + self.held + self.act;
        if needed > self.committed_chunks() - self.segments.len(segment) {
            return None;
        }
        // A segment a residency copy still reads or writes is never
        // evacuated: the copy's addresses were resolved at its submit, and
        // a segment released (unmapped) under a copy queued on another
        // stream is read unmapped (a device fault on an eviction's
        // kv-to-disk copy).
        if self.moving_out.range(range.start..range.end).next().is_some() {
            return None;
        }
        // Nor one holding chunks a served restore's reservation keeps for
        // its unpark and bind ([`Self::reserve_chunks`]): released, they
        // left the reservation unable to open them, and the bind they were
        // held for came up short, its process suspended again (the
        // simulation's seed 263: a run of three held in a segment the walk
        // gave back).
        if self.reserved_in(range.start, range.end) {
            return None;
        }
        let sources: Vec<u32> = (range.start..range.end)
            .filter(|&chunk| !self.free.get(chunk))
            .collect();
        let mut targets: BTreeSet<u32> = self.open.iter_in(0, range.start).collect();
        let mut moves = Vec::with_capacity(sources.len());
        let mut in_run = BTreeSet::new();
        for &from in &sources {
            let logical = self.physical_to_logical[from as usize];
            let Some(&run) = self.runs.get(&logical) else {
                continue;
            };
            let to = self.free_run_in(&targets, run)?;
            for k in 0..run {
                targets.remove(&(to + k));
                in_run.insert(from + k);
                moves.push(ChunkMove {
                    logical: self.physical_to_logical[(from + k) as usize],
                    from: from + k,
                    to: to + k,
                });
            }
        }
        for &from in sources.iter().filter(|from| !in_run.contains(from)) {
            let to = targets.pop_first()?;
            moves.push(ChunkMove {
                logical: self.physical_to_logical[from as usize],
                from,
                to,
            });
        }
        Some(Evacuation { segment, moves })
    }

    /// The lowest of `free` chunks that opens `run` of them at adjacent
    /// physical ids inside one segment.
    fn free_run_in(&self, free: &BTreeSet<u32>, run: u32) -> Option<u32> {
        let mut stretch = 0u32;
        let mut last = None;
        for &chunk in free {
            let adjacent = last.is_some_and(|last: u32| {
                last + 1 == chunk && self.segments.of(last) == self.segments.of(chunk)
            });
            stretch = if adjacent { stretch + 1 } else { 1 };
            last = Some(chunk);
            if stretch == run {
                return Some(chunk + 1 - run);
            }
        }
        None
    }

    /// Carries out `evacuation`, which the fire at `epoch` copies: the
    /// segment then drains, handing nothing out, until that fire retires.
    pub fn apply_evacuation(
        &mut self,
        evacuation: &Evacuation,
        epoch: u64,
    ) -> Result<(), SegmentError> {
        let segment = evacuation.segment;
        self.segments.check(segment)?;
        let stale = SegmentError::Stale(segment);
        if let Some((head, run, at)) = self.splits_run(&evacuation.moves) {
            return Err(SegmentError::SplitsRun {
                segment,
                head,
                run,
                at,
            });
        }
        if segment >= self.top_segment || self.segments.state(segment) != SegmentState::Committed {
            return Err(stale);
        }
        let range = self.segments.range(segment);
        let needed = self.reserved + self.held + self.act;
        let committed = self.committed_chunks() - self.segments.len(segment);
        if needed > committed {
            return Err(SegmentError::Budget {
                segment,
                committed,
                needed,
            });
        }
        if evacuation.moves.len() != self.segments.used(segment) as usize {
            return Err(stale);
        }
        let mut targets = BTreeSet::new();
        for step in &evacuation.moves {
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && range.contains(&step.from)
                && step.to < range.start
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(stale);
            }
        }
        self.segments
            .set_state(segment, SegmentState::Draining { until: epoch });
        for chunk in range {
            self.open.clear(chunk);
        }
        for step in &evacuation.moves {
            self.move_chunk(*step);
        }
        Ok(())
    }

    /// Takes back `evacuation`, applied with [`Self::apply_evacuation`] and
    /// refused by the engine before its copy: each moved chunk goes back to
    /// where it lay (the engine's table never changed), held still or let
    /// go since, and the segment is committed again, its free chunks open.
    pub fn undo_evacuation(&mut self, evacuation: &Evacuation) {
        let segment = evacuation.segment;
        if !matches!(
            self.segments.state(segment),
            SegmentState::Draining { .. }
        ) {
            return;
        }
        self.segments.set_state(segment, SegmentState::Committed);
        for step in evacuation.moves.iter().rev() {
            if self.logical_to_physical[step.logical as usize] != step.to {
                continue;
            }
            let back = ChunkMove {
                logical: step.logical,
                from: step.to,
                to: step.from,
            };
            if self.chunks[step.logical as usize].shape == NO_SHAPE {
                // Let go since: two free chunks change places.
                let displaced = self.physical_to_logical[back.to as usize];
                self.logical_to_physical[back.logical as usize] = back.to;
                self.logical_to_physical[displaced as usize] = back.from;
                self.physical_to_logical[back.to as usize] = back.logical;
                self.physical_to_logical[back.from as usize] = displaced;
            } else {
                self.move_chunk(back);
            }
        }
        for chunk in self.segments.range(segment) {
            self.reopen(chunk);
        }
    }

    /// The draining segment planned to wait past `epoch` waits on it
    /// instead: the engine took its copy with the fires through `epoch`
    /// handed out, the last that may read its chunks.
    pub fn drains_through(&mut self, epoch: u64) {
        for segment in 0..self.segments.count() {
            if let SegmentState::Draining { until } = self.segments.state(segment)
                && until > epoch
            {
                self.segments
                    .set_state(segment, SegmentState::Draining { until: epoch });
            }
        }
    }

    /// Gives `segment` back: a drained one once its evacuation's fire has
    /// retired, or a committed one holding nothing.
    pub fn release_segment(&mut self, segment: u32) -> Result<(), SegmentError> {
        self.segments.check(segment)?;
        if segment >= self.top_segment {
            return Err(SegmentError::State {
                segment,
                state: self.segments.state(segment),
                wanted: "below the activation committed from the load",
            });
        }
        let held = self.segments.used(segment);
        if held > 0 {
            return Err(SegmentError::Held { segment, held });
        }
        // Nor released while a residency copy still reads or writes one of
        // its chunks: it waits for the copy to land ([`Self::moved_out`]).
        let range = self.segments.range(segment);
        // Nor while a served restore's reservation keeps one of its chunks
        // ([`Self::plan_evacuation`]).
        // Nor while a chunk move's copy still reads or writes one of its
        // chunks (fenced until the fire carrying it retires): a compaction
        // moving chunks out leaves the segment empty and committed, and
        // given back then, the copy read memory the engine unmapped.
        let fenced = range.clone().filter(|&chunk| self.fenced.get(chunk)).count();
        let copying = u32::try_from(self.moving_out.range(range.start..range.end).count() + fenced)
            .unwrap_or(u32::MAX)
            .saturating_add(u32::from(self.reserved_in(range.start, range.end)));
        if copying > 0 {
            return Err(SegmentError::Held { segment, held: copying });
        }
        match self.segments.state(segment) {
            SegmentState::Draining { until } if until > self.retired => {
                Err(SegmentError::Draining {
                    segment,
                    until,
                    retired: self.retired,
                })
            }
            SegmentState::Draining { .. } => {
                self.segments.set_state(segment, SegmentState::Uncommitted);
                Ok(())
            }
            SegmentState::Committed => {
                let needed = self.reserved + self.held + self.act;
                let committed = self.committed_chunks() - self.segments.len(segment);
                if needed > committed {
                    return Err(SegmentError::Budget {
                        segment,
                        committed,
                        needed,
                    });
                }
                self.segments.set_state(segment, SegmentState::Uncommitted);
                for chunk in self.segments.range(segment) {
                    self.reopen(chunk);
                }
                Ok(())
            }
            state => Err(SegmentError::State {
                segment,
                state,
                wanted: "drained or empty",
            }),
        }
    }

    /// Marks `segment` committed once the backend has mapped it.
    pub fn commit_segment(&mut self, segment: u32) -> Result<(), SegmentError> {
        self.segments.check(segment)?;
        match self.segments.state(segment) {
            SegmentState::Uncommitted => {
                self.segments.set_state(segment, SegmentState::Committed);
                for chunk in self.segments.range(segment) {
                    debug_assert!(self.free.get(chunk));
                    self.reopen(chunk);
                }
                Ok(())
            }
            state => Err(SegmentError::State {
                segment,
                state,
                wanted: "uncommitted",
            }),
        }
    }

    /// The lowest uncommitted segments to commit so `bytes` of headroom
    /// stand ready, off the fire path; empty when they already do. Never
    /// more than there are.
    pub fn commit_ahead(&self, bytes: u64) -> Vec<u32> {
        let want = bytes.div_ceil(self.chunk_bytes.max(1));
        let mut room = u64::from(self.headroom());
        let mut segments = Vec::new();
        for segment in self
            .segments
            .uncommitted()
            .filter(|&segment| segment < self.top_segment && self.backs(segment))
        {
            if room >= want {
                break;
            }
            room += u64::from(self.segments.len(segment));
            segments.push(segment);
        }
        // The prefix whole, the span the rung lends ([`Self::lent_unbacked`]).
        for (segment, lent) in self.lent_unbacked() {
            if room >= want {
                break;
            }
            room += u64::from(lent);
            segments.push(segment);
        }
        segments
    }

    /// Checks every internal invariant; for tests.
    #[cfg(test)]
    pub(crate) fn check(&self) {
        for (&head, &run) in &self.runs {
            let start = self.logical_to_physical[head as usize];
            for at in start..start + run {
                assert!(
                    at < self.total
                        && !self.free.get(at)
                        && self.segments.of(at) == self.segments.of(start),
                    "the run of {run} from logical chunk {head} (physical {start}) is split at {at}"
                );
            }
        }
        let mut held = 0;
        let mut seg_used = vec![0u32; self.segments.count() as usize];
        for logical in 0..self.total {
            let physical = self.logical_to_physical[logical as usize];
            let meta = self.chunks[logical as usize];
            assert_eq!(self.physical_to_logical[physical as usize], logical);
            if meta.shape == NO_SHAPE && logical >= self.reserved {
                assert!(self.free.get(physical), "a free logical id on a held chunk");
                continue;
            }
            assert!(!self.free.get(physical));
            assert!(!self.open.get(physical));
            seg_used[self.segments.of(physical) as usize] += 1;
            if logical < self.reserved {
                assert_eq!(physical, logical);
                continue;
            }
            held += 1;
            assert!(meta.used > 0, "an open chunk with nothing out");
            let state = &self.shapes[meta.shape as usize];
            let base = meta.frame as usize * state.words;
            let free: u32 = state.free[base..base + state.words]
                .iter()
                .map(|word| word.count_ones())
                .sum();
            assert_eq!(free + meta.used, state.spc());
            let partial = meta.used < state.spc();
            assert_eq!(
                state.partial.get(&meta.class).is_some_and(|chunks| chunks
                    .contains(&(Reverse(meta.used), physical, logical))),
                partial
            );
            assert!(
                matches!(
                    self.segments.state(self.segments.of(physical)),
                    SegmentState::Committed
                ),
                "a held chunk outside a committed segment"
            );
        }
        assert_eq!(held, self.held);
        for segment in 0..self.segments.count() {
            assert_eq!(seg_used[segment as usize], self.segments.used(segment));
            if segment >= self.top_segment && !self.span_on_demand {
                assert_eq!(self.segments.state(segment), SegmentState::Committed);
            }
            for chunk in self.segments.range(segment) {
                assert_eq!(self.open.get(chunk), self.openable(chunk), "chunk {chunk}");
                if !self.free.get(chunk) && chunk >= self.reserved {
                    assert!(chunk < self.state_limit(), "state in the activation span");
                }
            }
        }
        for state in &self.shapes {
            let mut live = 0;
            let mut cached = 0;
            let mut partial_free = 0;
            for (&class, chunks) in &state.partial {
                assert!(!chunks.is_empty(), "an empty class is no entry");
                for &(Reverse(used), physical, logical) in chunks {
                    assert_eq!(self.logical_to_physical[logical as usize], physical);
                    assert_eq!(self.chunks[logical as usize].used, used);
                    assert_eq!(self.chunks[logical as usize].class, class);
                    if class == SINGLE {
                        partial_free += u64::from(state.spc() - used);
                    }
                }
            }
            assert_eq!(partial_free, state.partial_free);
            for logical in 0..self.total {
                let meta = self.chunks[logical as usize];
                if meta.shape == NO_SHAPE || !std::ptr::eq(&self.shapes[meta.shape as usize], state)
                {
                    continue;
                }
                for index in 0..state.spc() {
                    let refs = state.refs[(meta.frame * state.spc() + index) as usize];
                    live += u64::from(refs.live > 0);
                    cached += u64::from(refs.live == 0 && refs.cache > 0);
                }
            }
            assert_eq!(live, state.live);
            assert_eq!(cached, state.cached);
        }
        assert!(self.reserved + self.held + self.act <= self.committed_chunks());
        assert!(self.committed_chunks() <= self.total);
    }
}

#[cfg(test)]
mod tests;
