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

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use state::key::Hash256;

pub use super::segments::{ChunkMove, Evacuation, SegmentError, SegmentState, Segments};

/// A slot shape's id: its index in [`SlotPoolFacts::shapes`].
pub type ShapeId = u16;

/// Fixed sets each working set holds: the one its fires read. Derived, not
/// configured.
pub const FIXED_SETS_PER_WORKING_SET: u32 = 1;

/// Copy-on-write fixed sets each running lane holds while it fires: the one
/// the fire writes before the working set's set is let go. Charged per lane
/// a fire carries, not per working set.
pub const COW_SETS_PER_LANE: u32 = 1;

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
    /// Slot sets each working set holds of this shape; a running lane holds
    /// [`COW_SETS_PER_LANE`] more of a fixed one.
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
}

/// What one registered program's fires add to every rung's activation
/// (REDESIGN §6, [`::engine::ProgramFacts::fire_bytes`]): a fire of a rung
/// of `r` tokens carries `min(r, FireLimits::max_lanes, max_lanes)` lanes
/// of it and needs `fixed + lanes × a_lane` bytes beside the model's own
/// activation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct FireTerm {
    pub fixed: u64,
    pub a_lane: u64,
    /// The most lanes one fire of the program flies; 0: no cap of its own.
    pub max_lanes: u32,
}

impl FireTerm {
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.fixed == 0 && self.a_lane == 0
    }

    /// Bytes a fire carrying `lanes` lanes needs for the program: its own
    /// lane cap taken.
    #[must_use]
    pub fn bytes(&self, lanes: u32) -> u64 {
        let lanes = if self.max_lanes == 0 {
            lanes
        } else {
            lanes.min(self.max_lanes)
        };
        self.fixed
            .saturating_add(u64::from(lanes).saturating_mul(self.a_lane))
    }
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
}

/// The state chunks in the way of a rung's activation, each moved to a free
/// chunk below it; logical ids stand. The engine copies `from` -> `to` and
/// rewrites its chunk table before the first fire at the rung.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RungEvacuation {
    pub rung: u32,
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
}

const FREE_CHUNK: ChunkMeta = ChunkMeta {
    shape: NO_SHAPE,
    frame: NONE,
    used: 0,
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
    /// Partly filled chunks, fullest first, then lowest physical address:
    /// `(Reverse(used), physical, logical)`.
    partial: BTreeSet<(Reverse<u32>, u32, u32)>,
    /// Free slots in `partial`.
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
            partial: BTreeSet::new(),
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
    fn free_run(&self, frame: u32, n: u32) -> Option<u32> {
        let base = frame as usize * self.words;
        let free = |index: u32| {
            let (word, bit) = Bits::locate(index);
            self.free[base + word] & bit != 0
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

    /// Moves a chunk's place among the partly filled as its count goes from
    /// `was` to `now`.
    fn reposition(&mut self, physical: u32, logical: u32, was: u32, now: u32) {
        let spc = self.spc();
        if was > 0 && was < spc {
            self.partial.remove(&(Reverse(was), physical, logical));
            self.partial_free -= u64::from(spc - was);
        }
        if now > 0 && now < spc {
            self.partial.insert((Reverse(now), physical, logical));
            self.partial_free += u64::from(spc - now);
        }
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

/// See the module docs.
#[derive(Clone, Debug)]
pub struct SlotPool {
    chunk_bytes: u64,
    total: u32,
    reserved: u32,
    shapes: Vec<ShapeState>,
    act_chunks: Vec<(u32, u32)>,
    act: u32,
    /// The fire terms of the programs that can be in a fire, on top of
    /// every rung's activation ([`Self::act_chunks`]).
    fire: Vec<FireTerm>,
    /// The most lanes a fire carries (the engine's `FireLimits::max_lanes`):
    /// a rung of `r` tokens allows `min(r, max_lanes)`.
    max_lanes: u32,
    /// The largest rung's activation chunks, the top of the range.
    act_max: u32,
    /// The first segment under the largest rung's activation: it and those
    /// above are committed from the load and never evacuated or released.
    top_segment: u32,
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
        // The largest rung's activation is backed from the load.
        let top_segment = if act_max == 0 {
            segments.count()
        } else {
            segments.of(total - act_max)
        };
        for segment in top_segment..segments.count() {
            if segments.state(segment) == SegmentState::Uncommitted {
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
            fire: Vec::new(),
            max_lanes: u32::MAX,
            act_max,
            top_segment,
            chunks: vec![FREE_CHUNK; total as usize],
            logical_to_physical: (0..total).collect(),
            physical_to_logical: (0..total).collect(),
            free,
            open: Bits::new(total, false),
            fences: BTreeMap::new(),
            fenced: Bits::new(total, false),
            ceiling: total,
            segments,
            held: 0,
            pending: BTreeMap::new(),
            retired: 0,
            holders: HashMap::new(),
            waiters: HashMap::new(),
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
    }

    /// No chunk at or past physical id `ceiling` is opened from now on, the
    /// engine binding none there yet (while it holds back part of the range
    /// it stated, [`::engine::live::Live::held`]); chunks below it open as
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
            self.open.set(chunk);
        } else {
            self.open.clear(chunk);
        }
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

    pub fn committed_chunks(&self) -> u32 {
        self.segments.committed_chunks()
    }

    /// Activation chunks a fire at `rung` tokens takes: the smallest rung at
    /// or above it, else the largest; with the chunks the registered
    /// programs' fire term takes at that rung's lanes ([`Self::set_fire`]).
    pub fn act_chunks(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(0, |&(tokens, chunks)| {
                chunks.saturating_add(self.fire_chunks(tokens))
            })
    }

    /// The chunks the fire terms take at a rung of `tokens`: `min(tokens,
    /// max_lanes)` lanes, each program's own cap taken.
    fn fire_chunks(&self, tokens: u32) -> u32 {
        let lanes = tokens.min(self.max_lanes);
        let bytes = self
            .fire
            .iter()
            .fold(0u64, |sum, term| sum.saturating_add(term.bytes(lanes)));
        u32::try_from(bytes.div_ceil(self.chunk_bytes.max(1))).unwrap_or(u32::MAX)
    }

    /// The fire terms of the programs that can be in a fire from now on.
    /// Nothing moves here: terms that raise the rung's need are taken by a
    /// raise ([`Self::plan_rung`], [`Self::apply_rung`]), and terms that
    /// drop give the room back at the next lower ([`Self::set_rung`]).
    pub fn set_fire(&mut self, mut fire: Vec<FireTerm>) {
        fire.retain(|term| !term.is_zero());
        fire.sort_unstable();
        self.fire = fire;
    }

    /// See [`Self::set_fire`].
    pub fn fire(&self) -> &[FireTerm] {
        &self.fire
    }

    /// The most lanes a fire carries: the engine's `FireLimits::max_lanes`.
    pub fn set_max_lanes(&mut self, lanes: u32) {
        self.max_lanes = lanes.max(1);
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

    /// The first segment under the largest rung's activation (the segment
    /// count where activation lives apart).
    pub fn top_segment(&self) -> u32 {
        self.top_segment
    }

    /// State chunks lie at physical ids below this: the activation of the
    /// current rung takes the rest.
    pub fn state_limit(&self) -> u32 {
        self.total - self.act
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
        if needed > self.committed_chunks() {
            return Err(SlotError::Budget {
                needed,
                committed: self.committed_chunks(),
            });
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
        let (was, now) = (self.state_limit(), self.total - act);
        self.act = act;
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
        let sources = self.held_in(limit, self.state_limit());
        let moves: Vec<ChunkMove> = sources
            .iter()
            .zip(self.open.iter_in(0, limit))
            .map(|(&from, to)| ChunkMove {
                logical: self.physical_to_logical[from as usize],
                from,
                to,
            })
            .collect();
        if moves.len() < sources.len() {
            return Err(SlotError::NoRoom {
                rung,
                chunks: sources.len() as u32,
            });
        }
        Ok(RungEvacuation { rung, act, moves })
    }

    /// The chunks raising the activation to `rung` lacks now: past what is
    /// committed, and in free chunks below its span for the state in it.
    /// 0 when the raise fits.
    pub fn rung_short(&self, rung: u32) -> u32 {
        let act = self.act_chunks(rung);
        let budget = (self.reserved + self.held + act).saturating_sub(self.committed_chunks());
        let limit = self.total - act;
        let sources = self.held_in(limit, self.state_limit()).len() as u32;
        let targets = self.open.iter_in(0, limit).take(sources as usize).count() as u32;
        budget.max(sources - targets)
    }

    /// Carries out `evacuation` and takes its rung; the fire at `epoch`
    /// carries the copy, and the chunks moved off stay closed until it
    /// retires. All or nothing.
    pub fn apply_rung(&mut self, evacuation: &RungEvacuation, epoch: u64) -> Result<(), SlotError> {
        let act = self.act_chunks(evacuation.rung);
        self.rung_budget(act)?;
        let limit = self.total - act;
        if act != evacuation.act
            || evacuation.moves.len() != self.held_in(limit, self.state_limit()).len()
        {
            return Err(SlotError::StaleRung);
        }
        let mut targets = BTreeSet::new();
        for step in &evacuation.moves {
            let fits = (step.logical as usize) < self.logical_to_physical.len()
                && self.chunks[step.logical as usize].shape != NO_SHAPE
                && self.logical_to_physical[step.logical as usize] == step.from
                && step.from >= limit
                && step.to < limit
                && self.open.get(step.to)
                && targets.insert(step.to);
            if !fits {
                return Err(SlotError::StaleRung);
            }
        }
        for step in &evacuation.moves {
            self.move_chunk(*step);
            self.fenced.set(step.from);
        }
        if !evacuation.moves.is_empty() {
            self.fences
                .entry(epoch)
                .or_default()
                .extend(evacuation.moves.iter().map(|step| step.from));
        }
        self.set_act(act);
        Ok(())
    }

    /// Moves a held chunk onto free physical chunk `step.to`, as the
    /// engine's `ChunkTable::apply` does: the free logical id there takes the
    /// moved one's old place.
    fn move_chunk(&mut self, step: ChunkMove) {
        let meta = self.chunks[step.logical as usize];
        let displaced = self.physical_to_logical[step.to as usize];
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
            state
                .partial
                .remove(&(Reverse(meta.used), step.from, step.logical));
            state
                .partial
                .insert((Reverse(meta.used), step.to, step.logical));
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

    /// Records that `waiter` is parked short `slots` of `shape`, replacing
    /// what it was short before.
    pub fn wait(&mut self, waiter: u64, shape: ShapeId, slots: u64) {
        self.unwait(waiter);
        if let Some(state) = self.shapes.get_mut(shape as usize) {
            state.waiting += slots;
            self.waiters.insert(waiter, (shape, slots));
        }
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
            let chunk = match self.shapes[shape as usize].partial.first() {
                Some(&(_, _, logical)) => logical,
                None => self.open_chunk(shape),
            };
            self.fill(chunk, holder.is_live(), n - taken.len() as u64, &mut taken);
        }
        if n > 0 {
            *self.holders.entry(holder).or_default() += n;
        }
        Ok(taken)
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
        let found = state.partial.iter().find_map(|&(_, _, logical)| {
            let frame = self.chunks[logical as usize].frame;
            state.free_run(frame, n).map(|first| (logical, first))
        });
        let (chunk, first) = match found {
            Some(found) => found,
            None if self.free_chunks() > 0 => (self.open_chunk(shape), 0),
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

    /// Exactly slot refs `first .. first + n` of `shape` for `holder`, in
    /// ref order: the chunks they lie in, `first` its chunk's slot 0, each
    /// opened for the shape where it is named rather than lowest first (a
    /// last chunk's slots past `n` stay free for the shape). What a load's
    /// own placement is adopted by. A segment of theirs not yet committed
    /// is counted committed: the backend backed it. All or nothing.
    ///
    /// # Errors
    ///
    /// [`SlotError::Unclaimable`] where `first` is not a slot 0 or a chunk
    /// is past the pool, reserved, open already, not free, in the
    /// activation span or draining; [`SlotError::Budget`] where the chunks
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
        let uncommitted: BTreeSet<u32> = range
            .clone()
            .map(|chunk| self.segments.of(self.logical_to_physical[chunk as usize]))
            .filter(|&segment| self.segments.state(segment) == SegmentState::Uncommitted)
            .collect();
        let added: u32 = uncommitted
            .iter()
            .map(|&segment| self.segments.len(segment))
            .sum();
        let needed = self.reserved + self.held + self.act + count;
        let committed = self.committed_chunks() + added;
        if needed > committed {
            return Err(SlotError::Budget { needed, committed });
        }
        for segment in uncommitted {
            self.segments.set_state(segment, SegmentState::Committed);
            for chunk in self.segments.range(segment) {
                self.reopen(chunk);
            }
        }
        let mut taken = Vec::with_capacity(usize::try_from(n).unwrap_or(0));
        for chunk in range {
            let physical = self.logical_to_physical[chunk as usize];
            let opened = self.open_at(physical, shape);
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
        } else {
            None
        }
    }

    /// Adds `holder`'s reference to slots someone already holds.
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
                assert!(now.live > 0, "{slot:?} has no live reference to drop");
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
        for slot in immediate {
            self.give_back(slot);
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
        if epoch == u64::MAX {
            // split_off(MAX) leaves MAX itself behind.
            let rest = std::mem::take(&mut self.pending);
            for slot in rest.into_values().flatten() {
                self.give_back(slot);
            }
        }
        for slot in due.into_values().flatten() {
            self.give_back(slot);
        }
    }

    /// Slots let go and waiting on a fire.
    pub fn pending_slots(&self) -> usize {
        self.pending.values().map(Vec::len).sum()
    }

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
    fn open_chunk(&mut self, shape: ShapeId) -> u32 {
        let physical = self.open.first().expect("room was checked");
        self.open_at(physical, shape)
    }

    /// Opens free physical chunk `physical` for `shape`; its logical id.
    fn open_at(&mut self, physical: u32, shape: ShapeId) -> u32 {
        self.open.clear(physical);
        self.free.clear(physical);
        let logical = self.physical_to_logical[physical as usize];
        debug_assert_eq!(self.chunks[logical as usize].shape, NO_SHAPE);
        self.segments.hold(physical);
        self.held += 1;
        let frame = self.shapes[shape as usize].open_frame();
        self.chunks[logical as usize] = ChunkMeta {
            shape,
            frame,
            used: 0,
        };
        logical
    }

    /// Takes up to `want` of `chunk`'s lowest free slots.
    fn fill(&mut self, chunk: u32, live: bool, want: u64, taken: &mut Vec<Slot>) {
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
        state.reposition(physical, chunk, meta.used, now);
        self.chunks[chunk as usize].used = now;
    }

    /// Takes slots `first .. first + n` of `chunk`, every one free.
    fn fill_run(&mut self, chunk: u32, live: bool, first: u32, n: u32, taken: &mut Vec<Slot>) {
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
        state.reposition(physical, chunk, meta.used, now);
        self.chunks[chunk as usize].used = now;
    }

    fn give_back(&mut self, slot: Slot) {
        let meta = self.chunks[slot.chunk as usize];
        let physical = self.logical_to_physical[slot.chunk as usize];
        let state = &mut self.shapes[meta.shape as usize];
        let (word, bit) = Bits::locate(slot.index);
        let at = meta.frame as usize * state.words + word;
        debug_assert_eq!(state.free[at] & bit, 0, "{slot:?} freed twice");
        state.free[at] |= bit;
        let now = meta.used - 1;
        state.reposition(physical, slot.chunk, meta.used, now);
        self.chunks[slot.chunk as usize].used = now;
        if now == 0 {
            state.spare.push(meta.frame);
            self.close_chunk(slot.chunk, physical);
        }
    }

    fn close_chunk(&mut self, logical: u32, physical: u32) {
        self.chunks[logical as usize] = FREE_CHUNK;
        self.free.set(physical);
        self.segments.let_go(physical);
        self.held -= 1;
        self.reopen(physical);
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
    /// lower segments, lowest first; none when they do not fit there, when
    /// the segment holds reserved chunks, or when the budget would not
    /// survive losing the segment.
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
        let sources: Vec<u32> = (range.start..range.end)
            .filter(|&chunk| !self.free.get(chunk))
            .collect();
        let moves: Vec<ChunkMove> = sources
            .iter()
            .zip(self.open.iter_in(0, range.start))
            .map(|(&from, to)| ChunkMove {
                logical: self.physical_to_logical[from as usize],
                from,
                to,
            })
            .collect();
        (moves.len() == sources.len()).then_some(Evacuation { segment, moves })
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

    /// Every draining segment waits on the fire at `epoch` too, where it
    /// waited on an earlier one: the fire known, once the engine took the
    /// evacuation, to come after its copy.
    pub fn fence_drains(&mut self, epoch: u64) {
        for segment in 0..self.segments.count() {
            if let SegmentState::Draining { until } = self.segments.state(segment)
                && until < epoch
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
            .filter(|&segment| self.backs(segment))
        {
            if room >= want {
                break;
            }
            room += u64::from(self.segments.len(segment));
            segments.push(segment);
        }
        segments
    }

    /// Checks every internal invariant; for tests.
    #[cfg(test)]
    fn check(&self) {
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
                state
                    .partial
                    .contains(&(Reverse(meta.used), physical, logical)),
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
            if segment >= self.top_segment {
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
            for &(Reverse(used), physical, logical) in &state.partial {
                assert_eq!(self.logical_to_physical[logical as usize], physical);
                assert_eq!(self.chunks[logical as usize].used, used);
                partial_free += u64::from(state.spc() - used);
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
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// splitmix64: a seeded stream, so a failing case replays.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn below(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
    }

    fn facts(
        total: u32,
        segment: u32,
        committed: u32,
        shapes: &[(ShapeKind, u32)],
    ) -> SlotPoolFacts {
        facts_with(
            total,
            segment,
            committed,
            shapes,
            &[(16, 1), (256, 3), (4096, 6)],
        )
    }

    fn facts_with(
        total: u32,
        segment: u32,
        committed: u32,
        shapes: &[(ShapeKind, u32)],
        act_chunks: &[(u32, u32)],
    ) -> SlotPoolFacts {
        SlotPoolFacts {
            chunk_bytes: 1 << 20,
            total_chunks: total,
            segment_chunks: segment,
            committed_segments: committed,
            reserved_chunks: 1,
            shapes: shapes
                .iter()
                .map(|&(kind, slots_per_chunk)| SlotShape {
                    kind,
                    slot_bytes: (1 << 20) / u64::from(slots_per_chunk),
                    slots_per_chunk,
                })
                .collect(),
            act_chunks: act_chunks.to_vec(),
        }
    }

    fn cache(key: u8) -> Holder {
        Holder::CacheEntry([key; 32])
    }

    /// What the test believes each holder holds.
    #[derive(Default)]
    struct Model {
        held: HashMap<Holder, Vec<Slot>>,
        /// Let go, and the epoch it waits on.
        pending: HashMap<Slot, u64>,
        retired: u64,
        touched: HashMap<Slot, u64>,
    }

    impl Model {
        fn refs(&self, slot: Slot) -> (u32, u32) {
            let mut refs = (0, 0);
            for (holder, slots) in &self.held {
                let count = slots.iter().filter(|&&s| s == slot).count() as u32;
                if holder.is_live() {
                    refs.0 += count;
                } else {
                    refs.1 += count;
                }
            }
            refs
        }
    }

    fn property_run(seed: u64, shapes: &[(ShapeKind, u32)]) {
        let mut rng = Rng(seed);
        let mut pool = SlotPool::new(facts(96, 16, 3, shapes));
        let mut model = Model::default();
        let mut epoch = 1u64;
        let holders = |rng: &mut Rng| match rng.below(3) {
            0 => Holder::WorkingSet(rng.below(6)),
            1 => Holder::Snapshot(rng.below(3)),
            _ => cache(rng.below(4) as u8),
        };
        for step in 0..3000 {
            match rng.below(12) {
                0..=3 => {
                    let holder = holders(&mut rng);
                    let shape = rng.below(shapes.len() as u64) as ShapeId;
                    let n = rng.below(12);
                    let available = pool.available(shape);
                    match pool.acquire(holder, shape, n) {
                        Ok(slots) => {
                            assert!(n <= available);
                            assert_eq!(slots.len() as u64, n);
                            let unique: HashSet<_> = slots.iter().collect();
                            assert_eq!(unique.len(), slots.len(), "a slot handed out twice");
                            for &slot in &slots {
                                assert_eq!(
                                    model.refs(slot),
                                    (0, 0),
                                    "{slot:?} handed out while held"
                                );
                                if let Some(&after) = model.pending.get(&slot) {
                                    assert!(
                                        after <= model.retired,
                                        "{slot:?} reused before epoch {after}"
                                    );
                                    model.pending.remove(&slot);
                                }
                                model.touched.remove(&slot);
                            }
                            model.held.entry(holder).or_default().extend(slots);
                        }
                        Err(SlotError::Short { .. }) => assert!(n > available),
                        Err(other) => panic!("{other}"),
                    }
                }
                4 => {
                    // A holder takes a reference on another's slots.
                    let from = holders(&mut rng);
                    let to = holders(&mut rng);
                    let mut slots: Vec<Slot> = model.held.get(&from).cloned().unwrap_or_default();
                    slots.retain(|slot| {
                        !model.held.get(&to).is_some_and(|held| held.contains(slot))
                    });
                    slots.sort();
                    slots.dedup();
                    slots.truncate(rng.below(6) as usize);
                    pool.share(to, &slots);
                    model.held.entry(to).or_default().extend(slots);
                }
                5..=7 => {
                    let holder = holders(&mut rng);
                    let Some(held) = model.held.get_mut(&holder) else {
                        continue;
                    };
                    let count = rng.below(held.len() as u64 + 1) as usize;
                    let gone: Vec<Slot> = held.drain(..count).collect();
                    let at = epoch.saturating_sub(rng.below(3));
                    pool.release(holder, &gone, at);
                    for slot in gone {
                        if model.refs(slot) == (0, 0) {
                            let after = at.max(model.touched.get(&slot).copied().unwrap_or(0));
                            model.pending.insert(slot, after);
                        }
                    }
                }
                8 => {
                    // A fire touches some held slots.
                    let mut slots: Vec<Slot> = model.held.values().flatten().copied().collect();
                    slots.sort();
                    slots.dedup();
                    slots.retain(|_| rng.below(3) == 0);
                    pool.touch(&slots, epoch);
                    for slot in slots {
                        model.touched.insert(slot, epoch);
                    }
                    epoch += 1;
                }
                9 => {
                    let through = epoch.saturating_sub(rng.below(4));
                    pool.retire_through(through);
                    model.retired = model.retired.max(through);
                }
                10 => {
                    let rung = [0, 200, 1000, 5000][rng.below(4) as usize];
                    if pool.set_rung(rung).is_err()
                        && let Ok(evacuation) = pool.plan_rung(rung)
                    {
                        let refs: Vec<(Slot, u32)> = model
                            .held
                            .values()
                            .flatten()
                            .map(|&slot| (slot, pool.slot_ref(slot)))
                            .collect();
                        pool.apply_rung(&evacuation, epoch).unwrap();
                        for (slot, engine) in refs {
                            assert_eq!(pool.slot_ref(slot), engine);
                            assert!(pool.physical(slot.chunk).unwrap() < pool.state_limit());
                        }
                        epoch += 1;
                    }
                }
                _ => {
                    // Shrink or grow the pool.
                    if rng.below(2) == 0 {
                        if let Some(segment) = pool.emptiest_segment()
                            && let Some(evacuation) = pool.plan_evacuation(segment)
                        {
                            let refs: Vec<(Slot, (u32, u32), u32)> = model
                                .held
                                .values()
                                .flatten()
                                .map(|&slot| (slot, pool.refs(slot), pool.slot_ref(slot)))
                                .collect();
                            pool.apply_evacuation(&evacuation, epoch).unwrap();
                            assert_eq!(pool.segments().used(segment), 0);
                            for (slot, refs, engine) in refs {
                                assert_eq!(pool.refs(slot), refs);
                                assert_eq!(pool.slot_ref(slot), engine);
                                let physical = pool.physical(slot.chunk).unwrap();
                                assert_ne!(pool.segments().of(physical), segment);
                            }
                            epoch += 1;
                        }
                        for segment in 0..pool.segments().count() {
                            let _ = pool.release_segment(segment);
                        }
                    } else if let Some(&segment) = pool.commit_ahead(8 << 20).first() {
                        pool.commit_segment(segment).unwrap();
                    }
                }
            }
            if step % 7 == 0 {
                pool.check();
            }
            // Refcount conservation, holder by holder and slot by slot.
            let mut total = 0;
            for (holder, slots) in &model.held {
                assert_eq!(pool.held_by(holder), slots.len() as u64, "{holder:?}");
                total += slots.len() as u64;
            }
            let pool_total: u64 = pool.holders.values().sum();
            assert_eq!(pool_total, total);
            if step % 50 == 0 {
                let mut all: Vec<Slot> = model.held.values().flatten().copied().collect();
                all.sort();
                all.dedup();
                for slot in all {
                    assert_eq!(pool.refs(slot), model.refs(slot));
                }
            }
            assert!(
                pool.reserved_chunks() + pool.state_chunks() + pool.act_reserved()
                    <= pool.committed_chunks()
            );
            assert!(pool.committed_chunks() <= pool.total_chunks());
        }
        // Let everything go: the pool comes back whole.
        for (holder, slots) in std::mem::take(&mut model.held) {
            pool.release(holder, &slots, epoch);
        }
        pool.retire_through(u64::MAX);
        assert_eq!(pool.state_chunks(), 0);
        assert_eq!(pool.demand(), Demand::default());
        pool.check();
    }

    /// A shrink's evacuation (suspect d): a fire built before it may still
    /// read or write the drained segment's old place, so the segment is not
    /// given back (`release_segments`) until the fire at the evacuation's
    /// epoch retires.
    #[test]
    fn a_drained_segment_is_held_until_the_fires_that_named_it_retire() {
        let mut pool = SlotPool::new(facts(96, 16, 3, &[(ShapeKind::List, 1)]));
        let holder = Holder::WorkingSet(0);
        let slots = pool.acquire(holder, 0, 40).unwrap();
        // The low chunks go back: room below for what sits in segment 2.
        let (low, kept): (Vec<Slot>, Vec<Slot>) = slots
            .into_iter()
            .partition(|slot| pool.physical(slot.chunk).unwrap() < 24);
        pool.release(holder, &low, 0);
        pool.retire_through(0);
        let segment = 2;
        assert!(
            kept.iter()
                .any(|slot| pool.segments().of(pool.physical(slot.chunk).unwrap()) == segment),
            "state sits in the segment to drain"
        );
        let evacuation = pool.plan_evacuation(segment).expect("room below it");
        assert!(!evacuation.moves.is_empty());
        const FIRE: u64 = 7;
        pool.apply_evacuation(&evacuation, FIRE).unwrap();
        assert_eq!(pool.segments().used(segment), 0);
        pool.retire_through(FIRE - 1);
        assert!(
            matches!(
                pool.release_segment(segment),
                Err(SegmentError::Draining { .. })
            ),
            "a drained segment went back while a fire naming its old place was in flight"
        );
        pool.retire_through(FIRE);
        pool.release_segment(segment).unwrap();
    }

    /// A raise's evacuation (suspect d): the chunks it moves state off stay
    /// out of the pool until the fire at its epoch retires, even when the
    /// rung comes back down at once: a fire built before the move may still
    /// read or write them, and no slot of another holder may lie there.
    #[test]
    fn a_raise_s_moved_off_chunks_open_for_no_one_until_their_fires_retire() {
        let mut pool = SlotPool::new(facts(96, 16, 6, &[(ShapeKind::List, 1)]));
        pool.set_rung(16).unwrap();
        let holder = Holder::WorkingSet(0);
        let all = pool.acquire(holder, 0, pool.available(0)).unwrap();
        let (low, kept): (Vec<Slot>, Vec<Slot>) = all
            .into_iter()
            .partition(|slot| pool.physical(slot.chunk).unwrap() < 40);
        pool.release(holder, &low, 0);
        pool.retire_through(0);
        let evacuation = pool.plan_rung(4096).unwrap();
        assert!(!evacuation.moves.is_empty(), "state sits in the span");
        let moved_off: HashSet<u32> = evacuation.moves.iter().map(|step| step.from).collect();
        const FIRE: u64 = 9;
        pool.apply_rung(&evacuation, FIRE).unwrap();
        // Down again before the fire that named them retires.
        pool.set_rung(16).unwrap();
        pool.retire_through(FIRE - 1);
        let other = Holder::WorkingSet(1);
        let taken = pool.acquire(other, 0, pool.available(0)).unwrap();
        for slot in &taken {
            let physical = pool.physical(slot.chunk).unwrap();
            assert!(
                !moved_off.contains(&physical),
                "{slot:?} handed out on chunk {physical}, which a fire in flight still names"
            );
        }
        pool.retire_through(FIRE);
        let after = pool.acquire(other, 0, pool.available(0)).unwrap();
        assert!(
            after
                .iter()
                .any(|slot| moved_off.contains(&pool.physical(slot.chunk).unwrap())),
            "the moved-off chunks open once the fire retires"
        );
        let _ = kept;
    }

    #[test]
    fn random_runs_keep_every_invariant() {
        for seed in 0..24 {
            property_run(
                seed,
                &[
                    (ShapeKind::List, 4),
                    (ShapeKind::Fixed, 1),
                    (ShapeKind::List, 70),
                ],
            );
        }
    }
}
