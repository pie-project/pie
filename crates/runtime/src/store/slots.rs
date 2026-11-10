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
            act_max,
            top_segment,
            chunks: vec![FREE_CHUNK; total as usize],
            logical_to_physical: (0..total).collect(),
            physical_to_logical: (0..total).collect(),
            free,
            open: Bits::new(total, false),
            fences: BTreeMap::new(),
            fenced: Bits::new(total, false),
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
            && self.segments.state(self.segments.of(chunk)) == SegmentState::Committed
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
    /// or above it, else the largest.
    pub fn act_chunks(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(0, |(_, chunks)| *chunks)
    }

    /// The activation chunks set aside now.
    pub fn act_reserved(&self) -> u32 {
        self.act
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

    /// Committed chunks neither held nor set aside for activation.
    pub fn headroom(&self) -> u32 {
        self.committed_chunks()
            .saturating_sub(self.reserved + self.held + self.act)
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
        for segment in self.segments.uncommitted() {
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

    #[test]
    fn a_shape_fills_its_fullest_partly_filled_chunk_first() {
        let mut pool = SlotPool::new(facts(16, 8, 2, &[(ShapeKind::List, 4)]));
        let ws = Holder::WorkingSet(1);
        let slots = pool.acquire(ws, 0, 12).unwrap();
        let chunks: Vec<u32> = slots.iter().map(|slot| slot.chunk).collect();
        assert_eq!(chunks, [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
        // Chunk 1 keeps 3, chunk 2 keeps 1, chunk 3 keeps 2.
        pool.release(ws, &[slots[0]], 0);
        pool.release(ws, &slots[4..7], 0);
        pool.release(ws, &slots[8..10], 0);
        pool.check();
        let next = pool.acquire(ws, 0, 4).unwrap();
        assert_eq!(
            next,
            [
                Slot { chunk: 1, index: 0 },
                Slot { chunk: 3, index: 0 },
                Slot { chunk: 3, index: 1 },
                Slot { chunk: 2, index: 0 },
            ]
        );
        pool.check();
    }

    #[test]
    fn a_let_go_slot_waits_for_the_last_fire_that_touched_it() {
        let mut pool = SlotPool::new(facts_with(3, 3, 1, &[(ShapeKind::Fixed, 1)], &[(16, 1)]));
        pool.set_rung(0).unwrap();
        let ws = Holder::WorkingSet(7);
        let slots = pool.acquire(ws, 0, 1).unwrap();
        pool.touch(&slots, 9);
        pool.release(ws, &slots, 4);
        assert!(pool.acquire(ws, 0, 1).is_err(), "act takes the other chunk");
        pool.retire_through(8);
        assert_eq!(pool.pending_slots(), 1);
        pool.retire_through(9);
        assert_eq!(pool.pending_slots(), 0);
        assert_eq!(pool.state_chunks(), 0, "an empty chunk is free again");
        assert_eq!(pool.acquire(ws, 0, 1).unwrap(), slots);
        pool.check();
    }

    #[test]
    fn a_shared_slot_lives_until_its_last_holder_lets_go() {
        let mut pool = SlotPool::new(facts(8, 8, 1, &[(ShapeKind::Fixed, 2)]));
        let ws = Holder::WorkingSet(1);
        let slots = pool.acquire(ws, 0, 2).unwrap();
        pool.share(cache(3), &slots[..1]);
        pool.share(Holder::Snapshot(2), &slots[1..]);
        let bytes = 1 << 19;
        assert_eq!(
            pool.demand(),
            Demand {
                live: 2 * bytes,
                wanted: 2 * bytes
            }
        );
        pool.release(ws, &slots, 0);
        assert_eq!(
            pool.demand(),
            Demand {
                live: bytes,
                wanted: 2 * bytes
            }
        );
        pool.wait(5, 0, 3);
        assert_eq!(pool.demand().wanted, 5 * bytes);
        pool.unwait(5);
        pool.release(cache(3), &slots[..1], 0);
        pool.release(Holder::Snapshot(2), &slots[1..], 0);
        assert_eq!(pool.state_chunks(), 0);
        assert_eq!(pool.held_by(&ws), 0);
        pool.check();
    }

    #[test]
    fn a_rung_that_does_not_fit_beside_the_state_is_refused() {
        let mut pool = SlotPool::new(facts(8, 8, 1, &[(ShapeKind::Fixed, 1)]));
        let ws = Holder::WorkingSet(1);
        pool.acquire(ws, 0, 4).unwrap();
        assert!(pool.set_rung(256).is_ok());
        assert_eq!(pool.headroom(), 0);
        assert!(matches!(
            pool.set_rung(4096),
            Err(SlotError::Budget {
                needed: 11,
                committed: 8
            })
        ));
        assert!(matches!(
            pool.acquire(ws, 0, 1),
            Err(SlotError::Short {
                slots: 1,
                chunks: 1,
                ..
            })
        ));
        pool.check();
    }

    #[test]
    fn commit_ahead_names_the_lowest_segments_short_of_the_bytes() {
        let mut pool = SlotPool::new(facts(32, 8, 1, &[(ShapeKind::Fixed, 1)]));
        // Segment 0 and the top segment under the largest rung's 6 chunks
        // are committed: 16 - 1 reserved - 1 act = 14 headroom.
        assert_eq!(pool.committed_chunks(), 16);
        assert_eq!(pool.top_segment(), 3);
        assert!(pool.commit_ahead(14 << 20).is_empty());
        assert_eq!(pool.commit_ahead(15 << 20), [1]);
        assert_eq!(pool.commit_ahead(100 << 20), [1, 2]);
        pool.commit_segment(2).unwrap();
        assert_eq!(pool.prefix_segments(), 1, "a prefix stops at the gap");
        assert_eq!(pool.commit_ahead(23 << 20), [1]);
        pool.commit_segment(1).unwrap();
        assert_eq!(pool.prefix_segments(), 3);
        pool.check();
    }

    #[test]
    fn an_evacuation_empties_its_segment_and_keeps_every_logical_slot() {
        let mut pool = SlotPool::new(facts_with(
            32,
            8,
            3,
            &[(ShapeKind::Fixed, 1), (ShapeKind::List, 2)],
            &[(16, 1)],
        ));
        let a = Holder::WorkingSet(1);
        let b = Holder::WorkingSet(2);
        // Fixed slots in chunks 1..=20, then list slots in 21 and 22 (22 half full).
        let fixed = pool.acquire(a, 0, 20).unwrap();
        let list = pool.acquire(b, 1, 3).unwrap();
        pool.share(cache(9), &list[2..]);
        assert_eq!(pool.physical(list[2].chunk), Some(22));
        // Too much is held below for the top segment to fit.
        assert!(pool.plan_evacuation(2).is_none());
        pool.release(a, &fixed[..15], 0);
        let kept: Vec<Slot> = fixed[15..].iter().chain(&list).copied().collect();
        let before: Vec<(Slot, (u32, u32), u32)> = kept
            .iter()
            .map(|&slot| (slot, pool.refs(slot), pool.slot_ref(slot)))
            .collect();
        assert_eq!(pool.emptiest_segment(), Some(1));
        let evacuation = pool.plan_evacuation(2).unwrap();
        assert_eq!(
            evacuation
                .moves
                .iter()
                .map(|step| step.to)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4, 5, 6, 7]
        );
        pool.apply_evacuation(&evacuation, 5).unwrap();
        pool.check();
        assert_eq!(pool.segments().used(2), 0);
        for (slot, refs, engine) in before {
            assert_eq!(pool.refs(slot), refs);
            assert_eq!(pool.slot_ref(slot), engine, "a held slot is never renamed");
            assert_eq!(pool.segments().of(pool.physical(slot.chunk).unwrap()), 0);
        }
        assert!(matches!(
            pool.release_segment(2),
            Err(SegmentError::Draining { until: 5, .. })
        ));
        pool.retire_through(5);
        pool.release_segment(2).unwrap();
        assert_eq!(pool.segments().state(2), SegmentState::Uncommitted);
        assert_eq!(pool.committed_chunks(), 24);
        assert!(
            matches!(
                pool.release_segment(3),
                Err(SegmentError::State { segment: 3, .. })
            ),
            "the activation's top segment is never released"
        );
        assert!(pool.plan_evacuation(3).is_none());
        assert!(matches!(
            pool.release_segment(0),
            Err(SegmentError::Held { .. })
        ));
        // The half-full list chunk moved and is still filled first.
        let next = pool.acquire(b, 1, 1).unwrap();
        assert_eq!(next[0].chunk, list[2].chunk);
        pool.check();
    }

    #[test]
    fn a_stale_evacuation_is_refused_whole() {
        let mut pool = SlotPool::new(facts_with(24, 8, 2, &[(ShapeKind::Fixed, 1)], &[(16, 1)]));
        let ws = Holder::WorkingSet(1);
        let slots = pool.acquire(ws, 0, 9).unwrap();
        pool.release(ws, &slots[..4], 0);
        let evacuation = pool.plan_evacuation(1).unwrap();
        // A chunk planned as a target is taken first.
        assert_eq!(pool.acquire(ws, 0, 1).unwrap()[0].chunk, 1);
        assert_eq!(
            pool.apply_evacuation(&evacuation, 1),
            Err(SegmentError::Stale(1))
        );
        // A chunk planned as a source is let go.
        let evacuation = pool.plan_evacuation(1).unwrap();
        pool.release(ws, &slots[8..], 0);
        assert_eq!(
            pool.apply_evacuation(&evacuation, 1),
            Err(SegmentError::Stale(1))
        );
        pool.check();
    }

    #[test]
    fn state_lies_below_the_activation_and_the_top_span_is_committed_from_load() {
        // 64 chunks of 8-chunk segments, the lowest committed; the largest
        // rung's 12 chunks put segments 6 and 7 under activation.
        let mut pool = SlotPool::new(facts_with(
            64,
            8,
            1,
            &[(ShapeKind::Fixed, 1)],
            &[(16, 2), (256, 6), (4096, 12)],
        ));
        assert_eq!(pool.top_segment(), 6);
        assert_eq!(pool.committed_chunks(), 8 + 16);
        assert_eq!(pool.state_limit(), 62);
        let ws = Holder::WorkingSet(1);
        // 7 below in segment 0, then the top span's free chunks under the
        // rung: 48..62.
        let slots = pool.acquire(ws, 0, 21).unwrap();
        let physical: Vec<u32> = slots
            .iter()
            .map(|slot| pool.physical(slot.chunk).unwrap())
            .collect();
        assert_eq!(physical[..7], [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(physical[7..], (48..62).collect::<Vec<_>>()[..]);
        assert!(pool.acquire(ws, 0, 1).is_err(), "the rung's 2 chunks stay");
        // Committing segment 1 opens it, lowest first.
        pool.commit_segment(1).unwrap();
        let next = pool.acquire(ws, 0, 1).unwrap();
        assert_eq!(pool.physical(next[0].chunk), Some(8));
        pool.check();
    }

    #[test]
    fn raising_the_rung_first_moves_state_out_of_the_new_activation_span() {
        // Segments 0..3 committed, segment 3 under the largest rung's 8.
        let mut pool = SlotPool::new(facts_with(
            32,
            8,
            3,
            &[(ShapeKind::Fixed, 1), (ShapeKind::List, 4)],
            &[(16, 1), (4096, 8)],
        ));
        let a = Holder::WorkingSet(1);
        // Fixed slots on physical 1..=28, the list chunk on 29.
        let fixed = pool.acquire(a, 0, 28).unwrap();
        let list = pool.acquire(a, 1, 3).unwrap();
        assert_eq!(pool.physical(list[0].chunk), Some(29));
        pool.release(a, &fixed[..10], 0);
        let refs: Vec<u32> = fixed[10..]
            .iter()
            .chain(&list)
            .map(|&slot| pool.slot_ref(slot))
            .collect();
        assert!(matches!(
            pool.set_rung(4096),
            Err(SlotError::Occupied { chunks: 6, .. })
        ));
        let evacuation = pool.plan_rung(4096).unwrap();
        assert_eq!(evacuation.act, 8);
        let from: Vec<u32> = evacuation.moves.iter().map(|step| step.from).collect();
        let to: Vec<u32> = evacuation.moves.iter().map(|step| step.to).collect();
        assert_eq!(from, [24, 25, 26, 27, 28, 29]);
        assert_eq!(to, [1, 2, 3, 4, 5, 6], "lowest free first");
        pool.apply_rung(&evacuation, 7).unwrap();
        pool.check();
        assert_eq!(pool.act_reserved(), 8);
        assert_eq!(pool.state_limit(), 24);
        let after: Vec<u32> = fixed[10..]
            .iter()
            .chain(&list)
            .map(|&slot| pool.slot_ref(slot))
            .collect();
        assert_eq!(after, refs, "slot refs are logical: a move renames none");
        for slot in fixed[10..].iter().chain(&list) {
            assert!(pool.physical(slot.chunk).unwrap() < 24);
        }
        // The table is the permutation the engine's comes to.
        let mut table = ::engine::chunk::ChunkTable::identity(32);
        table
            .apply(&::engine::chunk::Evacuation {
                moves: evacuation
                    .moves
                    .iter()
                    .map(|step| (step.logical, step.to))
                    .collect(),
            })
            .unwrap();
        assert_eq!(table.as_slice(), pool.chunk_table());
        // A stale plan is refused whole.
        assert_eq!(pool.apply_rung(&evacuation, 8), Err(SlotError::StaleRung));
        // Lowering the rung opens the span again, the chunks moved off once
        // the fire carrying the copy retires.
        pool.set_rung(16).unwrap();
        assert_eq!(pool.available(0), 4 + 1, "the moved-off chunks are fenced");
        pool.retire_through(7);
        assert_eq!(pool.available(0), 4 + 1 + 6);
        // New chunks open lowest first, below where the moved ones went.
        let next = pool.acquire(a, 0, 1).unwrap();
        assert_eq!(pool.physical(next[0].chunk), Some(7));
        pool.check();
    }

    #[test]
    fn a_rung_with_no_room_below_is_refused_and_moves_nothing() {
        let mut pool = SlotPool::new(facts_with(
            16,
            8,
            2,
            &[(ShapeKind::Fixed, 1)],
            &[(16, 1), (4096, 4)],
        ));
        let a = Holder::WorkingSet(1);
        pool.acquire(a, 0, 13).unwrap();
        assert!(matches!(
            pool.plan_rung(4096),
            Err(SlotError::Budget { .. })
        ));
        let mut pool = SlotPool::new(facts_with(
            16,
            8,
            2,
            &[(ShapeKind::Fixed, 1)],
            &[(16, 1), (4096, 4)],
        ));
        let slots = pool.acquire(a, 0, 14).unwrap();
        pool.release(a, &slots[..3], 0);
        // 3 free below, 3 in the way of the 4-chunk span: they fit.
        assert_eq!(pool.plan_rung(4096).unwrap().moves.len(), 3);
        pool.check();
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

    #[test]
    fn a_working_set_holds_one_set_of_each_shape() {
        let fixed = SlotShape {
            kind: ShapeKind::Fixed,
            slot_bytes: 1,
            slots_per_chunk: 1,
        };
        assert_eq!(fixed.sets_per_working_set(), 1);
        assert_eq!(
            SlotShape {
                kind: ShapeKind::List,
                ..fixed
            }
            .sets_per_working_set(),
            1
        );
    }

    /// A million single-page chunks, slot by slot and in one batch, against
    /// today's page pool. Run with `--release --ignored --nocapture`.
    #[test]
    #[ignore = "a timing check; run in release"]
    fn a_million_single_page_chunks_keep_pace_with_the_page_pool() {
        use super::super::pool::{Pool, PoolId};
        use std::time::Instant;

        #[derive(Clone, Copy)]
        struct Page(u32);
        impl PoolId for Page {
            fn from_index(index: u32) -> Self {
                Page(index)
            }
            fn index(self) -> u32 {
                self.0
            }
        }

        const N: u32 = 1 << 20;
        let ws = Holder::WorkingSet(1);
        let fresh = || {
            SlotPool::new(SlotPoolFacts {
                chunk_bytes: 1 << 16,
                total_chunks: N,
                segment_chunks: 1 << 12,
                committed_segments: N >> 12,
                reserved_chunks: 0,
                shapes: vec![SlotShape {
                    kind: ShapeKind::List,
                    slot_bytes: 1 << 16,
                    slots_per_chunk: 1,
                }],
                act_chunks: Vec::new(),
            })
        };

        // One at a time.
        let mut pages = Pool::<Page>::new(N);
        let at = Instant::now();
        let mut held = Vec::with_capacity(N as usize);
        for _ in 0..N {
            held.extend(pages.try_alloc_n(1).unwrap());
        }
        for page in held.drain(..) {
            pages.recycle_after_epoch(vec![page], 1);
        }
        pages.retire_through(1);
        let pool_single = at.elapsed();

        let mut slots = fresh();
        let at = Instant::now();
        let mut taken = Vec::with_capacity(N as usize);
        for _ in 0..N {
            taken.extend(slots.acquire(ws, 0, 1).unwrap());
        }
        for slot in taken.drain(..) {
            slots.release(ws, &[slot], 1);
        }
        slots.retire_through(1);
        let slot_single = at.elapsed();
        assert_eq!(slots.state_chunks(), 0);

        // In one batch.
        let at = Instant::now();
        let held = pages.try_alloc_n(N as usize).unwrap();
        pages.recycle_after_epoch(held, 2);
        pages.retire_through(2);
        let pool_batch = at.elapsed();

        let at = Instant::now();
        let taken = slots.acquire(ws, 0, u64::from(N)).unwrap();
        let acquired = at.elapsed();
        slots.release(ws, &taken, 2);
        let released = at.elapsed();
        slots.retire_through(2);
        let slot_batch = at.elapsed();
        eprintln!(
            "slots batch: acquire {acquired:?}, release {:?}, retire {:?}",
            released - acquired,
            slot_batch - released
        );

        eprintln!(
            "1M single-page chunks, alloc+free: one at a time pool {pool_single:?} slots {slot_single:?} ({:.1}x); \
             batch pool {pool_batch:?} slots {slot_batch:?} ({:.1}x)",
            slot_single.as_secs_f64() / pool_single.as_secs_f64(),
            slot_batch.as_secs_f64() / pool_batch.as_secs_f64(),
        );
        if !cfg!(debug_assertions) {
            assert!(slot_single.as_secs_f64() < 10.0 * pool_single.as_secs_f64());
        }
    }
}
