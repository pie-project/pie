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
//! evacuated or released. Live state opens only below the top rung's span
//! ([`SlotPool::live_limit`]) and no activation is set aside past it
//! ([`SlotError::PastSpan`]), so raising the rung moves nothing; it only
//! clears the idle cache a lowered rung left in the span
//! ([`SlotPool::plan_rung`], [`SlotError::SpanCached`]).
//!
//! Logical ids mirror the engine's `ChunkTable`: logical and physical ids are
//! a permutation of `0..total`, the identity at load, and only a segment's
//! evacuation (a shrink) changes it, the way `ChunkTable::apply` does (the
//! moved chunk takes the target's place, the free logical id there takes
//! the moved one's old place). Opening physical chunk `p` hands out the logical id the table maps
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
    /// `(rung tokens, activation chunks)`, ascending. None stated (a load
    /// whose activation lies outside the pool) is one rung of fires of any
    /// tokens whose carve takes no chunk of it, its span the room the load
    /// backs for it (`act_span`, else none): its readout and standing bytes
    /// are cut and refused against that span as against any top rung's.
    pub act_chunks: Vec<(u32, u32)>,
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
    #[error("rung {rung}'s activation takes {chunks} chunks state holds")]
    Occupied { rung: u32, chunks: u32 },
    /// A rung whose activation, its readout rows and the engine's standing
    /// guest bytes included, takes more than the top rung's span: never set
    /// aside (live state lies below the span), so fires are cut to the
    /// widest rung that fits (`SharedPool::widest_fit`).
    #[error("rung {rung}'s activation takes {chunks} chunks, past the top rung's span of {span}")]
    PastSpan { rung: u32, chunks: u32, span: u32 },
    /// A rung whose activation lies in span segments yield gave back under
    /// pressure (its last shed): set aside once the governor's grow backs
    /// them again (`Engine::commit_span`), never before.
    #[error(
        "rung {rung}'s activation needs the top {segments} span segments, which yield gave \
         back: the governor's grow backs them again"
    )]
    SpanShed { rung: u32, segments: u32 },
    #[error("a rung evacuation no longer matches the pool")]
    StaleRung,
    /// A claim named a chunk that cannot be taken whole as it stands.
    #[error("chunk {chunk} cannot be claimed: {why}")]
    Unclaimable { chunk: u32, why: &'static str },
    /// Live state asked for chunks only the top rung's span has free: live
    /// state opens no chunk at or past the span's base
    /// ([`SlotPool::live_limit`]), whatever rung is set aside, so a raise
    /// never has live state there to move.
    #[error(
        "shape {shape}'s live slots need {chunks} more chunks below the top rung's span: \
         the top rung's span holds only idle cache"
    )]
    LiveInSpan { shape: ShapeId, chunks: u32 },
    /// A raise to `rung` finds idle cache in the span its activation takes:
    /// the cache's owner lets it go first (dropped, or demoted by its keys,
    /// [`SlotPool::clear_span`]); a raise only ever clears cache, never
    /// moves it.
    #[error("rung {rung}'s activation takes {chunks} chunks idle cache holds; clear them first")]
    SpanCached { rung: u32, chunks: u32 },
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

/// A raise of the activation, as planned: nothing moves, live state never
/// lying in the top rung's span and no rung's activation reaching past it
/// ([`SlotPool::plan_rung`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RungRaise {
    pub rung: u32,
    /// The rung the activation was set aside for when it was planned.
    pub from: u32,
    /// The activation chunks the rung takes.
    pub act: u32,
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
    /// The engine's standing guest bytes, below every rung's carve
    /// ([`Self::set_standing`]).
    standing: u64,
    /// The readout term ([`Self::set_readouts`]): the bytes a row a fire
    /// reads out takes beside the model's activation (0: inside the span),
    /// and the most rows one fire reads out.
    readout_row_bytes: u64,
    readout_rows: u32,
    /// The largest rung's activation chunks, the top of the range.
    act_max: u32,
    /// The first segment under the largest rung's activation: it and those
    /// above (the top rung's span) are committed from the load and never
    /// evacuated; given back only by yield, its last shed, below the
    /// activation set aside ([`Self::release_span`]), and backed again by
    /// the governor's grow ([`Self::commit_span`]).
    top_segment: u32,
    /// What the engine states its activation span covers as it stands
    /// ([`Self::set_act_span`]): the statement itself, nothing derived.
    act_span: Option<u32>,
    chunks: Vec<ChunkMeta>,
    /// A permutation of `0..total`, as the engine's chunk table.
    logical_to_physical: Vec<u32>,
    physical_to_logical: Vec<u32>,
    /// Physical chunks never handed out: the reserved ones from 0 and the
    /// skipped ids ([`Self::skip`]). Each stays where it lies.
    fixed: Bits,
    /// The skipped runs of chunk ids, `first..end` ([`Self::skip`]).
    skips: Vec<(u32, u32)>,
    /// Free physical chunks.
    free: Bits,
    /// Free physical chunks of committed segments below the activation:
    /// what may be opened.
    open: Bits,
    /// No chunk at or past this physical id is opened: the engine binds
    /// none there yet ([`Self::set_ceiling`]).
    ceiling: u32,
    /// Idle cache opens and fills no chunk at or past this physical id: a
    /// raise asked for the span from here clear ([`Self::clear_span`]),
    /// until the activation is set aside again. `total` while none asks.
    clearing: u32,
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
}

impl SlotPool {
    pub fn new(facts: SlotPoolFacts) -> Self {
        let total = facts.total_chunks;
        assert!(facts.reserved_chunks <= total);
        assert!(facts.shapes.len() < NO_SHAPE as usize);
        let mut act_chunks = facts.act_chunks;
        act_chunks.sort_unstable();
        // The top rung's span: its carve's chunks; a load stating no rung
        // is the one rung of no carve, its span the room the load backs.
        let act_max = match act_chunks.last() {
            Some(&(_, chunks)) => chunks,
            None => {
                act_chunks.push((u32::MAX, 0));
                facts.act_span.unwrap_or(0)
            }
        };
        assert!(
            facts.reserved_chunks + act_max <= total,
            "the reserved chunks and the largest rung's activation outgrow the pool"
        );
        let mut segments = Segments::new(total, facts.segment_chunks, facts.committed_segments);
        // The top rung's span is backed from the load.
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
        let mut fixed = Bits::new(total, false);
        for chunk in 0..facts.reserved_chunks {
            free.clear(chunk);
            fixed.set(chunk);
            segments.hold(chunk);
        }
        let mut pool = Self {
            chunk_bytes: facts.chunk_bytes,
            total,
            reserved: facts.reserved_chunks,
            shapes: facts.shapes.into_iter().map(ShapeState::new).collect(),
            act_chunks,
            act: 0,
            standing: 0,
            readout_row_bytes: 0,
            readout_rows: u32::MAX,
            act_max,
            top_segment,
            act_span: facts.act_span,
            chunks: vec![FREE_CHUNK; total as usize],
            logical_to_physical: (0..total).collect(),
            physical_to_logical: (0..total).collect(),
            fixed,
            skips: Vec::new(),
            free,
            open: Bits::new(total, false),
            ceiling: total,
            clearing: total,
            segments,
            held: 0,
            pending: BTreeMap::new(),
            retired: 0,
            holders: HashMap::new(),
            waiters: HashMap::new(),
            runs: HashMap::new(),
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

    /// The chunk ids `first..first + chunks` are skipped: the backend
    /// never backs them, so none is handed out, counted in a segment's
    /// chunks or moved, and no run of adjacent chunks crosses them; the
    /// ids above keep their place.
    pub fn skip(&mut self, first: u32, chunks: u32) {
        let end = first.saturating_add(chunks).min(self.total);
        if first < end {
            self.skips.push((first, end));
        }
        for chunk in first..end {
            if self.fixed.get(chunk) {
                continue;
            }
            self.free.clear(chunk);
            self.fixed.set(chunk);
            self.segments.skip(chunk);
            self.reopen(chunk);
        }
    }

    /// Whether physical chunk `chunk` is never handed out: reserved or
    /// skipped. Its logical id is its physical one.
    fn is_fixed(&self, chunk: u32) -> bool {
        self.fixed.get(chunk)
    }

    /// Whether physical chunk `chunk` may be opened: free, in a committed
    /// segment, below the activation.
    fn openable(&self, chunk: u32) -> bool {
        self.free.get(chunk)
            && chunk < self.state_limit()
            && chunk < self.ceiling
            && self.segments.state(self.segments.of(chunk)) == SegmentState::Committed
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

    /// Chunks with memory under them for the pool's use: the committed
    /// segments'.
    pub fn committed_chunks(&self) -> u32 {
        self.segments.committed_chunks()
    }

    /// Activation chunks a fire at `rung` tokens takes: the smallest rung at
    /// or above it, else the largest; with the engine's standing guest
    /// bytes ([`Self::set_standing`]) and the rung's readout rows, one ceil
    /// over the two. A guest program takes nothing here: its scratch is
    /// state of its instance (`::engine::GuestState`).
    pub fn act_chunks(&self, rung: u32) -> u32 {
        self.act_chunks
            .iter()
            .find(|(tokens, _)| *tokens >= rung)
            .or(self.act_chunks.last())
            .map_or(0, |&(tokens, chunks)| {
                chunks.saturating_add(self.fire_chunks(tokens))
            })
    }

    /// The chunks a rung of `tokens` takes beside its carve: what the
    /// engine keeps standing for every live instance whatever the fire, and
    /// the rows such a fire reads out where they land beside the activation
    /// ([`Self::readout_bytes`]).
    fn fire_chunks(&self, tokens: u32) -> u32 {
        let bytes = self.standing.saturating_add(self.readout_bytes(tokens));
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
    /// bound but the fire's tokens). Nothing moves here: a term that raises
    /// the rung's need is taken by a raise ([`Self::plan_rung`]).
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

    /// The engine's standing guest bytes from now on
    /// ([`::engine::Engine::standing_after`]): the span bytes it keeps for
    /// the live instances of registered programs beside every fire, with
    /// what the binds under way may add. Set aside below the carve at
    /// every rung ([`Self::act_chunks`]); nothing moves here, as with
    /// [`Self::set_readouts`]. The bytes it replaces are handed back.
    pub fn set_standing(&mut self, bytes: u64) -> u64 {
        std::mem::replace(&mut self.standing, bytes)
    }

    /// See [`Self::set_standing`].
    #[must_use]
    pub fn standing(&self) -> u64 {
        self.standing
    }

    /// Whether nothing takes the span beside a rung's carve at any rung:
    /// nothing standing, and no readout row beside the activation.
    #[must_use]
    pub fn guests_zero(&self) -> bool {
        self.standing == 0 && self.readout_row_bytes == 0
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
    /// chunks it holds, the standing and readout bytes included (0 where
    /// no rung is stated, or where standing bytes booked since leave it
    /// holding none until a raise).
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
        let committed = self.committed_chunks();
        if needed > committed {
            return Err(SlotError::Budget { needed, committed });
        }
        Ok(())
    }

    /// State chunks at physical ids `from..to`.
    fn held_in(&self, from: u32, to: u32) -> Vec<u32> {
        (from..to)
            .filter(|&chunk| !self.free.get(chunk) && !self.is_fixed(chunk))
            .collect()
    }

    /// Sets the activation aside for fires up to `rung` tokens; refused when
    /// the state already held leaves too little, or when state sits in the
    /// span the rung's activation takes (raise it with [`Self::plan_rung`]
    /// and [`Self::apply_rung`] then).
    pub fn set_rung(&mut self, rung: u32) -> Result<(), SlotError> {
        let act = self.act_in_span(rung)?;
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
        // The activation is set aside anew: what a raise asked the cache
        // to clear is taken, or no longer asked.
        self.clearing = self.total;
        for chunk in was.min(now)..was.max(now) {
            self.reopen(chunk);
        }
    }

    /// The chunks in the way of an activation from physical `limit`: idle
    /// cache in the top rung's span, which a raise only clears. Nothing
    /// else lies there: live state opens below the span's base, and no
    /// activation set aside reaches past it ([`SlotError::PastSpan`]).
    fn in_way(&self, limit: u32) -> Vec<u32> {
        debug_assert!(limit >= self.live_limit(), "an activation past the span");
        let held = self.held_in(limit, self.state_limit());
        debug_assert!(
            held.iter().all(|&chunk| !self.holds_live(chunk)),
            "live state lies in the top rung's span"
        );
        held
    }

    /// The activation chunks rung `rung` takes, refused where they reach
    /// past the top rung's span, or into span segments yield gave back
    /// ([`SlotError::SpanShed`]: the governor's grow backs them again).
    fn act_in_span(&self, rung: u32) -> Result<u32, SlotError> {
        let act = self.act_chunks(rung);
        if act > self.act_max {
            return Err(SlotError::PastSpan {
                rung,
                chunks: act,
                span: self.act_max,
            });
        }
        let segments = self.span_for(act);
        if segments > self.span_committed() {
            return Err(SlotError::SpanShed { rung, segments });
        }
        Ok(act)
    }

    /// The span segments, counted from the top, an activation of `act`
    /// chunks lies in.
    fn span_for(&self, act: u32) -> u32 {
        if act == 0 {
            return 0;
        }
        self.segments.count() - self.segments.of(self.total - act)
    }

    /// Whether any slot of physical chunk `physical` has a live reference.
    fn holds_live(&self, physical: u32) -> bool {
        let meta = self.chunks[self.physical_to_logical[physical as usize] as usize];
        if meta.shape == NO_SHAPE {
            return false;
        }
        let state = &self.shapes[meta.shape as usize];
        let spc = state.spc() as usize;
        let at = meta.frame as usize * spc;
        state.refs[at..at + spc].iter().any(|refs| refs.live > 0)
    }

    /// What raising the activation to rung `rung` takes: nothing moves,
    /// live state never lying in the top rung's span ([`Self::live_limit`])
    /// and no rung's activation reaching past it ([`SlotError::PastSpan`]);
    /// idle cache there is cleared first (refused with
    /// [`SlotError::SpanCached`] while it lies there).
    pub fn plan_rung(&self, rung: u32) -> Result<RungRaise, SlotError> {
        let act = self.act_in_span(rung)?;
        let cached = self.in_way(self.total - act);
        if !cached.is_empty() {
            return Err(SlotError::SpanCached {
                rung,
                chunks: cached.len() as u32,
            });
        }
        self.rung_budget(act)?;
        Ok(RungRaise {
            rung,
            from: self.rung(),
            act,
        })
    }

    /// The chunks of idle cache in the span raising the activation to
    /// `rung` takes: what its owner clears before the raise
    /// ([`Self::clear_span`]).
    pub fn span_cached(&self, rung: u32) -> u32 {
        let act = self.act_chunks(rung).min(self.act_max);
        self.in_way(self.total - act).len() as u32
    }

    /// A raise asks the span from physical chunk `from` clear: idle cache
    /// opens and fills nothing there from now until the activation is set
    /// aside again, and the cache's owner lets go what lies there (the
    /// weight cache drops it, [`super::weights::WeightCache::clear_span`]).
    pub fn clear_span(&mut self, from: u32) {
        self.clearing = self.clearing.min(from.max(self.live_limit()));
    }

    /// See [`Self::clear_span`]: `total` while no raise asks.
    #[must_use]
    pub fn clearing(&self) -> u32 {
        self.clearing
    }

    /// Takes `raise`'s rung. All or nothing.
    pub fn apply_rung(&mut self, raise: &RungRaise) -> Result<(), SlotError> {
        let act = self.act_in_span(raise.rung)?;
        let cached = self.in_way(self.total - act);
        if !cached.is_empty() {
            return Err(SlotError::SpanCached {
                rung: raise.rung,
                chunks: cached.len() as u32,
            });
        }
        self.rung_budget(act)?;
        if act != raise.act {
            return Err(SlotError::StaleRung);
        }
        self.set_act(act);
        Ok(())
    }

    /// Takes back a raise applied with [`Self::apply_rung`] and refused by
    /// the engine (its carve): the activation is `act` again, the rung in
    /// hand.
    pub fn undo_rung(&mut self, act: u32) {
        self.set_act(act);
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

    /// Committed chunks live state may open now: neither held nor set
    /// aside for activation, below the ceiling ([`Self::set_ceiling`]) and
    /// below the top rung's span ([`Self::live_limit`]). A committed chunk
    /// the engine binds none of yet is no room, so the walker commits
    /// ahead of a waiter rather than count it, and nor is a chunk of the
    /// span a lowered rung leaves: it holds only idle cache.
    pub fn headroom(&self) -> u32 {
        let limit = self.live_limit().min(self.state_limit()).min(self.ceiling);
        let below: u32 = (0..self.segments.count())
            .filter(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .map(|segment| {
                let range = self.segments.range(segment);
                range.end.min(limit).saturating_sub(range.start)
            })
            .sum::<u32>()
            - self.skipped_in_committed(limit);
        let held_below = self.held - self.held_count(limit, self.total);
        self.headroom_any()
            .min(below.saturating_sub(self.reserved + held_below))
    }

    /// The room idle cache may open: [`Self::headroom`] with the span a
    /// lowered rung leaves counted.
    fn headroom_any(&self) -> u32 {
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
            .sum::<u32>()
            - self.skipped_in_committed(self.ceiling);
        all.min(below.saturating_sub(self.reserved + self.held))
    }

    /// State chunks at physical ids `from..to`, counted.
    fn held_count(&self, from: u32, to: u32) -> u32 {
        (from..to)
            .filter(|&chunk| !self.free.get(chunk) && !self.is_fixed(chunk))
            .count() as u32
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
        (self.is_fixed(chunk) || self.chunks[chunk as usize].shape != NO_SHAPE)
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

    /// Free chunks live state could open now without breaking the budget:
    /// below the top rung's span ([`Self::live_limit`]).
    pub fn free_chunks(&self) -> u32 {
        self.free_chunks_for(true)
    }

    /// Free chunks a holder, `live` or not, could open now. Idle cache may
    /// open a chunk of the top rung's span a lowered rung leaves, short of
    /// what a raise asks cleared ([`Self::clear_span`]); live state none.
    fn free_chunks_for(&self, live: bool) -> u32 {
        let (limit, room) = if live {
            (self.live_limit(), self.headroom())
        } else {
            (self.clearing, self.headroom_any())
        };
        let past = self.open.iter_in(limit, self.total).count() as u32;
        (self.open.count - past).min(room)
    }

    /// Live state opens no chunk at or past this physical id: the base of
    /// the top rung's span, the largest rung's activation, on every pool
    /// and at every rung. A lowered rung leaves the span below its own
    /// activation to idle cache alone (what may go: resident weights above
    /// their floor), so a raise only ever clears cache, and moves no live
    /// state. The span is reserved at load and given back only by yield,
    /// as its last shed under pressure (the rung, after idle prefixes and
    /// the weight cache); a raise the engine then refuses for want of it
    /// waits on the governor's grow (`SharedGrow`), not on a fire.
    #[must_use]
    pub fn live_limit(&self) -> u32 {
        self.total - self.act_max
    }

    /// Free slots `shape` could take now for live state without breaking
    /// the budget.
    pub fn available(&self, shape: ShapeId) -> u64 {
        self.available_for(shape, true)
    }

    /// [`Self::available`] for idle cache: the span a lowered rung leaves
    /// counted ([`Self::free_chunks_for`]).
    pub fn available_cache(&self, shape: ShapeId) -> u64 {
        self.available_for(shape, false)
    }

    fn available_for(&self, shape: ShapeId, live: bool) -> u64 {
        let Some(state) = self.shapes.get(shape as usize) else {
            return 0;
        };
        let chunks = self.free_chunks_for(live);
        self.partial_free_for(shape, live) + u64::from(chunks) * u64::from(state.spc())
    }

    /// Whether every `(shape, slots)` of `needs` could be taken at once by
    /// live state: each shape's partly filled chunks below the top rung's
    /// span first, then free chunks below it, which the shapes share.
    pub fn fits(&self, needs: &[(ShapeId, u64)]) -> bool {
        self.chunks_for(needs)
            .is_some_and(|chunks| chunks <= u64::from(self.free_chunks()))
    }

    /// Free chunks live state could open once every segment below the
    /// activation the facts allow were committed: what the pool can grow
    /// to below the top rung's span, short of the reserved chunks, the
    /// activation and what is held.
    pub fn growable_chunks(&self) -> u32 {
        let limit = self.state_limit().min(self.live_limit());
        let free = (0..limit)
            .filter(|&chunk| {
                self.free.get(chunk)
                    && !matches!(
                        self.segments.state(self.segments.of(chunk)),
                        SegmentState::Draining { .. }
                    )
            })
            .count() as u32;
        let held_below = self.held - self.held_count(limit, self.total);
        free.min(
            (self.total - self.skipped_below(self.total))
                .saturating_sub(self.reserved + self.held + self.act),
        )
        .min((limit - self.skipped_below(limit)).saturating_sub(self.reserved + held_below))
    }

    /// Skipped chunk ids below physical id `limit` ([`Self::skip`]).
    pub fn skipped_below(&self, limit: u32) -> u32 {
        self.skips
            .iter()
            .map(|&(first, end)| end.min(limit).saturating_sub(first))
            .sum()
    }

    /// Skipped chunk ids below `limit` in committed segments.
    fn skipped_in_committed(&self, limit: u32) -> u32 {
        let mut skipped = 0;
        for &(first, end) in &self.skips {
            let end = end.min(limit);
            if first >= end {
                continue;
            }
            for segment in self.segments.of(first)..=self.segments.of(end - 1) {
                if self.segments.state(segment) == SegmentState::Committed {
                    let range = self.segments.range(segment);
                    skipped += range.end.min(end).saturating_sub(range.start.max(first));
                }
            }
        }
        skipped
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
                .saturating_sub(self.partial_free_for(shape, true))
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
        let live = holder.is_live();
        let spc = u64::from(state.spc());
        let partial_free = self.partial_free_for(shape, live);
        let fresh = n.saturating_sub(partial_free).div_ceil(spc);
        let room = u64::from(self.free_chunks_for(live));
        if fresh > room {
            if live && fresh <= u64::from(self.free_chunks_for(false)) {
                return Err(SlotError::LiveInSpan {
                    shape,
                    chunks: (fresh - room) as u32,
                });
            }
            let short = fresh - room;
            return Err(SlotError::Short {
                shape,
                slots: n - partial_free - room * spc,
                chunks: short as u32,
            });
        }
        let mut taken = Vec::with_capacity(n as usize);
        while (taken.len() as u64) < n {
            let chunk = match self.partial_for(shape, live) {
                Some(logical) => logical,
                None => self.open_chunk(shape, live),
            };
            self.fill(chunk, live, n - taken.len() as u64, &mut taken);
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
        let live = holder.is_live();
        let limit = self.fill_limit(live);
        let found = state
            .partial
            .iter()
            .filter(|&&(_, physical, _)| physical < limit)
            .find_map(|&(_, _, logical)| {
                let frame = self.chunks[logical as usize].frame;
                state.free_run(frame, n).map(|first| (logical, first))
            });
        let (chunk, first) = match found {
            Some(found) => found,
            None if self.free_chunks_for(live) > 0 => (self.open_chunk(shape, live), 0),
            None if live && self.free_chunks_for(false) > 0 => {
                return Err(SlotError::LiveInSpan { shape, chunks: 1 });
            }
            None => {
                return Err(SlotError::Short {
                    shape,
                    slots: u64::from(n),
                    chunks: 1,
                });
            }
        };
        let mut taken = Vec::with_capacity(n as usize);
        self.fill_run(chunk, live, first, n, &mut taken);
        *self.holders.entry(holder).or_default() += u64::from(n);
        Ok(taken)
    }

    /// The free runs of `run` adjacent chunks a holder, `live` or not, may
    /// open now, each inside one committed segment, below what it may fill
    /// ([`Self::fill_limit`]): each run's first physical chunk, lowest
    /// first, no two sharing a chunk, and no more than the budget's room
    /// holds ([`Self::headroom`]).
    fn chunk_runs_for(&self, run: u32, live: bool) -> Vec<u32> {
        let run = run.max(1);
        if run > self.segments.segment_chunks() {
            return Vec::new();
        }
        let limit = self.fill_limit(live);
        let room = if live {
            self.headroom()
        } else {
            self.headroom_any()
        };
        let most = (room / run) as usize;
        let mut firsts = Vec::new();
        for segment in 0..self.segments.count() {
            let range = self.segments.range(segment);
            let mut stretch = 0u32;
            for chunk in range.start..range.end.min(limit) {
                if !self.open.get(chunk) {
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

    /// Free runs of `run` adjacent chunks inside one segment live state may
    /// take now ([`Self::acquire_chunk_runs`]), counted disjoint.
    #[must_use]
    pub fn free_chunk_runs(&self, run: u32) -> u32 {
        u32::try_from(self.chunk_runs_for(run, true).len()).unwrap_or(u32::MAX)
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
        let firsts = self.chunk_runs_for(run, live);
        if firsts.len() < pieces as usize {
            return Err(SlotError::NoChunkRun {
                run,
                pieces,
                free_runs: u32::try_from(firsts.len()).unwrap_or(u32::MAX),
                free_chunks: self.free_chunks_for(live),
            });
        }
        let mut taken = Vec::with_capacity((pieces * run * spc) as usize);
        for &first in &firsts[..pieces as usize] {
            let head = self.physical_to_logical[first as usize];
            for physical in first..first + run {
                let chunk = self.open_at(physical, shape);
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
        let range = self.claimable_for(shape, first, n, holder.is_live())?;
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
        self.claimable_for(shape, first, n, true)
    }

    /// [`Self::claimable`] for a holder `live` or not: live state claims
    /// nothing in the top rung's span.
    fn claimable_for(
        &self,
        shape: ShapeId,
        first: u32,
        n: u64,
        live: bool,
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
            if let Some(why) = self.unclaimable(chunk, live) {
                return Err(SlotError::Unclaimable { chunk, why });
            }
        }
        Ok(range)
    }

    /// Why logical chunk `chunk` cannot be claimed now, if it cannot.
    fn unclaimable(&self, chunk: u32, live: bool) -> Option<&'static str> {
        if chunk >= self.total {
            return Some("past the pool");
        }
        if self.is_fixed(chunk) {
            return Some("reserved");
        }
        if self.chunks[chunk as usize].shape != NO_SHAPE {
            return Some("open for a shape already");
        }
        let physical = self.logical_to_physical[chunk as usize];
        if !self.free.get(physical) {
            Some("not free")
        } else if physical >= self.state_limit() {
            Some("in the activation span")
        } else if live && physical >= self.live_limit() {
            Some("in the top rung's span, which holds only idle cache")
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
            debug_assert!(
                !live || self.logical_to_physical[slot.chunk as usize] < self.live_limit(),
                "a live reference on {slot:?} in the top rung's span"
            );
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

    pub fn retire_through(&mut self, epoch: u64) {
        self.retired = self.retired.max(epoch);
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

    /// Opens a free committed chunk for `shape`, for a holder `live` or
    /// not. The caller has checked the room: for live state, room below
    /// the top rung's span, so the lowest free chunk lies below it. Idle
    /// cache takes the lowest free chunk of the span a lowered rung leaves
    /// first, short of what a raise asks cleared, and only then the
    /// lowest below it: the room below the span is live state's.
    fn open_chunk(&mut self, shape: ShapeId, live: bool) -> u32 {
        let in_span = if live {
            None
        } else {
            self.open.iter_in(self.live_limit(), self.clearing).next()
        };
        let physical = in_span
            .or_else(|| self.open.first())
            .expect("room was checked");
        debug_assert!(
            physical < self.fill_limit(live),
            "chunk {physical} is past what a holder live {live} may open"
        );
        self.open_at(physical, shape)
    }

    /// No holder `live` or not opens or fills a chunk at or past this
    /// physical id: live state the top rung's span's base, idle cache what
    /// a raise asks cleared.
    fn fill_limit(&self, live: bool) -> u32 {
        if live {
            self.live_limit()
        } else {
            self.clearing
        }
    }

    /// The partly filled chunk of `shape` a holder, `live` or not, fills
    /// next: the fullest, lowest first, short of [`Self::fill_limit`].
    fn partial_for(&self, shape: ShapeId, live: bool) -> Option<u32> {
        let limit = self.fill_limit(live);
        self.shapes[shape as usize]
            .partial
            .iter()
            .find(|&&(_, physical, _)| physical < limit)
            .map(|&(_, _, logical)| logical)
    }

    /// Free slots of `shape` in partly filled chunks a holder, `live` or
    /// not, may fill.
    fn partial_free_for(&self, shape: ShapeId, live: bool) -> u64 {
        let state = &self.shapes[shape as usize];
        let limit = self.fill_limit(live);
        let past: u64 = state
            .partial
            .iter()
            .filter(|&&(_, physical, _)| physical >= limit)
            .map(|&(_, _, logical)| u64::from(state.spc() - self.chunks[logical as usize].used))
            .sum();
        state.partial_free - past
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
        self.runs.remove(&logical);
        self.chunks[logical as usize] = FREE_CHUNK;
        self.free.set(physical);
        self.segments.let_go(physical);
        self.held -= 1;
        self.reopen(physical);
    }

    // ---- the span's yield shed and grow ----

    /// Span segments committed, counted from the top of the pool: the `n`
    /// of `commit_span(n)` and `release_span(n)`. Every one of them from
    /// the load until yield gives some back.
    pub fn span_committed(&self) -> u32 {
        (self.top_segment..self.segments.count())
            .rev()
            .take_while(|&segment| self.segments.state(segment) == SegmentState::Committed)
            .count() as u32
    }

    /// The span's segments, the top rung's span's whole.
    pub fn span_segments(&self) -> u32 {
        self.segments.count() - self.top_segment
    }

    /// The span segment counted `n`th from the top (`n` from 1).
    fn span_segment(&self, n: u32) -> u32 {
        self.segments.count() - n
    }

    /// The `n` of the governor's grow's `commit_span(n)`: the whole span,
    /// where yield gave some of it back and the engine binds it (no
    /// ceiling held below it). `None` where the span is whole.
    pub fn span_to_regrow(&self) -> Option<u32> {
        let whole = self.span_segments();
        (self.span_committed() < whole && self.ceiling >= self.total).then_some(whole)
    }

    /// The engine backed the top `n` span segments again
    /// (`Engine::commit_span(n)`, the governor's grow): their chunks open
    /// as before yield gave them back.
    pub fn commit_span(&mut self, n: u32) {
        let n = n.min(self.span_segments());
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

    /// The `n` of yield's last shed, `release_span(n)`: the lowest span
    /// segments committed that the activation set aside does not reach
    /// and that hold no chunk (the span a lowered rung leaves to idle
    /// cache, the cache gone), given back from the bottom up. `None` where
    /// none is.
    pub fn span_to_shed(&self) -> Option<u32> {
        let committed = self.span_committed();
        let limit = self.state_limit();
        let mut keep = committed;
        while keep > 0 {
            let segment = self.span_segment(keep);
            let range = self.segments.range(segment);
            let idle = range.end <= limit && self.segments.used(segment) == 0;
            if !idle {
                break;
            }
            keep -= 1;
        }
        (keep < committed).then_some(keep)
    }

    /// Gives back the span segments below the top `n` committed
    /// ([`Self::span_to_shed`] says which may go), ahead of the engine's
    /// `release_span(n)`.
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
        if range.clone().any(|chunk| self.is_fixed(chunk)) {
            return None;
        }
        let needed = self.reserved + self.held + self.act;
        if needed > self.committed_chunks() - self.segments.len(segment) {
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
            .filter(|&segment| segment < self.top_segment && self.backs(segment))
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
    pub(crate) fn check(&self) {
        let mut held = 0;
        let mut seg_used = vec![0u32; self.segments.count() as usize];
        for logical in 0..self.total {
            let physical = self.logical_to_physical[logical as usize];
            let meta = self.chunks[logical as usize];
            assert_eq!(self.physical_to_logical[physical as usize], logical);
            if meta.shape == NO_SHAPE && !self.is_fixed(logical) {
                assert!(self.free.get(physical), "a free logical id on a held chunk");
                continue;
            }
            assert!(!self.free.get(physical));
            assert!(!self.open.get(physical));
            seg_used[self.segments.of(physical) as usize] += 1;
            if self.is_fixed(logical) {
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
            for chunk in self.segments.range(segment) {
                assert_eq!(self.open.get(chunk), self.openable(chunk), "chunk {chunk}");
                if !self.free.get(chunk) && !self.is_fixed(chunk) {
                    assert!(chunk < self.state_limit(), "state in the activation span");
                    assert!(
                        chunk < self.live_limit() || !self.holds_live(chunk),
                        "live state on chunk {chunk}, at or past the top rung's span's base {}",
                        self.live_limit()
                    );
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
            act_span: None,
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
                    let available = if holder.is_live() {
                        pool.available(shape)
                    } else {
                        pool.available_cache(shape)
                    };
                    match pool.acquire(holder, shape, n) {
                        Ok(slots) => {
                            assert!(n <= available);
                            for &slot in &slots {
                                assert!(
                                    !holder.is_live()
                                        || pool.physical(slot.chunk).unwrap() < pool.live_limit(),
                                    "a live open at or past the top rung's span's base"
                                );
                            }
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
                        Err(SlotError::LiveInSpan { .. }) => {
                            assert!(holder.is_live() && n > available);
                        }
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
                    // Live state takes no reference in the top rung's span:
                    // only cache lies there.
                    if to.is_live() {
                        slots.retain(|slot| pool.physical(slot.chunk).unwrap() < pool.live_limit());
                    }
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
                    // Now and then the guests' standing bytes change: no
                    // rung whose activation they carry past the span is
                    // set aside.
                    if rng.below(4) == 0 {
                        pool.set_standing(rng.below(4) << 20);
                    }
                    let rung = [0, 200, 1000, 5000][rng.below(4) as usize];
                    if pool.set_rung(rung).is_err()
                        && let Err(SlotError::SpanCached { .. }) = pool.plan_rung(rung)
                    {
                        // A raise over idle cache in the span: its owners
                        // let it go (the weight cache drops it), nothing
                        // there live; it lands once the fires reading it
                        // retire.
                        let from = pool.total_chunks() - pool.act_chunks(rung);
                        pool.clear_span(from);
                        let mut gone: Vec<(Holder, Vec<Slot>)> = Vec::new();
                        for (holder, held) in &mut model.held {
                            if holder.is_live() {
                                continue;
                            }
                            let (out, kept): (Vec<Slot>, Vec<Slot>) = held
                                .iter()
                                .partition(|slot| pool.physical(slot.chunk).unwrap() >= from);
                            *held = kept;
                            gone.push((*holder, out));
                        }
                        for (holder, out) in gone {
                            pool.release(holder, &out, epoch);
                            for slot in out {
                                if model.refs(slot) == (0, 0) {
                                    let after =
                                        epoch.max(model.touched.get(&slot).copied().unwrap_or(0));
                                    model.pending.insert(slot, after);
                                }
                            }
                        }
                        pool.retire_through(epoch);
                        model.retired = model.retired.max(epoch);
                        assert_eq!(pool.span_cached(rung), 0, "the raise's span is clear");
                    }
                    if pool.set_rung(rung).is_err()
                        && let Ok(raise) = pool.plan_rung(rung)
                    {
                        let refs: Vec<(Slot, u32)> = model
                            .held
                            .values()
                            .flatten()
                            .map(|&slot| (slot, pool.slot_ref(slot)))
                            .collect();
                        // No live state lies in the top rung's span and no
                        // activation reaches past it: a raise moves nothing.
                        let table = pool.chunk_table().to_vec();
                        pool.apply_rung(&raise).unwrap();
                        assert_eq!(pool.chunk_table(), table.as_slice(), "a raise moved state");
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

    /// A pool of 96 chunks, every one committed, lowered to the smallest
    /// rung: the top rung's span is 90..96, the smallest rung's activation
    /// 95..96, so 90..95 is the span the lowered rung leaves. Live state
    /// holds every chunk below the span.
    fn lowered_and_full() -> (SlotPool, Vec<Slot>) {
        let mut pool = SlotPool::new(facts(96, 16, 6, &[(ShapeKind::List, 4)]));
        pool.set_rung(16).unwrap();
        assert_eq!(pool.live_limit(), 90, "the top rung's span's base");
        assert_eq!(pool.state_limit(), 95, "the smallest rung's activation");
        let room = pool.available(0);
        assert_eq!(room, 89 * 4, "below the span only: chunks 1..90");
        let live = pool.acquire(Holder::WorkingSet(1), 0, room).unwrap();
        assert!(
            live.iter()
                .all(|slot| pool.physical(slot.chunk).unwrap() < 90)
        );
        (pool, live)
    }

    /// The ruled rung shape: a committed span chunk a lowered rung leaves
    /// refuses a live open, by name, and takes a cache one; live state
    /// takes no claim there either.
    #[test]
    fn a_committed_span_chunk_refuses_a_live_open_and_takes_a_cache_one() {
        let (mut pool, live) = lowered_and_full();
        assert_eq!(pool.available(0), 0, "no live room: the span is cache's");
        assert_eq!(pool.available_cache(0), 5 * 4, "90..95 for cache");
        assert!(!pool.fits(&[(0, 1)]));
        assert_eq!(pool.growable_chunks(), 0);
        let refused = pool.acquire(Holder::WorkingSet(2), 0, 1).unwrap_err();
        assert_eq!(
            refused,
            SlotError::LiveInSpan {
                shape: 0,
                chunks: 1
            }
        );
        assert!(
            refused
                .to_string()
                .contains("the top rung's span holds only idle cache"),
            "{refused}"
        );
        assert_eq!(
            pool.acquire_run(Holder::Snapshot(1), 0, 2).unwrap_err(),
            SlotError::LiveInSpan {
                shape: 0,
                chunks: 1
            }
        );
        let cached = pool.acquire(cache(1), 0, 6).unwrap();
        let chunks: BTreeSet<u32> = cached
            .iter()
            .map(|slot| pool.physical(slot.chunk).unwrap())
            .collect();
        assert_eq!(chunks, BTreeSet::from([90, 91]), "cache opens in the span");
        // A live claim of a span chunk is refused; the free part of a
        // cache chunk there is no live room.
        let free = pool.physical_to_logical[92];
        assert_eq!(
            pool.claim(Holder::WorkingSet(3), 0, free * 4, 4)
                .unwrap_err(),
            SlotError::Unclaimable {
                chunk: free,
                why: "in the top rung's span, which holds only idle cache",
            }
        );
        assert!(pool.acquire(Holder::WorkingSet(2), 0, 1).is_err());
        pool.check();
        pool.release(cache(1), &cached, 0);
        pool.release(Holder::WorkingSet(1), &live, 0);
        pool.check();
    }

    /// A raise over cached span chunks moves nothing: it is refused while
    /// cache lies in its span (nothing to fund, no live state to move),
    /// nothing more opens there for cache once it asks, and once the
    /// cache's owner lets it go and the fires reading it retire the rung is
    /// taken with no move.
    #[test]
    fn a_raise_over_cached_span_chunks_clears_them_and_moves_no_live_state() {
        let (mut pool, live) = lowered_and_full();
        let cached = pool.acquire(cache(1), 0, 10).unwrap();
        assert_eq!(pool.span_cached(4096), 3, "chunks 90..93 hold cache");
        assert_eq!(
            pool.plan_rung(4096).unwrap_err(),
            SlotError::SpanCached {
                rung: 4096,
                chunks: 3
            }
        );
        let from = pool.total_chunks() - pool.act_chunks(4096);
        pool.clear_span(from);
        assert_eq!(pool.clearing(), 90);
        assert_eq!(
            pool.available_cache(0),
            0,
            "nothing opens there for cache now"
        );
        assert!(pool.acquire(cache(2), 0, 1).is_err());
        // The owner lets it go behind the fire at epoch 3.
        pool.release(cache(1), &cached, 3);
        assert!(pool.plan_rung(4096).is_err(), "not before the fire retires");
        pool.retire_through(3);
        let raise = pool.plan_rung(4096).unwrap();
        let table = pool.chunk_table().to_vec();
        pool.apply_rung(&raise).unwrap();
        assert_eq!(pool.chunk_table(), table.as_slice(), "nothing moved");
        assert_eq!(pool.clearing(), 96, "the ask is taken");
        assert_eq!(pool.state_limit(), 90);
        pool.check();
        // Lowered again, the span is cache's again.
        pool.set_rung(16).unwrap();
        assert_eq!(pool.available(0), 0);
        assert_eq!(pool.available_cache(0), 5 * 4);
        pool.release(Holder::WorkingSet(1), &live, 0);
    }

    /// No rung whose activation, its standing bytes included, reaches past
    /// the top rung's span is set aside: neither raised to nor lowered to,
    /// so no raise ever has live state in its way (fires are cut to the
    /// widest rung that fits instead, `SharedPool::widest_fit`).
    #[test]
    fn a_rung_whose_activation_reaches_past_the_span_is_never_set_aside() {
        let mut pool = SlotPool::new(facts(96, 16, 6, &[(ShapeKind::List, 4)]));
        pool.set_rung(16).unwrap();
        let live = pool.acquire(Holder::WorkingSet(1), 0, 89 * 4).unwrap();
        // Two chunks' standing guest bytes at every rung: the top rung's
        // activation would reach 88, past the span's base at 90.
        pool.set_standing(2 << 20);
        assert_eq!(pool.act_chunks(4096), 8);
        let past = SlotError::PastSpan {
            rung: 4096,
            chunks: 8,
            span: 6,
        };
        assert_eq!(pool.plan_rung(4096).unwrap_err(), past);
        assert_eq!(pool.set_rung(4096).unwrap_err(), past);
        // The rung under it fits with the term: taken, nothing moved.
        assert_eq!(pool.act_chunks(256), 5);
        let table = pool.chunk_table().to_vec();
        let raise = pool.plan_rung(256).unwrap();
        pool.apply_rung(&raise).unwrap();
        assert_eq!(pool.chunk_table(), table.as_slice());
        assert_eq!(pool.state_limit(), 91);
        pool.check();
        pool.release(Holder::WorkingSet(1), &live, 0);
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

    /// Runs of adjacent chunks are taken inside one segment, lowest first,
    /// never across a segment's end; a run past a segment is refused by
    /// name, and too few free runs are refused naming what was free. An
    /// evacuation moves a run whole onto adjacent free chunks of one lower
    /// segment, its chunks in order.
    #[test]
    fn chunk_runs_lie_inside_a_segment_and_move_whole() {
        let fixed = [(ShapeKind::Fixed, 4)];
        // Four segments of 4 chunks, the top one under the activation.
        let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
        let holder = Holder::WorkingSet(1);
        let physical = |pool: &SlotPool, slots: &[Slot]| -> Vec<u32> {
            let mut chunks: Vec<u32> = slots
                .iter()
                .map(|slot| pool.physical(slot.chunk).unwrap())
                .collect();
            chunks.dedup();
            chunks
        };
        // Segment 0 holds the null chunk: its run of 3 is 1..4.
        assert_eq!(pool.free_chunk_runs(3), 3);
        let first = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
        assert_eq!(physical(&pool, &first), [1, 2, 3]);
        assert_eq!(first.len(), 12, "every slot of each chunk");
        // A run of 2 then lies in segment 1, never across 3|4.
        let second = pool.acquire_chunk_runs(holder, 0, 2, 2).unwrap();
        assert_eq!(physical(&pool, &second), [4, 5, 6, 7]);
        assert_eq!(
            pool.acquire_chunk_runs(holder, 0, 5, 1),
            Err(SlotError::RunPastSegment { run: 5, segment: 4 })
        );
        assert_eq!(
            pool.acquire_chunk_runs(holder, 0, 3, 2),
            Err(SlotError::NoChunkRun {
                run: 3,
                pieces: 2,
                free_runs: 1,
                free_chunks: 4,
            })
        );
        // Segment 0 empties but for the null chunk; segment 2's run of 3
        // moves whole into 1..4.
        let third = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
        assert_eq!(physical(&pool, &third), [8, 9, 10]);
        pool.release(holder, &first, 0);
        let evacuation = pool.plan_evacuation(2).expect("the run fits segment 0");
        let moves: Vec<(u32, u32)> = evacuation
            .moves
            .iter()
            .map(|step| (step.from, step.to))
            .collect();
        assert_eq!(moves, [(8, 1), (9, 2), (10, 3)]);
        pool.apply_evacuation(&evacuation, 1).unwrap();
        assert_eq!(physical(&pool, &third), [1, 2, 3], "adjacent, in order");
        // Where no lower segment has the run free, the segment stays.
        let mut pool = SlotPool::new(facts_with(16, 4, 4, &fixed, &[(16, 4)]));
        let run = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
        let singles = pool.acquire_chunk_runs(holder, 0, 1, 2).unwrap();
        assert_eq!(physical(&pool, &singles), [4, 5]);
        let far = pool.acquire_chunk_runs(holder, 0, 3, 1).unwrap();
        assert_eq!(physical(&pool, &far), [8, 9, 10]);
        pool.release(holder, &run, 0);
        // Segment 0 has 1..4 free again: segment 2's run fits there.
        assert!(pool.plan_evacuation(2).is_some());
        let taken = pool.acquire_chunk_runs(holder, 0, 1, 1).unwrap();
        assert_eq!(physical(&pool, &taken), [1]);
        assert!(
            pool.plan_evacuation(2).is_none(),
            "2..4 and 6..8 free, four chunks, and no run of 3 in a lower segment"
        );
    }

    /// Skipped ids mid-segment, with ids above them: ids 6..10 of 20 in
    /// segments of 4 (from the middle of segment 1 to the middle of segment
    /// 2), the top segment under the activation. No skipped id is handed
    /// out, counted in the budget or moved; a run of adjacent chunks lies
    /// up to the skip's edge or past it, never across it; and every id
    /// above the skip is handed out as itself.
    #[test]
    fn skipped_ids_mid_segment_stay_gaps_no_run_crosses() {
        let fixed = [(ShapeKind::Fixed, 4)];
        let mut pool = SlotPool::new(facts_with(20, 4, 5, &fixed, &[(16, 4)]));
        let committed = pool.committed_chunks();
        pool.skip(6, 4);
        assert_eq!(pool.committed_chunks(), committed - 4, "nothing backs them");
        pool.check();
        let holder = Holder::WorkingSet(1);
        let physical = |pool: &SlotPool, slots: &[Slot]| -> Vec<u32> {
            let mut chunks: Vec<u32> = slots
                .iter()
                .map(|slot| pool.physical(slot.chunk).unwrap())
                .collect();
            chunks.dedup();
            chunks
        };
        // Runs of 2: 1..3 (segment 0 holds the null chunk), 4..6 up to the
        // skip's edge, then 10..12, 12..14 and 14..16 past it; 5|10 is no
        // run.
        assert_eq!(pool.free_chunk_runs(2), 5);
        let runs = pool.acquire_chunk_runs(holder, 0, 2, 4).unwrap();
        assert_eq!(physical(&pool, &runs), [1, 2, 4, 5, 10, 11, 12, 13]);
        for slot in &runs {
            assert_eq!(
                pool.physical(slot.chunk),
                Some(slot.chunk),
                "ids keep their place"
            );
        }
        assert_eq!(pool.free_chunk_runs(2), 1, "14..16");
        pool.check();
        // Singles take every free chunk but the skipped ones.
        let singles = pool.acquire_chunk_runs(holder, 0, 1, 3).unwrap();
        assert_eq!(physical(&pool, &singles), [3, 14, 15]);
        assert!(pool.acquire_chunk_runs(holder, 0, 1, 1).is_err());
        // A segment that holds skipped ids is never evacuated.
        pool.release(holder, &singles, 0);
        assert!(pool.plan_evacuation(1).is_none());
        assert!(pool.plan_evacuation(2).is_none());
        pool.check();
    }
}
