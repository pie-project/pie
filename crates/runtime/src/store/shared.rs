//! The runtime on one shared chunk pool: what an engine that states real
//! (unpinned) [`ChunkFacts`] runs on.
//!
//! One [`SlotPool`] holds every slot shape the engine states, kv list, window
//! lists and fixed alike, with the streamable weights' shapes after them; a
//! chunk holds slots of one shape while it is open, and any shape opens any
//! free chunk. The stores draw from it through
//! [`super::chunks::ShapePool::shared`]:
//!
//! - the kv store's pages are slot refs of the list shape;
//! - a working set's window pages (CONTRACT-BC 19, 21) are slot refs of
//!   each window list (one a distinct window, `T_w` tokens a slot of its
//!   own), refcounted by the rs store and dropped list by list once a slot
//!   lies wholly behind that list's window; a lane's table of each window
//!   list is in its own token order, ref 0 (the null chunk, never handed
//!   out) where it holds none;
//! - the rs store's state slots are **records**: one working set's fixed
//!   state, a slot of every fixed shape for each of its pieces;
//! - the rs store's buffer pages are slot refs of the **rs list** (a list
//!   shape of role [`SlotRole::RsBuffer`], CONTRACT-BC 18): taken as
//!   unfolded rows accumulate, refcounted by the store so forks share their
//!   common unfolded prefix, and given back as rows fold.
//!
//! A lane names its slots in [`::engine::fire::Lane::tables`] (kv list and
//! window list refs in token order) and [`::engine::fire::Lane::fixed`] (one `[run, cow]`
//! a piece). `cow == run` writes the record in place; a lane whose record is
//! shared (a fork, or a snapshot restored, in that fire) writes a fresh
//! record the engine copies `run` onto first. The copy-on-write records come
//! out of a reserve of one set per lane a fire runs, which the planner keeps
//! back from admission (`ResidencyPlanner::cow_lanes`).
//!
//! Position (CONTRACT-BC item 2): state lies below `total - act_chunks(rung)`;
//! raising the rung first moves state out of the new activation span
//! ([`SharedPool::raise_rung`]), the engine copying it by
//! [`::engine::Engine::evacuate`] before the next fire. The top
//! `act_chunks_max` span is committed from the load and counts as held from
//! then on. Segments commit as a prefix (`commit_segments(n)`) ahead of need,
//! off the fire path; a shrink evacuates the highest committed segment of
//! the prefix (the only one `release_segments` can give back without the
//! ones above it) and releases it once the fire carrying the copy retires.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

use ::engine::chunk::{
    ChunkFacts, Evacuation as EngineEvacuation, Growth as SegmentGrowth, ShapeId as EngineShape,
    SlotKind, SlotRole,
};

use super::chunks::STORE;
use super::segments::{Evacuation, SegmentState};
use super::slots::{
    Demand, Holder, RungEvacuation, ShapeId, ShapeKind, Slot, SlotError, SlotPool, SlotPoolFacts,
    SlotShape,
};
use crate::governor::{Demand as Asked, Growth, Retarget};

/// The pool behind an engine's stores, shared by every store and the
/// governor; each call takes the lock for itself, innermost of every store
/// lock.
pub type Shared = Arc<Mutex<SharedPool>>;

/// What a pool keeps committed ahead of what it has handed out, in bytes.
pub const COMMIT_AHEAD_BYTES: u64 = super::grow::COMMIT_AHEAD_BYTES;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SharedError {
    #[error("chunk facts: {0}")]
    Facts(String),
    #[error("shape {0} is pinned; a shared pool takes unpinned shapes only")]
    Pinned(u32),
    #[error("the reserved chunks are not one run from chunk 0")]
    Reserved,
    #[error(
        "{kind} shapes {shapes:?}: the stores hold one kv list, one rs list and one window list a distinct window"
    )]
    Shapes {
        kind: &'static str,
        shapes: Vec<u32>,
    },
}

/// A per-token shape the stores take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenShape {
    pub id: EngineShape,
    pub pool: ShapeId,
    pub tokens: u32,
    /// The rows a window list retains; 0 for the kv list and the rs list.
    pub window: u32,
    pub slot_bytes: u64,
    pub slots_per_chunk: u32,
}

/// A fixed shape: `pieces` slots of it a record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedShape {
    pub id: EngineShape,
    pub pool: ShapeId,
    pub pieces: u32,
    pub slot_bytes: u64,
}

/// One step toward a segment target, as [`SharedPool::next_move`] plans it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentMove {
    /// Back segments `0..prefix` (`commit_segments(prefix)`); the pool hands
    /// out their chunks once the engine has, [`SharedPool::committed`].
    Commit { prefix: u32 },
    /// Copy `evacuation` (applied in the pool already), ahead of the next
    /// fire; [`SharedPool::moved`] once the engine has it.
    Evacuate { evacuation: EngineEvacuation },
    /// Give back segments `prefix..` (`release_segments(prefix)`), released
    /// in the pool already.
    Release { prefix: u32 },
    /// The highest committed segment cannot be emptied into the rest.
    NoRoom,
    /// A drained segment waits for the fire carrying its copy.
    Wait,
}

/// The engine side of the pool: what a segment move and a rung raise ask of
/// it. Every engine answers; one that does not grow answers unsupported.
pub trait SegmentPort {
    fn commit_segments(&mut self, segments: u32) -> ::engine::Result<()>;
    fn release_segments(&mut self, segments: u32) -> ::engine::Result<()>;
    fn evacuate(&mut self, evacuation: &EngineEvacuation) -> ::engine::Result<()>;
}

impl<E: ::engine::Engine + ?Sized> SegmentPort for E {
    fn commit_segments(&mut self, segments: u32) -> ::engine::Result<()> {
        ::engine::Engine::commit_segments(self, segments)
    }

    fn release_segments(&mut self, segments: u32) -> ::engine::Result<()> {
        ::engine::Engine::release_segments(self, segments)
    }

    fn evacuate(&mut self, evacuation: &EngineEvacuation) -> ::engine::Result<()> {
        ::engine::Engine::evacuate(self, evacuation)
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct SharedPool {
    pool: SlotPool,
    growth: SegmentGrowth,
    list: Option<TokenShape>,
    /// The rs buffer's list shape (CONTRACT-BC 18).
    rs_list: Option<TokenShape>,
    /// The window lists, narrowest window first (CONTRACT-BC 19).
    windows: Vec<TokenShape>,
    fixed: Vec<FixedShape>,
    /// The first weight shape's pool id; the weight shapes follow.
    weights_first: ShapeId,
    /// Record id -> its slots, each fixed shape's pieces in order.
    records: Vec<Option<Vec<Vec<Slot>>>>,
    free_records: BTreeSet<u32>,
    records_capacity: u32,
    cow_lanes: u32,
    /// Chunks one working set takes at the longest context.
    least_chunks: u32,
}

/// The engine's moves expressed as `(logical, new physical)`.
fn engine_moves(moves: &[super::segments::ChunkMove]) -> EngineEvacuation {
    EngineEvacuation {
        moves: moves.iter().map(|step| (step.logical, step.to)).collect(),
    }
}

impl SharedPool {
    /// Whether `facts` state a shared pool: shapes, none of them pinned.
    #[must_use]
    pub fn states_shared(facts: &ChunkFacts) -> bool {
        !facts.shapes.is_empty() && facts.shapes.iter().all(|shape| shape.pinned.is_none())
    }

    /// The pool `facts` state, a copy-on-write reserve of `cow_lanes` records
    /// for the lanes one fire runs, and `max_context` tokens the longest a
    /// working set holds. The rung starts at the largest: state never takes
    /// the top span until a rung is lowered.
    pub fn new(facts: &ChunkFacts, cow_lanes: u32, max_context: u32) -> Result<Self, SharedError> {
        facts
            .validate()
            .map_err(|error| SharedError::Facts(error.to_string()))?;
        if let Some(shape) = facts.shapes.iter().find(|shape| shape.pinned.is_some()) {
            return Err(SharedError::Pinned(shape.id.0));
        }
        let mut reserved = 0u32;
        for run in &facts.reserved {
            if run.first != reserved {
                return Err(SharedError::Reserved);
            }
            reserved += run.chunks;
        }
        let mut shapes = Vec::with_capacity(facts.shapes.len() + facts.weights.len());
        let (mut lists, mut rs_lists, mut windows, mut fixed) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for shape in &facts.shapes {
            let pool = ShapeId::try_from(shapes.len()).expect("shapes fit a shape id");
            let kind = match shape.kind {
                SlotKind::List { tokens, window, .. } => {
                    let found = match window {
                        Some(_) => &mut windows,
                        None if shape.role == SlotRole::RsBuffer => &mut rs_lists,
                        None => &mut lists,
                    };
                    found.push(TokenShape {
                        id: shape.id,
                        pool,
                        tokens,
                        window: window.unwrap_or(0),
                        slot_bytes: shape.slot_bytes,
                        slots_per_chunk: shape.slots_per_chunk,
                    });
                    ShapeKind::List
                }
                SlotKind::Fixed { pieces } => {
                    fixed.push(FixedShape {
                        id: shape.id,
                        pool,
                        pieces,
                        slot_bytes: shape.slot_bytes,
                    });
                    ShapeKind::Fixed
                }
            };
            shapes.push(SlotShape {
                kind,
                slot_bytes: shape.slot_bytes,
                slots_per_chunk: shape.slots_per_chunk,
            });
        }
        for (kind, found) in [("kv list", &lists), ("rs list", &rs_lists)] {
            if found.len() > 1 {
                return Err(SharedError::Shapes {
                    kind,
                    shapes: found.iter().map(|shape| shape.id.0).collect(),
                });
            }
        }
        windows.sort_by_key(|shape: &TokenShape| shape.window);
        if windows.len() > crate::store::rs::MAX_WINDOW_LISTS {
            return Err(SharedError::Shapes {
                kind: "window list",
                shapes: windows.iter().map(|shape| shape.id.0).collect(),
            });
        }
        if let Some(pair) = windows.windows(2).find(|w| w[0].window == w[1].window) {
            return Err(SharedError::Shapes {
                kind: "window list",
                shapes: pair.iter().map(|shape| shape.id.0).collect(),
            });
        }
        let weights_first = ShapeId::try_from(shapes.len()).expect("shapes fit a shape id");
        shapes.extend(super::weights::pool_shapes(&facts.weights));
        let mut pool = SlotPool::new(SlotPoolFacts {
            chunk_bytes: facts.chunk_bytes,
            total_chunks: facts.total_chunks,
            segment_chunks: facts.segment_chunks,
            committed_segments: facts.segments.committed,
            reserved_chunks: reserved,
            shapes,
            act_chunks: facts.act_chunks.clone(),
        });
        if let Some(&(rung, _)) = facts.act_chunks.last() {
            pool.set_rung(rung)
                .expect("an empty pool takes its largest rung");
        }
        let mut shared = Self {
            pool,
            growth: facts.segments.growth,
            list: lists.first().copied(),
            rs_list: rs_lists.first().copied(),
            windows,
            fixed,
            weights_first,
            records: Vec::new(),
            free_records: BTreeSet::new(),
            records_capacity: 0,
            cow_lanes,
            least_chunks: 0,
        };
        // Under the smallest rung state may take every chunk but the
        // reserved ones and that rung's activation.
        let state_chunks = facts
            .total_chunks
            .saturating_sub(reserved)
            .saturating_sub(facts.act_chunks.first().map_or(0, |&(_, chunks)| chunks));
        shared.records_capacity = shared.records_fitting(state_chunks);
        shared.least_chunks = shared.working_set_chunks(max_context);
        tracing::debug!(
            chunk_bytes = facts.chunk_bytes,
            total_chunks = facts.total_chunks,
            segment_chunks = facts.segment_chunks,
            committed = facts.segments.committed,
            shapes = facts.shapes.len(),
            records = shared.records_capacity,
            "a shared pool"
        );
        Ok(shared)
    }

    #[must_use]
    pub fn pool(&self) -> &SlotPool {
        &self.pool
    }

    pub fn pool_mut(&mut self) -> &mut SlotPool {
        &mut self.pool
    }

    #[must_use]
    pub fn list(&self) -> Option<TokenShape> {
        self.list
    }

    /// The rs buffer's list shape, where the model buffers rs rows.
    #[must_use]
    pub fn rs_list(&self) -> Option<TokenShape> {
        self.rs_list
    }

    /// The window lists, narrowest window first.
    #[must_use]
    pub fn windows(&self) -> &[TokenShape] {
        &self.windows
    }

    #[must_use]
    pub fn fixed(&self) -> &[FixedShape] {
        &self.fixed
    }

    #[must_use]
    pub fn weights_first(&self) -> ShapeId {
        self.weights_first
    }

    /// Copy-on-write records kept back for the lanes of one fire.
    #[must_use]
    pub fn cow_reserve(&self) -> u32 {
        if self.fixed.is_empty() {
            0
        } else {
            self.cow_lanes
        }
    }

    /// The chunks `n` records take from empty chunks.
    fn record_chunks(&self, n: u64) -> u64 {
        self.fixed
            .iter()
            .map(|shape| {
                let spc = u64::from(self.pool.shape(shape.pool).map_or(1, |s| s.slots_per_chunk));
                (n * u64::from(shape.pieces)).div_ceil(spc.max(1))
            })
            .sum()
    }

    /// The most records `chunks` empty chunks hold.
    fn records_fitting(&self, chunks: u32) -> u32 {
        if self.fixed.is_empty() {
            return 0;
        }
        let (mut low, mut high) = (0u64, u64::from(u32::MAX));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if self.record_chunks(mid) <= u64::from(chunks) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        u32::try_from(low).unwrap_or(u32::MAX)
    }

    /// Chunks one working set takes at `context` tokens: its list slots, a
    /// slots of each window list its window covers and the one its oldest
    /// row retires from, its record, and an rs list slot for the rows a fire
    /// leaves unfolded.
    fn working_set_chunks(&self, context: u32) -> u32 {
        let per_chunk = |slots: u64, spc: u32| slots.div_ceil(u64::from(spc.max(1)));
        let list = self.list.map_or(0, |list| {
            per_chunk(
                u64::from(context.div_ceil(list.tokens.max(1))),
                list.slots_per_chunk,
            )
        });
        let windows: u64 = self
            .windows
            .iter()
            .map(|list| {
                let tokens = list.tokens.max(1);
                let slots = context
                    .div_ceil(tokens)
                    .min(list.window.div_ceil(tokens) + 1);
                per_chunk(u64::from(slots), list.slots_per_chunk)
            })
            .sum();
        let rs = u64::from(self.rs_list.is_some());
        u32::try_from(list + windows + rs + self.record_chunks(1)).unwrap_or(u32::MAX)
    }

    // ---- slot refs of the list shapes ----

    /// The slot slot ref `slot_ref` of pool shape `shape` names.
    #[must_use]
    pub fn slot(&self, shape: ShapeId, slot_ref: u32) -> Slot {
        let spc = self
            .pool
            .shape(shape)
            .map_or(1, |s| s.slots_per_chunk.max(1));
        Slot {
            chunk: slot_ref / spc,
            index: slot_ref % spc,
        }
    }

    /// `n` fresh slots of `shape` for the store, as slot refs; none when
    /// they do not all fit.
    pub fn take_refs(&mut self, shape: ShapeId, n: u64) -> Option<Vec<u32>> {
        let slots = self.pool.acquire(STORE, shape, n).ok()?;
        Some(slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect())
    }

    /// The store lets `refs` of `shape` go; no fire reads them any more.
    pub fn give_refs(&mut self, shape: ShapeId, refs: &[u32]) {
        let slots: Vec<Slot> = refs.iter().map(|&r| self.slot(shape, r)).collect();
        self.pool.release(STORE, &slots, 0);
    }

    // ---- records ----

    /// Records the pool's state chunks hold at most.
    #[must_use]
    pub fn records_capacity(&self) -> u32 {
        self.records_capacity
    }

    /// Records that fit now, each fixed shape's pieces from its partly
    /// filled chunks first.
    #[must_use]
    pub fn records_available(&self) -> u32 {
        if self.fixed.is_empty() {
            return 0;
        }
        let fits = |n: u64| {
            let needs: Vec<(ShapeId, u64)> = self
                .fixed
                .iter()
                .map(|shape| (shape.pool, n * u64::from(shape.pieces)))
                .collect();
            self.pool.fits(&needs)
        };
        let (mut low, mut high) = (0u64, u64::from(self.records_capacity));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if fits(mid) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let free = self.records_capacity as usize - self.live_records();
        (low as u32).min(free as u32)
    }

    fn live_records(&self) -> usize {
        self.records.len() - self.free_records.len()
    }

    /// `n` fresh records, ids lowest first; none when they do not all fit.
    pub fn take_records(&mut self, n: usize) -> Option<Vec<u32>> {
        if n == 0 {
            return Some(Vec::new());
        }
        if self.live_records() + n > self.records_capacity as usize {
            return None;
        }
        let needs: Vec<(ShapeId, u64)> = self
            .fixed
            .iter()
            .map(|shape| (shape.pool, n as u64 * u64::from(shape.pieces)))
            .collect();
        if !self.pool.fits(&needs) {
            return None;
        }
        let mut taken: Vec<Vec<Slot>> = Vec::with_capacity(self.fixed.len());
        for &(shape, slots) in &needs {
            taken.push(
                self.pool
                    .acquire(STORE, shape, slots)
                    .expect("the records were checked to fit"),
            );
        }
        let mut ids = Vec::with_capacity(n);
        for at in 0..n {
            let record: Vec<Vec<Slot>> = self
                .fixed
                .iter()
                .zip(&taken)
                .map(|(shape, slots)| {
                    let pieces = shape.pieces as usize;
                    slots[at * pieces..(at + 1) * pieces].to_vec()
                })
                .collect();
            let id = match self.free_records.pop_first() {
                Some(id) => {
                    self.records[id as usize] = Some(record);
                    id
                }
                None => {
                    self.records.push(Some(record));
                    (self.records.len() - 1) as u32
                }
            };
            ids.push(id);
        }
        Some(ids)
    }

    /// The store lets records `ids` go; no fire reads them any more.
    pub fn give_records(&mut self, ids: &[u32]) {
        for &id in ids {
            let record = self
                .records
                .get_mut(id as usize)
                .and_then(Option::take)
                .unwrap_or_else(|| panic!("record {id} is not out"));
            for slots in &record {
                self.pool.release(STORE, slots, 0);
            }
            self.free_records.insert(id);
        }
    }

    /// Each fixed shape's slot refs record `id` holds, pieces in order.
    #[must_use]
    pub fn record_refs(&self, id: u32) -> Vec<(EngineShape, Vec<u32>)> {
        let Some(Some(record)) = self.records.get(id as usize) else {
            return Vec::new();
        };
        self.fixed
            .iter()
            .zip(record)
            .map(|(shape, slots)| {
                (
                    shape.id,
                    slots.iter().map(|&slot| self.pool.slot_ref(slot)).collect(),
                )
            })
            .collect()
    }

    /// The slots record `id` holds, every shape's.
    #[must_use]
    pub fn record_slots(&self, id: u32) -> Vec<Slot> {
        self.records
            .get(id as usize)
            .and_then(Option::as_ref)
            .map_or_else(Vec::new, |record| {
                record.iter().flatten().copied().collect()
            })
    }

    // ---- lane inputs ----

    /// A lane's [`::engine::fire::Lane::fixed`]: one `[run, cow]` a piece of
    /// each fixed shape. `cow == run` writes in place.
    #[must_use]
    pub fn lane_fixed(&self, run: u32, cow: u32) -> Vec<(EngineShape, Vec<[u32; 2]>)> {
        let cow_refs = self.record_refs(cow);
        self.record_refs(run)
            .into_iter()
            .zip(cow_refs)
            .map(|((shape, run), (_, cow))| {
                (
                    shape,
                    run.into_iter()
                        .zip(cow)
                        .map(|(run, cow)| [run, cow])
                        .collect(),
                )
            })
            .collect()
    }

    /// A lane's [`::engine::fire::Lane::tables`]: its list slot refs, and
    /// each window list's own table `windows[l]` (slot refs in its token
    /// order, 0 where trimmed: the null chunk); a shape the pool lacks or
    /// the lane does not read is left out.
    #[must_use]
    pub fn lane_tables(&self, list: &[u32], windows: &[Vec<u32>]) -> Vec<(EngineShape, Vec<u32>)> {
        let mut tables = Vec::new();
        if let Some(shape) = self.list.filter(|_| !list.is_empty()) {
            tables.push((shape.id, list.to_vec()));
        }
        for (shape, table) in self.windows.iter().zip(windows) {
            if !table.is_empty() {
                tables.push((shape.id, table.clone()));
            }
        }
        tables
    }

    /// Whole-slot copies `[src, dst]` of each window list, from slot refs
    /// copied `(list, src, dst)`, as [`::engine::fire::Lane::slot_copies`]
    /// states them; a list with none left out.
    #[must_use]
    pub fn window_copies(&self, copies: &[(usize, u32, u32)]) -> Vec<(EngineShape, Vec<[u32; 2]>)> {
        self.windows
            .iter()
            .enumerate()
            .map(|(at, shape)| {
                (
                    shape.id,
                    copies
                        .iter()
                        .filter(|&&(list, ..)| list == at)
                        .map(|&(_, src, dst)| [src, dst])
                        .collect::<Vec<_>>(),
                )
            })
            .filter(|(_, pairs)| !pairs.is_empty())
            .collect()
    }

    /// The slot ref a snapshot of record `id` is named by in
    /// [`::engine::fire::RsSnapshot::slot`]: its first fixed shape's first
    /// piece.
    #[must_use]
    pub fn snapshot_ref(&self, id: u32) -> u32 {
        self.record_refs(id)
            .first()
            .and_then(|(_, refs)| refs.first().copied())
            .unwrap_or(0)
    }

    /// Every piece of record `id`'s first fixed shape, in piece order, for
    /// [`::engine::fire::RsSnapshot::pieces`] (CONTRACT-BC 16); empty where
    /// the record is one piece, which [`Self::snapshot_ref`] names alone.
    #[must_use]
    pub fn snapshot_pieces(&self, id: u32) -> Vec<u32> {
        self.record_refs(id)
            .into_iter()
            .next()
            .map(|(_, refs)| refs)
            .filter(|refs| refs.len() > 1)
            .unwrap_or_default()
    }

    // ---- rungs ----

    /// Sets the activation aside for fires of up to `rung` tokens. A rung
    /// no larger than now takes nothing more; a larger one first moves the
    /// state in its span down: the engine copies it before the next fire
    /// (`port.evacuate`), and the chunks moved off stay closed until the
    /// fire at `epoch` retires. Nothing moves when the port refuses.
    pub fn raise_rung(
        &mut self,
        rung: u32,
        epoch: u64,
        port: &mut dyn SegmentPort,
    ) -> Result<(), RungError> {
        if self.pool.act_chunks(rung) <= self.pool.act_reserved() {
            return self.pool.set_rung(rung).map_err(RungError::Pool);
        }
        let evacuation = self.pool.plan_rung(rung).map_err(RungError::Pool)?;
        if !evacuation.moves.is_empty() {
            port.evacuate(&engine_moves(&evacuation.moves))
                .map_err(|error| RungError::Engine(error.to_string()))?;
        }
        self.apply_rung(&evacuation, epoch)
    }

    fn apply_rung(&mut self, evacuation: &RungEvacuation, epoch: u64) -> Result<(), RungError> {
        self.pool
            .apply_rung(evacuation, epoch)
            .map_err(RungError::Pool)
    }

    /// Lets state take the span above `rung`'s activation again.
    pub fn lower_rung(&mut self, rung: u32) -> Result<(), RungError> {
        if self.pool.act_chunks(rung) > self.pool.act_reserved() {
            return Err(RungError::Pool(SlotError::Occupied { rung, chunks: 0 }));
        }
        self.pool.set_rung(rung).map_err(RungError::Pool)
    }

    // ---- segments ----

    #[must_use]
    pub fn segment_bytes(&self) -> u64 {
        u64::from(self.pool.segments().segment_chunks()) * self.pool.chunk_bytes()
    }

    /// Segments with memory under them: the prefix (draining ones in it
    /// included) and the top span.
    #[must_use]
    pub fn mapped_segments(&self) -> u32 {
        self.pool.segments().mapped_segments()
    }

    /// The next step toward `target` mapped segments, the top span counted;
    /// `None` once there. A grow commits the lowest uncommitted segment, the
    /// commit ahead of need included; a shrink evacuates the highest
    /// committed segment of the prefix (applied here, for the engine to
    /// copy before the next fire; its chunks stay out until `epoch`
    /// retires), and releases it once drained.
    pub fn next_move(&mut self, target: u32, epoch: u64) -> Option<SegmentMove> {
        let segments = self.pool.segments();
        let top = self.pool.top_segment();
        let prefix = (0..top)
            .take_while(|&s| segments.state(s) != SegmentState::Uncommitted)
            .count() as u32;
        let mapped = self.mapped_segments();
        if mapped < target && prefix < top {
            return Some(SegmentMove::Commit { prefix: prefix + 1 });
        }
        if mapped <= target || prefix == 0 {
            return None;
        }
        let last = prefix - 1;
        match segments.state(last) {
            SegmentState::Draining { .. } => Some(match self.pool.release_segment(last) {
                Ok(()) => SegmentMove::Release { prefix: last },
                Err(error) => {
                    tracing::trace!(%error, epoch, "a draining segment waits");
                    SegmentMove::Wait
                }
            }),
            SegmentState::Committed if segments.used(last) == 0 => {
                Some(match self.pool.release_segment(last) {
                    Ok(()) => SegmentMove::Release { prefix: last },
                    Err(_) => SegmentMove::NoRoom,
                })
            }
            SegmentState::Committed => {
                let Some(evacuation) = self.pool.plan_evacuation(last) else {
                    return Some(SegmentMove::NoRoom);
                };
                self.apply_evacuation(&evacuation, epoch);
                Some(SegmentMove::Evacuate {
                    evacuation: engine_moves(&evacuation.moves),
                })
            }
            SegmentState::Uncommitted => None,
        }
    }

    /// The engine took the evacuation [`Self::next_move`] planned, with
    /// fires through `current` (the kv store's per-fire epoch, CONTRACT-BC
    /// 17) handed out by then. Any of those may run ahead of the copy and
    /// read the old location; the first fire after them runs after it. The
    /// segment drains until that one, `current + 1`, retires.
    pub fn evacuated(&mut self, current: u64) {
        self.pool.fence_drains(current.saturating_add(1));
    }

    fn apply_evacuation(&mut self, evacuation: &Evacuation, epoch: u64) {
        self.pool
            .apply_evacuation(evacuation, epoch)
            .expect("an evacuation planned under the lock applies");
    }

    /// The engine backed segments `0..prefix`: their chunks go out now.
    pub fn committed(&mut self, prefix: u32) {
        for segment in 0..prefix.min(self.pool.top_segment()) {
            if self.pool.segments().state(segment) == SegmentState::Uncommitted {
                self.pool
                    .commit_segment(segment)
                    .expect("an uncommitted segment commits");
            }
        }
    }

    /// The segments to commit ahead of need: enough that `bytes` (what
    /// waiters lack, plus [`COMMIT_AHEAD_BYTES`]) stand ready; the mapped
    /// target that takes, never past the pool.
    #[must_use]
    pub fn ahead_target(&self, bytes: u64) -> u32 {
        let wanted = self
            .pool
            .commit_ahead(bytes.saturating_add(COMMIT_AHEAD_BYTES));
        self.mapped_segments() + wanted.len() as u32
    }

    /// Steps toward `target` mapped segments through `port` until there, or
    /// until a shrink finds no room or waits on a fire; the fire at `epoch`
    /// is the last that may read what an evacuation moves. A move the port
    /// refuses stops the walk; what the pool applied stands, as the engine
    /// would have it once the port does.
    pub fn steer(&mut self, target: u32, epoch: u64, port: &mut dyn SegmentPort) -> Retarget {
        loop {
            match self.next_move(target, epoch) {
                None => return Retarget::Reached,
                Some(SegmentMove::Commit { prefix }) => {
                    if port.commit_segments(prefix).is_err() {
                        return Retarget::Reached;
                    }
                    self.committed(prefix);
                }
                Some(SegmentMove::Evacuate { evacuation }) => {
                    let _ = port.evacuate(&evacuation);
                    return Retarget::Reached;
                }
                Some(SegmentMove::Release { prefix }) => {
                    let _ = port.release_segments(prefix);
                }
                Some(SegmentMove::Wait) => return Retarget::Reached,
                Some(SegmentMove::NoRoom) => {
                    return Retarget::NoRoom {
                        held: u64::from(self.mapped_segments()),
                    };
                }
            }
        }
    }

    // ---- governor ----

    /// What is asked of the pool, in bytes: the slots held and wanted, the
    /// reserved chunks, and the top span, committed from the load.
    #[must_use]
    pub fn demand(&self) -> Asked {
        let Demand { live, wanted } = self.pool.demand();
        let fixed = self.fixed_bytes();
        Asked {
            wanted: wanted + fixed,
            live: live + fixed,
        }
    }

    /// The reserved chunks and the top span, held from the load.
    fn fixed_bytes(&self) -> u64 {
        u64::from(self.pool.reserved_chunks() + self.pool.act_max()) * self.pool.chunk_bytes()
    }

    /// Bytes below which one lane cannot run: one working set at the longest
    /// context, the reserved chunks and the top span.
    #[must_use]
    pub fn floor(&self) -> u64 {
        u64::from(self.least_chunks) * self.pool.chunk_bytes() + self.fixed_bytes()
    }

    #[must_use]
    pub fn hard_limit(&self) -> u64 {
        u64::from(self.pool.total_chunks()) * self.pool.chunk_bytes()
    }

    /// Bytes under the mapped segments, the top span's included.
    #[must_use]
    pub fn held(&self) -> u64 {
        let segments = self.pool.segments();
        (0..segments.count())
            .filter(|&s| segments.state(s) != SegmentState::Uncommitted)
            .map(|s| u64::from(segments.len(s)) * self.pool.chunk_bytes())
            .sum()
    }

    #[must_use]
    pub fn growth(&self) -> Growth {
        match self.growth {
            SegmentGrowth::Fixed => Growth::Fixed,
            SegmentGrowth::Bottom | SegmentGrowth::TwoEnded => Growth::Elastic {
                least_cut: self.segment_bytes(),
            },
        }
    }

    /// The holder record slots are taken under.
    #[must_use]
    pub const fn store_holder() -> Holder {
        STORE
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RungError {
    #[error("{0}")]
    Pool(SlotError),
    #[error("the engine refused the evacuation: {0}")]
    Engine(String),
}

#[cfg(test)]
mod tests;

mod target;
pub use target::SharedGrow;
