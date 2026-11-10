//! The stores' slot ids, taken from [`SlotPool`]s.
//!
//! The engine states its pools as chunks ([`ChunkFacts`]); until a backend
//! switches to one shared pool, each of today's pools is a *pinned* shape:
//! a run of chunks of its own, one slot a chunk, whose slots keep the ids
//! the engine numbers them by today ([`ChunkFacts::degenerate`]). Each such
//! run is one [`ShapePool`]: a `SlotPool` of one shape whose chunk `c` is
//! the store's id `base + c`. A `SlotPool` opens its lowest free chunk first,
//! as the stores' own pools did, so the ids handed to the engine are the
//! ones it was handed before.
//!
//! Each chunk is a segment of its own here: a pool that gains memory as it
//! is asked for commits a prefix of its segments and stops handing out
//! chunks past it. A held chunk past that limit stays committed until it is
//! let go; then its segment is released.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::policy::pressure::{Bytes, Meter, Tier};
use crate::store::registry;

use ::engine::chunk::{
    ChunkFacts, DEGENERATE_BUFFER, DEGENERATE_KV, DEGENERATE_STATE, DEGENERATE_WINDOW, Pinned,
    SlotKind,
};

use super::kv::page_table::WorkingSetId;
use super::pool::PoolId;
use super::shared::{Shared, SharedPool, TokenShape};
use super::slots::{
    Demand, Holder, SegmentState, ShapeId, ShapeKind, Slot, SlotPool, SlotPoolFacts, SlotShape,
};

/// The holder a store's own allocations are taken under: the store's
/// table, which refcounts them, holds each id it hands out.
pub const STORE: Holder = Holder::WorkingSet(u64::MAX);

/// Where a [`ShapePool`]'s ids come from.
#[derive(Clone, Debug)]
enum Backing {
    /// A pinned shape's own pool: id `base + c` is chunk `c`.
    Own(Box<SlotPool>),
    /// A list or ring shape of an engine's shared pool: an id is the slot
    /// ref, and the null chunk's ref 0 is never handed out.
    Shape(Shared, ShapeId),
    /// The fixed records of an engine's shared pool: id `base + r` is
    /// record `r`, a slot of every fixed shape for each of its pieces.
    Records(Shared),
}

/// One store's ids of one shape: a pinned shape's own run (see the module
/// docs), or a shape of an engine's shared pool ([`super::shared`]).
#[derive(Clone, Debug)]
pub struct ShapePool<I> {
    slots: Backing,
    kind: ShapeKind,
    slot_bytes: u64,
    base: u32,
    capacity: u32,
    /// One past the highest id there may be.
    ids_end: u32,
    /// Chunks at and past it are not handed out.
    limit: u32,
    /// Ids let go, each batch waiting on the fire at its epoch.
    pending: Vec<(u64, Vec<I>)>,
    id: PhantomData<I>,
}

/// The waiter a store's shortfall is recorded under in a shared pool, one a
/// store: the shape's id, the records' past every shape's.
fn waiter(backing: &Backing) -> u64 {
    match backing {
        Backing::Shape(_, shape) => u64::from(*shape),
        Backing::Own(_) | Backing::Records(_) => u64::MAX,
    }
}

impl<I: PoolId> ShapePool<I> {
    /// `capacity` ids from `base`, every one committed.
    pub fn new(kind: ShapeKind, slot_bytes: u64, base: u32, capacity: u32) -> Self {
        base.checked_add(capacity)
            .expect("pool id range overflows u32");
        let slots = SlotPool::new(SlotPoolFacts {
            chunk_bytes: slot_bytes,
            total_chunks: capacity,
            segment_chunks: 1,
            committed_segments: capacity,
            reserved_chunks: 0,
            shapes: vec![SlotShape {
                kind,
                slot_bytes,
                slots_per_chunk: 1,
            }],
            act_chunks: Vec::new(),
        });
        Self {
            slots: Backing::Own(Box::new(slots)),
            kind,
            slot_bytes,
            base,
            capacity,
            ids_end: base + capacity,
            limit: capacity,
            pending: Vec::new(),
            id: PhantomData,
        }
    }

    /// The slot refs of `shared`'s per-token shape `shape` as ids. Its
    /// capacity is what the state chunks would hold of it alone; its ids run
    /// to every ref the pool's chunks could name.
    pub fn shared(shared: &Shared, shape: TokenShape) -> Self {
        Self::shared_from(shared, shape, 0)
    }

    /// [`Self::shared`] with id `base + r` naming slot ref `r`, for a store
    /// that numbers another pool's ids below them.
    pub fn shared_from(shared: &Shared, shape: TokenShape, base: u32) -> Self {
        let (kind, capacity, ids_end) = {
            let pool = shared.lock().unwrap();
            let slots = pool.pool();
            let kind = slots.shape(shape.pool).map_or(ShapeKind::List, |s| s.kind);
            let refs = |chunks: u32| {
                u32::try_from(u64::from(chunks) * u64::from(shape.slots_per_chunk))
                    .unwrap_or(u32::MAX)
            };
            let state = slots
                .total_chunks()
                .saturating_sub(slots.reserved_chunks())
                .saturating_sub(slots.act_chunks(0));
            (kind, refs(state), refs(slots.total_chunks()))
        };
        base.checked_add(ids_end)
            .expect("pool id range overflows u32");
        Self {
            slots: Backing::Shape(Arc::clone(shared), shape.pool),
            kind,
            slot_bytes: shape.slot_bytes,
            base,
            capacity,
            ids_end: base + ids_end,
            limit: capacity,
            pending: Vec::new(),
            id: PhantomData,
        }
    }

    /// `shared`'s fixed records as ids from `base`.
    pub fn records(shared: &Shared, base: u32) -> Self {
        let (capacity, slot_bytes) = {
            let pool = shared.lock().unwrap();
            let bytes = pool
                .fixed()
                .iter()
                .map(|shape| shape.slot_bytes * u64::from(shape.pieces))
                .sum();
            (pool.records_capacity(), bytes)
        };
        base.checked_add(capacity)
            .expect("pool id range overflows u32");
        Self {
            slots: Backing::Records(Arc::clone(shared)),
            kind: ShapeKind::Fixed,
            slot_bytes,
            base,
            capacity,
            ids_end: base + capacity,
            limit: capacity,
            pending: Vec::new(),
            id: PhantomData,
        }
    }

    /// One past the highest id the pool may hand out.
    pub fn ids_end(&self) -> u32 {
        self.ids_end
    }

    /// Whether the ids are an engine's shared pool's.
    pub fn is_shared(&self) -> bool {
        !matches!(self.slots, Backing::Own(_))
    }

    fn slot(&self, id: I) -> Slot {
        debug_assert!(self.contains(id), "id {} is not this pool's", id.index());
        Slot {
            chunk: id.index() - self.base,
            index: 0,
        }
    }

    fn slots_of(&self, ids: &[I]) -> Vec<Slot> {
        ids.iter().map(|&id| self.slot(id)).collect()
    }

    /// The shared pool's slots `ids` name.
    fn shared_slots(&self, pool: &SharedPool, ids: &[I]) -> Vec<Slot> {
        match &self.slots {
            Backing::Shape(_, shape) => ids
                .iter()
                .map(|&id| pool.slot(*shape, id.index() - self.base))
                .collect(),
            Backing::Records(_) => ids
                .iter()
                .flat_map(|&id| pool.record_slots(id.index() - self.base))
                .collect(),
            Backing::Own(_) => unreachable!("a pinned pool's slots are its own"),
        }
    }

    /// Runs `f` on the slot pool `ids` live in, with their slots there.
    fn with_slots<T>(&mut self, ids: &[I], f: impl FnOnce(&mut SlotPool, Vec<Slot>) -> T) -> T {
        match &self.slots {
            Backing::Own(_) => {
                let slots = self.slots_of(ids);
                let Backing::Own(pool) = &mut self.slots else {
                    unreachable!()
                };
                f(pool, slots)
            }
            Backing::Shape(shared, _) | Backing::Records(shared) => {
                let shared = Arc::clone(shared);
                let mut pool = shared.lock().unwrap();
                let slots = self.shared_slots(&pool, ids);
                f(pool.pool_mut(), slots)
            }
        }
    }

    /// The `n` lowest free ids, ascending, or none when fewer than `n`
    /// stand free under the limit.
    pub fn try_alloc_n(&mut self, n: usize) -> Option<Vec<I>> {
        match &mut self.slots {
            Backing::Own(pool) => {
                let taken = pool.acquire(STORE, 0, n as u64).ok()?;
                Some(
                    taken
                        .into_iter()
                        .map(|slot| I::from_index(self.base + slot.chunk))
                        .collect(),
                )
            }
            Backing::Shape(shared, shape) => {
                let refs = shared.lock().unwrap().take_refs(*shape, n as u64)?;
                Some(
                    refs.into_iter()
                        .map(|r| I::from_index(self.base + r))
                        .collect(),
                )
            }
            Backing::Records(shared) => {
                let records = shared.lock().unwrap().take_records(n)?;
                Some(
                    records
                        .into_iter()
                        .map(|record| I::from_index(self.base + record))
                        .collect(),
                )
            }
        }
    }

    /// `ids` go back once the fire at `epoch` retires.
    pub fn recycle_after_epoch(&mut self, ids: Vec<I>, epoch: u64) {
        if !ids.is_empty() {
            self.pending.push((epoch, ids));
        }
    }

    /// `ids`, which no fire has touched, go back now.
    pub fn release_reserved(&mut self, ids: Vec<I>) {
        self.give_back(&ids);
    }

    pub fn retire_through(&mut self, epoch: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= epoch {
                let (_, ids) = self.pending.swap_remove(i);
                self.give_back(&ids);
            } else {
                i += 1;
            }
        }
    }

    fn give_back(&mut self, ids: &[I]) {
        if ids.is_empty() {
            return;
        }
        match &mut self.slots {
            Backing::Own(pool) => {
                let slots: Vec<Slot> = ids
                    .iter()
                    .map(|&id| Slot {
                        chunk: id.index() - self.base,
                        index: 0,
                    })
                    .collect();
                pool.release(STORE, &slots, 0);
                self.release_past_limit(&slots);
            }
            Backing::Shape(shared, shape) => {
                let refs: Vec<u32> = ids.iter().map(|id| id.index() - self.base).collect();
                shared.lock().unwrap().give_refs(*shape, &refs);
            }
            Backing::Records(shared) => {
                let records: Vec<u32> = ids.iter().map(|id| id.index() - self.base).collect();
                shared.lock().unwrap().give_records(&records);
            }
        }
    }

    /// Releases the segments of `slots` past the limit that hold nothing
    /// now, so they are not handed out again.
    fn release_past_limit(&mut self, slots: &[Slot]) {
        let Backing::Own(pool) = &mut self.slots else {
            return;
        };
        for slot in slots {
            if slot.chunk >= self.limit && pool.segments().used(slot.chunk) == 0 {
                pool.release_segment(slot.chunk)
                    .expect("an empty committed segment past the limit goes");
            }
        }
    }

    /// Stops handing out ids at `limit` and up, or hands them out again.
    /// An id already out stays out until it is given back; [`Self::tail_held`]
    /// counts those. A shared pool moves by its segments, not by a limit.
    pub fn set_limit(&mut self, limit: u32) {
        let Backing::Own(pool) = &mut self.slots else {
            return;
        };
        let limit = limit.min(self.capacity);
        for segment in 0..self.capacity {
            let state = pool.segments().state(segment);
            if segment < limit {
                if state == SegmentState::Uncommitted {
                    pool.commit_segment(segment)
                        .expect("an uncommitted segment commits");
                }
            } else if state == SegmentState::Committed && pool.segments().used(segment) == 0 {
                pool.release_segment(segment)
                    .expect("an empty committed segment goes");
            }
        }
        self.limit = limit;
    }

    /// The ids at and past the limit still out, waiting on an epoch or on
    /// their holder.
    pub fn tail_held(&self) -> usize {
        let Backing::Own(pool) = &self.slots else {
            return 0;
        };
        (self.limit..self.capacity)
            .filter(|&segment| pool.segments().used(segment) > 0)
            .count()
    }

    /// One past the highest offset still out, at least the limit.
    pub fn held_end(&self) -> u32 {
        self.high_end().max(self.limit)
    }

    /// One past the highest offset still out.
    pub fn high_end(&self) -> u32 {
        let Backing::Own(pool) = &self.slots else {
            return self.capacity;
        };
        (0..self.capacity)
            .rev()
            .find(|&segment| pool.segments().used(segment) > 0)
            .map_or(0, |segment| segment + 1)
    }

    /// Free ids under the limit.
    pub fn available(&self) -> usize {
        match &self.slots {
            Backing::Own(pool) => pool.available(0) as usize,
            Backing::Shape(shared, shape) => {
                shared.lock().unwrap().pool().available(*shape) as usize
            }
            Backing::Records(shared) => shared.lock().unwrap().records_available() as usize,
        }
    }

    /// Ids handed out under the limit. A shared pool's list or ring shape
    /// counts the slots held of it: its segments not yet committed are room
    /// it grows into, not ids out.
    pub fn out(&self) -> u32 {
        match &self.slots {
            Backing::Shape(shared, shape) => {
                let demand = shared.lock().unwrap().pool().shape_demand(*shape);
                u32::try_from(demand.live + demand.cached).unwrap_or(u32::MAX)
            }
            Backing::Own(_) | Backing::Records(_) => {
                self.limit.saturating_sub(self.available() as u32)
            }
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn limit(&self) -> u32 {
        self.limit
    }

    pub fn base(&self) -> u32 {
        self.base
    }

    pub fn contains(&self, id: I) -> bool {
        (self.base..self.base.saturating_add(self.capacity)).contains(&id.index())
    }

    pub fn kind(&self) -> ShapeKind {
        self.kind
    }

    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Adds `holder`'s reference to ids the store holds: a cache entry or a
    /// snapshot keeping them. Dropped with [`Self::unshare`] before the
    /// store lets them go.
    pub fn share(&mut self, holder: Holder, ids: &[I]) {
        self.with_slots(ids, |pool, slots| pool.share(holder, &slots));
    }

    /// Drops `holder`'s reference again. An id holding no cache reference
    /// is passed over: the store's own count, not this, decides when an id
    /// goes, so a hold that went astray never stops a release.
    pub fn unshare(&mut self, holder: Holder, ids: &[I]) {
        self.with_slots(ids, |pool, slots| {
            let slots: Vec<Slot> = slots
                .into_iter()
                .filter(|&slot| {
                    let held = pool.refs(slot).1 > 0;
                    debug_assert!(held, "{slot:?} lets go a cache hold it never took");
                    held
                })
                .collect();
            if pool.held_by(&holder) >= slots.len() as u64 {
                pool.release(holder, &slots, 0);
            }
        });
    }

    /// The cache entries holding a reference on `id`.
    pub fn cache_refs(&self, id: I) -> u32 {
        match &self.slots {
            Backing::Own(pool) => pool.refs(self.slot(id)).1,
            Backing::Shape(shared, _) | Backing::Records(shared) => {
                let pool = shared.lock().unwrap();
                self.shared_slots(&pool, &[id])
                    .first()
                    .map_or(0, |&slot| pool.pool().refs(slot).1)
            }
        }
    }

    /// Ids `holder` holds a reference on (slots, in a shared pool's records).
    pub fn held_by(&self, holder: &Holder) -> u64 {
        match &self.slots {
            Backing::Own(pool) => pool.held_by(holder),
            Backing::Shape(shared, _) | Backing::Records(shared) => {
                shared.lock().unwrap().pool().held_by(holder)
            }
        }
    }

    /// Records that waiters are short `ids` of this pool, replacing what
    /// they were short before.
    pub fn wait(&mut self, ids: u64) {
        let waiter = waiter(&self.slots);
        match &mut self.slots {
            Backing::Own(pool) => {
                if ids == 0 {
                    pool.unwait(0);
                } else {
                    pool.wait(0, 0, ids);
                }
            }
            Backing::Shape(shared, shape) => {
                let mut shared = shared.lock().unwrap();
                if ids == 0 {
                    shared.pool_mut().unwait(waiter);
                } else {
                    shared.pool_mut().wait(waiter, *shape, ids);
                }
            }
            Backing::Records(shared) => {
                let mut shared = shared.lock().unwrap();
                let first = shared.fixed().first().copied();
                match first {
                    Some(shape) if ids > 0 => {
                        shared
                            .pool_mut()
                            .wait(waiter, shape.pool, ids * u64::from(shape.pieces))
                    }
                    _ => shared.pool_mut().unwait(waiter),
                }
            }
        }
    }

    /// What is held and wanted of this pool, in bytes.
    pub fn demand(&self) -> Demand {
        let shape_demand = |pool: &SlotPool, shape: ShapeId| {
            let demand = pool.shape_demand(shape);
            let bytes = pool.shape(shape).map_or(0, |s| s.slot_bytes);
            Demand {
                live: demand.live * bytes,
                wanted: (demand.live + demand.cached + demand.waiting) * bytes,
            }
        };
        match &self.slots {
            Backing::Own(pool) => pool.demand(),
            Backing::Shape(shared, shape) => shape_demand(shared.lock().unwrap().pool(), *shape),
            Backing::Records(shared) => {
                let shared = shared.lock().unwrap();
                shared
                    .fixed()
                    .iter()
                    .map(|shape| shape_demand(shared.pool(), shape.pool))
                    .fold(Demand::default(), |sum, demand| Demand {
                        live: sum.live + demand.live,
                        wanted: sum.wanted + demand.wanted,
                    })
            }
        }
    }
}

/// Where a lane's state rides when the runtime binds none of its own: a
/// fixed slot of the seat shape, which takes no memory, held by the lane's
/// working set. A working set holds one for each lane it fires at once;
/// lane `i` of its group rides its `i`th.
#[derive(Debug)]
pub struct Seats {
    slots: SlotPool,
    capacity: u32,
    held: HashMap<WorkingSetId, Vec<Slot>>,
    /// A working set let its seats go: a fire short of seats looks again.
    freed: Arc<tokio::sync::Notify>,
}

/// The seats a model without state slots holds at once at most.
const SEATS_UNBOUNDED: u32 = 1 << 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeatsShort {
    pub need: u32,
    pub have: u32,
    pub capacity: u32,
}

impl std::fmt::Display for SeatsShort {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "this fire seats {} more sequence(s) than the pools have room for \
             ({} seat(s) free of the {} the state slots give)",
            self.need, self.have, self.capacity
        )
    }
}

impl Seats {
    /// As many seats as there are state slots; a model without any seats
    /// as many lanes as it fires.
    pub fn new(state_slots: u32) -> Self {
        let capacity = if state_slots == 0 {
            SEATS_UNBOUNDED
        } else {
            state_slots
        };
        Self {
            slots: SlotPool::new(SlotPoolFacts {
                chunk_bytes: 0,
                total_chunks: capacity,
                segment_chunks: capacity.max(1),
                committed_segments: u32::from(capacity > 0),
                reserved_chunks: 0,
                shapes: vec![SlotShape {
                    kind: ShapeKind::Fixed,
                    slot_bytes: 0,
                    slots_per_chunk: 1,
                }],
                act_chunks: Vec::new(),
            }),
            capacity: state_slots,
            held: HashMap::new(),
            freed: Arc::new(tokio::sync::Notify::new()),
        }
    }

    /// The seats of `ws`'s first `lanes` lanes, taking what it lacks.
    pub fn seats(&mut self, ws: WorkingSetId, lanes: usize) -> Result<Vec<u32>, SeatsShort> {
        let lanes = u32::try_from(lanes).unwrap_or(u32::MAX);
        let have = self.held.get(&ws).map_or(0, |held| held.len() as u32);
        if have < lanes {
            let need = lanes - have;
            let fresh = self
                .slots
                .acquire(Holder::WorkingSet(ws.to_bits()), 0, u64::from(need))
                .map_err(|_| SeatsShort {
                    need,
                    have: self.slots.available(0) as u32,
                    capacity: self.capacity,
                })?;
            self.held.entry(ws).or_default().extend(fresh);
        }
        Ok(self.held.get(&ws).map_or_else(Vec::new, |held| {
            held[..lanes as usize]
                .iter()
                .map(|slot| slot.chunk)
                .collect()
        }))
    }

    /// `ws` lets its seats go.
    pub fn release(&mut self, ws: WorkingSetId) {
        if let Some(held) = self.held.remove(&ws) {
            self.slots
                .release(Holder::WorkingSet(ws.to_bits()), &held, 0);
            self.freed.notify_waiters();
        }
    }

    /// Fires once a working set lets its seats go.
    pub fn freed(&self) -> Arc<tokio::sync::Notify> {
        Arc::clone(&self.freed)
    }

    /// Whether `need` seats could ever stand free at once.
    pub fn could_seat(&self, need: u32) -> bool {
        need <= self.slots.total_chunks()
    }

    #[cfg(test)]
    pub fn available(&self) -> u32 {
        self.slots.available(0) as u32
    }
}

/// One pinned shape's runs, as the stores take them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShapeRun {
    pub slot_bytes: u64,
    /// Device ids, from [`Pinned::ids_from`].
    pub device: u32,
    pub ids_from: u32,
    pub host: u32,
    pub disk: u32,
}

impl ShapeRun {
    fn of(pinned: Pinned, slot_bytes: u64, slots_per_chunk: u32) -> Self {
        Self {
            slot_bytes,
            device: pinned.device.chunks * slots_per_chunk,
            ids_from: pinned.ids_from,
            host: pinned.host.chunks * slots_per_chunk,
            disk: pinned.disk.chunks * slots_per_chunk,
        }
    }
}

/// What a store needs of an engine's [`ChunkFacts`]: each of today's pools,
/// as the pinned shape the degenerate mode states it as; or, where the
/// engine states a shared pool, its geometry, the stores drawing their ids
/// from a [`SharedPool`] built of the same facts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StorePools {
    /// The shapes share one pool: the counts below are not stated.
    pub shared: bool,
    pub page_size: u32,
    pub kv: ShapeRun,
    /// The window ring's pages but the null page 0; its ids run from 1.
    pub window: ShapeRun,
    pub window_tokens: u32,
    pub state: ShapeRun,
    pub buffer: ShapeRun,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ChunkFactsError {
    #[error(
        "shape {0} takes chunks from the shared pool beside pinned shapes; an engine states every shape pinned or none"
    )]
    Mixed(u32),
    #[error("shape {0} is not one of the degenerate mode's")]
    Unknown(u32),
    #[error("shape {0} cuts a slot into pieces, which the stores do not take yet")]
    Pieces(u32),
}

impl StorePools {
    /// The stores' pools as `facts` states them: every shape pinned (the
    /// degenerate mode), or none (a shared pool).
    pub fn of(facts: &ChunkFacts) -> Result<Self, ChunkFactsError> {
        if SharedPool::states_shared(facts) {
            return Ok(Self::shared(facts));
        }
        let mut pools = Self::default();
        for shape in &facts.shapes {
            let pinned = shape.pinned.ok_or(ChunkFactsError::Mixed(shape.id.0))?;
            let run = ShapeRun::of(pinned, shape.slot_bytes, shape.slots_per_chunk);
            match (shape.id, shape.kind) {
                (DEGENERATE_KV, SlotKind::List { tokens, .. }) => {
                    pools.page_size = tokens;
                    pools.kv = run;
                }
                (DEGENERATE_WINDOW, SlotKind::Ring { window, tokens }) => {
                    pools.page_size = tokens;
                    pools.window = run;
                    pools.window_tokens = window;
                }
                (DEGENERATE_STATE | DEGENERATE_BUFFER, SlotKind::Fixed { pieces })
                    if pieces != 1 =>
                {
                    return Err(ChunkFactsError::Pieces(shape.id.0));
                }
                (DEGENERATE_STATE, SlotKind::Fixed { .. }) => pools.state = run,
                (DEGENERATE_BUFFER, SlotKind::Fixed { .. }) => pools.buffer = run,
                _ => return Err(ChunkFactsError::Unknown(shape.id.0)),
            }
        }
        Ok(pools)
    }

    /// A shared pool's geometry: the kv list and ring shapes' tokens a
    /// slot and the window, and the rs list's slot bytes as the buffer's.
    fn shared(facts: &ChunkFacts) -> Self {
        let mut pools = Self {
            shared: true,
            ..Self::default()
        };
        for shape in &facts.shapes {
            let run = ShapeRun {
                slot_bytes: shape.slot_bytes,
                ..ShapeRun::default()
            };
            match shape.kind {
                // The rs buffer's list (CONTRACT-BC 18): its tokens are
                // the rs store's, not the kv page size.
                SlotKind::List { .. } if shape.role == ::engine::chunk::SlotRole::RsBuffer => {
                    pools.buffer = run;
                }
                SlotKind::List { tokens, .. } => {
                    pools.page_size = tokens;
                    pools.kv = run;
                }
                SlotKind::Ring { window, tokens } => {
                    if pools.page_size == 0 {
                        pools.page_size = tokens;
                    }
                    pools.window = run;
                    pools.window_tokens = window;
                }
                SlotKind::Fixed { pieces } => {
                    pools.state.slot_bytes += shape.slot_bytes * u64::from(pieces);
                }
            }
        }
        pools
    }

    /// The engine's window pages, the null page 0 counted.
    pub fn window_pages(&self) -> u32 {
        if self.window.device == 0 && self.window.host == 0 {
            0
        } else {
            self.window.device + self.window.ids_from
        }
    }
}

/// The worker policy's byte attrs, read from the stores' slot pools: each
/// shape's slots at its own size. What a cache entry alone holds is cache:
/// the kv store's idle cache pages, the rs checkpoints' rows.
pub struct SlotMeter;

impl SlotMeter {
    fn engine(stores: &registry::Stores, tier: Tier) -> Option<Bytes> {
        let (kv, kv_page_bytes, capacity, cached, host_free, host_capacity) =
            registry::with_kv_lock(&stores.kv, "policy-pressure", |kv| {
                let pool = kv.page_pool();
                (
                    pool.demand(),
                    pool.slot_bytes(),
                    u64::from(pool.capacity()),
                    u64::from(kv.reclaimable_cache_pages()),
                    kv.host_swap_available() as u64,
                    u64::from(kv.host_swap_capacity()),
                )
            });
        // A page of unknown size leaves the tier unread.
        if capacity > 0 && kv_page_bytes == 0 {
            return None;
        }
        let rs = stores.rs.lock().unwrap();
        let (slot_bytes, page_bytes) = rs.unit_bytes();
        let rows = |units: super::rs::RsUnits| {
            u64::from(units.slots) * slot_bytes + u64::from(units.pages) * page_bytes
        };
        Some(match tier {
            Tier::Device => {
                let state = rs.device_demand();
                let total = kv_page_bytes * capacity + rows(rs.capacity());
                let kv_cached = cached * kv_page_bytes;
                Bytes {
                    total,
                    state: kv.live + state.live,
                    live: kv.live.saturating_sub(kv_cached)
                        + state.live.saturating_sub(rows(rs.cache_only())),
                    wanted: kv.wanted + state.wanted,
                    act: 0,
                }
            }
            Tier::Host => {
                let total = host_capacity * kv_page_bytes + rows(rs.host_capacity());
                let held = (host_capacity - host_free.min(host_capacity)) * kv_page_bytes
                    + rows(rs.host_capacity().sub(rs.host_available()));
                Bytes {
                    total,
                    state: held,
                    live: held,
                    wanted: held,
                    act: 0,
                }
            }
        })
    }
}

impl Meter for SlotMeter {
    fn bytes(&self, tier: Tier) -> Option<Bytes> {
        let all = registry::all_for_model(0);
        if all.is_empty() {
            return None;
        }
        let mut sum = Bytes::default();
        for stores in &all {
            let bytes = Self::engine(stores, tier)?;
            sum = Bytes {
                total: sum.total + bytes.total,
                state: sum.state + bytes.state,
                live: sum.live + bytes.live,
                wanted: sum.wanted + bytes.wanted,
                act: sum.act + bytes.act,
            };
        }
        Some(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::super::pool::Pool;
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Id(u32);

    impl PoolId for Id {
        fn from_index(index: u32) -> Self {
            Self(index)
        }
        fn index(self) -> u32 {
            self.0
        }
    }

    /// splitmix64, seeded so a failing case replays.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            if n == 0 { 0 } else { (z ^ (z >> 31)) % n }
        }
    }

    /// Every call the stores make, on the store pool and the slot pool
    /// side by side: the same ids, counts and limits throughout.
    #[test]
    fn a_shape_pool_hands_out_the_ids_the_store_pool_did() {
        for seed in 0..40u64 {
            let mut rng = Rng(seed);
            let base = rng.below(3) as u32 * 7;
            let capacity = 1 + rng.below(150) as u32;
            let mut old = Pool::<Id>::new_range(base, capacity);
            let mut new = ShapePool::<Id>::new(ShapeKind::List, 64, base, capacity);
            let mut held: Vec<Id> = Vec::new();
            let mut epoch = 0u64;
            for step in 0..600 {
                match rng.below(8) {
                    0 | 1 => {
                        let n = rng.below(6) as usize;
                        let (a, b) = (old.try_alloc_n(n), new.try_alloc_n(n));
                        assert_eq!(a, b, "seed {seed} step {step}: alloc {n}");
                        held.extend(a.unwrap_or_default());
                    }
                    2 if !held.is_empty() => {
                        let at = rng.below(held.len() as u64) as usize;
                        let id = held.swap_remove(at);
                        old.release_reserved(vec![id]);
                        new.release_reserved(vec![id]);
                    }
                    3 if !held.is_empty() => {
                        let at = rng.below(held.len() as u64) as usize;
                        let id = held.swap_remove(at);
                        let at = epoch + rng.below(3);
                        old.recycle_after_epoch(vec![id], at);
                        new.recycle_after_epoch(vec![id], at);
                    }
                    4 => {
                        epoch += 1;
                        old.retire_through(epoch);
                        new.retire_through(epoch);
                    }
                    5 => {
                        let limit = rng.below(u64::from(capacity) + 10) as u32;
                        old.set_limit(limit);
                        new.set_limit(limit);
                    }
                    _ => {}
                }
                assert_eq!(old.available(), new.available(), "seed {seed} step {step}");
                assert_eq!(old.limit(), new.limit());
                assert_eq!(old.tail_held(), new.tail_held(), "seed {seed} step {step}");
                assert_eq!(old.held_end(), new.held_end(), "seed {seed} step {step}");
                assert_eq!(old.high_end(), new.high_end(), "seed {seed} step {step}");
            }
        }
    }

    #[test]
    fn demand_is_counted_in_bytes_and_a_waiter_is_wanted() {
        let mut pool = ShapePool::<Id>::new(ShapeKind::Fixed, 1000, 0, 8);
        let ids = pool.try_alloc_n(3).unwrap();
        pool.wait(2);
        assert_eq!(
            pool.demand(),
            Demand {
                live: 3000,
                wanted: 5000
            }
        );
        pool.share(Holder::CacheEntry([7; 32]), &ids[..1]);
        assert_eq!(pool.held_by(&Holder::CacheEntry([7; 32])), 1);
        pool.unshare(Holder::CacheEntry([7; 32]), &ids[..1]);
        pool.wait(0);
        pool.release_reserved(ids);
        assert_eq!(pool.demand(), Demand::default());
        assert_eq!(pool.available(), 8);
    }

    fn two_working_sets() -> (WorkingSetId, WorkingSetId) {
        let mut kv = super::super::kv::KvStore::new(8, [0; 32]);
        (kv.create_working_set(), kv.create_working_set())
    }

    #[test]
    fn two_working_sets_never_share_a_seat_and_a_released_one_is_reissued() {
        let (a, b) = two_working_sets();
        let mut seats = Seats::new(2);
        let first = seats.seats(a, 1).unwrap();
        assert_eq!(seats.seats(a, 1).unwrap(), first, "a set keeps its seat");
        let second = seats.seats(b, 1).unwrap();
        assert_ne!(first, second, "two live sequences, two seats");
        assert_eq!(seats.available(), 0);
        seats.release(a);
        seats.release(a);
        assert_eq!(seats.available(), 1, "released once, returned once");
        assert_eq!(seats.seats(b, 2).unwrap(), vec![second[0], first[0]]);
    }

    #[test]
    fn a_fire_wider_than_the_seats_is_refused_and_takes_nothing() {
        let (a, b) = two_working_sets();
        let mut seats = Seats::new(2);
        seats.seats(a, 1).unwrap();
        let refusal = seats.seats(b, 3).unwrap_err();
        assert_eq!(
            refusal,
            SeatsShort {
                need: 3,
                have: 1,
                capacity: 2
            }
        );
        assert_eq!(seats.available(), 1, "a refused ask seats nothing");
        assert!(!seats.could_seat(3));
        assert!(
            Seats::new(0).could_seat(64),
            "no state slots: no cap to speak of"
        );
    }

    #[test]
    fn the_degenerate_facts_come_back_as_todays_pools() {
        let pools = ::engine::PoolFacts {
            kv_pages: 10,
            kv_page_size: 16,
            state_slots: 4,
            state_slot_bytes: 4096,
            buffer_pages: 6,
            window_pages: 5,
            window_tokens: 64,
            host_state_slots: 2,
            host_window_pages: 3,
            ..Default::default()
        };
        let facts = ChunkFacts::degenerate(
            &pools,
            &::engine::chunk::DegenerateBytes {
                kv_page: 100,
                window_page: 50,
                buffer_page: 25,
            },
        );
        let stores = StorePools::of(&facts).unwrap();
        assert_eq!(stores.page_size, 16);
        assert_eq!(stores.kv.device, 10);
        assert_eq!(stores.kv.slot_bytes, 100);
        assert_eq!(stores.window.device, 4);
        assert_eq!(stores.window.ids_from, 1);
        assert_eq!(stores.window_pages(), 5);
        assert_eq!(stores.window.host, 3);
        assert_eq!(stores.window_tokens, 64);
        assert_eq!(stores.state.device, 4);
        assert_eq!(stores.state.host, 2);
        assert_eq!(stores.buffer.device, 6);
    }
}
