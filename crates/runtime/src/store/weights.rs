//! The weight cache: which atoms of the streamable weights (routed experts,
//! PLE row blocks) are resident in the chunk pool.
//!
//! A streamable weight's atoms live in the pool as read-only slots of a
//! weight shape ([`::engine::chunk::WeightShape`]), held by
//! [`Holder::Weights`]. Two kinds of hold:
//!
//! - the **floor**: each shape's staging slots, what a fire cannot run
//!   without (the top-k experts of this layer and the next; this fire's
//!   rows). Pinned: live demand, never evicted.
//! - the **cache**: resident atoms above the floor. Wanted demand, the
//!   first thing to go when the pool is short.
//!
//! Residency is the runtime's, not PLEX's:
//!
//! - LRU order, kept from the engine's use record ([`WeightPort::uses`]):
//!   the atoms read in the latest window first, the most read first among
//!   them.
//! - Membership changes only at a governor tick, and only as each window
//!   of [`WINDOW_TICKS`] closes. A miss never inserts: the engine streams
//!   the atom through a staging slot and drops it; the use is only recorded.
//!   This keeps the per-token layer cycle from thrashing the cache.
//!
//! Deployment mode ([`WeightStreaming`]): `off` holds every atom pinned and
//! never evicts one, so pool pressure goes to kv and state through
//! `tier-pressure` as before. `on` sheds the cache to its floor before
//! anything else gives way: [`WeightFirst`] does it ahead of the inner
//! pool's tier-pressure, with no PLEX involvement, since weights are
//! read-only and come back from the host copy. Recovery runs the other way:
//! kv and state waiters are served first, and the cache refills only into
//! room left once none is short. Weights the pool cannot seat fail an `off`
//! load; `auto` streams them instead ([`boot_cache`]).
//!
//! The engine hears the resident set it starts with at load, before any
//! fire ([`WeightFirst::boot`]), and each change after at a governor tick
//! or a shed. Where the load seated a shape's atoms itself
//! ([`WeightShape::seated`]), the cache claims those exact slots and the
//! starting set names them there, so no atom moves; seated slots something
//! else holds fail the load.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::RwLock;

use ::engine::chunk::{
    ChunkFacts, ResidentWeight, WeightAtom, WeightResidency, WeightShape, WeightUses,
};
use serde::{Deserialize, Serialize};

use super::slots::{
    Holder, ShapeId, ShapeKind, Slot, SlotError, SlotPool, SlotPoolFacts, SlotShape,
};
use crate::governor::{Demand, Growth, PoolTarget, Retarget};

/// Whether streamable weights may be evicted from the device
/// (`[model] weight_streaming`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WeightStreaming {
    /// Off when the model's weights fit the device at load, else on.
    #[default]
    Auto,
    /// Under pressure the weight cache shrinks to its floor first.
    On,
    /// Every weight stays resident; pressure goes to kv and state.
    Off,
}

impl WeightStreaming {
    /// Whether this mode streams, for a load whose weights did or did not
    /// fit the device budget.
    #[must_use]
    pub const fn streams(self, weights_fit: bool) -> bool {
        match self {
            WeightStreaming::Auto => !weights_fit,
            WeightStreaming::On => true,
            WeightStreaming::Off => false,
        }
    }
}

static STREAMING: RwLock<BTreeMap<usize, bool>> = RwLock::new(BTreeMap::new());

/// Records the mode resolved at load for engine `engine`.
pub fn set_streaming(engine: usize, on: bool) {
    STREAMING
        .write()
        .expect("the weight streaming modes are poisoned")
        .insert(engine, on);
}

/// Whether engine `engine` streams its weights; off when never told.
#[must_use]
pub fn streaming(engine: usize) -> bool {
    STREAMING
        .read()
        .expect("the weight streaming modes are poisoned")
        .get(&engine)
        .copied()
        .unwrap_or(false)
}

/// The weight cache engine `engine` loads with, its shapes from `first` in
/// `pool` ([`pool_shapes`] of `weights`), in the mode resolved for it
/// ([`set_streaming`]): not streaming, every atom pinned; streaming, the
/// floor.
///
/// Weights that do not fit the pool: `auto` streams them instead (and
/// records it for the engine), `off` fails the load. A floor that does not
/// fit fails it in any mode: no fire could run.
///
/// # Errors
///
/// As above, naming the bytes the weights need and the pool has free.
pub fn boot_cache(
    engine: usize,
    mode: WeightStreaming,
    weights: &[WeightShape],
    first: ShapeId,
    pool: &mut SlotPool,
) -> anyhow::Result<WeightCache> {
    let needs = |all: bool| -> Vec<(ShapeId, u64)> {
        weights
            .iter()
            .enumerate()
            .map(|(at, facts)| {
                let shape = first + ShapeId::try_from(at).expect("weight shapes fit a shape id");
                let atoms = if all { facts.count } else { 0 };
                (shape, u64::from(facts.floor_slots) + u64::from(atoms))
            })
            .collect()
    };
    let floor: u64 = weights.iter().map(WeightShape::floor_bytes).sum();
    let all: u64 = weights.iter().map(WeightShape::all_bytes).sum();
    let free = u64::from(pool.free_chunks()) * pool.chunk_bytes();
    let mut streams = streaming(engine);
    if !streams && !pool.fits(&needs(true)) {
        anyhow::ensure!(
            mode != WeightStreaming::Off,
            "weight_streaming = \"off\": engine {engine}'s weights do not fit its pool: \
             every weight resident needs {} bytes ({all} of weights over {} shapes, \
             {floor} of staging), the pool has {free} bytes free ({} chunks of {} bytes); \
             set weight_streaming = \"auto\" or \"on\" to stream them",
            all.saturating_add(floor),
            weights.len(),
            pool.free_chunks(),
            pool.chunk_bytes(),
        );
        tracing::info!(
            engine,
            need_bytes = all.saturating_add(floor),
            free_bytes = free,
            floor_bytes = floor,
            "the weights do not fit the pool: they stream, the cache at its floor"
        );
        set_streaming(engine, true);
        streams = true;
    }
    anyhow::ensure!(
        pool.fits(&needs(!streams)),
        "engine {engine}'s pool cannot seat the weight staging: it needs {floor} bytes \
         over {} shapes, the pool has {free} bytes free ({} chunks of {} bytes)",
        weights.len(),
        pool.free_chunks(),
        pool.chunk_bytes(),
    );
    WeightCache::new(weights, first, streams, pool)
        .map_err(|error| anyhow::anyhow!("engine {engine}'s weight cache: {error}"))
}

/// The pool shapes the weight shapes take, in order: appended after the
/// state shapes, the first at the [`ShapeId`] given to [`WeightCache::new`].
#[must_use]
pub fn pool_shapes(weights: &[WeightShape]) -> Vec<SlotShape> {
    weights
        .iter()
        .map(|weight| SlotShape {
            // A whole record no working set holds: fixed is the nearest
            // kind, and no admission counts it.
            kind: ShapeKind::Fixed,
            slot_bytes: weight.slot_bytes,
            slots_per_chunk: weight.slots_per_chunk,
        })
        .collect()
}

/// The pool the weight shapes take their slots from until the pool switch
/// sets them beside kv and state: the device's chunks above every reserved
/// and pinned run, short of the activation at the top, in the engine's
/// chunk ids; [`pool_shapes`] of `facts.weights` from shape 0. The
/// degenerate mode's pinned runs take every chunk, so it has none.
#[must_use]
pub fn slot_pool(facts: &ChunkFacts) -> SlotPool {
    let total = facts.total_chunks.saturating_sub(facts.act_chunks_max());
    let below = facts
        .reserved
        .iter()
        .map(|run| u64::from(run.first) + u64::from(run.chunks))
        .chain(
            facts
                .shapes
                .iter()
                .filter_map(|shape| shape.pinned.map(|pinned| pinned.device.end())),
        )
        .max()
        .unwrap_or(0);
    let reserved = u32::try_from(below).unwrap_or(u32::MAX).min(total);
    SlotPool::new(SlotPoolFacts {
        chunk_bytes: facts.chunk_bytes,
        total_chunks: total,
        segment_chunks: 1,
        committed_segments: total,
        reserved_chunks: reserved,
        shapes: pool_shapes(&facts.weights),
        act_chunks: Vec::new(),
    })
}

/// Why a weight cache could not be seated at load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SeatError {
    #[error(transparent)]
    Pool(#[from] SlotError),
    /// The slots the load seated a shape in ([`WeightShape::seated`]) are
    /// not free to adopt: something else holds them. The load fails rather
    /// than seat the weights again elsewhere.
    #[error(
        "weight shape {name} ({count} atoms): the load seated it at slot refs \
         {first}..{end}, which the runtime cannot adopt: {error}"
    )]
    Seated {
        name: String,
        count: u32,
        first: u32,
        end: u64,
        error: SlotError,
    },
}

impl SeatError {
    fn seated(facts: &WeightShape, first: u32, error: SlotError) -> Self {
        SeatError::Seated {
            name: facts.name.clone(),
            count: facts.count,
            first,
            end: u64::from(first) + u64::from(facts.count),
            error,
        }
    }
}

/// Window length in governor ticks: membership changes once a window.
pub const WINDOW_TICKS: u32 = 4;

/// The engine side of the cache: reads the use record, takes the resident
/// set. Every engine answers; one that does not stream answers unsupported,
/// which the cache reads as nothing used.
pub trait WeightPort {
    fn uses(&mut self) -> ::engine::Result<WeightUses>;
    fn set_residency(&mut self, residency: &WeightResidency) -> ::engine::Result<()>;
}

impl<E: ::engine::Engine + ?Sized> WeightPort for E {
    fn uses(&mut self) -> ::engine::Result<WeightUses> {
        self.weight_uses()
    }

    fn set_residency(&mut self, residency: &WeightResidency) -> ::engine::Result<()> {
        self.weight_residency(residency)
    }
}

/// When an atom was last read, and how often in that window.
#[derive(Debug, Clone, Copy, Default)]
struct Seen {
    window: u64,
    uses: u64,
}

/// One weight shape's slots.
#[derive(Debug)]
struct Kept {
    facts: WeightShape,
    pool: ShapeId,
    staging: Vec<Slot>,
    /// Resident atoms above the floor (or, not streaming, every atom).
    resident: BTreeMap<u32, Slot>,
}

impl Kept {
    const fn holder(&self, pinned: bool) -> Holder {
        Holder::Weights {
            shape: self.pool,
            pinned,
        }
    }
}

/// See the module docs.
#[derive(Debug)]
pub struct WeightCache {
    streaming: bool,
    shapes: Vec<Kept>,
    /// The window being recorded.
    window: u64,
    ticks: u32,
    record: HashMap<WeightAtom, Seen>,
    changed: bool,
}

impl WeightCache {
    /// Takes each weight shape's floor from `pool`, whose shapes from
    /// `first` on are [`pool_shapes`] of `weights`; not streaming, every
    /// atom too, pinned.
    ///
    /// A shape the load seated itself ([`WeightShape::seated`]) keeps its
    /// atoms where they lie: those exact slots are claimed first, before
    /// any is taken lowest first. Not streaming, every one of them, pinned;
    /// streaming, as cache above the floor, whole seated chunks in atom
    /// order while every shape's floor still fits beside them.
    ///
    /// # Errors
    ///
    /// The pool cannot seat the floor (or, not streaming, the weights), or
    /// the slots a shape was seated in are not free to claim
    /// ([`SeatError::Seated`]).
    pub fn new(
        weights: &[WeightShape],
        first: ShapeId,
        streaming: bool,
        pool: &mut SlotPool,
    ) -> Result<Self, SeatError> {
        let shape_of =
            |at: usize| first + ShapeId::try_from(at).expect("weight shapes fit a shape id");
        // What the load seated itself is adopted where it lies, before any
        // slot is taken lowest first: nothing else may land in its chunks.
        // Held, every seated chunk must be free to claim, checked for all
        // shapes before any is claimed; streaming, each chunk kept is
        // checked as it is claimed.
        let mut resident: Vec<BTreeMap<u32, Slot>> = vec![BTreeMap::new(); weights.len()];
        if !streaming {
            for (at, facts) in weights.iter().enumerate() {
                let Some(seated) = facts.seated else { continue };
                pool.claimable(shape_of(at), seated, u64::from(facts.count))
                    .map_err(|error| SeatError::seated(facts, seated, error))?;
            }
        }
        // Streaming, the seated atoms stay as cache while every floor still
        // fits beside them, whole chunks in atom order.
        let floor_chunks: u64 = weights
            .iter()
            .map(|facts| {
                u64::from(facts.floor_slots).div_ceil(u64::from(facts.slots_per_chunk.max(1)))
            })
            .sum();
        for (at, facts) in weights.iter().enumerate() {
            let Some(seated) = facts.seated else { continue };
            let shape = shape_of(at);
            let spc = facts.slots_per_chunk.max(1);
            let atoms: Vec<(u32, u32)> = if streaming {
                (0..facts.count)
                    .step_by(spc as usize)
                    .map(|atom| (atom, spc.min(facts.count - atom)))
                    .collect()
            } else {
                vec![(0, facts.count)]
            };
            for (from, n) in atoms {
                if streaming && u64::from(pool.free_chunks()) <= floor_chunks {
                    break;
                }
                let holder = Holder::Weights {
                    shape,
                    pinned: !streaming,
                };
                let slots = pool
                    .claim(holder, shape, seated + from, u64::from(n))
                    .map_err(|error| SeatError::seated(facts, seated, error))?;
                resident[at].extend((from..from + n).zip(slots));
            }
        }
        let mut shapes = Vec::with_capacity(weights.len());
        for ((at, facts), mut resident) in weights.iter().enumerate().zip(resident) {
            let shape = shape_of(at);
            let holder = Holder::Weights {
                shape,
                pinned: true,
            };
            let staging = pool.acquire(holder, shape, u64::from(facts.floor_slots))?;
            if !streaming && facts.seated.is_none() {
                resident = (0..facts.count)
                    .zip(pool.acquire(holder, shape, u64::from(facts.count))?)
                    .collect();
            }
            shapes.push(Kept {
                facts: facts.clone(),
                pool: shape,
                staging,
                resident,
            });
        }
        Ok(Self {
            streaming,
            shapes,
            window: 1,
            ticks: 0,
            record: HashMap::new(),
            changed: true,
        })
    }

    #[must_use]
    pub const fn streaming(&self) -> bool {
        self.streaming
    }

    /// The weight shapes it holds.
    #[must_use]
    pub fn shapes(&self) -> usize {
        self.shapes.len()
    }

    /// Whether `shape` is one of the pool shapes the weights take.
    #[must_use]
    pub fn holds_shape(&self, shape: ShapeId) -> bool {
        self.shapes.iter().any(|kept| kept.pool == shape)
    }

    /// The bytes held whatever the pressure: the floor, and not streaming
    /// every atom.
    #[must_use]
    pub fn pinned_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|kept| {
                let floor = kept.facts.floor_bytes();
                if self.streaming {
                    floor
                } else {
                    floor.saturating_add(kept.facts.all_bytes())
                }
            })
            .sum()
    }

    /// Resident atoms above the floor.
    #[must_use]
    pub fn cached(&self) -> usize {
        if self.streaming {
            self.shapes.iter().map(|kept| kept.resident.len()).sum()
        } else {
            0
        }
    }

    /// The slot `atom` is resident in.
    #[must_use]
    pub fn resident(&self, atom: WeightAtom) -> Option<Slot> {
        self.kept(atom.shape)?.resident.get(&atom.atom).copied()
    }

    /// The bytes of atoms read in the latest whole window and not resident:
    /// what the cache would take with room, wanted demand.
    #[must_use]
    pub fn unmet_bytes(&self) -> u64 {
        if !self.streaming {
            return 0;
        }
        let last = self.window.saturating_sub(1);
        self.record
            .iter()
            .filter(|(atom, seen)| seen.window >= last && self.resident(**atom).is_none())
            .filter_map(|(atom, _)| self.kept(atom.shape).map(|kept| kept.facts.slot_bytes))
            .sum()
    }

    fn kept(&self, shape: ::engine::chunk::ShapeId) -> Option<&Kept> {
        self.shapes.iter().find(|kept| kept.facts.id == shape)
    }

    /// Adds the engine's reads to the record. Membership does not change.
    pub fn record(&mut self, uses: &WeightUses) {
        for &(atom, reads) in &uses.uses {
            let known = self
                .kept(atom.shape)
                .is_some_and(|kept| atom.atom < kept.facts.count);
            if !known || reads == 0 {
                continue;
            }
            let seen = self.record.entry(atom).or_default();
            if seen.window != self.window {
                *seen = Seen {
                    window: self.window,
                    uses: 0,
                };
            }
            seen.uses = seen.uses.saturating_add(u64::from(reads));
        }
    }

    /// Lets every resident atom above the floor go, the fire at `epoch`
    /// the last that may read them; the slots it gives back. Nothing while
    /// not streaming.
    pub fn shed(&mut self, pool: &mut SlotPool, epoch: u64) -> u64 {
        if !self.streaming {
            return 0;
        }
        let mut shed = 0;
        for kept in &mut self.shapes {
            if kept.resident.is_empty() {
                continue;
            }
            let slots: Vec<Slot> = std::mem::take(&mut kept.resident).into_values().collect();
            pool.release(kept.holder(false), &slots, epoch);
            shed += slots.len() as u64;
        }
        self.changed |= shed > 0;
        shed
    }

    /// One governor tick. When it closes a window, each shape keeps the
    /// atoms the record ranks first, as many as it holds now, plus, when
    /// `refill`, as many more as the pool has room for; atoms are taken in
    /// only from the window just closed. A swapped-out atom's slot goes to
    /// the atom that displaces it.
    pub fn tick(&mut self, pool: &mut SlotPool, epoch: u64, refill: bool) {
        self.ticks += 1;
        if self.ticks < WINDOW_TICKS {
            return;
        }
        self.ticks = 0;
        if self.streaming {
            for at in 0..self.shapes.len() {
                self.settle(at, pool, epoch, refill);
            }
        }
        self.window += 1;
    }

    fn settle(&mut self, at: usize, pool: &mut SlotPool, epoch: u64, refill: bool) {
        let window = self.window;
        let kept = &self.shapes[at];
        let id = kept.facts.id;
        let mut ranked: Vec<(u32, Seen)> = self
            .record
            .iter()
            .filter(|(atom, seen)| {
                atom.shape == id
                    && (seen.window == window || kept.resident.contains_key(&atom.atom))
            })
            .map(|(atom, seen)| (atom.atom, *seen))
            .collect();
        // Most recent window first, the most read first within it.
        ranked
            .sort_unstable_by(|a, b| (b.1.window, b.1.uses, a.0).cmp(&(a.1.window, a.1.uses, b.0)));
        let room = if refill {
            usize::try_from(pool.available(kept.pool)).unwrap_or(usize::MAX)
        } else {
            0
        };
        let capacity = kept.resident.len().saturating_add(room);
        let keep: BTreeSet<u32> = ranked
            .iter()
            .take(capacity)
            .map(|&(atom, _)| atom)
            .collect();
        let kept = &mut self.shapes[at];
        let out: Vec<u32> = kept
            .resident
            .keys()
            .filter(|atom| !keep.contains(atom))
            .copied()
            .collect();
        let incoming: Vec<u32> = keep
            .iter()
            .filter(|atom| !kept.resident.contains_key(atom))
            .copied()
            .collect();
        let mut freed: Vec<Slot> = out
            .iter()
            .filter_map(|atom| kept.resident.remove(atom))
            .collect();
        let mut fresh = Vec::new();
        let reused = freed.len().min(incoming.len());
        let more = incoming.len() - reused;
        if more > 0 {
            let more = (more as u64).min(pool.available(kept.pool));
            if let Ok(slots) = pool.acquire(kept.holder(false), kept.pool, more) {
                fresh = slots;
            }
        }
        let mut slots = freed.drain(..reused).chain(fresh);
        for &atom in &incoming {
            let Some(slot) = slots.next() else { break };
            kept.resident.insert(atom, slot);
        }
        drop(slots);
        if !freed.is_empty() {
            pool.release(kept.holder(false), &freed, epoch);
        }
        self.changed |= !out.is_empty() || !incoming.is_empty();
    }

    /// The whole resident set, as the engine takes it.
    #[must_use]
    pub fn residency(&self, pool: &SlotPool) -> WeightResidency {
        // Not streaming, every atom is pinned in its slot for good.
        let mut residency = WeightResidency {
            held: !self.streaming,
            ..WeightResidency::default()
        };
        for kept in &self.shapes {
            residency.staging.push((
                kept.facts.id,
                kept.staging
                    .iter()
                    .map(|&slot| pool.slot_ref(slot))
                    .collect(),
            ));
            residency
                .resident
                .extend(kept.resident.iter().map(|(&atom, &slot)| ResidentWeight {
                    atom: WeightAtom {
                        shape: kept.facts.id,
                        atom,
                    },
                    slot: pool.slot_ref(slot),
                }));
        }
        residency
    }

    /// The resident set, when it changed since this last answered.
    pub fn take_changed(&mut self, pool: &SlotPool) -> Option<WeightResidency> {
        std::mem::take(&mut self.changed).then(|| self.residency(pool))
    }
}

/// Whether a kv or state waiter (any shape but the weights') is short.
fn kv_short(pool: &SlotPool, cache: &WeightCache) -> bool {
    (0..=ShapeId::MAX)
        .map_while(|shape| pool.shape(shape).map(|_| shape))
        .any(|shape| !cache.holds_shape(shape) && pool.shape_demand(shape).waiting > 0)
}

/// What [`WeightFirst`] needs of the pool it wraps: its slots, and the
/// epoch of the last fire submitted (the last that may read a slot let go
/// now).
pub trait SlotAccess {
    fn with_slots(&mut self, f: &mut dyn FnMut(&mut SlotPool, u64));
}

/// A pool target with a weight cache in it, shed first.
///
/// - Demand: the floor is live (its holder is), resident atoms above it
///   are wanted (theirs is not), and atoms read but not resident are
///   wanted too.
/// - A shrink the inner pool has no room for sheds the cache to its floor
///   and tries again; only with the cache at its floor does a shortfall
///   come back as [`Retarget::NoRoom`], the governor's `tier-pressure`.
/// - At each tick: a kv or state waiter short sheds the cache; with none
///   short the cache settles its window and may refill into what room is
///   left. The engine hears the resident set whenever it changed.
///
/// Not streaming, it is the inner pool with its weights pinned.
pub struct WeightFirst<P, W> {
    inner: P,
    cache: WeightCache,
    port: W,
    /// The cache was shed since the last tick: it does not refill on it.
    shed: bool,
}

impl<P: PoolTarget + SlotAccess, W: WeightPort> WeightFirst<P, W> {
    pub const fn new(inner: P, cache: WeightCache, port: W) -> Self {
        Self {
            inner,
            cache,
            port,
            shed: false,
        }
    }

    pub const fn inner(&self) -> &P {
        &self.inner
    }

    pub const fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    pub const fn cache(&self) -> &WeightCache {
        &self.cache
    }

    pub const fn port(&self) -> &W {
        &self.port
    }

    pub const fn port_mut(&mut self) -> &mut W {
        &mut self.port
    }

    /// Sheds the cache to its floor; the slots given back. What a planner
    /// calls before it raises `tier-pressure` for a waiter, so weights go
    /// before any kv or state does.
    pub fn shed(&mut self) -> u64 {
        let mut shed = 0;
        let cache = &mut self.cache;
        self.inner.with_slots(&mut |pool, epoch| {
            shed = cache.shed(pool, epoch);
        });
        // Asked at all, a waiter is short: the next tick does not refill,
        // whether or not anything was above the floor.
        self.shed |= self.cache.streaming();
        if shed > 0 {
            self.publish();
        }
        shed
    }

    /// Tells the engine the resident set it starts with: what the load
    /// calls before any fire, so none reads a row the cache holds from the
    /// host.
    ///
    /// # Errors
    ///
    /// The engine refused it. One that does not stream (it answers
    /// unsupported) is no error: it reads its weights as it loaded them.
    pub fn boot(&mut self) -> ::engine::Result<()> {
        let mut residency = None;
        let cache = &mut self.cache;
        self.inner.with_slots(&mut |pool, _| {
            residency = cache.take_changed(pool);
        });
        match residency.map(|residency| self.port.set_residency(&residency)) {
            Some(Err(::engine::Error::Unsupported { .. })) | Some(Ok(())) | None => Ok(()),
            Some(Err(error)) => Err(error),
        }
    }

    fn publish(&mut self) {
        let mut residency = None;
        let cache = &mut self.cache;
        self.inner.with_slots(&mut |pool, _| {
            residency = cache.take_changed(pool);
        });
        if let Some(residency) = residency
            && let Err(error) = self.port.set_residency(&residency)
        {
            tracing::warn!(%error, "the engine did not take the weight residency");
        }
    }
}

impl<P: PoolTarget + SlotAccess, W: WeightPort> PoolTarget for WeightFirst<P, W> {
    fn demand(&mut self) -> Demand {
        let demand = self.inner.demand();
        Demand {
            wanted: demand.wanted.saturating_add(self.cache.unmet_bytes()),
            live: demand.live,
        }
    }

    fn segment_bytes(&self) -> u64 {
        self.inner.segment_bytes()
    }

    /// The inner pool's floor for kv, state and activation, and the
    /// weights' pinned bytes beside it.
    fn floor(&self) -> u64 {
        self.inner
            .floor()
            .saturating_add(self.cache.pinned_bytes())
            .min(self.inner.hard_limit())
    }

    fn hard_limit(&self) -> u64 {
        self.inner.hard_limit()
    }

    fn held(&self) -> u64 {
        self.inner.held()
    }

    fn growth(&self) -> Growth {
        self.inner.growth()
    }

    fn retarget(&mut self, segments: u64) -> Retarget {
        let first = self.inner.retarget(segments);
        if !matches!(first, Retarget::NoRoom { .. }) || !self.cache.streaming() {
            return first;
        }
        if self.shed() == 0 {
            return first;
        }
        match self.inner.retarget(segments) {
            // What was shed comes back as the fires reading it retire: the
            // pool is on its way, and the next tick tries again with the
            // cache at its floor.
            Retarget::NoRoom { .. } => Retarget::Reached,
            reached => reached,
        }
    }

    fn take_refused(&mut self) -> bool {
        self.inner.take_refused()
    }

    fn shrink_rung(&mut self) {
        self.inner.shrink_rung();
    }

    fn restore_rung(&mut self) {
        self.inner.restore_rung();
    }

    fn ticked(&mut self) {
        match self.port.uses() {
            Ok(uses) => self.cache.record(&uses),
            Err(error) => tracing::trace!(%error, "no weight use record"),
        }
        let mut short = false;
        let cache = &mut self.cache;
        let shed = self.shed;
        self.inner.with_slots(&mut |pool, epoch| {
            short = kv_short(pool, cache);
            if short {
                cache.shed(pool, epoch);
            }
            cache.tick(pool, epoch, !short && !shed);
        });
        self.shed = false;
        self.publish();
        self.inner.ticked();
    }
}

pub mod port;

#[cfg(test)]
mod tests;
