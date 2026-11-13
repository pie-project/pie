//! The weight cache: which atoms of the streamable weights (routed experts,
//! PLE row blocks) are resident in the chunk pool.
//!
//! A streamable weight's atoms live in the pool as read-only slots of a
//! weight shape ([`::engine::chunk::WeightShape`]), held by
//! [`Holder::Weights`]. A weight atom is a clean page: read-only, with a
//! copy on the host, so dropping one loses no work, only future misses.
//! Residency is the runtime's, by three rules with no mode and no threshold
//! (REDESIGN 2.7a; rules 1 and 3 here, rule 2 waits on measurement):
//!
//! 1. **Live demand before cache.** When every atom fits the pool at load,
//!    every atom is resident and pinned (a *held* set: nothing misses,
//!    nothing streams through a staging slot). When live demand needs the
//!    chunks (a kv or state waiter short, or a governor cut with no room),
//!    the cache leaves held: each shape keeps its staging slots (the
//!    **floor**: the top-k experts of this layer and the next, this fire's
//!    rows; pinned, live demand, never evicted) and gives the rest up. The
//!    engine is asked first and the slots move only once it took the
//!    streaming set ([`WeightPort::propose`]); one that refuses keeps the
//!    set held, and the pressure goes to kv and state as it always did.
//!    Weights that do not fit at load stream from the start.
//! 2. (Not yet: among cache entries, the lowest value per byte goes first.
//!    Until it is measured, the cache goes to its floor before kv or state
//!    gives way, [`WeightFirst`].)
//! 3. **An atom comes back at break-even**: once the bytes its misses have
//!    streamed since it left reach the bytes loading it takes (the
//!    ski-rental rule; with a miss streaming the whole atom, as every
//!    engine's does, that is its first miss). The atoms that have cost most
//!    come back first, into
//!    room no waiter is short of, else in place of a resident atom no fire
//!    read for a whole governor tick. A miss never inserts on the fire
//!    path: the engine streams the atom through a staging slot and the use
//!    is only recorded ([`WeightPort::uses`]).
//!
//! The engine hears the resident set it starts with at load, before any
//! fire ([`WeightFirst::boot`]), and each change after at a governor tick
//! or a shed. Where the load seated a shape's atoms itself
//! ([`WeightShape::seated`]), the cache claims those exact slots and the
//! starting set names them there, so no atom moves; seated slots something
//! else holds fail the load.

use std::collections::{BTreeMap, HashMap};

use super::slots::{
    Holder, ShapeId, ShapeKind, Slot, SlotError, SlotPool, SlotPoolFacts, SlotShape,
};
use crate::governor::{Demand, Growth, PoolTarget, Retarget};
use ::engine::chunk::{
    ChunkFacts, ResidentWeight, WeightAtom, WeightResidency, WeightShape, WeightUses,
};

/// The weight cache engine `engine` loads with, its shapes from `first` in
/// `pool` ([`pool_shapes`] of `weights`): every atom pinned where the pool
/// seats them all (rule 1), else the floor, streaming.
///
/// # Errors
///
/// A floor that does not fit fails the load: no fire could run. The error
/// names the bytes the weights need and the pool has free.
pub fn boot_cache(
    engine: usize,
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
                // Held, every atom and no staging; streaming, the floor.
                let slots = if all { facts.count } else { facts.floor_slots };
                (shape, u64::from(slots))
            })
            .collect()
    };
    let floor: u64 = weights.iter().map(WeightShape::floor_bytes).sum();
    let all: u64 = weights.iter().map(WeightShape::all_bytes).sum();
    // What the pool can grow to, not what the load committed: the weights
    // commit the segments they take (`WeightCache::new`).
    let free = u64::from(pool.growable_chunks()) * pool.chunk_bytes();
    let streams = !pool.fits_grown(&needs(true));
    if streams && !weights.is_empty() {
        tracing::info!(
            engine,
            need_bytes = all,
            free_bytes = free,
            floor_bytes = floor,
            "the weights do not fit the pool: they stream, the cache at its floor"
        );
    }
    anyhow::ensure!(
        pool.fits_grown(&needs(!streams)),
        "engine {engine}'s pool cannot seat the weight staging: it needs {floor} bytes \
         over {} shapes, the pool has {free} bytes free ({} chunks of {} bytes)",
        weights.len(),
        pool.growable_chunks(),
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

/// The engine side of the cache: reads the use record, takes the resident
/// set. Every engine answers; one that does not stream answers unsupported,
/// which the cache reads as nothing used.
pub trait WeightPort {
    fn uses(&mut self) -> ::engine::Result<WeightUses>;
    fn set_residency(&mut self, residency: &WeightResidency) -> ::engine::Result<()>;
    /// Offers a resident set the engine may refuse without harm: the one a
    /// held set leaves for (rule 1). Nothing it names moves until the
    /// engine took it.
    fn propose(&mut self, residency: &WeightResidency) -> Proposal;
}

/// What became of a [`WeightPort::propose`].
pub enum Proposal {
    Taken,
    Refused,
    /// Queued on the engine's lane; its answer comes later.
    Pending(port::Answer),
}

impl<E: ::engine::Engine + ?Sized> WeightPort for E {
    fn uses(&mut self) -> ::engine::Result<WeightUses> {
        self.weight_uses()
    }

    fn set_residency(&mut self, residency: &WeightResidency) -> ::engine::Result<()> {
        self.weight_residency(residency)
    }

    fn propose(&mut self, residency: &WeightResidency) -> Proposal {
        match self.weight_residency(residency) {
            Ok(()) => Proposal::Taken,
            Err(_) => Proposal::Refused,
        }
    }
}

/// What the use record says of one atom.
#[derive(Debug, Clone, Copy, Default)]
struct Seen {
    /// The tick it was last read in.
    tick: u64,
    /// Not resident: the bytes its misses have streamed since it left.
    owed: u64,
}

/// One weight shape's slots.
#[derive(Debug)]
struct Kept {
    facts: WeightShape,
    pool: ShapeId,
    staging: Vec<Slot>,
    /// Resident atoms above the floor (or, held, every atom).
    resident: BTreeMap<u32, Slot>,
}

/// A held set leaving for a streaming one (rule 1): the engine was offered
/// the set in which each shape's `staging` slots, taken from resident
/// atoms, stage what misses, and holds its answer.
struct Leaving {
    staging: Vec<Vec<u32>>,
    answer: port::Answer,
}

impl std::fmt::Debug for Leaving {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Leaving")
            .field("staging", &self.staging)
            .finish_non_exhaustive()
    }
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
    /// Held, the streaming set the engine was offered and has not answered.
    leaving: Option<Leaving>,
    /// Held, the engine refused to leave: the set stays held for the load.
    stays_held: bool,
    shapes: Vec<Kept>,
    /// The governor ticks seen.
    tick: u64,
    record: HashMap<WeightAtom, Seen>,
    changed: bool,
    /// The committed prefix the engine must back before the weights land:
    /// the segments [`Self::new`] committed past what the load committed,
    /// taken by the load ([`Self::take_backing`]).
    backing: Option<u32>,
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
        // The pool may hold only part of its segments committed at load
        // (one, on an engine that grows from the bottom): commit, as a
        // prefix, what the weights take before claiming it. Held, every
        // seated range and every other shape's atoms; streaming, the floor
        // (seated atoms are kept as cache only where room stands).
        let needs: Vec<(ShapeId, u64)> = weights
            .iter()
            .enumerate()
            .map(|(at, facts)| {
                let slots = match (streaming, facts.seated) {
                    (true, _) => facts.floor_slots,
                    (false, Some(_)) => 0,
                    (false, None) => facts.count,
                };
                (shape_of(at), u64::from(slots))
            })
            .collect();
        let through = if streaming {
            0
        } else {
            weights
                .iter()
                .filter_map(|facts| {
                    let spc = facts.slots_per_chunk.max(1);
                    facts
                        .seated
                        .map(|seated| seated / spc + facts.count.div_ceil(spc))
                })
                .max()
                .unwrap_or(0)
        };
        let backing =
            (!pool.commit_for(&needs, through).is_empty()).then(|| pool.prefix_segments());
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
                if streaming && u64::from(pool.growable_chunks()) <= floor_chunks {
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
            // A held set has every atom resident: nothing misses, so
            // nothing streams through a staging slot and none is taken.
            let staging = if streaming {
                pool.acquire(holder, shape, u64::from(facts.floor_slots))?
            } else {
                Vec::new()
            };
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
            leaving: None,
            stays_held: false,
            shapes,
            tick: 1,
            record: HashMap::new(),
            changed: true,
            backing,
        })
    }

    /// The committed prefix the engine must back before it is told the
    /// residency (`commit_segments`), once: `None` where the load's own
    /// commit held the weights.
    pub fn take_backing(&mut self) -> Option<u32> {
        self.backing.take()
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

    /// The bytes held whatever the pressure: streaming, the floor; not
    /// streaming, every atom (and no staging, which nothing streams
    /// through).
    #[must_use]
    pub fn pinned_bytes(&self) -> u64 {
        self.shapes
            .iter()
            .map(|kept| {
                if self.streaming {
                    kept.facts.floor_bytes()
                } else {
                    kept.facts.all_bytes()
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

    /// The bytes of atoms past break-even and not resident: what the cache
    /// would take with room, wanted demand.
    #[must_use]
    pub fn unmet_bytes(&self) -> u64 {
        if !self.streaming {
            return 0;
        }
        self.record
            .iter()
            .filter(|(atom, _)| self.resident(**atom).is_none())
            .filter_map(|(atom, seen)| {
                let bytes = self.kept(atom.shape)?.facts.slot_bytes;
                (seen.owed >= bytes).then_some(bytes)
            })
            .sum()
    }

    /// The bytes of resident atoms above the floor: the model cache, what
    /// live demand may take.
    #[must_use]
    pub fn cache_bytes(&self) -> u64 {
        if !self.streaming {
            return 0;
        }
        self.shapes
            .iter()
            .map(|kept| kept.facts.slot_bytes * kept.resident.len() as u64)
            .sum()
    }

    fn kept(&self, shape: ::engine::chunk::ShapeId) -> Option<&Kept> {
        self.shapes.iter().find(|kept| kept.facts.id == shape)
    }

    /// Adds the engine's reads to the record: a resident atom's last read,
    /// and what a missed one has streamed since it left. Membership does
    /// not change.
    pub fn record(&mut self, uses: &WeightUses) {
        for &(atom, reads) in &uses.uses {
            let Some(kept) = self.kept(atom.shape) else {
                continue;
            };
            if atom.atom >= kept.facts.count || reads == 0 {
                continue;
            }
            let bytes = kept.facts.slot_bytes;
            let missed = !kept.resident.contains_key(&atom.atom);
            let seen = self.record.entry(atom).or_default();
            seen.tick = self.tick;
            if missed {
                seen.owed = seen
                    .owed
                    .saturating_add(u64::from(reads).saturating_mul(bytes));
            }
        }
    }

    /// `atoms` of the shape at `at` left the cache: what they owe starts
    /// again from nothing.
    fn left(&mut self, at: usize, atoms: impl IntoIterator<Item = u32>) {
        let shape = self.shapes[at].facts.id;
        for atom in atoms {
            if let Some(seen) = self.record.get_mut(&WeightAtom { shape, atom }) {
                seen.owed = 0;
            }
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
        for at in 0..self.shapes.len() {
            let kept = &mut self.shapes[at];
            if kept.resident.is_empty() {
                continue;
            }
            let gone = std::mem::take(&mut kept.resident);
            let slots: Vec<Slot> = gone.values().copied().collect();
            pool.release(kept.holder(false), &slots, epoch);
            shed += slots.len() as u64;
            self.left(at, gone.into_keys());
        }
        self.changed |= shed > 0;
        shed
    }

    /// Held, the streaming set live demand asks for (rule 1): each shape
    /// stages through `floor_slots` of the slots its highest atoms hold,
    /// and every other atom stays resident until a shed lets it go. `None`
    /// once it streams, while an offer waits, or once the engine refused.
    fn leaving_set(&self, pool: &SlotPool) -> Option<(Vec<Vec<u32>>, WeightResidency)> {
        if self.streaming || self.leaving.is_some() || self.stays_held || self.shapes.is_empty() {
            return None;
        }
        let staging: Vec<Vec<u32>> = self
            .shapes
            .iter()
            .map(|kept| {
                kept.resident
                    .keys()
                    .rev()
                    .take(kept.facts.floor_slots as usize)
                    .copied()
                    .collect()
            })
            .collect();
        let mut residency = WeightResidency::default();
        for (kept, staged) in self.shapes.iter().zip(&staging) {
            residency.staging.push((
                kept.facts.id,
                staged
                    .iter()
                    .map(|atom| pool.slot_ref(kept.resident[atom]))
                    .collect(),
            ));
            residency.resident.extend(
                kept.resident
                    .iter()
                    .filter(|(atom, _)| !staged.contains(atom))
                    .map(|(&atom, &slot)| ResidentWeight {
                        atom: WeightAtom {
                            shape: kept.facts.id,
                            atom,
                        },
                        slot: pool.slot_ref(slot),
                    }),
            );
        }
        Some((staging, residency))
    }

    /// The engine took the streaming set offered: the staging slots stay
    /// pinned (the floor), every other atom becomes cache above it.
    fn left_held(&mut self, staging: &[Vec<u32>], pool: &mut SlotPool) {
        for (at, staged) in staging.iter().enumerate() {
            let kept = &mut self.shapes[at];
            kept.staging = staged
                .iter()
                .filter_map(|atom| kept.resident.remove(atom))
                .collect();
            let cached: Vec<Slot> = kept.resident.values().copied().collect();
            pool.share(kept.holder(false), &cached);
            pool.release(kept.holder(true), &cached, 0);
            let gone: Vec<u32> = staged.clone();
            self.left(at, gone);
        }
        self.streaming = true;
    }

    /// One governor tick: each shape takes back the atoms past break-even
    /// (rule 3), most owed first, into free room when `refill`, else in
    /// place of resident atoms no fire read for a whole tick.
    pub fn tick(&mut self, pool: &mut SlotPool, epoch: u64, refill: bool) {
        if self.streaming {
            for at in 0..self.shapes.len() {
                self.settle(at, pool, epoch, refill);
            }
        }
        self.tick += 1;
    }

    fn settle(&mut self, at: usize, pool: &mut SlotPool, epoch: u64, refill: bool) {
        let kept = &self.shapes[at];
        let (id, bytes) = (kept.facts.id, kept.facts.slot_bytes);
        // Past break-even and not resident, most owed first.
        let mut back: Vec<(u32, u64)> = self
            .record
            .iter()
            .filter(|(atom, seen)| {
                atom.shape == id && seen.owed >= bytes && !kept.resident.contains_key(&atom.atom)
            })
            .map(|(atom, seen)| (atom.atom, seen.owed))
            .collect();
        if back.is_empty() {
            return;
        }
        back.sort_unstable_by(|a, b| (b.1, a.0).cmp(&(a.1, b.0)));
        // Resident atoms no fire read in the tick now closing, longest
        // unread first: what a returning atom may take the place of.
        let last = self.tick;
        let mut idle: Vec<(u32, u64)> = kept
            .resident
            .keys()
            .map(|&atom| {
                let read = self
                    .record
                    .get(&WeightAtom { shape: id, atom })
                    .map_or(0, |seen| seen.tick);
                (atom, read)
            })
            .filter(|&(_, read)| read < last)
            .collect();
        idle.sort_unstable_by(|a, b| (a.1, a.0).cmp(&(b.1, b.0)));
        let room = if refill {
            usize::try_from(pool.available(kept.pool)).unwrap_or(usize::MAX)
        } else {
            0
        };
        let fresh_n = back.len().min(room);
        let swapped = (back.len() - fresh_n).min(idle.len());
        let mut fresh = Vec::new();
        if fresh_n > 0
            && let Ok(slots) = pool.acquire(kept.holder(false), kept.pool, fresh_n as u64)
        {
            fresh = slots;
        }
        let kept = &mut self.shapes[at];
        let out: Vec<u32> = idle[..swapped].iter().map(|&(atom, _)| atom).collect();
        let mut slots: Vec<Slot> = fresh
            .into_iter()
            .chain(out.iter().filter_map(|atom| kept.resident.remove(atom)))
            .collect();
        let mut incoming = Vec::new();
        for &(atom, _) in &back {
            let Some(slot) = slots.pop() else { break };
            kept.resident.insert(atom, slot);
            incoming.push(atom);
        }
        // A slot nothing came back into (an acquire short of what it
        // asked) goes back.
        if !slots.is_empty() {
            pool.release(kept.holder(false), &slots, epoch);
        }
        self.changed |= !out.is_empty() || !incoming.is_empty();
        self.left(at, out);
        self.left(at, incoming);
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
            // Held, nothing streams: the staging list is empty.
            if self.streaming {
                residency.staging.push((
                    kept.facts.id,
                    kept.staging
                        .iter()
                        .map(|&slot| pool.slot_ref(slot))
                        .collect(),
                ));
            }
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
///   short the atoms past break-even come back into what room is left. The
///   engine hears the resident set whenever it changed.
///
/// Held, it is the inner pool with its weights pinned, until live demand
/// asks for their chunks: a shed then offers the engine the streaming set
/// (rule 1), and sheds for real once the engine took it.
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

    pub const fn cache(&self) -> &WeightCache {
        &self.cache
    }

    /// See [`WeightCache::take_backing`].
    pub fn take_backing(&mut self) -> Option<u32> {
        self.cache.take_backing()
    }

    pub const fn port(&self) -> &W {
        &self.port
    }

    pub const fn port_mut(&mut self) -> &mut W {
        &mut self.port
    }

    /// Sheds the cache to its floor; the slots given back. What a planner
    /// calls before it raises `tier-pressure` for a waiter, so weights go
    /// before any kv or state does. Held, it offers the engine the
    /// streaming set and gives nothing back until the engine took it.
    pub fn shed(&mut self) -> u64 {
        if !self.cache.streaming() {
            self.leave_held();
            return 0;
        }
        let mut shed = 0;
        let cache = &mut self.cache;
        self.inner.with_slots(&mut |pool, epoch| {
            shed = cache.shed(pool, epoch);
        });
        // Asked at all, a waiter is short: the next tick does not refill,
        // whether or not anything was above the floor.
        self.shed = true;
        if shed > 0 {
            self.publish();
        }
        shed
    }

    /// Held, offers the engine the streaming set live demand asks for
    /// (rule 1); an engine that answers at once is heard at once.
    fn leave_held(&mut self) {
        let mut offer = None;
        let cache = &self.cache;
        self.inner.with_slots(&mut |pool, _| {
            offer = cache.leaving_set(pool);
        });
        let Some((staging, residency)) = offer else {
            return;
        };
        match self.port.propose(&residency) {
            Proposal::Taken => self.left_held(&staging),
            Proposal::Refused => self.stay_held(),
            Proposal::Pending(answer) => {
                self.cache.leaving = Some(Leaving { staging, answer });
            }
        }
    }

    /// Reads the engine's answer to the streaming set offered, once it
    /// came.
    fn heard(&mut self) {
        let Some(mut leaving) = self.cache.leaving.take() else {
            return;
        };
        match leaving.answer.try_recv() {
            Ok(Ok(_)) => self.left_held(&leaving.staging),
            Ok(Err(error)) => {
                tracing::info!(%error, "the engine keeps its weights held");
                self.stay_held();
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                self.cache.leaving = Some(leaving);
            }
            Err(tokio::sync::oneshot::error::TryRecvError::Closed) => self.stay_held(),
        }
    }

    fn left_held(&mut self, staging: &[Vec<u32>]) {
        let cache = &mut self.cache;
        self.inner
            .with_slots(&mut |pool, _| cache.left_held(staging, pool));
        tracing::info!("live demand took the weight cache off held: the weights stream");
        // The cache goes to its floor now that it may.
        self.shed();
    }

    fn stay_held(&mut self) {
        self.cache.stays_held = true;
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
        if !matches!(first, Retarget::NoRoom { .. }) {
            return first;
        }
        if !self.cache.streaming() {
            self.leave_held();
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

    fn ticked(&mut self) {
        self.heard();
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
        if short && !self.cache.streaming() {
            self.leave_held();
        }
        self.shed = false;
        self.publish();
        self.inner.ticked();
    }
}

pub mod port;

#[cfg(test)]
mod tests;
