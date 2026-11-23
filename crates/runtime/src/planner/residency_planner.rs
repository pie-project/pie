use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::departure::Departure;
use super::grant::{AllocationGrant, Demand, DevicePageReservation, RoomReturns, RsSlotReservation};
use super::holdings::{held_page_count, held_rs_units};
use super::inner::Inner;
use super::pool::{PoolPort, RunNeed};
use super::proc::{PendingBind, Proc, Residency};
use super::rank::Rank;
use super::reclaim::Reclaim;
use super::short::{Shortfall, Sink, WeightShed};
use super::stats::PlannerStats;
use super::wait::{EntryKey, WaitKind, Waiter};
use super::{ProcessId, exec};

/// The fixed state sets a working set holds, resting or running
/// (`state-chunk` decision 4). The copy-on-write set a lane's write takes
/// while a fork or snapshot still reads the first is no part of it: the
/// fire that writes it asks for it with the rest of its demand, and once
/// held it counts against the seats ([`ResidencyPlanner::state_seats`]).
pub(super) const FIXED_SETS_PER_WORKING_SET: u32 = 1;

/// It suspends no process of its own.
pub(super) const SWEEP: ProcessId = ProcessId::nil();
/// The key a demote's pages are expected under while it moves: they come
/// back once its copy has landed, and a shortfall read meanwhile counts them
/// arriving instead of evicting another process for room on its way.
const DEMOTE: ProcessId = ProcessId::max();

pub struct ResidencyPlanner {
    pub(super) inner: parking_lot::Mutex<Inner>,
    /// The head's need read this pass ([`Self::head_room`]), by the pass,
    /// the head and the pages gathered: the step, the map, the wants and
    /// the stall line read it once a pass, not once each.
    pub(super) need_cache: parking_lot::Mutex<Option<(u64, EntryKey, u32, Option<RunNeed>)>>,
    /// Events that can change a pass's answer the pool's own counts do not
    /// show: an arrival, a departure, a landing, a policy answer, each a
    /// poke ([`Self::poke`]). Part of [`Self::generation`].
    pub(super) events: AtomicU64,
    /// The last pass settled with a waiting head and nothing in flight
    /// (the progress invariant broken, [`Self::dump_stall`]): the next
    /// pass plans the head's need as if its placement did not stand, in
    /// chunks only moves empty, never a wait.
    pub(super) force_moves: AtomicBool,
    /// Fires landed as the running pass began: a wait it chose on
    /// something in flight is no quiet wait where one landed meanwhile.
    pub(super) pass_landed: AtomicU64,
    /// The generation the last pass ended on where it took no step (it
    /// ended where it began): a wake on that generation plans nothing new,
    /// and is skipped, counted ([`super::stats::PlannerStats::skipped_wakes`]).
    pub(super) settled_at: parking_lot::Mutex<Option<u64>>,
    /// The occupancy map last built, by the generation and the candidates
    /// it was built for: the same pool and candidates read the same map.
    pub(super) map_cache: parking_lot::Mutex<Option<(u64, Vec<ProcessId>, super::pool::ChunkMap)>>,
    /// The last restriction read of it, by generation and candidates: a
    /// list read again in the same generation is no new restriction.
    pub(super) restricted_cache: parking_lot::Mutex<Option<(u64, Vec<ProcessId>, super::pool::ChunkMap)>>,
    pub(super) pause_due: AtomicBool,
    pub(super) nonresident: AtomicUsize,
    /// Room grants give back as they drop, the queue's length and the drain
    /// task's wake: shared with every reservation handed out.
    pub(super) returns: Arc<RoomReturns>,
    idle_reclaim_exhausted: std::sync::atomic::AtomicBool,
    pub(super) sweeping: AtomicBool,
    sweep_exhausted: AtomicBool,
    pub(super) demoting: AtomicBool,
    /// A host -> disk spill failed: the engine moves no slots between
    /// tiers below the device, and none is tried again.
    spill_failed: AtomicBool,
    /// Parked state is on its way down a tier ([`Self::park_down`]).
    parking_down: AtomicBool,
    /// A move of parked state down a tier failed: the pool's room count
    /// then (`PoolPort::room_gives`), `u64::MAX` for none. It is tried
    /// again once the pool has moved past it, never on the same map twice:
    /// latched for good, one failure left every parked-in-place process's
    /// state holding the device while their owners waited to be served.
    pub(super) park_down_failed: AtomicU64,
    /// A list compaction moved nothing: the room and the fires landed then
    /// ([`Self::compaction_room`]), `u64::MAX` for none. It is planned
    /// again once either has moved past it: on the same room it would find
    /// the same readers with fires in flight, or the same chunks taken.
    pub(super) list_compaction_failed: AtomicU64,
    /// The segment walk made no grow it promised: the next pass goes on
    /// past its grow steps ([`Self::no_grow`]).
    pub(super) no_grow: AtomicBool,
    /// The last eviction pass found no swap room for a victim: the head
    /// waits on swap slots coming free ([`Self::publish_wants`]).
    pub(super) swap_bound: AtomicBool,
    pub(super) port: Arc<dyn PoolPort>,
    pub(super) stats: PlannerStats,
    /// `PoolFacts::seated_first`: a seated lane's growth before a prompt
    /// waiting to be seated, which waits for a lane to finish.
    pub(super) seated_first: bool,
    /// Held while an uncontended claim reads the rings' free room, weighs it
    /// against the seated lanes' growth and takes its grant, so prompts seated
    /// at once each see the others' pages gone.
    pub(super) seating: parking_lot::Mutex<()>,
    /// `PoolFacts::seat_to_end`: see [`super::to_end::ToEnd`].
    seat_to_end: bool,
    to_end_notify: Notify,
    /// Told when the state pool may seat more: a charged process left, a
    /// set was let go, or the pool backed more slots.
    seat_room: Notify,
    /// What its waiters lack that only the policy can free.
    pub(super) short: parking_lot::Mutex<Shortfall>,
    /// Where that is said (`pressure::raise`, for this planner's engine).
    sink: Sink,
    /// The engine's weight cache, shed before the device's shortfall is
    /// said; `None` where no cache stands in front of the pool.
    weight_shed: Option<WeightShed>,
}

/// How long a process waits at admission before it says why, and again
/// after each such span: what a stall at the gates is read from.
pub(super) const ADMIT_SAYS_AFTER: std::time::Duration = std::time::Duration::from_secs(5);

/// Waits on `wait` for [`ADMIT_SAYS_AFTER`] at most: `false` once that
/// passed, for the caller to say why it waits and look again. With no
/// timer to wait on (a future polled outside a runtime), waits it out.
async fn waits_said(wait: impl std::future::Future<Output = ()>) -> bool {
    if tokio::runtime::Handle::try_current().is_err() {
        wait.await;
        return true;
    }
    crate::rt::time::timeout(ADMIT_SAYS_AFTER, wait)
        .await
        .is_ok()
}

impl ResidencyPlanner {
    pub fn new(port: Arc<dyn PoolPort>) -> Self {
        let sink = super::short::sink(port.locus().1);
        // Device room coming free on any path (a release, a fire's epoch
        // retiring, a sequence ending, an eviction's pages let go) wakes the
        // planner as a dropped grant does: lock-free, never planning in the
        // release itself.
        let returns: Arc<RoomReturns> = Arc::default();
        let signal = Arc::clone(&returns);
        returns.want_nothing();
        port.set_room_signal(crate::store::slots::RoomSignal(Arc::new(move |count| {
            signal.slots_freed(count);
        })));
        Self {
            inner: parking_lot::Mutex::new(Inner::default()),
            need_cache: parking_lot::Mutex::new(None),
            events: AtomicU64::new(0),
            force_moves: AtomicBool::new(false),
            pass_landed: AtomicU64::new(0),
            settled_at: parking_lot::Mutex::new(None),
            map_cache: parking_lot::Mutex::new(None),
            restricted_cache: parking_lot::Mutex::new(None),
            pause_due: AtomicBool::new(false),
            nonresident: AtomicUsize::new(0),
            returns,
            idle_reclaim_exhausted: std::sync::atomic::AtomicBool::new(false),
            sweeping: AtomicBool::new(false),
            sweep_exhausted: AtomicBool::new(false),
            demoting: AtomicBool::new(false),
            spill_failed: AtomicBool::new(false),
            parking_down: AtomicBool::new(false),
            park_down_failed: AtomicU64::new(u64::MAX),
            list_compaction_failed: AtomicU64::new(u64::MAX),
            no_grow: AtomicBool::new(false),
            swap_bound: AtomicBool::new(false),
            port,
            stats: PlannerStats::default(),
            seated_first: false,
            seating: parking_lot::Mutex::new(()),
            seat_to_end: false,
            to_end_notify: Notify::new(),
            seat_room: Notify::new(),
            short: parking_lot::Mutex::new(Shortfall::default()),
            sink,
            weight_shed: None,
        }
    }

    /// Sheds `shed` (the engine's weight cache) before a device shortfall
    /// is said, so the policy lets no kv or state go while streamed weights
    /// above their floor could go instead.
    #[must_use]
    pub fn weight_shed(mut self, shed: WeightShed) -> Self {
        self.weight_shed = Some(shed);
        self
    }

    /// What `units` take, in bytes, at this engine's scale; a unit of
    /// unknown size counts as a byte, so a shortfall never reads as none.
    pub(super) fn bytes_of(&self, units: Reclaim) -> u64 {
        let scale = crate::policy::pressure::scale_of(self.port.locus().1);
        let page = scale.kv_page_bytes.max(1);
        let slot = scale.state_slot_bytes.max(1);
        u64::from(units.pages)
            .saturating_add(u64::from(units.rs.pages))
            .saturating_mul(page)
            .saturating_add(u64::from(units.rs.slots).saturating_mul(slot))
    }

    /// The host rows one set's state takes, which a new ask leaves free for
    /// a restore: a set parked on the host comes back by parking another in
    /// its place, which needs that much room while both stand.
    pub(super) fn restore_reserve(&self) -> RsUnits {
        let (_, total) = self.port.rs_host_stats();
        let mut windows = total.windows;
        for list in &mut windows {
            *list = (*list).min(1);
        }
        RsUnits {
            slots: total.slots.min(1),
            pages: total.pages.min(1),
            windows,
        }
    }

    /// Waiter `key` lacks `device` bytes on the device and `host` on the
    /// host that the planner's own moves cannot give it; both 0 once it
    /// lacks nothing the policy must free.
    pub(super) fn note_short(&self, key: EntryKey, device: u64, host: u64) {
        self.short.lock().note(key.1, [device, host]);
    }

    /// The head is short of room only the policy can make, and there is
    /// something the policy could let go: idle cache, a checkpoint, or a
    /// handle at rest on the device. A starved fleet then waits on the
    /// policy instead of restarting anyone.
    pub(super) fn waits_on_policy(&self) -> bool {
        let Some((head, resting)) = self.with_inner(|inner| {
            let (head, _) = inner.unmet_head()?;
            let resting = inner
                .procs
                .values()
                .any(|proc| proc.holder && proc.state == Residency::Resident);
            Some((head, resting))
        }) else {
            return false;
        };
        if !self.short.lock().is_short(head.1) {
            return false;
        }
        // The host is what is short: the policy frees it by letting a set at
        // rest go, and the head waits until one does.
        if self.short.lock().host_short(head.1) {
            return true;
        }
        let (model, engine) = self.port.locus();
        resting
            || self.port.reclaimable_idle() > 0
            || crate::store::registry::try_get(model, engine)
                .is_some_and(|stores| stores.rs.lock().unwrap().holds_checkpoints())
    }

    /// A suspend for the head found no host rows for the state it would
    /// move: it moved and dropped nothing, and the host is short of `rows`.
    pub(super) fn note_host_short(&self, rows: RsUnits) {
        let Some(head) = self.with_inner(|inner| inner.unmet_head().map(|(key, _)| key)) else {
            return;
        };
        let bytes = self.bytes_of(Reclaim { pages: 0, rs: rows });
        self.short.lock().note_host(head.1, bytes);
        self.report_short();
    }

    /// Says how short each tier is now, for the waiters still waiting:
    /// again while a shortfall stands (the gauge tells the policy again
    /// only when it has stood a while), and once as it clears.
    pub fn report_short(&self) {
        if self.short.lock().quiet() {
            return;
        }
        let waiting: HashSet<u64> = self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .filter(|(_, waiter)| waiter.is_unmet())
                .map(|(key, _)| key.1)
                .collect()
        });
        let said = {
            let mut short = self.short.lock();
            short.retain(|id| waiting.contains(&id));
            short.take_said()
        };
        for (tier, bytes) in said {
            // What the weight cache gave back comes free as the fires
            // reading it retire; the next report says what is still short.
            if tier == crate::policy::pressure::Tier::Device
                && bytes > 0
                && self.weight_shed.as_ref().is_some_and(|shed| shed() > 0)
            {
                tracing::debug!(
                    bytes,
                    "planner: the weight cache sheds before tier pressure"
                );
                continue;
            }
            (self.sink)(tier, bytes);
        }
    }

    /// The capacity gate as it stands ([`super::inner::CapacityGate`]): the
    /// finisher, the oldest admitted process not idle, and what every tier's
    /// room leaves past what is held (idle cache counted free: the policy
    /// lets it go) and the finisher's headroom, its end less the kv pages
    /// it holds on every tier. `None` where the pool states no end or keeps
    /// no chunks, or nothing is admitted.
    #[cfg(test)]
    pub(super) fn capacity_gate(&self) -> Option<super::inner::CapacityGate> {
        self.capacity_order().map(|(gate, _)| gate)
    }

    /// The gate and each process's place in its order
    /// ([`super::inner::CapacityGate`]).
    pub(super) fn capacity_order(
        &self,
    ) -> Option<(super::inner::CapacityGate, std::collections::HashMap<ProcessId, u64>)> {
        self.capacity_reads().map(|(read, places, last)| {
            let mut gate = read.gate();
            // What one not in the order (a new process's first ask) may
            // take: the place after the last.
            gate.spare = last;
            (gate, places.into_iter().collect())
        })
    }

    /// The figures the capacity gate is read from ([`Self::capacity_gate`]).
    #[cfg(test)]
    pub(super) fn capacity_read(&self) -> Option<super::inner::CapacityRead> {
        self.capacity_reads().map(|(read, ..)| read)
    }

    /// [`Self::capacity_read`] and each process's allowance in the gate's
    /// order: past the first, the least room any process ahead of it would
    /// be left with to finish in turn.
    #[allow(clippy::type_complexity)]
    fn capacity_reads(&self) -> Option<(super::inner::CapacityRead, Vec<(ProcessId, u64)>, u64)> {
        let end = self.port.context_pages()?;
        // Pages the planner gathered and holds for its heads are free too:
        // counted taken, gathering for a head held it at the gate, given
        // back it went again, a pass that never ended.
        let (candidates, gathered, residents) = self.with_inner(|inner| {
            let mut candidates: Vec<(u64, ProcessId, bool, u64)> = inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.admitted && !proc.idle && !proc.held)
                .map(|(pid, proc)| {
                    // The room its binds wait for: queued, or the bind that
                    // suspended it.
                    let queued: u64 = inner
                        .queue
                        .values()
                        .filter(|waiter| {
                            waiter.pid == *pid
                                && matches!(waiter.kind, WaitKind::BindRoom { rise: None, .. })
                        })
                        .map(|waiter| u64::from(waiter.room_pages()))
                        .sum();
                    let pending = proc.bind_room.map_or(0, |bind| u64::from(bind.pages));
                    (proc.seq, *pid, proc.state == Residency::Resident, queued.max(pending))
                })
                .collect();
            candidates.sort_unstable_by_key(|&(seq, ..)| seq);
            let residents: Vec<ProcessId> = inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.state == Residency::Resident)
                .map(|(pid, _)| *pid)
                .collect();
            (candidates, inner.accum.len() as u64, residents)
        });
        if candidates.is_empty() {
            return None;
        }
        let census = self.port.capacity_census()?;
        let (model, engine) = self.port.locus();
        // What one eviction of each resident moves below whole: its kv
        // pages on the device, its recurrent state and its guest state and
        // tables, as the census counts them. The largest two, read once.
        let mut moves: Vec<(u64, ProcessId)> = residents
            .iter()
            .map(|&pid| {
                let kv = super::holdings::held_page_count(pid, model, engine)
                    .saturating_sub(super::holdings::swapped_page_count(pid, model, engine));
                let rs = self
                    .port
                    .rs_as_pages(super::holdings::held_rs_units(pid, model, engine));
                (u64::from(kv) + u64::from(rs) + self.port.guest_pages_of(pid), pid)
            })
            .collect();
        moves.sort_unstable_by(|a, b| b.cmp(a));
        // The largest and the second largest move of anyone but `finisher`.
        let unit = |finisher: ProcessId| {
            let mut others = moves.iter().filter(|&&(_, pid)| pid != finisher);
            let largest = others.next().copied();
            let second = others.next().map_or(0, |&(pages, _)| pages);
            (largest.map_or(0, |(pages, _)| pages), largest.map(|(_, pid)| pid), second)
        };
        let read = |&(_, finisher, resident, binding): &(u64, ProcessId, bool, u64)| {
            let guest = self.port.guest_pages_of(finisher);
            let below = u64::from(super::holdings::swapped_page_count(finisher, model, engine))
                + if resident { 0 } else { guest };
            super::inner::CapacityRead {
                finisher,
                end: u64::from(end),
                // Every page of its working sets, on every tier.
                kv: u64::from(super::holdings::held_page_count(finisher, model, engine)),
                below,
                binding,
                guest,
                whole: census.whole,
                held: census.held,
                slack: census.slack,
                idle: census.idle,
                moving: census.moving,
                gathered,
                unit: unit(finisher),
            }
        };
        // An order every admitted process can finish in, one after another
        // (a banker's order, one resource: every tier's free pages): each in
        // turn needs its growth to its end, its restore's state, its binds'
        // room and the margin a whole move takes, and gives back all it
        // holds as it ends. Fewest needed first: if any order finishes them
        // all, that one does. The first is the finisher. An ask of any
        // other waits while it would leave someone ahead of it unable to
        // finish in turn; one in the finisher's place never waits. Held to
        // the first alone, every ask past it was let in while the next
        // process in line no longer fit (the simulation's field mix: the
        // finisher ended, the oldest then needed 74 pages with 19 free, and
        // every other ask was held while nothing could move).
        let mut order: Vec<(u64, u64, super::inner::CapacityRead)> = candidates
            .iter()
            .map(|candidate| {
                let read = read(candidate);
                let rs = self.port.rs_as_pages(super::holdings::held_rs_units(
                    read.finisher,
                    model,
                    engine,
                ));
                (read.need(), read.kv + read.guest + u64::from(rs), read)
            })
            .collect();
        // Fewest needed first, the oldest first among equals (candidates
        // come by admission, and the sort is stable).
        order.sort_by_key(|&(need, ..)| need);
        let (_, _, first) = *order.first()?;
        let mut free = first.free() as i128;
        let mut least = i128::MAX;
        let mut places = Vec::with_capacity(order.len());
        for &(need, held, read) in &order {
            places.push((read.finisher, u64::try_from(least.max(0)).unwrap_or(u64::MAX)));
            least = least.min(free - need as i128);
            free += held as i128;
        }
        // The first's place is the finisher's: it never waits. One not in
        // the order goes after the last.
        places.remove(0);
        Some((first, places, u64::try_from(least.max(0)).unwrap_or(u64::MAX)))
    }

    /// Whether an allocation of `pages` for `pid` waits on the capacity
    /// gate now, said at debug with the finisher and the bound its end came
    /// from, and counted.
    /// Read off the gate the last pass left ([`super::inner::Inner::capacity`]),
    /// each grant taken past it out of its spare: a fire's grant pays one
    /// lock, never a read of every tier.
    pub(super) fn capacity_holds(&self, pid: ProcessId, pages: u32) -> bool {
        // No pass has read it yet: read now, before the first grant.
        if self.with_counts(|inner| inner.capacity.is_none()) {
            let order = self.capacity_order();
            self.with_counts(|inner| {
                if inner.capacity.is_none() {
                    inner.set_capacity(order);
                }
            });
        }
        let Some((gate, holds)) = self.with_counts(|inner| {
            let place = inner.capacity_allow.get(&pid).copied();
            let gate = inner.capacity.as_mut()?;
            let holds = gate.holds(pid, pages, place);
            if !holds && pid != gate.finisher {
                gate.used = gate.used.saturating_add(gate.cost(pid, pages));
            }
            Some((*gate, holds))
        }) else {
            return false;
        };
        if holds {
            self.stats.capacity_waits.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(
                %pid,
                pages,
                finisher = %gate.finisher,
                allowance = gate.allowance(None),
                free = gate.free,
                end = "the declared context",
                "planner: an allocation waits for the finisher's headroom on all tiers"
            );
        }
        holds
    }

    /// [`Self::capacity_holds`] for a bind's `chunks` of fresh guest state.
    pub fn capacity_holds_chunks(&self, pid: ProcessId, chunks: u32) -> bool {
        self.capacity_holds(pid, self.port.chunks_as_pages(chunks))
    }

    /// Whether the gate would hold `chunks` of fresh state for `pid` now,
    /// read without taking them out of its spare.
    pub fn capacity_would_hold(&self, pid: ProcessId, chunks: u32) -> bool {
        let pages = self.port.chunks_as_pages(chunks);
        self.with_inner(|inner| {
            let place = inner.capacity_allow.get(&pid).copied();
            inner.capacity.is_some_and(|gate| gate.holds(pid, pages, place))
        })
    }

    /// The error waiter `key` is refused with where its need could never
    /// fit the device pool, even with nothing else resident: its kv pages
    /// and the pages that free the runs of whole chunks it asks (a bind's,
    /// or a restore's for its roles or the bind that suspended it), together
    /// past the pool's total ([`super::PlannerError::PastDevice`]). `None`
    /// for every other need, however short and for however long: it waits
    /// (its owner parked) while the planner suspends and evicts by its
    /// victim choice and the chunk map, and room given back, an eviction
    /// landed or a completion wakes it. Over-demand is served slower, never
    /// failed; no clock and no answer of the policy refuses a need that fits.
    /// The device's room for state as pages, one figure for every
    /// never-fits check (restores, binds, a fire's unit alone): the state
    /// room at the narrowest rung, the lent span included
    /// ([`super::pool::PoolPort::state_chunks_most`]); the kv pool's pages
    /// where the kv pages are a pool of their own.
    pub(super) fn device_pages_most(&self) -> u32 {
        match self.port.state_chunks_most() {
            u32::MAX => self.port.device_stats().1,
            most => self.port.chunks_as_pages(most),
        }
    }

    /// [`super::pool::PoolPort::chunk_occupants`] of `pids`, the pass
    /// counted as one that built the occupancy map
    /// ([`super::stats::PlannerStats::map_passes`]).
    pub(super) fn occupants(&self, pids: &[ProcessId]) -> super::pool::ChunkMap {
        if !pids.is_empty() {
            self.stats.map_built.store(true, Ordering::Relaxed);
        }
        if pids.is_empty() {
            return self.port.chunk_occupants(pids);
        }
        // One map a generation, over every process the planner knows and
        // those asked: each candidate list (the eviction plan's, the stall
        // line's, the parked map's) reads its own restriction of it. Built
        // per list, a pass built it up to three times (the L40S's 14 s of
        // map passes in a 334 s run).
        let generation = self.generation();
        if let Some((at, of, map)) = self.restricted_cache.lock().as_ref()
            && *at == generation
            && of == pids
        {
            self.stats.map_cache_hits.fetch_add(1, Ordering::Relaxed);
            return map.clone();
        }
        let mut all: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        all.extend(pids.iter().copied());
        all.sort_unstable();
        all.dedup();
        let cached = self.map_cache.lock().as_ref().and_then(|(at, of, map)| {
            (*at == generation && pids.iter().all(|pid| of.binary_search(pid).is_ok()))
                .then(|| map.restricted(pids))
        });
        let restricted = match cached {
            Some(map) => {
                self.stats.map_cache_hits.fetch_add(1, Ordering::Relaxed);
                map
            }
            None => {
                let map = self.port.chunk_occupants(&all);
                let restricted = map.restricted(pids);
                *self.map_cache.lock() = Some((generation, all, map));
                restricted
            }
        };
        *self.restricted_cache.lock() = Some((generation, pids.to_vec(), restricted.clone()));
        restricted
    }

    /// One count of everything a pass reads that can change its answer:
    /// the pool's changes (every take, give and move of slots, room given
    /// back, [`PoolPort::room_changes`]), slots come free on any tier, grants
    /// returned unspent, fires landed, the planner's events (each poke), and
    /// the planner's own book: its queue, gathered pages, moves in flight,
    /// and each process's residency and fires. Unmoved, a pass reads what
    /// the last read, and its answer is the same.
    pub(super) fn generation(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.port.room_changes().hash(&mut hasher);
        self.returns.freed_count().hash(&mut hasher);
        self.returns.epoch().hash(&mut hasher);
        self.stats.fires_landed.load(Ordering::Acquire).hash(&mut hasher);
        self.events.load(Ordering::Acquire).hash(&mut hasher);
        self.with_inner(|inner| inner.book_print()).hash(&mut hasher);
        hasher.finish()
    }

    pub(super) fn never_fits(&self, key: EntryKey) -> Option<super::PlannerError> {
        // One device figure on every path: the state room at the narrowest
        // rung, lent span included (Mac Pro on ba229ff25 read 365 pages on
        // one path and 428 on the other, the 63 lent chunks apart).
        let most = self.port.state_chunks_most();
        let total = self.device_pages_most();
        // Its runs, its unpark's and its state's chunks together past every
        // chunk state may take: with every other process evicted and all
        // parked state moved down it still would not fit, and it is refused
        // by name, never waited on (the L40S on 3b45462d4: a restore's 1,058
        // slots of guest state against what the state limit leaves).
        if most != u32::MAX
            && let Some(room) = self.head_room_within(key, false)
            && room.chunks() > u64::from(most)
        {
            let pages = |chunks: u64| {
                self.port
                    .chunks_as_pages(u32::try_from(chunks).unwrap_or(u32::MAX))
            };
            return Some(super::PlannerError::PastDevice {
                need: pages(room.chunks()),
                total,
                runs: room.longest(),
            });
        }
        let (need, runs) = self.with_inner(|inner| {
            let waiter = inner.queue.get(&key)?;
            let runs = match waiter.kind {
                WaitKind::BindRoom { run, pieces, .. } => Some((run, pieces)),
                WaitKind::Restore { .. } => inner.procs.get(&waiter.pid).and_then(|proc| {
                    proc.roles_short
                        .into_iter()
                        .chain(proc.bind_room)
                        .max_by_key(|need| need.pages)
                        .map(|need| (need.run, need.pieces))
                }),
                WaitKind::Allocation { .. } => None,
            };
            Some((
                waiter.kv_need().saturating_add(inner.room_pages(waiter)),
                runs,
            ))
        })?;
        (need > total).then_some(super::PlannerError::PastDevice { need, total, runs })
    }

    /// With `PoolFacts::seat_to_end`, a process starts only while the pools
    /// hold the started ones to the end (see [`super::to_end::ToEnd`]).
    #[must_use]
    pub fn seat_to_end(mut self, on: bool) -> Self {
        self.seat_to_end = on;
        self
    }

    /// With `PoolFacts::seat_spread`, the starts the pools do not hold to the
    /// end at once are spread over a life (see [`super::to_end::ToEnd`]).
    #[must_use]
    pub fn seat_spread(mut self, on: bool) -> Self {
        self.inner.get_mut().to_end.spread = on;
        self
    }

    /// With `PoolFacts::seated_first`, seated growth doesn't park behind a waiting
    /// prompt, which is seated only within the rings' growth room
    /// (`PoolPort::rs_growth`) and suspends only an idle lane.
    #[must_use]
    pub fn seated_first(mut self, on: bool) -> Self {
        self.seated_first = on;
        self
    }

    /// The rs units a seated lane may still claim, when the planner keeps
    /// them for it.
    pub(super) fn rs_growth(&self) -> RsUnits {
        if self.seated_first {
            self.port.rs_growth()
        } else {
            RsUnits::default()
        }
    }

    pub fn arm_drain_task(self: &Arc<Self>) {
        if !crate::rt::has_runtime() {
            return;
        };
        let notify = Arc::new(Notify::new());
        if self.returns.drain.set(notify.clone()).is_err() {
            return;
        }
        let planner = self.clone();
        crate::rt::spawn(async move {
            loop {
                // With no one queued it sleeps until a poke (an enqueue
                // pokes, so the first waiter is seen at once).
                if planner.returns.waiters.load(Ordering::Acquire) == 0 {
                    notify.notified().await;
                    planner.plan();
                    continue;
                }
                // Not on the policy's answer of nothing freed: nothing
                // changed, and a pass taken on it said the shortfall again,
                // drew the next such answer and planned again, sixty times a
                // second with nothing to issue (the L40S: a stuck head, 98
                // passes between two stall reports). What the policy does
                // free comes back through the pool's room signal.
                tokio::select! {
                    () = notify.notified() => planner.plan(),
                    _ = crate::rt::time::timeout(
                        crate::policy::pressure::AGAIN,
                        std::future::pending::<()>(),
                    ) => planner.tick(),
                }
            }
        });
    }

    /// The drain task's beat while anyone waits, once a
    /// [`crate::policy::pressure::AGAIN`] for the whole planner, never one
    /// per waiter: a shortfall that stands is said again (the default
    /// responder looks again when its last discard freed too little),
    /// fires stood unsettled past their window are noted (a fleet that
    /// stalls so is planned for), a waiter whose need the pool shrank under
    /// is woken to be refused, and a stall that goes on is said again. It
    /// plans nothing on the clock: events plan.
    pub(super) fn tick(self: &Arc<Self>) {
        // The backstop, never the main path: room that came free since the
        // last pass began and now meets what the head lacks (a free that
        // raced that pass, or a path that woke no one) is planned on within
        // a beat; frees that meet nothing plan nothing.
        let beat = || {
            self.returns
                .wake_cause
                .store(super::grant::WAKE_BEAT, Ordering::Release);
        };
        if self.returns.freed_since_plan() && self.wants_met_now() {
            beat();
            self.plan();
        } else if self.quiet_unplanned() {
            beat();
            self.plan();
            // Read after the pass: room its own steps took and gave back is
            // the room it planned on, never a change that plans again.
            self.quiet_unplanned();
        }
        self.say_held_in_flight();
        self.report_short();
        self.note_lapsed_fires();
        // One read of the queue, as `never_fits` reads each need.
        let (_, total) = self.port.device_stats();
        let past: Vec<Arc<Notify>> = self.with_inner(|inner| {
            inner
                .queue
                .values()
                .filter(|waiter| waiter.is_unmet())
                .filter_map(|waiter| match &waiter.kind {
                    WaitKind::Allocation { notify, .. }
                        if waiter.kv_need().saturating_add(inner.room_pages(waiter)) > total =>
                    {
                        Some(notify.clone())
                    }
                    _ => None,
                })
                .collect()
        });
        for notify in past {
            notify.notify_waiters();
        }
        self.dump_stall();
    }

    /// Whether the queue's head waits unserved with nothing in flight on
    /// room no pass has planned it on since the fleet went quiet: the fleet
    /// going quiet is an event no store tells (the last fire landing, an
    /// eviction settling), and with no pass after it a restore head left
    /// unfunded while fires were in flight waited for good (Mac Pro's
    /// 2 GiB: one stall line in three minutes, 0 chunks free, 308
    /// emptiable, no eviction issued). Once per head and room: the same
    /// room plans nothing again. Records the room it reads.
    fn quiet_unplanned(&self) -> bool {
        let room = [
            self.port.room_gives(),
            u64::from(self.port.free_chunks()),
            u64::from(self.port.device_stats().0),
            u64::from(self.raw_swap_free()),
        ];
        self.with_inner(|inner| {
            let Some((key, _)) = inner.unmet_head() else {
                inner.quiet_planned = None;
                return false;
            };
            if !inner.nothing_in_flight() || inner.quiet_planned == Some((key, room)) {
                return false;
            }
            inner.quiet_planned = Some((key, room));
            true
        })
    }

    /// A head unserved past [`super::plan::STALL_SAYS_AGAIN`], spinning or
    /// quiet: what is in flight and whose slots hold the opened chunks are
    /// said, once per head each such span. Mac Pro's 2 GiB mix went silent
    /// after one stall line, a head waiting on something no line named.
    fn say_held_in_flight(&self) {
        let gives = self.port.room_gives();
        let held = self.with_inner(|inner| {
            let (key, pid, kind) = inner
                .unmet_head()
                .map(|(key, waiter)| (key, waiter.pid, waiter.kind_name()))?;
            // Once per head each STALL_SAYS_AGAIN, spinning or quiet: a
            // head re-planned continuously never goes quiet, and its line
            // was never said (Mac Pro on 22b0a3e83).
            let now = crate::rt::Instant::now();
            match inner.held_in_flight {
                Some((at, _, since, _)) if at == key => {
                    if since.elapsed() < super::plan::STALL_SAYS_AGAIN {
                        return None;
                    }
                    inner.held_in_flight = Some((key, gives, now, true));
                }
                _ => {
                    inner.held_in_flight = Some((key, gives, now, false));
                    return None;
                }
            }
            let firing: Vec<ProcessId> = inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.firing())
                .map(|(pid, _)| *pid)
                .collect();
            let moving: Vec<(ProcessId, Residency)> = inner
                .procs
                .iter()
                .filter(|(_, proc)| matches!(proc.state, Residency::Evicting | Residency::Restoring))
                .map(|(pid, proc)| (*pid, proc.state))
                .collect();
            let (plan, blocked) = inner.plan_fields(key);
            Some((
                plan,
                blocked,
                pid,
                kind,
                firing.len(),
                firing.into_iter().take(4).collect::<Vec<_>>(),
                moving,
                inner.evicting.keys().copied().collect::<Vec<_>>(),
                inner.kill_in_flight(),
                inner.parked_moving.iter().copied().collect::<Vec<_>>(),
            ))
        });
        if let Some((plan, blocked, pid, kind, firing, some_firing, moving, evicting, killing, parked_moving)) =
            held
        {
            let quiet = self.with_inner(|inner| inner.nothing_in_flight());
            let held_by = self.held_by_state();
            let store = self.store_holders();
            tracing::warn!(
                %pid,
                kind,
                quiet,
                %plan,
                %blocked,
                capped = %self.capped_line(),
                swap = ?self.swap_stats(),
                %store,
                %held_by,
                firing,
                ?some_firing,
                ?moving,
                ?evicting,
                killing,
                ?parked_moving,
                "planner: the queue's head stays unserved: what is in flight and whose slots \
                 hold the opened chunks"
            );
        }
    }

    /// Publishes what the queue's head lacks for room coming free to be
    /// weighed against without a lock ([`RoomReturns::met`]): the free
    /// chunks its runs take, the kv list slots its pages take, the swap
    /// slots its victims' moves wait on. A head whose lack none of those
    /// measures (state slots, a gate) is woken by any chunk coming free.
    pub(super) fn publish_wants(&self) {
        // No one waits: nothing to weigh, nothing read.
        if self.returns.waiters.load(Ordering::Acquire) == 0 {
            self.returns.want_nothing();
            return;
        }
        let head = self.with_inner(|inner| {
            inner.unmet_head().map(|(key, waiter)| {
                (
                    key,
                    waiter.kv_need().saturating_sub(inner.accum.len() as u32),
                    inner
                        .procs
                        .get(&waiter.pid)
                        .is_some_and(|proc| proc.roles_short.is_some()),
                    waiter.rs_need(),
                )
            })
        });
        let Some((key, kv_missing, refused, rs_need)) = head else {
            self.returns.want_nothing();
            return;
        };
        let rs_short = !rs_need.is_zero() && !self.port.rs_stats().0.covers(rs_need);
        // What it lacks, as the pass's step reads it (kv, then state slots,
        // then runs): said with its wait if that runs long.
        self.with_inner(|inner| {
            let moving = inner
                .queue
                .get(&key)
                .is_some_and(|waiter| inner.parked_moving.contains(&waiter.pid));
            let bind = inner
                .queue
                .get(&key)
                .and_then(|waiter| inner.procs.get(&waiter.pid))
                .is_some_and(|proc| proc.bind_room.is_some());
            if let Some(waiter) = inner.queue.get_mut(&key) {
                waiter.lacked = match waiter.kind {
                    WaitKind::BindRoom { rise: Some(_), .. } => "rise room",
                    WaitKind::BindRoom { .. } => "bind room",
                    _ if moving => "its parked state moving down",
                    _ if kv_missing > 0 => "kv pages",
                    _ if rs_short && refused => "state slots, refused in its unpark before",
                    _ if rs_short => "state slots",
                    _ if refused => "runs its unpark was refused",
                    _ if bind => "its bind's runs",
                    _ => "nothing the pool lacks",
                };
            }
        });
        // A head refused on the room standing now (its unpark came up short
        // there): a count that room already meets moves nothing, so its
        // want is past what stands. Its chunk count met by the free chunks
        // (a restore short of a run of four, 231 chunks free), every chunk
        // given back anywhere, the pass's own among them, woke a pass that
        // found no step (Mac Pro's mx-3: ~470 a second, nothing in flight).
        // A head never refused is tried against the room standing, its
        // want its need.
        let past = |count: u32| if refused { u64::from(count).saturating_add(1) } else { 0 };
        // Refused, it goes again on any chunk more than its refusal left
        // where its placement stands, which reads its own parked parts
        // retaken first: fewer chunks free than its whole need's count can
        // stand it. A floor of that count left a chunk given back that
        // stood it unwoken, the head waiting on a pass some other event
        // ran (the simulation's lost wakes: "restore short of runs:
        // something in flight, not funded", then sent at once).
        let mut want_chunks = self.head_room(key).map_or(u64::MAX, |need| {
            if refused {
                past(self.port.free_chunks())
            } else {
                need.chunks()
            }
        });
        let want_list = if kv_missing > 0 {
            u64::from(kv_missing).max(if refused { past(self.port.device_stats().0) } else { 0 })
        } else {
            u64::MAX
        };
        let want_swap = if self.swap_bound.load(Ordering::Acquire) {
            u64::from(self.raw_swap_free()).saturating_add(1)
        } else {
            u64::MAX
        };
        // Short of state slots (records, rs list or window slots), never
        // refused: slots of those shapes given back fill no chunk and move
        // no kv slot, so no count the signal carries measures them; any
        // room given back wakes it. Left to the free chunks' count, a
        // restore short of its state waited for a fire landing's poke, up
        // to 15 s (the L40S at 6 GiB on 22b0a3e83: restores served almost
        // all by pokes, about one a run by the room signal).
        // Refused before in its unpark or not: its step is state slots
        // (read ahead of its runs), and the refusal's want of more chunks
        // than stood was a count state given back inside open chunks never
        // moves (the L40S at 6 GiB on 5120cbc3c: restore waits served by
        // pokes, up to 16.5 s, a few by the room signal).
        if rs_short && (!refused || kv_missing == 0) {
            want_chunks = 0;
        }
        if want_chunks == u64::MAX && want_list == u64::MAX && want_swap == u64::MAX {
            want_chunks = u64::from(self.port.free_chunks()).saturating_add(1);
        }
        let (free_chunks, free_pages) = (self.port.free_chunks(), self.port.device_stats().0);
        self.with_inner(|inner| {
            if let Some(waiter) = inner.queue.get_mut(&key) {
                waiter.wanted = ([want_chunks, want_list, want_swap], [free_chunks, free_pages]);
            }
        });
        self.returns.want_chunks.store(want_chunks, Ordering::Release);
        self.returns.want_list.store(want_list, Ordering::Release);
        self.returns.want_swap.store(want_swap, Ordering::Release);
    }

    /// Whether the pool as it stands meets what the head lacks
    /// ([`Self::publish_wants`]).
    pub(super) fn wants_met_now(&self) -> bool {
        // Only the counts a want is set for are read: each takes a store's
        // lock, and this runs at the end of plan passes on the fire path.
        let want = |field: &AtomicU64| field.load(Ordering::Acquire);
        let chunks = want(&self.returns.want_chunks);
        if chunks != u64::MAX && u64::from(self.port.free_chunks()) >= chunks {
            return true;
        }
        let list = want(&self.returns.want_list);
        if list != u64::MAX && u64::from(self.port.device_stats().0) >= list {
            return true;
        }
        let swap = want(&self.returns.want_swap);
        swap != u64::MAX && u64::from(self.raw_swap_free()) >= swap
    }

    /// The host's and the disk's free slots, idle cache's not counted.
    pub(super) fn raw_swap_free(&self) -> u32 {
        self.port.host_stats().0.saturating_add(self.port.disk_stats().0)
    }

    pub(super) fn poke(self: &Arc<Self>) {
        self.events.fetch_add(1, Ordering::AcqRel);
        self.wake_planner();
    }

    /// Wakes the planner on no event of its own: a parked waiter's wake on
    /// room given back, a rung moved or guest state given back, whose
    /// change the generation already reads (the pool's counts, the grant
    /// returns). A wake whose generation the last no-step pass ended on
    /// plans nothing ([`Self::settled_at`]): waking each parked KV-short
    /// restore on every poke cost the L40S ~280 s of waits a run.
    pub(super) fn wake_planner(self: &Arc<Self>) {
        self.returns
            .wake_cause
            .store(super::grant::WAKE_POKE, Ordering::Release);
        match self.returns.drain.get() {
            Some(notify) => notify.notify_one(),
            None => self.plan(),
        }
    }

    /// Drops idle cache for what the head is `short` of, past what earlier
    /// reclaims freed that is still returning, and no more
    /// ([`super::pool::PoolPort::reclaim_idle`]).
    pub(super) fn reclaim_idle_once(&self, short: super::pool::IdleShort) -> IdleOutcome {
        if self.idle_reclaim_exhausted.swap(true, Ordering::AcqRel) {
            return IdleOutcome::Nothing;
        }
        let reclaim = self.port.reclaim_idle(short);
        if reclaim.freed > 0 {
            self.idle_reclaim_exhausted.store(false, Ordering::Release);
            return IdleOutcome::Freed;
        }
        if reclaim.returning > 0 {
            self.idle_reclaim_exhausted.store(false, Ordering::Release);
            return IdleOutcome::Returning;
        }
        IdleOutcome::Nothing
    }

    /// Idle cache the device is short of room for moves down a tier, host
    /// first then disk, rather than being dropped, where a policy keys it
    /// above the retention floor (the stores gate it: floor-keyed cache is
    /// dropped by [`Self::reclaim_idle_once`] instead, never copied): whether
    /// a move is on its way (its landing frees the pages and pokes the
    /// planner). Only under
    /// a waiter's want, never while pages stand free. No share of the host
    /// is held back from it: a suspend takes its host room by dropping the
    /// cache there first (`KvStore::prepare_suspend`), so what is paused
    /// goes before what is cached with no fraction kept.
    ///
    /// `wanted`: the pages the head lacks. No more than that moves: a head
    /// short of one prompt's pages waited behind the copy of every idle page
    /// on the device.
    pub(super) fn demote_idle(self: &Arc<Self>, wanted: u32) -> bool {
        if !self.port.suspend_capable() {
            return false;
        }
        if self.demoting.swap(true, Ordering::AcqRel) {
            return true;
        }
        let (mut host_free, host_total) = self.port.host_stats();
        let (disk_free, _) = self.port.disk_stats();
        // A full host is under pressure too: its own idle cache, lowest key
        // first, goes on down to disk before the device's lands there, as
        // much as the demote wants, so the tiers keep the newer above the
        // older and nothing on the host is dropped for want of room.
        if host_total > 0
            && host_free == 0
            && disk_free > 0
            && !self.spill_failed.load(Ordering::Acquire)
        {
            if let Some(txn) = self.port.prepare_spill(wanted.max(1) as usize) {
                exec::spawn_spill(self.clone(), txn, Self::spill_done);
                return true;
            }
            // Settling pages on their disk copies may have freed some.
            host_free = self.port.host_stats().0;
        }
        let (disk_free, _) = self.port.disk_stats();
        let room = (host_free + disk_free).min(wanted.max(1));
        let txn = (room > 0)
            .then(|| self.port.prepare_demote(room as usize))
            .flatten();
        let Some(txn) = txn else {
            self.demoting.store(false, Ordering::Release);
            return false;
        };
        let pages = txn.page_count() as u32;
        ptrace!("demote begins pages={}", pages);
        self.with_inner(|inner| {
            inner.evicting.insert(
                DEMOTE,
                Reclaim {
                    pages,
                    rs: RsUnits::default(),
                },
            )
        });
        exec::spawn_move_out(self.clone(), txn, "demote", Self::demote_done);
        true
    }

    /// A spill that failed is not tried again: the host's cache stays where
    /// it is and the device's goes to disk past it, as before spilling.
    fn spill_done(self: &Arc<Self>, moved: bool) {
        if !moved {
            self.spill_failed.store(true, Ordering::Release);
        }
        self.demote_done(moved);
    }

    fn demote_done(self: &Arc<Self>, moved: bool) {
        ptrace!("demote done moved={}", moved);
        self.with_inner(|inner| inner.settle_eviction(DEMOTE));
        self.demoting.store(false, Ordering::Release);
        if moved {
            self.re_arm_idle_reclaim();
        }
        self.poke();
    }

    /// The runs of whole chunks the head `key`'s step takes when it runs,
    /// `(run, pieces)`: a bind waiting for room, its runs; the restore of a
    /// process a bind suspended, its bind's runs beside what its own unpark
    /// takes back first ([`Self::beside_own`]); a restore short of its
    /// roles, the shortfall its unpark named. One reading for the stall
    /// record, the census, the chunk map's ask and the step: read apart,
    /// the map asked less than the step took and moved no one. `None` for
    /// a head short of kv alone.
    pub(super) fn head_room(&self, key: EntryKey) -> Option<RunNeed> {
        // Read once a step of a pass: outside a pass (the wants published
        // after it, a test, a report) it is read as it stands.
        if !self.returns.planning.load(Ordering::Acquire) {
            return self.head_room_within(key, true);
        }
        let pass = self.stats.plan_passes.load(Ordering::Relaxed);
        let gathered = self.with_inner(|inner| inner.accum.len() as u32);
        if let Some((at, head, pages, need)) = self.need_cache.lock().as_ref()
            && (*at, *head, *pages) == (pass, key, gathered)
        {
            return need.clone();
        }
        let need = self.head_room_within(key, true);
        *self.need_cache.lock() = Some((pass, key, gathered, need.clone()));
        need
    }

    /// [`Self::head_room`], `capped` to the pool's state chunks or not:
    /// uncapped, what the head's step takes whole, for whether it ever
    /// fits alone ([`Self::never_fits`]).
    fn head_room_within(&self, key: EntryKey, capped: bool) -> Option<RunNeed> {
        let (need, suspended, kv_missing) = self.with_inner(|inner| {
            let waiter = inner.queue.get(&inner.locate(key)?)?;
            match &waiter.kind {
                WaitKind::BindRoom { run, pieces, .. } => {
                    Some(((*run, *pieces), None, (0, RsUnits::default())))
                }
                WaitKind::Restore { .. } => {
                    let proc = inner.procs.get(&waiter.pid)?;
                    let need = proc.restore_need()?;
                    let suspended = proc.bind_room.is_some() && proc.roles_short.is_none();
                    // The run its last unpark was refused, kept across a
                    // re-send: its need holds it until a restore serves it.
                    let refused = proc.roles_refused;
                    // A bind it was suspended for, of another length than
                    // its refusal's: its runs stand beside the unpark's.
                    let bind = proc
                        .bind_room
                        .filter(|bind| !suspended && bind.run != need.0)
                        .map(|bind| (bind.run, bind.pieces));
                    // Its kv pages not yet gathered take chunks of the same
                    // free pool as its runs, and so does its state (records,
                    // rs list slots) its serve takes before its unpark.
                    let kv_missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                    Some((
                        need,
                        Some((waiter.pid, suspended, bind, refused)),
                        (kv_missing, waiter.rs_need()),
                    ))
                }
                WaitKind::Allocation { .. } => None,
            }
        })?;
        let refused = suspended.and_then(|(_, _, _, refused)| refused);
        let mut room = match suspended {
            // A bind's runs beside what its own unpark takes back first.
            Some((pid, true, _, _)) => self.beside_own(pid, need),
            // Short of its roles: the whole unpark, not the part it came up
            // short on (its own runs count that part among them), and the
            // runs of the bind that suspended it, of their own length.
            Some((pid, false, bind, _)) => {
                let mut room = self.whole_unpark(pid, need);
                if let Some((run, pieces)) = bind {
                    room.add(run, pieces);
                }
                room
            }
            None => RunNeed::one(need.0, need.1),
        };
        if let Some((run, pieces)) = refused {
            room.at_least(run, pieces);
        }
        // A restore's step gathers its kv pages, then lays its runs: the
        // chunks the pages still need are asked in runs of its longest
        // length, laid after the runs (they need not be adjacent; asked so,
        // no page gathered first breaks a run the need counts on). Read as
        // the runs alone, fitting runs stood (Mac Pro: a restore of (3, 3)
        // and 41 kv pages, 8 runs open), the pages broke them as they were
        // gathered, and the unpark came up short.
        // Its state the serve takes first (Mac Pro's 2 GiB on 3b45462d4: a
        // restore's records opened chunks between the map's read and its
        // unpark, which came up short of a run of four with no chunk free,
        // the map showing its need standing, round after round) is asked
        // the same way.
        let (kv_missing, rs) = kv_missing;
        let pages = kv_missing.saturating_add(self.port.rs_as_pages(rs));
        let per_chunk = self.port.chunks_as_pages(1);
        // As single chunks beside its runs, never runs of their own: asked
        // as runs of the longest length, a restore with many pages to
        // gather asked runs no pool of this size held (Mac Pro at 2 GiB on
        // c10d7de5d: (7, 1) grew to (7, 31), 217 chunks against 30 free runs
        // of 7, on a device 310 of 365 pages free, and nothing was moved).
        // Where its gathered pages break runs the map stood, the store's
        // placement (its kv taken first) places them first, and the map
        // lays its runs in what they leave ([`Self::head_placement`]).
        if per_chunk > 0 && pages > 0 {
            room.add(1, pages.div_ceil(per_chunk));
        }
        if !capped {
            return Some(room);
        }
        Some(room.within(self.port.state_chunks(), Some(need)))
    }

    /// The runs a restore short of its roles takes when its unpark runs:
    /// the whole unpark (`PoolPort::unpark_need`, its parts' runs, one
    /// entry per run length, the one it came up short on among them) where
    /// that is more than the part its refusal named, never more than the
    /// pool stands. Asked the part alone, the map saw it free and moved no
    /// one, the refusal came again with those runs standing, and the need
    /// grew a run a refusal (9x3, then 9x4) while nothing was issued.
    pub(super) fn whole_unpark(&self, pid: ProcessId, (run, pieces): (u32, u32)) -> RunNeed {
        let mut need = self.port.unpark_need(pid);
        need.at_least(run, pieces);
        need
    }

    /// The runs a restore a bind suspended asks of the chunk map: its bind's
    /// `pieces` runs of `run` chunks beside those its own unpark lays first
    /// (`PoolPort::unpark_need`, one entry per run length; the pool's bound
    /// read by the caller). The census reads the bind's runs beside its own
    /// retake (`room_beside_own`); a map asked for the bind's runs alone saw
    /// them free, found no one to move, and the restore waited with no step.
    pub(super) fn beside_own(&self, pid: ProcessId, (run, pieces): (u32, u32)) -> RunNeed {
        let mut need = self.port.unpark_need(pid);
        need.add(run, pieces);
        need
    }

    /// Parked state lying on the device, for a map for `head`: role slots
    /// parked where they lie ([`PoolPort::parked_occupants`]), and the state
    /// evicted processes left on the device at their eviction (records, rs
    /// list slots, [`PoolPort::chunk_occupants`] of them), the head's own
    /// once it is evicted. A move down a tier frees either
    /// ([`Self::park_down`]); counted as no one's, queued processes waiting
    /// to be restored held chunks no move freed (Mac Pro on 22b0a3e83).
    pub(super) fn parked_map(&self, head: ProcessId) -> super::pool::ChunkMap {
        let except = self.parked_except(head);
        let evicted: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    proc.state == Residency::Evicted
                        && **pid != except
                        && !inner.parked_moving.contains(*pid)
                })
                .map(|(pid, _)| *pid)
                .collect()
        });
        let mut map = self.port.parked_occupants(except);
        if !evicted.is_empty() {
            map.merge(self.occupants(&evicted));
        }
        map
    }

    /// Whose parked state a map for `head` leaves out: its own while it is
    /// resident (its state is live, whatever of it lies parked), nobody's
    /// once it is evicted. Nothing runs on an evicted head's state parked in
    /// place; moved down a tier, its restore lays it back compactly. Left
    /// out, an evicted head's own tables and rings (126 instances on the
    /// M1) held the chunks its bind's runs needed, and nothing moved them.
    pub(super) fn parked_except(&self, head: ProcessId) -> ProcessId {
        let evicted = self.with_inner(|inner| {
            inner
                .procs
                .get(&head)
                .is_some_and(|proc| proc.state == Residency::Evicted)
        });
        if evicted { SWEEP } else { head }
    }

    /// Parked state lying on the device (role slots its park found no host
    /// room for) is room a move makes, as idle cache is: for the head's
    /// runs (`need`, each length laid as the map lays it), the parked processes whose
    /// state alone empties the cheapest such runs, free chunks beside it,
    /// move it down a tier ([`Self::park_down`]). Whether a move is on its
    /// way (its landing pokes the planner).
    pub(super) fn park_down_for(self: &Arc<Self>, need: &RunNeed) -> bool {
        let Some(head) = self.with_inner(|inner| inner.unmet_head().map(|(_, w)| w.pid)) else {
            return false;
        };
        match self.parked_map(head).cheapest_need(need) {
            Some(pids) if !pids.is_empty() => self.park_down(head, pids),
            _ => false,
        }
    }

    /// Moves the parked state lying on the device down a tier for `head`
    /// short of `pages` kv pages ([`Self::park_down`]): the occupants of
    /// the cheapest chunks that, emptied, hold them, as many of those as
    /// stand (a part funds a part; residents move for the rest). No one
    /// runs on parked state, so it goes before any resident does, as for a
    /// head short of runs. Whether a move is on its way.
    pub(super) fn park_down_for_pages(self: &Arc<Self>, head: ProcessId, pages: u32) -> bool {
        let per_chunk = self.port.chunks_as_pages(1);
        if pages == 0 || per_chunk == 0 {
            return false;
        }
        let mut map = self.parked_map(head);
        let emptiable = map.held.iter().filter(|chunk| chunk.emptiable()).count() as u32;
        let chunks = pages.div_ceil(per_chunk).min(emptiable);
        if chunks == 0 {
            return false;
        }
        // Chunks free already gave the head what they hold (it absorbed the
        // free pages first): only emptied ones are new room.
        map.free.clear();
        match map.cheapest_runs(1, chunks) {
            Some(pids) if !pids.is_empty() => self.park_down(head, pids),
            _ => false,
        }
    }

    /// Moves the parked state `pids` left on the device down a tier: the
    /// host where it has room, else host kv goes on down to disk first to
    /// make it, the evicted processes' in the order they would be evicted
    /// (lowest first), `head`'s last of all, its own only once it is evicted: the
    /// tiers keep the newer above the older, as idle cache's do
    /// ([`Self::demote_idle`]). Only evicted processes' state moves, and
    /// none of them is restored until it lands. Whether a move is on its
    /// way.
    /// Whether `pid` is to be sent once whatever its placement reads (a
    /// disagreement with nothing to move), taken as read.
    pub(super) fn take_send_regardless(&self, pid: ProcessId) -> bool {
        self.with_inner(|inner| {
            inner
                .procs
                .get_mut(&pid)
                .is_some_and(|proc| std::mem::take(&mut proc.send_regardless))
        })
    }

    /// The processes whose restore is under way.
    pub(super) fn restoring_pids(&self) -> Vec<ProcessId> {
        self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.state == Residency::Restoring)
                .map(|(pid, _)| *pid)
                .collect()
        })
    }

    pub(super) fn park_down(self: &Arc<Self>, head: ProcessId, pids: Vec<ProcessId>) -> bool {
        let failed_at = self.park_down_failed.load(Ordering::Acquire);
        if failed_at != u64::MAX && failed_at == self.park_down_room() {
            return false;
        }
        if self.parking_down.swap(true, Ordering::AcqRel) {
            return true;
        }
        let (pids, order) = self.with_inner(|inner| {
            // Parked state of a process the planner no longer knows (gone
            // while parked) moves down too: no one runs on it, and passed
            // over it held its chunks for good.
            // An evicted head's own parked state moves too: nothing runs on
            // it, and its restore lays it back compactly. A resident head's
            // is never evicted, so never moved.
            let evicted = |pid: &ProcessId| {
                inner
                    .procs
                    .get(pid)
                    .is_none_or(|proc| proc.state == Residency::Evicted)
            };
            let (pids, passed): (Vec<ProcessId>, Vec<ProcessId>) =
                pids.into_iter().partition(|pid| evicted(pid));
            if !passed.is_empty() {
                // One evicting or restoring is in flight; the head's own
                // state is never moved for it. Named, never silent.
                let states: Vec<String> = passed
                    .iter()
                    .map(|pid| {
                        let state = inner.procs.get(pid).map(|proc| proc.state);
                        format!("{pid}: {state:?}{}", if *pid == head { " (the head)" } else { "" })
                    })
                    .collect();
                tracing::debug!(%head, ?states, "planner: parked state passed over for a move down");
            }
            let mut order: Vec<(std::cmp::Reverse<Rank>, ProcessId)> = inner
                .procs
                .iter()
                .filter(|(pid, _)| evicted(pid))
                .map(|(pid, proc)| (std::cmp::Reverse(proc.rank()), *pid))
                .collect();
            order.sort_unstable();
            inner.parked_moving.extend(pids.iter().copied());
            let order: Vec<ProcessId> = order.into_iter().map(|(_, pid)| pid).collect();
            (pids, order)
        });
        if pids.is_empty() || !self.port.move_parked_down(self, &pids, &order) {
            // Refused here as at its landing: not tried again on this room
            // (its take and give-back of the slots below counted room freed,
            // and the pass ran again on it, for good: seed 3392).
            if !pids.is_empty() {
                self.park_down_failed
                    .store(self.park_down_room(), Ordering::Release);
            }
            self.with_inner(|inner| inner.parked_moving.clear());
            self.parking_down.store(false, Ordering::Release);
            return false;
        }
        true
    }

    /// The room a move of parked state down a tier reads: the device's
    /// room given back and the free slots below. One refused is tried again
    /// once either moves.
    /// The room a list compaction is planned on: the pool's room count, the
    /// swap tiers' free slots and the fires landed.
    pub(super) fn compaction_room(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.park_down_room().hash(&mut hasher);
        self.stats.fires_landed.load(Ordering::Acquire).hash(&mut hasher);
        hasher.finish()
    }

    pub(super) fn park_down_room(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        self.port.room_gives().hash(&mut hasher);
        self.raw_swap_free().hash(&mut hasher);
        hasher.finish()
    }

    /// A move of parked state down a tier ended: its processes may be
    /// restored again, and the plan looks at the room it made. One that
    /// failed is not tried again on the same map: it would fail the same
    /// way until room moves (`park_down_failed`).
    pub(super) fn parked_down_done(self: &Arc<Self>, moved: bool) {
        let failed_at = if moved { u64::MAX } else { self.park_down_room() };
        self.park_down_failed.store(failed_at, Ordering::Release);
        self.with_inner(|inner| inner.parked_moving.clear());
        self.parking_down.store(false, Ordering::Release);
        self.poke();
    }

    pub(super) fn re_arm_idle_reclaim(&self) {
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        self.sweep_exhausted.store(false, Ordering::Release);
    }

    /// Not what the head and restores under way read: they would bring it
    /// straight back.
    /// With `tails_only`, only the tails past a head a key keeps go: a set
    /// keyed whole stays as it is (what a head short of state reaches for,
    /// as a whole set moves with its state instead).
    pub(super) fn sweep_stranded(self: &Arc<Self>, wanted: u32, tails_only: bool) -> bool {
        if self.sweep_exhausted.load(Ordering::Acquire)
            || self.sweeping.swap(true, Ordering::AcqRel)
        {
            return false;
        }
        // The slots free now, idle cache's not counted: a sweep takes only
        // what stands free, no share of it held back.
        let room = (self.port.host_stats().0 + self.port.disk_stats().0) as usize;
        let (model, engine) = self.port.locus();
        // What rests goes in the order it would be evicted, and no more of
        // it than is wanted: a policy's keys hold here too, the head of a set
        // they keep going last.
        let ordered: Vec<(crate::store::kv::page_table::WorkingSetId, u64)> = self
            .with_inner(|inner| {
                let mut holders: Vec<_> = inner
                    .procs
                    .iter()
                    .filter(|(_, proc)| proc.holder)
                    .map(|(pid, proc)| (std::cmp::Reverse(proc.rank()), *pid))
                    .collect();
                holders.sort_unstable();
                holders
            })
            .into_iter()
            .flat_map(|(_, pid)| {
                let head = holder_head(pid);
                crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine)
                    .into_iter()
                    .map(move |ws| (ws, head))
            })
            .collect();
        // Tails only: a set keyed whole stays where it is (a sweep moves
        // every suspended set it is not told to keep).
        let (order, whole): (Vec<_>, Vec<_>) = ordered
            .into_iter()
            .partition(|&(_, head)| !tails_only || head > 0);
        if order.is_empty() && tails_only {
            self.sweeping.store(false, Ordering::Release);
            return false;
        }
        let mut keep: HashSet<_> = self
            .with_inner(|inner| {
                let mut pids: Vec<ProcessId> = inner
                    .procs
                    .iter()
                    .filter(|(_, proc)| proc.state == Residency::Restoring)
                    .map(|(pid, _)| *pid)
                    .collect();
                pids.extend(inner.unmet_head().map(|(_, waiter)| waiter.pid));
                pids
            })
            .into_iter()
            .flat_map(|pid| {
                crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine)
            })
            .collect();
        keep.extend(whole.into_iter().map(|(ws, _)| ws));
        let txn = (room > 0)
            .then(|| {
                if tails_only {
                    self.port
                        .prepare_tail_sweep(&keep, room, &order, wanted as usize)
                } else {
                    self.port
                        .prepare_sweep(&keep, room, &order, wanted as usize)
                }
            })
            .flatten();
        let Some(txn) = txn else {
            self.sweeping.store(false, Ordering::Release);
            self.sweep_exhausted.store(true, Ordering::Release);
            return false;
        };
        let pages = txn.page_count() as u32;
        self.with_inner(|inner| {
            inner.evicting.insert(
                SWEEP,
                Reclaim {
                    pages,
                    rs: RsUnits::default(),
                },
            )
        });
        exec::spawn_move_out(self.clone(), txn, "sweep", Self::sweep_done);
        true
    }

    fn sweep_done(self: &Arc<Self>, moved: bool) {
        self.with_inner(|inner| inner.settle_eviction(SWEEP));
        self.sweeping.store(false, Ordering::Release);
        if moved {
            self.re_arm_idle_reclaim();
        } else {
            // Not again until something changes: a failed copy would fail
            // the same way.
            self.sweep_exhausted.store(true, Ordering::Release);
        }
        self.poke();
    }

    pub fn locus(&self) -> (usize, usize) {
        self.port.locus()
    }

    pub(super) fn with_inner<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut inner = self.lock_inner();
        let result = f(&mut inner);
        self.returns
            .waiters
            .store(inner.queue.len(), Ordering::Release);
        self.nonresident
            .store(inner.nonresident_count(), Ordering::Release);
        inner.admit_unpreempted();
        result
    }

    /// [`Self::with_inner`] for `f` that touches no process's residency,
    /// rank or parking and no queue: what `with_inner` refreshes after it
    /// (the waiters, the nonresident count, the admissions a preemption
    /// held back) reads all the processes, and a fire's own count moves
    /// twice a lane a step.
    pub(super) fn with_counts<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        f(&mut self.lock_inner())
    }

    pub fn register(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            let seq = inner.next_seq;
            inner.next_seq += 1;
            let mut proc = Proc::new(seq);
            proc.key = crate::policy::schedule::key(pid);
            proc.held = crate::policy::schedule::held(pid);
            inner.procs.insert(pid, proc);
        });
    }

    /// Registers what holds a handle's working set between processes: idle
    /// from the start, so the first to want its pages takes them. It ranks
    /// past every process, the longest parked last of all: a turn resumes
    /// what it just left, so the handle left longest goes first, where the
    /// youngest process would. `parked` is what it owes the device, for a set
    /// that comes already evicted.
    pub fn register_holder(self: &Arc<Self>, pid: ProcessId, parked: Option<Demand>) {
        self.with_inner(|inner| {
            let seq = inner.next_seq;
            inner.next_seq += 1;
            let mut proc = Proc::new(0);
            proc.rest(seq);
            proc.kept = holder_key(pid);
            proc.holder = true;
            proc.admitted = true;
            proc.idle = true;
            if let Some(demand) = parked {
                proc.state = Residency::Evicted;
                proc.resident.store(false, Ordering::Release);
                proc.parked = Some(demand);
            }
            inner.procs.insert(pid, proc);
        });
        self.re_arm_idle_reclaim();
        self.poke();
    }

    /// Reads each holder's retention key again: the table has changed.
    pub fn rekey_holders(&self) {
        self.with_inner(|inner| {
            for (pid, proc) in inner.procs.iter_mut().filter(|(_, proc)| proc.holder) {
                proc.kept = holder_key(*pid);
            }
        });
    }

    /// Whether a holder's state is on the device, if it is one here.
    pub fn holder_resident(&self, pid: ProcessId) -> Option<bool> {
        self.with_inner(|inner| {
            let proc = inner.procs.get(&pid).filter(|proc| proc.holder)?;
            Some(proc.state == Residency::Resident)
        })
    }

    /// A launch is taking this holder's set: no one's victim from here.
    pub fn claim_holder(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.idle = false;
            }
        });
    }

    /// The launch did not happen.
    pub fn unclaim_holder(self: &Arc<Self>, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.idle = true;
            }
        });
        self.poke();
    }

    /// Whether an eviction has this holder's set in hand.
    /// Waits until `pid` is not being moved off the device: its eviction
    /// ends (or rolls back) and tells its signal. At once where it is not.
    pub async fn wait_not_moving(&self, pid: ProcessId) {
        loop {
            let Some(signal) = self
                .lock_inner()
                .procs
                .get(&pid)
                .map(|proc| proc.signal.clone())
            else {
                return;
            };
            let notified = signal.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if !self.holder_moving(pid) {
                return;
            }
            notified.await;
        }
    }

    pub fn holder_moving(&self, pid: ProcessId) -> bool {
        self.lock_inner()
            .procs
            .get(&pid)
            .is_some_and(|proc| proc.state == Residency::Evicting)
    }

    /// `pid` took sets that are off the device: it restores them before it
    /// runs, as if it had been evicted with them.
    pub fn owe(self: &Arc<Self>, pid: ProcessId, demand: Demand) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            proc.state = Residency::Evicted;
            // Held below the schedule's floor: parked with what it owes,
            // restored once the policy lifts it.
            if proc.held {
                let owed = proc.parked.take().unwrap_or_default();
                proc.parked = Some(Demand {
                    kv_pages: owed.kv_pages.saturating_add(demand.kv_pages),
                    rs: owed.rs.add(demand.rs),
                });
                return;
            }
            inner.enqueue(pid, WaitKind::Restore { demand });
        });
        self.poke();
    }

    pub fn register_with_seq(&self, pid: ProcessId, seq: u64) {
        self.with_inner(|inner| {
            let mut proc = Proc::new(seq);
            proc.key = crate::policy::schedule::key(pid);
            proc.held = crate::policy::schedule::held(pid);
            proc.respawn = inner
                .killing
                .iter()
                .any(|killed| inner.procs.get(killed).is_some_and(|p| p.seq == seq));
            inner.procs.insert(pid, proc);
        });
    }

    pub fn spawn_seq(&self, pid: ProcessId) -> Option<u64> {
        self.lock_inner().procs.get(&pid).map(|proc| proc.seq)
    }

    /// Holds `pid` back from starting while an older process is preempted,
    /// so new work does not take the pages that process needs to come back.
    pub async fn admit(&self, pid: ProcessId) {
        loop {
            let notify = self.with_inner(|inner| {
                let rank = inner.procs.get(&pid)?.rank();
                if inner.preempted_floor().is_none_or(|floor| rank <= floor) {
                    return None;
                }
                let notify = Arc::new(Notify::new());
                inner.admitting.insert((rank, pid), notify.clone());
                Some(notify)
            });
            match notify {
                Some(notify) => {
                    if !waits_said(notify.notified()).await {
                        let holders: Vec<String> = self.with_inner(|inner| {
                            inner
                                .procs
                                .iter()
                                .filter(|(holder, proc)| {
                                    (proc.state != Residency::Resident && proc.parked.is_none())
                                        || proc.respawn
                                        || inner.killing.contains(holder)
                                })
                                .map(|(holder, proc)| format!("{holder}: {:?}", proc.state))
                                .collect()
                        });
                        tracing::warn!(
                            %pid,
                            ?holders,
                            "a process waits at admission behind a preempted one"
                        );
                    }
                }
                None => break,
            }
        }
        self.admit_state(pid).await;
        if self.seat_to_end {
            self.admit_to_end(pid).await;
        }
    }

    /// The processes the state pool runs at once: each holds
    /// [`FIXED_SETS_PER_WORKING_SET`] fixed sets, and the sets an admitted
    /// process holds past those (a fork's copy-on-write sets, a second
    /// context's) seat no one else while held. At least one runs. `None`
    /// where the model keeps no fixed state.
    pub(super) fn state_seats(&self) -> Option<usize> {
        let (_, total) = self.port.rs_stats();
        if total.slots == 0 {
            return None;
        }
        let (model, engine) = self.port.locus();
        let charged: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.charged)
                .map(|(&pid, _)| pid)
                .collect()
        });
        // Read off the planner's lock: the holdings take the store's.
        let past: u32 = charged
            .into_iter()
            .map(|pid| {
                held_rs_units(pid, model, engine)
                    .slots
                    .saturating_sub(FIXED_SETS_PER_WORKING_SET)
            })
            .sum();
        Some((total.slots.saturating_sub(past) / FIXED_SETS_PER_WORKING_SET).max(1) as usize)
    }

    /// Holds `pid` back until the state pool has the fixed set it holds
    /// free of those the admitted hold or are owed, so a pool of fixed state
    /// never runs more working sets than it can write: the runtime frees no
    /// state on its own to make room for one admitted past it. A fork's
    /// copy-on-write set is no seat's: its fire asks for it as any demand
    /// is asked, and waits or is refused by name there.
    async fn admit_state(&self, pid: ProcessId) {
        loop {
            let notified = self.seat_room.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Read again on each wake: the pool may have backed more, or a
            // fork let its copies go.
            let (_, total) = self.port.rs_stats();
            if total.slots == 0 {
                return;
            }
            // The holdings are read only where the whole pool would seat it.
            let most = (total.slots / FIXED_SETS_PER_WORKING_SET).max(1) as usize;
            let room = self.with_inner(|inner| {
                let charged = inner.procs.values().filter(|proc| proc.charged).count();
                inner
                    .procs
                    .get(&pid)
                    .is_none_or(|proc| proc.charged || charged < most)
            });
            let seats = if room { self.state_seats() } else { None };
            let passed = seats.is_some_and(|seats| {
                self.with_inner(|inner| {
                    let charged = inner.procs.values().filter(|proc| proc.charged).count();
                    let Some(proc) = inner.procs.get_mut(&pid) else {
                        return true;
                    };
                    if proc.charged || charged < seats {
                        proc.charged = true;
                        return true;
                    }
                    false
                })
            });
            if passed {
                return;
            }
            if !waits_said(notified).await {
                let charged = self
                    .with_inner(|inner| inner.procs.values().filter(|proc| proc.charged).count());
                let seats = seats.unwrap_or(most);
                tracing::warn!(
                    %pid,
                    seats,
                    charged,
                    "a process waits at admission for a state seat"
                );
            }
        }
    }

    /// What `pid` holds once `demand` lands, kept as its peak.
    pub fn note_peak(&self, pid: ProcessId, demand: Demand) {
        if !self.seat_to_end {
            return;
        }
        let (model, engine) = self.port.locus();
        let kv = held_page_count(pid, model, engine) + demand.kv_pages;
        let rs = super::holdings::held_rs_units(pid, model, engine).add(demand.rs);
        if let Some(proc) = self.lock_inner().procs.get_mut(&pid) {
            if proc.peak.0 == 0 && proc.peak.1.is_zero() {
                proc.first = (kv, rs);
            }
            proc.peak = (
                proc.peak.0.max(kv),
                RsUnits {
                    slots: proc.peak.1.slots.max(rs.slots),
                    pages: proc.peak.1.pages.max(rs.pages),
                    ..RsUnits::default()
                },
            );
        }
    }

    /// Holds `pid` back, oldest first, until the pools hold it to the end
    /// beside the processes already started.
    async fn admit_to_end(&self, pid: ProcessId) {
        loop {
            let notified = self.to_end_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let (_, kv_total) = self.port.device_stats();
            let (_, rs_total) = self.port.rs_stats();
            let now = std::time::Instant::now();
            let passed = self.with_inner(|inner| {
                let Some(rank) = inner.procs.get(&pid).map(Proc::rank) else {
                    return Ok(());
                };
                inner.to_end.waiting.insert((rank, pid));
                // By pid: a reschedule may have moved its rank since it queued.
                if inner
                    .to_end
                    .waiting
                    .first()
                    .is_none_or(|(_, first)| *first != pid)
                {
                    return Err(None);
                }
                inner.to_end.room(&inner.procs, kv_total, rs_total, now)?;
                inner.to_end.waiting.retain(|(_, waiting)| *waiting != pid);
                inner.to_end.live.insert(pid, now);
                Ok(())
            });
            match passed {
                Ok(()) => {
                    self.to_end_notify.notify_waiters();
                    return;
                }
                Err(None) => notified.await,
                Err(Some(spread)) => {
                    let _ = crate::rt::time::timeout(spread, notified).await;
                }
            }
        }
    }

    /// Queues the restore of a parked `pid` because a message for it is on
    /// its way. It stays idle until it reads the message, so a process
    /// restored early remains the first victim and never pins pages.
    pub fn wake(self: &Arc<Self>, pid: ProcessId) {
        let queued = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            if proc.held || proc.seatless {
                return None;
            }
            let demand = proc.parked.take()?;
            inner.enqueue(pid, WaitKind::Restore { demand });
            Some(())
        });
        if queued.is_some() {
            self.poke();
        }
    }

    /// Marks `pid` as waiting at a bind for room, or past it
    /// ([`super::proc::Proc::bind_waiting`]): waiting, it is a victim as an
    /// idle process is, and the planner looks again for a head it may now
    /// fund.
    pub fn set_bind_waiting(self: &Arc<Self>, pid: ProcessId, waiting: bool) {
        let (changed, released) = self.with_inner(|inner| {
            let changed = inner.procs.get_mut(&pid).is_some_and(|proc| {
                let was = proc.bind_waiting;
                proc.bind_waiting = waiting;
                was != waiting
            });
            // The suspended bind it was restored for waits for room again:
            // its restore is over, and the next bind-suspended restore may
            // go. Held for it, every other one waited on a bind that waits
            // behind them (Mac Pro's beam mix: a restored process short at
            // its bind again, the queue's head a bind-suspended restore,
            // flat with every thread parked).
            let released = waiting && inner.bind_restoring == Some(pid);
            if released {
                inner.bind_restoring = None;
            }
            (changed, released)
        });
        if (changed && waiting) || released {
            self.poke();
        }
    }

    /// Marks `pid` as waiting on its client, or as awake again. Waking a
    /// parked process queues its restore, ahead of its next fire.
    pub fn set_idle(self: &Arc<Self>, pid: ProcessId, idle: bool) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            proc.idle = idle;
            if self.seat_to_end {
                self.to_end_notify.notify_waiters();
            }
            let parked = if idle || proc.held || proc.seatless {
                None
            } else {
                proc.parked.take()
            };
            if let Some(demand) = parked {
                inner.enqueue(pid, WaitKind::Restore { demand });
            }
        });
        self.poke();
    }

    pub fn reschedule(self: &Arc<Self>) {
        let schedule = crate::policy::schedule::current();
        self.with_inner(|inner| {
            inner.reschedule(|pid| {
                schedule
                    .as_ref()
                    .map_or((0, false), |s| (s.key(*pid), s.held(*pid)))
            });
        });
        self.vacate_held_seats();
        self.pause_due.store(true, Ordering::Release);
        self.poke();
    }

    /// A paused process must not keep out one the schedule chose.
    pub(super) fn vacate_held_seats(&self) {
        if !crate::inferlet::process::seats_capped() {
            return;
        }
        let pids: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter_mut()
                .filter(|(_, p)| {
                    p.held && !p.seatless && p.state == Residency::Evicted && p.parked.is_some()
                })
                .map(|(pid, p)| {
                    p.seatless = true;
                    *pid
                })
                .collect()
        });
        for pid in pids {
            crate::inferlet::process::vacate_seat(pid);
        }
    }

    pub(super) fn reseat_released(self: &Arc<Self>) {
        let pids: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter_mut()
                .filter(|(_, p)| {
                    p.seatless
                        && !p.reseating
                        && !p.held
                        && p.state == Residency::Evicted
                        && p.parked.is_some()
                })
                .map(|(pid, p)| {
                    p.reseating = true;
                    *pid
                })
                .collect()
        });
        for pid in pids {
            exec::spawn_reseat(self.clone(), pid);
        }
    }

    pub(super) fn reseated(self: &Arc<Self>, pid: ProcessId, seat: crate::inferlet::process::Seat) {
        let queued = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            proc.reseating = false;
            if proc.held || proc.state != Residency::Evicted {
                return None;
            }
            let demand = proc.parked.take()?;
            proc.seatless = false;
            inner.enqueue(pid, WaitKind::Restore { demand });
            Some(())
        });
        if queued.is_some() {
            crate::inferlet::process::reseat(pid, seat);
            self.poke();
        }
    }

    /// The next plan tries again; poking here would recurse off the runtime.
    pub(super) fn reseat_failed(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.reseating = false;
            }
        });
    }

    pub fn can_pause(&self) -> bool {
        self.port.suspend_capable()
    }

    /// Suspends `pid` to give back the room its parked items hold on the
    /// device: a bind of its short of room it holds part of itself would
    /// wait on itself ([`crate::store::roles::bind_when_resident`]).
    /// Whether a suspension began. The bind asks `pieces` runs of `run`
    /// adjacent chunks: its restore waits for them beside its kv pages
    /// ([`super::proc::Proc::bind_room`]), never served into the same
    /// shortfall.
    pub fn suspend_for_room(self: &Arc<Self>, pid: ProcessId, run: u32, pieces: u32) -> bool {
        let need = PendingBind {
            run: run.max(1),
            pieces,
            pages: self.port.chunks_as_pages(run.max(1).saturating_mul(pieces)),
        };

        // A bind the capacity gate holds waits for the gate in the planner:
        // suspended, it parked its process for room that was there, and
        // came back to the same hold.
        if self.capacity_would_hold(pid, run.max(1).saturating_mul(pieces)) {
            return false;
        }
        let before = self.stats.evictions_started.load(Ordering::Relaxed);
        ptrace!(
            "suspend-for-room pid={} run={} pieces={}",
            pid,
            need.run,
            need.pieces
        );
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.bind_room = Some(need);
            }
        });
        self.pause(vec![pid], |_| true);
        // Not suspended, it waits as any bind short of room: no restore
        // reads the room.
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid)
                && proc.state == Residency::Resident
            {
                proc.bind_room = None;
            }
        });
        let suspended = self.stats.evictions_started.load(Ordering::Relaxed) > before;
        // Its suspension begun changes who heads the queue (a restore ahead
        // of new work goes first only while the fleet runs): the waiters
        // plan on it now, not on its landing (the simulation's lost wakes:
        // allocations served at once by a pass forced after a bind's
        // suspend, none planned since).
        if suspended && self.returns.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        }
        suspended
    }

    /// A worker with no host or disk tier leaves it running.
    pub(super) fn pause_held(self: &Arc<Self>) {
        let held: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.held)
                .map(|(pid, _)| *pid)
                .collect()
        });
        self.pause(held, |proc| proc.held);
    }

    /// Suspends those of `pids` that are running and a lower tier can hold.
    fn pause(self: &Arc<Self>, pids: Vec<ProcessId>, still: fn(&Proc) -> bool) {
        if !self.port.suspend_capable() {
            ptrace!("pause: cannot suspend");
            return;
        }
        let pausable: Vec<(ProcessId, Rank)> = self.with_inner(|inner| {
            pids.iter()
                .filter_map(|pid| {
                    let proc = inner.procs.get(pid)?;
                    (proc.admitted
                        && proc.state == Residency::Resident
                        && !inner.is_finisher(*pid)
                        && !inner.host_swap_blocked.contains(pid)
                        && !inner.prepare_blocked.contains(pid))
                    .then(|| (*pid, proc.rank()))
                })
                .collect()
        });
        ptrace!("pause candidates={}", pausable.len());
        if pausable.is_empty() {
            return;
        }
        let (model, engine) = self.port.locus();
        let everything = Reclaim {
            pages: u32::MAX,
            rs: RsUnits {
                slots: u32::MAX,
                pages: u32::MAX,
                ..RsUnits::default()
            },
        };
        let (picks, unhostable) =
            self.quote_and_pick(pausable, everything, self.swap_room(), 0, model, engine);
        // No tier takes them: they keep running.
        ptrace!(
            "pause picks={} unhostable={} room={:?}",
            picks.len(),
            unhostable.len(),
            self.swap_room()
        );
        let (wake, spawned) = self.with_inner(|inner| {
            let (mut wake, mut spawned) = (Vec::new(), Vec::new());
            for (pid, expected) in picks {
                if !inner
                    .procs
                    .get(&pid)
                    .is_some_and(|proc| still(proc) && proc.state == Residency::Resident)
                {
                    continue;
                }
                wake.extend(inner.begin_eviction(pid, expected));
                spawned.push(pid);
            }
            (wake, spawned)
        });
        self.start_evictions(wake, spawned);
    }

    pub(super) fn start_evictions(self: &Arc<Self>, wake: Vec<Arc<Notify>>, pids: Vec<ProcessId>) {
        for notify in wake {
            notify.notify_waiters();
        }
        let unfired = self.with_inner(|inner| {
            pids.iter()
                .filter(|pid| inner.procs.get(pid).is_some_and(|proc| proc.unfired))
                .count() as u64
        });
        self.stats.evicted_unfired.fetch_add(unfired, Ordering::Relaxed);
        for pid in pids {
            self.stats.evictions_started.fetch_add(1, Ordering::Relaxed);
            exec::spawn_evict(self.clone(), pid);
        }
    }

    pub fn note_admitted(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.admitted = true;
                proc.respawn = false;
            }
        });
    }

    /// The hold `pid` keeps on its place here past its exit (see
    /// [`Departure`]): the same one for every holder while any lives, none
    /// once `pid` has left.
    pub fn departure(self: &Arc<Self>, pid: ProcessId) -> Option<Arc<Departure>> {
        self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            if let Some(departure) = proc.departure.upgrade() {
                return Some(departure);
            }
            let departure = Arc::new(Departure::new(self, pid));
            proc.departure = Arc::downgrade(&departure);
            Some(departure)
        })
    }

    /// `pid`'s last hold dropped: what its working sets held is back.
    pub(super) fn departed(self: &Arc<Self>, pid: ProcessId) {
        let settled = self.with_inner(|inner| {
            !inner.procs.contains_key(&pid) && inner.evicting.remove(&pid).is_some()
        });
        if settled {
            self.re_arm_idle_reclaim();
            self.poke();
        }
    }

    /// `pid` leaves: its client has heard, its seat and asks go now. What
    /// its working sets hold stays on its way back, as an eviction's would,
    /// until its last [`Departure`] hold drops (each set's release drops its
    /// own, however long a fire leases it), so the fleet is not read as
    /// wedged for pages no one registered holds.
    pub fn unregister(self: &Arc<Self>, pid: ProcessId) {
        crate::store::shared::forget_process(pid);
        // What was held for its restore goes back with it.
        self.port.release_restore(pid);
        let (model, engine) = self.port.locus();
        let held = held_page_count(pid, model, engine);
        let held_rs = held_rs_units(pid, model, engine);
        // A seat taken again after the process ended goes back.
        crate::inferlet::process::vacate_seat(pid);
        let (signal, removed, charged) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
            if inner.bind_restoring == Some(pid) {
                inner.bind_restoring = None;
            }
            let charged = departed.as_ref().is_some_and(|proc| proc.charged);
            if departed.is_some() {
                inner.admitting.retain(|(_, waiting), _| *waiting != pid);
            }
            if self.seat_to_end {
                if let Some(passed) = inner.to_end.live.remove(&pid)
                    && let Some(proc) = departed.as_ref().filter(|proc| {
                        proc.admitted
                            && proc.state == Residency::Resident
                            && !proc.respawn
                            && proc.peak.0 > 0
                    })
                {
                    inner.to_end.learn(proc.peak, proc.first, passed.elapsed());
                }
                inner.to_end.waiting.retain(|(_, waiting)| *waiting != pid);
            }
            if departed.as_ref().is_some_and(|proc| {
                proc.admitted && !proc.holder && proc.state == Residency::Resident
            }) {
                inner.completion_credit = inner.completion_credit.saturating_add(1);
            }
            inner.settle_eviction(pid);
            // Its holds outlive it only while one lives; the last one's drop
            // settles this, and cannot have run before it is put here.
            // Its state is on its way back as its pages are: a pool dry of
            // the slots its sets still hold is not wedged.
            if (held > 0 || !held_rs.is_zero())
                && departed
                    .as_ref()
                    .is_some_and(|proc| proc.departure.strong_count() > 0)
            {
                inner.evicting.insert(
                    pid,
                    Reclaim {
                        pages: held,
                        rs: held_rs,
                    },
                );
            }
            let signal = departed.map(|proc| proc.signal);
            inner.killing.remove(&pid);
            inner.host_swap_blocked.clear();
            inner.prepare_blocked.clear();
            let keys: Vec<EntryKey> = inner
                .queue
                .iter()
                .filter(|(_, waiter)| waiter.pid == pid)
                .map(|(key, _)| *key)
                .collect();
            let removed: Vec<Waiter> = keys
                .iter()
                .filter_map(|key| inner.queue.remove(key))
                .collect();
            (signal, removed, charged)
        });
        if charged {
            self.seat_room.notify_waiters();
        }
        for waiter in &removed {
            if let WaitKind::Allocation { notify, .. } = &waiter.kind {
                notify.notify_waiters();
            }
        }
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        drop(removed);
        if self.seat_to_end {
            self.to_end_notify.notify_waiters();
        }
        self.re_arm_idle_reclaim();
        self.poke();
    }

    /// Whether a process's demand waits for room: a kv allocation, a
    /// restore or a guest bind, unmet. A rise is ordered among them by its
    /// own key and is not counted: it is what the room it waits on goes to
    /// in turn, not what a rung walk gives room back for.
    #[must_use]
    pub fn demand_waits(&self) -> bool {
        self.with_inner(|inner| {
            inner.queue.values().any(|waiter| {
                waiter.is_unmet()
                    && !matches!(
                        waiter.kind,
                        super::wait::WaitKind::Allocation { rise: true, .. }
                            | super::wait::WaitKind::BindRoom { rise: Some(_), .. }
                    )
            })
        })
    }

    /// What [`Self::pages_freed`] does short of planning: the seats are
    /// read again, idle reclaim re-armed.
    pub(super) fn pages_freed_quietly(&self) {
        self.port.freed();
        self.seat_room.notify_waiters();
        self.re_arm_idle_reclaim();
    }

    /// The queue's head is a restore refused on room that has not moved
    /// since: no more chunks free within the state budget than its refusal
    /// left ([`super::proc::Proc::roles_short_free`]), the key it goes again
    /// on. A pass for it finds no step.
    pub(super) fn head_refused_on_room_unmoved(&self) -> bool {
        let free = self.port.free_chunks();
        self.with_inner(|inner| {
            inner.unmet_head().is_some_and(|(_, waiter)| {
                matches!(waiter.kind, WaitKind::Restore { .. })
                    && inner.procs.get(&waiter.pid).is_some_and(|proc| {
                        proc.roles_short.is_some()
                            && proc.roles_short_free.is_some_and(|left| free <= left)
                    })
            })
        })
    }

    pub fn pages_freed(self: &Arc<Self>) {
        self.port.freed();
        // A set let go or a segment backed: admission reads its seats again.
        self.seat_room.notify_waiters();
        self.re_arm_idle_reclaim();
        if self.returns.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        }
    }

    pub(super) fn try_reserve(&self, demand: Demand) -> Option<AllocationGrant> {
        let kv = if demand.kv_pages > 0 {
            let pages = self.port.reserve_device(demand.kv_pages)?;
            DevicePageReservation::new(pages, self.port.clone(), self.returns.clone())
        } else {
            DevicePageReservation::empty()
        };
        let rs = self.reserve_rs_slots(demand.rs)?;
        Some(AllocationGrant::new(kv, rs))
    }

    pub(super) fn reserve_rs_slots(&self, need: RsUnits) -> Option<RsSlotReservation> {
        if need.is_zero() {
            return Some(RsSlotReservation::empty());
        }
        let grant = self.port.reserve_rs(need)?;
        Some(RsSlotReservation::new(grant, self.port.clone(), self.returns.clone()))
    }

    pub(super) fn absorb_free(&self, count: u32) -> bool {
        let pages = self.port.reserve_device_up_to(count);
        if pages.is_empty() {
            return false;
        }
        let reservation = DevicePageReservation::new(pages, self.port.clone(), self.returns.clone());
        self.with_inner(|inner| inner.accum.absorb(reservation));
        true
    }
}

/// What a holder ranks by as a victim. Moving it whole takes its state
/// too (on a shared pool its record, rs list and window slots), which every
/// token it keeps reads: so it ranks by the highest key on its set, the
/// key on its head where a policy keeps one (`Retain::upto`). A tail the
/// head's key does not cover goes first by itself ([`holder_head`], the
/// sweep), its pages at the whole set's key.
fn holder_key(pid: ProcessId) -> i64 {
    use crate::policy::retention::{Installed, Keys};
    Installed.key_at(Some(pid), 0)
}

/// The tokens a holder's keys keep above the whole of it.
fn holder_head(pid: ProcessId) -> u64 {
    use crate::policy::retention::{Installed, Keys};
    let whole = Installed.key_at(Some(pid), u64::MAX);
    Installed
        .steps(pid)
        .into_iter()
        .filter(|&upto| upto > 0 && Installed.key_at(Some(pid), upto - 1) > whole)
        .max()
        .unwrap_or(0)
}

/// What one idle reclaim came to ([`ResidencyPlanner::reclaim_idle_once`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum IdleOutcome {
    /// Something went: the head looks again in what it left.
    Freed,
    /// Nothing more went: pages earlier reclaims freed cover what is short
    /// and wait on a fire's retirement. The head waits for it (the settle
    /// that retires them wakes the planner), never on a clock, and nothing
    /// else is taken for it meanwhile.
    Returning,
    /// Nothing went and nothing is returning.
    Nothing,
}
