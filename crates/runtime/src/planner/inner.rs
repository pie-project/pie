use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::ProcessId;
use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::proc::{Proc, Residency};
use super::rank::Rank;
use super::reclaim::Reclaim;
use super::wait::{EntryKey, WaitKind, Waiter};

#[derive(Default)]
pub(super) struct Inner {
    /// The process suspended for a bind whose restore was served and that
    /// has not fired since: one at a time. Restored together, each took its
    /// items back and every bind came up short against the others' (the M1
    /// stress: ten processes suspended, restored and short in turn, 190
    /// times each); restored alone, a bind competes only with the
    /// residents.
    pub(super) bind_restoring: Option<ProcessId>,
    pub(super) next_seq: u64,
    next_id: u64,
    pub(super) procs: HashMap<ProcessId, Proc>,
    pub(super) queue: BTreeMap<EntryKey, Waiter>,
    pub(super) accum: DevicePageReservation,
    /// Pages on their way back: a suspend's in flight, keyed by its process,
    /// or a departed process's still held by its working sets, keyed by the
    /// process gone, until its last [`super::Departure`] hold drops.
    pub(super) evicting: HashMap<ProcessId, Reclaim>,
    pub(super) completion_credit: u32,
    pub(super) host_swap_blocked: HashSet<ProcessId>,
    pub(super) prepare_blocked: HashSet<ProcessId>,
    /// The pool's room count and the swap tiers' free slots when the last
    /// of those blocks was set ([`super::ResidencyPlanner::lapse_blocks`]):
    /// a block holds only on the pool it was set on.
    pub(super) blocked_at: Option<(u64, u32)>,
    pub(super) killing: HashSet<ProcessId>,
    pub(super) admitting: BTreeMap<(Rank, ProcessId), Arc<Notify>>,
    pub(super) to_end: super::to_end::ToEnd,
    /// The evicted processes whose parked state is on its way down a tier
    /// ([`super::ResidencyPlanner::park_down`]): none is restored until it
    /// lands, so no restore misses an item moving under it.
    pub(super) parked_moving: HashSet<ProcessId>,
    /// The unservable head the last stall dump named, the pool's room
    /// count then, and when ([`super::ResidencyPlanner::dump_stall`]): one
    /// dump per head and room change, and again each
    /// [`super::plan::STALL_SAYS_AGAIN`] while it stays unserved.
    pub(super) stall_dumped: Option<(super::wait::EntryKey, u64, crate::rt::Instant)>,
    /// The room a pass last planned a quiet head on (nothing in flight):
    /// the head and the pool's counts ([`super::ResidencyPlanner::tick`]).
    pub(super) quiet_planned: Option<(super::wait::EntryKey, [u64; 4])>,
    /// Waits served this pass past [`super::wait::LONG_WAIT`]: the kind and
    /// how long.
    pub(super) long_waits: Vec<(&'static str, std::time::Duration)>,
    /// The head held unserved on something in flight: since when, on what
    /// room, and whether that was said ([`super::ResidencyPlanner::tick`]).
    pub(super) held_in_flight:
        Option<(super::wait::EntryKey, u64, crate::rt::Instant, bool)>,
}

impl Inner {
    pub(super) fn kill_in_flight(&self) -> bool {
        self.killing.iter().any(|pid| {
            self.procs.contains_key(pid)
                && !self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == *pid && waiter.is_unmet())
        })
    }

    /// The pages that, moved out whole, free the runs of chunks `waiter`
    /// asks past its kv: a bind short of guest-state room, or the restore
    /// of a process such a bind suspended ([`super::proc::Proc::bind_room`]).
    pub(super) fn room_pages(&self, waiter: &super::wait::Waiter) -> u32 {
        let pending = match waiter.kind {
            // A bind its suspension waits for, or role slots its last
            // unpark was short of: pages that, moved out, free their chunks.
            WaitKind::Restore { .. } => self.procs.get(&waiter.pid).map_or(0, |proc| {
                let bind = proc.bind_room.map_or(0, |need| need.pages);
                let roles = proc.roles_short.map_or(0, |need| need.pages);
                bind.max(roles)
            }),
            _ => 0,
        };
        waiter.room_pages().max(pending)
    }

    pub(super) fn settle_eviction(&mut self, pid: ProcessId) {
        self.evicting.remove(&pid);
    }

    /// What the evictions in flight are expected to free.
    pub(super) fn expected_reclaim(&self) -> Reclaim {
        self.evicting
            .values()
            .fold(Reclaim::default(), |sum, &mark| sum.add(mark))
    }

    pub(super) fn nonresident_count(&self) -> usize {
        let mut n = 0;
        for proc in self.procs.values() {
            let resident = proc.state == Residency::Resident;
            // A handle at rest off the device is the point of parking it.
            if !resident && !proc.holder {
                n += 1;
            }
            proc.resident.store(resident, Ordering::Release);
        }
        n
    }

    pub(super) fn preempted_floor(&self) -> Option<Rank> {
        self.procs
            .iter()
            .filter(|(pid, proc)| {
                (proc.state != Residency::Resident && proc.parked.is_none())
                    || proc.respawn
                    || self.killing.contains(pid)
            })
            .map(|(_, proc)| proc.rank())
            .min()
    }

    pub(super) fn admit_unpreempted(&mut self) {
        if self.admitting.is_empty() {
            return;
        }
        let floor = self.preempted_floor();
        let (go, blocked): (BTreeMap<_, _>, BTreeMap<_, _>) = std::mem::take(&mut self.admitting)
            .into_iter()
            .partition(|((rank, _), _)| floor.is_none_or(|floor| *rank <= floor));
        self.admitting = blocked;
        for notify in go.into_values() {
            notify.notify_one();
        }
    }

    /// What an unseated claim on `need` leaves free for seated lanes to grow into
    /// (`growth` each), so seating a prompt never leaves one short at its next page.
    pub(super) fn seating_reserve(
        &self,
        pid: ProcessId,
        growth: RsUnits,
        need: RsUnits,
        capacity: RsUnits,
    ) -> RsUnits {
        // Nothing in flight: no seated lane grows before the claim.
        if need.is_zero()
            || growth.is_zero()
            || self.procs.get(&pid).is_none_or(|proc| proc.seated)
            || self.nothing_in_flight()
        {
            return RsUnits::default();
        }
        // The seated lanes whose next step needs room now: seated, running
        // and resident, not waiting in the queue (one waiting asks for its
        // room there, and grows before nothing). Counting every seated lane,
        // or yielding only once the fleet read stalled by its waiters'
        // kinds, held back the growth of lanes that were all blocked, and an
        // allocation with 38 free runs was not granted (the M1).
        let waiting: HashSet<ProcessId> = self
            .queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .map(|waiter| waiter.pid)
            .collect();
        let seated = u32::try_from(
            self.procs
                .iter()
                .filter(|(lane, proc)| {
                    proc.seated
                        && !proc.idle
                        && proc.state == Residency::Resident
                        && !waiting.contains(*lane)
                })
                .count(),
        )
        .unwrap_or(u32::MAX);
        // One growth for each seated lane that needs room now (its next
        // step's pages), never more than the pool.
        RsUnits {
            slots: growth.slots.saturating_mul(seated).min(capacity.slots),
            pages: growth.pages.saturating_mul(seated).min(capacity.pages),
            ..RsUnits::default()
        }
    }

    /// A seated lane's growth takes free pages at once rather than parking behind
    /// claims served after it anyway (a waiting prompt holding nothing); a
    /// higher-ranked seated waiter or preempted process still goes first.
    pub(super) fn growth_first(&self, pid: ProcessId) -> bool {
        let Some(proc) = self.procs.get(&pid) else {
            return false;
        };
        let rank = proc.rank();
        proc.state == Residency::Resident
            && !proc.respawn
            && !self.killing.contains(&pid)
            && self.queue.iter().all(|(&(waiter_rank, _), waiter)| {
                waiter.pid != pid
                    && (waiter_rank > rank
                        || !waiter.is_unmet()
                        || self.procs.get(&waiter.pid).is_some_and(|w| !w.seated))
            })
            && self.procs.iter().all(|(other, proc)| {
                proc.rank() > rank
                    || (proc.state == Residency::Resident
                        && !proc.respawn
                        && !self.killing.contains(other))
            })
    }

    pub(super) fn enqueue(&mut self, pid: ProcessId, kind: WaitKind) -> EntryKey {
        let key = (self.procs[&pid].rank(), self.next_id);
        self.next_id += 1;
        // The one restored for its bind asks for pages: its bind went
        // through, and the next suspended bind may come back. Held to its
        // first fire, the hold outlived a grant it waited on behind the
        // held restore (the M1 stress: 86 at the planner's grant, the
        // restore head held, nothing moved).
        if matches!(kind, WaitKind::Allocation { .. }) && self.bind_restoring == Some(pid) {
            self.bind_restoring = None;
        }
        self.queue.insert(
            key,
            Waiter {
                pid,
                kind,
                since: crate::rt::Instant::now(),
            },
        );
        key
    }

    /// Wherever a reschedule has since moved it.
    pub(super) fn locate(&self, key: EntryKey) -> Option<EntryKey> {
        if self.queue.contains_key(&key) {
            return Some(key);
        }
        self.queue.keys().find(|(_, id)| *id == key.1).copied()
    }

    pub(super) fn reschedule(&mut self, order_of: impl Fn(&ProcessId) -> (i64, bool)) {
        for (pid, proc) in self.procs.iter_mut().filter(|(_, proc)| !proc.holder) {
            (proc.key, proc.held) = order_of(pid);
        }
        let held_restores: Vec<EntryKey> = self
            .queue
            .iter()
            .filter(|(_, w)| matches!(w.kind, WaitKind::Restore { .. }))
            .filter(|(_, w)| self.procs.get(&w.pid).is_some_and(|p| p.held))
            .map(|(key, _)| *key)
            .collect();
        for key in held_restores {
            if let Some(Waiter {
                pid,
                kind: WaitKind::Restore { demand },
                ..
            }) = self.queue.remove(&key)
                && let Some(proc) = self.procs.get_mut(&pid)
            {
                proc.parked = Some(demand);
            }
        }
        let released: Vec<(ProcessId, Demand)> = self
            .procs
            .iter_mut()
            .filter(|(_, p)| !p.held && !p.idle && !p.seatless && p.state == Residency::Evicted)
            .filter_map(|(pid, p)| Some((*pid, p.parked.take()?)))
            .collect();
        for (pid, demand) in released {
            self.enqueue(pid, WaitKind::Restore { demand });
        }
        let procs = &self.procs;
        let rank = |pid: &ProcessId, was: Rank| procs.get(pid).map_or(was, Proc::rank);
        self.queue = std::mem::take(&mut self.queue)
            .into_iter()
            .map(|((was, id), waiter)| ((rank(&waiter.pid, was), id), waiter))
            .collect();
        self.admitting = std::mem::take(&mut self.admitting)
            .into_iter()
            .map(|((was, pid), notify)| ((rank(&pid, was), pid), notify))
            .collect();
    }

    pub(super) fn unmet_head(&self) -> Option<(EntryKey, &Waiter)> {
        let mut oldest_restore: Option<EntryKey> = None;
        let mut allocation: Option<EntryKey> = None;
        // A process evicted for a head that still waits unserved is not
        // restored before it (the anti-thrash rule's second half): its
        // restore is passed over, the queue not held, so the head it was
        // evicted for is planned for. Restored first, the victim was
        // evicted again for the same head every ~65 ms, the head never
        // served.
        // Read only where a restore of a victim is met (a process with an
        // evicted-for head): built on every call it was a scan and a set of
        // the whole queue each time, and this runs many times a pass.
        let waiting = std::cell::OnceCell::<HashSet<ProcessId>>::new();
        let waiting = || {
            waiting.get_or_init(|| {
                self.queue
                    .values()
                    .filter(|waiter| waiter.is_unmet())
                    .map(|waiter| waiter.pid)
                    .collect()
            })
        };
        // Its head served: no unmet waiter of its, and resident. A restore
        // head leaves the queue while its restore runs and comes back if
        // its unpark is short: read by its queue alone, it looked served
        // in that gap, and the victim's restore went first (the L40S: one
        // victim evicted 5,894 times for one restore head, never served).
        let unserved = |head: ProcessId| {
            waiting().contains(&head)
                || self
                    .procs
                    .get(&head)
                    .is_some_and(|proc| proc.state != Residency::Resident)
        };
        let held_for_head = |pid: ProcessId| {
            self.procs
                .get(&pid)
                .and_then(|proc| proc.evicted_for)
                .is_some_and(|head| head != pid && unserved(head))
        };
        // Evicted while it waited in the queue: held while any process
        // ahead of it (older) has an entry in the queue.
        let behind_queue = |pid: ProcessId| {
            let Some(seq) = self
                .procs
                .get(&pid)
                .filter(|proc| proc.evicted_queued)
                .map(|proc| proc.seq)
            else {
                return false;
            };
            self.queue.values().any(|other| {
                other.pid != pid
                    && !matches!(other.kind, WaitKind::Restore { .. })
                    && self.procs.get(&other.pid).is_some_and(|proc| proc.seq < seq)
            })
        };
        for (&key, waiter) in &self.queue {
            if !waiter.is_unmet() {
                continue;
            }
            if matches!(waiter.kind, WaitKind::Restore { .. })
                && (held_for_head(waiter.pid) || behind_queue(waiter.pid))
            {
                continue;
            }
            match &waiter.kind {
                // A bind short of room asks as an admission does.
                WaitKind::Allocation { .. } | WaitKind::BindRoom { .. } => {
                    allocation = Some(key);
                    break;
                }
                WaitKind::Restore { .. } => {
                    oldest_restore.get_or_insert(key);
                }
            }
        }
        let head = match (allocation, oldest_restore) {
            (Some(allocation), Some(restore)) => {
                // An older restore comes back ahead of new work: the
                // admission of a prompt not yet seated waits behind it
                // (Mac Pro's M5: processes evicted in the ramp waited
                // behind every later arrival, then froze the tail). A seated
                // process's own growth keeps its turn, or the decoders whose
                // completions free the restore's room would wait on it.
                let admits_new = self
                    .queue
                    .get(&allocation)
                    .and_then(|waiter| self.procs.get(&waiter.pid))
                    .is_some_and(|proc| !proc.seated);
                if restore < allocation
                    && (admits_new || self.fleet_stalled() || self.eviction_unfundable())
                {
                    restore
                } else {
                    allocation
                }
            }
            (Some(allocation), None) => allocation,
            (None, restore) => restore?,
        };
        self.queue
            .get_key_value(&head)
            .map(|(&key, waiter)| (key, waiter))
    }

    pub(super) fn queued_quantum(&self) -> u32 {
        self.queue
            .values()
            .filter(|waiter| waiter.is_unmet())
            .filter(|waiter| {
                matches!(&waiter.kind, WaitKind::Allocation { demand, .. } if demand.kv_pages == 1)
            })
            .count() as u32
    }

    pub(super) fn begin_eviction(&mut self, pid: ProcessId, expected: Reclaim) -> Vec<Arc<Notify>> {
        if let Some(proc) = self.procs.get_mut(&pid) {
            proc.state = Residency::Evicting;
        }
        self.evicting.insert(pid, expected);
        let mut wake = Vec::new();
        for waiter in self.queue.values_mut().filter(|w| w.pid == pid) {
            match &mut waiter.kind {
                WaitKind::Allocation {
                    notify, yielded, ..
                } => {
                    *yielded = true;
                    wake.push(notify.clone());
                }
                // Its own bind's wait for room ends: evicted, the bind
                // waits for its restore, and nothing is funded for it
                // meanwhile. Kept, the eviction waited on fires whose
                // finalizer was that bind, and the bind on the eviction
                // (Mac Pro's M5 soak: evicting=1 for good, pool free 0).
                WaitKind::BindRoom {
                    notify,
                    served: false,
                    withdrawn,
                    ..
                } => {
                    *withdrawn = true;
                    wake.push(notify.clone());
                }
                _ => {}
            }
        }
        wake
    }

    pub(super) fn eviction_unfundable(&self) -> bool {
        !self.host_swap_blocked.is_empty()
    }

    pub(super) fn burst_shortfall(&self) -> u32 {
        let mut supply = self.accum.len() as u32;
        let mut extra = 0u32;
        for waiter in self.queue.values() {
            if !waiter.is_unmet() {
                continue;
            }
            let WaitKind::Allocation { demand, .. } = &waiter.kind else {
                continue;
            };
            if !demand.rs.is_zero() {
                break;
            }
            let need = demand.kv_pages;
            let covered = need.min(supply);
            supply -= covered;
            extra = extra.saturating_add(need - covered);
        }
        extra
    }

    pub(super) fn serve_burst(&mut self) -> Vec<(EntryKey, u32, Arc<Notify>)> {
        let mut wake = Vec::new();
        let Some((head, waiter)) = self.unmet_head() else {
            return wake;
        };
        if !matches!(waiter.kind, WaitKind::Allocation { .. }) {
            return wake;
        }
        let Inner {
            queue,
            accum,
            long_waits,
            ..
        } = self;
        for (&key, waiter) in queue.range_mut(head..) {
            if !waiter.is_unmet() {
                continue;
            }
            let waited = waiter.since.elapsed();
            let WaitKind::Allocation {
                demand,
                notify,
                outcome,
                ..
            } = &mut waiter.kind
            else {
                continue;
            };
            let demand = *demand;
            if !demand.rs.is_zero() {
                break;
            }
            if (accum.len() as u32) < demand.kv_pages {
                break;
            }
            let kv = accum.donate(demand.kv_pages);
            debug_assert!(outcome.is_none(), "an unmet waiter carries no outcome");
            *outcome = Some(Ok(AllocationGrant::new(kv, RsSlotReservation::empty())));
            if waited > super::wait::LONG_WAIT {
                long_waits.push(("allocation", waited));
            }
            wake.push((key, demand.kv_pages, notify.clone()));
        }
        wake
    }

    /// Whether the restore of `pid`, the head, goes now, its kv gathered:
    /// suspended processes come back as soon as their room stands, fires
    /// in flight or none, every one it holds at once, not one a
    /// completion. One whose role slots found no room last time
    /// ([`super::proc::Proc::roles_short`]) goes again on the runs that
    /// stand free, and has room made for its runs (idle cache, a coming
    /// grow, then the occupants of the chunks it could take) only once
    /// nothing in flight can free room, a fleet waiting at binds included:
    /// funding it while the fleet decoded thrashed.
    pub(super) fn restore_gate(&mut self, pid: ProcessId) -> RestoreGate {
        let roles_short = self
            .procs
            .get(&pid)
            .filter(|proc| proc.roles_short.is_some())
            .and_then(super::proc::Proc::restore_need);
        if self.completion_credit > 0 {
            self.completion_credit -= 1;
            return RestoreGate::Serve;
        }
        let stalled = self.nothing_in_flight();
        // Short of runs for its roles: it goes again on runs that stand
        // free whatever the fleet does (the plan looks, outside this lock);
        // room is made for it by moving others only once nothing in flight
        // can free room (a fleet waiting at binds included). Waiting for a
        // completion while its runs stood free, it waited on an event that
        // need not come (Mac Pro's beam mix: 2 runs of 4 short, 440 such
        // runs free, flat for 191 s with every thread parked).
        // Funded too while another process waits at a bind behind it: that
        // bind goes only once the head is served, so the head's room comes
        // from no completion of the waiter's, and a fleet whose decoders
        // never stop firing is never stalled (a roles-short head starved
        // behind a steady decoder with binds waiting on it).
        if let Some(need) = roles_short {
            return RestoreGate::Evict {
                need,
                fund: stalled || self.bind_waits_behind(pid),
            };
        }
        // Its kv gathered (the step before), its rs and its bind's runs
        // standing: it goes, fires in flight or none. Held for a page for
        // each running lane past the pages it had already gathered, it held
        // them idle while a fleet of 248 decoded, its need standing and
        // never attempted (Mac Pro's 2 GiB mix); seated lanes' growth goes
        // ahead of it anyway (`growth_first`).
        RestoreGate::Serve
    }

    /// The head restore waiting on room for its roles goes again: room
    /// was made for it ([`Step::EvictForRoles`](super::plan)).
    pub(super) fn roles_retry_head(&mut self) {
        let Some((_, waiter)) = self.unmet_head() else {
            return;
        };
        let pid = waiter.pid;
        if let Some(proc) = self.procs.get_mut(&pid) {
            proc.roles_short = None;
        }
    }

    /// Whether `pid`'s anti-thrash mark ([`super::proc::Proc::unfired`])
    /// keeps it out of the victims: only while it can fire, by state, never
    /// a clock. Waiting in the queue itself (its lanes blocked on the
    /// planner, which serves the head first) or with nothing in flight
    /// anywhere (nothing of it runs), it cannot, and protected it would hold
    /// the head that waits on its room for good: it is a candidate.
    pub(super) fn unfired_protects(&self, pid: ProcessId) -> bool {
        self.procs.get(&pid).is_some_and(|proc| proc.unfired)
            && !self
                .queue
                .values()
                .any(|waiter| waiter.pid == pid && waiter.is_unmet())
            && !self.nothing_in_flight()
    }

    /// Whether the one bind-suspended restore at a time
    /// ([`Self::bind_restoring`]) no longer holds the next: its process
    /// cannot fire (nothing in flight anywhere, or it waits in the queue on
    /// something other than its bind) and its bind is not queued for room
    /// (which the gate funds first). By state, never a clock: held while it
    /// could not fire, every other bind-suspended restore waited for good
    /// with its runs standing (Mac Pro's no-step lines with no refusal).
    pub(super) fn bind_restoring_released(&self) -> bool {
        let Some(restoring) = self.bind_restoring else {
            return true;
        };
        let mut bind_queued = false;
        let mut waits = false;
        for waiter in self.queue.values().filter(|waiter| waiter.pid == restoring) {
            if !waiter.is_unmet() {
                continue;
            }
            match waiter.kind {
                WaitKind::BindRoom { .. } => bind_queued = true,
                _ => waits = true,
            }
        }
        !bind_queued && (waits || self.nothing_in_flight())
    }

    /// Nothing in flight can free room: no fire submitted and not landed,
    /// no eviction, restore or kill under way. What the others wait on
    /// (grants, binds, gates, anything not yet named) does not matter:
    /// with nothing moving, only the planner moving a resident makes room.
    /// Counting waiter kinds instead missed one each time (stalled, then
    /// waiting at binds, then one resident between steps), and the fleet
    /// stood still with nothing coming.
    pub(super) fn nothing_in_flight(&self) -> bool {
        self.evicting.is_empty()
            && !self.kill_in_flight()
            && self.procs.values().all(|proc| {
                !proc.firing()
                    && !matches!(proc.state, Residency::Evicting | Residency::Restoring)
            })
    }

    /// Whether a process other than `head` waits at a bind for room
    /// ([`super::proc::Proc::bind_waiting`]): read by each process's own
    /// flag, never by counting the queue's waiter kinds.
    pub(super) fn bind_waits_behind(&self, head: ProcessId) -> bool {
        self.procs
            .iter()
            .any(|(pid, proc)| *pid != head && proc.bind_waiting)
    }

    pub(super) fn fleet_stalled(&self) -> bool {
        if !self.evicting.is_empty() {
            return false;
        }
        let mut parked = HashSet::new();
        for waiter in self.queue.values() {
            match &waiter.kind {
                WaitKind::Allocation {
                    outcome: Some(_), ..
                } => return false,
                _ => {
                    parked.insert(waiter.pid);
                }
            }
        }
        self.procs.iter().all(|(pid, proc)| match proc.state {
            // Parked, it is stalled only with no fire of its own in flight:
            // one parked lane beside its peers' running fires waits for them.
            Residency::Resident => {
                !proc.admitted || proc.idle || (parked.contains(pid) && !proc.firing())
            }
            Residency::Evicting | Residency::Restoring => false,
            Residency::Evicted => true,
        })
    }
}

/// What the head restore does now ([`Inner::restore_gate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreGate {
    Serve,
    /// Its role slots are short of `need`, `(run, pieces)` runs of whole
    /// chunks: it goes again where they stand free, and (`fund`) room is
    /// made for it ([`Step::EvictForRoles`](super::plan)).
    Evict { need: (u32, u32), fund: bool },
}
