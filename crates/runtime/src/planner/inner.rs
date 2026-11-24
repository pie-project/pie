use std::collections::{BTreeMap, HashSet};
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
    /// has not fired since: one at a time. Restored together, each takes
    /// its items back and every bind comes up short against the others',
    /// in turn without end; restored alone, a bind competes only with the
    /// residents.
    pub(super) bind_restoring: Option<ProcessId>,
    pub(super) next_seq: u64,
    next_id: u64,
    pub(super) procs: super::PidMap<Proc>,
    pub(super) queue: BTreeMap<EntryKey, Waiter>,
    pub(super) accum: DevicePageReservation,
    /// Pages on their way back: a suspend's in flight, keyed by its process,
    /// or a departed process's still held by its working sets, keyed by the
    /// process gone, until its last [`super::Departure`] hold drops.
    pub(super) evicting: super::PidMap<Reclaim>,
    pub(super) completion_credit: u32,
    pub(super) host_swap_blocked: super::PidSet,
    pub(super) prepare_blocked: super::PidSet,
    /// The pool's room count and the swap tiers' free slots when the last
    /// of those blocks was set ([`super::ResidencyPlanner::lapse_blocks`]):
    /// a block holds only on the pool it was set on.
    pub(super) blocked_at: Option<(u64, u32)>,
    pub(super) killing: super::PidSet,
    /// Processes contention failed ([`super::ResidencyPlanner::contend`]),
    /// with why: each of their waits ends with it.
    pub(super) failed: std::collections::HashMap<ProcessId, super::PlannerError>,
    pub(super) admitting: BTreeMap<(Rank, ProcessId), Arc<Notify>>,
    pub(super) to_end: super::to_end::ToEnd,
    /// The evicted processes whose parked state is on its way down a tier
    /// ([`super::ResidencyPlanner::park_down`]): none is restored until it
    /// lands, so no restore misses an item moving under it.
    pub(super) parked_moving: super::PidSet,
    /// The unservable head the last stall dump named, the pool's room
    /// count then, and when ([`super::ResidencyPlanner::dump_stall`]): one
    /// dump per head and room change, and again each
    /// [`super::plan::STALL_SAYS_AGAIN`] while it stays unserved.
    pub(super) stall_dumped: Option<(super::wait::EntryKey, u64, crate::rt::Instant)>,
    /// The head a stall last forced a pass for, with the chunks free and
    /// the fires landed then: forced once on that room
    /// ([`super::ResidencyPlanner::dump_stall`]).
    pub(super) stall_forced: Option<(super::wait::EntryKey, u32, u64)>,
    /// The head and room the policy was last asked for with no pick
    /// ([`super::eviction`]): asked once a head and room.
    pub(super) policy_asked: Option<(super::wait::EntryKey, u64)>,
    /// The head whose stall was last said, and when: said once a stall
    /// episode, whatever room moves meanwhile.
    pub(super) stall_said: Option<(super::wait::EntryKey, crate::rt::Instant)>,
    /// The room a pass last planned a quiet head on (nothing in flight):
    /// the head and the pool's counts ([`super::ResidencyPlanner::tick`]).
    pub(super) quiet_planned: Option<(super::wait::EntryKey, [u64; 4])>,
    /// Waits served this pass past [`super::wait::LONG_WAIT`]: the kind and
    /// how long.
    pub(super) long_waits: Vec<super::wait::LongWait>,
    /// The head held unserved on something in flight: since when, on what
    /// room, and whether that was said ([`super::ResidencyPlanner::tick`]).
    pub(super) held_in_flight:
        Option<(super::wait::EntryKey, u64, crate::rt::Instant, bool)>,
    /// How the last eviction plan ended, for the stall dump: the head it
    /// planned for, where it stopped, whether the room below was short,
    /// and the candidates its quote found unhostable.
    pub(super) last_plan: Option<super::victim::PlanEnd>,
    /// Each queued process's own last plan end: a stall line names its own
    /// head's, never another entry's ([`Self::plan_fields`]).
    pub(super) ends: super::PidMap<super::victim::PlanEnd>,
    /// Heads this pass planned and left stuck, and their processes: no
    /// move named for them and nothing in flight
    /// ([`super::ResidencyPlanner::stuck_head`]). Passed over for the
    /// rest of the pass, so a waiter behind that can go does, and holding
    /// no victim's restore behind them meanwhile (no move is there for
    /// them to keep room for); cleared as the pass ends, each heading the
    /// next pass again.
    pub(super) passed: Vec<(super::wait::EntryKey, ProcessId)>,
    /// A list compaction on its way ([`super::ResidencyPlanner::compact_lists`]):
    /// the processes whose sets it fenced (resident, but not so for their
    /// fires until it lands), or `Some` of none for one that fenced no one.
    /// One at a time, and in flight as a move is.
    pub(super) relocating: Option<super::PidSet>,
    /// Every eviction landed (`Some` of the head it was made for, if any)
    /// and restore landed (`None`), in order, for the simulation's progress
    /// gate.
    #[cfg(test)]
    pub(super) moves: Vec<(ProcessId, Option<Option<ProcessId>>)>,
}

impl Inner {
    /// The fields every line of a head left unserved carries: where the
    /// last plan for `head` stopped, whether the room below was short and
    /// whom its quote found unhostable (`ended=none` where no plan has
    /// stopped for it yet); and every victim blocked now and every restore
    /// waiting on its rebinds.
    pub(super) fn plan_fields(&self, head: super::wait::EntryKey) -> (String, String) {
        use super::plan::some_pids;
        // Its own process's last end, whatever entry it was queued under
        // then (a refused restore is queued again under a new one): read
        // as the planner's last end of any entry, a stall line would name
        // another head's.
        let pid = self.queue.get(&head).map(|waiter| waiter.pid);
        // An entry served and gone from the queue: the last end, where it
        // was this entry's.
        let own = pid.and_then(|pid| self.ends.get(&pid)).or_else(|| {
            self.last_plan.as_ref().filter(|end| end.head.1 == head.1)
        });
        let plan = match own {
            Some(end) => format!(
                "ended={:?}{} host_short={} unhostable={}",
                end.ended,
                if end.head.1 == head.1 {
                    String::new()
                } else {
                    format!(" (as entry {})", end.head.1)
                },
                end.host_short,
                some_pids(end.unhostable.iter())
            ),
            None => "ended=none host_short=false unhostable=0[]".to_owned(),
        };
        let rebinding: Vec<String> = self
            .procs
            .iter()
            .filter_map(|(pid, proc)| {
                proc.rebind_waiting.as_ref().map(|unbound| format!("{pid}: {unbound}"))
            })
            .collect();
        // Rung raises waiting for room (capped frames' and rises'): each
        // with its process, its rung and whether it was served.
        let raises: Vec<String> = self
            .queue
            .values()
            .filter_map(|waiter| match waiter.kind {
                WaitKind::BindRoom {
                    rise: Some(rung),
                    served,
                    withdrawn,
                    ..
                } => Some(format!(
                    "{}: rung={rung} served={served} withdrawn={withdrawn}",
                    waiter.pid
                )),
                _ => None,
            })
            .collect();
        let blocked = format!(
            "host_swap_blocked={} prepare_blocked={} rebind_waiting={rebinding:?} raises={raises:?}",
            some_pids(self.host_swap_blocked.iter()),
            some_pids(self.prepare_blocked.iter())
        );
        (plan, blocked)
    }

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
        // kinds, holds back the growth of lanes that are all blocked, and an
        // allocation with free runs standing goes ungranted.
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
        // first fire, the hold would outlive a grant it waits on behind the
        // held restore, nothing moving.
        if matches!(kind, WaitKind::Allocation { .. }) && self.bind_restoring == Some(pid) {
            self.bind_restoring = None;
        }
        self.queue.insert(
            key,
            Waiter {
                pid,
                kind,
                since: crate::rt::Instant::now(),
                lacked: "behind the head",
                wanted: ([u64::MAX; 3], [0; 2]),
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

    /// Reads the keys and holds of the processes `only` names (by PLEX id),
    /// or all, and reorders what waits by them. Whether any changed.
    pub(super) fn reschedule(
        &mut self,
        only: Option<&std::collections::HashSet<ProcessId>>,
        order_of: impl Fn(&ProcessId) -> (i64, bool),
    ) -> bool {
        let mut changed = false;
        for (pid, proc) in self.procs.iter_mut().filter(|(pid, proc)| {
            !proc.holder
                && only.is_none_or(|only| only.contains(&crate::policy::schedule::plex_id(**pid)))
        }) {
            let now = order_of(pid);
            changed |= (proc.key, proc.held) != now;
            (proc.key, proc.held) = now;
        }
        if !changed {
            return false;
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
        true
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
                self.queue.values().map(|waiter| waiter.pid).collect()
            })
        };
        // Its head served: no unmet waiter of its, and resident. A restore
        // head leaves the queue while its restore runs and comes back if
        // its unpark is short: read by its queue alone, it looks served
        // in that gap, and the victim's restore would go first.
        // Or served and not fired since (restored or bound for its wait,
        // [`super::proc::Proc::unfired`]): the victim restored would take
        // the room back before the head ran on it, the head's next ask evict
        // it again, and the two go round.
        // Served and not yet taken (a bind's room served, an allocation's
        // grant not yet taken) is unserved still: the room it was served
        // is its until it takes it. Read as served, the victim's restore
        // would go first and take that room, the bind come back short, ask
        // again and evict the victim again, round and round.
        // And served, still in its request: the room it was made goes on
        // growing as it runs, and a victim restored beside it would be
        // evicted again for its next page, and again. The victim
        // comes back once that one ends, idles or is moved itself: evicted
        // once for it each time it arrives.
        let unserved = |head: ProcessId| {
            waiting().contains(&head)
                || self
                    .procs
                    .get(&head)
                    .is_some_and(|proc| {
                        proc.state != Residency::Resident
                            || (proc.unfired && !proc.idle)
                            || proc.bind_waiting
                            || Self::in_request(proc)
                    })
        };
        let passed = |head: ProcessId| self.passed.iter().any(|&(_, pid)| pid == head);
        // Its head a binder still in its request: between two binds it
        // stands out of the queue a moment, and a victim restored then is
        // moved for the next bind (the Metal mix's seed 25: a beam's verify
        // each step, its victim evicted twice). Held until the head's
        // request ends: an event, no deadline.
        let binder_in_request = |head: ProcessId| {
            self.procs
                .get(&head)
                .is_some_and(|proc| proc.binds_served > 0 && Self::in_request(proc))
        };
        let held_for_head = |pid: ProcessId| {
            self.procs
                .get(&pid)
                .and_then(|proc| proc.evicted_for)
                .is_some_and(|head| {
                    head != pid && (unserved(head) || binder_in_request(head)) && !passed(head)
                })
        };
        let head_at_bind = |pid: ProcessId| {
            self.procs
                .get(&pid)
                .and_then(|proc| proc.evicted_for)
                .filter(|&head| !passed(head))
                .and_then(|head| self.procs.get(&head))
                .is_some_and(|head| head.state == Residency::Resident && head.bind_waiting)
        };
        // Evicted while it waited in the queue: held while any process
        // ahead of it has an entry in the queue. Ahead is the policy's
        // priority, the engine's own tie (older first) among equals.
        let behind_queue = |pid: ProcessId| {
            let Some(rank) = self
                .procs
                .get(&pid)
                .filter(|proc| proc.evicted_queued)
                .map(Proc::rank)
            else {
                return false;
            };
            self.queue.values().any(|other| {
                other.pid != pid
                    && !matches!(other.kind, WaitKind::Restore { .. })
                    && self.procs.get(&other.pid).is_some_and(|proc| proc.rank() < rank)
            })
        };
        // The oldest restore passed over for the head it was evicted for:
        // where nothing else heads the queue, it does. With every waiter
        // an evicted restore, each held for another's (none resident,
        // nothing in flight), no head would be chosen and no restore ever
        // planned.
        let mut held_restore: Option<EntryKey> = None;
        for (&key, waiter) in &self.queue {
            if !waiter.is_unmet() || self.passed.iter().any(|&(passed, _)| passed == key) {
                continue;
            }
            if matches!(waiter.kind, WaitKind::Restore { .. })
                && (held_for_head(waiter.pid) || behind_queue(waiter.pid))
            {
                // Held for a process at a bind, between its asks (out of
                // the queue a moment as it takes again what it was
                // served, or asks again): never the one that goes for want
                // of a head. Gone then, the victim's restore would go, take
                // the room the bind was made, and the bind evict it again.
                if !head_at_bind(waiter.pid) {
                    held_restore.get_or_insert(key);
                }
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
                // (else processes evicted in the ramp wait behind every
                // later arrival). A seated
                // process's own growth keeps its turn, or the decoders whose
                // completions free the restore's room would wait on it.
                let admits_new = self
                    .queue
                    .get(&allocation)
                    .and_then(|waiter| self.procs.get(&waiter.pid))
                    .is_some_and(|proc| !proc.seated);
                if restore < allocation
                    && (admits_new
                        || self.fleet_stalled()
                        || self.eviction_unfundable())
                {
                    restore
                } else {
                    allocation
                }
            }
            (Some(allocation), None) => allocation,
            (None, Some(restore)) => restore,
            // Nothing else waits that could be served: an older entry still
            // to take its grant holds the queue (`behind_queue`'s case), as
            // does anything in flight; else, nothing moving, the restore
            // held for its head goes.
            (None, None) => {
                let held = held_restore?;
                let served_untaken = self.queue.values().any(|waiter| match &waiter.kind {
                    WaitKind::Allocation { outcome, .. } => outcome.is_some(),
                    WaitKind::BindRoom { served, .. } => *served,
                    WaitKind::Restore { .. } => false,
                });
                // A head restoring (or anything else in flight) may serve
                // it yet: held, as the anti-thrash rule asks.
                if served_untaken || !self.nothing_in_flight() {
                    return None;
                }
                // The evictor's restore before its victims': where the head a
                // held restore waits for is itself a restore waiting, that
                // one goes (and so on up the chain), never the victim ahead
                // of the head it was moved for. The oldest held restore
                // going instead would take the room its evictor's restore
                // needs, and be evicted again for it.
                let mut key = held;
                let mut seen = HashSet::new();
                while seen.insert(key) {
                    let Some(head) = self
                        .queue
                        .get(&key)
                        .and_then(|waiter| self.procs.get(&waiter.pid))
                        .and_then(|proc| proc.evicted_for)
                    else {
                        break;
                    };
                    // One passed over is no head: chosen again, the pass
                    // over every stuck head would choose it for good, a
                    // core burned.
                    let Some((&up, _)) = self.queue.iter().find(|&(&at, waiter)| {
                        waiter.pid == head
                            && waiter.is_unmet()
                            && matches!(waiter.kind, WaitKind::Restore { .. })
                            && !self.passed.iter().any(|&(passed, _)| passed == at)
                    }) else {
                        break;
                    };
                    key = up;
                }
                // Evicted for a binder still in its request: held until its
                // request ends (an event that plans again). Sent as the last
                // resort while the head stood out of the queue between two
                // binds, it was moved for the next one at once (the Metal
                // mix's seed 25: a beam's verify each step). A head waiting
                // on room itself is never held behind: the restore goes.
                if let Some(proc) = self.queue.get(&key).and_then(|waiter| self.procs.get(&waiter.pid))
                    && let Some(head) = proc.evicted_for
                    && let Some(at) = self.procs.get(&head)
                    && at.binds_served > 0
                    && Self::in_request(at)
                    && !self.queue.values().any(|waiter| waiter.pid == head && waiter.is_unmet())
                {
                    return None;
                }
                key
            }
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
        let mut withdrawn_bind = None;
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
                // meanwhile. Kept, the eviction would wait on fires whose
                // finalizer is that bind, and the bind on the eviction.
                // A rung raise for its capped frame is not: the frame is
                // admitted work its eviction waits on, and it goes only
                // once raised (withdrawn, the eviction waits on a capped
                // frame for good).
                WaitKind::BindRoom {
                    run,
                    pieces,
                    singles,
                    pages,
                    notify,
                    served: false,
                    withdrawn,
                    rise: None,
                } => {
                    *withdrawn = true;
                    wake.push(notify.clone());
                    withdrawn_bind.get_or_insert(super::proc::PendingBind {
                        singles: *singles,
                        run: *run,
                        pieces: *pieces,
                        pages: *pages,
                    });
                }
                _ => {}
            }
        }
        // The room its bind waited for is what its restore brings back
        // with it, as a bind suspended for room asks
        // ([`Proc::bind_room`]): counted in its need and laid beside its
        // restore. Forgotten, an evicted process's need read its bind's
        // room short: made the capacity gate's finisher, restored, it would
        // ask the room again past what stood free, lose its place and be
        // evicted for the new finisher, round and round.
        if let Some(bind) = withdrawn_bind
            && let Some(proc) = self.procs.get_mut(&pid)
        {
            proc.bind_room.get_or_insert(bind);
        }
        wake
    }

    /// `pid` arrived (its restore landed, a new request, or it left): the
    /// processes restored past it while it stood passed over may be moved
    /// for it again ([`super::proc::Proc::restored_past`]).
    pub(super) fn arrived(&mut self, pid: ProcessId) {
        for proc in self.procs.values_mut() {
            if proc.restored_past == Some(pid) {
                proc.restored_past = None;
            }
            if proc.last_resort_for == Some(pid) {
                proc.last_resort_for = None;
            }
        }
    }

    /// Whether `head` was evicted for `pid` and is not back yet: no victim
    /// for `head`'s asks, `pid` is.
    pub(super) fn evicted_for_by(&self, head: ProcessId, pid: ProcessId) -> bool {
        self.procs
            .get(&head)
            .is_some_and(|proc| proc.state != Residency::Resident && proc.evicted_for == Some(pid))
    }

    /// Whether `pid`'s bind was served its room and has not taken it yet:
    /// room made for it, it takes it as it wakes. Moved meanwhile, the room
    /// would go to others, the bind find its process parked, and its
    /// restore come back without the room, short again.
    pub(super) fn bind_served(&self, pid: ProcessId) -> bool {
        self.queue.values().any(|waiter| {
            waiter.pid == pid && matches!(waiter.kind, WaitKind::BindRoom { served: true, .. })
        })
    }

    /// Whether `proc` is resident in its request: admitted, not a handle's
    /// stand-in, not idle on its client nor held below the policy's floor.
    pub(super) fn in_request(proc: &Proc) -> bool {
        proc.admitted
            && !proc.holder
            && !proc.idle
            && !proc.held
            && proc.state == Residency::Resident
    }

    /// Whether `pid` keeps its room against a head the policy ranks at
    /// `head`: admitted and resident in its request, a process runs to its
    /// end, moved only for room something ranked above it asks (the rank
    /// is the policy's victim order, [`Proc::rank`]). Moved for its peers'
    /// asks whenever it stands between fires, a lane goes round with them,
    /// evicted and restored while a few lanes step, and none finish.
    ///
    /// It keeps none where it cannot go on until room comes: idle or held
    /// (the policy's), at a bind for room, or waiting in the queue with
    /// nothing in flight (`quiet`, [`Self::nothing_in_flight`] as the
    /// caller read it). Where no one can go on, contention
    /// ([`super::ResidencyPlanner::contend`]) suspends a victim whole.
    pub(super) fn keeps_room(&self, pid: ProcessId, head: Rank, quiet: bool) -> bool {
        self.procs
            .get(&pid)
            .is_some_and(|proc| proc.rank() <= head && Self::in_request(proc) && !proc.bind_waiting)
            && !(quiet
                && self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == pid && waiter.is_unmet()))
    }

    /// Nothing in flight (no fire, no landing, no move) and no process
    /// that could fire if served: every resident waits in the queue, is
    /// idle or held, or is not admitted, and no ask stands served and not
    /// taken. Only so is a full set of tiers a reason to fail a process
    /// ([`super::ResidencyPlanner::fail_tiers_full`]); short of it, an ask
    /// waits for the departure, free or landing that comes.
    pub(super) fn truly_wedged(&self) -> bool {
        self.nothing_in_flight()
            && self.parked_moving.is_empty()
            && !self.queue.values().any(Waiter::untaken)
            && !self.queue.values().any(Waiter::rise_untaken)
            && !self.runnable()
    }

    /// Whether a process could fire as it stands: resident, admitted, not
    /// idle or held, with no wait of its own in the queue. Its fire and
    /// what it frees as it ends are the event a waiting head waits on.
    pub(super) fn runnable(&self) -> bool {
        self.procs.iter().any(|(pid, proc)| {
            proc.state == Residency::Resident
                && proc.admitted
                && !proc.idle
                && !proc.held
                && !self.queue.values().any(|waiter| waiter.pid == *pid && waiter.is_unmet())
        })
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
            let (lacked, wanted) = (waiter.lacked, waiter.wanted);
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
                long_waits.push(super::wait::LongWait {
                    kind: "allocation",
                    lacked,
                    wanted,
                    waited,
                });
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
        // completion while its runs stand free, it would wait on an event
        // that need not come.
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
        // each running lane past the pages it had already gathered, it would
        // hold them idle while the fleet decodes, its need standing and
        // never attempted; seated lanes' growth goes
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

    /// Sets the room the victims blocked now are blocked on: those blocked
    /// on another room go back among the candidates first. One stamp for
    /// every block, refreshed by a later block on more room, would hold an
    /// earlier victim out on room it fits.
    pub(super) fn block_on(&mut self, room: (u64, u32)) {
        if self.blocked_at != Some(room) {
            self.host_swap_blocked.clear();
            self.prepare_blocked.clear();
        }
        self.blocked_at = Some(room);
    }

    /// Whether `pid`'s anti-thrash mark ([`super::proc::Proc::unfired`])
    /// keeps it out of the victims: only while it can fire, by state, never
    /// a clock. Waiting in the queue itself (its lanes blocked on the
    /// planner, which serves the head first) or at a bind for room, it
    /// cannot, and protected it would hold the head that waits on its
    /// room for good: it is a candidate. Nothing in flight is no such
    /// state: restored a moment ago it fires as soon as its lanes wake, and
    /// moved back first, it would go round with the head it made room for,
    /// restored ahead of it and evicted for it again without a fire between.
    /// A head no move serves while such a one holds its room
    /// is passed over ([`super::ResidencyPlanner::stuck_head`]).
    pub(super) fn unfired_protects(&self, pid: ProcessId) -> bool {
        self
                .procs
                .get(&pid)
                .is_some_and(|proc| proc.unfired && !proc.bind_waiting)
            && (!self
                .queue
                .values()
                .any(|waiter| waiter.pid == pid && waiter.is_unmet())
                || !self.nothing_in_flight())
    }

    /// Whether the one bind-suspended restore at a time
    /// ([`Self::bind_restoring`]) no longer holds the next: its process
    /// cannot fire (nothing in flight anywhere, or it waits in the queue on
    /// something other than its bind) and its bind is not queued for room
    /// (which the gate funds first). By state, never a clock: held while it
    /// cannot fire, every other bind-suspended restore would wait for good
    /// with its runs standing.
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
    /// The kv pages restore `key` still gathers: its kv need less what the
    /// head has gathered (`accum`, the head's alone).
    pub(super) fn restore_kv_missing(&self, key: EntryKey) -> u32 {
        let Some(waiter) = self.queue.get(&key) else {
            return 0;
        };
        let gathered = if self.unmet_head().is_some_and(|(head, _)| head == key) {
            u32::try_from(self.accum.len()).unwrap_or(u32::MAX)
        } else {
            0
        };
        waiter.kv_need().saturating_sub(gathered)
    }

    /// A print of the planner's own book a pass reads: the queue (each
    /// entry and whether it is unmet), the pages gathered, the moves in
    /// flight and parked state moving, and each process's residency,
    /// fires, idleness and recorded needs. Part of the planner's generation
    /// ([`super::ResidencyPlanner::generation`]).
    pub(super) fn book_print(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        // Unordered sets fold by a wrapping sum of each member's hash: no
        // sort, no allocation, the same print whatever the map's order.
        fn one(value: impl Hash) -> u64 {
            let mut hasher = rustc_hash::FxHasher::default();
            value.hash(&mut hasher);
            hasher.finish()
        }
        let mut print = one((self.next_id, self.accum.len(), self.bind_restoring));
        for (key, waiter) in &self.queue {
            print = print
                .rotate_left(5)
                ^ one((key, waiter.pid, waiter.is_unmet(), waiter.kind_name()));
        }
        let evicting = self.evicting.keys().fold(0u64, |sum, pid| sum.wrapping_add(one(pid)));
        let moving = self.parked_moving.iter().fold(0u64, |sum, pid| sum.wrapping_add(one(pid)));
        let procs = self.procs.iter().fold(0u64, |sum, (pid, proc)| {
            sum.wrapping_add(one((
                pid,
                proc.state as u8,
                proc.fires,
                proc.fire_lapsed,
                proc.idle,
                proc.roles_short.map(|need| (need.run, need.pieces)),
                proc.bind_room.map(|need| (need.run, need.pieces)),
            )))
        });
        one((print, evicting, moving, procs))
    }

    /// [`Self::nothing_in_flight`] but for `pid`'s own move.
    pub(super) fn nothing_in_flight_but(&self, pid: ProcessId) -> bool {
        self.relocating.is_none()
            && self.evicting.keys().all(|evicting| *evicting == pid)
            && !self.kill_in_flight()
            && self.procs.iter().all(|(at, proc)| {
                *at == pid
                    || ((!proc.firing() || proc.bind_waiting)
                        && match proc.state {
                            Residency::Evicting => false,
                            Residency::Restoring => proc.rebind_waiting.is_some(),
                            _ => true,
                        })
            })
    }

    /// Whether `pid` is resident and none of its pages move under a list
    /// compaction: its fires may run.
    pub(super) fn settled_resident(&self, pid: ProcessId) -> bool {
        self.procs
            .get(&pid)
            .is_none_or(|proc| proc.state == Residency::Resident)
            && !self
                .relocating
                .as_ref()
                .is_some_and(|fenced| fenced.contains(&pid))
    }

    pub(super) fn nothing_in_flight(&self) -> bool {
        self.relocating.is_none()
            && self.evicting.is_empty()
            && !self.kill_in_flight()
            && self.procs.values().all(|proc| {
                // A fire held at a bind the store has not served is no fire
                // submitted: nothing it waits on comes from it.
                (!proc.firing() || proc.bind_waiting)
                    && match proc.state {
                        Residency::Evicting => false,
                        // Waiting on its rebinds, it moves nothing: it is
                        // waiting, not in flight ([`Proc::rebind_waiting`]).
                        Residency::Restoring => proc.rebind_waiting.is_some(),
                        _ => true,
                    }
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
