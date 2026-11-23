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
    /// has not fired since: one at a time. Restored together, each took its
    /// items back and every bind came up short against the others' (the M1
    /// stress: ten processes suspended, restored and short in turn, 190
    /// times each); restored alone, a bind competes only with the
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
    /// The capacity gate as the pass reads it ([`CapacityGate`]): an
    /// allocation of any process but the finisher's waits while it would
    /// leave less free on every tier together than the finisher's headroom.
    pub(super) capacity: Option<CapacityGate>,
    /// Each process's place in the gate's order ([`CapacityGate`]): what it
    /// may take before the ones ahead of it could no longer finish in turn.
    /// One not named takes what the last may.
    pub(super) capacity_allow: std::collections::HashMap<ProcessId, u64>,
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

/// The capacity gate (the mechanism; who parks for whom stays the
/// policy's): state let in must always leave one process able to finish.
/// The finisher is the oldest admitted process not idle, picked by its
/// admission order alone; its headroom is its end (the declared context,
/// in pages) less what it holds on every tier. Every other process's
/// allocation, a new one's first included, waits while granting it would
/// leave the device, the host and the disk together with less free than
/// that headroom ("waits for the finisher's headroom on all tiers"); the
/// finisher's own never waits on it, and its eviction plan parks the others
/// below, where the gate kept room for them. As the finisher ends, the next
/// oldest inherits the headroom (its departure plans again). Let in past
/// that, every resident a page short with nothing anywhere to move into
/// deadlocked the fleet (the simulation's seeds 2547, 3243, 3591; the
/// L40S's 3 GiB disk-only stress, 0 of 150 decodes done in 553 s).
///
/// An eviction moves a process whole, so the headroom also covers the
/// largest other resident's unit less a page ([`CapacityRead::margin`]):
/// the room the finisher's deficit is made in on the device overshoots it
/// by up to one move, and that move must find room below. Counted as
/// fungible pages alone, the free room lay below in pieces smaller than
/// every resident, and nobody could move (the field mix's stalls).
///
/// The finisher is the oldest admitted process whose need fits, resident
/// or not: preferred resident, an evicted old process never became the
/// finisher while any resident fitted, and its restore waited behind every
/// younger one (the field mix: a decode evicted at step 131 restored at
/// 607, the L40S's decodes unfinished while beams verified).
#[derive(Clone, Copy, Debug)]
pub(super) struct CapacityGate {
    pub(super) finisher: ProcessId,
    /// Pages free on every tier together past the finisher's headroom.
    pub(super) spare: u64,
    /// Pages free on every tier together.
    pub(super) free: u64,
    /// Taken past the order since it was read, by every ask but the
    /// finisher's: each such ask leaves every place after the finisher that
    /// much less ([`Self::allowance`]).
    pub(super) used: u64,
    /// The largest move another process makes ([`CapacityRead::unit`]),
    /// the process that makes it, and the second largest: an ask of the
    /// largest raises the margin by its pages, any other's by no more
    /// than it takes the second past the largest.
    pub(super) unit: (u64, Option<ProcessId>, u64),
}

/// What the capacity gate is read from, in kv pages
/// ([`super::ResidencyPlanner::capacity_read`]): the finisher and what it
/// still needs, and every tier's room and what is held in it
/// ([`super::pool::CapacityCensus`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CapacityRead {
    pub(super) finisher: ProcessId,
    /// Its end: the declared context.
    pub(super) end: u64,
    /// The kv pages it holds on every tier.
    pub(super) kv: u64,
    /// What of its state lies off the device, its restore's to bring back:
    /// its kv pages below, and its guest state and tables where it is not
    /// resident.
    pub(super) below: u64,
    /// The room its binds wait for: new state of its own.
    pub(super) binding: u64,
    /// Its guest state and tables, counted as `held` counts them.
    pub(super) guest: u64,
    /// Every tier's room for state together.
    pub(super) whole: u64,
    /// What every tier holds, each item once wherever it lies.
    pub(super) held: u64,
    /// The room the guest shapes hold open that no state fills: not free.
    pub(super) slack: u64,
    /// Of `held`, what only idle cache holds: free to the gate.
    pub(super) idle: u64,
    /// Of `held`, the second slots moves under way hold.
    pub(super) moving: u64,
    /// Of `held`, the device pages the planner gathered for its heads:
    /// free to the gate, as the head they go to asks them again (counted
    /// taken, gathering for a head held it at the gate, given back it went
    /// again, a pass that never ended).
    pub(super) gathered: u64,
    /// The largest state one other process holds on the device, the
    /// process, and the second largest: what one eviction moves below
    /// whole ([`CapacityRead::margin`]).
    pub(super) unit: (u64, Option<ProcessId>, u64),
}

impl CapacityRead {
    /// The finisher's headroom: its kv growth to its end, its restore's
    /// state, and its binds' room.
    /// What it needs to finish ([`Self::headroom`]): its place in the
    /// gate's order.
    pub(super) fn need(&self) -> u64 {
        self.headroom()
    }

    fn headroom(&self) -> u64 {
        self.end.saturating_sub(self.kv) + self.below + self.binding + self.margin()
    }

    /// What the finisher's room on the device costs below past its own
    /// pages: an eviction moves a process's state whole, so the pages that
    /// make the finisher's room overshoot its deficit by up to one move
    /// less a page. Free pages counted on every tier together, a room below
    /// smaller than every resident's state moved nobody: the finisher a
    /// page short on a full device with free pages left only below, every
    /// other resident too large for them (the simulation's field mix, seed
    /// 641: four pages free on the disk, residents of five, six and nine).
    fn margin(&self) -> u64 {
        self.unit.0.saturating_sub(1)
    }

    /// Every tier's room past what is held and the slack held open.
    pub(super) fn free(&self) -> u64 {
        self.free_of_state().saturating_sub(self.slack)
    }

    /// Every tier's room past what is held: the room the state leaves,
    /// slack not counted.
    fn free_of_state(&self) -> u64 {
        // A move under way holds two slots for one item; exactly one comes
        // back whichever way it ends (a landed move gives back its source,
        // a refused or aborted one its destination). The order is read after
        // the moves under way: counted taken, every grant's own eviction or
        // restore left the next read short by what it put in flight, and the
        // order turned short before anyone grew (F57: the field mix's seed
        // 401, least -6 at step 7 with 9 moving). Asks still wait for room on
        // the tiers themselves until the moves land.
        let taken = self
            .held
            .saturating_sub(self.idle)
            .saturating_sub(self.gathered)
            .saturating_sub(self.moving);
        self.whole.saturating_sub(taken)
    }

    /// The gate: an ask waits while what is held, the ask and the
    /// finisher's headroom together pass every tier's room. A move between
    /// tiers changes no term, so only new state trips it.
    pub(super) fn gate(&self) -> CapacityGate {
        let (headroom, free) = (self.headroom(), self.free());
        CapacityGate {
            finisher: self.finisher,
            spare: free.saturating_sub(headroom),
            free,
            unit: self.unit,
            used: 0,
        }
    }


    /// What the room the state leaves falls short of the finisher's growth
    /// to its end by, the finisher's own guest state counted free (its
    /// binds, its own asks, take what they take; its moves change nothing):
    /// what no ask but the finisher's may raise. Read past the slack, which
    /// a move may change (a park opening a chunk below): the gate holds
    /// asks on the room past it, so no ask raises this.
    #[cfg(test)]
    pub(super) fn others_short(&self) -> u64 {
        self.end
            .saturating_sub(self.kv)
            .saturating_sub(self.free_of_state() + self.guest)
    }
}

impl CapacityGate {
    /// Whether an allocation of `pages` for `pid` waits on the gate.
    pub(super) fn holds(&self, pid: ProcessId, pages: u32, place: Option<u64>) -> bool {
        pid != self.finisher && pages > 0 && self.cost(pid, pages) > self.allowance(place)
    }

    /// What an ask may take at `place` in the order (its allowance as read,
    /// none: the last's), less what asks took since.
    pub(super) fn allowance(&self, place: Option<u64>) -> u64 {
        place.unwrap_or(self.spare).saturating_sub(self.used)
    }

    /// What `pages` asked for `pid` take of the spare: the pages, and what
    /// they raise the largest move by (its own growth for the largest; past
    /// the largest, no more than the second's growth would take it).
    pub(super) fn cost(&self, pid: ProcessId, pages: u32) -> u64 {
        let pages = u64::from(pages);
        let (largest, of, second) = self.unit;
        let raised = if Some(pid) == of {
            pages
        } else {
            (second + pages).saturating_sub(largest)
        };
        pages + raised
    }

    /// Whether `waiter` waits on the gate: an allocation's pages, or a
    /// bind's room (its guest state's fresh chunks), new state either way;
    /// a restore brings back what is held already, and a rung raise moves
    /// the requester's own.
    pub(super) fn holds_waiter(&self, waiter: &Waiter, place: Option<u64>) -> bool {
        match waiter.kind {
            WaitKind::Allocation { rise: false, .. } => {
                self.holds(waiter.pid, waiter.kv_need(), place)
            }
            WaitKind::BindRoom { rise: None, .. } => {
                self.holds(waiter.pid, waiter.room_pages(), place)
            }
            _ => false,
        }
    }
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
        // as the planner's last end of any entry, a stall line named
        // another head's (Mac Pro at 2 GiB: "ended=none (last ... for
        // entry ...)").
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
                self.queue.values().map(|waiter| waiter.pid).collect()
            })
        };
        // Its head served: no unmet waiter of its, and resident. A restore
        // head leaves the queue while its restore runs and comes back if
        // its unpark is short: read by its queue alone, it looked served
        // in that gap, and the victim's restore went first (the L40S: one
        // victim evicted 5,894 times for one restore head, never served).
        // Or served and not fired since (restored or bound for its wait,
        // [`super::proc::Proc::unfired`]): the victim restored then took the
        // room back before the head ran on it, the head's next ask evicted
        // it again, and the two went round (the L40S at 3 GiB on 4cef0c7d3:
        // 613 evictions within 200 ms of their process's restore, about one
        // eviction a serve).
        // Served and not yet taken (a bind's room served, an allocation's
        // grant not yet taken) is unserved still: the room it was served
        // is its until it takes it. Read as served, the victim's restore
        // went first and took that room, the bind came back short, asked
        // again and evicted the victim again, and round (seed 361: one
        // process evicted and restored for one bind for good).
        // And served, still in its request: the room it was made goes on
        // growing as it runs, and a victim restored beside it was evicted
        // again for its next page, and again (the field mix over 50 seeds:
        // 1,811 evictions for 1,405 finished, 55 of them a lane's again for
        // the same head with no restore of that head between). The victim
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
        let held_for_head = |pid: ProcessId| {
            self.procs
                .get(&pid)
                .and_then(|proc| proc.evicted_for)
                .is_some_and(|head| head != pid && unserved(head) && !passed(head))
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
        // The oldest restore passed over for the head it was evicted for:
        // where nothing else heads the queue, it does. With every waiter
        // an evicted restore, each held for another's (the L40S at 3 GiB
        // on 5120cbc3c: six beams evicted for one another, none resident,
        // nothing in flight), no head was chosen, no restore was ever
        // planned, and even the beat named none.
        let mut held_restore: Option<EntryKey> = None;
        for (&key, waiter) in &self.queue {
            if !waiter.is_unmet() || self.passed.iter().any(|&(passed, _)| passed == key) {
                continue;
            }
            // The finisher's restore is held for no other head: the room
            // every tier keeps is its ([`CapacityGate`]), and the heads
            // ahead of it may wait on the gate until it finishes.
            if matches!(waiter.kind, WaitKind::Restore { .. })
                && !self.is_finisher(waiter.pid)
                && (held_for_head(waiter.pid) || behind_queue(waiter.pid))
            {
                // Held for a process at a bind, between its asks (out of
                // the queue a moment as it takes again what it was
                // served, or asks again): never the one that goes for want
                // of a head. Gone then, the victim's restore went, took the
                // room the bind was made, and the bind evicted it again
                // (seed 1610).
                if !head_at_bind(waiter.pid) {
                    held_restore.get_or_insert(key);
                }
                continue;
            }
            // An allocation past what every tier keeps for the finisher
            // waits on the gate: no head ([`CapacityGate`]).
            if self.capacity.is_some_and(|gate| {
                gate.holds_waiter(waiter, self.capacity_allow.get(&waiter.pid).copied())
            }) {
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
                let finisher_restore = self
                    .queue
                    .get(&restore)
                    .is_some_and(|waiter| self.is_finisher(waiter.pid));
                if restore < allocation
                    && (admits_new
                        || finisher_restore
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
                held
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
                // meanwhile. Kept, the eviction waited on fires whose
                // finalizer was that bind, and the bind on the eviction
                // (Mac Pro's M5 soak: evicting=1 for good, pool free 0).
                // A rung raise for its capped frame is not: the frame is
                // admitted work its eviction waits on, and it goes only
                // once raised (Mac Pro at 2 GiB on 4cef0c7d3: the eviction
                // waited on a capped frame whose raise it withdrew).
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
        // room short: made the capacity gate's finisher, restored, it asked
        // the room again past what stood free, lost its place and was
        // evicted for the new finisher, round and round (the field mix's
        // wide shape, seed 7).
        if let Some(bind) = withdrawn_bind
            && let Some(proc) = self.procs.get_mut(&pid)
        {
            proc.bind_room.get_or_insert(bind);
        }
        wake
    }

    /// Whether `pid` is the capacity gate's finisher
    /// ([`CapacityGate`]): its restore and its growth go first and are
    /// funded whatever is in flight, so the one process the tiers keep
    /// room for finishes while younger work runs on.
    /// Whether the capacity gate holds an ask of `pid`'s: it fires only
    /// once the ones ahead of it in the gate's order finish. Spared as one
    /// restored and not fired since, it held the room they wait on while
    /// the moves around it kept something in flight for good (the field
    /// mix, seed 178: the finisher's bind of two adjacent chunks unserved
    /// 2,400 steps, one move away).
    pub(super) fn gate_holds(&self, pid: ProcessId) -> bool {
        self.capacity.is_some_and(|gate| {
            let place = self.capacity_allow.get(&pid).copied();
            self.queue
                .values()
                .any(|waiter| waiter.pid == pid && waiter.is_unmet() && gate.holds_waiter(waiter, place))
        })
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
    /// went to others, the bind found its process parked, and its restore
    /// came back without the room, short again (the simulation's field
    /// mix, seed 13: suspended twice for one bind).
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
    /// asks whenever it stood between fires, a lane went round with them,
    /// evicted and restored while about three lanes stepped (the L40S at
    /// 3 GiB: 1,923 evictions, restores 1,922, 0 of 150 decodes finished in
    /// 542 s).
    ///
    /// It keeps none where it cannot go on until room comes: idle or held
    /// (the policy's), at a bind for room, its ask held by the capacity
    /// gate (it goes only once those ahead of it in the gate's order
    /// finish, and they may need its room), or waiting in the queue with
    /// nothing in flight (`quiet`, [`Self::nothing_in_flight`] as the
    /// caller read it). So whatever holds the room the gate's first needs
    /// either goes on to its end or yields, and the gate's order still
    /// finishes everyone.
    pub(super) fn keeps_room(&self, pid: ProcessId, head: Rank, quiet: bool) -> bool {
        self.procs
            .get(&pid)
            .is_some_and(|proc| proc.rank() <= head && Self::in_request(proc) && !proc.bind_waiting)
            && !(quiet
                && self
                    .queue
                    .values()
                    .any(|waiter| waiter.pid == pid && waiter.is_unmet()))
            && !self.gate_holds(pid)
    }

    /// The gate and its order as a pass reads them
    /// ([`super::ResidencyPlanner::capacity_order`]).
    pub(super) fn set_capacity(
        &mut self,
        order: Option<(CapacityGate, std::collections::HashMap<ProcessId, u64>)>,
    ) {
        let (gate, places) = order.unzip();
        self.capacity = gate;
        self.capacity_allow = places.unwrap_or_default();
    }

    pub(super) fn is_finisher(&self, pid: ProcessId) -> bool {
        self.capacity.is_some_and(|gate| gate.finisher == pid)
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
            capacity,
            capacity_allow,
            ..
        } = self;
        let mut used = capacity.map(|gate| gate.used);
        for (&key, waiter) in queue.range_mut(head..) {
            if !waiter.is_unmet() {
                continue;
            }
            // Past the finisher's headroom: it waits on the gate.
            let pid = waiter.pid;
            let pages = waiter.kv_need();
            if let (Some(gate), Some(used)) = (capacity.as_ref(), used.as_mut())
                && matches!(waiter.kind, WaitKind::Allocation { rise: false, .. })
            {
                let gate = CapacityGate { used: *used, ..*gate };
                if gate.holds(pid, pages, capacity_allow.get(&pid).copied()) {
                    continue;
                }
                if pid != gate.finisher {
                    *used = used.saturating_add(gate.cost(pid, pages));
                }
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
                fund: stalled || self.bind_waits_behind(pid) || self.is_finisher(pid),
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

    /// Sets the room the victims blocked now are blocked on: those blocked
    /// on another room go back among the candidates first. One stamp for
    /// every block, refreshed by a later block on more room, held an
    /// earlier victim out on room it fits (seed 3392: a victim of seven
    /// pages blocked at six free below, kept out at nineteen, every waiter
    /// left with nothing in flight).
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
    /// moved back first, it went round with the head it made room for,
    /// restored ahead of it and evicted for it again without a fire between
    /// (seed 4139). A head no move serves while such a one holds its room
    /// is passed over ([`super::ResidencyPlanner::stuck_head`]).
    pub(super) fn unfired_protects(&self, pid: ProcessId) -> bool {
        !self.gate_holds(pid)
            && self
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
