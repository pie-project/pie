use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::departure::Departure;
use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::holdings::held_page_count;
use super::inner::Inner;
use super::pool::PoolPort;
use super::proc::{Proc, Residency};
use super::rank::Rank;
use super::reclaim::Reclaim;
use super::stats::PlannerStats;
use super::wait::{EntryKey, WaitKind, Waiter};
use super::{ProcessId, exec};

pub(super) const HOST_RESERVE_DIVISOR: u32 = 8;

/// It suspends no process of its own.
const SWEEP: ProcessId = ProcessId::nil();

pub struct ResidencyPlanner {
    pub(super) inner: parking_lot::Mutex<Inner>,
    pub(super) pause_due: AtomicBool,
    pub(super) waiters: AtomicUsize,
    pub(super) nonresident: AtomicUsize,
    drain: OnceLock<Arc<Notify>>,
    idle_reclaim_exhausted: std::sync::atomic::AtomicBool,
    pub(super) sweeping: AtomicBool,
    sweep_exhausted: AtomicBool,
    pub(super) demoting: AtomicBool,
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
}

impl ResidencyPlanner {
    pub fn new(port: Arc<dyn PoolPort>) -> Self {
        Self {
            inner: parking_lot::Mutex::new(Inner::default()),
            pause_due: AtomicBool::new(false),
            waiters: AtomicUsize::new(0),
            nonresident: AtomicUsize::new(0),
            drain: OnceLock::new(),
            idle_reclaim_exhausted: std::sync::atomic::AtomicBool::new(false),
            sweeping: AtomicBool::new(false),
            sweep_exhausted: AtomicBool::new(false),
            demoting: AtomicBool::new(false),
            port,
            stats: PlannerStats::default(),
            seated_first: false,
            seating: parking_lot::Mutex::new(()),
            seat_to_end: false,
            to_end_notify: Notify::new(),
        }
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
        if self.drain.set(notify.clone()).is_err() {
            return;
        }
        let planner = self.clone();
        crate::rt::spawn(async move {
            loop {
                notify.notified().await;
                planner.plan();
            }
        });
    }

    pub(super) fn poke(self: &Arc<Self>) {
        match self.drain.get() {
            Some(notify) => notify.notify_one(),
            None => self.plan(),
        }
    }

    pub(super) fn reclaim_idle_once(&self) -> bool {
        if self.idle_reclaim_exhausted.swap(true, Ordering::AcqRel) || self.port.reclaim_idle() == 0
        {
            return false;
        }
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        true
    }

    pub(super) fn re_arm_idle_reclaim(&self) {
        self.idle_reclaim_exhausted.store(false, Ordering::Release);
        self.sweep_exhausted.store(false, Ordering::Release);
    }

    /// Not what the head and restores under way read: they would bring it
    /// straight back.
    pub(super) fn sweep_stranded(self: &Arc<Self>, wanted: u32) -> bool {
        if self.sweep_exhausted.load(Ordering::Acquire)
            || self.sweeping.swap(true, Ordering::AcqRel)
        {
            return false;
        }
        let (swap_free, swap_total) = self.swap_stats();
        let room = swap_free.saturating_sub(swap_total / HOST_RESERVE_DIVISOR) as usize;
        let (model, engine) = self.port.locus();
        // What rests goes in the order it would be evicted, and no more of
        // it than is wanted: a policy's keys hold here too, the head of a set
        // they keep going last.
        let order: Vec<_> = self
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
        let keep: HashSet<_> = self
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
        let txn = (room > 0)
            .then(|| {
                self.port
                    .prepare_sweep(&keep, room, &order, wanted as usize)
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
        self.waiters.store(inner.queue.len(), Ordering::Release);
        self.nonresident
            .store(inner.nonresident_count(), Ordering::Release);
        inner.admit_unpreempted();
        result
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

    /// The recurrent slots holders keep on the device, a slot forks share
    /// counted once: each would need the host if it went there.
    pub fn resting_rs_slots(&self) -> u32 {
        let (model, engine) = self.port.locus();
        let holders: Vec<ProcessId> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(_, proc)| proc.holder && proc.state == Residency::Resident)
                .map(|(pid, _)| *pid)
                .collect()
        });
        let resting: HashSet<_> = holders
            .into_iter()
            .flat_map(|pid| {
                crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine)
            })
            .collect();
        crate::store::registry::try_get(model, engine).map_or(0, |stores| {
            stores.rs.lock().unwrap().suspendable(&resting).slots
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
                Some(notify) => notify.notified().await,
                None => break,
            }
        }
        if self.seat_to_end {
            self.admit_to_end(pid).await;
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

    /// Suspends the processes reading pages past the pool's limit: they come
    /// back below it. A process moves whole, never a page at a time, so a
    /// shrink does not turn into a fault per decode.
    pub fn vacate_tail(self: &Arc<Self>) {
        // What a waiter has gathered may lie in the tail: it gathers again.
        let gathered = self.with_inner(|inner| std::mem::take(&mut inner.accum));
        drop(gathered);
        self.poke();
        let (model, engine) = self.port.locus();
        let tail = self.port.working_sets_past_limit();
        if tail.is_empty() {
            return;
        }
        let pids: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        let readers = pids
            .into_iter()
            .filter(|&pid| {
                crate::inferlet::process::residency::kv_working_set_ids(pid, model, engine)
                    .iter()
                    .any(|ws| tail.contains(ws))
            })
            .collect();
        self.pause(readers, |_| true);
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
            },
        };
        let (picks, unhostable) =
            self.quote_and_pick(pausable, everything, self.swap_room(), model, engine);
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
        let (model, engine) = self.port.locus();
        let held = held_page_count(pid, model, engine);
        // A seat taken again after the process ended goes back.
        crate::inferlet::process::vacate_seat(pid);
        let (signal, removed) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
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
            if held > 0
                && departed
                    .as_ref()
                    .is_some_and(|proc| proc.departure.strong_count() > 0)
            {
                inner.evicting.insert(
                    pid,
                    Reclaim {
                        pages: held,
                        rs: RsUnits::default(),
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
            (signal, removed)
        });
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

    pub fn pages_freed(self: &Arc<Self>) {
        self.re_arm_idle_reclaim();
        if self.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        }
    }

    pub(super) fn try_reserve(&self, demand: Demand) -> Option<AllocationGrant> {
        let kv = if demand.kv_pages > 0 {
            let pages = self.port.reserve_device(demand.kv_pages)?;
            DevicePageReservation::new(pages, self.port.clone())
        } else {
            DevicePageReservation::empty()
        };
        let rs = self.reserve_rs_slots(demand.rs)?;
        Some(AllocationGrant::new(demand, kv, rs))
    }

    pub(super) fn reserve_rs_slots(&self, need: RsUnits) -> Option<RsSlotReservation> {
        if need.is_zero() {
            return Some(RsSlotReservation::empty());
        }
        let grant = self.port.reserve_rs(need)?;
        Some(RsSlotReservation::new(grant, self.port.clone()))
    }

    pub(super) fn absorb_free(&self, count: u32) -> bool {
        let pages = self.port.reserve_device_up_to(count);
        if pages.is_empty() {
            return false;
        }
        let reservation = DevicePageReservation::new(pages, self.port.clone());
        self.with_inner(|inner| inner.accum.absorb(reservation));
        true
    }
}

/// What the whole of a holder keeps: a key on a prefix keeps only that prefix.
fn holder_key(pid: ProcessId) -> i64 {
    use crate::policy::retention::{Installed, Keys};
    Installed.key_at(Some(pid), u64::MAX)
}

/// The tokens a holder's keys keep above the whole of it.
fn holder_head(pid: ProcessId) -> u64 {
    use crate::policy::retention::{Installed, Keys};
    let whole = holder_key(pid);
    Installed
        .steps(pid)
        .into_iter()
        .filter(|&upto| upto > 0 && Installed.key_at(Some(pid), upto - 1) > whole)
        .max()
        .unwrap_or(0)
}
