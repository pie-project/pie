use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use tokio::sync::Notify;

use crate::store::rs::RsUnits;

use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
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
    demote_after: AtomicU64,
    pub(super) port: Arc<dyn PoolPort>,
    pub(super) stats: PlannerStats,
    /// `PoolFacts::seated_first`: a seated lane's growth before a prompt
    /// waiting to be seated, which waits for a lane to finish.
    pub(super) seated_first: bool,
    /// Held while an uncontended claim reads the rings' free room, weighs it
    /// against the seated lanes' growth and takes its grant, so prompts seated
    /// at once each see the others' pages gone.
    pub(super) seating: parking_lot::Mutex<()>,
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
            demote_after: AtomicU64::new(0),
            port,
            stats: PlannerStats::default(),
            seated_first: false,
            seating: parking_lot::Mutex::new(()),
        }
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
    pub(super) fn sweep_stranded(self: &Arc<Self>) -> bool {
        if self.sweep_exhausted.load(Ordering::Acquire)
            || self.sweeping.swap(true, Ordering::AcqRel)
        {
            return false;
        }
        let (swap_free, swap_total) = self.swap_stats();
        let room = swap_free.saturating_sub(swap_total / HOST_RESERVE_DIVISOR) as usize;
        let (model, engine) = self.port.locus();
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
            .then(|| self.port.prepare_sweep(&keep, room))
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
                None => return,
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
        if !self.port.suspend_capable() {
            ptrace!("pause_held: cannot suspend");
            return;
        }
        let held: Vec<(ProcessId, Rank)> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    proc.held
                        && proc.admitted
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                })
                .map(|(pid, proc)| (*pid, proc.rank()))
                .collect()
        });
        ptrace!("pause_held candidates={}", held.len());
        if held.is_empty() {
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
            self.quote_and_pick(held, everything, self.swap_room(), model, engine);
        // No tier takes them: they keep running.
        ptrace!(
            "pause_held picks={} unhostable={} room={:?}",
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
                    .is_some_and(|proc| proc.held && proc.state == Residency::Resident)
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

    pub fn unregister(self: &Arc<Self>, pid: ProcessId) {
        // A seat taken again after the process ended goes back.
        crate::inferlet::process::vacate_seat(pid);
        let (signal, removed) = self.with_inner(|inner| {
            let departed = inner.procs.remove(&pid);
            if departed.is_some() {
                inner.admitting.retain(|(_, waiting), _| *waiting != pid);
            }
            if departed
                .as_ref()
                .is_some_and(|proc| proc.admitted && proc.state == Residency::Resident)
            {
                inner.completion_credit = inner.completion_credit.saturating_add(1);
            }
            let signal = departed.map(|proc| proc.signal);
            inner.settle_eviction(pid);
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
        self.re_arm_idle_reclaim();
        self.poke();
    }

    pub fn pages_freed(self: &Arc<Self>) {
        self.re_arm_idle_reclaim();
        if self.waiters.load(Ordering::Acquire) != 0 {
            self.poke();
        }
    }

    fn demote_idle(self: &Arc<Self>) {
        const BACKOFF_MS: u64 = 100;
        static START: std::sync::OnceLock<crate::rt::Instant> = std::sync::OnceLock::new();
        let now = START
            .get_or_init(crate::rt::Instant::now)
            .elapsed()
            .as_millis() as u64;
        if now < self.demote_after.load(Ordering::Acquire)
            || self.demoting.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let (device_free, device_total) = self.port.device_stats();
        let (host_free, host_total) = self.port.host_stats();
        let room = host_free.saturating_sub(host_total / HOST_RESERVE_DIVISOR) as usize;
        let txn = (device_free.saturating_mul(HOST_RESERVE_DIVISOR) < device_total && room > 0)
            .then(|| self.port.prepare_demote(room))
            .flatten();
        let Some(txn) = txn else {
            self.demoting.store(false, Ordering::Release);
            self.demote_after.store(now + BACKOFF_MS, Ordering::Release);
            return;
        };
        exec::spawn_move_out(self.clone(), txn, "demote", Self::demote_done);
    }

    fn demote_done(self: &Arc<Self>, moved: bool) {
        self.demoting.store(false, Ordering::Release);
        if moved {
            self.re_arm_idle_reclaim();
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
