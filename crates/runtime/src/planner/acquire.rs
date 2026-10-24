use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

use super::error::PlannerError;
use super::grant::{AllocationGrant, Demand};
use super::proc::Residency;
use super::wait::{EntryKey, WaitKind};
use super::{ProcessId, ResidencyPlanner};

pub enum Acquired {
    Granted(AllocationGrant),
    Yield,
    /// The snapshots the demand counted were dropped; quote it again.
    Requote,
}

enum Collect {
    Ready(Result<AllocationGrant, PlannerError>),
    Yield,
    Requote,
    Wait,
}

struct WaitRegistration<'a> {
    planner: &'a Arc<ResidencyPlanner>,
    key: EntryKey,
    active: bool,
}

impl WaitRegistration<'_> {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for WaitRegistration<'_> {
    fn drop(&mut self) {
        if self.active {
            self.planner.cancel_waiter(self.key);
        }
    }
}

impl ResidencyPlanner {
    /// The grant `demand` takes without parking, if the pools hand it out
    /// now; see `acquire`.
    pub fn grant_now(self: &Arc<Self>, pid: ProcessId, demand: Demand) -> Option<AllocationGrant> {
        if demand.is_zero() {
            if self.waiters.load(Ordering::Acquire) != 0
                || self.nonresident.load(Ordering::Acquire) != 0
            {
                self.note_progress(pid);
            }
            return Some(AllocationGrant::empty());
        }
        let uncontended = self.waiters.load(Ordering::Acquire) == 0
            && self.nonresident.load(Ordering::Acquire) == 0;
        if !self.seated_first {
            if !uncontended {
                self.note_progress(pid);
            }
            if uncontended && let Some(grant) = self.try_reserve(demand) {
                return Some(grant);
            }
            return None;
        }
        if uncontended {
            // The room read, the growth held back and the grant taken as one
            // step: a burst of prompts read apart from their grants would each
            // see the room the others are about to take, and seat lanes their
            // first page boundary finds short.
            let _seating = self.seating.lock();
            if !demand.rs.is_zero() {
                let (rs_free, rs_total) = self.port.rs_stats();
                let reserve =
                    self.lock_inner()
                        .seating_reserve(pid, self.rs_growth(), demand.rs, rs_total);
                if !rs_free.covers(demand.rs.add(reserve)) {
                    return None;
                }
            }
            let grant = self.try_reserve(demand)?;
            if let Some(proc) = self.lock_inner().procs.get_mut(&pid) {
                proc.seated = true;
            }
            return Some(grant);
        }
        let first = {
            let mut inner = self.lock_inner();
            inner.procs.get_mut(&pid).is_some_and(|proc| {
                proc.progressed = true;
                proc.seated
            }) && inner.growth_first(pid)
        };
        if first && let Some(grant) = self.try_reserve(demand) {
            return Some(grant);
        }
        None
    }

    pub async fn acquire(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<Acquired, PlannerError> {
        if let Some(grant) = self.grant_now(pid, demand) {
            return Ok(Acquired::Granted(grant));
        }
        let (_, kv_total) = self.port.device_stats();
        if demand.kv_pages > kv_total {
            return Err(PlannerError::Impossible {
                need: demand.kv_pages,
                total: kv_total,
            });
        }
        if !demand.rs.is_zero() {
            let (rs_free, rs_total) = self.port.rs_stats();
            if !rs_total.covers(demand.rs) {
                let (need, total) = if demand.rs.slots > rs_total.slots {
                    (demand.rs.slots, rs_total.slots)
                } else {
                    (demand.rs.pages, rs_total.pages)
                };
                return Err(PlannerError::Impossible { need, total });
            }
            // A cache entry never makes live work wait: short of state, a
            // resident process quotes again once the idle cache is dropped,
            // or else the snapshots sharing its own state, instead of parking.
            let resident = self.with_inner(|inner| {
                inner.procs.get(&pid).is_some_and(|proc| {
                    proc.state == Residency::Resident && !inner.killing.contains(&pid)
                })
            });
            if !rs_free.covers(demand.rs) && resident {
                // What this frees may be what an older waiter parked on.
                if self.reclaim_idle_once() {
                    self.pages_freed();
                    return Ok(Acquired::Requote);
                }
                if self.yield_own_checkpoints(pid) {
                    self.pages_freed();
                    return Ok(Acquired::Requote);
                }
            }
        }
        enum Parked {
            Entry(EntryKey, Arc<Notify>),
            NotResident,
            Gone,
        }
        let parked = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get(&pid) else {
                return Parked::Gone;
            };
            if proc.state != Residency::Resident {
                return Parked::NotResident;
            }
            let notify = Arc::new(Notify::new());
            let key = inner.enqueue(
                pid,
                WaitKind::Allocation {
                    demand,
                    notify: notify.clone(),
                    outcome: None,
                    yielded: false,
                },
            );
            Parked::Entry(key, notify)
        });
        match parked {
            Parked::Gone => Err(PlannerError::Cancelled),
            Parked::NotResident => Ok(Acquired::Yield),
            Parked::Entry(key, notify) => {
                self.stats.parks.fetch_add(1, Ordering::Relaxed);
                ptrace!("park key={:?} pid={} kv={:?}", key, pid, demand.kv_pages);
                crate::scheduler::worker::notify_lane_close(quorum_id, Some(pid));
                let mut registration = WaitRegistration {
                    planner: self,
                    key,
                    active: true,
                };
                self.poke();
                loop {
                    let notified = notify.notified();
                    tokio::pin!(notified);
                    notified.as_mut().enable();
                    let acquired = match self.collect_outcome(key) {
                        Collect::Wait => {
                            notified.await;
                            continue;
                        }
                        Collect::Ready(outcome) => {
                            ptrace!("collect key={:?} ok={}", key, outcome.is_ok());
                            if matches!(outcome, Err(PlannerError::Cancelled)) {
                                self.stats.cancelled_waits.fetch_add(1, Ordering::Relaxed);
                            }
                            outcome.map(Acquired::Granted)
                        }
                        Collect::Yield => Ok(Acquired::Yield),
                        Collect::Requote => Ok(Acquired::Requote),
                    };
                    registration.disarm();
                    self.poke();
                    return acquired;
                }
            }
        }
    }

    fn note_progress(&self, pid: ProcessId) {
        if let Some(proc) = self.lock_inner().procs.get_mut(&pid) {
            proc.progressed = true;
        }
    }

    async fn await_signal(&self, signal: &Notify, settled: impl Fn() -> bool) {
        let notified = signal.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if settled() {
            return;
        }
        notified.await;
    }

    fn collect_outcome(&self, key: EntryKey) -> Collect {
        self.with_inner(|inner| {
            let Some(key) = inner.locate(key) else {
                return Collect::Ready(Err(PlannerError::Cancelled));
            };
            let Some(waiter) = inner.queue.get_mut(&key) else {
                return Collect::Ready(Err(PlannerError::Cancelled));
            };
            let WaitKind::Allocation {
                outcome, yielded, ..
            } = &mut waiter.kind
            else {
                unreachable!("allocation keys never collide with restore entries");
            };
            match outcome.take() {
                Some(outcome) => {
                    let pid = waiter.pid;
                    inner.queue.remove(&key);
                    if self.seated_first
                        && outcome.is_ok()
                        && let Some(proc) = inner.procs.get_mut(&pid)
                    {
                        proc.seated = true;
                    }
                    Collect::Ready(outcome)
                }
                None if *yielded => {
                    let pid = waiter.pid;
                    inner.queue.remove(&key);
                    // A resident waiter yielded only its cache: it quotes
                    // again, with no residency to wait for.
                    match inner.procs.get(&pid) {
                        Some(proc) if proc.state == Residency::Resident => Collect::Requote,
                        _ => Collect::Yield,
                    }
                }
                None => Collect::Wait,
            }
        })
    }

    fn cancel_waiter(self: &Arc<Self>, key: EntryKey) {
        let removed = self.with_inner(|inner| {
            let key = inner.locate(key)?;
            inner.queue.remove(&key)
        });
        if let Some(waiter) = removed {
            drop(waiter);
            self.poke();
        }
    }

    pub fn is_resident(&self, pid: ProcessId) -> bool {
        self.inner
            .lock()
            .procs
            .get(&pid)
            .is_none_or(|proc| proc.state == Residency::Resident)
    }

    pub fn residency_flag(&self, pid: ProcessId) -> Option<Arc<AtomicBool>> {
        self.lock_inner()
            .procs
            .get(&pid)
            .map(|proc| proc.resident.clone())
    }

    pub async fn wait_resident(&self, pid: ProcessId) -> Result<(), PlannerError> {
        let mut left_pipeline = false;
        loop {
            let signal = {
                let inner = self.lock_inner();
                let Some(proc) = inner.procs.get(&pid) else {
                    return Err(PlannerError::Cancelled);
                };
                if proc.state == Residency::Resident {
                    return Ok(());
                }
                proc.signal.clone()
            };
            if !left_pipeline {
                left_pipeline = true;
                crate::scheduler::worker::notify_process_suspend(pid);
            }
            self.stats.gate_parks.fetch_add(1, Ordering::Relaxed);
            self.await_signal(&signal, || self.is_resident(pid)).await;
        }
    }
}
