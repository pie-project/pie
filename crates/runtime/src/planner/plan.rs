use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::Notify;

use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::holdings::{swapped_page_count, swapped_rs_units};
use super::proc::Residency;
use super::wait::{EntryKey, WaitKind, Waiter};
use super::{ProcessId, ResidencyPlanner, exec};

enum Step {
    Absorb { count: u32, fund_by_eviction: bool },
    ShortRs { fund_by_eviction: bool },
    ServeAllocation { key: EntryKey, demand: Demand },
    ServeAllocationBurst { extra: u32 },
    ServeRestore { key: EntryKey, pid: ProcessId },
    Release(DevicePageReservation),
    Done,
}

pub(super) enum ServeOutcome {
    Served(Arc<Notify>),
    Stale(RsSlotReservation),
}

enum Board {
    Go(DevicePageReservation),
    Replan,
}

impl ResidencyPlanner {
    pub fn plan(self: &Arc<Self>) {
        if self.pause_due.swap(false, Ordering::AcqRel) {
            self.pause_held();
        }
        self.reseat_released();
        loop {
            let (rs_free, rs_total) = self.port.rs_stats();
            let rs_growth = self.rs_growth();
            let step = self.with_inner(|inner| {
                let Some((key, waiter)) = inner.unmet_head() else {
                    if inner.accum.len() > 0 && inner.queue.is_empty() {
                        let stranded = std::mem::take(&mut inner.accum);
                        return Step::Release(stranded);
                    }
                    return Step::Done;
                };
                let is_restore = matches!(waiter.kind, WaitKind::Restore { .. });
                let pid = waiter.pid;
                let demand = match &waiter.kind {
                    WaitKind::Allocation { demand, .. } => Some(*demand),
                    WaitKind::Restore { .. } => None,
                };
                let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                if missing > 0 {
                    return Step::Absorb {
                        count: missing,
                        fund_by_eviction: !is_restore || inner.fleet_stalled(),
                    };
                }
                // rs slots are taken whole when the head is served; a short
                // pool is funded as paged pages are, by suspending holders.
                let need = waiter.rs_need();
                if !rs_free.covers(need.add(inner.seating_reserve(pid, rs_growth, need, rs_total)))
                {
                    return Step::ShortRs {
                        fund_by_eviction: !is_restore || inner.fleet_stalled(),
                    };
                }
                match demand {
                    Some(demand) if !demand.rs.is_zero() => Step::ServeAllocation { key, demand },
                    Some(_) => Step::ServeAllocationBurst {
                        extra: inner.burst_shortfall(),
                    },
                    None => {
                        if inner.completion_credit == 0 && !inner.fleet_stalled() {
                            return Step::Done;
                        }
                        inner.completion_credit = inner.completion_credit.saturating_sub(1);
                        Step::ServeRestore { key, pid }
                    }
                }
            });
            let fund_by_eviction = match step {
                Step::Done => return,
                Step::Release(stranded) => {
                    drop(stranded);
                    return;
                }
                Step::Absorb {
                    count,
                    fund_by_eviction,
                } => {
                    if self.absorb_free(count) || self.reclaim_idle_once() {
                        continue;
                    }
                    if self.sweep_stranded() {
                        return;
                    }
                    fund_by_eviction
                }
                Step::ShortRs { fund_by_eviction } => {
                    if self.reclaim_idle_once() || self.yield_head_checkpoints() {
                        continue;
                    }
                    fund_by_eviction
                }
                Step::ServeAllocation { key, demand } => {
                    let Some(rs) = self.reserve_rs_slots(demand.rs) else {
                        continue;
                    };
                    let outcome = self.with_inner(|inner| {
                        match inner.unmet_head() {
                            Some((head, _)) if head == key => {}
                            _ => return ServeOutcome::Stale(rs),
                        }
                        if (inner.accum.len() as u32) < demand.kv_pages {
                            return ServeOutcome::Stale(rs);
                        }
                        let kv = inner.accum.donate(demand.kv_pages);
                        let waiter = inner.queue.get_mut(&key).expect("head exists");
                        let WaitKind::Allocation {
                            outcome, notify, ..
                        } = &mut waiter.kind
                        else {
                            unreachable!("ServeAllocation only targets allocation entries");
                        };
                        debug_assert!(outcome.is_none(), "unserved head carries no outcome");
                        *outcome = Some(Ok(AllocationGrant::new(demand, kv, rs)));
                        ServeOutcome::Served(notify.clone())
                    });
                    match outcome {
                        ServeOutcome::Served(notify) => {
                            self.stats.serves.fetch_add(1, Ordering::Relaxed);
                            notify.notify_waiters();
                            continue;
                        }
                        ServeOutcome::Stale(rs) => {
                            drop(rs);
                            continue;
                        }
                    }
                }
                Step::ServeAllocationBurst { extra } => {
                    if extra > 0 {
                        self.absorb_free(extra);
                    }
                    let wake = self.with_inner(|inner| inner.serve_burst());
                    if !wake.is_empty() {
                        self.stats
                            .serves
                            .fetch_add(wake.len() as u64, Ordering::Relaxed);
                    }
                    for (key, pages, notify) in wake {
                        ptrace!("serve key={:?} kv={:?}", key, pages);
                        notify.notify_waiters();
                    }
                    continue;
                }
                Step::ServeRestore { key, pid } => {
                    let (model, engine) = self.port.locus();
                    let need = Demand {
                        kv_pages: swapped_page_count(pid, model, engine),
                        rs: swapped_rs_units(pid, model, engine),
                    };
                    let stale = self.with_inner(|inner| match inner.queue.get_mut(&key) {
                        Some(Waiter {
                            kind: WaitKind::Restore { demand },
                            ..
                        }) if *demand != need => {
                            *demand = need;
                            true
                        }
                        _ => false,
                    });
                    if stale {
                        continue;
                    }
                    let Some(slots) = self.reserve_rs_slots(need.rs) else {
                        continue;
                    };
                    let boarded = self.with_inner(|inner| {
                        match inner.unmet_head() {
                            Some((head, _)) if head == key => {}
                            _ => return Board::Replan,
                        }
                        if (inner.accum.len() as u32) < need.kv_pages {
                            return Board::Replan;
                        }
                        let Some(proc) = inner.procs.get_mut(&pid) else {
                            inner.queue.remove(&key);
                            return Board::Replan;
                        };
                        if proc.state != Residency::Evicted {
                            return Board::Replan;
                        }
                        proc.state = Residency::Restoring;
                        inner.queue.remove(&key);
                        Board::Go(inner.accum.donate(need.kv_pages))
                    });
                    match boarded {
                        Board::Go(pages) => {
                            exec::spawn_restore(self.clone(), pid, pages, slots);
                            continue;
                        }
                        Board::Replan => {
                            drop(slots);
                            continue;
                        }
                    }
                }
            };
            if fund_by_eviction {
                self.plan_eviction();
            } else {
                self.stats
                    .restore_absorb_short
                    .fetch_add(1, Ordering::Relaxed);
            }
            return;
        }
    }
}
