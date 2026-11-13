use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::Notify;

use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::holdings::{swapped_page_count, swapped_rs_units};
use super::proc::Residency;
use super::wait::{EntryKey, WaitKind, Waiter};
use super::{ProcessId, ResidencyPlanner, exec};
use crate::store::rs::RsUnits;

enum Step {
    Absorb {
        count: u32,
        fund_by_eviction: bool,
    },
    ShortRs {
        need: RsUnits,
        fund_by_eviction: bool,
        /// Pages gathered past the head's own kv need.
        surplus: u32,
    },
    ServeAllocation {
        key: EntryKey,
        demand: Demand,
    },
    ServeAllocationBurst {
        extra: u32,
    },
    ServeRestore {
        key: EntryKey,
        pid: ProcessId,
    },
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
        self.plan_once();
        // A waiter served or gone lacks nothing now.
        self.report_short();
    }

    fn plan_once(self: &Arc<Self>) {
        if self.pause_due.swap(false, Ordering::AcqRel) {
            self.pause_held();
        }
        self.reseat_released();
        loop {
            let (rs_free, rs_total) = self.port.rs_stats();
            let rs_growth = self.rs_growth();
            // What the device has free past the pages the head gathered:
            // room a restore may take.
            let spare = self.port.device_stats().0;
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
                let seated = need.add(inner.seating_reserve(pid, rs_growth, need, rs_total));
                if !rs_free.covers(seated) {
                    return Step::ShortRs {
                        need: seated.sub(rs_free),
                        fund_by_eviction: !is_restore || inner.fleet_stalled(),
                        surplus: (inner.accum.len() as u32).saturating_sub(waiter.kv_need()),
                    };
                }
                match demand {
                    Some(demand) if !demand.rs.is_zero() => Step::ServeAllocation { key, demand },
                    Some(_) => Step::ServeAllocationBurst {
                        extra: inner.burst_shortfall(),
                    },
                    None => {
                        // A restore goes back once a resident process has
                        // finished (its credit), or the fleet can do nothing
                        // else, or the device has room for it and still a
                        // page for each resident lane's next one: suspended
                        // processes come back as soon as room appears, every
                        // one it holds at once, not one a completion.
                        if inner.completion_credit > 0 {
                            inner.completion_credit -= 1;
                        } else if spare < inner.growth_reserve() && !inner.fleet_stalled() {
                            return Step::Done;
                        }
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
                    if self.absorb_free(count) {
                        continue;
                    }
                    if self.port.grow_pages(count) {
                        return;
                    }
                    // Idle cache a policy keys above the floor moves down
                    // a tier before any is dropped; the rest is dropped.
                    if self.demote_idle(count) {
                        return;
                    }
                    if self.reclaim_idle_once() {
                        continue;
                    }
                    if self.sweep_stranded(count, false) {
                        return;
                    }
                    fund_by_eviction
                }
                Step::ShortRs {
                    need,
                    fund_by_eviction,
                    surplus,
                } => {
                    // On a shared pool the pages gathered for an earlier head
                    // (one that left, or a restart's) hold chunks this head's
                    // state could take: they go back before anything moves.
                    // Kept, they starved every replay of a restarted process
                    // of the state slot its lanes asked for.
                    if surplus > 0 && self.port.rs_as_pages(need) > 0 {
                        let spare = self.with_inner(|inner| {
                            let keep = inner.unmet_head().map_or(0, |(_, head)| head.kv_need());
                            let spare = (inner.accum.len() as u32).saturating_sub(keep);
                            inner.accum.donate(spare)
                        });
                        if spare.len() > 0 {
                            drop(spare);
                            continue;
                        }
                    }
                    if self.port.grow_rs(need) {
                        return;
                    }
                    // On a shared pool the chunks idle cache holds open for
                    // state too: what is keyed above the floor moves down a
                    // tier before any is dropped.
                    let as_pages = self.port.rs_as_pages(need);
                    if as_pages > 0 && self.demote_idle(as_pages) {
                        return;
                    }
                    if self.reclaim_idle_once() || self.reclaim_head_sharing() {
                        continue;
                    }
                    // And the tails of sets at rest past what their keys
                    // keep, lowest key first, before any set moves whole
                    // (its state with it).
                    if as_pages > 0 && self.sweep_stranded(as_pages, true) {
                        return;
                    }
                    fund_by_eviction
                }
                Step::ServeAllocation { key, demand } => {
                    let Some(rs) = self.reserve_rs_slots(demand.rs) else {
                        if self.port.grow_rs(demand.rs) {
                            return;
                        }
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
                        *outcome = Some(Ok(AllocationGrant::new(kv, rs)));
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
                        if self.port.grow_rs(need.rs) {
                            return;
                        }
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
