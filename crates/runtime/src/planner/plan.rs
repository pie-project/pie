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
        /// What the free slots lack for the head.
        need: RsUnits,
        /// The head's whole ask, the free slots it will take included: what
        /// a shared pool's segment walker grows for. The free slots are no
        /// one's yet, so a shortfall alone would under-count the demand by
        /// them, and a pool whose free slots fall just short of a head
        /// would never be asked to grow.
        ask: RsUnits,
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
        if crate::policy::retention::take_idle_cache_new() {
            self.sweep_tiers();
        }
        let on_grow = self.plan_once();
        // The waiters parked behind a grow the walk was promised, now.
        let parked = if on_grow {
            self.with_inner(|inner| inner.queue.values().filter(|w| w.is_unmet()).count())
        } else {
            0
        };
        self.stats
            .grow_parked
            .store(parked as u64, Ordering::Relaxed);
        // A waiter served or gone lacks nothing now.
        self.report_short();
    }

    /// The segment walk made no grow it promised: the plan this wakes goes
    /// on past the grow, to demote, sweep and evict, on this event alone.
    pub fn no_grow(self: &Arc<Self>) {
        self.stats.no_grows.fetch_add(1, Ordering::Relaxed);
        ptrace!("no grow: the walk mapped nothing it promised");
        self.no_grow.store(true, Ordering::Release);
        self.poke();
    }

    /// Asks the pool to grow by `need`, unless the walk said this plan no
    /// grow is coming (`skip`): true where the plan waits on the grow, which
    /// `pages_freed` (a segment mapped) or [`Self::no_grow`] ends.
    fn wait_on_grow(&self, skip: bool, need: Demand) -> bool {
        if skip || !self.port.grow(need) {
            return false;
        }
        self.stats.grow_waits.fetch_add(1, Ordering::Relaxed);
        ptrace!("plan waits on a grow kv={} rs={:?}", need.kv_pages, need.rs);
        true
    }

    /// One pass; true where it ends waiting on a grow.
    fn plan_once(self: &Arc<Self>) -> bool {
        if self.pause_due.swap(false, Ordering::AcqRel) {
            self.pause_held();
        }
        self.reseat_released();
        // Taken once a pass: every grow step of this pass goes on past it.
        let no_grow = self.no_grow.swap(false, Ordering::AcqRel);
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
                        ask: seated,
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
                Step::Done => return false,
                Step::Release(stranded) => {
                    drop(stranded);
                    return false;
                }
                Step::Absorb {
                    count,
                    fund_by_eviction,
                } => {
                    if self.absorb_free(count) {
                        continue;
                    }
                    if self.wait_on_grow(
                        no_grow,
                        Demand {
                            kv_pages: count,
                            rs: RsUnits::default(),
                        },
                    ) {
                        return true;
                    }
                    // The device holds the highest keys: idle cache at or
                    // below the floor (which never moves down) goes first,
                    // then what is keyed above it moves down a tier.
                    if self.reclaim_idle_once() {
                        continue;
                    }
                    if self.demote_idle(count) {
                        return false;
                    }
                    if self.sweep_stranded(count, false) {
                        return false;
                    }
                    fund_by_eviction
                }
                Step::ShortRs {
                    need,
                    ask,
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
                    if self.wait_on_grow(
                        no_grow,
                        Demand {
                            kv_pages: 0,
                            rs: ask,
                        },
                    ) {
                        return true;
                    }
                    // On a shared pool the chunks idle cache holds open for
                    // state too: what is keyed above the floor moves down a
                    // tier before any is dropped.
                    let as_pages = self.port.rs_as_pages(need);
                    if self.reclaim_idle_once() || self.reclaim_head_sharing() {
                        continue;
                    }
                    if as_pages > 0 && self.demote_idle(as_pages) {
                        return false;
                    }
                    // And the tails of sets at rest past what their keys
                    // keep, lowest key first, before any set moves whole
                    // (its state with it).
                    if as_pages > 0 && self.sweep_stranded(as_pages, true) {
                        return false;
                    }
                    fund_by_eviction
                }
                Step::ServeAllocation { key, demand } => {
                    let Some(rs) = self.reserve_rs_slots(demand.rs) else {
                        if self.wait_on_grow(
                            no_grow,
                            Demand {
                                kv_pages: 0,
                                rs: demand.rs,
                            },
                        ) {
                            return true;
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
                        if self.wait_on_grow(
                            no_grow,
                            Demand {
                                kv_pages: 0,
                                rs: need.rs,
                            },
                        ) {
                            return true;
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
            return false;
        }
    }
}
