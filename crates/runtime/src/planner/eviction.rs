use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::store::kv::page_table::ReclaimQuote;
use crate::store::rs::RsUnits;

use super::error::StarveCause;
use super::grant::AllocationGrant;
use super::holdings::suspendable_rs_units;
use super::plan::ServeOutcome;
use super::proc::Residency;
use super::rank::Rank;
use super::reclaim::Reclaim;
use super::residency_planner::HOST_RESERVE_DIVISOR;
use super::victim::{Victim, VictimSet};
use super::wait::{EntryKey, WaitKind};
use super::{ProcessId, ResidencyPlanner};

impl ResidencyPlanner {
    pub(super) fn quote_and_pick(
        &self,
        candidates: Vec<(ProcessId, Rank)>,
        deficit: Reclaim,
        room: Reclaim,
        model: usize,
        engine: usize,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        let mut ordered: Vec<(ProcessId, Rank, bool)> = candidates
            .into_iter()
            .map(|(pid, rank)| {
                let quiescent =
                    crate::inferlet::process::residency::kv_lease_quiescent(pid, model, engine);
                (pid, rank, quiescent)
            })
            .collect();
        ordered.sort_by_key(|&(_, rank, quiescent)| (!quiescent, std::cmp::Reverse(rank)));
        let pids: Vec<ProcessId> = ordered.iter().map(|(pid, ..)| *pid).collect();
        let rs_quotes: Vec<RsUnits> = pids
            .iter()
            .map(|&pid| suspendable_rs_units(pid, model, engine))
            .collect();
        Self::pick_with_budget_escalation(&pids, &rs_quotes, deficit, room, |pids, budget| {
            crate::inferlet::process::residency::kv_reclaim_quotes(pids, model, engine, budget)
        })
    }

    fn pick_with_budget_escalation(
        pids: &[ProcessId],
        rs_quotes: &[RsUnits],
        deficit: Reclaim,
        room: Reclaim,
        quote: impl Fn(&[ProcessId], u32) -> Vec<Option<ReclaimQuote>>,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        // A pick moves its pages whole, so every candidate is quoted when
        // slots, not pages, are what the head lacks.
        let budget = if !deficit.rs.is_zero() {
            u32::MAX
        } else {
            deficit.pages
        };
        let (picks, unhostable) =
            Self::pick_from_quotes(pids, quote(pids, budget), rs_quotes, deficit, room);
        let covered = picks
            .iter()
            .fold(Reclaim::default(), |sum, &(_, pick)| sum.add(pick));
        if !covered.covers(deficit) && !unhostable.is_empty() {
            return Self::pick_from_quotes(pids, quote(pids, u32::MAX), rs_quotes, deficit, room);
        }
        (picks, unhostable)
    }

    /// Picks candidates in order until what they free covers the deficit.
    /// A pick's pages move whole, so one whose pages exceed the room left
    /// is unhostable; its rs slots park only when host rows hold them all,
    /// and stay on the device otherwise. One that adds nothing the head
    /// still lacks stays.
    fn pick_from_quotes(
        pids: &[ProcessId],
        quotes: Vec<Option<ReclaimQuote>>,
        rs_quotes: &[RsUnits],
        deficit: Reclaim,
        room: Reclaim,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        let mut picks = Vec::new();
        let mut unhostable = Vec::new();
        let (mut covered, mut room) = (Reclaim::default(), room);
        for ((&pid, quote), &rs) in pids.iter().zip(quotes).zip(rs_quotes) {
            if covered.covers(deficit) {
                break;
            }
            let pages = match quote {
                Some(ReclaimQuote::Pages(pages)) => pages,
                _ => 0,
            };
            let rs = if room.rs.covers(rs) {
                rs
            } else {
                RsUnits::default()
            };
            let frees = Reclaim { pages, rs };
            let helps = (pages > 0 && covered.pages < deficit.pages)
                || (rs.slots > 0 && covered.rs.slots < deficit.rs.slots)
                || (rs.pages > 0 && covered.rs.pages < deficit.rs.pages);
            if !helps {
                continue;
            }
            if !frees.fits(room) {
                unhostable.push(pid);
                continue;
            }
            room = room.sub(frees);
            covered = covered.add(frees);
            picks.push((pid, frees));
        }
        (picks, unhostable)
    }

    /// Free and total slots suspended kv can go to: host, then disk.
    pub(super) fn swap_stats(&self) -> (u32, u32) {
        let (host_free, host_total) = self.port.host_stats();
        let (disk_free, disk_total) = self.port.disk_stats();
        (host_free + disk_free, host_total + disk_total)
    }

    /// The room below the device: swap slots for pages, host rows for rs.
    pub(super) fn swap_room(&self) -> Reclaim {
        Reclaim {
            pages: self.swap_stats().0,
            rs: self.port.rs_host_stats().0,
        }
    }

    pub(super) fn plan_eviction(self: &Arc<Self>) {
        let Some(victims) = self.victim_set() else {
            return;
        };
        let (swap_free, swap_total) = self.swap_stats();
        if victims.deficit.pages > 0 {
            if !self.port.suspend_capable() || swap_free == 0 {
                self.check_starvation(StarveCause::NoSwapRoom);
                return;
            }
            if (swap_free.saturating_mul(HOST_RESERVE_DIVISOR) <= swap_total && !self.is_wedged())
                || !self.supply_stalled()
            {
                self.stats
                    .eviction_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let (model, engine) = self.port.locus();
        let (picks, unhostable) = self.quote_and_pick(
            victims.preferred(),
            victims.deficit,
            self.swap_room(),
            model,
            engine,
        );
        if !unhostable.is_empty() {
            self.with_inner(|inner| {
                for pid in &unhostable {
                    inner.host_swap_blocked.insert(*pid);
                }
            });
            self.record_host_swap_exhaustion();
        }
        if picks.is_empty() {
            if victims.deficit.pages == 0 {
                self.check_rs_starvation(victims.head);
            } else if !unhostable.is_empty() {
                self.check_starvation(StarveCause::NoSwapRoom);
            } else if !self.check_hog(victims.head, victims.head_pid) {
                self.check_starvation(StarveCause::NoEligibleVictim);
            }
            return;
        }
        self.commit_evictions(victims.head, picks, false);
    }

    pub(super) fn victim_set(&self) -> Option<VictimSet> {
        let (free, _) = self.port.device_stats();
        let (rs_free, rs_total) = self.port.rs_stats();
        let rs_growth = self.rs_growth();
        self.with_inner(|inner| {
            let (head, waiter) = inner.unmet_head()?;
            // What the device still has free and an eviction in flight
            // returns is supply, not deficit, as `supply_stalled` counts it.
            let missing = waiter
                .kv_need()
                .max(inner.queued_quantum())
                .saturating_sub(inner.accum.len() as u32);
            let expected = inner.expected_reclaim();
            let deficit = Reclaim {
                pages: missing.saturating_sub(expected.pages.saturating_add(free)),
                rs: waiter
                    .rs_need()
                    .add(inner.seating_reserve(waiter.pid, rs_growth, waiter.rs_need(), rs_total))
                    .sub(rs_free.add(expected.rs)),
            };
            if deficit == Reclaim::default() {
                return None;
            }
            let head_rank = head.0;
            // A prompt not yet seated waits for a lane to finish rather than
            // suspending one mid-decode: only an idle or held process yields
            // to it.
            let seating = self.seated_first
                && matches!(waiter.kind, WaitKind::Allocation { .. })
                && inner
                    .procs
                    .get(&waiter.pid)
                    .is_some_and(|proc| !proc.seated);
            let members = inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    proc.admitted
                        && if seating {
                            proc.idle || proc.held
                        } else {
                            proc.rank() > head_rank || proc.idle || proc.held
                        }
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                })
                .map(|(pid, proc)| Victim {
                    pid: *pid,
                    rank: proc.rank(),
                    e6_fresh: proc.progressed || proc.idle || proc.held,
                })
                .collect();
            Some(VictimSet {
                head,
                head_pid: waiter.pid,
                deficit,
                members,
            })
        })
    }

    fn commit_evictions(
        self: &Arc<Self>,
        head_key: EntryKey,
        picks: Vec<(ProcessId, Reclaim)>,
        e6_relaxed: bool,
    ) -> bool {
        let (wake, spawned) = self.with_inner(|inner| {
            let (mut wake, mut spawned) = (Vec::new(), Vec::new());
            let head_rank = match inner.unmet_head() {
                Some((key, _)) if key == head_key => key.0,
                _ => return (wake, spawned),
            };
            for (pid, expected) in picks {
                let Some(proc) = inner.procs.get(&pid) else {
                    continue;
                };
                if proc.state != Residency::Resident
                    || (proc.rank() <= head_rank && !proc.idle && !proc.held)
                    || (!proc.progressed && !proc.idle && !proc.held && !e6_relaxed)
                {
                    continue;
                }
                wake.extend(inner.begin_eviction(pid, expected));
                spawned.push(pid);
            }
            (wake, spawned)
        });
        let any = !spawned.is_empty();
        for pid in &spawned {
            ptrace!("evict pid={} head={:?}", pid, head_key);
        }
        self.start_evictions(wake, spawned);
        any
    }

    pub(super) fn yield_own_checkpoints(&self, pid: ProcessId) -> bool {
        let (model, engine) = self.port.locus();
        let working_sets: Vec<_> =
            crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine)
                .into_iter()
                .collect();
        !working_sets.is_empty() && self.port.yield_checkpoints(&working_sets) > 0
    }

    /// A cache entry never makes live work wait: when the head's ask is
    /// short of state, the snapshots sharing its state are dropped so its
    /// write lands in place, and the head re-quotes as a yielded waiter.
    pub(super) fn yield_head_checkpoints(self: &Arc<Self>) -> bool {
        let Some((key, pid)) = self.with_inner(|inner| {
            inner
                .unmet_head()
                .and_then(|(key, waiter)| match waiter.kind {
                    WaitKind::Allocation { .. } => Some((key, waiter.pid)),
                    WaitKind::Restore { .. } => None,
                })
        }) else {
            return false;
        };
        if !self.yield_own_checkpoints(pid) {
            return false;
        }
        let wake = self.with_inner(|inner| {
            let waiter = inner.queue.get_mut(&key)?;
            let WaitKind::Allocation {
                notify, yielded, ..
            } = &mut waiter.kind
            else {
                return None;
            };
            *yielded = true;
            Some(notify.clone())
        });
        match wake {
            Some(notify) => {
                notify.notify_waiters();
                true
            }
            None => false,
        }
    }

    pub(super) fn supply_stalled(&self) -> bool {
        let (free, _) = self.port.device_stats();
        self.with_inner(|inner| {
            let accum = inner.accum.len() as u32;
            if inner.kill_in_flight() {
                return false;
            }
            if inner.unmet_head().is_none() {
                return false;
            }
            let expected = inner.expected_reclaim().pages;
            let supply = free + accum + expected;
            if inner.queued_quantum() > supply {
                return true;
            }
            free == 0
                && inner
                    .unmet_head()
                    .is_some_and(|(_, head)| head.kv_need() > accum + expected)
        })
    }

    pub(super) fn is_wedged(&self) -> bool {
        self.with_inner(|inner| inner.fleet_stalled())
    }

    pub(super) fn last_resort_evict(self: &Arc<Self>) -> bool {
        let (model, engine) = self.port.locus();
        let Some(stores) = crate::store::registry::try_get(model, engine) else {
            return false;
        };
        let pids: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
        if pids.is_empty() {
            return false;
        }
        let working_sets =
            crate::inferlet::process::residency::kv_working_sets_for(&pids, model, engine);
        let rs_quotes: Vec<RsUnits> = pids
            .iter()
            .map(|&pid| suspendable_rs_units(pid, model, engine))
            .collect();

        let (head, picks) =
            crate::store::registry::with_kv_lock(&stores.kv, "planner-endgame", |kv| {
                self.with_inner(|inner| {
                    let Some((head, waiter)) = inner.unmet_head() else {
                        return (None, Vec::new());
                    };
                    let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                    let deficit = missing.saturating_sub(inner.expected_reclaim().pages);
                    if deficit == 0 {
                        return (None, Vec::new());
                    }
                    let head_rank = head.0;
                    let quotes = crate::inferlet::process::residency::quote_locked(
                        kv,
                        working_sets,
                        u32::MAX,
                    );
                    let mut legal: Vec<(ProcessId, Rank, bool, Reclaim)> = pids
                        .iter()
                        .zip(quotes)
                        .zip(&rs_quotes)
                        .filter_map(|((pid, quote), &rs)| {
                            let proc = inner.procs.get(pid)?;
                            if (proc.rank() <= head_rank && !proc.idle && !proc.held)
                                || proc.state != Residency::Resident
                                || inner.host_swap_blocked.contains(pid)
                                || inner.prepare_blocked.contains(pid)
                            {
                                return None;
                            }
                            match quote? {
                                ReclaimQuote::Pages(pages) if pages > 0 => Some((
                                    *pid,
                                    proc.rank(),
                                    proc.progressed,
                                    Reclaim { pages, rs },
                                )),
                                _ => None,
                            }
                        })
                        .collect();
                    legal.sort_by_key(|&(_, rank, fresh, _)| (!fresh, std::cmp::Reverse(rank)));
                    let mut picks = Vec::new();
                    let mut covered = 0u32;
                    for (pid, _, _, frees) in legal {
                        if covered >= deficit {
                            break;
                        }
                        covered += frees.pages;
                        picks.push((pid, frees));
                    }
                    (Some(head), picks)
                })
            });
        let (Some(head), false) = (head, picks.is_empty()) else {
            return false;
        };
        if !self.commit_evictions(head, picks, true) {
            return false;
        }
        self.stats.e6_relaxations.fetch_add(1, Ordering::Relaxed);
        true
    }

    pub(super) fn salvage_free_pages(&self) -> bool {
        let missing = self.with_inner(|inner| {
            let (_, head) = inner.unmet_head()?;
            Some(head.kv_need().saturating_sub(inner.accum.len() as u32))
        });
        let salvaged = missing.is_some_and(|missing| missing > 0 && self.absorb_free(missing));
        if salvaged {
            self.stats.salvages.fetch_add(1, Ordering::Relaxed);
        }
        salvaged
    }

    pub(super) fn serve_from_hoard(self: &Arc<Self>) -> bool {
        let Some((key, demand)) = self.with_inner(|inner| {
            let (head_key, _) = inner.unmet_head()?;
            let hoard = inner.accum.len() as u32;
            inner.queue.iter().find_map(|(&key, waiter)| {
                if key == head_key || !waiter.is_unmet() {
                    return None;
                }
                match &waiter.kind {
                    WaitKind::Allocation { demand, .. } if hoard >= demand.kv_pages => {
                        Some((key, *demand))
                    }
                    _ => None,
                }
            })
        }) else {
            return false;
        };
        let Some(rs) = self.reserve_rs_slots(demand.rs) else {
            return false;
        };
        let outcome = self.with_inner(|inner| {
            if (inner.accum.len() as u32) < demand.kv_pages {
                return ServeOutcome::Stale(rs);
            }
            let Some(waiter) = inner.queue.get_mut(&key) else {
                return ServeOutcome::Stale(rs);
            };
            let WaitKind::Allocation {
                demand: queued,
                outcome,
                notify,
                yielded,
            } = &mut waiter.kind
            else {
                return ServeOutcome::Stale(rs);
            };
            if outcome.is_some() || *yielded || *queued != demand {
                return ServeOutcome::Stale(rs);
            }
            let kv = inner.accum.donate(demand.kv_pages);
            *outcome = Some(Ok(AllocationGrant::new(demand, kv, rs)));
            ServeOutcome::Served(notify.clone())
        });
        match outcome {
            ServeOutcome::Served(notify) => {
                self.stats.serves.fetch_add(1, Ordering::Relaxed);
                self.stats.hoard_bypasses.fetch_add(1, Ordering::Relaxed);
                notify.notify_waiters();
                self.poke();
                true
            }
            ServeOutcome::Stale(rs) => {
                drop(rs);
                false
            }
        }
    }
}
