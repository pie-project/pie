use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::store::kv::page_table::ReclaimQuote;
use crate::store::rs::RsUnits;

use super::error::StarveCause;
use super::grant::AllocationGrant;
use super::holdings::suspendable_rs_units;
use super::plan::ServeOutcome;
use super::proc::Residency;
use super::rank::VictimOrder;
use super::reclaim::Reclaim;
use super::residency_planner::HOST_RESERVE_DIVISOR;
use super::victim::{Victim, VictimSet};
use super::wait::{EntryKey, WaitKind};
use super::{ProcessId, ResidencyPlanner};

impl ResidencyPlanner {
    pub(super) fn quote_and_pick(
        &self,
        candidates: Vec<(ProcessId, VictimOrder)>,
        deficit: Reclaim,
        room: Reclaim,
        state_pages: u32,
        model: usize,
        engine: usize,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        let mut ordered: Vec<(ProcessId, VictimOrder, bool)> = candidates
            .into_iter()
            .map(|(pid, order)| {
                let quiescent =
                    crate::inferlet::process::residency::kv_lease_quiescent(pid, model, engine);
                (pid, order, quiescent)
            })
            .collect();
        ordered.sort_by_key(|&(_, order, quiescent)| (!quiescent, order));
        let pids: Vec<ProcessId> = ordered.iter().map(|(pid, ..)| *pid).collect();
        // On a shared pool the state a victim parks frees chunks kv takes:
        // it is worth that many kv pages to a head short of kv.
        let rs_quotes: Vec<(RsUnits, u32)> = pids
            .iter()
            .map(|&pid| {
                let rs = suspendable_rs_units(pid, model, engine);
                (rs, self.port.rs_as_pages(rs))
            })
            .collect();
        Self::pick_with_budget_escalation(
            &pids,
            &rs_quotes,
            deficit,
            room,
            state_pages,
            |pids, budget| {
                crate::inferlet::process::residency::kv_reclaim_quotes(pids, model, engine, budget)
            },
        )
    }

    fn pick_with_budget_escalation(
        pids: &[ProcessId],
        rs_quotes: &[(RsUnits, u32)],
        deficit: Reclaim,
        room: Reclaim,
        state_pages: u32,
        quote: impl Fn(&[ProcessId], u32) -> Vec<Option<ReclaimQuote>>,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        // A pick moves its pages whole, so every candidate is quoted when
        // slots, not pages, are what the head lacks.
        let budget = if !deficit.rs.is_zero() {
            u32::MAX
        } else {
            deficit.pages
        };
        let (picks, unhostable) = Self::pick_from_quotes(
            pids,
            quote(pids, budget),
            rs_quotes,
            deficit,
            room,
            state_pages,
        );
        let covered = picks
            .iter()
            .fold(Reclaim::default(), |sum, &(_, pick)| sum.add(pick));
        if !covered.covers(deficit) && !unhostable.is_empty() {
            return Self::pick_from_quotes(
                pids,
                quote(pids, u32::MAX),
                rs_quotes,
                deficit,
                room,
                state_pages,
            );
        }
        (picks, unhostable)
    }

    /// Picks candidates in order until what they free covers the deficit.
    /// A pick's pages move whole, so one whose pages exceed the room left
    /// is unhostable; its rs slots park only when host rows hold them all,
    /// and stay on the device otherwise. One that adds nothing the head
    /// still lacks stays. Each quote's rs comes with what it is worth in kv
    /// pages (on a shared pool the chunks it frees; else 0), which counts
    /// toward the head's pages.
    ///
    /// Of `deficit.pages`, `state_pages` stand for state the head lacks on a
    /// shared pool. A victim whose own state the host cannot take would
    /// park its kv and leave its record where it lies, freeing no chunk
    /// that state opens: it is picked only for kv the head itself lacks.
    pub(super) fn pick_from_quotes(
        pids: &[ProcessId],
        quotes: Vec<Option<ReclaimQuote>>,
        rs_quotes: &[(RsUnits, u32)],
        deficit: Reclaim,
        room: Reclaim,
        state_pages: u32,
    ) -> (Vec<(ProcessId, Reclaim)>, Vec<ProcessId>) {
        let kv_part = deficit.pages.saturating_sub(state_pages);
        let mut picks = Vec::new();
        let mut unhostable = Vec::new();
        let (mut covered, mut room) = (Reclaim::default(), room);
        for ((&pid, quote), &(rs, worth)) in pids.iter().zip(quotes).zip(rs_quotes) {
            if covered.covers(deficit) {
                break;
            }
            let pages = match quote {
                Some(ReclaimQuote::Pages(pages)) => pages,
                _ => 0,
            };
            let stranded = !rs.is_zero() && !room.rs.covers(rs);
            let (rs, worth) = if stranded {
                (RsUnits::default(), 0)
            } else {
                (rs, worth)
            };
            let frees = Reclaim { pages, rs };
            let wanted = if stranded && state_pages > 0 {
                kv_part
            } else {
                deficit.pages
            };
            let helps = (pages.saturating_add(worth) > 0 && covered.pages < wanted)
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
            covered = covered.add(Reclaim {
                pages: pages.saturating_add(worth),
                rs,
            });
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
        let deficit = self.bytes_of(victims.deficit);
        // A new ask leaves one set's state room on the host: a set at rest
        // there comes back by parking another in its place, so what rests
        // still resumes once the host is full.
        let mut room = self.swap_room();
        if !victims.restore {
            room.rs = room.rs.sub(self.restore_reserve());
        }
        // The host is short where it is the move's way out: its kv swap is
        // full, or its rows cannot take the state the head lacks (on a
        // shared pool, the state a victim must park to free its chunks).
        let state = victims.deficit.rs.add(victims.rs_short);
        let host_short = self.port.suspend_capable()
            && ((victims.deficit.pages > 0 && swap_free == 0) || !room.rs.covers(state));
        // Kv the head itself lacks waits on the supply first; kv that stands
        // for state it lacks moves now, as nothing else frees its chunks.
        let kv_deficit = victims.deficit.pages.saturating_sub(victims.rs_pages);
        if victims.deficit.pages > 0 {
            if !self.port.suspend_capable() || swap_free == 0 {
                self.note_short(victims.head, deficit, if host_short { deficit } else { 0 });
                self.check_starvation(StarveCause::NoSwapRoom);
                return;
            }
            if kv_deficit > 0
                && ((swap_free.saturating_mul(HOST_RESERVE_DIVISOR) <= swap_total
                    && !self.is_wedged())
                    || !self.supply_stalled())
            {
                self.stats
                    .eviction_deferrals
                    .fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let (model, engine) = self.port.locus();
        // A bind short of whole chunks (a run of slots in a chunk, or runs
        // of adjacent chunks): pages moved anywhere free none of them, so
        // the occupants of the cheapest chunks it could take, each free or
        // emptied whole by evicting them, are the picks, every one of them.
        // So is a restore whose role slots are short of runs of chunks
        // ([`super::proc::Proc::roles_short`]): its unpark retakes its dead
        // parts fresh, in their class.
        let shape = self.with_inner(|inner| match inner.queue.get(&victims.head) {
            Some(waiter) => match &waiter.kind {
                WaitKind::BindRoom { run, pieces, .. } => Some((*run, *pieces)),
                WaitKind::Restore { .. } => inner
                    .procs
                    .get(&waiter.pid)
                    .and_then(|proc| proc.roles_short)
                    .map(|need| (need.run, need.pieces)),
                WaitKind::Allocation { .. } => None,
            },
            None => None,
        });
        let candidates = victims.preferred();
        let map = shape.map(|_| {
            let pids: Vec<ProcessId> = candidates.iter().map(|&(pid, _)| pid).collect();
            self.port.chunk_occupants(&pids)
        });
        let (candidates, wanted, occupants) =
            chunk_picks(candidates, victims.deficit, shape, map.as_ref());
        let (mut picks, unhostable) =
            self.quote_and_pick(candidates, wanted, room, victims.rs_pages, model, engine);
        // An occupant with no kv or state to move (only guest scratch,
        // given back at its park) quotes nothing and is no pick by pages:
        // it is one all the same, or its chunk does not empty.
        for pid in occupants {
            if !picks.iter().any(|&(picked, _)| picked == pid) && !unhostable.contains(&pid) {
                picks.push((pid, Reclaim::default()));
            }
        }
        if let Some((run, pieces)) = shape {
            tracing::debug!(
                run,
                pieces,
                picks = ?picks.iter().map(|&(pid, _)| pid).collect::<Vec<_>>(),
                chunks = ?map.as_ref().map(|map| (map.held.len(), map.free.len())),
                "planner: a bind short of whole chunks: the occupants of the cheapest it could \
                 take, or none qualified (every chunk holds a slot no candidate moves)"
            );
        }
        if !unhostable.is_empty() {
            self.with_inner(|inner| {
                for pid in &unhostable {
                    inner.host_swap_blocked.insert(*pid);
                }
            });
            self.record_host_swap_exhaustion();
        }
        if picks.is_empty() {
            // Nothing the planner may move covers the head: it parks, and
            // the policy is asked for room (`tier-pressure`). The host is
            // short too where a victim's move found no room there.
            let host = if host_short || !unhostable.is_empty() {
                deficit
            } else {
                0
            };
            self.note_short(victims.head, deficit, host);
            if victims.deficit.pages == 0 {
                self.check_rs_starvation(victims.head);
            } else if !unhostable.is_empty() {
                self.check_starvation(StarveCause::NoSwapRoom);
            } else if !self.check_hog(victims.head, victims.head_pid) {
                self.check_starvation(StarveCause::NoEligibleVictim);
            }
            return;
        }
        if self.commit_evictions(victims.head, picks, false) {
            // Moves on their way fund it: nothing for the policy yet.
            self.note_short(victims.head, 0, 0);
        } else {
            // Every pick declined at its start: the head is short as with
            // none, and says so, rather than waiting on moves never begun.
            tracing::debug!(head = ?victims.head_pid, "planner: no pick could start its move");
            self.note_short(victims.head, deficit, if host_short { deficit } else { 0 });
        }
    }

    pub(super) fn victim_set(&self) -> Option<VictimSet> {
        let (free, _) = self.port.device_stats();
        let (rs_free, rs_total) = self.port.rs_stats();
        let rs_growth = self.rs_growth();
        let (mut victims, spare) = self.with_inner(|inner| {
            let (head, waiter) = inner.unmet_head()?;
            // What the device still has free and an eviction in flight
            // returns is supply, not deficit, as `supply_stalled` counts it.
            let missing = waiter
                .kv_need()
                .max(inner.queued_quantum())
                .saturating_sub(inner.accum.len() as u32);
            let expected = inner.expected_reclaim();
            // In-flight moves past what the head's own kv takes free chunks
            // for its state too.
            let spare = expected.pages.saturating_sub(missing);
            // A bind short of room is short the pages that, moved out
            // whole, free its chunks: free pages are no runs of chunks. So
            // is the restore of a process such a bind suspended.
            let room = inner.room_pages(waiter).saturating_sub(expected.pages);
            let deficit = Reclaim {
                pages: missing
                    .saturating_sub(expected.pages.saturating_add(free))
                    .saturating_add(room),
                rs: waiter
                    .rs_need()
                    .add(inner.seating_reserve(waiter.pid, rs_growth, waiter.rs_need(), rs_total))
                    .sub(rs_free.add(expected.rs)),
            };
            if deficit == Reclaim::default() {
                return None;
            }
            let head_rank = head.0;
            // Never a set keyed above the head's demand while running work
            // will free room. With the fleet standing still, a resting set
            // (idle, or a handle's at rest) frees nothing by itself, so it
            // moves down a tier for the head whatever its key; running work
            // never does.
            let head_key = crate::policy::schedule::current()
                .and(inner.procs.get(&waiter.pid))
                .map_or(i64::MAX, |proc| proc.key);
            let stalled = inner.fleet_stalled();
            // A set at rest is weighed by the head's retention key, like
            // with like: what keeps sets, not the order of service.
            let head_kept =
                super::residency_planner::holder_key(crate::policy::schedule::plex_id(waiter.pid));
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
                        // On its way into a process: it was asked back.
                        && (!proc.holder || proc.idle)
                        // Waiting at a bind for room, it makes no progress
                        // until room comes: it yields as an idle one does,
                        // to a prompt not yet seated too.
                        && ((stalled && (proc.idle || proc.holder))
                            || (proc.key <= head_key
                                // A set at rest is keyed by what keeps it.
                                && !(proc.holder && proc.kept > head_kept)))
                        && if seating {
                            proc.idle || proc.held || proc.bind_waiting
                        } else {
                            proc.rank() > head_rank
                                || proc.idle
                                || proc.held
                                || proc.bind_waiting
                        }
                        && proc.state == Residency::Resident
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                        // Being failed: it frees its pages as it ends.
                        && !inner.killing.contains(*pid)
                })
                .map(|(pid, proc)| Victim {
                    pid: *pid,
                    order: proc.victim_order(),
                    // A bind waiter makes no progress until room comes: it
                    // counts as fresh, as an idle one does, or it is never
                    // a pick and the bind waits on it for good.
                    e6_fresh: proc.progressed || proc.idle || proc.held || proc.bind_waiting,
                })
                .collect();
            Some((
                VictimSet {
                    head,
                    head_pid: waiter.pid,
                    deficit,
                    // The bind's runs are state the head lacks, not kv to
                    // wait on the supply for: moved now, as nothing else
                    // frees their chunks.
                    rs_pages: room,
                    rs_short: RsUnits::default(),
                    restore: matches!(waiter.kind, WaitKind::Restore { .. }),
                    members,
                },
                spare,
            ))
        })?;
        // State the head lacks on a shared pool is kv to move: the chunks it
        // frees open for the state's shapes. Without this the planner moves
        // nothing for it and the policy is asked to discard instead.
        let as_pages = self.port.rs_as_pages(victims.deficit.rs);
        if as_pages > 0 {
            victims.rs_short = victims.deficit.rs;
            victims.rs_pages = victims
                .rs_pages
                .saturating_add(as_pages.saturating_sub(spare));
            victims.deficit.pages = victims.deficit.pages.saturating_add(victims.rs_pages);
            victims.deficit.rs = RsUnits::default();
            if victims.deficit == Reclaim::default() {
                return None;
            }
        }
        Some(victims)
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
                // Waiting at a bind for room it makes no progress: it
                // yields as an idle one does, as the victim set chose it
                // (dropped here, it was chosen again on every quote and
                // nothing moved: AMD's wedge).
                let yields = proc.idle || proc.held || proc.bind_waiting;
                if proc.state != Residency::Resident
                    || (proc.rank() <= head_rank && !yields)
                    || (!proc.progressed && !yields && !e6_relaxed)
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
            ptrace!(
                "evict pid={} head={:?} landed={}",
                pid,
                head_key,
                self.with_inner(|inner| inner.procs.get(pid).map_or(0, |p| p.landed))
            );
        }
        self.start_evictions(wake, spawned);
        any
    }

    /// Drops the checkpoints sharing `pid`'s state that their keys let the
    /// runtime drop unasked; whether any went.
    pub(super) fn reclaim_own_sharing(&self, pid: ProcessId) -> bool {
        let (model, engine) = self.port.locus();
        let working_sets: Vec<_> =
            crate::inferlet::process::residency::rs_working_set_ids(pid, model, engine)
                .into_iter()
                .collect();
        !working_sets.is_empty() && self.port.reclaim_sharing(&working_sets) > 0
    }

    /// When the head's ask is short of state, the snapshots sharing its
    /// state that their keys let go are dropped so its write lands in place,
    /// and the head re-quotes as a yielded waiter. One a policy keeps stays:
    /// the head asks for room with tier pressure instead.
    pub(super) fn reclaim_head_sharing(self: &Arc<Self>) -> bool {
        let Some((key, pid)) = self.with_inner(|inner| {
            inner
                .unmet_head()
                .and_then(|(key, waiter)| match waiter.kind {
                    WaitKind::Allocation { .. } => Some((key, waiter.pid)),
                    WaitKind::Restore { .. } | WaitKind::BindRoom { .. } => None,
                })
        }) else {
            return false;
        };
        if !self.reclaim_own_sharing(pid) {
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
                    // Its kv, and the pages that free the runs a bind short
                    // of room asks, as the victim set counts them.
                    let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                    let expected = inner.expected_reclaim().pages;
                    let deficit = missing
                        .saturating_sub(expected)
                        .saturating_add(inner.room_pages(waiter).saturating_sub(expected));
                    if deficit == 0 {
                        return (None, Vec::new());
                    }
                    let head_rank = head.0;
                    let head_key = crate::policy::schedule::current()
                        .and(inner.procs.get(&waiter.pid))
                        .map_or(i64::MAX, |proc| proc.key);
                    let stalled = inner.fleet_stalled();
                    let head_kept = super::residency_planner::holder_key(
                        crate::policy::schedule::plex_id(waiter.pid),
                    );
                    let quotes = crate::inferlet::process::residency::quote_locked(
                        kv,
                        working_sets,
                        u32::MAX,
                    );
                    let mut legal: Vec<(ProcessId, VictimOrder, bool, Reclaim)> = pids
                        .iter()
                        .zip(quotes)
                        .zip(&rs_quotes)
                        .filter_map(|((pid, quote), &rs)| {
                            let proc = inner.procs.get(pid)?;
                            if (proc.rank() <= head_rank && !proc.idle && !proc.held)
                                || (!(stalled && (proc.idle || proc.holder))
                                    && (proc.key > head_key
                                        || (proc.holder && proc.kept > head_kept)))
                                || (proc.holder && !proc.idle)
                                || proc.state != Residency::Resident
                                || inner.host_swap_blocked.contains(pid)
                                || inner.prepare_blocked.contains(pid)
                            {
                                return None;
                            }
                            match quote? {
                                ReclaimQuote::Pages(pages) if pages > 0 => Some((
                                    *pid,
                                    proc.victim_order(),
                                    proc.progressed,
                                    Reclaim { pages, rs },
                                )),
                                _ => None,
                            }
                        })
                        .collect();
                    legal.sort_by_key(|&(_, order, fresh, _)| (!fresh, order));
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
                ..
            } = &mut waiter.kind
            else {
                return ServeOutcome::Stale(rs);
            };
            if outcome.is_some() || *yielded || *queued != demand {
                return ServeOutcome::Stale(rs);
            }
            let kv = inner.accum.donate(demand.kv_pages);
            *outcome = Some(Ok(AllocationGrant::new(kv, rs)));
            ServeOutcome::Served(notify.clone())
        });
        match outcome {
            ServeOutcome::Served(notify) => {
                self.stats.serves.fetch_add(1, Ordering::Relaxed);
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

/// The candidates a plan quotes, and what it covers: for a bind short of
/// whole chunks (`shape`, its run and pieces), the occupants of the
/// cheapest chunks it could take ([`super::pool::ChunkMap::cheapest_runs`]),
/// every one of them (the deficit past what they free, so none is left
/// out); else `candidates` against `deficit`, as pages counted anywhere.
fn chunk_picks(
    candidates: Vec<(ProcessId, VictimOrder)>,
    deficit: Reclaim,
    shape: Option<(u32, u32)>,
    map: Option<&super::pool::ChunkMap>,
) -> (Vec<(ProcessId, VictimOrder)>, Reclaim, Vec<ProcessId>) {
    let occupants = shape
        .zip(map)
        .and_then(|((run, pieces), map)| map.cheapest_runs(run, pieces))
        .filter(|occupants| !occupants.is_empty());
    match occupants {
        Some(occupants) => (
            candidates
                .into_iter()
                .filter(|(pid, _)| occupants.contains(pid))
                .collect(),
            Reclaim {
                pages: u32::MAX,
                rs: deficit.rs,
            },
            occupants,
        ),
        None => (candidates, deficit, Vec::new()),
    }
}

#[cfg(test)]
mod chunk_picks_tests {
    use super::super::pool::{ChunkHeld, ChunkMap};
    use super::*;

    fn pid(n: u128) -> ProcessId {
        ProcessId::from_u128(n)
    }

    fn held(chunk: u32, used: u32, by: &[(u128, u32)]) -> ChunkHeld {
        ChunkHeld {
            chunk,
            used,
            by: by.iter().map(|&(p, pages)| (pid(p), pages)).collect(),
        }
    }

    /// Chunk 1 is held by A alone (5 pages), chunk 2 by B and a cache
    /// entry (B's 2 of 4: not emptiable), chunk 3 by C (3 pages): a bind
    /// short of one chunk evicts C, the cheapest chunk that empties.
    #[test]
    fn a_bind_short_of_a_chunk_evicts_the_occupants_of_the_cheapest_emptiable_one() {
        let map = ChunkMap {
            held: vec![
                held(1, 5, &[(1, 5)]),
                held(2, 4, &[(2, 2)]),
                held(3, 3, &[(3, 3)]),
            ],
            free: vec![],
            headroom: None,
            segment_chunks: 0,
        };
        assert_eq!(map.cheapest_runs(1, 1), Some(vec![pid(3)]));
        // Two chunks: C's and A's, B's never.
        let mut two = map.cheapest_runs(1, 2).unwrap();
        two.sort_unstable();
        assert_eq!(two, vec![pid(1), pid(3)]);
        assert_eq!(map.cheapest_runs(1, 3), None, "B's chunk does not empty");
    }

    /// A run of two adjacent chunks: a free chunk beside an emptiable one
    /// costs only that one's pages; chunks apart are no run.
    #[test]
    fn a_run_of_chunks_lies_in_adjacent_chunks_free_or_emptied() {
        let map = ChunkMap {
            held: vec![held(4, 6, &[(1, 6)]), held(7, 2, &[(2, 1), (3, 1)])],
            free: vec![5, 8],
            headroom: None,
            segment_chunks: 0,
        };
        let mut pids = map.cheapest_runs(2, 1).unwrap();
        pids.sort_unstable();
        assert_eq!(pids, vec![pid(2), pid(3)], "7 and 8: two pages to move");
        assert_eq!(
            map.cheapest_runs(2, 2),
            Some(vec![pid(1), pid(2), pid(3)]).map(|mut v| {
                v.sort_unstable();
                v
            })
        );
        assert_eq!(map.cheapest_runs(3, 1), None, "no three adjacent");
    }

    /// The plan's candidates for a bind short of a chunk: the chosen
    /// chunk's occupants only, every one quoted (the deficit past them);
    /// for any other head, or where no chunk empties, the candidates as
    /// they were.
    #[test]
    fn a_bind_room_head_quotes_exactly_one_chunk_s_occupants() {
        let rank = |came: u64| VictimOrder::of(0, 0, came);
        let candidates = vec![(pid(1), rank(1)), (pid(2), rank(2)), (pid(3), rank(3))];
        let deficit = Reclaim {
            pages: 7,
            rs: Default::default(),
        };
        let map = ChunkMap {
            held: vec![held(1, 5, &[(1, 5)]), held(3, 3, &[(3, 3)])],
            free: vec![],
            headroom: None,
            segment_chunks: 0,
        };
        let (picked, wanted, occupants) =
            chunk_picks(candidates.clone(), deficit, Some((1, 1)), Some(&map));
        assert_eq!(occupants, vec![pid(3)]);
        assert_eq!(
            picked.iter().map(|&(p, _)| p).collect::<Vec<_>>(),
            vec![pid(3)]
        );
        assert_eq!(wanted.pages, u32::MAX, "every occupant is picked");
        let (kept, same, _) = chunk_picks(candidates.clone(), deficit, None, Some(&map));
        assert_eq!(kept.len(), 3);
        assert_eq!(same, deficit);
        let none = ChunkMap::default();
        let (kept, same, _) = chunk_picks(candidates, deficit, Some((1, 1)), Some(&none));
        assert_eq!(
            (kept.len(), same),
            (3, deficit),
            "no chunk empties: as before"
        );
    }

    /// Mac Pro's M5 shape B: the cheapest runs stood free, yet the head
    /// waited with no one moved. Free chunks the budget cannot open (no
    /// headroom) are no room: each needs a chunk emptied elsewhere, the
    /// cheapest, whose budget comes back.
    #[test]
    fn a_free_run_past_the_budgets_headroom_empties_a_chunk_elsewhere() {
        let mut map = ChunkMap {
            held: vec![held(10, 4, &[(1, 4)]), held(20, 2, &[(2, 2)])],
            free: vec![3, 4, 5],
            headroom: Some(3),
            segment_chunks: 0,
        };
        assert_eq!(
            map.cheapest_runs(3, 1),
            Some(vec![]),
            "room for the run: no one moves"
        );
        map.headroom = Some(1);
        assert_eq!(
            map.cheapest_runs(3, 1),
            Some(vec![pid(1), pid(2)]),
            "two chunks short of the budget: the two cheapest elsewhere are emptied"
        );
        map.headroom = Some(0);
        assert_eq!(
            map.cheapest_runs(3, 1),
            None,
            "three short and two to empty: no runs stand"
        );
    }

    /// A run lies inside one segment: free chunks either side of a
    /// segment's edge are no run.
    #[test]
    fn a_run_across_a_segments_edge_is_no_run() {
        let map = ChunkMap {
            held: Vec::new(),
            free: vec![6, 7, 8, 9],
            headroom: None,
            segment_chunks: 8,
        };
        assert_eq!(
            map.cheapest_runs(4, 1),
            None,
            "6..10 crosses chunk 8's segment edge"
        );
        assert_eq!(
            map.cheapest_runs(2, 2),
            Some(vec![]),
            "6..8 and 8..10 each lie in one"
        );
    }
}
