use std::collections::HashSet;
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
use super::pool::RunNeed;
use super::reclaim::Reclaim;
use super::victim::{Victim, VictimSet};
use super::wait::{EntryKey, WaitKind};
use super::{ProcessId, ResidencyPlanner};

/// Plan ends that wait for room something in flight gives back: chosen
/// with nothing in flight, each is a planner bug ([`ResidencyPlanner::plan_ended`]).
const WAITS_ON_FLIGHT: &[&str] = &[
    "something in flight",
    "once more chunks stand free",
    "waits for the restored one's bind to fire",
];

impl ResidencyPlanner {
    pub(super) fn quote_and_pick(
        &self,
        candidates: Vec<(ProcessId, Rank)>,
        deficit: Reclaim,
        room: Reclaim,
        state_pages: u32,
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

    /// Free and total slots suspended kv can go to: host, then disk. The
    /// slots only idle cache holds there count as free: a suspend makes its
    /// room by dropping them (`KvStore::make_swap_room`). Counted as full,
    /// a disk tier holding idle cache made every victim unhostable, and a
    /// head the pool fits alone waited with nothing issued.
    pub(super) fn swap_stats(&self) -> (u32, u32) {
        let (host_free, host_total) = self.port.host_stats();
        let (disk_free, disk_total) = self.port.disk_stats();
        let (host_idle, disk_idle) = self.port.idle_below_device();
        let free = host_free
            .saturating_add(disk_free)
            .saturating_add(host_idle)
            .saturating_add(disk_idle)
            .min(host_total.saturating_add(disk_total));
        (free, host_total + disk_total)
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
        let (swap_free, _) = self.swap_stats();
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
        // The queue's quantum (other waiters' one-page asks, counted in the
        // deficit) is no kv of the head's: a bind or a restore whose kv is
        // gathered waited on a full disk for it, its runs one queued
        // resident away and nothing issued (Mac Pro's M5 on 5120cbc3c:
        // need_moves Some(Some(1)), 69 to 91 candidates). A victim the
        // room below cannot take is found unhostable at its quote.
        let kv_deficit = victims.deficit.pages.saturating_sub(victims.rs_pages);
        let kv_own = victims.own_pages.min(kv_deficit);
        if victims.deficit.pages > 0 {
            // No room below for kv: only kv the head itself lacks waits on
            // it. Chunks a head's runs need empty as their occupants' guest
            // scratch is given back and their state moves, which a full
            // swap tier does not stop; returning here for them left a
            // restore head with a plan (an eviction, or the compaction
            // after it) issuing nothing on a disk full of parked kv.
            if kv_own > 0 && (!self.port.suspend_capable() || swap_free == 0) {
                self.plan_ended(victims.head, "no room below for the head's own kv", host_short, &[]);
                self.swap_bound.store(true, Ordering::Release);
                self.note_short(victims.head, deficit, if host_short { deficit } else { 0 });
                self.check_starvation(StarveCause::NoSwapRoom);
                return;
            }
            // Kv the head lacks waits on the supply only while some is on
            // its way (`supply_stalled`: pages free, gathered, or an
            // eviction landing). No share of the swap tiers is kept back
            // from a victim: a fraction held there deferred the head for
            // good under a tight disk budget, nothing in flight to free
            // anything (the disk-only M1 hang), and a victim the swap
            // cannot take is found unhostable when it is quoted.
            if kv_deficit > 0 && !self.supply_stalled() {
                self.plan_ended(victims.head, "kv on its way", host_short, &[]);
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
        let mut shape = self.head_room(victims.head);
        // The map stands a restore's need while the store's placement of
        // its whole restore does not (room there only by emptying, the
        // map's free chunks laid otherwise than the takes lay them): one
        // more piece of its longest run is asked, so the map names the
        // occupants whose move makes it (the L40S on 4cef0c7d3: 1,195 of
        // 1,199 lines with free runs and emptiable chunks enough and no
        // move planned). Once the restore stands, it is sent again.
        if victims.restore
            && let Some(need) = shape.as_mut()
        {
            let mut map = self.port.chunk_occupants(&[]);
            map.merge(self.parked_map(victims.head_pid));
            self.lay_serve_state(victims.head, &mut map);
            self.widen_to_store(victims.head, victims.head_pid, need, &map);
        }
        // Kv the head lacks, with parked state lying on the device (role
        // slots its park found no room below for, as on a home with no host
        // tier): that state moves down a tier first, no one running on it.
        // Left there, a first frame short of kv waited behind chunks only a
        // move of parked state could empty.
        if shape.is_none()
            && victims.deficit.pages > 0
            && self.park_down_for_pages(victims.head_pid, victims.deficit.pages)
        {
            self.plan_ended(victims.head, "parked state moves down", host_short, &[]);
            self.note_short(victims.head, 0, 0);
            return;
        }
        // A process made no progress since its last move is spared only
        // while something in flight may move it on: with nothing in flight
        // (every waiter parked, none progressing), the fresh alone were a
        // map of two stuck chunks, and the head's window never qualified
        // while 371 chunks stood emptiable. Then every member is a
        // candidate; the anti-thrash rule still spares what can fire.
        let quiet = self.with_inner(|inner| inner.nothing_in_flight());
        let candidates = victims.candidates(quiet);
        let residents: Vec<ProcessId> = candidates.iter().map(|&(pid, _)| pid).collect();
        // Parked state lying on the device is an occupant too: a chunk it
        // shares with residents empties once both move, it down a tier and
        // they off the device.
        let map = shape.as_ref().map(|_| {
            let mut map = self.occupants(&residents);
            map.merge(self.parked_map(victims.head_pid));
            self.lay_serve_state(victims.head, &mut map);
            map
        });
        // No eviction forms the head's runs (a chunk in every window holds
        // what no move off the device clears: a resident ranked above the
        // head, the head's own live parts, a shared item a resident holds,
        // a victim no swap room takes): moving chunks is the only way, and
        // a compaction is asked for them. Taken out of the heads' paths
        // after a soak asked one that never came (4a267be69), such heads
        // waited for good with the pool able to stand their runs.
        // Only with nothing in flight: an eviction on its way (its victim
        // no candidate now, the map empty of it) forms the runs as it lands,
        // and a compaction asked meanwhile waited on what it would make.
        if let (Some((run, pieces)), Some(need), Some(map)) =
            (shape.as_ref().and_then(RunNeed::longest), shape.as_ref(), map.as_ref())
            && map.cheapest_need(need).is_none()
            && self.with_inner(|inner| inner.nothing_in_flight())
            && self.port.compaction_comes(run, pieces)
        {
            self.stats.compaction_waits.fetch_add(1, Ordering::Relaxed);
            self.plan_ended(victims.head, "compaction asked", host_short, &[]);
            tracing::debug!(
                head = %victims.head_pid,
                run,
                pieces,
                "planner: no eviction forms the head's runs; a compaction moves chunks for them"
            );
            self.note_short(victims.head, 0, 0);
            return;
        }
        let (candidates, wanted, occupants) =
            chunk_picks(candidates, victims.deficit, shape.as_ref(), map.as_ref());
        let (occupants, parked): (Vec<ProcessId>, Vec<ProcessId>) = occupants
            .into_iter()
            .partition(|pid| residents.contains(pid));
        let parking_down = !parked.is_empty() && self.park_down(victims.head_pid, parked);
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
        if let Some(need) = shape.as_ref() {
            let (run, pieces) = need.longest().unwrap_or((1, 0));
            tracing::debug!(
                ?need,
                picks = ?picks.iter().map(|&(pid, _)| pid).collect::<Vec<_>>(),
                chunks = ?map.as_ref().map(|map| (map.held.len(), map.free.len())),
                "planner: a bind short of whole chunks: the occupants of the cheapest it could \
                 take, or none qualified (every chunk holds a slot no candidate moves)"
            );
            // None qualified: what stands between the free chunks and a run,
            // the planner's map and the pool's own count, so a stall names
            // its cause.
            if picks.is_empty() {
                // A restore head: what was recorded for it (a bind's need
                // at its suspend, or its last unpark's shortfall) and what
                // its restore would take back.
                let (recorded, head) = self.with_inner(|inner| {
                    let head = inner.queue.get(&victims.head).and_then(|waiter| {
                        matches!(waiter.kind, WaitKind::Restore { .. }).then_some(waiter.pid)
                    });
                    let recorded = head.and_then(|pid| inner.procs.get(&pid)).map(|proc| {
                        format!(
                            "bind_room={:?} roles_short={:?} roles_short_why={:?} \
                             roles_free_short={:?} map_vs_refusal_runs={:?}",
                            proc.bind_room.map(|need| (need.run, need.pieces)),
                            proc.roles_short.map(|need| (need.run, need.pieces)),
                            proc.roles_short_why,
                            proc.roles_free_short,
                            proc.roles_short_mismatch,
                        )
                    });
                    (recorded, head)
                });
                tracing::debug!(
                    map = %map.as_ref().map(|map| map.explain(run)).unwrap_or_default(),
                    pool = %self.port.chunk_census(run),
                    head_recorded = %recorded.unwrap_or_default(),
                    head_unpark = %head
                        .map(|pid| self.port.unpark_census(pid, Some((run, pieces))))
                        .unwrap_or_default(),
                    "planner: no run for the head and no one to move: census"
                );
            }
        }
        if !unhostable.is_empty() {
            self.swap_bound.store(true, Ordering::Release);
            let pool_now = self.pool_now();
            self.with_inner(|inner| {
                for pid in &unhostable {
                    inner.host_swap_blocked.insert(*pid);
                }
                inner.blocked_at = Some(pool_now);
            });
            self.record_host_swap_exhaustion();
        }
        if picks.is_empty() && parking_down {
            self.plan_ended(victims.head, "parked state moves down", host_short, &unhostable);
            // The parked state's move funds it: nothing for the policy yet.
            self.note_short(victims.head, 0, 0);
            return;
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
            // Every resident left out for a block its move set (room
            // below short, or nothing found to move) is said as the end,
            // not as the policy's: lifted by the event that ends its cause
            // ([`Self::lapse_blocks`]).
            let blocked = self.with_inner(|inner| {
                victims.members.is_empty()
                    && !(inner.host_swap_blocked.is_empty() && inner.prepare_blocked.is_empty())
            });
            // A restore head the planner reads as fitting is never the
            // policy's to make room for (the M1 on ca5c710a7: need_moves 0,
            // "no pick: the policy is asked" among 111 candidates): either
            // the store's placement stands it, and it is sent again once
            // more chunks stand free than its refusal left, or its own
            // check refuses what the map stands, and that is said.
            let fits = match (victims.restore, shape.as_ref(), map.as_ref()) {
                (true, Some(need), Some(map)) => map
                    .cheapest_need(need)
                    .is_some_and(|moves| moves.is_empty())
                    .then(|| self.store_stands(victims.head, victims.head_pid, need)),
                _ => None,
            };
            if let Some(stands) = fits
                && unhostable.is_empty()
            {
                // Its own check's refusal is the line's `why`
                // (`roles_short_why`); a free chunk's event pokes the plan.
                let end = if stands {
                    "the head's placement stands: sent again once more chunks stand free than its refusal left"
                } else {
                    "the map stands the head and its own check refuses it (why=)"
                };
                self.plan_ended(victims.head, end, host_short, &unhostable);
                self.note_short(victims.head, 0, 0);
                return;
            }
            self.plan_ended(
                victims.head,
                if !unhostable.is_empty() {
                    "every pick unhostable: the policy is asked"
                } else if blocked {
                    "every candidate blocked: its move found no room below or nothing to move"
                } else {
                    "no pick: the policy is asked"
                },
                host_short,
                &unhostable,
            );
            self.note_short(victims.head, deficit, host);
            // The policy asked with nothing in flight while the map of every
            // process names moves for the head: a quiet wait too, said with
            // who those moves name and why this pass moved none of them
            // (the L40S at 3 GiB on f623134fc: kv resident 0, need_moves 7,
            // "no pick: the policy is asked", 185 s with no serve).
            if unhostable.is_empty()
                && !blocked
                && let Some(need) = shape.as_ref()
                && self.with_inner(|inner| inner.nothing_in_flight())
            {
                let everyone: Vec<ProcessId> = self.with_inner(|inner| inner.procs.keys().copied().collect());
                let mut all = self.occupants(&everyone);
                all.merge(self.parked_map(victims.head_pid));
                if let Some(named) = all.cheapest_need(need).filter(|named| !named.is_empty()) {
                    self.stats.quiet_waits.fetch_add(1, Ordering::Relaxed);
                    let states: Vec<String> = self.with_inner(|inner| {
                        named
                            .iter()
                            .map(|pid| match inner.procs.get(pid) {
                                Some(proc) => format!(
                                    "{pid}: {:?} queued={} idle={} held={} holder={} candidate={}",
                                    proc.state,
                                    inner.queue.values().any(|w| w.pid == *pid && w.is_unmet()),
                                    proc.idle,
                                    proc.held,
                                    proc.holder,
                                    victims.members.iter().any(|v| v.pid == *pid),
                                ),
                                None => format!("{pid}: gone"),
                            })
                            .collect()
                    });
                    // Each named one's attribution by source beside the
                    // map's per-chunk count (the L40S on a05b286be: every
                    // named process evicted, holding nothing the warn said).
                    let sources: Vec<String> = named
                        .iter()
                        .map(|pid| format!("{pid}: {}", self.port.occupancy_sources(*pid)))
                        .collect();
                    let chunks: Vec<String> = all
                        .held
                        .iter()
                        .filter(|held| held.by.iter().any(|(pid, _)| named.contains(pid)))
                        .map(|held| format!("chunk {} used {} by {:?}", held.chunk, held.used, held.by))
                        .collect();
                    tracing::warn!(
                        head = %victims.head_pid,
                        ?need,
                        ?states,
                        ?sources,
                        ?chunks,
                        parked_down_refused_at_this_room = self.park_down_failed.load(Ordering::Acquire)
                            == self.port.room_gives(),
                        "planner: the policy asked with nothing in flight while the map names moves \
                         for the head (a planner bug: no step found)"
                    );
                }
            }
            if victims.deficit.pages == 0 {
                self.check_rs_starvation(victims.head);
            } else if !unhostable.is_empty() {
                self.check_starvation(StarveCause::NoSwapRoom);
            } else if !self.check_hog(victims.head, victims.head_pid) {
                self.check_starvation(StarveCause::NoEligibleVictim);
            }
            return;
        }
        if self.commit_evictions(victims.head, picks, quiet) {
            self.plan_ended(victims.head, "evictions issued", host_short, &unhostable);
            // Moves on their way fund it: nothing for the policy yet.
            self.note_short(victims.head, 0, 0);
        } else {
            // Every pick declined at its start: the head is short as with
            // none, and says so, rather than waiting on moves never begun.
            tracing::debug!(head = ?victims.head_pid, "planner: no pick could start its move");
            self.plan_ended(victims.head, "no pick could start its move", host_short, &unhostable);
            self.note_short(victims.head, deficit, if host_short { deficit } else { 0 });
        }
    }

    /// A restore head's `need` widened until the map no longer stands it
    /// where the store's placement of its whole restore does not (its
    /// serve's state, every piece of its unpark, its pending bind's runs):
    /// one more piece of its longest run at a time, never past what the
    /// map's free chunks could hold. The map then names the moves the
    /// store's room needs: read as the map's alone, the M1 on ca5c710a7
    /// read need_moves 0 while its own check found one run of three of the
    /// three the map stood, and the pass asked the policy.
    pub(super) fn widen_to_store(
        &self,
        head: EntryKey,
        pid: ProcessId,
        need: &mut RunNeed,
        map: &super::pool::ChunkMap,
    ) {
        let Some((run, _)) = need.longest() else {
            return;
        };
        // Standing where its last refusal was taken in this same room is
        // no standing: the restore found otherwise there, and holding it
        // for more room with nothing in flight lost its wake (Mac Pro at
        // 2 GiB on f623134fc, twice). Its moves are named instead.
        if !self.refused_in_this_room(pid) && self.store_stands(head, pid, need) {
            return;
        }
        let most = u32::try_from(map.free.len()).unwrap_or(u32::MAX) / run.max(1) + 1;
        for _ in 0..most {
            if !map.cheapest_need(need).is_some_and(|moves| moves.is_empty()) {
                break;
            }
            need.add(run, 1);
        }
    }

    /// Whether `pid`'s last refusal (of any kind: its unpark short, its
    /// pending bind short) was taken in the free room standing now, no
    /// chunk more free than it left: then a placement check that stands it
    /// is wrong, and the plan makes room instead of sending it again. The
    /// backstop for any mismatch between the check and the restore's own
    /// takes (the M1 at 2 GiB on 05c8624c5: 117,270 no-step lines, the check
    /// standing a run of 7 the restore, its pages gathered, found none of).
    pub(super) fn refused_in_this_room(&self, pid: ProcessId) -> bool {
        let free = self.port.free_chunks();
        self.with_inner(|inner| {
            inner
                .procs
                .get(&pid)
                .and_then(|proc| proc.roles_short_free)
                .is_some_and(|left| free <= left)
        })
    }

    /// Whether the store's placement of restore head `head`'s whole
    /// restore stands `need` now: its serve's state, every piece of its
    /// unpark, its pending bind's runs ([`PoolPort::restore_stands`]).
    pub(super) fn store_stands(&self, head: EntryKey, pid: ProcessId, need: &RunNeed) -> bool {
        let (state, bind, kv) = self.with_inner(|inner| {
            let state = inner.queue.get(&head).map(super::wait::Waiter::rs_need);
            let bind = inner
                .procs
                .get(&pid)
                .and_then(|proc| proc.bind_room)
                .map(|need| (need.run, need.pieces));
            (state.unwrap_or_default(), bind, inner.restore_kv_missing(head))
        });
        self.port.restore_stands(pid, need, state, kv, bind)
    }

    /// Records where a plan pass stopped for the queue's head short of
    /// the eviction plan (a grow, parked state moving down, something in
    /// flight): every stall line then says where its head's plan stopped,
    /// with the same fields an eviction plan's end carries.
    pub(super) fn plan_stopped(&self, ended: &'static str) {
        let Some(head) = self.with_inner(|inner| inner.unmet_head().map(|(key, _)| key)) else {
            return;
        };
        self.plan_ended(head, ended, false, &[]);
    }

    /// Records where the eviction plan for `head` stopped, for the stall
    /// dump ([`super::victim::PlanEnd`]).
    pub(super) fn plan_ended(&self, head: EntryKey, ended: &'static str, host_short: bool, unhostable: &[ProcessId]) {
        // A head waits for room only while something in flight or moving
        // gives room back (a fire, a copy, a restore landing): a wait so
        // chosen with nothing in flight is a planner bug, said as one. Mac
        // Pro at 2 GiB on f623134fc stalled twice on such an end.
        if WAITS_ON_FLIGHT.iter().any(|wait| ended.contains(wait))
            && self.with_inner(|inner| inner.nothing_in_flight())
        {
            self.stats.quiet_waits.fetch_add(1, Ordering::Relaxed);
            // Planned again at once: a completion that raced this pass is
            // seen and funded; a wait that stands with nothing in flight is
            // said again, never silent.
            if let Some(notify) = self.returns.drain.get() {
                notify.notify_one();
            }
            tracing::warn!(
                ended,
                "planner: a wait only something in flight could end, chosen with nothing in \
                 flight (a planner bug: no step found)"
            );
        }
        self.with_inner(|inner| {
            inner.last_plan = Some(super::victim::PlanEnd {
                head,
                ended,
                host_short,
                unhostable: unhostable.to_vec(),
            });
        });
    }

    /// The pool's room count and the swap tiers' free slots now: what a
    /// block of a victim is set on ([`Self::lapse_blocks`]).
    pub(super) fn pool_now(&self) -> (u64, u32) {
        let (host_free, _) = self.port.host_stats();
        let (disk_free, _) = self.port.disk_stats();
        // A fire landed is a change too: a victim whose move found nothing
        // to move (its state still settling) may find it settled now, and
        // stayed blocked for good where no room came back (the L40S on
        // 22b0a3e83: a queued process's chunk, no candidate).
        let landed = self.stats.fires_landed.load(Ordering::Relaxed);
        (
            self.port.room_gives().wrapping_add(landed),
            host_free.saturating_add(disk_free),
        )
    }

    /// Victims blocked (their move found no swap room, or nothing to move)
    /// go back among the candidates once the pool has moved past the
    /// block: room given back, or swap slots freed. Cleared only by a
    /// restore or a departure, a block outlived the pool it was set on, and
    /// a stall where neither came kept every such resident out of the
    /// victim set for good. Never on the same pool twice.
    pub(super) fn lapse_blocks(&self) {
        let blocked = self.with_inner(|inner| inner.blocked_at);
        let Some(at) = blocked else {
            return;
        };
        let now = self.pool_now();
        if now.0 == at.0 && now.1 <= at.1 {
            return;
        }
        self.with_inner(|inner| {
            if inner.blocked_at == Some(at) {
                inner.host_swap_blocked.clear();
                inner.prepare_blocked.clear();
                inner.blocked_at = None;
            }
        });
    }

    pub(super) fn victim_set(&self) -> Option<VictimSet> {
        self.lapse_blocks();
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
            let own_pages = waiter
                .kv_need()
                .saturating_sub(inner.accum.len() as u32)
                .saturating_sub(expected.pages.saturating_add(free));
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
            // With nothing in flight, a process waiting in the queue runs
            // nothing: its resident slots are a victim's like any other's,
            // whatever its rank or the head's seating (the M1 on 3b45462d4:
            // 138 queued, 364 opened chunks, no candidate holding one).
            // Read once for every member, each only where some member
            // needs it: per member, the anti-thrash rule's own reads (the
            // queue, then every process) cost a pass tenfold.
            // A process whose frame waits for a raise (cut wider than the
            // cap) is no victim: its eviction waits on that frame, which
            // goes only once raised (Mac Pro at 2 GiB on dc1414ae8: the
            // capped frame's owner evicted, its eviction waiting on its own
            // frame for good).
            let capped_owner = crate::store::shared::owed_by(self.port.locus().1)
                .filter(|_| crate::store::shared::owed_cut_rows(self.port.locus().1) > 0);
            let quiet_cell = std::cell::OnceCell::new();
            let quiet = || *quiet_cell.get_or_init(|| inner.nothing_in_flight());
            let waiting_cell = std::cell::OnceCell::<HashSet<ProcessId>>::new();
            let queued = |pid: &ProcessId| {
                waiting_cell
                    .get_or_init(|| {
                        inner
                            .queue
                            .values()
                            .filter(|waiter| waiter.is_unmet())
                            .map(|waiter| waiter.pid)
                            .collect()
                    })
                    .contains(pid)
            };
            let members = inner
                .procs
                .iter()
                .filter(|(pid, proc)| {
                    // Never the head's own process: a bind short of room
                    // waiting is "bind waiting" as its victims are, and
                    // picked, it evicted itself, came back, and was evicted
                    // again (the M1: three times over, itself first).
                    // Not admitted to run yet (waiting for its execution
                    // seat after its bind): with nothing in flight it runs
                    // nothing and holds what its bind took, a victim like
                    // any other. Left out, the M1 on 7e75af541 stalled with
                    // 342 such processes holding 356 chunks and no
                    // candidate (candidates=0, "no pick").
                    **pid != waiter.pid
                        && Some(**pid) != capped_owner
                        // One rule: any resident process not in a fire in
                        // flight is a candidate for the victim order (rank
                        // first) to choose from, whatever its admission,
                        // its queue or its seat. Read by admission, rank and
                        // seating, 102 residents between fires fell in no
                        // set while one restore was in flight (Mac Pro at
                        // 2 GiB on ca5c710a7), and 342 on the M1 before.
                        // Waiting at a bind, held or idle, it makes no
                        // progress whatever is in flight of its own.
                        && (!proc.firing() || proc.bind_waiting || proc.idle || proc.held)
                        // On its way into a process: it was asked back.
                        && (!proc.holder || proc.idle)
                        && proc.state == Residency::Resident
                        // Restored or bound for a waiting head and not
                        // fired since: never moved back before it runs,
                        // while it can run.
                        // ([`Inner::unfired_protects`], its reads shared.)
                        && !(proc.unfired && !queued(*pid) && !quiet())
                        && !inner.host_swap_blocked.contains(*pid)
                        && !inner.prepare_blocked.contains(*pid)
                        // Being failed: it frees its pages as it ends.
                        && !inner.killing.contains(*pid)
                })
                .map(|(pid, proc)| Victim {
                    pid: *pid,
                    rank: proc.rank(),
                    // A bind waiter makes no progress until room comes: it
                    // counts as fresh, as an idle one does, or it is never
                    // a pick and the bind waits on it for good.
                    e6_fresh: proc.progressed
                        || proc.idle
                        || proc.held
                        || proc.bind_waiting
                        || quiet(),
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
                    own_pages,
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
            let (head_rank, head_pid) = match inner.unmet_head() {
                Some((key, waiter)) if key == head_key => (key.0, waiter.pid),
                _ => return (wake, spawned),
            };
            let quiet = inner.nothing_in_flight();
            let queued: HashSet<ProcessId> = inner
                .queue
                .values()
                .filter(|waiter| waiter.is_unmet() && !matches!(waiter.kind, WaitKind::Restore { .. }))
                .map(|waiter| waiter.pid)
                .collect();
            for (pid, expected) in picks {
                let Some(proc) = inner.procs.get(&pid) else {
                    continue;
                };
                // Waiting at a bind for room it makes no progress: it
                // yields as an idle one does, as the victim set chose it
                // (dropped here, it was chosen again on every quote and
                // nothing moved: AMD's wedge).
                // So does one waiting in the queue with nothing in flight
                // (the victim set's rule; dropped here, the queued-victim
                // change never took: the L40S on 22b0a3e83).
                let waits = queued.contains(&pid);
                // Not in a fire in flight, it runs nothing now: it yields as
                // the victim set chose it (the one rule there).
                let yields =
                    proc.idle || proc.held || proc.bind_waiting || quiet || !proc.firing();
                if proc.state != Residency::Resident
                    || inner.unfired_protects(pid)
                    || (proc.rank() <= head_rank && !yields)
                    || (!proc.progressed && !yields && !e6_relaxed)
                {
                    continue;
                }
                wake.extend(inner.begin_eviction(pid, expected));
                // Evicted for this head: not restored while it waits. And
                // evicted while it waited in the queue: restored only once
                // nothing ahead of it waits ([`Inner::unmet_head`]).
                if let Some(proc) = inner.procs.get_mut(&pid) {
                    proc.evicted_for = Some(head_pid);
                    proc.evicted_queued = waits;
                }
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
    candidates: Vec<(ProcessId, super::rank::Rank)>,
    deficit: Reclaim,
    shape: Option<&RunNeed>,
    map: Option<&super::pool::ChunkMap>,
) -> (Vec<(ProcessId, super::rank::Rank)>, Reclaim, Vec<ProcessId>) {
    let occupants = shape
        .zip(map)
        .and_then(|(need, map)| map.cheapest_need(need))
        .filter(|occupants| !occupants.is_empty());
    // Short of whole chunks with the map read and no runs its candidates
    // could empty: no one is moved by pages counted anywhere, which frees
    // no run (Mac Pro's soak: one resident evicted a line, each freeing
    // nothing the run could take, until none was resident).
    if occupants.is_none() && shape.is_some() && map.is_some() {
        return (Vec::new(), deficit, Vec::new());
    }
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
            held: vec![held(1, 5, &[(1, 5)]), held(2, 4, &[(2, 2)]), held(3, 3, &[(3, 3)])],
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

    /// The L40S's line (a restore of kv 0 whose need read (1, 16) while
    /// its unpark was refused a run of two, 15 chunks open, headroom one):
    /// a need of one entry per run length lays its run of two first and
    /// its single chunks beside it, the budget's headroom read against all
    /// of them. Read as single chunks, four scattered free ones stood the
    /// need, no one moved, and the run of two was never made.
    #[test]
    fn a_need_lays_its_longer_runs_first_and_its_singles_beside_them() {
        let map = ChunkMap {
            held: vec![held(1, 4, &[(1, 4)]), held(9, 1, &[(2, 1)])],
            free: vec![0, 2, 4, 6],
            headroom: None,
            segment_chunks: 0,
        };
        assert_eq!(map.cheapest_runs(1, 4), Some(vec![]), "four singles stand: no one moves");
        let need = RunNeed::of([(1, 2), (2, 1)]);
        assert_eq!(need.entries(), &[(2, 1), (1, 2)], "the longest first");
        assert_eq!(
            map.cheapest_need(&need),
            Some(vec![pid(1)]),
            "the run of two takes 0-1 or 1-2: chunk 1's occupant moves"
        );
        // Headroom for two chunks: the third free chunk the need opens
        // empties a chunk elsewhere, the cheapest.
        let tight = ChunkMap {
            headroom: Some(2),
            ..map.clone()
        };
        let mut pids = tight.cheapest_need(&RunNeed::of([(2, 1), (1, 2)])).unwrap();
        pids.sort_unstable();
        assert_eq!(pids, vec![pid(1), pid(2)], "every chunk it opens read against the headroom");
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
        assert_eq!(map.cheapest_runs(2, 2), Some(vec![pid(1), pid(2), pid(3)]).map(|mut v| {
            v.sort_unstable();
            v
        }));
        assert_eq!(map.cheapest_runs(3, 1), None, "no three adjacent");
    }

    /// Runs chosen together, not cheapest first: free runs at 1-3 and 6-8
    /// of eleven chunks, A's pages at 0, 4, 5 and 9, three runs of three
    /// asked. The two free runs leave no third beside them; moving A stands
    /// 0-2, 3-5 and 6-8. Taken cheapest first, the map said no runs, and
    /// the head (fitting the pool alone) waited with no one moved.
    #[test]
    fn runs_that_fit_only_with_an_occupant_moved_are_found_beside_free_ones() {
        let map = ChunkMap {
            held: vec![
                held(0, 1, &[(1, 1)]),
                held(4, 1, &[(1, 1)]),
                held(5, 1, &[(1, 1)]),
                held(9, 1, &[(1, 1)]),
            ],
            free: vec![1, 2, 3, 6, 7, 8, 10],
            headroom: None,
            segment_chunks: 0,
        };
        assert_eq!(map.cheapest_runs(3, 2), Some(vec![]), "two free runs stand");
        assert_eq!(map.cheapest_runs(3, 3), Some(vec![pid(1)]), "A moves for the third");
        assert_eq!(map.cheapest_runs(3, 4), None, "no four runs of three in eleven");
    }

    /// The plan's candidates for a bind short of a chunk: the chosen
    /// chunk's occupants only, every one quoted (the deficit past them);
    /// for any other head, or where no chunk empties, the candidates as
    /// they were.
    #[test]
    fn a_bind_room_head_quotes_exactly_one_chunk_s_occupants() {
        let rank = |seq: u64| super::super::rank::Rank::of(0, seq);
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
            chunk_picks(candidates.clone(), deficit, Some(&RunNeed::one(1, 1)), Some(&map));
        assert_eq!(occupants, vec![pid(3)]);
        assert_eq!(picked.iter().map(|&(p, _)| p).collect::<Vec<_>>(), vec![pid(3)]);
        assert_eq!(wanted.pages, u32::MAX, "every occupant is picked");
        let (kept, same, _) = chunk_picks(candidates.clone(), deficit, None, Some(&map));
        assert_eq!(kept.len(), 3);
        assert_eq!(same, deficit);
        // No chunk empties: no one is moved by pages counted anywhere,
        // which frees no run (Mac Pro's cascade).
        let none = ChunkMap::default();
        let (kept, _, occupants) =
            chunk_picks(candidates.clone(), deficit, Some(&RunNeed::one(1, 1)), Some(&none));
        assert!(kept.is_empty() && occupants.is_empty(), "no run its eviction makes: none moved");
        // Without a map (no shared pool) the candidates stand as before.
        let (kept, same, _) = chunk_picks(candidates, deficit, Some(&RunNeed::one(1, 1)), None);
        assert_eq!((kept.len(), same), (3, deficit));
    }

    /// Mac Pro's soak, the dominant shape: 1867 chunks free and runs of 4
    /// standing in the planner's map (cost 0), the head short all the same
    /// (its need understated what its unpark takes). The cheapest runs were
    /// "no one", so the plan fell back to evicting by pages: one resident a
    /// line, each freeing nothing the run could take, until none was
    /// resident. A run standing free moves no one; nor does a map where no
    /// candidate's eviction empties a run.
    #[test]
    fn a_head_short_of_a_run_moves_no_one_whose_eviction_makes_none() {
        let rank = |seq: u64| super::super::rank::Rank::of(0, seq);
        let candidates = vec![(pid(1), rank(1)), (pid(2), rank(2))];
        let deficit = Reclaim::default();
        // Free 4..8 and 12..16: runs of 4 stand; the residents hold 9 and 11.
        let standing = ChunkMap {
            held: vec![held(9, 2, &[(1, 2)]), held(11, 1, &[(2, 1)])],
            free: (4..8).chain(12..16).collect(),
            headroom: Some(64),
            segment_chunks: 0,
        };
        assert_eq!(standing.cheapest_runs(4, 1), Some(vec![]), "a run stands");
        let (kept, _, occupants) =
            chunk_picks(candidates.clone(), deficit, Some(&RunNeed::one(4, 1)), Some(&standing));
        assert!(kept.is_empty() && occupants.is_empty(), "a free run: no one moved");
        // No candidate's chunk lies where a run could form: none moved.
        let scattered = ChunkMap {
            held: vec![held(1, 3, &[(1, 1)]), held(3, 3, &[(2, 1)])],
            free: vec![0, 2, 4],
            headroom: Some(64),
            segment_chunks: 0,
        };
        assert_eq!(scattered.cheapest_runs(3, 1), None, "their chunks do not empty");
        let (kept, _, _) = chunk_picks(candidates, deficit, Some(&RunNeed::one(3, 1)), Some(&scattered));
        assert!(kept.is_empty(), "evicting them makes no run: none moved");
        assert!(scattered.explain(3).contains("stuck=2"), "{}", scattered.explain(3));
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
        assert_eq!(map.cheapest_runs(3, 1), Some(vec![]), "room for the run: no one moves");
        map.headroom = Some(1);
        assert_eq!(
            map.cheapest_runs(3, 1),
            Some(vec![pid(1), pid(2)]),
            "two chunks short of the budget: the two cheapest elsewhere are emptied"
        );
        map.headroom = Some(0);
        assert_eq!(map.cheapest_runs(3, 1), None, "three short and two to empty: no runs stand");
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
        assert_eq!(map.cheapest_runs(4, 1), None, "6..10 crosses chunk 8's segment edge");
        assert_eq!(map.cheapest_runs(2, 2), Some(vec![]), "6..8 and 8..10 each lie in one");
    }
}
