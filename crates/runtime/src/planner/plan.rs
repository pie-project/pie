use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::Notify;

use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::holdings::{swapped_page_count, swapped_rs_units};
use super::inner::RestoreGate;
use super::proc::{PendingBind, Residency};
use super::pool::RunNeed;
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
    /// The head is a restore whose role slots found no room, and nothing
    /// else will free any: a resident process is evicted for it.
    /// The head restore's role slots are short of `need`, `(run, pieces)`
    /// runs of whole chunks.
    /// Room is made for it only where `fund` (the fleet stalled or
    /// waiting at binds); free runs of its need send it again either way.
    EvictForRoles { need: (u32, u32), fund: bool },
    /// The head is a restore of a process a bind short of guest-state
    /// room suspended, its kv gathered: it waits for the room that bind
    /// asks as the bind would ([`Self::BindRoom`]).
    RestoreBindRoom {
        key: EntryKey,
        need: PendingBind,
    },
    /// The head is a bind short of device room for its guest state, or a
    /// rung raise short of room (`rise`, its rung's tokens).
    BindRoom {
        key: EntryKey,
        run: u32,
        pieces: u32,
        singles: u32,
        rise: Option<u32>,
    },
    Release(DevicePageReservation),
    /// Nothing to do this pass, and why (recorded as the pass's end).
    Done(&'static str),
}

pub(super) enum ServeOutcome {
    Served(Arc<Notify>),
    Stale(RsSlotReservation),
}

enum Board {
    Go(DevicePageReservation),
    Replan,
}

/// While a head stays unserved with nothing in flight, its stall dump is
/// said again as often as a process waiting at admission says why
/// ([`super::residency_planner::ADMIT_SAYS_AFTER`]), the drain task looking
/// with no plan pass: a stall that goes on is never silent, and no timer
/// ends it.
pub(super) const STALL_SAYS_AGAIN: std::time::Duration =
    super::residency_planner::ADMIT_SAYS_AFTER;

/// A count of processes and the first few of them, for a log line.
pub(super) fn some_pids<'a>(pids: impl ExactSizeIterator<Item = &'a ProcessId>) -> String {
    const SHOWN: usize = 4;
    let count = pids.len();
    let shown: Vec<String> = pids.take(SHOWN).map(ToString::to_string).collect();
    format!("{count}{:?}", shown)
}

// Tests that hold the planner to the progress invariant panic where a settled
// pass leaves a head waiting with nothing in flight (`dump_stall`). Per
// thread: a test's passes run on its own thread, and tests run side
// by side.
#[cfg(test)]
thread_local! {
    pub(super) static PROGRESS_STRICT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl ResidencyPlanner {
    pub fn plan(self: &Arc<Self>) {
        // A grant dropped unspent during the pass gave back room the pass
        // may have read before it came: the pass runs again on the pool as
        // it stands ([`super::grant::RoomReturns`]), so the drop, which
        // cannot plan itself, wakes the head within this step.
        // Edge-triggered: a wake on the generation the last no-step pass
        // ended on reads what that pass read, and plans nothing new. A
        // stuck head is said once and costs no pass until an event (the
        // Mac Pro's g2 spin: 56,671 no-step lines, ~199 s of map passes).
        let generation = self.generation();
        ptrace!("plan pass settled={}", *self.settled_at.lock() == Some(generation));
        if *self.settled_at.lock() == Some(generation) {
            self.stats.skipped_wakes.fetch_add(1, Ordering::Relaxed);
            self.returns
                .wake_cause
                .store(super::grant::WAKE_CALL, Ordering::Release);
            return;
        }
        self.stats.plan_passes.fetch_add(1, Ordering::Relaxed);
        self.pass_landed.store(
            self.stats.fires_landed.load(Ordering::Acquire),
            Ordering::Release,
        );
        let began = crate::rt::Instant::now();
        struct Timed<'a>(&'a super::stats::PlannerStats, crate::rt::Instant);
        impl Drop for Timed<'_> {
            fn drop(&mut self) {
                self.0
                    .pass_took(u64::try_from(self.1.elapsed().as_micros()).unwrap_or(u64::MAX));
            }
        }
        let _timed = Timed(&self.stats, began);
        let cause = self
            .returns
            .wake_cause
            .swap(super::grant::WAKE_CALL, Ordering::AcqRel);
        let on_grow = loop {
            self.returns.planning.store(true, Ordering::Release);
            self.returns.plan_begins();
            self.swap_bound.store(false, Ordering::Release);
            let steps = crate::rt::Instant::now();
            let on_grow = loop {
                let seen = self.returns.epoch();
                let on_grow = self.plan_once();
                if self.returns.epoch() == seen {
                    break on_grow;
                }
            };
            self.phase_took(0, steps);
            let wants = crate::rt::Instant::now();
            self.publish_wants();
            self.returns.planning.store(false, Ordering::Release);
            self.phase_took(1, wants);
            // Room that came free while the pass ran (its signal woke no one
            // then) and meets what the head lacks now: planned on at once.
            if self.returns.waiters.load(Ordering::Acquire) == 0
                || !(self.returns.freed_since_plan() && self.wants_met_now())
            {
                break on_grow;
            }
        };
        // A head left stuck is passed over and the pass planned on for the
        // waiters behind it, until one moves or none is left
        // ([`Self::stuck_head`]); the stall dump then reads the queue's
        // true head.
        let mut on_grow = on_grow;
        let mut passed = false;
        while !on_grow && let Some(stuck) = self.stuck_head() {
            self.with_inner(|inner| inner.passed.push(stuck));
            passed = true;
            on_grow = self.plan_once();
        }
        if passed {
            self.with_inner(|inner| inner.passed.clear());
            self.publish_wants();
        }
        let rest = crate::rt::Instant::now();
        // The waiters parked behind a grow the walk was promised, now.
        let parked = if on_grow {
            self.with_inner(|inner| inner.queue.values().filter(|w| w.is_unmet()).count())
        } else {
            0
        };
        self.stats
            .grow_parked
            .store(parked as u64, Ordering::Relaxed);
        // A force spans the one pass after the stall that set it: one this
        // pass did not take (it ended before its eviction step, or on
        // something in flight) is stale, read later beside evictions and
        // restores in flight (the L40S at 3 GiB on c8077be0f: forced passes
        // said with evicting=2 restoring=1, and after "kv on its way").
        self.force_moves.store(false, Ordering::Release);
        // Stuck as it began, its head is said: only where a head was passed
        // over (every head after it found stuck too, or moved) can one be.
        if !on_grow && passed {
            self.dump_stall();
        }
        // A waiter served or gone lacks nothing now.
        self.report_short();
        self.say_long_waits(cause);
        self.phase_took(2, rest);
        // Ended where it began, it took no step: settled until an event.
        let ended = self.generation();
        *self.settled_at.lock() = (ended == generation).then_some(ended);
    }

    fn phase_took(&self, phase: usize, since: crate::rt::Instant) {
        if let Some(total) = self.stats.plan_phase_us.get(phase) {
            total.fetch_add(
                u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX),
                Ordering::Relaxed,
            );
        }
    }

    /// Passes that built the occupancy map, their total microseconds, and
    /// the quiet waits ([`super::stats::PlannerStats::quiet_waits`]).
    #[must_use]
    pub fn map_census(&self) -> (u64, u64, u64) {
        (
            self.stats.map_passes.load(Ordering::Relaxed),
            self.stats.map_pass_us.load(Ordering::Relaxed),
            self.stats.quiet_waits.load(Ordering::Relaxed),
        )
    }

    /// Wakes skipped on a settled generation, and occupancy maps read from
    /// the cache ([`super::stats::PlannerStats::skipped_wakes`],
    /// [`super::stats::PlannerStats::map_cache_hits`]).
    #[must_use]
    pub fn edge_census(&self) -> (u64, u64) {
        (
            self.stats.skipped_wakes.load(Ordering::Relaxed),
            self.stats.map_cache_hits.load(Ordering::Relaxed),
        )
    }

    /// Settled passes that found a head stuck (each repeat), passes forced
    /// for one, and how long the oldest waiting head has waited (ms): a run
    /// that ends stuck reads so, never clean.
    #[must_use]
    pub fn stuck_census(&self) -> (u64, u64, u64) {
        let oldest = self.with_inner(|inner| {
            inner
                .queue
                .values()
                .filter(|waiter| waiter.is_unmet())
                .map(|waiter| waiter.since.elapsed())
                .max()
                .unwrap_or_default()
        });
        (
            self.stats.stuck_settles.load(Ordering::Relaxed),
            self.stats.forced_passes.load(Ordering::Relaxed),
            u64::try_from(oldest.as_millis()).unwrap_or(u64::MAX),
        )
    }

    /// Restores refused again on the room their last refusal left.
    #[must_use]
    pub fn refusal_repeats(&self) -> u64 {
        self.stats.refusal_repeats.load(Ordering::Relaxed)
    }

    /// A pass's time by phase, in total microseconds: its steps, the wants
    /// published after them, the rest ([`super::stats::PlannerStats::plan_phase_us`]).
    #[must_use]
    pub fn phase_census(&self) -> [u64; 3] {
        [0, 1, 2].map(|phase| self.stats.plan_phase_us[phase].load(Ordering::Relaxed))
    }

    /// Every opened chunk, by the live slots it holds and the state of the
    /// processes holding them: `firing` / `running` (resident, not queued,
    /// a fire in flight or none), `queued` / `queued-firing` (resident,
    /// waiting in the queue), `queued-evicted` (evicted, waiting to be
    /// restored, its state still on the device), `parked` (evicted, not
    /// queued), `other` (slots no process is named for: cache, a
    /// checkpoint). Read as `class:slots-bucket=chunks`. The M1 on
    /// 3b45462d4: 364 opened chunks, none emptiable, no candidate holding
    /// any: whose slots held them was unsaid.
    pub(super) fn held_by_state(&self) -> String {
        let (all, class) = self.with_inner(|inner| {
            let queued: std::collections::HashSet<ProcessId> = inner
                .queue
                .values()
                .filter(|waiter| waiter.is_unmet())
                .map(|waiter| waiter.pid)
                .collect();
            let class: std::collections::HashMap<ProcessId, &'static str> = inner
                .procs
                .iter()
                .map(|(pid, proc)| {
                    let resident = proc.state == Residency::Resident;
                    let label = match (queued.contains(pid), resident) {
                        (true, true) if proc.firing() => "queued-firing",
                        (true, true) => "queued",
                        (true, false) => "queued-evicted",
                        (false, true) if proc.firing() => "firing",
                        (false, true) => "running",
                        (false, false) => "parked",
                    };
                    (*pid, label)
                })
                .collect();
            (inner.procs.keys().copied().collect::<Vec<_>>(), class)
        });
        let mut map = self.occupants(&all);
        map.merge(self.port.parked_occupants(super::residency_planner::SWEEP));
        let bucket = |slots: u32| match slots {
            0 => "0",
            1 => "1",
            2..=3 => "2-3",
            4..=15 => "4-15",
            16..=63 => "16-63",
            _ => "64+",
        };
        let mut counts: std::collections::BTreeMap<(String, &'static str), u32> =
            std::collections::BTreeMap::new();
        for chunk in &map.held {
            let mut labels: std::collections::BTreeSet<&'static str> = chunk
                .by
                .iter()
                .map(|(pid, _)| class.get(pid).copied().unwrap_or("parked"))
                .collect();
            let named: u32 = chunk.by.iter().map(|&(_, slots)| slots).sum();
            if named < chunk.used {
                labels.insert("other");
            }
            let label = labels.into_iter().collect::<Vec<_>>().join("+");
            *counts.entry((label, bucket(chunk.used))).or_default() += 1;
        }
        counts
            .into_iter()
            .map(|((label, slots), chunks)| format!("{label}:{slots}={chunks}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Whether `pid` is a process this planner holds: a request of its that
    /// came back cancelled was dropped under it, not by its departure.
    #[must_use]
    pub fn registered(&self, pid: ProcessId) -> bool {
        self.with_inner(|inner| inner.procs.contains_key(&pid))
    }

    /// Plan passes so far, their total time (us), mean (us) and the power of
    /// two of microseconds 99 of each 100 fell under: the census the field
    /// reads a pass's cost by.
    #[must_use]
    pub fn pass_census(&self) -> (u64, u64, u64, u64) {
        self.stats.pass_census()
    }

    /// Frames the scheduler holds out under the fire cap (capped: complete,
    /// wider than the rung in hand) as the engine's rung gate states them:
    /// the widest one's rows and the process its raise is funded for.
    pub(super) fn capped_line(&self) -> String {
        let (_, engine) = self.port.locus();
        format!(
            "rows={} owner={:?}",
            crate::store::shared::owed_cut_rows(engine),
            crate::store::shared::owed_by(engine)
        )
    }

    /// A restore head's whole restore placed by the store's takes
    /// ([`super::pool::PoolPort::restore_placement`]): its kv pages still to
    /// gather, its serve's state and its unpark's pieces taken first, its
    /// pending bind's runs in what they leave. `None` for any other head.
    pub(super) fn head_placement(
        &self,
        head: EntryKey,
        need: &RunNeed,
    ) -> Option<super::pool::RestorePlacement> {
        let (pid, state, bind, kv) = self.with_inner(|inner| {
            let waiter = inner
                .queue
                .get(&head)
                .filter(|waiter| matches!(waiter.kind, WaitKind::Restore { .. }))?;
            let bind = inner
                .procs
                .get(&waiter.pid)
                .and_then(|proc| proc.bind_room)
                .map(|need| need.entries())
                .unwrap_or_default();
            Some((waiter.pid, waiter.rs_need(), bind, inner.restore_kv_missing(head)))
        })?;
        Some(self.port.restore_placement(pid, need, state, kv, &bind))
    }

    /// Whose the device's kv pages and state units are: every process the
    /// planner knows (resident or evicted), idle cache, and the rest no
    /// one here names (state of a process gone, a checkpoint). The L40S at
    /// 3 GiB on 5120cbc3c held 150 pages and 17 chunks with no process
    /// resident, the pool's census naming only "store" as their holder.
    pub(super) fn store_holders(&self) -> String {
        let (model, engine) = self.port.locus();
        let pids: Vec<(ProcessId, bool)> = self.with_inner(|inner| {
            inner
                .procs
                .iter()
                .map(|(pid, proc)| (*pid, proc.state == Residency::Resident))
                .collect()
        });
        let (mut kv_resident, mut kv_away) = (0u32, 0u32);
        let (mut rs_resident, mut rs_away) = (RsUnits::default(), RsUnits::default());
        for &(pid, resident) in &pids {
            let pages = super::holdings::held_page_count(pid, model, engine);
            let rs = super::holdings::held_rs_units(pid, model, engine);
            if resident {
                kv_resident += pages;
                rs_resident = rs_resident.add(rs);
            } else {
                kv_away += pages;
                rs_away = rs_away.add(rs);
            }
        }
        let kv_out = self.port.device_out();
        let idle = self.port.reclaimable_idle();
        let (rs_free, rs_total) = self.port.rs_stats();
        let rs_held = rs_total.sub(rs_free);
        format!(
            "kv: out={kv_out} resident={kv_resident} not_resident={kv_away} idle={idle} \
             unnamed={}; state: held=({},{}) resident=({},{}) not_resident=({},{}) unnamed=({},{})",
            kv_out.saturating_sub(kv_resident + kv_away + idle),
            rs_held.slots,
            rs_held.pages,
            rs_resident.slots,
            rs_resident.pages,
            rs_away.slots,
            rs_away.pages,
            rs_held.sub(rs_resident.add(rs_away)).slots,
            rs_held.sub(rs_resident.add(rs_away)).pages,
        )
    }

    /// Each wait served this pass past [`super::wait::LONG_WAIT`], with
    /// what woke the pass: a field's gaps between dispatches past 100 ms
    /// (the L40S at 6 GiB) are attributed to the room signal, a poke, the
    /// beat or a call.
    fn say_long_waits(&self, cause: u8) {
        let waits = self.with_inner(|inner| std::mem::take(&mut inner.long_waits));
        if waits.is_empty() {
            return;
        }
        let woken_by = match cause {
            super::grant::WAKE_SIGNAL => "room signal",
            super::grant::WAKE_POKE => "poke (a fire landing, a move ending, pages freed)",
            super::grant::WAKE_BEAT => "beat",
            _ => "call (an arrival, a release)",
        };
        let now = (self.port.free_chunks(), self.port.device_stats().0, self.raw_swap_free());
        for super::wait::LongWait {
            kind,
            lacked,
            wanted: ([want_chunks, want_list, want_swap], [free_then, pages_then]),
            waited,
        } in waits
        {
            let want = |n: u64| if n == u64::MAX { "-".to_owned() } else { n.to_string() };
            if let Some(count) = self.stats.long_waits.get(usize::from(cause)) {
                count.fetch_add(1, Ordering::Relaxed);
            }
            tracing::info!(
                kind,
                lacked,
                want_chunks = %want(want_chunks),
                want_list = %want(want_list),
                want_swap = %want(want_swap),
                free_chunks_then = free_then,
                free_pages_then = pages_then,
                free_chunks_after_pass = now.0,
                free_pages_after_pass = now.1,
                swap_free_after_pass = now.2,
                waited_ms = waited.as_millis() as u64,
                woken_by,
                "planner: a wait past 100 ms served"
            );
        }
    }

    /// The pass ended with the queue's head unserved and nothing in flight
    /// (no eviction, restore, fire or kill, no parked state moving down, no
    /// grow promised): only an outside event (an arrival, a departure, room
    /// given back, a policy answer) plans again, and the engine's stall
    /// watchdog sees no fire to time. Said once per head and room change,
    /// with what the head asks, the queue, and, for a head short of whole
    /// chunks, the pool's own census and the planner's map over every
    /// process, so a silent scheduler names the head that holds it.
    pub(super) fn dump_stall(self: &Arc<Self>) {
        // Room the governor's target holds back is what the head waits on:
        // the target's rise, an outside event, plans again
        // ([`Self::target_raised`]). No bug, and no stall.
        let held_back = self.port.held_back();
        if held_back > 0 {
            self.with_inner(|inner| {
                let Some((key, pid, kind)) = inner
                    .unmet_head()
                    .map(|(key, waiter)| (key, waiter.pid, waiter.kind_name()))
                else {
                    return;
                };
                if inner.nothing_in_flight()
                    && inner.stall_dumped.is_none_or(|(at, _, _)| at != key)
                {
                    inner.stall_dumped = Some((key, 0, crate::rt::Instant::now()));
                    tracing::info!(
                        ?key,
                        %pid,
                        kind,
                        held_back,
                        "planner: the queue's head waits on the governor's target (room it holds back)"
                    );
                }
            });
            return;
        }
        let gives = self.port.room_gives();
        let stalled = self.with_inner(|inner| {
            let Some((key, waiter)) = inner.unmet_head() else {
                inner.stall_dumped = None;
                return None;
            };
            // A rung raise taking the room it was served is in flight.
            // So is a grant or a bind's room served and not yet taken:
            // its waiter takes it and fires (a head passed over for it,
            // [`Self::stuck_head`]).
            if !inner.nothing_in_flight()
                || !inner.parked_moving.is_empty()
                || inner.queue.values().any(Waiter::rise_untaken)
                || inner.queue.values().any(Waiter::untaken)
            {
                return None;
            }
            // A compaction on its way moves the head's room as a move does.
            if self.port.compaction_under_way() {
                return None;
            }
            // A frame held for a raise the walk has since taken (the cap up
            // to it): its fire goes on the scheduler's next turn, and lands.
            // Read as nothing in flight, the pass between the raise and that
            // fire was a stall (the simulation: a raise beside a restore
            // waiting on the raiser's room).
            let (_, engine) = self.port.locus();
            let owed = crate::store::shared::owed_cut_rows(engine);
            if owed > 0 && crate::store::shared::fire_cap(engine) >= owed {
                return None;
            }
            // Found again: counted each time, said once a head and room.
            self.stats.stuck_settles.fetch_add(1, Ordering::Relaxed);
            if let Some((at, seen, said)) = inner.stall_dumped
                && (at, seen) == (key, gives)
                && said.elapsed() < STALL_SAYS_AGAIN
            {
                return None;
            }
            let (pid, kind) = (waiter.pid, waiter.kind_name());
            let pids: Vec<ProcessId> = inner.procs.keys().copied().collect();
            // What its last unpark refusal said, and what was recorded as
            // its need: set beside the map, a refusal with fitting runs
            // standing names the part or shape the map does not count.
            let refusal = inner.procs.get(&waiter.pid).map(|proc| {
                format!(
                    "refused_on_this_pool={} roles_short={:?} bind_room={:?} why={:?} \
                     refusal_room=(run, free_runs)={:?} free_chunks={:?}",
                    proc.roles_short_gives == Some(gives),
                    proc.roles_short.map(|need| (need.run, need.pieces)),
                    proc.bind_room.map(|need| (need.run, need.pieces)),
                    proc.roles_short_why,
                    proc.roles_short_runs,
                    proc.roles_short_free,
                )
            });
            let refused_run = inner
                .procs
                .get(&waiter.pid)
                .and_then(|proc| proc.roles_short_runs.map(|(run, _)| run));
            // Its kv pages still to gather: an allocation's need, and a
            // restore's beside its runs.
            let kv_missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
            let again = inner.stall_dumped.is_some_and(|(at, seen, _)| (at, seen) == (key, gives));
            inner.stall_dumped = Some((key, gives, crate::rt::Instant::now()));
            // Said once a stall episode (the head stuck, whatever room moves
            // meanwhile), and again only as a stall that goes on: said on
            // each change of room, the M5 at beam w32 said it 0.5 M times
            // in 13 minutes as gathered pages came and went.
            let say = !inner
                .stall_said
                .is_some_and(|(at, said)| at == key && said.elapsed() < STALL_SAYS_AGAIN);
            if say {
                inner.stall_said = Some((key, crate::rt::Instant::now()));
            }
            let (plan, blocked) = inner.plan_fields(key);
            Some((key, pid, kind, pids, again, refusal, kv_missing, plan, blocked, refused_run, say))
        });
        let Some((key, pid, kind, pids, again, refusal, kv_missing, plan, blocked, refused_run, say)) =
            stalled
        else {
            return;
        };
        // What the head's step takes, read as the map and the step read it.
        let need = self.head_room(key);
        self.stats.stall_dumps.fetch_add(1, Ordering::Relaxed);
        let (free, total) = self.port.device_stats();
        // A head short of kv reads the map at runs of one chunk: what holds
        // the chunks it could take, parked state lying on the device among
        // it (a no-host pool's parked-in-place sets).
        let run = need.as_ref().and_then(RunNeed::longest).map_or(1, |(run, _)| run);
        // The candidates the eviction pass reads (nothing is in flight
        // here: every member), so the dump's map is the plan's.
        let pids: Vec<ProcessId> = self
            .victim_set()
            .map(|set| set.candidates(true).into_iter().map(|(pid, _)| pid).collect())
            .unwrap_or(pids);
        let mut map = self.occupants(&pids);
        map.merge(self.parked_map(pid));
        // The moves the whole need takes, every length laid together (the
        // map's own line reads one run of the longest length alone), a
        // restore's own takes placed first by the store's takes, as the
        // eviction plan reads it ([`Self::head_placement`]).
        let placement = need.as_ref().and_then(|need| self.head_placement(key, need));
        if let Some(placement) = &placement {
            placement.lay(&mut map);
        }
        let short = match (&placement, &need) {
            (Some(placement), _) => Some(placement.short.clone()),
            (None, need) => need.clone(),
        };
        let need_moves = short
            .as_ref()
            .map(|short| map.cheapest_need(short).map(|pids| pids.len()));
        // The map's figure for the room the refusal read, at its run: the
        // map's free runs once the head's own takes are laid, and its free
        // chunks, beside the refusal's own.
        let map_room = refused_run.map(|run| (run, map.free_runs(run), map.free.len()));
        let held_by = self.held_by_state();
        let store = self.store_holders();
        let (map, census) = (map.explain(run), self.port.chunk_census(run));
        // Every head queued fits the pool alone (one past it is refused at
        // its ask): unserved with nothing in flight, the planner found no
        // step for room that is there, which is its bug, never a wait.
        if say {
        tracing::error!(
            ?key,
            %pid,
            kind,
            need = ?need,
            short = ?short,
            own_opened = ?placement.as_ref().map(|placement| placement.opened.len()),
            bind_free_runs_after_own = ?placement.as_ref().and_then(|placement| placement.bind_free_runs),
            kv_missing,
            kv_chunks = kv_missing.div_ceil(self.port.chunks_as_pages(1).max(1)),
            again,
            refusal = %refusal.unwrap_or_default(),
            map_room_at_refused_run = ?map_room,
            ?need_moves,
            %plan,
            %blocked,
            capped = %self.capped_line(),
            passes = ?self.pass_census(),
            skipped_wakes = self.stats.skipped_wakes.load(Ordering::Relaxed),
            map_cache_hits = self.stats.map_cache_hits.load(Ordering::Relaxed),
            swap = ?self.swap_stats(),
            %held_by,
            %store,
            free_runs = ?need.as_ref().map(|need| {
                need.entries()
                    .iter()
                    .map(|&(run, _)| (run, self.port.guest_free_runs(run)))
                    .collect::<Vec<_>>()
            }),
            device = ?(free, total),
            queue = %self.debug_queue(),
            %map,
            pool = %census,
            "planner: the queue's head fits the pool alone and is unserved with nothing in \
             flight (a planner bug: no step found)"
        );
        }
        // A planner bug, never the head's: it waits on the next room, rung
        // or arrival event, never refused (only a need past the pool alone
        // is), and a debug build stops on it by name. The simulation's
        // progress invariant fails a run on it instead.
        assert!(
            !cfg!(all(debug_assertions, not(test))),
            "planner bug: the queue's head {pid} ({kind}) fits the pool alone and is unserved \
             with nothing in flight, no step found: {plan}"
        );
        // The progress invariant: a settled pass leaves every waiting head
        // served or something in flight that moves the generation. Broken,
        // tests fail by name; served, the next pass plans the head's need
        // as if its placement did not stand, and never waits.
        #[cfg(test)]
        if PROGRESS_STRICT.with(std::cell::Cell::get) {
            panic!(
                "progress invariant: head {key:?} ({kind}, pid {pid}) waits with nothing in \
                 flight; need {need:?}, short {short:?}, need_moves {need_moves:?}, placement \
                 {placement:?}, plan {plan}"
            );
        }
        // A victim whose move found the tier below full waits for the pool
        // to move, and with nothing in flight nothing moves it: it goes
        // again in the forced pass (the simulation's seed 431: the one
        // resident with pages to move, its move refused while the disk tier
        // stood full a step, kept out with nothing else moving and the
        // finisher's one page unserved). Lifted once a stall, as the stall
        // is said. One whose move found nothing to move finds nothing again
        // on the same pool: it stays out.
        self.with_inner(|inner| {
            inner.host_swap_blocked.clear();
            if inner.prepare_blocked.is_empty() {
                inner.blocked_at = None;
            }
        });
        self.force_moves.store(true, Ordering::Release);
        self.events.fetch_add(1, Ordering::AcqRel);
        if let Some(notify) = self.returns.drain.get() {
            notify.notify_one();
        }
    }

    /// The governor's target rose past what the pool maps: a commit can
    /// come where none could, room no event of the pool's own has told
    /// yet. Waiters plan again on it (the Mac's verified beam mix beside
    /// another suite: the target swung 693 to 779 MiB for 25 minutes,
    /// processes parked as "plan: resident again" and none planned).
    pub fn target_raised(self: &Arc<Self>) {
        if self.returns.waiters.load(Ordering::Acquire) != 0 {
            ptrace!("target raised: a commit can come");
            self.poke();
        }
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

    /// One pass over the queue's heads; true where it ends waiting on a
    /// grow. Heads passed over this pass ([`Self::stuck_head`]) stay passed
    /// until [`Self::plan`] clears them.
    pub(super) fn plan_once(self: &Arc<Self>) -> bool {
        self.plan_heads()
    }

    /// The queue's head left stuck as a pass settles: no move named for it,
    /// nothing in flight, no parked state or idle cache moving down, no
    /// tails being swept, no grant or raise served and not yet taken, no
    /// compaction on its way. [`Self::plan`] passes such a head over for
    /// the rest of the pass and plans the next waiter, so one behind it
    /// that can go does (a decoder's page served from pages free, a restore
    /// whose room below the head's moves need); each heads the next pass
    /// again, planned first on whatever came back. Held at the stuck head,
    /// nothing moved and nothing came that could move it (seeds 134, 624: a
    /// bind short of chunks only a full disk could empty, decoders behind
    /// it that free pages would serve, nothing in flight). Read once a pass
    /// where the stall dump read its head before: a firing fleet's pass
    /// pays nothing more.
    fn stuck_head(&self) -> Option<(EntryKey, ProcessId)> {
        let stuck = self.with_inner(|inner| {
            let Some((key, waiter)) = inner.unmet_head() else {
                // No head at all (none passed over): none to say again.
                if inner.passed.is_empty() {
                    inner.stall_dumped = None;
                }
                return None;
            };
            let pid = waiter.pid;
            // The capacity gate's finisher is never passed over: the heads
            // behind it (others' restores) took the room its moves made,
            // kept something in flight, and two of them went round, each
            // restored and evicted for the other 125 times while its bind
            // waited (the field mix, seed 178).
            if inner.is_finisher(pid) {
                return None;
            }
            (inner.nothing_in_flight()
                && inner.parked_moving.is_empty()
                && !inner.queue.values().any(Waiter::rise_untaken)
                && !inner.queue.values().any(Waiter::untaken))
            .then_some((key, pid))
        })?;
        (!self.demoting.load(Ordering::Acquire)
            && !self.sweeping.load(Ordering::Acquire)
            && !self.port.compaction_under_way())
        .then_some(stuck)
    }

    fn plan_heads(self: &Arc<Self>) -> bool {
        if self.pause_due.swap(false, Ordering::AcqRel) {
            self.pause_held();
        }
        self.reseat_released();
        // Taken once a pass: every grow step of this pass goes on past it.
        let no_grow = self.no_grow.swap(false, Ordering::AcqRel);
        // The head restore whose pending bind's room was seen free this
        // pass: what it asks past its kv stands.
        let mut room_stood: Option<EntryKey> = None;
        loop {
            // The capacity gate as this step reads it: a waiter it holds is
            // no head ([`super::inner::CapacityGate`]).
            let order = self.capacity_order();
            self.with_inner(|inner| inner.set_capacity(order));
            // Each step reads the head's need afresh (the cache spans one
            // step's reads: its own, the map's, the eviction's).
            *self.need_cache.lock() = None;
            let (rs_free, rs_total) = self.port.rs_stats();
            let rs_growth = self.rs_growth();
            // What the device has free past the pages the head gathered:
            // room a restore may take.
            let step = self.with_inner(|inner| {
                // A rung raise served and not yet taken holds the queue:
                // the room it was funded is its, and its leaving plans
                // again.
                if inner.queue.values().any(Waiter::rise_untaken) {
                    return Step::Done("a rung raise served and not yet taken holds the queue");
                }
                // A list compaction moves pages of the processes it names,
                // running and parked: nothing is served or moved until it
                // lands (a restore of one, a move of its parked state), and
                // its landing plans again.
                if inner.relocating.is_some() {
                    return Step::Done("a list compaction is on its way");
                }
                let Some((key, waiter)) = inner.unmet_head() else {
                    if inner.accum.len() > 0 && inner.queue.is_empty() {
                        let stranded = std::mem::take(&mut inner.accum);
                        return Step::Release(stranded);
                    }
                    return Step::Done("no unmet head");
                };
                if let WaitKind::BindRoom {
                    run,
                    pieces,
                    singles,
                    rise,
                    ..
                } = waiter.kind
                {
                    return Step::BindRoom {
                        key,
                        run,
                        pieces,
                        singles,
                        rise,
                    };
                }
                let is_restore = matches!(waiter.kind, WaitKind::Restore { .. });
                // Pages gathered past what a restore head needs were gathered
                // for another head (one a restore went ahead of): held while
                // this one waits, they spent the state budget its runs need
                // (the L40S: 8 of 9 opened chunks held by no candidate and
                // nothing parked, nothing in flight). They go back. An
                // allocation's surplus stays: a burst serves several from it.
                // Not while a head this pass passed over holds them: given
                // back, they were gathered again for it on the next pass,
                // and given back again, a pass that never settled (seed
                // 3392).
                if is_restore
                    && inner.passed.is_empty()
                    && inner.accum.len() as u32 > waiter.kv_need()
                {
                    let spare = inner.accum.len() as u32 - waiter.kv_need();
                    return Step::Release(inner.accum.donate(spare));
                }
                let pid = waiter.pid;
                // Its parked state is on its way down a tier: restored now,
                // it would miss the items moving. The landing pokes.
                if is_restore && inner.parked_moving.contains(&pid) {
                    return Step::Done("its parked state moves down a tier: restored once it lands");
                }
                let demand = match &waiter.kind {
                    WaitKind::Allocation { demand, .. } => Some(*demand),
                    WaitKind::Restore { .. } | WaitKind::BindRoom { .. } => None,
                };
                let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                if missing > 0 {
                    return Step::Absorb {
                        count: missing,
                        fund_by_eviction: !is_restore
                            || inner.nothing_in_flight()
                            || inner.is_finisher(pid),
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
                        fund_by_eviction: !is_restore
                            || inner.nothing_in_flight()
                            || inner.is_finisher(pid),
                        surplus: (inner.accum.len() as u32).saturating_sub(waiter.kv_need()),
                    };
                }
                match demand {
                    Some(demand) if !demand.rs.is_zero() => Step::ServeAllocation { key, demand },
                    Some(_) => Step::ServeAllocationBurst {
                        extra: inner.burst_shortfall(),
                    },
                    None => match inner.procs.get(&pid).and_then(|proc| proc.bind_room) {
                        // Another restored for its bind and not yet fired
                        // ([`Inner::bind_restoring`]): its bind goes first,
                        // served or funded as a bind is, and with none
                        // waiting the planner waits for it to fire. Funding
                        // this restore instead moved every resident for a
                        // restore the planner would not serve, and the
                        // restored one's bind waited behind it for good.
                        Some(need)
                            if inner
                                .bind_restoring
                                .is_some_and(|restoring| restoring != pid)
                                && !inner.bind_restoring_released() =>
                        {
                            let restoring = inner.bind_restoring;
                            inner
                                .queue
                                .iter()
                                .find_map(|(key, waiter)| match waiter.kind {
                                    WaitKind::BindRoom {
                                        run,
                                        pieces,
                                        singles,
                                        served: false,
                                        withdrawn: false,
                                        ..
                                    } if Some(waiter.pid) == restoring => Some(Step::BindRoom {
                                        key: *key,
                                        run,
                                        pieces,
                                        singles,
                                        rise: None,
                                    }),
                                    _ => None,
                                })
                                .unwrap_or_else(|| {
                                    // Its bind served and not yet taken: the
                                    // hand-off is under way, no wait on a
                                    // fire. Withdrawn (evicted again): it
                                    // holds this head no longer.
                                    let bind = inner.queue.values().find_map(|waiter| match waiter.kind {
                                        WaitKind::BindRoom { served, withdrawn, .. }
                                            if Some(waiter.pid) == restoring && waiter.is_unmet() =>
                                        {
                                            Some((served, withdrawn))
                                        }
                                        _ => None,
                                    });
                                    match bind {
                                        Some((true, _)) => {
                                            Step::Done("the restored one's bind is served and not yet taken")
                                        }
                                        Some((_, true)) => Step::RestoreBindRoom { key, need },
                                        // Nothing in flight to fire it: its
                                        // room is planned, never waited on.
                                        _ if inner.nothing_in_flight() => {
                                            Step::RestoreBindRoom { key, need }
                                        }
                                        _ => Step::Done("waits for the restored one's bind to fire"),
                                    }
                                })
                        }
                        // Its whole need: the room its bind asks too.
                        Some(need) if room_stood != Some(key) => {
                            Step::RestoreBindRoom { key, need }
                        }
                        _ => match inner.restore_gate(pid) {
                            RestoreGate::Serve => Step::ServeRestore { key, pid },
                            RestoreGate::Evict { need, fund } => {
                                Step::EvictForRoles { need, fund }
                            }
                        },
                    },
                }
            });
            let fund_by_eviction = match step {
                // Every pass that stops says where, for the stall line
                // (Mac Pro at 2 GiB on 3e0f0f18f: need_moves 1, ended=none,
                // 4,435 no-step lines).
                Step::Done(ended) => {
                    self.plan_stopped(ended);
                    return false;
                }
                Step::BindRoom {
                    key,
                    run,
                    pieces,
                    singles,
                    rise,
                } => {
                    // A rung raise asks what its raise lacks now, as a run
                    // of one chunk a piece: the chunks free now and those
                    // to empty, funded from here exactly as a bind's runs
                    // (idle cache, a grow, parked state down a tier, then
                    // the chunk map's occupants under the key rule). One
                    // that fits, or never will, is served: its waiter
                    // takes the raise or refuses it by name.
                    let (run, pieces) = match rise {
                        None => (run, pieces),
                        Some(rung) => match self
                            .with_inner(|inner| inner.queue.get(&key).map(|waiter| waiter.pid))
                            .map_or(super::pool::RiseRoom::Fits, |pid| self.port.rise_room(pid, rung))
                        {
                            super::pool::RiseRoom::Short { pieces } => {
                                let pages = self.port.chunks_as_pages(pieces);
                                self.with_inner(|inner| {
                                    if let Some(Waiter {
                                        kind:
                                            WaitKind::BindRoom {
                                                pieces: asked,
                                                pages: room,
                                                ..
                                            },
                                        ..
                                    }) = inner.queue.get_mut(&key)
                                    {
                                        *asked = pieces;
                                        *room = pages;
                                    }
                                });
                                (1, pieces)
                            }
                            super::pool::RiseRoom::Fits | super::pool::RiseRoom::Past { .. } => {
                                self.serve_bind_room(key);
                                continue;
                            }
                        },
                    };
                    // Its runs stand free, and its single chunks beside
                    // them: the bind takes them, unless the capacity gate
                    // holds its fresh state. Served there, its take was
                    // held, it asked again and was served at once, a spin
                    // in one step: it waits for the gate, which moves as
                    // those ahead of it finish.
                    if rise.is_none()
                        && self.port.guest_room_free(run, pieces)
                        && (singles == 0
                            || self
                                .port
                                .guest_room_free(1, run.saturating_mul(pieces).saturating_add(singles)))
                    {
                        let pid = self.with_inner(|inner| inner.queue.get(&key).map(|waiter| waiter.pid));
                        if pid.is_some_and(|pid| {
                            self.capacity_would_hold(
                                pid,
                                run.saturating_mul(pieces).saturating_add(singles),
                            )
                        }) {
                            self.plan_stopped("bind room: its runs stand, held at the capacity gate");
                            return false;
                        }
                        self.serve_bind_room(key);
                        continue;
                    }
                    // Idle cache first, one set or a sweep, and it looks
                    // again in what that left; then residents are moved for
                    // it by the planner's victim choice, as for an
                    // allocation: a bind is new demand, not a restore, and
                    // it is funded whether or not the fleet has stalled (a
                    // fleet that keeps decoding never stalls, and the bind
                    // waited behind it for good).
                    if self.reclaim_idle_once() || self.sweep_stranded(u32::MAX, false) {
                        continue;
                    }
                    // A grow the walk will make holds its runs: it waits on
                    // that, moving no one. Only where none is coming (the
                    // governor's target reached) are residents moved.
                    let need = Demand {
                        kv_pages: self
                            .port
                            .chunks_as_pages(run.saturating_mul(pieces).saturating_add(singles)),
                        rs: RsUnits::default(),
                    };
                    if self.wait_on_grow(no_grow, need) {
                        self.plan_stopped("bind room: waits on a grow the walk makes");
                        return true;
                    }
                    // Parked state lying on the device moves down a tier
                    // before any resident does.
                    if self.park_down_for(&RunNeed::of([(run, pieces), (1, singles)])) {
                        self.plan_stopped("bind room: parked state moves down");
                        return false;
                    }
                    true
                }
                Step::RestoreBindRoom { key, need } => {
                    // As the bind it suspended for waits: served once the
                    // runs stand free past its gathered kv, else idle cache
                    // first, then, the fleet stalled, a resident parked
                    // for it by the planner's victim choice.
                    // Its restore retakes its own parked guest state first:
                    // the runs must stand beside that
                    // ([`super::pool::PoolPort::room_beside_own`]), read
                    // from the pool as it is, never from room given back
                    // since its park (nothing gives any once the fleet
                    // waits on this head: Mac Pro's beam mix, flat after a
                    // bind-suspend with 861 chunks free). And one such
                    // restore at a time ([`Inner::bind_restoring`]).
                    let (pid, other_restoring) = self.with_inner(|inner| {
                        let pid = inner.queue.get(&key).map(|waiter| waiter.pid);
                        (
                            pid,
                            inner.bind_restoring.is_some_and(|restoring| Some(restoring) != pid)
                                && !inner.bind_restoring_released(),
                        )
                    });
                    debug_assert!(
                        !other_restoring,
                        "selected only with no other restoring that can still fire"
                    );
                    // The one placement its restore is sent on: its serve's
                    // state, every piece of its unpark, then the bind's runs
                    // ([`super::pool::PoolPort::restore_stands`]). Read
                    // beside its dead parts alone, the bind stood here while
                    // the restore's own check, after its unpark took its live
                    // parts and its runs of two, did not: sent, refused
                    // (PendingBindShort), sent again (Mac Pro at 2 GiB on
                    // 7e75af541: 52k no-step lines, free runs (7, 3)).
                    let whole = self.head_room(key).unwrap_or_else(|| need.runs());
                    if pid.is_some_and(|pid| {
                        !self.no_more_room_than_refused(pid)
                            && (self
                                .head_placement(key, &whole)
                                .is_some_and(|placement| placement.stands())
                                || self.take_send_regardless(pid))
                    }) {
                        self.plan_stopped("bind-suspended restore sent: its placement stands");
                        room_stood = Some(key);
                        continue;
                    }
                    if self.reclaim_idle_once() || self.sweep_stranded(u32::MAX, false) {
                        continue;
                    }
                    // Parked state lying on the device moves down a tier for
                    // it, no one running on it (the M1 stress: every
                    // Instance chunk held by parked processes' role slots,
                    // the bind-suspended head's runs never funded): the
                    // bind's runs beside its own retake ([`Self::beside_own`]).
                    let room = match pid {
                        Some(pid) => {
                            let mut room = self.beside_own(pid, (need.run, need.pieces));
                            room.add(1, need.singles);
                            room
                        }
                        None => need.runs(),
                    }
                    .within(self.port.state_chunks(), Some((need.run, need.pieces)));
                    if self.park_down_for(&room) {
                        self.plan_stopped("bind-suspended restore: parked state moves down");
                        return false;
                    }
                    // Funded once nothing in flight can free room
                    // ([`Inner::nothing_in_flight`]).
                    if !self.with_inner(|inner| inner.nothing_in_flight()) {
                        self.plan_stopped("bind-suspended restore: something in flight, not funded");
                        return false;
                    }
                    true
                }
                Step::EvictForRoles {
                    need: (run, pieces),
                    fund,
                } => {
                    // What the restore's step takes, as the map and the
                    // census read it ([`Self::head_room`]).
                    // Every run length its unpark lays, each counted apart;
                    // its refusal's own length is `run` (the gate's need).
                    let room = self
                        .with_inner(|inner| inner.unmet_head().map(|(key, _)| key))
                        .and_then(|key| self.head_room(key))
                        .unwrap_or_else(|| RunNeed::one(run, pieces));
                    // Its runs stand free (a completion or a commit gave
                    // them back): it goes again, once a need. Short again
                    // with the same need, the need understates its unpark
                    // (Mac Pro's M5 livelock: free runs of it stood, every
                    // unpark came back short, 20M times, and nothing was
                    // ever reclaimed or moved); then room is made for it.
                    // Understated (refused while runs of its need stood), it
                    // goes again only once the pool holds more of them than
                    // its refusal saw: never twice on the same runs (the
                    // livelock), and on a change a quiescent pool shows (no
                    // completion needed: Mac Pro's disk-only home, 64 free
                    // runs, nothing firing, the head held for good).
                    let free_runs = self.port.guest_free_runs(run);
                    let understated = self.with_inner(|inner| {
                        inner.unmet_head().is_some_and(|(_, waiter)| {
                            inner.procs.get(&waiter.pid).is_some_and(|proc| {
                                proc.roles_free_short.is_some_and(|(short, _)| short == run)
                                    && free_runs <= proc.roles_free_short_at
                            })
                        })
                    });
                    // A plain short head goes again on runs that stand only
                    // once more runs of its need's length stand than its
                    // refusal saw: never on the same room twice. Room a
                    // refused restore took and gave back is no more room.
                    // Read at the length its refusal named, in the room
                    // the refusal read.
                    // And only once room was given back since it was
                    // refused ([`Self::no_more_room_than_refused`]):
                    // runs of four fluctuating with a decoding fleet's
                    // takes sent it again ~350 times a second, each unpark
                    // short of a chunk (Mac Pro on 3b45462d4).
                    // One placement decides: the store's own, its serve's
                    // state laid first ([`super::pool::PoolPort::guest_need_free`]). It
                    // goes again where that stands now, unless it stood as
                    // the refusal left the pool too (it read a room the
                    // unpark did not): then only once more chunks stand
                    // free than the refusal left
                    // ([`super::proc::Proc::roles_short_stood`]).
                    let (head_key, stood_at, free_left) = self.with_inner(|inner| {
                        inner
                            .unmet_head()
                            .map(|(key, waiter)| {
                                let proc = inner.procs.get(&waiter.pid);
                                (
                                    Some(key),
                                    proc.and_then(|proc| proc.roles_short_stood),
                                    proc.and_then(|proc| proc.roles_short_free),
                                )
                            })
                            .unwrap_or_default()
                    });
                    // Never on the room its last refusal was taken in.
                    let _ = stood_at;
                    let _ = free_left;
                    let moved = self
                        .with_inner(|inner| inner.unmet_head().map(|(_, waiter)| waiter.pid))
                        .is_none_or(|pid| !self.no_more_room_than_refused(pid));
                    if !understated
                        && moved
                        && head_key.is_some_and(|key| {
                            self.head_placement(key, &room)
                                .is_some_and(|placement| placement.stands())
                                || self
                                    .with_inner(|inner| inner.queue.get(&key).map(|waiter| waiter.pid))
                                    .is_some_and(|pid| self.take_send_regardless(pid))
                        })
                    {
                        self.plan_stopped("restore sent again: its placement stands");
                        self.with_inner(|inner| {
                            let pid = inner.unmet_head().map(|(_, waiter)| waiter.pid);
                            if let Some(proc) = pid.and_then(|pid| inner.procs.get_mut(&pid)) {
                                proc.roles_free_retry = Some((run, pieces));
                            }
                            inner.roles_retry_head();
                        });
                        continue;
                    }
                    // Not funded yet: it waits for a completion (one that
                    // frees room pokes the plan, which looks again here).
                    if !fund {
                        self.plan_stopped("restore short of runs: something in flight, not funded");
                        return false;
                    }
                    // Idle cache holds chunks too: one set goes, and the
                    // restore tries again in what it left, before any
                    // resident moves for it.
                    if self.reclaim_idle_once() || self.sweep_stranded(u32::MAX, false) {
                        self.with_inner(|inner| inner.roles_retry_head());
                        continue;
                    }
                    // A grow the walk will make holds its runs: it waits on
                    // that, moving no one.
                    let grow = Demand {
                        kv_pages: self
                            .port
                            .chunks_as_pages(u32::try_from(room.chunks()).unwrap_or(u32::MAX)),
                        rs: RsUnits::default(),
                    };
                    if self.wait_on_grow(no_grow, grow) {
                        self.plan_stopped("restore short of runs: waits on a grow the walk makes");
                        return true;
                    }
                    // Parked state lying on the device (role slots their
                    // park found no host room for) moves down a tier before
                    // any resident does: no one runs on it (Mac Pro's M5
                    // stall: host full, every process evicted, the head's
                    // runs held by parked role slots, nothing to evict).
                    if self.park_down_for(&room) {
                        self.plan_stopped("restore short of runs: parked state moves down");
                        return false;
                    }
                    // Else the occupants of the chunks it could take move
                    // for it (`plan_eviction`'s chunk branch), under the
                    // key rule.
                    true
                }
                Step::Release(stranded) => {
                    // Back to the pool; the head is planned on what came back.
                    drop(stranded);
                    continue;
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
                        self.plan_stopped("short of kv: waits on a grow the walk makes");
                        return true;
                    }
                    // Idle cache a policy keys above the floor moves down
                    // a tier before any is dropped; the rest is dropped.
                    if self.demote_idle(count) {
                        self.plan_stopped("short of kv: idle cache moves down");
                        return false;
                    }
                    if self.reclaim_idle_once() {
                        continue;
                    }
                    if self.sweep_stranded(count, false) {
                        self.plan_stopped("short of kv: stranded tails swept");
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
                    if surplus > 0
                        && self.port.rs_as_pages(need) > 0
                        && self.with_inner(|inner| inner.passed.is_empty())
                    {
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
                        self.plan_stopped("short of state: waits on a grow the walk makes");
                        return true;
                    }
                    // On a shared pool the chunks idle cache holds open for
                    // state too: what is keyed above the floor moves down a
                    // tier before any is dropped.
                    let as_pages = self.port.rs_as_pages(need);
                    if as_pages > 0 && self.demote_idle(as_pages) {
                        self.plan_stopped("short of state: idle cache moves down");
                        return false;
                    }
                    if self.reclaim_idle_once() || self.reclaim_head_sharing() {
                        continue;
                    }
                    // And the tails of sets at rest past what their keys
                    // keep, lowest key first, before any set moves whole
                    // (its state with it).
                    if as_pages > 0 && self.sweep_stranded(as_pages, true) {
                        self.plan_stopped("short of state: stranded tails swept");
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
                            self.plan_stopped("allocation: waits on a grow for its state");
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
                            self.plan_stopped("restore: waits on a grow for its state");
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
                        // Suspended for a bind: the one restored at a time.
                        if inner.procs.get(&pid).is_some_and(|proc| proc.bind_room.is_some()) {
                            inner.bind_restoring = Some(pid);
                        }
                        let Some(proc) = inner.procs.get_mut(&pid) else {
                            inner.queue.remove(&key);
                            return Board::Replan;
                        };
                        if proc.state != Residency::Evicted {
                            return Board::Replan;
                        }
                        // Held below the policy's floor: never restored, on
                        // any path, forced or not, until the policy lifts it
                        // (its restore is queued again then).
                        if proc.held {
                            inner.queue.remove(&key);
                            return Board::Replan;
                        }
                        // Restored past the head it was evicted for, that
                        // head passed over: no victim for it until that head
                        // arrives ([`Inner::arrived`]). Moved for it again
                        // once it fired, it went round with a head no move
                        // formed runs for, restored past it and moved back
                        // (the simulation's seeds 189, 276).
                        proc.restored_past = proc.evicted_for.filter(|head| {
                            inner.passed.iter().any(|&(_, passed)| passed == *head)
                        });
                        let proc = inner.procs.get_mut(&pid).expect("read above");
                        proc.state = Residency::Restoring;
                        if let Some(waiter) = inner.queue.remove(&key) {
                            let waited = waiter.since.elapsed();
                            if waited > super::wait::LONG_WAIT {
                                inner.long_waits.push(super::wait::LongWait {
                                    kind: "restore",
                                    lacked: waiter.lacked,
                                    wanted: waiter.wanted,
                                    waited,
                                });
                            }
                        }
                        Board::Go(inner.accum.donate(need.kv_pages))
                    });
                    match boarded {
                        Board::Go(pages) => {
                            // What its placement chose (its unpark's pieces,
                            // then its pending bind's runs) is held for it
                            // until its unpark and bind land: no head served
                            // meanwhile takes it (the L40S at 3 GiB on
                            // c201812ea: PendingBindShort, its run-1 bind's
                            // chunk taken between its serve and its unpark).
                            let bind = self.with_inner(|inner| {
                                inner.procs.get(&pid).and_then(|proc| proc.bind_room).map(|need| need.entries())
                            });
                            let held = self.port.reserve_restore(pid, bind.as_deref().unwrap_or_default());
                            // What held it for the head it was moved for, as
                            // that head stood when it went (the hold's terms,
                            // [`super::inner::Inner::unmet_head`]).
                            let held_for = tracing::enabled!(tracing::Level::DEBUG)
                                .then(|| {
                                    self.with_inner(|inner| {
                                        let head = inner.procs.get(&pid)?.evicted_for?;
                                        let queued = inner.queue.values().any(|waiter| waiter.pid == head);
                                        let state = inner.procs.get(&head).map(|proc| {
                                            format!(
                                                "{:?} unfired={} idle={} bind_waiting={} in_request={}",
                                                proc.state,
                                                proc.unfired,
                                                proc.idle,
                                                proc.bind_waiting,
                                                super::inner::Inner::in_request(proc)
                                            )
                                        });
                                        Some(format!(
                                            "head {head} queued={queued} passed={} finisher={} {state:?}",
                                            inner.passed.iter().any(|&(_, passed)| passed == head),
                                            inner.is_finisher(pid)
                                        ))
                                    })
                                })
                                .flatten();
                            tracing::debug!(%pid, ?bind, held, ?held_for, "planner: a restore sent, its placement held for it");
                            // Said as the pass's end for this head: a restore
                            // refused and queued again showed "ended=none".
                            self.plan_ended(key, "restore sent", false, &[]);
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
                self.plan_stopped("restore short of kv: something in flight, not funded");
                self.stats
                    .restore_absorb_short
                    .fetch_add(1, Ordering::Relaxed);
            }
            return false;
        }
    }
}
