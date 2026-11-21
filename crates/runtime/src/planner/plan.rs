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
        rise: Option<u32>,
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

/// While a head stays unserved with nothing in flight, its stall dump is
/// said again as often as a process waiting at admission says why
/// ([`super::residency_planner::ADMIT_SAYS_AFTER`]), the drain task looking
/// with no plan pass: a stall that goes on is never silent, and no timer
/// ends it.
pub(super) const STALL_SAYS_AGAIN: std::time::Duration =
    super::residency_planner::ADMIT_SAYS_AFTER;

impl ResidencyPlanner {
    pub fn plan(self: &Arc<Self>) {
        // A grant dropped unspent during the pass gave back room the pass
        // may have read before it came: the pass runs again on the pool as
        // it stands ([`super::grant::RoomReturns`]), so the drop, which
        // cannot plan itself, wakes the head within this step.
        self.stats.plan_passes.fetch_add(1, Ordering::Relaxed);
        let cause = self
            .returns
            .wake_cause
            .swap(super::grant::WAKE_CALL, Ordering::AcqRel);
        let on_grow = loop {
            self.returns.planning.store(true, Ordering::Release);
            self.returns.plan_begins();
            self.swap_bound.store(false, Ordering::Release);
            let on_grow = loop {
                let seen = self.returns.epoch();
                let on_grow = self.plan_once();
                if self.returns.epoch() == seen {
                    break on_grow;
                }
            };
            self.returns.planning.store(false, Ordering::Release);
            self.publish_wants();
            // Room that came free while the pass ran (its signal woke no one
            // then) and meets what the head lacks now: planned on at once.
            if self.returns.waiters.load(Ordering::Acquire) == 0
                || !(self.returns.freed_since_plan() && self.wants_met_now())
            {
                break on_grow;
            }
        };
        // The waiters parked behind a grow the walk was promised, now.
        let parked = if on_grow {
            self.with_inner(|inner| inner.queue.values().filter(|w| w.is_unmet()).count())
        } else {
            0
        };
        self.stats
            .grow_parked
            .store(parked as u64, Ordering::Relaxed);
        if !on_grow {
            self.dump_stall();
        }
        // A waiter served or gone lacks nothing now.
        self.report_short();
        self.say_long_waits(cause);
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
        let mut map = self.port.chunk_occupants(&all);
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
        for (kind, waited) in waits {
            if let Some(count) = self.stats.long_waits.get(usize::from(cause)) {
                count.fetch_add(1, Ordering::Relaxed);
            }
            tracing::info!(
                kind,
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
        let gives = self.port.room_gives();
        let stalled = self.with_inner(|inner| {
            let Some((key, waiter)) = inner.unmet_head() else {
                inner.stall_dumped = None;
                return None;
            };
            // A rung raise taking the room it was served is in flight.
            if !inner.nothing_in_flight()
                || !inner.parked_moving.is_empty()
                || inner.queue.values().any(Waiter::rise_untaken)
            {
                return None;
            }
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
                    "refused_on_this_pool={} roles_short={:?} bind_room={:?} why={:?}",
                    proc.roles_short_gives == Some(gives),
                    proc.roles_short.map(|need| (need.run, need.pieces)),
                    proc.bind_room.map(|need| (need.run, need.pieces)),
                    proc.roles_short_why,
                )
            });
            // Its kv pages still to gather: an allocation's need, and a
            // restore's beside its runs.
            let kv_missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
            let again = inner.stall_dumped.is_some_and(|(at, seen, _)| (at, seen) == (key, gives));
            inner.stall_dumped = Some((key, gives, crate::rt::Instant::now()));
            Some((key, pid, kind, pids, again, refusal, kv_missing))
        });
        let Some((key, pid, kind, pids, again, refusal, kv_missing)) = stalled else {
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
        let mut map = self.port.chunk_occupants(&pids);
        map.merge(self.parked_map(pid));
        // The moves the whole need takes, every length laid together (the
        // map's own line reads one run of the longest length alone).
        let need_moves = need
            .as_ref()
            .map(|need| map.cheapest_need(need).map(|pids| pids.len()));
        let held_by = self.held_by_state();
        let (map, census) = (map.explain(run), self.port.chunk_census(run));
        // Every head queued fits the pool alone (one past it is refused at
        // its ask): unserved with nothing in flight, the planner found no
        // step for room that is there, which is its bug, never a wait.
        tracing::error!(
            ?key,
            %pid,
            kind,
            need = ?need,
            kv_missing,
            kv_chunks = kv_missing.div_ceil(self.port.chunks_as_pages(1).max(1)),
            again,
            refusal = %refusal.unwrap_or_default(),
            ?need_moves,
            %held_by,
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
    pub(super) fn plan_once(self: &Arc<Self>) -> bool {
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
                    return Step::Done;
                }
                let Some((key, waiter)) = inner.unmet_head() else {
                    if inner.accum.len() > 0 && inner.queue.is_empty() {
                        let stranded = std::mem::take(&mut inner.accum);
                        return Step::Release(stranded);
                    }
                    return Step::Done;
                };
                if let WaitKind::BindRoom {
                    run, pieces, rise, ..
                } = waiter.kind
                {
                    return Step::BindRoom {
                        key,
                        run,
                        pieces,
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
                if is_restore && inner.accum.len() as u32 > waiter.kv_need() {
                    let spare = inner.accum.len() as u32 - waiter.kv_need();
                    return Step::Release(inner.accum.donate(spare));
                }
                let pid = waiter.pid;
                // Its parked state is on its way down a tier: restored now,
                // it would miss the items moving. The landing pokes.
                if is_restore && inner.parked_moving.contains(&pid) {
                    return Step::Done;
                }
                let demand = match &waiter.kind {
                    WaitKind::Allocation { demand, .. } => Some(*demand),
                    WaitKind::Restore { .. } | WaitKind::BindRoom { .. } => None,
                };
                let missing = waiter.kv_need().saturating_sub(inner.accum.len() as u32);
                if missing > 0 {
                    return Step::Absorb {
                        count: missing,
                        fund_by_eviction: !is_restore || inner.nothing_in_flight(),
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
                        fund_by_eviction: !is_restore || inner.nothing_in_flight(),
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
                        Some(_)
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
                                        served: false,
                                        withdrawn: false,
                                        ..
                                    } if Some(waiter.pid) == restoring => Some(Step::BindRoom {
                                        key: *key,
                                        run,
                                        pieces,
                                        rise: None,
                                    }),
                                    _ => None,
                                })
                                .unwrap_or(Step::Done)
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
                Step::Done => return false,
                Step::BindRoom {
                    key,
                    run,
                    pieces,
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
                        Some(rung) => match self.port.rise_room(rung) {
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
                    // Its runs stand free: the bind takes them.
                    if rise.is_none() && self.port.guest_room_free(run, pieces) {
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
                        kv_pages: self.port.chunks_as_pages(run.saturating_mul(pieces)),
                        rs: RsUnits::default(),
                    };
                    if self.wait_on_grow(no_grow, need) {
                        return true;
                    }
                    // Parked state lying on the device moves down a tier
                    // before any resident does.
                    if self.park_down_for(&RunNeed::one(run, pieces)) {
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
                    if pid.is_some_and(|pid| self.port.room_beside_own(pid, need.run, need.pieces)) {
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
                        Some(pid) => self.beside_own(pid, (need.run, need.pieces)),
                        None => RunNeed::one(need.run, need.pieces),
                    }
                    .within(self.port.state_chunks(), Some((need.run, need.pieces)));
                    if self.park_down_for(&room) {
                        return false;
                    }
                    // Funded once nothing in flight can free room
                    // ([`Inner::nothing_in_flight`]).
                    if !self.with_inner(|inner| inner.nothing_in_flight()) {
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
                    // And only once more chunks stand free than it left:
                    // runs of four fluctuating with a decoding fleet's
                    // takes sent it again ~350 times a second, each unpark
                    // short of a chunk (Mac Pro on 3b45462d4).
                    let refused = self.with_inner(|inner| {
                        inner.unmet_head().and_then(|(_, waiter)| {
                            inner
                                .procs
                                .get(&waiter.pid)
                                .map(|proc| (proc.roles_short_runs, proc.roles_short_free))
                        })
                    });
                    let map_moved = match refused {
                        Some((Some((at_run, at)), free)) => {
                            self.port.guest_free_runs(at_run) > at
                                && free.is_none_or(|free| self.port.free_chunks() > free)
                        }
                        Some((None, _)) => true,
                        None => false,
                    };
                    if !understated && map_moved && self.port.guest_need_free(&room) {
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
                        return true;
                    }
                    // Parked state lying on the device (role slots their
                    // park found no host room for) moves down a tier before
                    // any resident does: no one runs on it (Mac Pro's M5
                    // stall: host full, every process evicted, the head's
                    // runs held by parked role slots, nothing to evict).
                    if self.park_down_for(&room) {
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
                        return true;
                    }
                    // Idle cache a policy keys above the floor moves down
                    // a tier before any is dropped; the rest is dropped.
                    if self.demote_idle(count) {
                        return false;
                    }
                    if self.reclaim_idle_once() {
                        continue;
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
                    if as_pages > 0 && self.demote_idle(as_pages) {
                        return false;
                    }
                    if self.reclaim_idle_once() || self.reclaim_head_sharing() {
                        continue;
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
                        proc.state = Residency::Restoring;
                        if let Some(waiter) = inner.queue.remove(&key) {
                            let waited = waiter.since.elapsed();
                            if waited > super::wait::LONG_WAIT {
                                inner.long_waits.push(("restore", waited));
                            }
                        }
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
