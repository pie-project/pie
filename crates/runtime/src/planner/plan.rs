use std::sync::Arc;
use std::sync::atomic::Ordering;

use tokio::sync::Notify;

use super::grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use super::holdings::{swapped_page_count, swapped_rs_units};
use super::inner::RestoreGate;
use super::proc::{PendingBind, Residency};
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
    /// The head is a bind short of device room for its guest state.
    BindRoom {
        key: EntryKey,
        run: u32,
        pieces: u32,
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
        if !on_grow {
            self.dump_stall();
        }
        // A waiter served or gone lacks nothing now.
        self.report_short();
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
            if !inner.nothing_in_flight() || !inner.parked_moving.is_empty() {
                return None;
            }
            if inner.stall_dumped == Some((key, gives)) {
                return None;
            }
            let need = match &waiter.kind {
                WaitKind::BindRoom { run, pieces, .. } => Some((*run, *pieces)),
                WaitKind::Restore { .. } => inner
                    .procs
                    .get(&waiter.pid)
                    .and_then(super::proc::Proc::restore_need),
                WaitKind::Allocation { .. } => None,
            };
            let (pid, kind) = (waiter.pid, waiter.kind_name());
            let pids: Vec<ProcessId> = inner.procs.keys().copied().collect();
            inner.stall_dumped = Some((key, gives));
            Some((key, pid, kind, need, pids))
        });
        let Some((key, pid, kind, need, pids)) = stalled else {
            return;
        };
        self.stats.stall_dumps.fetch_add(1, Ordering::Relaxed);
        let (free, total) = self.port.device_stats();
        let (map, census) = need.map_or((String::new(), String::new()), |(run, _)| {
            let mut map = self.port.chunk_occupants(&pids);
            map.merge(self.port.parked_occupants(pid));
            (map.explain(run), self.port.chunk_census(run))
        });
        tracing::warn!(
            ?key,
            %pid,
            kind,
            need = ?need,
            free_runs = need.map(|(run, _)| self.port.guest_free_runs(run)),
            device = ?(free, total),
            queue = %self.debug_queue(),
            %map,
            pool = %census,
            "planner: the queue's head is unserved with nothing in flight"
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
    fn plan_once(self: &Arc<Self>) -> bool {
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
                if let WaitKind::BindRoom { run, pieces, .. } = waiter.kind {
                    return Step::BindRoom { key, run, pieces };
                }
                let is_restore = matches!(waiter.kind, WaitKind::Restore { .. });
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
                                .is_some_and(|restoring| restoring != pid) =>
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
                                    }),
                                    _ => None,
                                })
                                .unwrap_or(Step::Done)
                        }
                        // Its whole need: the room its bind asks too.
                        Some(need) if room_stood != Some(key) => {
                            Step::RestoreBindRoom { key, need }
                        }
                        _ => match inner.restore_gate(pid, spare) {
                            RestoreGate::Serve => Step::ServeRestore { key, pid },
                            RestoreGate::Wait => Step::Done,
                            RestoreGate::Evict { need, fund } => {
                                Step::EvictForRoles { need, fund }
                            }
                        },
                    },
                }
            });
            let fund_by_eviction = match step {
                Step::Done => return false,
                Step::BindRoom { key, run, pieces } => {
                    // Its runs stand free: the bind takes them.
                    if self.port.guest_room_free(run, pieces) {
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
                    if self.park_down_for(run, pieces) {
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
                            inner.bind_restoring.is_some_and(|restoring| Some(restoring) != pid),
                        )
                    });
                    debug_assert!(!other_restoring, "selected only with no other restoring");
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
                    // the bind-suspended head's runs never funded).
                    if self.park_down_for(need.run, need.pieces) {
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
                    // once the map has moved past its refusal: never twice on
                    // the same map.
                    let gives_now = self.port.room_gives();
                    let map_moved = self.with_inner(|inner| {
                        inner.unmet_head().is_some_and(|(_, waiter)| {
                            inner.procs.get(&waiter.pid).is_some_and(|proc| {
                                proc.roles_short_gives.is_none_or(|at| gives_now != at)
                            })
                        })
                    });
                    if !understated && map_moved && self.port.guest_room_free(run, pieces) {
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
                        kv_pages: self.port.chunks_as_pages(run.saturating_mul(pieces)),
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
                    if self.park_down_for(run, pieces) {
                        return false;
                    }
                    // Else the occupants of the chunks it could take move
                    // for it (`plan_eviction`'s chunk branch), under the
                    // key rule.
                    true
                }
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
