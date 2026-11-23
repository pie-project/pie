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
        // A fire of a process that is not resident (evicting, evicted,
        // restoring) is granted nothing, not even an empty grant: its
        // instance's slots may be on their way off the device, or given to
        // another owner. It yields and waits for its restore (`acquire`).
        if self.nonresident.load(Ordering::Acquire) != 0 && !self.is_resident(pid) {
            tracing::debug!(%pid, "planner: no grant for a fire of a process that is not resident");
            return None;
        }
        if demand.is_zero() {
            if self.returns.waiters.load(Ordering::Acquire) != 0
                || self.nonresident.load(Ordering::Acquire) != 0
            {
                self.note_progress(pid);
            }
            return Some(AllocationGrant::empty());
        }
        // Past what every tier keeps for the finisher, it queues and waits
        // there, named ([`super::inner::CapacityGate`]).
        if self.capacity_holds(pid, demand.kv_pages) {
            return None;
        }
        let uncontended = self.returns.waiters.load(Ordering::Acquire) == 0
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

    /// The refusal of `alone`, the most one unit of a fire takes on the
    /// device at once (one KV working set's lanes: every page they read
    /// and write, what they share counted, and their state's ask), where
    /// it is past the whole device pool: no split of the fire and no move
    /// of any other process makes it fit
    /// ([`PlannerError::PastDevice`]). `None` for every unit that fits
    /// alone, however short the pool is now.
    pub fn past_device(&self, alone: Demand) -> Option<PlannerError> {
        let total = self.device_pages_most();
        let need = alone.kv_pages.saturating_add(self.port.rs_as_pages(alone.rs));
        (need > total).then_some(PlannerError::PastDevice {
            need,
            total,
            runs: None,
        })
    }

    pub async fn acquire(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<Acquired, PlannerError> {
        let acquired = self.acquire_inner(pid, quorum_id, demand, false).await;
        if matches!(acquired, Ok(Acquired::Granted(_))) {
            self.note_peak(pid, demand);
        }
        acquired
    }

    /// `pid`'s ask of `pages` kv pages for an activation rise that cannot
    /// fire narrower (a piece no cut brings under the rung held, a
    /// program's fire term): a keyed demand, served in the requester's key
    /// order like any, its grant given back at once for the rise to take.
    /// The requester's own pages in the span are moved by the rise, never
    /// freed, so they weigh in no hog check, and a pool
    /// whose rise cannot seat what one process holds was refused at load
    /// (its largest rung beside one sequence at the context).
    pub async fn acquire_rise(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        pages: u32,
    ) -> Result<Acquired, PlannerError> {
        let demand = Demand {
            kv_pages: pages,
            rs: crate::store::rs::RsUnits::default(),
        };
        self.acquire_inner(pid, quorum_id, demand, true).await
    }

    async fn acquire_inner(
        self: &Arc<Self>,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
        rise: bool,
    ) -> Result<Acquired, PlannerError> {
        if let Some(grant) = self.grant_now(pid, demand) {
            return Ok(Acquired::Granted(grant));
        }
        let (_, kv_total) = self.port.device_stats();
        // The finisher grows on the device alone: an end past the device
        // never finishes, and is refused by name at its first ask, the
        // bound named (the declared context: a process states no end).
        if !rise
            && let Some(end) = self.port.context_pages()
            && end > self.device_pages_most()
            && let Some(room) = self.port.room()
        {
            return Err(PlannerError::EndPastDevice { end, room });
        }
        if demand.kv_pages > kv_total && !rise {
            if let Some(room) = self.port.room() {
                return Err(PlannerError::PastPool {
                    need: demand.kv_pages,
                    room,
                });
            }
            return Err(PlannerError::Impossible {
                need: demand.kv_pages,
                total: kv_total,
            });
        }
        // On a shared pool the ask's state takes chunks its kv would: kv and
        // state together past the pool never fit, whatever the process holds
        // (an empty-handed ask is caught here, not by a wedge's verdict).
        let state_pages = self.port.rs_as_pages(demand.rs);
        if !rise && state_pages > 0 && demand.kv_pages.saturating_add(state_pages) > kv_total {
            let need = demand.kv_pages.saturating_add(state_pages);
            if let Some(room) = self.port.room() {
                return Err(PlannerError::PastPool { need, room });
            }
            return Err(PlannerError::Impossible {
                need,
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
            // Short of state, a resident process quotes again once the idle
            // cache its keys let go is dropped, or else the snapshots sharing
            // its own state that may go, instead of parking. What a policy
            // keeps stays: the waiter parks and asks for room.
            let resident = self.with_inner(|inner| {
                inner.procs.get(&pid).is_some_and(|proc| {
                    proc.state == Residency::Resident && !inner.killing.contains(&pid)
                })
            });
            if !rs_free.covers(demand.rs) && resident && !self.port.grow_rs(demand.rs) {
                // What this frees may be what an older waiter parked on.
                if self.reclaim_idle_once() {
                    self.pages_freed();
                    return Ok(Acquired::Requote);
                }
                if self.reclaim_own_sharing(pid) {
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
                    rise,
                },
            );
            Parked::Entry(key, notify)
        });
        match parked {
            Parked::Gone => Err(PlannerError::Cancelled),
            Parked::NotResident => Ok(Acquired::Yield),
            Parked::Entry(key, notify) => {
                self.stats.parks.fetch_add(1, Ordering::Relaxed);
                // Queued for room: its owner is parked here until served.
                let _parked = crate::inferlet::process::parks_here();
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
                    // Room a grant gives back as it drops (no lock taken
                    // there, `RoomReturns`): read before it parks, and
                    // after every wake, a move plans again.
                    let returned = self.returns.returned();
                    tokio::pin!(returned);
                    returned.as_mut().enable();
                    let seen = self.returns.epoch();
                    let acquired = match self.collect_outcome(key) {
                        // Short, it waits however long: its own notify
                        // (served, room given back, an eviction landed, a
                        // completion) ends the wait. While it stands short
                        // of room only the policy can make it says so again
                        // (the gauge keeps a band only while it is told);
                        // only a need past the whole device pool, which the
                        // pool shrinking under it can make, is refused.
                        Collect::Wait if self.returns.epoch() != seen => {
                            self.wake_planner();
                            continue;
                        }
                        Collect::Wait => {
                            // Events alone wake it: its own notify, room a
                            // dropped grant gave back, or the drain task's
                            // tick finding its need past the pool
                            // ([`ResidencyPlanner::tick`]). On a 100 ms
                            // timer each parked waiter woke, re-reported
                            // and planned ten times a second, hundreds of
                            // them under pressure.
                            tokio::select! {
                                // In order, never at random: a seeded run takes the same branch.
                                biased;
                                () = &mut notified => {}
                                () = &mut returned => {}
                            }
                            // Room returned since: a dropped grant's, or
                            // slots the pool freed on any path.
                            if self.returns.epoch() != seen
                                || (self.returns.freed_since_plan() && self.wants_met_now())
                            {
                                self.wake_planner();
                            }
                            let Some(error) = self.never_fits(key) else {
                                continue;
                            };
                            match self.collect_outcome(key) {
                                Collect::Wait => {
                                    tracing::warn!(%pid, %error, "planner: an allocation past the device pool is refused");
                                    Err(error)
                                }
                                Collect::Ready(outcome) => outcome.map(Acquired::Granted),
                                Collect::Yield => Ok(Acquired::Yield),
                                Collect::Requote => Ok(Acquired::Requote),
                            }
                        }
                        Collect::Ready(outcome) => {
                            ptrace!("collect key={:?} ok={}", key, outcome.is_ok());
                            outcome.map(Acquired::Granted)
                        }
                        Collect::Yield => Ok(Acquired::Yield),
                        Collect::Requote => Ok(Acquired::Requote),
                    };
                    if !matches!(
                        acquired,
                        Err(PlannerError::PastDevice { .. } | PlannerError::RunPastPool { .. })
                    ) {
                        registration.disarm();
                    }
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

    /// Sleeps until what a non-resident process waits on changes: its own
    /// residency (`signal`, told as it is evicted, restored, rolled back or
    /// gone, or as its queued restore is found past the device pool). No
    /// clock: a planner poke for another process, the shortfall standing, or
    /// the policy answering a shortfall does not wake it. (On a 100 ms
    /// re-report it woke each waiter ten times a second while it stayed
    /// evicted.)
    async fn await_residency_change(
        signal: &Notify,
        settled: impl Fn() -> bool,
        answered: impl Fn(),
    ) {
        let notified = signal.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        loop {
            let policy = crate::policy::pressure::answered();
            tokio::pin!(policy);
            policy.as_mut().enable();
            if settled() {
                return;
            }
            // The policy answering a shortfall plans again, as the give-up's
            // check did (a grow, a commit or a move tried again), and
            // refuses nothing.
            tokio::select! {
                // In order, never at random: a seeded run takes the same branch.
                biased;
                () = &mut notified => return,
                () = &mut policy => answered(),
            }
        }
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
        self.inner.lock().settled_resident(pid)
    }

    /// Registers kv working set `ws` of process `pid` in `residency`,
    /// under the lock the planner moves `pid` off Resident under: the set
    /// either stands before an eviction's Resident → Evicting, and so is in
    /// that eviction's snapshot (fenced and quiesced with it), or comes
    /// after it and is born fenced, its fires waiting for the restore
    /// ([`Self::report_restored`] unfences it). The snapshot is complete by
    /// construction. A beam fork made mid-eviction was neither: its fires
    /// ran on the instance being parked, read mid-copy and reissued.
    pub(crate) fn register_kv_working_set(
        &self,
        pid: ProcessId,
        residency: &std::sync::Mutex<crate::inferlet::process::residency::ProcessResidency>,
        ws: &crate::store::kv::working_set::KvWorkingSet,
    ) {
        let inner = self.lock_inner();
        let resident = inner.settled_resident(pid);
        let handle = ws.suspend_handle();
        if !resident {
            handle.fence();
        }
        residency
            .lock()
            .unwrap()
            .kv_working_sets
            .insert((ws.model, ws.engine, ws.id), handle);
        drop(inner);
    }

    pub fn residency_flag(&self, pid: ProcessId) -> Option<Arc<AtomicBool>> {
        self.lock_inner()
            .procs
            .get(&pid)
            .map(|proc| proc.resident.clone())
    }

    pub async fn wait_resident(self: &Arc<Self>, pid: ProcessId) -> Result<(), PlannerError> {
        let mut left_pipeline = false;
        loop {
            let (signal, asked) = {
                let mut inner = self.lock_inner();
                if !inner.procs.contains_key(&pid) {
                    return Err(PlannerError::Cancelled);
                }
                if inner.settled_resident(pid) {
                    return Ok(());
                }
                let proc = inner.procs.get_mut(&pid).expect("present");
                // Parked owing its restore (it went idle): a process waiting
                // to be resident asks for it again.
                // Left parked, nothing queues it, and the wait never ends
                // (the M1 / M5 stress: the earliest admitted sat evicted
                // with no restore queued once the rest drained).
                // Held below the schedule's floor (or seatless, its seat
                // vacated for one the schedule chose), it stays parked
                // until the policy lifts it ([`super::inner::Inner::reschedule`] queues
                // its restore then): its own lanes waiting to run are no
                // lift, nor is room standing free.
                let demand = (proc.state == Residency::Evicted && !proc.held && !proc.seatless)
                    .then(|| proc.parked.take())
                    .flatten();
                let signal = proc.signal.clone();
                if let Some(demand) = demand {
                    inner.enqueue(pid, WaitKind::Restore { demand });
                }
                (signal, demand.is_some())
            };
            if asked {
                self.poke();
            }
            if !left_pipeline {
                left_pipeline = true;
                crate::scheduler::worker::notify_process_suspend(pid);
            }
            // A restore past the whole device pool is refused by name, at
            // once; any other waits for room, however long.
            if let Some(error) = self.restore_never_fits(pid) {
                tracing::warn!(%pid, %error, "planner: a restore past the device pool is refused");
                return Err(error);
            }
            self.stats.gate_parks.fetch_add(1, Ordering::Relaxed);
            // Waiting its restore turn: its owner is parked here.
            let _parked = crate::inferlet::process::parks_here();
            Self::await_residency_change(
                &signal,
                || self.is_resident(pid) || self.restore_never_fits_now(pid),
                // The policy answered nothing freed: plan again only where
                // room came free since the last pass (else a pass said the
                // shortfall again and drew the next such answer, a loop).
                || {
                    if self.returns.freed_since_plan() {
                        self.wake_planner();
                    }
                },
            )
            .await;
        }
    }

    /// `pid`'s queued restore, if one is queued.
    fn queued_restore(&self, pid: ProcessId) -> Option<EntryKey> {
        self.with_inner(|inner| {
            inner
                .queue
                .iter()
                .find(|(_, waiter)| {
                    waiter.pid == pid && matches!(waiter.kind, WaitKind::Restore { .. })
                })
                .map(|(key, _)| *key)
        })
    }

    /// Whether `pid`'s queued restore could never fit the device pool
    /// ([`Self::never_fits`]), read without acting on it.
    pub(super) fn restore_never_fits_now(&self, pid: ProcessId) -> bool {
        self.queued_restore(pid)
            .is_some_and(|key| self.never_fits(key).is_some())
    }

    /// A queued restore of `pid` whose need could never fit the device pool
    /// ([`Self::never_fits`]): it leaves the queue owing what it did
    /// (parked, as a holder's set is), and its error says what is past the
    /// pool. `None` for every need that fits: it waits.
    pub(super) fn restore_never_fits(self: &Arc<Self>, pid: ProcessId) -> Option<PlannerError> {
        let key = self.queued_restore(pid)?;
        let error = self.never_fits(key)?;
        let settled = self.with_inner(|inner| {
            if let Some(super::wait::Waiter {
                pid,
                kind: WaitKind::Restore { demand },
                ..
            }) = inner.queue.remove(&key)
                && let Some(proc) = inner.procs.get_mut(&pid)
                && proc.state == Residency::Evicted
            {
                proc.parked = Some(demand);
                // What its restore learned of the room is the room's then:
                // queued again ([`Self::wait_resident`]) it starts from
                // the room as it stands.
                proc.roles_short = None;
                proc.roles_free_retry = None;
                proc.roles_free_short = None;
                proc.roles_short_gives = None;
                proc.roles_short_runs = None;
                proc.roles_short_free = None;
                proc.last_refusal = None;
                proc.roles_short_stood = None;
                proc.roles_refused = None;
                proc.roles_short_mismatch = None;
                return true;
            }
            false
        });
        if !settled {
            // Served or gone meanwhile: the loop reads where it stands.
            return None;
        }
        self.note_short(key, 0, 0);
        self.poke();
        Some(error)
    }
}

/// How a bind's wait for room ([`ResidencyPlanner::wait_bind_room`]) ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindRoomWait {
    /// Its runs stand free: the bind takes them.
    Served,
    /// Its process is being evicted: the bind waits for its restore.
    Withdrawn,
}

impl ResidencyPlanner {
    /// Waits, queued in the planner beside admissions and restores, until
    /// the device pool holds `pieces` free runs of `run` adjacent chunks
    /// for a bind of `pid`'s short of room for its guest state
    /// (`store::roles::RolesError::BindShort`). The planner funds it as it
    /// funds a restore short of room for its roles: idle cache dropped
    /// first, then, the fleet stalled, a resident parked by its victim
    /// choice; the bind is sent again once served. `Err` where `pid` is
    /// gone. Woken by its own serve, never a clock.
    ///
    /// An eviction of `pid` itself withdraws the wait
    /// ([`BindRoomWait::Withdrawn`]): evicted, its bind waits for it to be
    /// resident again, never for room funded meanwhile.
    pub async fn wait_bind_room(
        self: &Arc<Self>,
        pid: ProcessId,
        run: u32,
        pieces: u32,
        singles: u32,
    ) -> Result<BindRoomWait, PlannerError> {
        let pages = self
            .port
            .chunks_as_pages(run.max(1).saturating_mul(pieces).saturating_add(singles));
        // Runs past the whole device pool never stand: refused by name, at
        // once. Any other bind waits for its runs, however long.
        let total = self.device_pages_most();
        if pages > total {
            return Err(PlannerError::PastDevice {
                need: pages,
                total,
                runs: Some((run.max(1), pieces)),
            });
        }
        // A run longer than any segment lays for state: never laid.
        let longest = self.port.longest_run_most();
        if pieces > 0 && run.max(1) > longest {
            return Err(PlannerError::RunPastPool {
                run: run.max(1),
                longest,
            });
        }
        self.wait_room(pid, (run, pieces, singles), pages, None)
            .await
            .map(|(wait, _registration)| wait)
    }

    /// `pid`'s raise of the activation to `rung` (tokens) short of room,
    /// a walk's raise to frames its scheduler holds out under a lowered cap
    /// (`store::shared::owe_cut_rows`): queued in the planner by `pid`'s
    /// key beside admissions, restores and binds, and funded exactly as a
    /// bind's room is (idle cache first, a grow, parked state down a tier,
    /// then the occupants of the chunks it takes by the chunk map, under
    /// the key rule), however long; completions, evictions landed and room
    /// moved quote it again. Once its room stands `take` runs (the raise,
    /// its moves and the engine's copy) with the queue held behind it, so
    /// no head after it is served into that room first; its result is
    /// returned. A rung that could never fit the pool alone is refused by
    /// name, at once ([`PlannerError::PastRung`]). `Ok(None)` where `pid`
    /// is being evicted meanwhile (its frames leave with it).
    pub async fn fund_rise<T, F>(
        self: &Arc<Self>,
        pid: ProcessId,
        rung: u32,
        take: impl FnOnce() -> F,
    ) -> Result<Option<T>, PlannerError>
    where
        F: std::future::Future<Output = T>,
    {
        let pieces = match self.port.rise_room(pid, rung) {
            super::pool::RiseRoom::Past { need, own, total } => {
                return Err(PlannerError::PastRung { rung, need, own, total });
            }
            super::pool::RiseRoom::Fits => return Ok(Some(take().await)),
            super::pool::RiseRoom::Short { pieces } => pieces,
        };
        let pages = self.port.chunks_as_pages(pieces);
        let (wait, registration) = self.wait_room(pid, (1, pieces, 0), pages, Some(rung)).await?;
        if wait == BindRoomWait::Withdrawn {
            return Ok(None);
        }
        // Served: it fits, or (the pool shrank under it) never will.
        if let super::pool::RiseRoom::Past { need, own, total } = self.port.rise_room(pid, rung) {
            return Err(PlannerError::PastRung { rung, need, own, total });
        }
        let taken = take().await;
        // Off the queue: the heads behind it are planned again.
        drop(registration);
        Ok(Some(taken))
    }

    /// Queues `pid`'s wait for room (a bind's runs, or a rung raise's
    /// chunks: `rise`) and waits until it is served or withdrawn; the
    /// registration that takes it off the queue is handed back, held for
    /// as long as the caller keeps the room the wait was served.
    async fn wait_room(
        self: &Arc<Self>,
        pid: ProcessId,
        (run, pieces, singles): (u32, u32, u32),
        pages: u32,
        rise: Option<u32>,
    ) -> Result<(BindRoomWait, Option<WaitRegistration<'_>>), PlannerError> {
        let notify = Arc::new(Notify::new());
        let key = self
            .with_inner(|inner| {
                // Evicting already, it waits for its restore instead. A
                // rung raise for its capped frame waits on: its eviction
                // waits on that frame, which goes only once raised.
                let proc = inner.procs.get(&pid)?;
                let raising = rise.is_some() && proc.state == Residency::Evicting;
                if proc.state != Residency::Resident && !raising {
                    return Some(None);
                }
                // Its bind asks its room again: what an eviction it was
                // withdrawn by left pending is this ask now.
                if rise.is_none()
                    && let Some(proc) = inner.procs.get_mut(&pid)
                {
                    proc.bind_room = None;
                }
                Some(Some({
                    inner.enqueue(
                        pid,
                        WaitKind::BindRoom {
                            run,
                            pieces,
                            singles,
                            pages,
                            notify: notify.clone(),
                            served: false,
                            withdrawn: false,
                            rise,
                        },
                    )
                }))
            })
            .ok_or(PlannerError::Cancelled)?;
        let Some(key) = key else {
            return Ok((BindRoomWait::Withdrawn, None));
        };
        // Queued for room: its owner is parked here until served.
        let _parked = crate::inferlet::process::parks_here();
        // Off the queue however the wait ends.
        let registration = WaitRegistration {
            planner: self,
            key,
            active: true,
        };
        self.poke();
        loop {
            let notified = notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let returned = self.returns.returned();
            tokio::pin!(returned);
            returned.as_mut().enable();
            let seen = self.returns.epoch();
            let served = self.with_inner(|inner| {
                let key = inner.locate(key)?;
                match inner.queue.get(&key)?.kind {
                    WaitKind::BindRoom {
                        served, withdrawn, ..
                    } => Some((served, withdrawn)),
                    _ => None,
                }
            });
            match served {
                None => return Err(PlannerError::Cancelled),
                Some((_, true)) => return Ok((BindRoomWait::Withdrawn, Some(registration))),
                Some((true, false)) => return Ok((BindRoomWait::Served, Some(registration))),
                Some((false, false)) if self.returns.epoch() != seen => self.wake_planner(),
                Some((false, false)) => {
                    // Room that comes free or moves (the rung lowered or
                    // raised, a segment committed, guest state given back)
                    // has the planner quote the waiting head again, not only
                    // its own serve: it never waits on room that is there.
                    let (model, engine) = self.port.locus();
                    let rung = crate::store::shared::rung_watch(engine);
                    let given = crate::store::registry::try_get(model, engine)
                        .and_then(|stores| stores.roles)
                        .map(|roles| roles.lock().unwrap().room_watch());
                    let given = async move {
                        match given {
                            Some(watch) => watch.freed().await,
                            None => std::future::pending().await,
                        }
                    };
                    tokio::select! {
                        // In order, never at random: a seeded run takes the same branch.
                        biased;
                        () = notified => {}
                        () = returned => self.wake_planner(),
                        () = rung => self.wake_planner(),
                        () = given => self.wake_planner(),
                    }
                }
            }
        }
    }

    /// The head bind's runs stand free: it is told, to take them.
    pub(super) fn serve_bind_room(self: &Arc<Self>, key: EntryKey) {
        let notify = self.with_inner(|inner| {
            let key = inner.locate(key)?;
            let waiter = inner.queue.get_mut(&key)?;
            let pid = waiter.pid;
            let notify = match &mut waiter.kind {
                WaitKind::BindRoom {
                    served,
                    notify,
                    withdrawn: false,
                    ..
                } => {
                    *served = true;
                    notify.clone()
                }
                _ => return None,
            };
            // Bound for a waiting head (its own bind): no victim until it
            // has fired once.
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.unfired = true;
            }
            Some(notify)
        });
        if let Some(notify) = notify {
            notify.notify_waiters();
        }
    }
}
