//! KV contention orchestration — FCFS preempt/restore (Task-B rework).
//!
//! In Gim's directive: KV-cache contention is handled by PREEMPT/RESTORE, not
//! admission. `max_concurrent_processes` is a large physical safety cap only;
//! when a forward's page allocation hits an exhausted pool
//! ([`WorkingSetError::OutOfBlocks`](crate::working_set::kv::WorkingSetError)),
//! the prep seam routes here instead of surfacing an inferlet error:
//!
//! 1. **Victim loop** — FCFS-preempt: suspend the YOUNGEST running process
//!    (latest spawn order; the oldest first-comer's progress is protected)
//!    until the request fits. The physical state-save (D2H stash / drop-to-
//!    replay, grace/refcount/txn guards) is behind [`ReclaimBackend`] —
//!    alpha's working-set wrappers; this module owns only the ORCHESTRATION.
//! 2. **Wait queue** — no victim (the requester is the youngest, or every
//!    peer is pinned/suspended): the requester parks FIFO in `waiters` and is
//!    woken as blocks free. Waiters hold NOTHING while parked (their forward
//!    txn was aborted before `acquire`), so waiting cannot deadlock.
//! 3. **Restore-on-free** — when blocks free and no waiter is parked, the
//!    oldest-suspended process restores (FCFS), gated by the anti-thrash
//!    guard: restore NEVER evicts (`free >= suspended_need`) and is paused
//!    while pool utilization exceeds `restore_pause_at_utilization` (the
//!    prior design's evict→restore→re-evict cascade guard, bootstrap.rs).
//! 4. **Exhaustion endgame (#19)** — a non-terminating guest can grow its
//!    context past the whole pool; the FCFS-oldest keystone then parks
//!    unsatisfiable forever (no victim; restore only consumes free). After
//!    `PIE_KV_EXHAUSTION_MS` of continuous quiet, `acquire` fails LOUD with
//!    [`ContentionError::Exhausted`] (→ `OutOfBlocks` to the guest) instead of
//!    wedging. (Alternative not built: *victim-the-hog* — cold-drop the
//!    survivor's own context to free the pool for the fleet; ship fail-loud.)
//!
//! Blueprint: the deleted `context/sched.rs` `when_allocated`/`drain_queues`
//! (GitHub main, `fa3fe140^`), stripped of its market layer. The prior code
//! deferred ops as closures because the context actor was synchronous; here
//! `execute_impl` is async, so `acquire().await` + retry-the-prep replaces
//! the closure machinery.
//!
//! Lock discipline: `inner` is a plain mutex ordered INSIDE everything else —
//! no [`ReclaimBackend`] call (which may take arena/working-set locks) is
//! ever made while holding it, and no await happens under it.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use crate::process::ProcessId;

/// How the runtime reacts to a KV pool exhaustion at the prep seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentionMode {
    /// Legacy: surface `OutOfBlocks` to the inferlet as a forward error.
    Error,
    /// Task-B: route to the orchestrator (preempt/restore) and retry.
    Preempt,
}

/// The contention mode, read once from `PIE_KV_CONTENTION`
/// (`preempt` ⇒ [`ContentionMode::Preempt`]; default/anything else ⇒ `Error`,
/// byte-for-byte legacy behavior).
pub fn contention_mode() -> ContentionMode {
    static MODE: OnceLock<ContentionMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("PIE_KV_CONTENTION")
            .map(|v| v.eq_ignore_ascii_case("preempt"))
            .unwrap_or(false)
        {
            true => ContentionMode::Preempt,
            false => ContentionMode::Error,
        }
    })
}

/// Outcome of a suspend attempt on a victim process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuspendOutcome {
    /// The victim's state was saved; this many pool blocks are free NOW.
    Suspended { freed_now: u32 },
    /// The victim's owned pages are all arena-PINNED by an in-flight forward
    /// (run-ahead co-batch): they can't be stashed now and free later at that
    /// forward's finalize (`on_blocks_freed` re-drains then). The victim stays
    /// running; the orchestrator tries the next victim.
    DeferredByGrace,
    /// SELF-SUSPEND protocol (B-refined): a park request was posted for the
    /// victim; it saves its own state at its next host-call boundary (the
    /// working set is process-local in its wasmtime ResourceTable — no
    /// cross-process access exists, by design). The orchestrator marks it
    /// `ParkRequested` and moves on; blocks arrive via
    /// [`ContentionOrchestrator::report_suspended`] when the victim complies.
    Requested,
    /// The backend cannot reclaim from this process (nothing suspendable, or
    /// suspend not wired yet) — exempt it from this request's victim loop.
    Unsupported,
}

/// The physical reclaim/restore surface the orchestrator drives. Implemented
/// over alpha's working-set state-save wrappers (`classify_for_suspend`,
/// `suspend_pages_warm`, `restore_pages_warm` + the cold/replay path); unit
/// tests use a mock. Methods may take arena/working-set locks — the
/// orchestrator never calls them while holding its own state lock.
pub trait ReclaimBackend: Send + Sync + 'static {
    /// `(free, total)` blocks of the contended pool.
    fn pool_stats(&self) -> (u32, u32);
    /// Save `victim`'s state and free its pool blocks.
    fn suspend(&self, victim: ProcessId) -> SuspendOutcome;
    /// Re-materialize a suspended process (H2D warm restore / replay). On
    /// `Err` the orchestrator re-queues it (never silently dropped).
    fn restore(&self, pid: ProcessId) -> anyhow::Result<()>;
}

/// A hard (non-contention) failure from `acquire`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentionError {
    /// The request can never fit: `need` exceeds the pool's total capacity.
    /// Surfaced to the inferlet as a real error (fail loud, prior design).
    Impossible { need: u32, total: u32 },
    /// The pool is EXHAUSTED for this requester (#19): after
    /// `PIE_KV_EXHAUSTION_MS` of CONTINUOUS unsatisfiability — no victim can
    /// yield, `free < need`, and the requester is the FCFS-oldest keystone that
    /// never self-yields — its request cannot be met by ANY orchestrator action.
    /// A non-terminating guest grew its context past what the pool can free.
    /// Fail LOUD to the guest (OutOfBlocks) instead of a silent wedge.
    Exhausted { need: u32, free: u32, total: u32 },
}

impl std::fmt::Display for ContentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentionError::Impossible { need, total } => write!(
                f,
                "allocation of {need} blocks can never fit (pool total {total})"
            ),
            ContentionError::Exhausted { need, free, total } => write!(
                f,
                "KV pool exhausted: request of {need} blocks unsatisfiable \
                 ({free} free of {total}; a non-terminating guest holds ~{})",
                total.saturating_sub(*free)
            ),
        }
    }
}

impl std::error::Error for ContentionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcState {
    Running,
    /// A suspend is in flight on another task (excluded from victim pick).
    Suspending,
    /// A self-suspend park request is posted ([`SuspendOutcome::Requested`]);
    /// the process saves its own state at its next host-call boundary
    /// (`should_park` → wrapper suspend → `report_suspended`). Excluded from
    /// victim pick; still running until it complies.
    ParkRequested,
    Suspended,
    /// A restore is in flight (excluded from victim pick and restore queue).
    Restoring,
}

#[derive(Debug)]
struct Proc {
    /// Registration order — the FCFS clock. Victim = max `spawn_seq` running.
    spawn_seq: u64,
    state: ProcState,
    /// Blocks freed when this process was suspended = the restore admission
    /// estimate (`can_restore`: restore NEVER evicts).
    suspended_need: u32,
    /// When the process entered `Suspended` — drives the restore-pause AGING
    /// override (a starved FIFO-head restore eventually proceeds even above
    /// the utilization pause, if it fits).
    suspended_at: Option<Instant>,
}

/// What the caller must do after [`ContentionOrchestrator::acquire_or_self_suspend`]
/// returns successfully.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acquired {
    /// Blocks are plausibly available — RETRY the allocation.
    Retry,
    /// No victim can yield and the requester itself holds reclaimable pages:
    /// the caller must run the SELF-SUSPEND protocol on its own working set
    /// (suspend → `report_suspended` → `park_until_restored` → restore),
    /// then retry. This is the prior design's `when_allocated` **step 6**
    /// ("NO VICTIM — requester self-suspends", sched.rs:861-866 @ fa3fe140^)
    /// — the designed deadlock-breaker: a parked page-HOLDER would otherwise
    /// strand its pages forever (the fleet=24/8 v2 deadlock).
    SelfSuspendFirst,
}

/// A parked allocation. The entry STAYS in the FIFO until its `acquire`
/// observes enough free blocks and removes itself (by `token`) — so phase 2
/// of `drain` (restores) is structurally blocked while any waiter is parked,
/// and a wake that loses the race to another allocator keeps its FIFO slot.
struct Waiter {
    token: u64,
    pid: ProcessId,
    need: u32,
    notify: Arc<Notify>,
}

#[derive(Default)]
struct Inner {
    next_seq: u64,
    next_token: u64,
    procs: HashMap<ProcessId, Proc>,
    /// FIFO alloc waiters — strict priority over restores (prior
    /// `drain_queues` phase 1).
    waiters: VecDeque<Waiter>,
    /// FCFS restore order: oldest-suspended first.
    restore_queue: VecDeque<ProcessId>,
    /// Self-suspended (parked) processes awaiting release. Entry created by
    /// `report_suspended` BEFORE any drain can release it, so a release that
    /// beats `park_until_restored` stores its permit in the `Notify` (no
    /// lost wakeup). Removed on wake-return and on unregister.
    parked: HashMap<ProcessId, Arc<Notify>>,
}

/// Engagement counters (lock-free reads) — the e2e's proof that contention
/// actually ENGAGED, so a trivially-passing run (pool never over-filled)
/// can't masquerade as a validated preempt/restore path. NOTE for asserts:
/// under the v1 passive backend (`ArenaPoolBackend`, suspend=Unsupported)
/// `suspends`/`restores` are ZERO BY DESIGN — v1 engagement is
/// `waiters_parked > 0` (+ `waiters_woken > 0`); suspend/restore counts
/// become meaningful with the v2 state-save backend.
#[derive(Debug, Default)]
pub struct ContentionStats {
    /// `acquire` calls that parked in the FIFO (no victim yielded blocks now).
    pub waiters_parked: std::sync::atomic::AtomicU64,
    /// Parked acquires that returned satisfied.
    pub waiters_woken: std::sync::atomic::AtomicU64,
    /// Victims whose state-save freed blocks (backend `Suspended` +
    /// self-suspend `report_suspended`).
    pub suspends: std::sync::atomic::AtomicU64,
    /// Suspended processes restored/released back to Running.
    pub restores: std::sync::atomic::AtomicU64,
}

pub struct ContentionOrchestrator {
    inner: Mutex<Inner>,
    backend: Box<dyn ReclaimBackend>,
    /// Pause restores while `used/total` exceeds this (anti-thrash; the
    /// existing `scheduler.restore_pause_at_utilization` config feeds it).
    restore_pause_at_utilization: f64,
    stats: ContentionStats,
    /// Broadcast per fire retire (`notify_waiters` semantics — no stored
    /// permit). Deep-running lanes gate on this to re-drain their own
    /// already-retired fires under contention (the carrier ⋈ contention fix:
    /// a lane's pins release only when ITS OWN task finalizes its fires, so
    /// each retire re-opens the drain window). Racy-by-design: a missed
    /// notification is covered by the next retire, and gate loops re-check
    /// state before every wait (`Notified::enable`).
    fire_retired: Arc<Notify>,
    /// #19 exhaustion deadline (ms): how long the FCFS-oldest keystone may stay
    /// CONTINUOUSLY unsatisfiable (no victim, `free < need`) before `acquire`
    /// fails LOUD with [`ContentionError::Exhausted`] instead of wedging. Read
    /// from `PIE_KV_EXHAUSTION_MS` (default 10000) at construction; the clock
    /// resets whenever the condition clears (reset-on-change kills transients).
    exhaustion_ms: u64,
}

impl ContentionOrchestrator {
    pub fn new(backend: Box<dyn ReclaimBackend>, restore_pause_at_utilization: f64) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            backend,
            restore_pause_at_utilization,
            stats: ContentionStats::default(),
            fire_retired: Arc::new(Notify::new()),
            exhaustion_ms: exhaustion_ms_from_env(),
        }
    }

    /// Override the #19 exhaustion deadline (tests; production uses the env
    /// default). Builder-style so `Arc::new(orch(..).with_exhaustion_ms(ms))`.
    pub fn with_exhaustion_ms(mut self, ms: u64) -> Self {
        self.exhaustion_ms = ms;
        self
    }

    /// Engagement counters (see [`ContentionStats`] for v1-vs-v2 semantics).
    pub fn stats(&self) -> &ContentionStats {
        &self.stats
    }

    /// True while any allocation waiter is parked — the pool is contended and
    /// deep-running lanes should self-serialize (drain their own fires, stop
    /// deepening) so their pins can release.
    pub fn contended(&self) -> bool {
        !self.inner.lock().unwrap().waiters.is_empty()
    }

    /// Is `pid` the FCFS-oldest registered process? The oldest is the
    /// completion keystone: never a victim (victim pick) and never a
    /// self-yielder (step-6 gate) — it parks and FCFS hands it the next freed
    /// blocks, guaranteeing fleet-wide progress one completion at a time.
    fn is_fcfs_oldest(&self, pid: ProcessId) -> bool {
        let inner = self.inner.lock().unwrap();
        let Some(p) = inner.procs.get(&pid) else {
            return false;
        };
        inner.procs.values().all(|q| q.spawn_seq >= p.spawn_seq)
    }

    /// The per-fire retire signal (see the field doc). Callers use the
    /// enable-then-check-then-await pattern to avoid lost wakeups.
    pub fn fire_retired(&self) -> Arc<Notify> {
        self.fire_retired.clone()
    }

    /// A fire retired — wake any lane gated on draining its own launches.
    pub fn on_fire_retired(&self) {
        self.fire_retired.notify_waiters();
    }

    /// Register a process at spawn — its registration order is the FCFS clock.
    pub fn register(&self, pid: ProcessId) {
        let mut inner = self.inner.lock().unwrap();
        let seq = inner.next_seq;
        inner.next_seq += 1;
        inner.procs.insert(
            pid,
            Proc {
                spawn_seq: seq,
                state: ProcState::Running,
                suspended_need: 0,
                suspended_at: None,
            },
        );
    }

    /// Unregister at process exit/terminate. Its parked waiters are removed
    /// (their `acquire` futures are being aborted with the process) and it
    /// leaves the restore queue. The exit frees the process's blocks — the
    /// caller runs [`on_blocks_freed`](Self::on_blocks_freed) after resource
    /// cleanup; this also drains defensively.
    pub fn unregister(&self, pid: ProcessId) {
        let parked = {
            let mut inner = self.inner.lock().unwrap();
            inner.procs.remove(&pid);
            inner.waiters.retain(|w| w.pid != pid);
            inner.restore_queue.retain(|&p| p != pid);
            inner.parked.remove(&pid)
        };
        if let Some(notify) = parked {
            // A parked task being torn down: wake it so `park_until_restored`
            // observes the removal and returns.
            notify.notify_one();
        }
        self.drain();
    }

    /// Whether `pid` is currently suspended (probe/test accessor; the
    /// scheduler wait-set Leave/rejoin coupling reads process state events,
    /// not this, but tests assert through it).
    pub fn is_suspended(&self, pid: ProcessId) -> bool {
        let inner = self.inner.lock().unwrap();
        matches!(
            inner.procs.get(&pid).map(|p| p.state),
            Some(ProcState::Suspended) | Some(ProcState::Restoring)
        )
    }

    /// Blocks freed somewhere (forward txn abort/commit released pages, a
    /// process exited, the M4 grace period released a deferred batch). Wakes
    /// FIFO waiters first, then restores suspended processes under the
    /// anti-thrash guard. Never called with arena/WS locks held.
    pub fn on_blocks_freed(&self) {
        self.drain();
    }

    // =========================================================================
    // Self-suspend protocol (B-refined) — the victim-side surface.
    //
    // A running victim's working set is process-local (wasmtime ResourceTable);
    // no cross-process access exists by design. So an active preempt is a
    // three-step handshake driven by the VICTIM's own task at its next
    // host-call boundary (execute_impl prologue):
    //
    //   1. `should_park(pid)`  — cheap check; true after a backend returned
    //      `SuspendOutcome::Requested` for this process.
    //   2. The victim saves its own state (working-set suspend wrappers, with
    //      its own table access) and calls `report_suspended(pid, freed_now)`.
    //   3. `park_until_restored(pid).await` — parks until the restore phase
    //      releases it (FCFS oldest-first, thrash-guarded); on wake the victim
    //      re-materializes its state (again with its own access) and, if the
    //      re-materialize itself hits contention, re-reports + re-parks.
    //
    // The scheduler wait-set Leave (a parked pipeline must not be awaited) is
    // emitted at `report_suspended` time once the lifecycle_rx wiring lands
    // (M-AB); until then the wave deadline + miss-demote bounds the wait.
    // =========================================================================

    /// Victim-side step 1: does `pid` have a pending park request?
    /// Checked at the host-call boundary before any allocation work.
    pub fn should_park(&self, pid: ProcessId) -> bool {
        let inner = self.inner.lock().unwrap();
        matches!(
            inner.procs.get(&pid).map(|p| p.state),
            Some(ProcState::ParkRequested)
        )
    }

    /// Victim-side step 2a — DECLINE: the victim reached its host-call
    /// boundary but has nothing to yield right now (zero materialized pages,
    /// or its state-save is grace-blocked by an in-flight pass). Clears the
    /// park request (`ParkRequested` → `Running`) so the state machine
    /// doesn't leak a permanently-flagged process; it stays a future victim
    /// candidate (a later `acquire` may re-request it once its state
    /// changes). Without this, a zero-yield victim would sit `ParkRequested`
    /// forever — excluded from victim picks yet never parked.
    pub fn decline_park(&self, pid: ProcessId) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(p) = inner.procs.get_mut(&pid) {
            if p.state == ProcState::ParkRequested {
                p.state = ProcState::Running;
            }
        }
    }

    /// Victim-side step 2: the process saved its own state and freed
    /// `freed_now` blocks. Marks it suspended, queues it for FCFS restore,
    /// and drains (its freed blocks wake waiters immediately).
    ///
    /// A×B coupling: the parked victim LEAVES the scheduler wait-set here —
    /// a frozen pipeline must never hold a wave. Rejoin is implicit on its
    /// first post-release request (the wave's implicit-join).
    pub fn report_suspended(&self, pid: ProcessId, freed_now: u32) {
        {
            let mut inner = self.inner.lock().unwrap();
            let Some(p) = inner.procs.get_mut(&pid) else {
                return;
            };
            p.state = ProcState::Suspended;
            p.suspended_need = freed_now;
            p.suspended_at = Some(Instant::now());
            inner.restore_queue.push_back(pid);
            self.stats
                .suspends
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Create the park entry BEFORE any drain can release the process,
            // so an early release stores its permit (no lost wakeup).
            inner.parked.entry(pid).or_default();
        }
        crate::inference::scheduler::notify_pipeline_leave(
            pid,
            crate::inference::scheduler::LeaveKind::Suspend,
        );
        self.drain();
    }

    /// Victim-side step 3: park until the restore phase releases this
    /// process (state back to Running), then return so the caller
    /// re-materializes its state. Returns immediately if the process was
    /// unregistered while parked (it is being torn down).
    pub async fn park_until_restored(&self, pid: ProcessId) {
        loop {
            let notify = {
                let mut inner = self.inner.lock().unwrap();
                match inner.procs.get(&pid).map(|p| p.state) {
                    // Released (or torn down): leave the park table and go.
                    Some(ProcState::Running) | None => {
                        inner.parked.remove(&pid);
                        return;
                    }
                    _ => match inner.parked.get(&pid) {
                        Some(n) => n.clone(),
                        None => {
                            // Defensive: parking without report_suspended.
                            inner.parked.entry(pid).or_default().clone()
                        }
                    },
                }
            };
            notify.notified().await;
        }
    }

    /// Wait until `need` free blocks are plausibly available, FCFS-preempting
    /// younger processes as needed. Returns when the caller should RETRY its
    /// allocation (the retry re-enters here on a lost race — progress is
    /// guaranteed because the oldest process is never a victim and waiters
    /// wake FIFO). The caller must hold NO arena/WS locks and no staged txn.
    pub async fn acquire(&self, requester: ProcessId, need: u32) -> Result<(), ContentionError> {
        self.acquire_or_self_suspend(requester, need, false)
            .await
            .map(|_| ())
    }

    /// The full v2 acquire: like [`acquire`](Self::acquire), but when NO
    /// victim can yield and the requester itself holds reclaimable pages
    /// (`holds_reclaimable`), returns [`Acquired::SelfSuspendFirst`] INSTEAD
    /// of parking — the prior design's step 6 (requester self-suspends),
    /// restoring the invariant that every parked page-holder has freed its
    /// pages (the fleet=24/8 deadlock-breaker). Callers that pass
    /// `holds_reclaimable=false` get the v1 park behavior unchanged.
    pub async fn acquire_or_self_suspend(
        &self,
        requester: ProcessId,
        need: u32,
        holds_reclaimable: bool,
    ) -> Result<Acquired, ContentionError> {
        let (_, total) = self.backend.pool_stats();
        if need > total {
            return Err(ContentionError::Impossible { need, total });
        }
        // Victims this request already tried that yielded nothing NOW
        // (grace-deferred or unsupported) — skipped until the world changes.
        let mut tried: HashSet<ProcessId> = HashSet::new();
        // This call's parked FIFO entry, if any: (token, notify). The entry
        // keeps its queue slot across lost wake races and is removed on exit.
        let mut parked: Option<(u64, Arc<Notify>)> = None;
        // #19: first instant this call became CONTINUOUSLY unsatisfiable as the
        // FCFS-oldest keystone (no victim, `free < need`). Reset whenever the
        // condition clears; on `exhaustion_ms` of continuous quiet ⇒ fail loud.
        let mut exhausted_since: Option<Instant> = None;

        let result = loop {
            let (free, _) = self.backend.pool_stats();
            if free >= need {
                break Ok(Acquired::Retry);
            }

            // Pick the youngest running peer as victim (FCFS), reserving it
            // as `Suspending` under the lock so concurrent acquires don't
            // double-suspend it. The backend call happens OUTSIDE the lock.
            let victim = {
                let mut inner = self.inner.lock().unwrap();
                // FIX (fleet=24/8 deadlock, alpha's mechanism): a lane parked
                // inside `acquire` stays `Running` but is stuck PAST its
                // execute-entry prologue — it can NEVER comply with a park
                // request. Picking it poisons it (`ParkRequested`, excluded
                // from re-pick, pages stranded). Exclude any pid with an
                // active waiter entry from the victim pick.
                let victim = inner
                    .procs
                    .iter()
                    .filter(|(p, s)| {
                        **p != requester
                            && s.state == ProcState::Running
                            && !tried.contains(*p)
                            && !inner.waiters.iter().any(|w| w.pid == **p)
                    })
                    .max_by_key(|(_, s)| s.spawn_seq)
                    .map(|(p, _)| *p);
                if let Some(v) = victim {
                    inner.procs.get_mut(&v).unwrap().state = ProcState::Suspending;
                }
                victim
            };

            match victim {
                Some(v) => {
                    // A victim appeared ⇒ the exhaustion condition is not met;
                    // reset the keystone deadline (reset-on-change, Q3).
                    exhausted_since = None;
                    match self.backend.suspend(v) {
                        SuspendOutcome::Suspended { freed_now } => {
                            {
                                let mut inner = self.inner.lock().unwrap();
                                if let Some(p) = inner.procs.get_mut(&v) {
                                    p.state = ProcState::Suspended;
                                    p.suspended_need = freed_now;
                                    p.suspended_at = Some(Instant::now());
                                    inner.restore_queue.push_back(v);
                                    self.stats
                                        .suspends
                                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                }
                            }
                            // The victim's blocks may also satisfy PARKED waiters,
                            // not just this requester — wake the FIFO prefix (a
                            // synchronous-suspend backend has no report_suspended
                            // drain to do it for us).
                            self.wake_waiters();
                            // Loop: re-check free (another requester may race the
                            // freed blocks; we then evict further or park).
                        }
                        SuspendOutcome::Requested => {
                            // Self-suspend protocol: the victim saves its own
                            // state at its next host-call boundary; its blocks
                            // arrive via `report_suspended` → drain. Skip it for
                            // this pass.
                            let mut inner = self.inner.lock().unwrap();
                            if let Some(p) = inner.procs.get_mut(&v) {
                                p.state = ProcState::ParkRequested;
                            }
                            tried.insert(v);
                        }
                        SuspendOutcome::DeferredByGrace | SuspendOutcome::Unsupported => {
                            let mut inner = self.inner.lock().unwrap();
                            if let Some(p) = inner.procs.get_mut(&v) {
                                p.state = ProcState::Running;
                            }
                            tried.insert(v);
                        }
                    }
                }
                None => {
                    // Step 6 (prior design): no victim can yield — if the
                    // requester itself holds reclaimable pages, tell the
                    // caller to SELF-SUSPEND instead of parking; a parked
                    // page-holder can neither comply with a park request
                    // (it's past its prologue) nor free its pages.
                    //
                    // FCFS COMPLETION KEYSTONE (alpha's livelock find): the
                    // OLDEST registered process must NEVER self-yield — the
                    // victim-pick already protects it as a victim, but an
                    // unguarded SelfSuspendFirst let it yield its own pages,
                    // breaking "the oldest first-comer's progress is
                    // protected" — with everyone yield-restoring and nobody
                    // finishing (the ~8-cycles/lane churn). The oldest PARKS
                    // instead; every younger peer still self-yields, so frees
                    // flow to the oldest (FIFO prefix), it completes, and the
                    // next-oldest inherits the protection — a completion
                    // chain instead of a livelock.
                    if holds_reclaimable && !self.is_fcfs_oldest(requester) {
                        return Ok(Acquired::SelfSuspendFirst);
                    }
                    // #19 EXHAUSTION endgame: we reach here as either (a) a
                    // page-less waiter, or (b) the FCFS-oldest keystone (which
                    // never SelfSuspendFirsts). For the keystone with no victim
                    // and `free < need`, NO orchestrator action can satisfy it —
                    // there is nothing to suspend, and restoring a suspended peer
                    // only CONSUMES `free`. If that persists for `exhaustion_ms`
                    // of continuous quiet, the guest has grown its context past
                    // the pool: fail LOUD (one self-triaging log) rather than
                    // wedge silently. Non-keystone waiters clear the clock — the
                    // keystone's eventual yield/failure cascades to satisfy them.
                    if self.is_fcfs_oldest(requester) {
                        let since = *exhausted_since.get_or_insert_with(Instant::now);
                        if since.elapsed() >= Duration::from_millis(self.exhaustion_ms) {
                            tracing::error!(
                                pid = %requester,
                                need,
                                free,
                                total,
                                resident = total.saturating_sub(free),
                                "KV pool exhausted (#19): FCFS-oldest lane's request \
                                 unsatisfiable for {}ms — a non-terminating guest grew \
                                 its context past the pool; failing loud, not wedging",
                                self.exhaustion_ms,
                            );
                            break Err(ContentionError::Exhausted { need, free, total });
                        }
                    } else {
                        exhausted_since = None;
                    }
                    // No victim: the requester is the youngest (or peers are
                    // pinned/suspended). Park FIFO (keeping any existing slot)
                    // and wait for a free event. Clear any stale park request
                    // on OURSELVES first — we cannot comply while parked, and
                    // a stranded `ParkRequested` would poison the state
                    // machine (self-heals otherwise only at the next execute
                    // entry).
                    let notify = {
                        let mut inner = self.inner.lock().unwrap();
                        if let Some(p) = inner.procs.get_mut(&requester) {
                            if p.state == ProcState::ParkRequested {
                                p.state = ProcState::Running;
                            }
                        }
                        match &parked {
                            Some((_, n)) => n.clone(),
                            None => {
                                let token = inner.next_token;
                                inner.next_token += 1;
                                let notify = Arc::new(Notify::new());
                                inner.waiters.push_back(Waiter {
                                    token,
                                    pid: requester,
                                    need,
                                    notify: notify.clone(),
                                });
                                parked = Some((token, notify.clone()));
                                self.stats
                                    .waiters_parked
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                drop(inner);
                                // A×B: a PLAIN-parked waiter must also LEAVE
                                // the wave (first park only). Without this it
                                // accrues wave misses while blocked here and
                                // the miss-limit TERMINATES a live lane (the
                                // BAR-2 "unresponsive pipeline" demote of a
                                // preempt-cycling carrier lane). Join is
                                // emitted on acquire exit.
                                crate::inference::scheduler::notify_pipeline_leave(
                                    requester,
                                    crate::inference::scheduler::LeaveKind::Suspend,
                                );
                                notify
                            }
                        }
                    };
                    // Poll-tick the park: the keystone must re-check its
                    // exhaustion deadline even when NO free event ever comes (the
                    // wedge has none). A real `notify` wins the race normally.
                    let poll = Duration::from_millis((self.exhaustion_ms / 4).clamp(20, 1000));
                    tokio::select! {
                        _ = notify.notified() => {}
                        _ = tokio::time::sleep(poll) => {}
                    }
                    // The world changed; previously-deferred victims may have
                    // drained their grace period.
                    tried.clear();
                }
            }
        };

        // Leave the FIFO (satisfied) and pass the wake along: waiters behind
        // us may also fit. Restores are NOT evaluated here — our blocks are
        // about to be consumed by the caller's retry; only a real free event
        // (`on_blocks_freed`) feeds the restore phase.
        if let Some((token, _)) = parked {
            let mut inner = self.inner.lock().unwrap();
            inner.waiters.retain(|w| w.token != token);
            drop(inner);
            if result.is_ok() {
                self.stats
                    .waiters_woken
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // A×B rejoin for the plain-park Leave above. On the
            // `SelfSuspendFirst` exit the caller immediately re-Leaves via
            // `report_suspended` — a benign Join/Leave pair; on an Err exit
            // the lane fails loud and terminate re-tombstones it.
            crate::inference::scheduler::notify_pipeline_join(requester);
            self.wake_waiters();
        }
        result
    }

    /// Phase 1 — waiters. Notify the FIFO prefix whose cumulative need fits
    /// the free pool. Entries KEEP their queue slot: a woken waiter that wins
    /// its re-check removes itself (and passes the wake along); one that
    /// loses a race stays parked in position. While ANY waiter is parked,
    /// the restore phase is skipped — alloc waiters have strict priority over
    /// restores, and their pending blocks are never given to a restore.
    fn wake_waiters(&self) {
        let (free, _total) = self.backend.pool_stats();
        let inner = self.inner.lock().unwrap();
        let mut budget = free;
        for w in &inner.waiters {
            if w.need <= budget {
                budget -= w.need;
                w.notify.notify_one();
            } else {
                break;
            }
        }
    }

    /// Central drain (prior `drain_queues`): phase 1 wakes FIFO waiters whose
    /// needs prefix-fit the free pool (strict priority); phase 2 restores the
    /// oldest-suspended processes while no waiter is parked, utilization is
    /// at/below the pause threshold, and the restore fits without evicting.
    fn drain(&self) {
        self.wake_waiters();

        // Phase 2 — restores (only when no waiter is parked).
        loop {
            let (free, total_now) = self.backend.pool_stats();
            let pid = {
                let mut inner = self.inner.lock().unwrap();
                if !inner.waiters.is_empty() {
                    return;
                }
                let total_f = total_now.max(1) as f64;
                let utilization = (total_now.saturating_sub(free)) as f64 / total_f;
                let Some(&pid) = inner.restore_queue.front() else {
                    return;
                };
                if utilization > self.restore_pause_at_utilization {
                    // Anti-thrash pause — WITH AGING: on a small pool the
                    // utilization can sit permanently above the threshold
                    // (e.g. 7/8 = 0.875 > 0.85), which would starve the
                    // FIFO-head restore forever. Once the head has waited
                    // past `PIE_KV_RESTORE_AGING_MS`, it proceeds anyway if
                    // it fits (still never evicts).
                    let aged = inner
                        .procs
                        .get(&pid)
                        .and_then(|p| p.suspended_at)
                        .is_some_and(|t| t.elapsed() >= Duration::from_millis(restore_aging_ms()));
                    if !aged {
                        return; // wait for utilization to drop (or the head to age)
                    }
                }
                let need = match inner.procs.get(&pid) {
                    Some(p) => p.suspended_need,
                    None => {
                        // Stale entry (unregistered) — lazy delete.
                        inner.restore_queue.pop_front();
                        continue;
                    }
                };
                if free < need {
                    return; // restore NEVER evicts (prior can_restore)
                }
                inner.restore_queue.pop_front();
                // Self-suspended (parked) process: release it directly — the
                // victim's own task re-materializes its state on wake (and
                // re-reports + re-parks if that hits contention). No backend
                // call; the wake IS the restore hand-off.
                if let Some(notify) = inner.parked.get(&pid).cloned() {
                    inner.procs.get_mut(&pid).unwrap().state = ProcState::Running;
                    inner.procs.get_mut(&pid).unwrap().suspended_need = 0;
                    drop(inner);
                    self.stats
                        .restores
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // A×B rejoin (Leave's other half): decay the scheduler
                    // tombstone BEFORE the victim wakes, so its next request
                    // re-enters the wait-set tracked (the sustained-f24
                    // balanced-freeze was this Join missing — every suspended
                    // lane stayed untracked forever and the wave set decayed
                    // to empty).
                    crate::inference::scheduler::notify_pipeline_join(pid);
                    notify.notify_one();
                    continue;
                }
                inner.procs.get_mut(&pid).unwrap().state = ProcState::Restoring;
                pid
            };

            match self.backend.restore(pid) {
                Ok(()) => {
                    {
                        let mut inner = self.inner.lock().unwrap();
                        if let Some(p) = inner.procs.get_mut(&pid) {
                            p.state = ProcState::Running;
                            p.suspended_need = 0;
                        }
                    }
                    self.stats
                        .restores
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // A×B rejoin — see the parked-release path above.
                    crate::inference::scheduler::notify_pipeline_join(pid);
                    // Loop: more restores may fit.
                }
                Err(e) => {
                    tracing::warn!(pid = %pid, "restore failed, re-queued: {e:#}");
                    let mut inner = self.inner.lock().unwrap();
                    if let Some(p) = inner.procs.get_mut(&pid) {
                        p.state = ProcState::Suspended;
                        inner.restore_queue.push_back(pid);
                    }
                    return; // don't hot-loop a failing restore; next event retries
                }
            }
        }
    }
}

// =============================================================================
// v1 backend — real pool stats over the unified arena; reclaim binds in M-B2
// =============================================================================

/// v1 [`ReclaimBackend`] over the unified arena's KV pool. Pool stats are
/// real, so preempt mode gives FIFO wait-for-free + restore-queue plumbing
/// end-to-end; `suspend` reports [`SuspendOutcome::Unsupported`] until the
/// working-set state-save wrappers bind here (M-B2, alpha's
/// `classify_for_suspend`/`suspend_pages_warm`/`restore_pages_warm`), which
/// turns the victim loop live.
pub struct ArenaPoolBackend {
    model_idx: usize,
    driver_idx: usize,
}

impl ArenaPoolBackend {
    pub fn new(model_idx: usize, driver_idx: usize) -> Self {
        Self {
            model_idx,
            driver_idx,
        }
    }
}

impl ReclaimBackend for ArenaPoolBackend {
    fn pool_stats(&self) -> (u32, u32) {
        use crate::arena::ArenaKind;
        let arena = crate::arena::registry::get(self.model_idx, self.driver_idx);
        let a = arena.lock().unwrap();
        let free = a.available(ArenaKind::KvPage);
        let total = free + a.used(ArenaKind::KvPage);
        (free, total)
    }

    fn suspend(&self, _victim: ProcessId) -> SuspendOutcome {
        // Passive v1: no reclaim from running processes; waiters ride the
        // natural frees (terminate-reclaim, finalize). Proven e2e (M-AB ②③).
        SuspendOutcome::Unsupported
    }

    fn restore(&self, _pid: ProcessId) -> anyhow::Result<()> {
        // Nothing is ever suspended by this backend.
        Ok(())
    }
}

/// v2 backend — ACTIVE FCFS preempt via the SELF-SUSPEND protocol
/// (B-refined): `suspend(victim)` returns [`SuspendOutcome::Requested`],
/// posting the park request; the victim saves its OWN state at its next
/// host-call boundary (`should_park` → the working-set suspend wrappers →
/// `report_suspended`) with its own ResourceTable access — no cross-process
/// working-set reach exists, by design. Restore of parked victims is the
/// orchestrator's direct release path (`park_until_restored` wake), so
/// `restore()` here never runs for them. Selected by
/// `PIE_KV_PREEMPT_ACTIVE=1` on top of `PIE_KV_CONTENTION=preempt`;
/// engagement shows as `stats().suspends/restores > 0` (the pre-wired v2
/// e2e assertion). Passive (`ArenaPoolBackend`) remains the default.
pub struct SelfSuspendBackend {
    pool: ArenaPoolBackend,
}

impl SelfSuspendBackend {
    pub fn new(model_idx: usize, driver_idx: usize) -> Self {
        Self {
            pool: ArenaPoolBackend::new(model_idx, driver_idx),
        }
    }
}

impl ReclaimBackend for SelfSuspendBackend {
    fn pool_stats(&self) -> (u32, u32) {
        self.pool.pool_stats()
    }

    fn suspend(&self, _victim: ProcessId) -> SuspendOutcome {
        // Post the park request; the orchestrator marks the victim
        // `ParkRequested` and its own task complies at the next
        // execute_impl prologue (`should_park` check — alpha's seam).
        SuspendOutcome::Requested
    }

    fn restore(&self, _pid: ProcessId) -> anyhow::Result<()> {
        // Parked victims are released by the orchestrator directly.
        Ok(())
    }
}

/// Restore-pause aging window (ms): 250. A FIFO-head restore older than this
/// proceeds even above the utilization pause (if it fits) — starvation-proofing
/// on small pools where utilization never drops below the threshold.
pub fn restore_aging_ms() -> u64 {
    250
}

/// #19 exhaustion deadline (ms): 10000 = 10s. Bounds how long the FCFS-oldest
/// keystone may stay continuously unsatisfiable before `acquire` fails LOUD
/// instead of wedging — turning a ~300s silent hang into a 10s named error.
/// Tests override via [`ContentionOrchestrator::with_exhaustion_ms`].
fn exhaustion_ms_from_env() -> u64 {
    10_000
}

/// Whether the ACTIVE self-suspend backend is selected (v2), read once from
/// `PIE_KV_PREEMPT_ACTIVE` (default off ⇒ the proven passive v1 backend).
pub fn preempt_active() -> bool {
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| {
        std::env::var("PIE_KV_PREEMPT_ACTIVE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

// =============================================================================
// Global instance (wired by bootstrap when PIE_KV_CONTENTION=preempt)
// =============================================================================

static CONTENTION: OnceLock<ContentionOrchestrator> = OnceLock::new();

/// Install the global orchestrator (bootstrap, once, only in preempt mode).
pub fn init_contention(orchestrator: ContentionOrchestrator) {
    let _ = CONTENTION.set(orchestrator);
}

/// The global orchestrator, if preempt mode is wired.
pub fn contention() -> Option<&'static ContentionOrchestrator> {
    CONTENTION.get()
}

/// Scheduler hook for payload-free fire retirement. No-op unless preempt mode
/// is wired (one `OnceLock` load on the hot path).
pub fn notify_fire_retired() {
    if let Some(o) = CONTENTION.get() {
        o.on_fire_retired();
    }
}
