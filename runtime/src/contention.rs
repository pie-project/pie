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

    /// A fire's responses were dispatched (scheduler response path) — wake any
    /// lane gated on draining its own retired fires.
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
                        .is_some_and(|t| {
                            t.elapsed() >= Duration::from_millis(restore_aging_ms())
                        });
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

/// Restore-pause aging window (ms), read once from `PIE_KV_RESTORE_AGING_MS`
/// (default 250): a FIFO-head restore older than this proceeds even above the
/// utilization pause (if it fits) — starvation-proofing on small pools where
/// utilization never drops below the threshold.
pub fn restore_aging_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("PIE_KV_RESTORE_AGING_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&m| m >= 1)
            .unwrap_or(250)
    })
}

/// #19 exhaustion deadline (ms), read from `PIE_KV_EXHAUSTION_MS` at
/// orchestrator construction (default 10000 = 10s). Bounds how long the
/// FCFS-oldest keystone may stay continuously unsatisfiable before `acquire`
/// fails LOUD instead of wedging — turning a ~300s silent hang into a 10s named
/// error. Read per-construct (not `OnceLock`-cached) so tests override via
/// [`ContentionOrchestrator::with_exhaustion_ms`].
fn exhaustion_ms_from_env() -> u64 {
    std::env::var("PIE_KV_EXHAUSTION_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&m| m >= 1)
        .unwrap_or(10_000)
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

/// Scheduler hook: a fire's responses were dispatched. No-op unless preempt
/// mode is wired (one `OnceLock` load — safe on the hot dispatch path).
pub fn notify_fire_retired() {
    if let Some(o) = CONTENTION.get() {
        o.on_fire_retired();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    /// Mock pool + per-process reclaim table. `free` is the shared counter;
    /// suspend/restore move each process's `reclaim` blocks in and out.
    struct MockBackend {
        free: AtomicU32,
        total: u32,
        reclaim: Mutex<HashMap<ProcessId, u32>>,
        grace: Mutex<HashSet<ProcessId>>,
        suspend_log: Mutex<Vec<ProcessId>>,
        restore_log: Mutex<Vec<ProcessId>>,
        fail_restores: AtomicU32,
    }

    impl MockBackend {
        fn new(free: u32, total: u32) -> Arc<Self> {
            Arc::new(Self {
                free: AtomicU32::new(free),
                total,
                reclaim: Mutex::new(HashMap::new()),
                grace: Mutex::new(HashSet::new()),
                suspend_log: Mutex::new(Vec::new()),
                restore_log: Mutex::new(Vec::new()),
                fail_restores: AtomicU32::new(0),
            })
        }

        fn set_reclaim(&self, pid: ProcessId, blocks: u32) {
            self.reclaim.lock().unwrap().insert(pid, blocks);
        }

        fn set_grace(&self, pid: ProcessId, deferred: bool) {
            let mut g = self.grace.lock().unwrap();
            if deferred {
                g.insert(pid);
            } else {
                g.remove(&pid);
            }
        }

        fn add_free(&self, n: u32) {
            self.free.fetch_add(n, Ordering::SeqCst);
        }

        fn take_free(&self, n: u32) {
            self.free.fetch_sub(n, Ordering::SeqCst);
        }
    }

    struct ArcBackend(Arc<MockBackend>);

    impl ReclaimBackend for ArcBackend {
        fn pool_stats(&self) -> (u32, u32) {
            (self.0.free.load(Ordering::SeqCst), self.0.total)
        }
        fn suspend(&self, victim: ProcessId) -> SuspendOutcome {
            if self.0.grace.lock().unwrap().contains(&victim) {
                return SuspendOutcome::DeferredByGrace;
            }
            match self.0.reclaim.lock().unwrap().get(&victim) {
                Some(&n) if n > 0 => {
                    self.0.free.fetch_add(n, Ordering::SeqCst);
                    self.0.suspend_log.lock().unwrap().push(victim);
                    SuspendOutcome::Suspended { freed_now: n }
                }
                _ => SuspendOutcome::Unsupported,
            }
        }
        fn restore(&self, pid: ProcessId) -> anyhow::Result<()> {
            if self.0.fail_restores.load(Ordering::SeqCst) > 0 {
                self.0.fail_restores.fetch_sub(1, Ordering::SeqCst);
                anyhow::bail!("injected restore failure");
            }
            let n = *self.0.reclaim.lock().unwrap().get(&pid).unwrap_or(&0);
            let free = self.0.free.load(Ordering::SeqCst);
            anyhow::ensure!(free >= n, "restore would evict (free {free} < {n})");
            self.0.free.fetch_sub(n, Ordering::SeqCst);
            self.0.restore_log.lock().unwrap().push(pid);
            Ok(())
        }
    }

    fn orch(mock: &Arc<MockBackend>, pause: f64) -> ContentionOrchestrator {
        ContentionOrchestrator::new(Box::new(ArcBackend(mock.clone())), pause)
    }

    fn pid() -> ProcessId {
        ProcessId::new_v4()
    }

    /// #19 — the EXHAUSTION endgame. A non-terminating FCFS-oldest survivor that
    /// grows past the whole pool wedges the fleet: every peer suspended, the
    /// survivor parked (keystone ⇒ never self-yields), restore gated by its own
    /// waiter entry. With NO endgame this hangs silently to the ~295s timeout;
    /// the fix fails LOUD — after `exhaustion_ms` of continuous unsatisfiability
    /// the survivor's `acquire` returns `Err(Exhausted)` (→ OutOfBlocks to the
    /// guest), never a silent wedge. (Reset-on-change guards transients: the
    /// clock only fires on CONTINUOUS unsatisfiable quiet.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn exhaustion_endgame_survivor_fails_loud_at_deadline() {
        // total 16, free 0 (the survivor already holds the whole pool). Short
        // deadline so the test is fast; production default is 10s.
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0).with_exhaustion_ms(150));
        let hog = pid(); // registered first ⇒ FCFS-oldest ⇒ the survivor keystone
        o.register(hog);

        // A fleet of peers, ALL already suspended (each was preempted earlier to
        // feed the survivor): they sit in `restore_queue` needing blocks back.
        for _ in 0..6 {
            let p = pid();
            o.register(p);
            let mut inner = o.inner.lock().unwrap();
            let pr = inner.procs.get_mut(&p).unwrap();
            pr.state = ProcState::Suspended;
            pr.suspended_need = 2;
            inner.restore_queue.push_back(p);
        }

        // The survivor needs 4 more, `free=0`, no running victim, and it is the
        // keystone (parks, never SelfSuspendFirst) — exactly the real
        // execute_impl path (`holds_reclaimable=true`). WITHOUT #19 this wedges;
        // WITH it, `acquire` must fail LOUD at the deadline (the 2s timeout is
        // the "never wedges" guard — the fix returns in ~150ms).
        let r = tokio::time::timeout(
            Duration::from_secs(2),
            o.acquire_or_self_suspend(hog, 4, true),
        )
        .await
        .expect("#19: survivor must FAIL LOUD at the deadline, never silently wedge");
        assert_eq!(
            r,
            Err(ContentionError::Exhausted { need: 4, free: 0, total: 16 }),
            "the over-capacity survivor gets a loud Exhausted (→ OutOfBlocks)"
        );
    }

    #[tokio::test]
    async fn fast_path_when_pool_has_room() {
        let mock = MockBackend::new(8, 16);
        let o = orch(&mock, 0.85);
        let a = pid();
        o.register(a);
        o.acquire(a, 4).await.unwrap();
        assert!(mock.suspend_log.lock().unwrap().is_empty(), "no preempt");
    }

    #[tokio::test]
    async fn impossible_request_fails_loud() {
        let mock = MockBackend::new(8, 16);
        let o = orch(&mock, 0.85);
        let a = pid();
        o.register(a);
        assert_eq!(
            o.acquire(a, 17).await,
            Err(ContentionError::Impossible { need: 17, total: 16 })
        );
    }

    /// FCFS: the YOUNGEST running peer is preempted first, then the next
    /// youngest, until the request fits. The oldest keeps its residency.
    #[tokio::test]
    async fn victims_are_youngest_first() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 0.85);
        let (old, mid, young) = (pid(), pid(), pid());
        o.register(old);
        o.register(mid);
        o.register(young);
        mock.set_reclaim(mid, 4);
        mock.set_reclaim(young, 4);

        o.acquire(old, 8).await.unwrap();
        assert_eq!(
            *mock.suspend_log.lock().unwrap(),
            vec![young, mid],
            "youngest first, then next-youngest"
        );
        assert!(o.is_suspended(young) && o.is_suspended(mid));
    }

    /// The requester itself is never a victim: a youngest requester with no
    /// peers parks and is woken by a free event.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_victim_parks_until_blocks_free() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 0.85));
        let a = pid();
        o.register(a);

        let o2 = o.clone();
        let task = tokio::spawn(async move { o2.acquire(a, 4).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished(), "parked while the pool is empty");

        mock.add_free(4);
        o.on_blocks_freed();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("woken by the free event")
            .unwrap()
            .unwrap();
    }

    /// A grace-deferred victim is skipped (stays running) and the next
    /// victim is taken instead.
    #[tokio::test]
    async fn grace_deferred_victim_skipped() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 0.85);
        let (old, mid, young) = (pid(), pid(), pid());
        o.register(old);
        o.register(mid);
        o.register(young);
        mock.set_reclaim(mid, 4);
        mock.set_reclaim(young, 4);
        mock.set_grace(young, true); // youngest has an in-flight pass

        o.acquire(old, 4).await.unwrap();
        assert_eq!(*mock.suspend_log.lock().unwrap(), vec![mid]);
        assert!(!o.is_suspended(young), "grace-deferred victim stays running");
    }

    /// Restore-on-free: FIFO oldest-suspended-first, only under the
    /// utilization pause threshold, and never evicting (free >= need).
    #[tokio::test]
    async fn restores_fifo_under_thrash_guard() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 0.85);
        let (old, mid, young) = (pid(), pid(), pid());
        o.register(old);
        o.register(mid);
        o.register(young);
        mock.set_reclaim(mid, 4);
        mock.set_reclaim(young, 4);
        o.acquire(old, 8).await.unwrap(); // suspends young then mid
        mock.take_free(8); // the requester consumed its allocation

        // Utilization 16/16 > 0.85 → restores paused.
        o.on_blocks_freed();
        assert!(mock.restore_log.lock().unwrap().is_empty(), "paused");

        // The requester frees a lot (e.g. finished): 12 free, util 4/16.
        mock.add_free(12);
        o.on_blocks_freed();
        // Both restore, FIFO: young was suspended first.
        assert_eq!(*mock.restore_log.lock().unwrap(), vec![young, mid]);
        assert!(!o.is_suspended(young) && !o.is_suspended(mid));
    }

    /// A restore that would not fit is held (restore never evicts).
    #[tokio::test]
    async fn restore_held_until_it_fits() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 1.0); // disable the utilization pause for this test
        let (old, young) = (pid(), pid());
        o.register(old);
        o.register(young);
        mock.set_reclaim(young, 8);
        o.acquire(old, 8).await.unwrap();
        mock.take_free(8);

        mock.add_free(4); // not enough for young's 8
        o.on_blocks_freed();
        assert!(mock.restore_log.lock().unwrap().is_empty());

        mock.add_free(4); // now 8 free
        o.on_blocks_freed();
        assert_eq!(*mock.restore_log.lock().unwrap(), vec![young]);
    }

    /// Parked waiters have strict priority over restores.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn waiters_outrank_restores() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let (old, young, newest) = (pid(), pid(), pid());
        o.register(old);
        o.register(young);
        mock.set_reclaim(young, 4);
        o.acquire(old, 4).await.unwrap(); // young suspended
        mock.take_free(4);

        // newest parks (no victim: old has nothing reclaimable).
        o.register(newest);
        let o2 = o.clone();
        let task = tokio::spawn(async move { o2.acquire(newest, 4).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished());

        mock.add_free(4);
        o.on_blocks_freed();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("waiter woken first")
            .unwrap()
            .unwrap();
        assert!(
            mock.restore_log.lock().unwrap().is_empty(),
            "the waiter outranks the restore; young stays suspended"
        );
    }

    /// A failing restore is re-queued (never dropped) and retried on the
    /// next free event.
    #[tokio::test]
    async fn failed_restore_requeues() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 1.0);
        let (old, young) = (pid(), pid());
        o.register(old);
        o.register(young);
        mock.set_reclaim(young, 4);
        o.acquire(old, 4).await.unwrap();
        mock.take_free(4);
        mock.add_free(4);

        mock.fail_restores.store(1, Ordering::SeqCst);
        o.on_blocks_freed();
        assert!(mock.restore_log.lock().unwrap().is_empty());
        assert!(o.is_suspended(young), "still suspended after the failure");

        o.on_blocks_freed(); // next event retries and succeeds
        assert_eq!(*mock.restore_log.lock().unwrap(), vec![young]);
    }

    /// The full self-suspend handshake (B-refined): a `Requested` victim is
    /// skipped by the victim loop; when it complies (report_suspended) its
    /// blocks wake the parked requester; a later free releases the parked
    /// victim FCFS and `park_until_restored` returns.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn self_suspend_handshake_roundtrip() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let (old, young) = (pid(), pid());
        o.register(old);
        o.register(young);
        // The mock backend yields nothing for young (Unsupported ≈ a
        // Requested victim that hasn't complied yet) → the requester parks;
        // the victim-side protocol is then driven directly, exactly as the
        // execute_impl prologue will drive it.
        let o2 = o.clone();
        let task = tokio::spawn(async move { o2.acquire(old, 4).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished(), "requester parked (no victim yields)");

        // ...meanwhile the victim-side protocol runs (as the execute_impl
        // prologue will): report_suspended frees blocks + drains.
        mock.add_free(4); // the victim's own state-save freed 4 blocks
        o.report_suspended(young, 4);
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("requester woken by the victim's report")
            .unwrap()
            .unwrap();
        assert!(o.is_suspended(young));

        // The victim parks; the requester consumes its blocks.
        mock.take_free(4);
        let o3 = o.clone();
        let park = tokio::spawn(async move { o3.park_until_restored(young).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!park.is_finished(), "victim parked while suspended");

        // Blocks free again → restore phase releases the parked victim.
        mock.add_free(4);
        o.on_blocks_freed();
        tokio::time::timeout(Duration::from_secs(1), park)
            .await
            .expect("victim released FCFS")
            .unwrap();
        assert!(!o.is_suspended(young), "released back to Running");
    }

    /// A release that lands BEFORE the victim parks is not lost: the Notify
    /// permit is stored at report_suspended's entry, so a late
    /// `park_until_restored` returns immediately.
    #[tokio::test]
    async fn early_release_is_not_lost() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 1.0);
        let a = pid();
        o.register(a);
        mock.add_free(4);
        o.report_suspended(a, 4); // drain releases it immediately (4 free ≥ 4)
        // The release already happened; parking now must not hang.
        tokio::time::timeout(Duration::from_millis(200), o.park_until_restored(a))
            .await
            .expect("early release stored; park returns immediately");
        assert!(!o.is_suspended(a));
    }

    /// `should_park` flips true only after a backend `Requested` outcome, and
    /// unregister wakes a parked task for teardown.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unregister_wakes_parked_victim() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let a = pid();
        o.register(a);
        assert!(!o.should_park(a), "no park request initially");
        mock.add_free(2); // not enough for the eventual restore (need 4)
        o.report_suspended(a, 4);
        let o2 = o.clone();
        let park = tokio::spawn(async move { o2.park_until_restored(a).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!park.is_finished());

        o.unregister(a); // teardown while parked
        tokio::time::timeout(Duration::from_secs(1), park)
            .await
            .expect("teardown wakes the parked task")
            .unwrap();
    }

    /// The v2 ACTIVE path end-to-end at the orchestrator level: a `Requested`
    /// backend (SelfSuspendBackend semantics) marks the victim ParkRequested
    /// (`should_park` flips true); the requester parks; the victim complies
    /// (report_suspended) → the requester wakes off the victim's freed
    /// blocks; suspends/restores counters arm (the v2 e2e assertion).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn active_backend_posts_park_request_and_victim_compliance_unblocks() {
        struct RequestingBackend(Arc<MockBackend>);
        impl ReclaimBackend for RequestingBackend {
            fn pool_stats(&self) -> (u32, u32) {
                (self.0.free.load(Ordering::SeqCst), self.0.total)
            }
            fn suspend(&self, _victim: ProcessId) -> SuspendOutcome {
                SuspendOutcome::Requested
            }
            fn restore(&self, _pid: ProcessId) -> anyhow::Result<()> {
                Ok(())
            }
        }

        let mock = MockBackend::new(0, 16);
        let o = Arc::new(ContentionOrchestrator::new(
            Box::new(RequestingBackend(mock.clone())),
            1.0,
        ));
        let (old, young) = (pid(), pid());
        o.register(old);
        o.register(young);

        let o2 = o.clone();
        let task = tokio::spawn(async move { o2.acquire(old, 4).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!task.is_finished(), "requester parked awaiting compliance");
        assert!(
            o.should_park(young),
            "the youngest victim has a pending park request"
        );

        // The victim complies at its host-call boundary (alpha's prologue):
        // saves its state (frees 4) + reports.
        mock.add_free(4);
        o.report_suspended(young, 4);
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("requester woken by the victim's compliance")
            .unwrap()
            .unwrap();
        assert!(o.is_suspended(young));
        assert_eq!(o.stats().suspends.load(Ordering::SeqCst), 1);

        // Requester consumes; later frees release the victim FCFS.
        mock.take_free(4);
        let o3 = o.clone();
        let park = tokio::spawn(async move { o3.park_until_restored(young).await });
        mock.add_free(4);
        o.on_blocks_freed();
        tokio::time::timeout(Duration::from_secs(1), park)
            .await
            .expect("victim released")
            .unwrap();
        assert_eq!(o.stats().restores.load(Ordering::SeqCst), 1);
    }

    /// The fleet=24/8 deadlock mechanism (alpha's root cause): a lane parked
    /// inside `acquire` stays Running but can never comply with a park
    /// request — the victim pick must EXCLUDE it, falling through to the
    /// next candidate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn victim_pick_excludes_parked_in_acquire_lanes() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let (old_p, mid, young) = (pid(), pid(), pid());
        o.register(old_p);
        o.register(mid);
        o.register(young);
        mock.set_reclaim(mid, 4);
        mock.set_reclaim(young, 4);

        // `young` OOMs first and parks in acquire (no victim yields for it:
        // mid+old have reclaim but mock returns Unsupported for zero-entry…
        // give young no one: exhaust by marking both tried via grace).
        mock.set_grace(mid, true);
        let o2 = o.clone();
        let young_task = tokio::spawn(async move { o2.acquire(young, 4).await });
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(!young_task.is_finished(), "young parked in acquire");

        // Now `old` OOMs. Victim candidates by age: young (parked — MUST be
        // skipped) then mid (grace-cleared now → suspendable).
        mock.set_grace(mid, false);
        o.acquire(old_p, 4).await.unwrap();
        assert!(
            !o.is_suspended(young),
            "parked-in-acquire lane must never be marked/suspended"
        );
        assert!(o.is_suspended(mid), "the pick fell through to mid");

        // young wakes off mid's freed blocks.
        tokio::time::timeout(Duration::from_secs(1), young_task)
            .await
            .expect("young woken")
            .unwrap()
            .unwrap();
    }

    /// Step 6 (prior design): no victim + the requester holds reclaimable
    /// pages ⇒ `SelfSuspendFirst` (never park a page-holder) — for any
    /// requester that is NOT the FCFS-oldest (the oldest is the completion
    /// keystone and parks instead; see below).
    #[tokio::test]
    async fn no_victim_page_holder_gets_self_suspend_first() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let (elder, a) = (pid(), pid());
        o.register(elder); // an older peer exists: `a` is NOT the keystone
        o.register(a);
        // elder is suspended (not a victim candidate), so `a` finds no victim.
        mock.set_reclaim(elder, 0);
        {
            let mut inner = o.inner.lock().unwrap();
            inner.procs.get_mut(&elder).unwrap().state = ProcState::Suspended;
        }
        let r = o.acquire_or_self_suspend(a, 4, true).await.unwrap();
        assert_eq!(r, Acquired::SelfSuspendFirst);
        // Without pages, the same call parks (verified by non-completion).
        let o2 = Arc::new(orch(&mock, 1.0));
        o2.register(elder);
        o2.register(a);
        {
            let mut inner = o2.inner.lock().unwrap();
            inner.procs.get_mut(&elder).unwrap().state = ProcState::Suspended;
        }
        let o3 = o2.clone();
        let t = tokio::spawn(async move { o3.acquire_or_self_suspend(a, 4, false).await });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!t.is_finished(), "page-less requester parks as before");
        o2.unregister(a);
        let _ = tokio::time::timeout(Duration::from_secs(1), t).await;
    }

    /// FCFS completion keystone (alpha's livelock find): the OLDEST process
    /// never self-yields — even page-holding with no victim, it PARKS, so one
    /// lane always runs to completion and the fleet drains instead of
    /// churning suspend/restore forever.
    #[tokio::test]
    async fn fcfs_oldest_parks_instead_of_self_yielding() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let (oldest, young) = (pid(), pid());
        o.register(oldest);
        o.register(young);
        // young is suspended → no victim for the oldest's acquire.
        {
            let mut inner = o.inner.lock().unwrap();
            inner.procs.get_mut(&young).unwrap().state = ProcState::Suspended;
        }
        let o2 = o.clone();
        let t =
            tokio::spawn(async move { o2.acquire_or_self_suspend(oldest, 4, true).await });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !t.is_finished(),
            "oldest page-holder PARKS (never SelfSuspendFirst)"
        );
        // Blocks free → the parked oldest wakes and proceeds (Retry).
        mock.add_free(4);
        o.on_blocks_freed();
        let r = tokio::time::timeout(Duration::from_secs(1), t)
            .await
            .expect("oldest woken")
            .unwrap()
            .unwrap();
        assert_eq!(r, Acquired::Retry, "keystone proceeds on the next free");
    }

    /// Carrier ⋈ contention gate primitives: `contended()` tracks parked
    /// waiters, and the `fire_retired` signal wakes an enabled waiter (the
    /// enable-then-check-then-await pattern the drain gate relies on).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn contended_and_fire_retired_gate_primitives() {
        let mock = MockBackend::new(0, 16);
        let o = Arc::new(orch(&mock, 1.0));
        let a = pid();
        o.register(a);
        assert!(!o.contended(), "no waiter yet");

        // Park a page-less requester (no victim can yield) → contended.
        let o2 = o.clone();
        let t = tokio::spawn(async move { o2.acquire(a, 4).await });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(o.contended(), "parked waiter ⇒ contended");

        // fire_retired: an ENABLED waiter is woken even if the notify lands
        // between its state-check and its await (no lost wakeup).
        let notify = o.fire_retired();
        let notified = notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        o.on_fire_retired(); // lands "before" the await
        tokio::time::timeout(Duration::from_millis(200), notified)
            .await
            .expect("enabled waiter woken by retire notify");

        o.unregister(a);
        let _ = tokio::time::timeout(Duration::from_secs(1), t).await;
        assert!(!o.contended(), "unregister purged the waiter");
    }

    /// Restore-pause AGING: on a small pool pegged above the utilization
    /// threshold, the FIFO-head restore proceeds once aged (never starves).
    #[tokio::test]
    async fn restore_pause_ages_out_instead_of_starving() {
        let mock = MockBackend::new(0, 8);
        let o = orch(&mock, 0.4); // pause above 40% utilization
        let (p1, p2) = (pid(), pid());
        o.register(p1);
        o.register(p2);
        mock.set_reclaim(p2, 4);
        o.acquire(p1, 4).await.unwrap(); // p2 suspended, 4 freed
        mock.take_free(4); // p1 consumed its allocation → 0 free, 8/8 used
        mock.add_free(4); // p1 released half → 4 free, util 4/8 = 0.5 > 0.4

        // Paused: utilization above threshold, head not yet aged.
        o.on_blocks_freed();
        assert!(o.is_suspended(p2), "paused: util above threshold, not aged");

        // Past the aging window the head restores anyway (it fits: 4 ≥ 4).
        std::thread::sleep(std::time::Duration::from_millis(300)); // > 250ms default
        o.on_blocks_freed();
        assert!(!o.is_suspended(p2), "aged head restores despite the pause");
    }

    /// Unregister removes the process from the restore queue and drops its
    /// parked waiters.
    #[tokio::test]
    async fn unregister_purges_queues() {
        let mock = MockBackend::new(0, 16);
        let o = orch(&mock, 1.0);
        let (old, young) = (pid(), pid());
        o.register(old);
        o.register(young);
        mock.set_reclaim(young, 4);
        o.acquire(old, 4).await.unwrap();
        mock.take_free(4);

        o.unregister(young); // exits while suspended
        mock.add_free(8);
        o.on_blocks_freed();
        assert!(
            mock.restore_log.lock().unwrap().is_empty(),
            "unregistered process is never restored"
        );
    }
}
