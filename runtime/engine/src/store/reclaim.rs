//! The reclaim ladder — FCFS preempt/restore over the KV physical pool. This
//! module manages pages, so it lives in `store/` even though the fire path
//! is what triggers it. `ContentionOrchestrator`/`ContentionMode`/etc. keep
//! their working names; the file/module name (`store::reclaim`) is the
//! contract.
//!
//! In Gim's directive: KV-cache contention is handled by PREEMPT/RESTORE, not
//! admission. `max_concurrent_processes` is a large physical safety cap only;
//! when a forward's page allocation hits an exhausted pool
//! ([`KvStoreError::OutOfPages`](crate::store::kv::KvStoreError) — typed, and
//! raised only at forward preparation because `reserve` is logical), the fire
//! path's prepare seam routes here instead of surfacing an inferlet error:
//!
//! 0. **Idle-lease rung** — the ladder's first step (kv_refact.md Scheduler):
//!    drop cache-root leases nothing live reaches
//!    ([`ReclaimBackend::reclaim_idle`] →
//!    `KvStore::drop_unused_cache_leases`). Pure cache, no work lost; only
//!    if pressure remains does the victim loop run.
//! 1. **Victim loop** — FCFS-preempt: suspend the YOUNGEST running process
//!    (latest spawn order; the oldest first-comer's progress is protected)
//!    until the request fits. The physical state-save (D2H stash / drop-to-
//!    replay, grace/refcount/txn guards) is behind [`ReclaimBackend`] —
//!    alpha's working-set wrappers; this module owns only the ORCHESTRATION.
//! 2. **Unified grant order** — blocked allocations and suspended restores
//!    compete by original `submit_seq`. Grants reserve concrete device ids, so
//!    a racing caller cannot bypass an older entitlement.
//! 3. **Restore-on-free** — the oldest eligible unified entry proceeds. A
//!    non-fitting restore holds its slot and never evicts; a utilization-paused
//!    restore voluntarily yields until its anti-thrash pause ages out.
//! 4. **Exhaustion endgame (#19)** — a non-terminating guest can grow its
//!    context past the whole pool; the FCFS-oldest keystone then parks
//!    unsatisfiable forever (no victim; restore only consumes free). After
//!    `PIE_KV_EXHAUSTION_MS` of continuous quiet, `acquire` fails LOUD with
//!    [`ContentionError::Exhausted`] (→ `OutOfPages` to the guest) instead of
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

/// Process identity the orchestrator tracks (FCFS clock key, victim pick,
/// wait-queue token). Kept as the leaf `uuid::Uuid` representation so the
/// store stays below the guest runtime in the layering.
pub type ProcessId = uuid::Uuid;

/// Whether a pipeline-leave was a full termination or a reclaim-driven
/// suspend. Kept separate from the scheduler's `LeaveKind` so the store stays
/// below the scheduler in the layering; the wait-all quorum wants to
/// know when the reclaim ladder forcibly suspends/restores a pipeline (a
/// frozen pipeline must never hold a wave), but that is a subscription the
/// scheduler installs on `store/`, not a call `store/` makes upward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    #[allow(dead_code)]
    Terminate,
    Suspend,
}

/// Pipeline-leave/join hooks. `notify_pipeline_leave` forwards to the
/// subscription the scheduler installs via [`set_pipeline_leave_hook`] (a
/// plain closure, so this module never names `crate::scheduler` and stays
/// below it in the layering — see [`LeaveKind`]'s doc for the A×B coupling
/// this seam serves); a no-op until installed. `notify_pipeline_join` stays
/// inert: the wait-all quorum's rejoin is implicit on a pipeline's next
/// wave request, so a join event has nothing to do on the scheduler side
/// either.
pub(crate) fn notify_pipeline_leave(pid: ProcessId, _kind: LeaveKind) {
    if let Some(hook) = PIPELINE_LEAVE_HOOK.get() {
        hook(pid);
    }
}
pub(crate) fn notify_pipeline_join(_pid: ProcessId) {}

type PipelineLeaveHook = Box<dyn Fn(ProcessId) + Send + Sync>;
static PIPELINE_LEAVE_HOOK: OnceLock<PipelineLeaveHook> = OnceLock::new();

/// Installs the scheduler's wait-all leave subscription (called once, from
/// `bootstrap`, wiring this to `scheduler::worker::notify_pipeline_leave`).
/// A plain closure keeps `store` below `scheduler` in the layering.
pub(crate) fn set_pipeline_leave_hook(hook: impl Fn(ProcessId) + Send + Sync + 'static) {
    let _ = PIPELINE_LEAVE_HOOK.set(Box::new(hook));
}

/// How the runtime reacts to a KV pool exhaustion at the prep seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentionMode {
    /// Legacy: surface `OutOfPages` to the inferlet as a forward error.
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
    fn host_pool_stats(&self) -> (u32, u32) {
        (0, 0)
    }
    /// Ladder rung 1: reclaim whatever costs no work — cache-root leases
    /// nothing live reaches. Returns blocks recycled NOW (0 when there is
    /// nothing idle); the orchestrator re-checks the pool before escalating
    /// to the victim loop.
    fn reclaim_idle(&self) -> u32 {
        0
    }
    /// Reserve concrete device page ids for one allocation or restore grant.
    /// The reservation owns the ids until consumed; dropping it returns them.
    fn reserve_pages(&self, count: u32) -> Option<DevicePageReservation>;
    /// Save `victim`'s state and free its pool blocks.
    fn suspend(&self, victim: ProcessId) -> SuspendOutcome;
    /// Re-materialize a suspended process (H2D warm restore / replay). On
    /// `Err` the orchestrator re-queues it (never silently dropped).
    fn restore(&self, pid: ProcessId) -> anyhow::Result<()>;
}

pub struct DevicePageReservation {
    pages: Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>>,
    returner: Option<Box<dyn FnOnce(Vec<crate::store::kv::page_table::PhysicalKvPageId>) + Send>>,
}

impl std::fmt::Debug for DevicePageReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DevicePageReservation")
            .field("pages", &self.pages)
            .finish_non_exhaustive()
    }
}

impl DevicePageReservation {
    pub fn new(
        pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>,
        returner: impl FnOnce(Vec<crate::store::kv::page_table::PhysicalKvPageId>) + Send + 'static,
    ) -> Self {
        Self {
            pages: Some(pages),
            returner: Some(Box::new(returner)),
        }
    }

    pub fn len(&self) -> usize {
        self.pages.as_ref().map_or(0, Vec::len)
    }

    fn take(mut self) -> Vec<crate::store::kv::page_table::PhysicalKvPageId> {
        self.returner = None;
        self.pages.take().unwrap_or_default()
    }
}

impl Drop for DevicePageReservation {
    fn drop(&mut self) {
        if let (Some(pages), Some(returner)) = (self.pages.take(), self.returner.take()) {
            returner(pages);
        }
    }
}

#[derive(Debug)]
pub struct AllocationGrant {
    pub process_id: ProcessId,
    pub request_id: u64,
    pub pages: u32,
    reservation: DevicePageReservation,
}

pub type ReclaimableProbe = Arc<dyn Fn() -> bool + Send + Sync>;

impl AllocationGrant {
    pub fn into_pages(self) -> Vec<crate::store::kv::page_table::PhysicalKvPageId> {
        self.reservation.take()
    }
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
    /// Fail LOUD to the guest (OutOfPages) instead of a silent wedge.
    Exhausted { need: u32, free: u32, total: u32 },
    /// The process was unregistered while waiting or parked.
    Cancelled,
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
            ContentionError::Cancelled => f.write_str("contention request cancelled"),
        }
    }
}

impl std::error::Error for ContentionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProcState {
    Running,
    Suspending,
    ParkRequested,
    Quiescing,
    Suspended,
    Restoring,
}

#[derive(Debug)]
struct Proc {
    /// Original process registration order: the authoritative FCFS clock.
    submit_seq: u64,
    state: ProcState,
    /// Blocks freed when this process was suspended = the restore admission
    /// estimate (`can_restore`: restore NEVER evicts).
    suspended_need: u32,
    /// When the process entered `Suspended` — drives the restore-pause AGING
    /// override (a starved FIFO-head restore eventually proceeds even above
    /// the utilization pause, if it fits).
    suspended_at: Option<Instant>,
    park_requested: Arc<Notify>,
}

/// What the caller must do after [`ContentionOrchestrator::acquire_or_self_suspend`]
/// returns successfully.
#[derive(Debug)]
pub enum Acquired {
    /// Concrete device ids reserved for this request.
    Granted(AllocationGrant),
    /// No victim can yield and the requester itself holds reclaimable pages:
    /// the caller must run the SELF-SUSPEND protocol on its own working set
    /// (suspend → `report_suspended` → `park_until_restore_granted` → restore),
    /// then retry. This is the prior design's `when_allocated` **step 6**
    /// ("NO VICTIM — requester self-suspends", sched.rs:861-866 @ fa3fe140^)
    /// — the designed deadlock-breaker: a parked page-HOLDER would otherwise
    /// strand its pages forever (the fleet=24/8 v2 deadlock).
    SelfSuspendFirst,
}

/// A parked allocation. The entry STAYS in the FIFO until its `acquire`
/// receives one concrete reservation-backed grant. Duplicate callers from
/// the same process aggregate into this one process-priority slot.
struct Waiter {
    request_id: u64,
    pid: ProcessId,
    need: u32,
    notify: Arc<Notify>,
    waiters: usize,
    grant: Option<AllocationGrant>,
    parked_reported: bool,
}

#[derive(Default)]
struct Inner {
    next_seq: u64,
    next_token: u64,
    procs: HashMap<ProcessId, Proc>,
    waiters: VecDeque<Waiter>,
    restore_queue: VecDeque<ProcessId>,
    /// Self-suspended (parked) processes awaiting release. Entry created by
    /// `report_suspended` BEFORE any drain can release it, so a release that
    /// beats `park_until_restore_granted` stores its permit in the `Notify` (no
    /// lost wakeup). Removed on wake-return and on unregister.
    parked: HashMap<ProcessId, Arc<Notify>>,
    restore_grants: HashMap<ProcessId, AllocationGrant>,
    restore_errors: HashMap<ProcessId, ContentionError>,
    restore_exhausted_since: Option<(ProcessId, Instant)>,
}

/// Engagement counters (lock-free reads) — the e2e's proof that contention
/// actually ENGAGED, so a trivially-passing run (pool never over-filled)
/// can't masquerade as a validated preempt/restore path. NOTE for asserts:
/// under the v1 passive backend (`KvPoolBackend`, suspend=Unsupported)
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
    pub allocation_grants: std::sync::atomic::AtomicU64,
    pub restore_grants: std::sync::atomic::AtomicU64,
    pub cancelled_waits: std::sync::atomic::AtomicU64,
    pub exhaustion_timeouts: std::sync::atomic::AtomicU64,
    pub grace_deferred: std::sync::atomic::AtomicU64,
    pub d2h_pages: std::sync::atomic::AtomicU64,
    pub h2d_pages: std::sync::atomic::AtomicU64,
    pub d2h_copy_us: std::sync::atomic::AtomicU64,
    pub h2d_copy_us: std::sync::atomic::AtomicU64,
    pub suspend_rollbacks: std::sync::atomic::AtomicU64,
    pub restore_rollbacks: std::sync::atomic::AtomicU64,
    pub host_swap_exhaustions: std::sync::atomic::AtomicU64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ContentionQueueEntry {
    pub process_id: String,
    pub submit_seq: u64,
    pub state: &'static str,
    pub pages: u32,
    pub granted: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ContentionDiagnostics {
    pub waiters: Vec<ContentionQueueEntry>,
    pub suspended: Vec<ContentionQueueEntry>,
    pub device_pages_free: u32,
    pub device_pages_total: u32,
    pub device_pages_reserved: u32,
    pub host_slots_free: u32,
    pub host_slots_total: u32,
    pub waiters_parked_total: u64,
    pub waiters_woken_total: u64,
    pub suspends_total: u64,
    pub restores_total: u64,
    pub allocation_grants_total: u64,
    pub restore_grants_total: u64,
    pub cancelled_waits_total: u64,
    pub exhaustion_timeouts_total: u64,
    pub grace_deferred_total: u64,
    pub d2h_pages_total: u64,
    pub h2d_pages_total: u64,
    pub d2h_copy_us_total: u64,
    pub h2d_copy_us_total: u64,
    pub suspend_rollbacks_total: u64,
    pub restore_rollbacks_total: u64,
    pub host_swap_exhaustions_total: u64,
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

struct WaitRegistration<'a> {
    orchestrator: &'a ContentionOrchestrator,
    pid: ProcessId,
    request_id: u64,
    active: bool,
}

#[derive(Clone, Copy)]
enum GrantCandidate {
    Allocation {
        pid: ProcessId,
        request_id: u64,
        need: u32,
    },
    Restore {
        pid: ProcessId,
        need: u32,
    },
}

impl WaitRegistration<'_> {
    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for WaitRegistration<'_> {
    fn drop(&mut self) {
        if self.active {
            self.orchestrator
                .cancel_waiter(self.pid, self.request_id, true);
        }
    }
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

    pub fn diagnostics(&self) -> ContentionDiagnostics {
        let (device_pages_free, device_pages_total) = self.backend.pool_stats();
        let (host_slots_free, host_slots_total) = self.backend.host_pool_stats();
        let inner = self.inner.lock().unwrap();
        let mut waiters: Vec<_> = inner
            .waiters
            .iter()
            .filter_map(|waiter| {
                inner
                    .procs
                    .get(&waiter.pid)
                    .map(|process| ContentionQueueEntry {
                        process_id: waiter.pid.to_string(),
                        submit_seq: process.submit_seq,
                        state: "allocation",
                        pages: waiter.need,
                        granted: waiter.grant.is_some(),
                    })
            })
            .collect();
        waiters.sort_by_key(|entry| entry.submit_seq);
        let mut suspended: Vec<_> = inner
            .procs
            .iter()
            .filter(|(_, process)| {
                matches!(process.state, ProcState::Suspended | ProcState::Restoring)
            })
            .map(|(pid, process)| ContentionQueueEntry {
                process_id: pid.to_string(),
                submit_seq: process.submit_seq,
                state: if process.state == ProcState::Restoring {
                    "restoring"
                } else {
                    "suspended"
                },
                pages: process.suspended_need,
                granted: inner.restore_grants.contains_key(pid),
            })
            .collect();
        suspended.sort_by_key(|entry| entry.submit_seq);
        let device_pages_reserved = inner
            .waiters
            .iter()
            .filter_map(|waiter| waiter.grant.as_ref())
            .chain(inner.restore_grants.values())
            .map(|grant| grant.pages)
            .sum();
        use std::sync::atomic::Ordering::Relaxed;
        ContentionDiagnostics {
            waiters,
            suspended,
            device_pages_free,
            device_pages_total,
            device_pages_reserved,
            host_slots_free,
            host_slots_total,
            waiters_parked_total: self.stats.waiters_parked.load(Relaxed),
            waiters_woken_total: self.stats.waiters_woken.load(Relaxed),
            suspends_total: self.stats.suspends.load(Relaxed),
            restores_total: self.stats.restores.load(Relaxed),
            allocation_grants_total: self.stats.allocation_grants.load(Relaxed),
            restore_grants_total: self.stats.restore_grants.load(Relaxed),
            cancelled_waits_total: self.stats.cancelled_waits.load(Relaxed),
            exhaustion_timeouts_total: self.stats.exhaustion_timeouts.load(Relaxed),
            grace_deferred_total: self.stats.grace_deferred.load(Relaxed),
            d2h_pages_total: self.stats.d2h_pages.load(Relaxed),
            h2d_pages_total: self.stats.h2d_pages.load(Relaxed),
            d2h_copy_us_total: self.stats.d2h_copy_us.load(Relaxed),
            h2d_copy_us_total: self.stats.h2d_copy_us.load(Relaxed),
            suspend_rollbacks_total: self.stats.suspend_rollbacks.load(Relaxed),
            restore_rollbacks_total: self.stats.restore_rollbacks.load(Relaxed),
            host_swap_exhaustions_total: self.stats.host_swap_exhaustions.load(Relaxed),
        }
    }

    pub fn record_d2h_copy(&self, elapsed: Duration) {
        self.stats.d2h_copy_us.fetch_add(
            elapsed.as_micros().min(u128::from(u64::MAX)) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    pub fn record_h2d_copy(&self, elapsed: Duration) {
        self.stats.h2d_copy_us.fetch_add(
            elapsed.as_micros().min(u128::from(u64::MAX)) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    pub fn record_suspend_rollback(&self) {
        self.stats
            .suspend_rollbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_restore_rollback(&self) {
        self.stats
            .restore_rollbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn record_host_swap_exhaustion(&self) {
        self.stats
            .host_swap_exhaustions
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn kv_pressure_bucket(&self) -> u8 {
        let diagnostics = self.diagnostics();
        let device_used = diagnostics
            .device_pages_total
            .saturating_sub(diagnostics.device_pages_free);
        let device_ratio = if diagnostics.device_pages_total == 0 {
            0.0
        } else {
            f64::from(device_used) / f64::from(diagnostics.device_pages_total)
        };
        let host_used = diagnostics
            .host_slots_total
            .saturating_sub(diagnostics.host_slots_free);
        let host_ratio = if diagnostics.host_slots_total == 0 {
            0.0
        } else {
            f64::from(host_used) / f64::from(diagnostics.host_slots_total)
        };
        let mut bucket = (device_ratio.max(host_ratio) * 255.0).round() as u8;
        if !diagnostics.suspended.is_empty() {
            bucket = bucket.max(224);
        }
        if !diagnostics.waiters.is_empty() {
            bucket = bucket.max(240);
        }
        bucket
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
        inner.procs.values().all(|q| q.submit_seq >= p.submit_seq)
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
                submit_seq: seq,
                state: ProcState::Running,
                suspended_need: 0,
                suspended_at: None,
                park_requested: Arc::new(Notify::new()),
            },
        );
    }

    /// Unregister at process exit/terminate. Its parked waiters are removed
    /// (their `acquire` futures are being aborted with the process) and it
    /// leaves the restore queue. The exit frees the process's blocks — the
    /// caller runs [`on_blocks_freed`](Self::on_blocks_freed) after resource
    /// cleanup; this also drains defensively.
    pub fn unregister(&self, pid: ProcessId) {
        let (parked, waiters) = {
            let mut inner = self.inner.lock().unwrap();
            inner.procs.remove(&pid);
            let waiters: Vec<_> = inner
                .waiters
                .iter()
                .filter(|waiter| waiter.pid == pid)
                .map(|waiter| waiter.notify.clone())
                .collect();
            inner.waiters.retain(|w| w.pid != pid);
            inner.restore_queue.retain(|&p| p != pid);
            inner.restore_grants.remove(&pid);
            inner.restore_errors.remove(&pid);
            (inner.parked.remove(&pid), waiters)
        };
        for notify in waiters {
            notify.notify_waiters();
        }
        if let Some(notify) = parked {
            // A parked task being torn down: wake it so the restore-grant wait
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
    //   3. `park_until_restore_granted(pid).await` — waits for concrete device
    //      ids, restores H2D, publishes remapped metadata, then calls
    //      `report_restored`; only that final report makes it runnable.
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

    /// Notification raced against long host awaits so an idle process can
    /// honor a park request without waiting for another guest call.
    pub fn park_signal(&self, pid: ProcessId) -> Option<Arc<Notify>> {
        self.inner
            .lock()
            .unwrap()
            .procs
            .get(&pid)
            .map(|process| process.park_requested.clone())
    }

    pub fn begin_quiesce(&self, pid: ProcessId) -> bool {
        let mut inner = self.inner.lock().unwrap();
        let Some(process) = inner.procs.get_mut(&pid) else {
            return false;
        };
        if process.state != ProcState::ParkRequested {
            return false;
        }
        process.state = ProcState::Quiescing;
        drop(inner);
        notify_pipeline_leave(pid, LeaveKind::Suspend);
        true
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
        let signal = {
            let mut inner = self.inner.lock().unwrap();
            let Some(p) = inner.procs.get_mut(&pid) else {
                return;
            };
            if matches!(p.state, ProcState::ParkRequested | ProcState::Quiescing) {
                p.state = ProcState::Running;
                Some(p.park_requested.clone())
            } else {
                None
            }
        };
        if let Some(signal) = signal {
            signal.notify_waiters();
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
            if !matches!(
                p.state,
                ProcState::Quiescing | ProcState::ParkRequested | ProcState::Suspending
            ) {
                return;
            }
            p.state = ProcState::Suspended;
            p.suspended_need = freed_now;
            p.suspended_at = Some(Instant::now());
            if !inner.restore_queue.contains(&pid) {
                inner.restore_queue.push_back(pid);
            }
            self.stats
                .suspends
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.stats
                .d2h_pages
                .fetch_add(u64::from(freed_now), std::sync::atomic::Ordering::Relaxed);
            // Create the park entry BEFORE any drain can release the process,
            // so an early release stores its permit (no lost wakeup).
            inner.parked.entry(pid).or_default();
        }
        self.drain();
    }

    /// Victim-side step 3: park until the restore phase releases this
    /// process (state back to Running), then return so the caller
    /// re-materializes its state. Returns immediately if the process was
    /// unregistered while parked (it is being torn down).
    pub async fn park_until_restore_granted(
        &self,
        pid: ProcessId,
    ) -> Result<AllocationGrant, ContentionError> {
        loop {
            self.drain();
            let notify = {
                let mut inner = self.inner.lock().unwrap();
                if let Some(error) = inner.restore_errors.remove(&pid) {
                    inner.parked.remove(&pid);
                    return Err(error);
                }
                if let Some(grant) = inner.restore_grants.remove(&pid) {
                    return Ok(grant);
                }
                match inner.procs.get(&pid).map(|p| p.state) {
                    None => {
                        inner.parked.remove(&pid);
                        return Err(ContentionError::Cancelled);
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
            let poll = Duration::from_millis((self.exhaustion_ms / 4).clamp(20, 1000));
            tokio::select! {
                _ = notify.notified() => {}
                _ = tokio::time::sleep(poll) => {}
            }
        }
    }

    pub fn report_restored(&self, pid: ProcessId, restored_pages: u32) {
        let signal = {
            let mut inner = self.inner.lock().unwrap();
            let Some(process) = inner.procs.get_mut(&pid) else {
                return;
            };
            if process.state != ProcState::Restoring {
                return;
            }
            process.state = ProcState::Running;
            process.suspended_need = 0;
            process.suspended_at = None;
            let signal = process.park_requested.clone();
            inner.parked.remove(&pid);
            signal
        };
        signal.notify_waiters();
        self.stats
            .restores
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.stats.h2d_pages.fetch_add(
            u64::from(restored_pages),
            std::sync::atomic::Ordering::Relaxed,
        );
        notify_pipeline_join(pid);
        self.drain();
    }

    pub fn report_restore_failed(&self, pid: ProcessId) {
        {
            let mut inner = self.inner.lock().unwrap();
            let Some(process) = inner.procs.get_mut(&pid) else {
                return;
            };
            process.state = ProcState::Suspended;
            if !inner.restore_queue.contains(&pid) {
                inner.restore_queue.push_back(pid);
            }
        }
        self.drain();
    }

    pub async fn acquire(
        &self,
        requester: ProcessId,
        need: u32,
    ) -> Result<AllocationGrant, ContentionError> {
        match self.acquire_or_self_suspend(requester, need, false).await? {
            Acquired::Granted(grant) => Ok(grant),
            Acquired::SelfSuspendFirst => unreachable!("self suspend disabled"),
        }
    }

    pub async fn acquire_or_self_suspend(
        &self,
        requester: ProcessId,
        need: u32,
        holds_reclaimable: bool,
    ) -> Result<Acquired, ContentionError> {
        self.acquire_or_self_suspend_live(requester, need, Arc::new(move || holds_reclaimable))
            .await
    }

    pub async fn acquire_or_self_suspend_live(
        &self,
        requester: ProcessId,
        need: u32,
        holds_reclaimable: ReclaimableProbe,
    ) -> Result<Acquired, ContentionError> {
        let (_, total) = self.backend.pool_stats();
        if need > total {
            return Err(ContentionError::Impossible { need, total });
        }
        let (mut request_id, mut notify) = loop {
            match self.register_waiter(requester, need) {
                Ok(registered) => break registered,
                Err(ContentionError::Cancelled) if self.is_registered(requester) => {
                    self.wait_until_running(requester).await?;
                }
                Err(error) => return Err(error),
            }
        };
        let mut registration = WaitRegistration {
            orchestrator: self,
            pid: requester,
            request_id,
            active: true,
        };
        let mut tried = HashSet::new();
        let mut exhausted_since = None;

        loop {
            self.drain();
            if let Some(grant) = self.take_allocation_grant(requester, request_id, need) {
                registration.disarm();
                notify_pipeline_join(requester);
                return Ok(Acquired::Granted(grant));
            }
            if !self.inner.lock().unwrap().procs.contains_key(&requester) {
                registration.disarm();
                return Err(ContentionError::Cancelled);
            }
            let registered = self
                .inner
                .lock()
                .unwrap()
                .waiters
                .iter()
                .any(|waiter| waiter.pid == requester && waiter.request_id == request_id);
            if !registered {
                registration.disarm();
                let registered = loop {
                    match self.register_waiter(requester, need) {
                        Ok(registered) => break registered,
                        Err(ContentionError::Cancelled) if self.is_registered(requester) => {
                            self.wait_until_running(requester).await?;
                        }
                        Err(error) => return Err(error),
                    }
                };
                request_id = registered.0;
                notify = registered.1;
                registration = WaitRegistration {
                    orchestrator: self,
                    pid: requester,
                    request_id,
                    active: true,
                };
                continue;
            }

            if self.backend.reclaim_idle() > 0 {
                self.drain();
                tried.clear();
                exhausted_since = None;
                continue;
            }

            let victim = {
                let mut inner = self.inner.lock().unwrap();
                let oldest = inner
                    .procs
                    .iter()
                    .min_by_key(|(_, process)| process.submit_seq)
                    .map(|(pid, _)| *pid);
                let victim = inner
                    .procs
                    .iter()
                    .filter(|(pid, process)| {
                        **pid != requester
                            && Some(**pid) != oldest
                            && process.state == ProcState::Running
                            && !tried.contains(*pid)
                            && !inner.waiters.iter().any(|waiter| waiter.pid == **pid)
                    })
                    .max_by_key(|(_, process)| process.submit_seq)
                    .map(|(pid, _)| *pid);
                if let Some(victim) = victim {
                    inner.procs.get_mut(&victim).unwrap().state = ProcState::Suspending;
                }
                victim
            };

            if let Some(victim) = victim {
                match self.backend.suspend(victim) {
                    SuspendOutcome::Suspended { freed_now } => {
                        if freed_now > 0 {
                            exhausted_since = None;
                        }
                        {
                            let mut inner = self.inner.lock().unwrap();
                            if let Some(process) = inner.procs.get_mut(&victim) {
                                process.state = ProcState::Suspended;
                                process.suspended_need = freed_now;
                                process.suspended_at = Some(Instant::now());
                                if !inner.restore_queue.contains(&victim) {
                                    inner.restore_queue.push_back(victim);
                                }
                            }
                        }
                        self.stats
                            .suspends
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        self.drain();
                    }
                    SuspendOutcome::Requested => {
                        let signal = {
                            let mut inner = self.inner.lock().unwrap();
                            inner.procs.get_mut(&victim).map(|process| {
                                process.state = ProcState::ParkRequested;
                                process.park_requested.clone()
                            })
                        };
                        if let Some(signal) = signal {
                            signal.notify_waiters();
                        }
                        tried.insert(victim);
                    }
                    SuspendOutcome::DeferredByGrace => {
                        self.stats
                            .grace_deferred
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let signal = {
                            let mut inner = self.inner.lock().unwrap();
                            inner.procs.get_mut(&victim).map(|process| {
                                process.state = ProcState::Running;
                                process.park_requested.clone()
                            })
                        };
                        if let Some(signal) = signal {
                            signal.notify_waiters();
                        }
                        tried.insert(victim);
                    }
                    SuspendOutcome::Unsupported => {
                        let signal = {
                            let mut inner = self.inner.lock().unwrap();
                            inner.procs.get_mut(&victim).map(|process| {
                                process.state = ProcState::Running;
                                process.park_requested.clone()
                            })
                        };
                        if let Some(signal) = signal {
                            signal.notify_waiters();
                        }
                        tried.insert(victim);
                    }
                }
                continue;
            }

            if holds_reclaimable() && !self.is_fcfs_oldest(requester) {
                registration.disarm();
                self.cancel_waiter(requester, request_id, false);
                let signal = {
                    let mut inner = self.inner.lock().unwrap();
                    inner.procs.get_mut(&requester).map(|process| {
                        process.state = ProcState::ParkRequested;
                        process.park_requested.clone()
                    })
                };
                if let Some(signal) = signal {
                    signal.notify_waiters();
                }
                notify_pipeline_join(requester);
                return Ok(Acquired::SelfSuspendFirst);
            }

            let (free, total) = self.backend.pool_stats();
            if self.is_oldest_requester(requester) {
                let since = *exhausted_since.get_or_insert_with(Instant::now);
                if since.elapsed() >= Duration::from_millis(self.exhaustion_ms) {
                    registration.disarm();
                    self.cancel_waiter(requester, request_id, false);
                    self.stats
                        .exhaustion_timeouts
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(ContentionError::Exhausted { need, free, total });
                }
            } else {
                exhausted_since = None;
            }

            self.mark_waiter_parked(requester, request_id);
            let poll = Duration::from_millis((self.exhaustion_ms / 4).clamp(20, 1000));
            tokio::select! {
                _ = notify.notified() => {}
                _ = tokio::time::sleep(poll) => {}
            }
            tried.clear();
        }
    }

    /// Any queued entitlement forces new allocations through the grant path;
    /// callers may not race a direct pool allocation around an older entry.
    pub fn allocation_requires_grant(&self) -> bool {
        true
    }

    pub fn is_running(&self, pid: ProcessId) -> bool {
        self.inner
            .lock()
            .unwrap()
            .procs
            .get(&pid)
            .is_some_and(|process| process.state == ProcState::Running)
    }

    pub fn is_registered(&self, pid: ProcessId) -> bool {
        self.inner.lock().unwrap().procs.contains_key(&pid)
    }

    async fn wait_until_running(&self, pid: ProcessId) -> Result<(), ContentionError> {
        loop {
            let signal = {
                let inner = self.inner.lock().unwrap();
                let Some(process) = inner.procs.get(&pid) else {
                    return Err(ContentionError::Cancelled);
                };
                if process.state == ProcState::Running {
                    return Ok(());
                }
                process.park_requested.clone()
            };
            let notified = signal.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = self.inner.lock().unwrap();
                let Some(process) = inner.procs.get(&pid) else {
                    return Err(ContentionError::Cancelled);
                };
                if process.state == ProcState::Running {
                    return Ok(());
                }
            }
            notified.await;
        }
    }

    fn register_waiter(
        &self,
        pid: ProcessId,
        need: u32,
    ) -> Result<(u64, Arc<Notify>), ContentionError> {
        let mut inner = self.inner.lock().unwrap();
        if !inner
            .procs
            .get(&pid)
            .is_some_and(|process| process.state == ProcState::Running)
        {
            return Err(ContentionError::Cancelled);
        }
        if let Some(waiter) = inner.waiters.iter_mut().find(|waiter| waiter.pid == pid) {
            if waiter.grant.is_none() {
                waiter.need = waiter.need.max(need);
            }
            waiter.waiters += 1;
            return Ok((waiter.request_id, waiter.notify.clone()));
        }
        let request_id = inner.next_token;
        inner.next_token += 1;
        let notify = Arc::new(Notify::new());
        inner.waiters.push_back(Waiter {
            request_id,
            pid,
            need,
            notify: notify.clone(),
            waiters: 1,
            grant: None,
            parked_reported: false,
        });
        Ok((request_id, notify))
    }

    fn mark_waiter_parked(&self, pid: ProcessId, request_id: u64) {
        let first = {
            let mut inner = self.inner.lock().unwrap();
            inner
                .waiters
                .iter_mut()
                .find(|waiter| waiter.pid == pid && waiter.request_id == request_id)
                .is_some_and(|waiter| {
                    if waiter.parked_reported {
                        false
                    } else {
                        waiter.parked_reported = true;
                        true
                    }
                })
        };
        if first {
            self.stats
                .waiters_parked
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            notify_pipeline_leave(pid, LeaveKind::Suspend);
        }
    }

    fn take_allocation_grant(
        &self,
        pid: ProcessId,
        request_id: u64,
        need: u32,
    ) -> Option<AllocationGrant> {
        let mut inner = self.inner.lock().unwrap();
        let position = inner
            .waiters
            .iter()
            .position(|waiter| waiter.pid == pid && waiter.request_id == request_id)?;
        if inner.waiters[position]
            .grant
            .as_ref()
            .is_none_or(|grant| grant.pages < need)
        {
            return None;
        }
        let grant = inner.waiters[position].grant.take()?;
        inner.waiters.remove(position);
        self.stats
            .waiters_woken
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(grant)
    }

    fn cancel_waiter(&self, pid: ProcessId, request_id: u64, cancelled: bool) {
        let removed = {
            let mut inner = self.inner.lock().unwrap();
            let Some(position) = inner
                .waiters
                .iter()
                .position(|waiter| waiter.pid == pid && waiter.request_id == request_id)
            else {
                return;
            };
            if inner.waiters[position].waiters > 1 {
                inner.waiters[position].waiters -= 1;
                false
            } else {
                inner.waiters.remove(position);
                true
            }
        };
        if cancelled {
            self.stats
                .cancelled_waits
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if removed {
            notify_pipeline_join(pid);
            self.drain();
        }
    }

    fn is_oldest_requester(&self, pid: ProcessId) -> bool {
        let inner = self.inner.lock().unwrap();
        let Some(process) = inner.procs.get(&pid) else {
            return false;
        };
        inner.waiters.iter().all(|waiter| {
            inner
                .procs
                .get(&waiter.pid)
                .is_none_or(|other| other.submit_seq >= process.submit_seq)
        })
    }

    fn drain(&self) {
        loop {
            let (free, total) = self.backend.pool_stats();
            let candidate = {
                let mut inner = self.inner.lock().unwrap();
                let mut candidates = Vec::new();
                for waiter in inner.waiters.iter().filter(|waiter| waiter.grant.is_none()) {
                    if let Some(process) = inner.procs.get(&waiter.pid) {
                        candidates.push((
                            process.submit_seq,
                            GrantCandidate::Allocation {
                                pid: waiter.pid,
                                request_id: waiter.request_id,
                                need: waiter.need,
                            },
                        ));
                    }
                }
                for &pid in &inner.restore_queue {
                    if let Some(process) = inner.procs.get(&pid) {
                        candidates.push((
                            process.submit_seq,
                            GrantCandidate::Restore {
                                pid,
                                need: process.suspended_need,
                            },
                        ));
                    }
                }
                candidates.sort_by_key(|(submit_seq, _)| *submit_seq);

                let utilization = f64::from(total.saturating_sub(free)) / f64::from(total.max(1));
                let mut selected = None;
                for (_, candidate) in candidates {
                    match candidate {
                        GrantCandidate::Restore { pid, need } => {
                            let Some(process) = inner.procs.get(&pid) else {
                                continue;
                            };
                            let aged = process.suspended_at.is_some_and(|since| {
                                since.elapsed() >= Duration::from_millis(restore_aging_ms())
                            });
                            if utilization > self.restore_pause_at_utilization && !aged {
                                continue;
                            }
                            if free < need {
                                let oldest = inner
                                    .procs
                                    .iter()
                                    .min_by_key(|(_, process)| process.submit_seq)
                                    .map(|(pid, _)| *pid);
                                if oldest == Some(pid) {
                                    let since = match inner.restore_exhausted_since {
                                        Some((head, since)) if head == pid => since,
                                        _ => {
                                            let now = Instant::now();
                                            inner.restore_exhausted_since = Some((pid, now));
                                            now
                                        }
                                    };
                                    if since.elapsed() >= Duration::from_millis(self.exhaustion_ms)
                                    {
                                        inner.restore_queue.retain(|queued| *queued != pid);
                                        inner.restore_exhausted_since = None;
                                        inner.restore_errors.insert(
                                            pid,
                                            ContentionError::Exhausted { need, free, total },
                                        );
                                        if let Some(notify) = inner.parked.get(&pid) {
                                            notify.notify_waiters();
                                        }
                                        self.stats
                                            .exhaustion_timeouts
                                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    }
                                } else {
                                    inner.restore_exhausted_since = None;
                                }
                                return;
                            }
                            inner.restore_exhausted_since = None;
                            selected = Some(candidate);
                            break;
                        }
                        GrantCandidate::Allocation { need, .. } => {
                            if free < need {
                                return;
                            }
                            inner.restore_exhausted_since = None;
                            selected = Some(candidate);
                            break;
                        }
                    }
                }
                selected
            };

            let Some(candidate) = candidate else {
                return;
            };
            let need = match candidate {
                GrantCandidate::Allocation { need, .. } | GrantCandidate::Restore { need, .. } => {
                    need
                }
            };
            let Some(reservation) = self.backend.reserve_pages(need) else {
                return;
            };

            let mut notify = None;
            let mut synchronous_restore = None;
            {
                let mut inner = self.inner.lock().unwrap();
                match candidate {
                    GrantCandidate::Allocation {
                        pid,
                        request_id,
                        need,
                    } => {
                        let Some(waiter) = inner.waiters.iter_mut().find(|waiter| {
                            waiter.pid == pid
                                && waiter.request_id == request_id
                                && waiter.grant.is_none()
                        }) else {
                            drop(inner);
                            drop(reservation);
                            continue;
                        };
                        waiter.grant = Some(AllocationGrant {
                            process_id: pid,
                            request_id,
                            pages: need,
                            reservation,
                        });
                        notify = Some(waiter.notify.clone());
                        self.stats
                            .allocation_grants
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    GrantCandidate::Restore { pid, need } => {
                        let valid = inner.restore_queue.contains(&pid)
                            && inner
                                .procs
                                .get(&pid)
                                .is_some_and(|process| process.state == ProcState::Suspended);
                        if !valid {
                            drop(inner);
                            drop(reservation);
                            continue;
                        }
                        inner.restore_queue.retain(|queued| *queued != pid);
                        let request_id = inner.next_token;
                        inner.next_token += 1;
                        if let Some(process) = inner.procs.get_mut(&pid) {
                            process.state = ProcState::Restoring;
                        }
                        if let Some(parked) = inner.parked.get(&pid).cloned() {
                            inner.restore_grants.insert(
                                pid,
                                AllocationGrant {
                                    process_id: pid,
                                    request_id,
                                    pages: need,
                                    reservation,
                                },
                            );
                            notify = Some(parked);
                        } else {
                            synchronous_restore = Some((pid, reservation));
                        }
                        self.stats
                            .restore_grants
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            }
            if let Some(notify) = notify {
                notify.notify_waiters();
            }
            if let Some((pid, reservation)) = synchronous_restore {
                drop(reservation);
                match self.backend.restore(pid) {
                    Ok(()) => self.report_restored(pid, need),
                    Err(error) => {
                        tracing::warn!(pid = %pid, "restore failed, re-queued: {error:#}");
                        self.report_restore_failed(pid);
                    }
                }
            }
        }
    }

    /// Wait until `need` free blocks are plausibly available, FCFS-preempting
    /// younger processes as needed. Returns when the caller should RETRY its
    /// allocation (the retry re-enters here on a lost race — progress is
    /// guaranteed because the oldest process is never a victim and waiters
    /// wake FIFO). The caller must hold NO arena/WS locks and no staged txn.
    #[cfg(any())]
    pub async fn acquire_legacy(
        &self,
        requester: ProcessId,
        need: u32,
    ) -> Result<(), ContentionError> {
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
    #[cfg(any())]
    pub async fn acquire_or_self_suspend_legacy(
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

            // Ladder rung 1: reclaim idle cache leases (no work lost) before
            // any preemption. If it yielded, re-check the pool — the freed
            // blocks may already satisfy us (and parked waiters).
            if self.backend.reclaim_idle() > 0 {
                self.wake_waiters();
                continue;
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
                                notify_pipeline_leave(requester, LeaveKind::Suspend);
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
            notify_pipeline_join(requester);
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
    #[cfg(any())]
    fn wake_waiters_legacy(&self) {
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
    #[cfg(any())]
    fn drain_legacy(&self) {
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
                    notify_pipeline_join(pid);
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
                    notify_pipeline_join(pid);
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
// v1 backend — real pool stats + idle-lease reclaim over the typed KvStore
// =============================================================================

/// v1 [`ReclaimBackend`] over the typed `KvStore`'s physical page pool. Pool
/// stats and the idle-lease rung are real, so preempt mode gives ladder rung
/// 1 + FIFO wait-for-free + restore-queue plumbing end-to-end; `suspend`
/// reports [`SuspendOutcome::Unsupported`] until the victim-side release
/// path binds here (phase 1 preempt = release the victim's WorkingSets and
/// recompute on resume — needs the recompute/replay machinery, a later
/// increment; `KvStore::exclusive_footprint` is the sizing primitive).
pub struct KvPoolBackend {
    model_idx: usize,
    driver_idx: usize,
}

impl KvPoolBackend {
    pub fn new(model_idx: usize, driver_idx: usize) -> Self {
        Self {
            model_idx,
            driver_idx,
        }
    }
}

impl ReclaimBackend for KvPoolBackend {
    fn pool_stats(&self) -> (u32, u32) {
        let stores = crate::store::registry::get(self.model_idx, self.driver_idx);
        let kv = stores.kv.lock().unwrap();
        let free = kv.available_pages() as u32;
        let total = kv.capacity_pages();
        (free, total)
    }

    fn host_pool_stats(&self) -> (u32, u32) {
        let stores = crate::store::registry::get(self.model_idx, self.driver_idx);
        let kv = stores.kv.lock().unwrap();
        (kv.host_swap_available() as u32, kv.host_swap_capacity())
    }

    fn reclaim_idle(&self) -> u32 {
        let stores = crate::store::registry::get(self.model_idx, self.driver_idx);
        let mut kv = stores.kv.lock().unwrap();
        let epoch = kv.current_epoch();
        let freed = kv.drop_unused_cache_leases(epoch);
        if freed > 0 {
            kv.retire_idle();
        }
        freed as u32
    }

    fn reserve_pages(&self, count: u32) -> Option<DevicePageReservation> {
        let stores = crate::store::registry::get(self.model_idx, self.driver_idx);
        let pages = stores
            .kv
            .lock()
            .unwrap()
            .reserve_device_pages(count as usize)?;
        let model_idx = self.model_idx;
        let driver_idx = self.driver_idx;
        Some(DevicePageReservation::new(pages, move |pages| {
            let stores = crate::store::registry::get(model_idx, driver_idx);
            stores.kv.lock().unwrap().release_device_reservation(pages);
        }))
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
/// posting the park request; process-owned orchestration freezes scheduler
/// preparation, drains its fires, and commits D2H before `report_suspended`.
/// Restore runs H2D after `park_until_restore_granted` and calls
/// `report_restored` only after mapping publication. `restore()` remains only
/// for synchronous backend compatibility. Selected by
/// `PIE_KV_PREEMPT_ACTIVE=1` on top of `PIE_KV_CONTENTION=preempt`;
/// engagement shows as `stats().suspends/restores > 0` (the pre-wired v2
/// e2e assertion). Passive (`KvPoolBackend`) remains the default.
pub struct SelfSuspendBackend {
    pool: KvPoolBackend,
}

impl SelfSuspendBackend {
    pub fn new(model_idx: usize, driver_idx: usize) -> Self {
        Self {
            pool: KvPoolBackend::new(model_idx, driver_idx),
        }
    }
}

impl ReclaimBackend for SelfSuspendBackend {
    fn pool_stats(&self) -> (u32, u32) {
        self.pool.pool_stats()
    }

    fn host_pool_stats(&self) -> (u32, u32) {
        self.pool.host_pool_stats()
    }

    fn reclaim_idle(&self) -> u32 {
        self.pool.reclaim_idle()
    }

    fn reserve_pages(&self, count: u32) -> Option<DevicePageReservation> {
        self.pool.reserve_pages(count)
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
    static AGING_MS: OnceLock<u64> = OnceLock::new();
    *AGING_MS.get_or_init(|| {
        std::env::var("PIE_KV_RESTORE_AGING_MS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(250)
    })
}

/// #19 exhaustion deadline (ms): 10000 = 10s. Bounds how long the FCFS-oldest
/// keystone may stay continuously unsatisfiable before `acquire` fails LOUD
/// instead of wedging — turning a ~300s silent hang into a 10s named error.
/// Tests override via [`ContentionOrchestrator::with_exhaustion_ms`].
fn exhaustion_ms_from_env() -> u64 {
    std::env::var("PIE_KV_EXHAUSTION_MS")
        .ok()
        .and_then(|value| value.parse().ok())
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

/// Scheduler hook for payload-free fire retirement. No-op unless preempt mode
/// is wired (one `OnceLock` load on the hot path).
pub fn notify_fire_retired() {
    if let Some(o) = CONTENTION.get() {
        o.on_fire_retired();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A pool with `free` blocks live plus `idle` blocks reclaimable at zero
    /// cost (rung 1: unused cache leases).
    struct MockPool {
        free: Arc<AtomicU32>,
        idle: AtomicU32,
        total: u32,
    }

    impl ReclaimBackend for MockPool {
        fn pool_stats(&self) -> (u32, u32) {
            (self.free.load(Ordering::SeqCst), self.total)
        }
        fn reclaim_idle(&self) -> u32 {
            let n = self.idle.swap(0, Ordering::SeqCst);
            self.free.fetch_add(n, Ordering::SeqCst);
            n
        }
        fn reserve_pages(&self, count: u32) -> Option<DevicePageReservation> {
            let reserved = self
                .free
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |free| {
                    (free >= count).then_some(free - count)
                })
                .ok()?;
            debug_assert!(reserved >= count);
            let free = self.free.clone();
            let pages = (0..count)
                .map(crate::store::kv::page_table::PhysicalKvPageId)
                .collect();
            Some(DevicePageReservation::new(pages, move |_| {
                free.fetch_add(count, Ordering::SeqCst);
            }))
        }
        fn suspend(&self, _victim: ProcessId) -> SuspendOutcome {
            SuspendOutcome::Unsupported
        }
        fn restore(&self, _pid: ProcessId) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn orch(free: u32, idle: u32, total: u32) -> (ContentionOrchestrator, ProcessId) {
        let (orch, pid, _) = orch_with_free(free, idle, total);
        (orch, pid)
    }

    fn orch_with_free(
        free: u32,
        idle: u32,
        total: u32,
    ) -> (ContentionOrchestrator, ProcessId, Arc<AtomicU32>) {
        let free = Arc::new(AtomicU32::new(free));
        let orch = ContentionOrchestrator::new(
            Box::new(MockPool {
                free: free.clone(),
                idle: AtomicU32::new(idle),
                total,
            }),
            0.85,
        )
        .with_exhaustion_ms(200);
        let pid = ProcessId::new_v4();
        orch.register(pid);
        (orch, pid, free)
    }

    fn suspend_for_test(orch: &ContentionOrchestrator, pid: ProcessId, pages: u32) {
        orch.inner
            .lock()
            .unwrap()
            .procs
            .get_mut(&pid)
            .unwrap()
            .state = ProcState::ParkRequested;
        assert!(orch.begin_quiesce(pid));
        orch.report_suspended(pid, pages);
    }

    #[tokio::test]
    async fn ladder_rung_1_reclaims_idle_leases_before_parking() {
        // 2 free + 6 idle-reclaimable, need 5: rung 1 must satisfy the
        // request without the requester ever parking.
        let (orch, pid) = orch(2, 6, 16);
        let got = orch.acquire(pid, 5).await;
        assert!(got.is_ok());
        assert_eq!(orch.stats().waiters_parked.load(Ordering::Relaxed), 0);
        assert_eq!(orch.backend.pool_stats().0, 3); // five pages are grant-reserved
        drop(got);
        assert_eq!(orch.backend.pool_stats().0, 8);
    }

    #[tokio::test]
    async fn keystone_with_nothing_reclaimable_parks_then_fails_loud() {
        // 1 free, nothing idle, no peers: the FCFS-oldest keystone parks and,
        // once the exhaustion deadline lapses with no free event, fails LOUD
        // (#19) instead of wedging.
        let (orch, pid) = orch(1, 0, 16);
        let orch = std::sync::Arc::new(orch);
        let waiter = {
            let orch = orch.clone();
            tokio::spawn(async move { orch.acquire(pid, 4).await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(orch.contended());
        let got = waiter.await.unwrap();
        assert!(matches!(
            got,
            Err(ContentionError::Exhausted {
                need: 4,
                free: 1,
                total: 16
            })
        ));
        assert_eq!(orch.stats().waiters_parked.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn restores_follow_submit_order_not_suspend_completion_order() {
        let (orch, a, free) = orch_with_free(0, 0, 10);
        let b = ProcessId::new_v4();
        let c = ProcessId::new_v4();
        orch.register(b);
        orch.register(c);
        suspend_for_test(&orch, c, 2);
        suspend_for_test(&orch, b, 2);

        free.store(2, Ordering::SeqCst);
        orch.on_blocks_freed();
        let inner = orch.inner.lock().unwrap();
        assert!(inner.restore_grants.contains_key(&b));
        assert!(!inner.restore_grants.contains_key(&c));
        drop(inner);
        orch.unregister(a);
    }

    #[test]
    fn older_restore_outranks_younger_allocation_waiter() {
        let (orch, older, free) = orch_with_free(0, 0, 10);
        let younger = ProcessId::new_v4();
        orch.register(younger);
        suspend_for_test(&orch, older, 2);
        let (request_id, _) = orch.register_waiter(younger, 2).unwrap();

        free.store(2, Ordering::SeqCst);
        orch.on_blocks_freed();
        let inner = orch.inner.lock().unwrap();
        assert!(inner.restore_grants.contains_key(&older));
        assert!(
            inner
                .waiters
                .iter()
                .find(|waiter| waiter.request_id == request_id)
                .unwrap()
                .grant
                .is_none()
        );
    }

    #[test]
    fn grants_are_reserved_and_returned_on_unregister() {
        let (orch, first, free) = orch_with_free(2, 0, 10);
        let second = ProcessId::new_v4();
        orch.register(second);
        let (first_request, _) = orch.register_waiter(first, 2).unwrap();
        let (second_request, _) = orch.register_waiter(second, 2).unwrap();
        orch.drain();
        assert_eq!(free.load(Ordering::SeqCst), 0);
        assert!(
            orch.inner
                .lock()
                .unwrap()
                .waiters
                .iter()
                .find(|waiter| waiter.request_id == first_request)
                .unwrap()
                .grant
                .is_some()
        );

        orch.unregister(first);
        let inner = orch.inner.lock().unwrap();
        assert!(
            inner
                .waiters
                .iter()
                .find(|waiter| waiter.request_id == second_request)
                .unwrap()
                .grant
                .is_some()
        );
        assert_eq!(free.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn duplicate_process_waits_share_one_priority_slot() {
        let (orch, pid) = orch(0, 0, 10);
        let (first, _) = orch.register_waiter(pid, 2).unwrap();
        let (second, _) = orch.register_waiter(pid, 4).unwrap();
        assert_eq!(first, second);
        let inner = orch.inner.lock().unwrap();
        assert_eq!(inner.waiters.len(), 1);
        assert_eq!(inner.waiters[0].need, 4);
        assert_eq!(inner.waiters[0].waiters, 2);
    }

    #[test]
    fn larger_duplicate_cannot_consume_an_issued_smaller_grant() {
        let (orch, pid, _) = orch_with_free(2, 0, 10);
        let (request, _) = orch.register_waiter(pid, 2).unwrap();
        orch.drain();
        let (same_request, _) = orch.register_waiter(pid, 4).unwrap();
        assert_eq!(request, same_request);
        assert!(orch.take_allocation_grant(pid, request, 4).is_none());
        assert!(orch.take_allocation_grant(pid, request, 2).is_some());
    }

    #[tokio::test]
    async fn unsatisfiable_suspended_keystone_times_out_without_an_allocator() {
        let (orch, pid, _) = orch_with_free(0, 0, 10);
        let orch = Arc::new(orch.with_exhaustion_ms(40));
        suspend_for_test(&orch, pid, 2);
        let result =
            tokio::time::timeout(Duration::from_secs(1), orch.park_until_restore_granted(pid))
                .await
                .expect("restore watchdog must wake the suspended keystone");
        assert!(matches!(
            result,
            Err(ContentionError::Exhausted {
                need: 2,
                free: 0,
                total: 10
            })
        ));
    }

    #[tokio::test]
    async fn younger_requester_self_suspends_while_older_peer_is_suspended() {
        let (orch, older, _) = orch_with_free(0, 0, 10);
        let younger = ProcessId::new_v4();
        orch.register(younger);
        suspend_for_test(&orch, older, 2);
        let acquired = orch
            .acquire_or_self_suspend(younger, 2, true)
            .await
            .unwrap();
        assert!(matches!(acquired, Acquired::SelfSuspendFirst));
        assert!(orch.should_park(younger));
    }

    #[tokio::test]
    async fn permanently_unsupported_victim_does_not_reset_exhaustion_clock() {
        let (orch, oldest, _) = orch_with_free(0, 0, 10);
        orch.register(ProcessId::new_v4());
        let orch = Arc::new(orch.with_exhaustion_ms(40));
        let result = tokio::time::timeout(Duration::from_secs(1), orch.acquire(oldest, 2))
            .await
            .expect("unsupported victim retries must still hit exhaustion");
        assert!(matches!(
            result,
            Err(ContentionError::Exhausted {
                need: 2,
                free: 0,
                total: 10
            })
        ));
    }

    #[tokio::test]
    async fn dropping_an_acquisition_future_removes_its_waiter() {
        let (orch, pid, _) = orch_with_free(0, 0, 10);
        let orch = Arc::new(orch.with_exhaustion_ms(5_000));
        let task = {
            let orch = orch.clone();
            tokio::spawn(async move { orch.acquire(pid, 2).await })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while orch.inner.lock().unwrap().waiters.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        let _ = task.await;
        tokio::task::yield_now().await;
        assert!(orch.inner.lock().unwrap().waiters.is_empty());
        assert_eq!(orch.stats.cancelled_waits.load(Ordering::Relaxed), 1);
    }
}
