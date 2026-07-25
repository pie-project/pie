//! The reclaim ladder — FCFS preempt/restore over the KV physical pool. This
//! module manages pages, so it lives in `store/` even though the fire path
//! is what triggers it. The file/module name (`store::reclaim`) is the
//! contract; internals live in `reclaim/{grant, state, queue}.rs`.
//!
//! In Gim's directive: KV-cache contention is handled by PREEMPT/RESTORE, not
//! admission — and there is NO off switch. `max_concurrent_processes` is a
//! large physical safety cap only; every fire's demand (KV pages and RS
//! slots together) routes through the orchestrator's grant path.
//!
//! The shape (kilimanjaro.md §5): **one owned grant, one FCFS queue, one
//! safe point, and a state machine the type system enforces.**
//!
//! - **One queue, head-first-claim (D1).** Allocation waiters and restore
//!   entries order by one spawn clock. The head — the oldest unmet entry, of
//!   either kind — has first claim on every freed page and accumulates until
//!   its demand fits; younger entries are served only from surplus beyond
//!   the head's accumulation. Drainage is provable (the head's demand is
//!   finite; freed pages flow to it monotonically) and victim-restore thrash
//!   is impossible (a victim is younger than the waiter it yielded to, so
//!   its restore entry queues behind).
//! - **The safe point is a state, not a place (D2).** Victim eligibility is
//!   "holds no pins"; park requests are honored at the next safe point
//!   (every fire prologue, every idle-await entry). The orchestrator posts
//!   requests and victims comply — there is no cross-process state save.
//! - **One clock (D4).** A single system-wide progress deadline, armed for
//!   the queue head and reset by any progress toward it (pages entering its
//!   accumulation; any suspend, restore, or completion). No per-victim
//!   timeout exists anywhere. Stale park requests are withdrawn by pressure
//!   relief, not timers.
//! - **One Notify story.** register → re-check → await; the only timer in
//!   the module is the progress deadline itself. No fallback polls.
//!
//! Lock discipline: `inner` is a plain mutex ordered INSIDE everything else —
//! no [`PoolPort`] call (which may take store locks) is ever made while
//! holding it, no reservation or grant is dropped under it, and no await
//! happens under it. Every mutation funnels through `with_inner`, which
//! recomputes the packed summary word (the lock-free hot path) — counts are
//! computed at one chokepoint, never mirrored.

mod grant;
mod queue;
mod state;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

pub use grant::{AllocationGrant, Demand, DevicePageReservation, RsSlotReservation};
use queue::{Entry, EntryKey, EntryKind, Queue};
use state::{Proc, ProcEvent, ProcState};

/// Process identity the orchestrator tracks (FCFS clock key, victim pick,
/// wait-queue token). Kept as the leaf `uuid::Uuid` representation so the
/// store stays below the guest runtime in the layering.
pub type ProcessId = uuid::Uuid;

/// Whether a pipeline-leave was an allocation wait or a reclaim-driven
/// suspend. Kept separate from the scheduler's `LeaveKind` so the store
/// stays below the scheduler in the layering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LeaveKind {
    AllocationWait,
    Suspend,
}

/// Pipeline-leave hook. Forwards to the subscription the scheduler installs
/// via [`set_pipeline_leave_hook`] (a plain closure, so this module never
/// names `crate::scheduler`); a no-op until installed. Rejoin is implicit on
/// a pipeline's next wave request, so no join event exists.
pub(crate) fn notify_pipeline_leave(pid: ProcessId, kind: LeaveKind) {
    if let Some(hook) = PIPELINE_LEAVE_HOOK.get() {
        hook(pid, kind);
    }
}

type PipelineLeaveHook = Box<dyn Fn(ProcessId, LeaveKind) + Send + Sync>;
static PIPELINE_LEAVE_HOOK: OnceLock<PipelineLeaveHook> = OnceLock::new();

/// Installs the scheduler's wait-all leave subscription (called once, from
/// `bootstrap`).
pub(crate) fn set_pipeline_leave_hook(hook: impl Fn(ProcessId, LeaveKind) + Send + Sync + 'static) {
    let _ = PIPELINE_LEAVE_HOOK.set(Box::new(hook));
}

/// The contention subsystem's whole configuration, parsed from the
/// environment exactly once at the bootstrap edge ([`Self::from_env`]) and
/// injected explicitly everywhere else — no read-once statics, no knobs
/// consulted mid-drain. Tests construct it directly. Contention handling
/// has no off switch: preempt/restore IS the allocation path.
#[derive(Clone, Copy, Debug)]
pub struct ContentionConfig {
    /// The single system-wide progress deadline: how long the queue head may
    /// go without ANY progress toward it before the escalation ladder's
    /// final rungs fire (fail-loud today; D7 kill lands with B6).
    pub exhaustion: Duration,
    /// Restore attempts per grant before the failure surfaces.
    pub restore_retries: u32,
}

impl Default for ContentionConfig {
    fn default() -> Self {
        Self {
            exhaustion: Duration::from_secs(10),
            restore_retries: 3,
        }
    }
}

impl ContentionConfig {
    /// Parse the operator-facing environment (`PIE_KV_EXHAUSTION_MS`,
    /// `PIE_KV_RESTORE_RETRIES`). Called once, from bootstrap.
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            exhaustion: std::env::var("PIE_KV_EXHAUSTION_MS")
                .ok()
                .and_then(|value| value.parse().ok())
                .map_or(defaults.exhaustion, Duration::from_millis),
            restore_retries: std::env::var("PIE_KV_RESTORE_RETRIES")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(defaults.restore_retries)
                .max(1),
        }
    }
}

/// Minimal physical port the orchestrator drives — pool stats, rung-0 idle
/// reclaim, and concrete reservations. The orchestrator owns ALL policy;
/// this port owns only physics. One production impl ([`RegistryPool`]); unit
/// tests use a mock.
pub trait PoolPort: Send + Sync + 'static {
    /// `(free, total)` device pages of the contended pool.
    fn device_stats(&self) -> (u32, u32);
    /// `(free, total)` host swap slots.
    fn host_stats(&self) -> (u32, u32);
    /// `(free, total)` RS folded slots.
    fn rs_stats(&self) -> (u32, u32);
    /// Rung 0: reclaim whatever costs no work — cache-root leases nothing
    /// live reaches. Returns pages recycled NOW.
    fn reclaim_idle(&self) -> u32;
    /// Whether the driver can physically move KV bytes to/from host swap.
    /// Arms the suspend rung; without it the ladder is pool-only (a
    /// capability degradation, not a mode).
    fn suspend_capable(&self) -> bool;
    /// The `(model, driver)` pair this pool belongs to — the safe-point
    /// machinery derives its store and scheduler targets from this instead
    /// of hardwiring `(0, 0)`.
    fn locus(&self) -> (usize, usize);
    fn reserve_device(&self, count: u32) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>>;
    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>);
    fn reserve_rs(&self, count: u32) -> Option<Vec<crate::store::rs::RsSlotId>>;
    fn release_rs(&self, slots: Vec<crate::store::rs::RsSlotId>);
}

/// Production [`PoolPort`] over the typed per-(model, driver) stores.
pub struct RegistryPool {
    model: usize,
    driver: usize,
    suspend_capable: bool,
}

impl RegistryPool {
    pub fn new(model: usize, driver: usize, suspend_capable: bool) -> Self {
        Self {
            model,
            driver,
            suspend_capable,
        }
    }

    fn stores(&self) -> crate::store::registry::Stores {
        crate::store::registry::get(self.model, self.driver)
    }
}

impl PoolPort for RegistryPool {
    fn device_stats(&self) -> (u32, u32) {
        let stores = self.stores();
        crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
            (kv.available_pages() as u32, kv.capacity_pages())
        })
    }

    fn host_stats(&self) -> (u32, u32) {
        let stores = self.stores();
        crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
            (kv.host_swap_available() as u32, kv.host_swap_capacity())
        })
    }

    fn rs_stats(&self) -> (u32, u32) {
        let stores = self.stores();
        let rs = stores.rs.lock().unwrap();
        (rs.available_slots() as u32, rs.capacity_slots())
    }

    fn reclaim_idle(&self) -> u32 {
        let stores = self.stores();
        crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
            let epoch = kv.current_epoch();
            let freed = kv.drop_unused_cache_leases(epoch);
            if freed > 0 {
                kv.retire_idle();
            }
            freed as u32
        })
    }

    fn suspend_capable(&self) -> bool {
        self.suspend_capable
    }

    fn locus(&self) -> (usize, usize) {
        (self.model, self.driver)
    }

    fn reserve_device(
        &self,
        count: u32,
    ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>> {
        let stores = self.stores();
        crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
            kv.reserve_device_pages(count as usize)
        })
    }

    fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>) {
        let stores = self.stores();
        crate::store::registry::with_kv_lock(&stores.kv, "reclaim", |kv| {
            kv.release_device_reservation(pages);
        });
    }

    fn reserve_rs(&self, count: u32) -> Option<Vec<crate::store::rs::RsSlotId>> {
        let stores = self.stores();
        stores.rs.lock().unwrap().reserve_slots(count as usize)
    }

    fn release_rs(&self, slots: Vec<crate::store::rs::RsSlotId>) {
        let stores = self.stores();
        stores.rs.lock().unwrap().release_slot_reservation(slots);
    }
}

/// A hard (non-contention) failure from `acquire`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentionError {
    /// The request can never fit: `need` exceeds the pool's total capacity.
    /// Surfaced to the inferlet as a real error (fail loud).
    Impossible { need: u32, total: u32 },
    /// The queue head made NO progress for the whole deadline — no victim
    /// yielded, nothing freed, nothing completed. A non-terminating guest
    /// grew its context past what the pool can free. Fail LOUD to the guest
    /// (OutOfPages) instead of a silent wedge.
    Exhausted { need: u32, free: u32, total: u32 },
    /// The process was unregistered while waiting or parked.
    Cancelled,
}

impl std::fmt::Display for ContentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContentionError::Impossible { need, total } => write!(
                f,
                "allocation of {need} units can never fit (pool total {total})"
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

/// Engagement counters (lock-free reads) — the e2e's proof that contention
/// actually ENGAGED, so a trivially-passing run (pool never over-filled)
/// can't masquerade as a validated preempt/restore path.
#[derive(Debug, Default)]
pub struct ContentionStats {
    /// `acquire` calls that parked in the queue (demand not met immediately).
    pub waiters_parked: AtomicU64,
    /// Parked acquires that returned satisfied.
    pub waiters_woken: AtomicU64,
    /// Victims whose state-save freed pages (`report_suspended`).
    pub suspends: AtomicU64,
    /// Suspended processes restored back to Running.
    pub restores: AtomicU64,
    pub allocation_grants: AtomicU64,
    pub restore_grants: AtomicU64,
    pub cancelled_waits: AtomicU64,
    pub exhaustion_timeouts: AtomicU64,
    pub grace_deferred: AtomicU64,
    pub d2h_pages: AtomicU64,
    pub h2d_pages: AtomicU64,
    pub d2h_copy_us: AtomicU64,
    pub h2d_copy_us: AtomicU64,
    pub suspend_rollbacks: AtomicU64,
    pub restore_rollbacks: AtomicU64,
    pub host_swap_exhaustions: AtomicU64,
    pub restore_prepared: AtomicU64,
    pub h2d_submitted: AtomicU64,
    /// Suspended → restore-grant-issued waits (µs total; count = restore_grants).
    pub restore_wait_us: AtomicU64,
    /// Restore-grant-issued → rejoined-running turnarounds (µs total).
    pub restore_turnaround_us: AtomicU64,
    /// Count for [`Self::restore_turnaround_us`].
    pub restore_turnarounds: AtomicU64,
    /// Progress-deadline breaches escalated to a kill (D7).
    pub deadline_kills: AtomicU64,
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
    pub processes: Vec<ContentionQueueEntry>,
    pub waiters: Vec<ContentionQueueEntry>,
    pub suspended: Vec<ContentionQueueEntry>,
    pub device_pages_free: u32,
    pub device_pages_total: u32,
    pub device_pages_reserved: u32,
    pub host_slots_free: u32,
    pub host_slots_total: u32,
    pub park_requests_current: usize,
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
    pub restore_prepared_total: u64,
    pub h2d_submitted_total: u64,
    pub restore_wait_us_total: u64,
    pub restore_turnaround_us_total: u64,
    pub restore_turnarounds_total: u64,
    pub deadline_kills_total: u64,
}

/// A restore grant issued by the drain, parked until its process collects it.
struct IssuedRestore {
    grant: AllocationGrant,
    issued_at: Instant,
}

#[derive(Default)]
struct Inner {
    next_seq: u64,
    next_entry_id: u64,
    procs: HashMap<ProcessId, Proc>,
    queue: Queue,
    restore_grants: HashMap<ProcessId, IssuedRestore>,
    /// Restore-grant-collected → `report_restored` turnaround starts.
    restoring_since: HashMap<ProcessId, Instant>,
    /// The single progress clock: armed for the queue head, reset by any
    /// progress toward it. `(head entry key, armed at)`.
    progress: Option<(EntryKey, Instant)>,
}

/// Packed summary word layout (the lock-free hot path):
/// bits 0..16 allocation entries, 16..32 restore entries, 32..48 park
/// requests. Recomputed from `Inner` at the single `with_inner` chokepoint.
fn summarize(inner: &Inner) -> u64 {
    let (allocations, restores) = inner.queue.counts();
    let parks = inner
        .procs
        .values()
        .filter(|proc| proc.state == ProcState::ParkRequested)
        .count();
    (allocations.min(0xFFFF) as u64)
        | ((restores.min(0xFFFF) as u64) << 16)
        | ((parks.min(0xFFFF) as u64) << 32)
}

/// Kills a process through the runtime abort path (`terminate`: quiesce
/// first, GPU lifetime respected, transactions unwound by the RAII guards)
/// — never an OS signal, never a cleanup bypass. Installed by bootstrap;
/// absent in unit tests unless a test installs its own.
pub type KillHook = Box<dyn Fn(ProcessId, String) + Send + Sync>;

/// Pages only this process's working sets could free on the orchestrator's
/// (model, driver) — the D6 cost figure for victim selection. Installed by
/// bootstrap (it crosses layers the store must not name).
pub type FootprintProbe = Box<dyn Fn(ProcessId, usize, usize) -> Option<u32> + Send + Sync>;

pub struct ContentionOrchestrator {
    inner: Mutex<Inner>,
    /// See [`summarize`]. Readers may observe one transition late; every
    /// gate loops and re-checks under the lock after making progress.
    summary: AtomicU64,
    port: Arc<dyn PoolPort>,
    stats: ContentionStats,
    config: ContentionConfig,
    kill: OnceLock<KillHook>,
    footprint: OnceLock<FootprintProbe>,
}

/// One victim-selection candidate, snapshotted under the lock.
struct VictimCandidate {
    pid: ProcessId,
    seq: u64,
    idle: bool,
}

/// What a progress-deadline breach escalated to (D7).
enum BreachAction {
    /// A younger victim was killed; the head got a fresh deadline window.
    Killed,
    /// No younger work exists (or no kill path is wired): the head fails
    /// itself — the final rung.
    FailLoud,
}

/// One step of the drain: computed under the lock, executed against the
/// port outside it.
enum Fill {
    /// Reserve `count` device pages and attach them to entry `key`.
    Device { key: EntryKey, count: u32 },
    /// Entry `key`'s KV side is covered; reserve `rs` slots and issue its
    /// grant.
    Finish { key: EntryKey, rs: u32 },
    /// A younger entry's stranded partial accumulation moved to the head
    /// (pure bookkeeping under the lock); re-scan.
    Reclaimed,
}

struct WaitRegistration<'a> {
    orchestrator: &'a ContentionOrchestrator,
    pid: ProcessId,
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
            self.orchestrator.cancel_waiter(self.pid, self.key, true);
        }
    }
}

impl ContentionOrchestrator {
    pub fn new(port: Arc<dyn PoolPort>, config: ContentionConfig) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            summary: AtomicU64::new(0),
            port,
            stats: ContentionStats::default(),
            config,
            kill: OnceLock::new(),
            footprint: OnceLock::new(),
        }
    }

    /// Install the runtime kill path (bootstrap, once).
    pub fn set_kill_hook(&self, hook: impl Fn(ProcessId, String) + Send + Sync + 'static) {
        let _ = self.kill.set(Box::new(hook));
    }

    /// Install the victim footprint probe (bootstrap, once).
    pub fn set_footprint_probe(
        &self,
        probe: impl Fn(ProcessId, usize, usize) -> Option<u32> + Send + Sync + 'static,
    ) {
        let _ = self.footprint.set(Box::new(probe));
    }

    /// Self-reported by the process side: `pid` entered (or left) a long
    /// idle host await — a permanent safe point, and D6's preferred victim
    /// class.
    pub fn note_idle_await(&self, pid: ProcessId, entered: bool) {
        self.with_inner(|inner| {
            if let Some(proc) = inner.procs.get_mut(&pid) {
                proc.idle = entered;
            }
        });
    }

    /// The injected configuration (read-only; parsed once at bootstrap).
    pub fn config(&self) -> ContentionConfig {
        self.config
    }

    /// The `(model, driver)` pair this orchestrator manages.
    pub fn locus(&self) -> (usize, usize) {
        self.port.locus()
    }

    /// Engagement counters (see [`ContentionStats`]).
    pub fn stats(&self) -> &ContentionStats {
        &self.stats
    }

    /// The single mutation chokepoint: every change to `Inner` goes through
    /// here, and the packed summary word is recomputed on the way out — one
    /// writer site, counts computed rather than mirrored.
    fn with_inner<R>(&self, f: impl FnOnce(&mut Inner) -> R) -> R {
        let mut inner = self.inner.lock().unwrap();
        let result = f(&mut inner);
        self.summary.store(summarize(&inner), Ordering::Release);
        result
    }

    /// True while any allocation waiter is queued — the pool is contended
    /// and deep-running lanes should self-serialize so their pins release.
    pub fn contended(&self) -> bool {
        self.summary.load(Ordering::Acquire) & 0xFFFF != 0
    }

    /// Register a process at spawn — its registration order is the FCFS
    /// clock, the single authoritative service order.
    pub fn register(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            let seq = inner.next_seq;
            inner.next_seq += 1;
            inner.procs.insert(pid, Proc::new(seq));
        });
    }

    /// Unregister at process exit/terminate. Its queue entries are removed,
    /// a parked task is woken for teardown, and freed capacity drains to the
    /// queue.
    pub fn unregister(&self, pid: ProcessId) {
        let (signal, entries, issued) = self.with_inner(|inner| {
            let signal = inner.procs.remove(&pid).map(|proc| proc.signal);
            let entries = inner.queue.extract(|entry| entry.pid == pid);
            let issued = inner.restore_grants.remove(&pid);
            inner.restoring_since.remove(&pid);
            inner.progress = None;
            (signal, entries, issued)
        });
        for entry in &entries {
            if let EntryKind::Allocation { notify, .. } = &entry.kind {
                notify.notify_waiters();
            }
        }
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
        // Reservations and grants return to the pool here, outside the lock.
        drop(entries);
        drop(issued);
        self.drain();
    }

    /// Pages or slots freed somewhere (a fire finalized, a process exited, a
    /// working set released). Drains if anyone is waiting.
    pub fn on_blocks_freed(&self) {
        if self.summary.load(Ordering::Acquire) & 0xFFFF_FFFF != 0 {
            self.drain();
        }
    }

    pub fn is_registered(&self, pid: ProcessId) -> bool {
        self.inner.lock().unwrap().procs.contains_key(&pid)
    }

    // =========================================================================
    // The acquisition path
    // =========================================================================

    /// Acquire one grant. The wait inside is a STANDING safe point (D2):
    /// the requester holds nothing, so the caller must race this future
    /// against the process's park signal and, when a request lands, drop
    /// it, honor the request (suspend/restore), and re-acquire — victim and
    /// requester are the same code at the same boundary (D3). Rank is the
    /// spawn clock, so cancel-and-retry never loses queue position.
    pub async fn acquire(
        &self,
        requester: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<AllocationGrant, ContentionError> {
        let (_, kv_total) = self.port.device_stats();
        if demand.kv_pages > kv_total {
            return Err(ContentionError::Impossible {
                need: demand.kv_pages,
                total: kv_total,
            });
        }
        if demand.rs_slots > 0 {
            let (_, rs_total) = self.port.rs_stats();
            if demand.rs_slots > rs_total {
                return Err(ContentionError::Impossible {
                    need: demand.rs_slots,
                    total: rs_total,
                });
            }
        }
        'register: loop {
            let (key, notify) = match self.register_waiter(requester, quorum_id, demand) {
                Ok(registered) => registered,
                Err(ContentionError::Cancelled) if self.is_registered(requester) => {
                    // Not Running — typically the requester itself was
                    // selected as a victim (ParkRequested). Wait; the caller
                    // observes the park signal at its safe point, cancels
                    // this future, and yields.
                    self.wait_until_running(requester).await?;
                    continue 'register;
                }
                Err(error) => return Err(error),
            };
            let mut registration = WaitRegistration {
                orchestrator: self,
                pid: requester,
                key,
                active: true,
            };
            loop {
                self.drain();
                if let Some(grant) = self.take_allocation_grant(requester, key, demand) {
                    registration.disarm();
                    return Ok(grant);
                }
                if !self.is_registered(requester) {
                    registration.disarm();
                    return Err(ContentionError::Cancelled);
                }
                if !self.has_entry(key) {
                    // Displaced — a park request landed on this process (it
                    // was selected as a victim) or a duplicate collected the
                    // grant. Fall back to registration, which resolves both.
                    registration.disarm();
                    continue 'register;
                }
                if self.port.reclaim_idle() > 0 {
                    continue; // rung 0 freed pages; re-drain before escalating
                }
                self.escalate();
                self.mark_waiter_parked(requester, key);
                let notified = notify.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                // Lost-wakeup guard: re-check after arming the waiter.
                if self.grant_ready(key) || !self.has_entry(key) {
                    continue;
                }
                match self.head_deadline(key) {
                    None => notified.await,
                    Some(deadline) => {
                        let sleep =
                            tokio::time::sleep(deadline.saturating_duration_since(Instant::now()));
                        tokio::pin!(sleep);
                        tokio::select! {
                            _ = &mut notified => {}
                            _ = &mut sleep => {
                                if let Some(error) = self.progress_breach(key) {
                                    match self.escalate_breach(key, &error) {
                                        // A kill reset the clock; keep waiting
                                        // for the teardown's frees.
                                        BreachAction::Killed => {}
                                        BreachAction::FailLoud => {
                                            registration.disarm();
                                            self.cancel_waiter(requester, key, false);
                                            self.stats
                                                .exhaustion_timeouts
                                                .fetch_add(1, Ordering::Relaxed);
                                            return Err(error);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn register_waiter(
        &self,
        pid: ProcessId,
        quorum_id: ProcessId,
        demand: Demand,
    ) -> Result<(EntryKey, Arc<Notify>), ContentionError> {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get(&pid) else {
                return Err(ContentionError::Cancelled);
            };
            if proc.state != ProcState::Running {
                return Err(ContentionError::Cancelled);
            }
            let seq = proc.seq;
            if let Some((key, entry)) = inner.queue.find_allocation(pid, quorum_id) {
                let key = *key;
                let EntryKind::Allocation {
                    notify,
                    waiters,
                    grant,
                    ..
                } = &mut entry.kind
                else {
                    unreachable!("find_allocation returns allocation entries");
                };
                let notify = notify.clone();
                *waiters += 1;
                if grant.is_none() {
                    entry.demand = entry.demand.max(demand);
                }
                return Ok((key, notify));
            }
            let id = inner.next_entry_id;
            inner.next_entry_id += 1;
            let notify = Arc::new(Notify::new());
            inner.queue.insert(
                (seq, id),
                Entry {
                    pid,
                    demand,
                    kind: EntryKind::Allocation {
                        quorum_id,
                        notify: notify.clone(),
                        waiters: 1,
                        parked_reported: false,
                        grant: None,
                    },
                    kv_accum: DevicePageReservation::empty(),
                },
            );
            Ok(((seq, id), notify))
        })
    }

    fn has_entry(&self, key: EntryKey) -> bool {
        self.inner.lock().unwrap().queue.get(&key).is_some()
    }

    fn grant_ready(&self, key: EntryKey) -> bool {
        self.inner
            .lock()
            .unwrap()
            .queue
            .get(&key)
            .is_some_and(|entry| {
                matches!(&entry.kind, EntryKind::Allocation { grant: Some(_), .. })
            })
    }

    fn take_allocation_grant(
        &self,
        pid: ProcessId,
        key: EntryKey,
        demand: Demand,
    ) -> Option<AllocationGrant> {
        let taken = self.with_inner(|inner| {
            let entry = inner.queue.get(&key)?;
            if entry.pid != pid {
                return None;
            }
            let EntryKind::Allocation { grant, .. } = &entry.kind else {
                return None;
            };
            if grant.as_ref().is_none_or(|g| !g.demand().covers(demand)) {
                return None;
            }
            inner.queue.remove(&key)
        })?;
        self.stats.waiters_woken.fetch_add(1, Ordering::Relaxed);
        let EntryKind::Allocation { grant, .. } = taken.kind else {
            unreachable!("checked above");
        };
        grant
    }

    fn cancel_waiter(&self, pid: ProcessId, key: EntryKey, cancelled: bool) {
        let removed = self.with_inner(|inner| {
            let entry = inner.queue.get_mut(&key)?;
            if entry.pid != pid {
                return None;
            }
            if let EntryKind::Allocation { waiters, .. } = &mut entry.kind {
                if *waiters > 1 {
                    *waiters -= 1;
                    return None;
                }
            }
            inner.queue.remove(&key)
        });
        if cancelled {
            self.stats.cancelled_waits.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(entry) = removed {
            drop(entry); // reservations/grants return outside the lock
            self.drain();
        }
    }

    fn mark_waiter_parked(&self, pid: ProcessId, key: EntryKey) {
        let quorum = self.with_inner(|inner| {
            let entry = inner.queue.get_mut(&key)?;
            if entry.pid != pid {
                return None;
            }
            let EntryKind::Allocation {
                quorum_id,
                parked_reported,
                ..
            } = &mut entry.kind
            else {
                return None;
            };
            (!std::mem::replace(parked_reported, true)).then_some(*quorum_id)
        });
        if let Some(quorum_id) = quorum {
            self.stats.waiters_parked.fetch_add(1, Ordering::Relaxed);
            notify_pipeline_leave(quorum_id, LeaveKind::AllocationWait);
        }
    }

    async fn wait_until_running(&self, pid: ProcessId) -> Result<(), ContentionError> {
        loop {
            let signal = {
                let inner = self.inner.lock().unwrap();
                let Some(proc) = inner.procs.get(&pid) else {
                    return Err(ContentionError::Cancelled);
                };
                if proc.state == ProcState::Running {
                    return Ok(());
                }
                proc.signal.clone()
            };
            let notified = signal.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            {
                let inner = self.inner.lock().unwrap();
                let Some(proc) = inner.procs.get(&pid) else {
                    return Err(ContentionError::Cancelled);
                };
                if proc.state == ProcState::Running {
                    return Ok(());
                }
            }
            notified.await;
        }
    }

    // =========================================================================
    // The drain — head-first-claim (D1)
    // =========================================================================

    fn drain(&self) {
        loop {
            self.withdraw_stale_park_requests();
            let (free, _) = self.port.device_stats();
            let step = self.with_inner(|inner| {
                let mut head: Option<(EntryKey, u32)> = None;
                let mut donor: Option<EntryKey> = None;
                for (key, entry) in inner.queue.iter() {
                    if !entry.is_unmet() {
                        continue;
                    }
                    let missing = entry.kv_missing();
                    if missing == 0 {
                        return Some(Fill::Finish {
                            key: *key,
                            rs: entry.demand.rs_slots,
                        });
                    }
                    if head.is_none() {
                        head = Some((*key, missing));
                        // With no pages for the head, younger entries may
                        // still Finish (RS-only demand) or hold a stranded
                        // accumulation — keep scanning.
                    } else if donor.is_none() && entry.kv_accum.len() > 0 {
                        // A younger entry partially accumulated while IT was
                        // the head, before this older process re-entered the
                        // boundary. Head-first-claim says those pages belong
                        // to the current head — stranding them would wedge
                        // the queue with a full pool.
                        donor = Some(*key);
                    }
                }
                let (head_key, missing) = head?;
                if free > 0 {
                    // Head-first-claim: every free page flows to the oldest
                    // unmet entry until its demand is covered.
                    return Some(Fill::Device {
                        key: head_key,
                        count: free.min(missing),
                    });
                }
                let donor = donor?;
                let moved = inner
                    .queue
                    .transfer_accum(&donor, &head_key, missing as usize);
                if moved == 0 {
                    return None;
                }
                inner.progress = None; // pages flowed to the head
                Some(Fill::Reclaimed)
            });
            match step {
                None => return,
                Some(Fill::Reclaimed) => continue,
                Some(Fill::Device { key, count }) => {
                    let Some(pages) = self.port.reserve_device(count) else {
                        return; // lost a race to the pool; the next free event re-drains
                    };
                    let reservation = DevicePageReservation::new(pages, self.port.clone());
                    let leftover = self.with_inner(|inner| {
                        match inner.queue.get_mut(&key) {
                            Some(entry) if entry.is_unmet() => {
                                entry.kv_accum.absorb(reservation);
                                // Pages flowed to the head: progress.
                                inner.progress = None;
                                None
                            }
                            _ => Some(reservation),
                        }
                    });
                    drop(leftover);
                }
                Some(Fill::Finish { key, rs }) => {
                    let rs_reservation = if rs > 0 {
                        let Some(slots) = self.port.reserve_rs(rs) else {
                            return; // RS pool short; the next RS free re-drains
                        };
                        RsSlotReservation::new(slots, self.port.clone())
                    } else {
                        RsSlotReservation::empty()
                    };
                    let outcome = self.finish_entry(key, rs_reservation);
                    match outcome {
                        FinishOutcome::Granted(notify) => {
                            notify.notify_waiters();
                            self.stats.allocation_grants.fetch_add(1, Ordering::Relaxed);
                        }
                        FinishOutcome::RestoreIssued { signal, waited } => {
                            signal.notify_waiters();
                            self.stats.restore_grants.fetch_add(1, Ordering::Relaxed);
                            self.stats.restore_wait_us.fetch_add(
                                waited.as_micros().min(u128::from(u64::MAX)) as u64,
                                Ordering::Relaxed,
                            );
                        }
                        FinishOutcome::Vanished(reservation, stale_entry) => {
                            drop(reservation);
                            drop(stale_entry);
                        }
                    }
                }
            }
        }
    }

    fn finish_entry(&self, key: EntryKey, rs_reservation: RsSlotReservation) -> FinishOutcome {
        // Reservations, grants, and displaced entries must never drop under
        // the inner lock (their Drop re-locks the pool). Anything this can't
        // consume is handed back out as `Vanished` and dropped by the drain.
        self.with_inner(|inner| {
            let Some(entry) = inner.queue.get(&key) else {
                return FinishOutcome::Vanished(rs_reservation, None);
            };
            if !entry.is_unmet() || entry.kv_missing() != 0 {
                return FinishOutcome::Vanished(rs_reservation, None);
            }
            // For a restore entry, the process must still be suspendable;
            // validate BEFORE the entry is consumed.
            if matches!(entry.kind, EntryKind::Restore { .. }) {
                let ready = inner
                    .procs
                    .get(&entry.pid)
                    .is_some_and(|proc| proc.state.apply(ProcEvent::GrantRestore).is_ok());
                if !ready {
                    // Teardown raced the grant; drop the stale entry too.
                    let stale = inner.queue.remove(&key);
                    return FinishOutcome::Vanished(rs_reservation, stale);
                }
            }
            let mut entry = inner.queue.remove(&key).expect("entry present");
            let kv = std::mem::replace(&mut entry.kv_accum, DevicePageReservation::empty());
            let grant = AllocationGrant::new(entry.pid, key.1, entry.demand, kv, rs_reservation);
            inner.progress = None; // a grant is progress by definition
            match entry.kind {
                EntryKind::Allocation {
                    quorum_id,
                    notify,
                    waiters,
                    parked_reported,
                    grant: existing,
                } => {
                    debug_assert!(existing.is_none(), "unmet entries carry no grant");
                    // Re-insert with the grant attached; the waiting future
                    // collects it via take_allocation_grant.
                    inner.queue.insert(
                        key,
                        Entry {
                            pid: entry.pid,
                            demand: entry.demand,
                            kind: EntryKind::Allocation {
                                quorum_id,
                                notify: notify.clone(),
                                waiters,
                                parked_reported,
                                grant: Some(grant),
                            },
                            kv_accum: DevicePageReservation::empty(),
                        },
                    );
                    FinishOutcome::Granted(notify)
                }
                EntryKind::Restore { suspended_at } => {
                    let pid = entry.pid;
                    let proc = inner.procs.get_mut(&pid).expect("validated above");
                    proc.state = proc
                        .state
                        .apply(ProcEvent::GrantRestore)
                        .expect("validated above");
                    let signal = proc.signal.clone();
                    inner.restore_grants.insert(
                        pid,
                        IssuedRestore {
                            grant,
                            issued_at: Instant::now(),
                        },
                    );
                    FinishOutcome::RestoreIssued {
                        signal,
                        waited: suspended_at.elapsed(),
                    }
                }
            }
        })
    }

    // =========================================================================
    // Escalation — the victim ladder (rung 1: suspend younger-than-head)
    // =========================================================================

    /// Post a park request to one victim younger than the queue head (D4's
    /// net-progress rule: suspending the head or its elders cannot make net
    /// progress — and the keystone exemption falls out as a theorem, since
    /// the globally-oldest process is never younger than any head).
    ///
    /// Eligibility is EVERY pin-free process younger than the head — hog
    /// reclaim: a young process holding a large resident footprint while
    /// making steady progress is a victim like any other; "pin-free" is
    /// enforced where the request is honored (the safe point), not here.
    ///
    /// Selection is cost-aware inside the FCFS invariant (D6): suspension
    /// is whole-process, so an oversized victim over-evicts. Prefer idle
    /// holders (no running lane lost), then the smallest footprint that
    /// covers the head's residual demand; tie-break youngest. Service
    /// order stays the spawn clock regardless of who is picked.
    fn escalate(&self) {
        if !self.port.suspend_capable() {
            return;
        }
        // Phase 1 (lock): the head's residual and the candidate set.
        let Some((residual, candidates)) = self.with_inner(|inner| {
            let (key, entry) = inner.queue.head()?;
            let missing = entry.kv_missing();
            if missing == 0 {
                return None; // the head only waits for RS or collection
            }
            let head_seq = key.0;
            let candidates: Vec<VictimCandidate> = inner
                .procs
                .iter()
                .filter(|(_, proc)| {
                    proc.state == ProcState::Running && !proc.killed && proc.seq > head_seq
                })
                .map(|(pid, proc)| VictimCandidate {
                    pid: *pid,
                    seq: proc.seq,
                    idle: proc.idle,
                })
                .collect();
            (!candidates.is_empty()).then_some((missing, candidates))
        }) else {
            return;
        };
        // Phase 2 (no lock — the probe takes store locks): D6 cost scoring.
        let victim = self.select_costed(residual, candidates);
        // Phase 3 (lock): re-validate against the current head and post.
        let notifications = self.with_inner(|inner| {
            let head_seq = {
                let (key, entry) = inner.queue.head()?;
                if entry.kv_missing() == 0 {
                    return None;
                }
                key.0
            };
            let Inner { procs, queue, .. } = &mut *inner;
            let proc = procs.get_mut(&victim)?;
            if proc.state != ProcState::Running || proc.killed || proc.seq <= head_seq {
                return None;
            }
            let Ok(next) = proc.state.apply(ProcEvent::RequestPark) else {
                return None;
            };
            proc.state = next;
            let park_signal = proc.signal.clone();
            // Displace the victim's queue entries: it can no longer be
            // served (its acquire will observe the park request and
            // self-suspend; its pages come back through report_suspended).
            let displaced = queue.extract(|entry| entry.pid == victim);
            Some((park_signal, displaced))
        });
        if let Some((park_signal, displaced)) = notifications {
            park_signal.notify_waiters();
            for entry in &displaced {
                if let EntryKind::Allocation { notify, .. } = &entry.kind {
                    notify.notify_waiters();
                }
            }
            drop(displaced); // accumulated reservations return outside the lock
        }
    }

    /// D6 cost order over a non-empty candidate set: idle holders first;
    /// then the smallest footprint covering `residual` (a covering victim
    /// ends the reclaim in one suspension), then the largest partial
    /// footprint (most progress per eviction), unknown footprints last of
    /// their class; ties break youngest.
    fn select_costed(&self, residual: u32, candidates: Vec<VictimCandidate>) -> ProcessId {
        let (model, driver) = self.port.locus();
        let probe = self.footprint.get();
        candidates
            .into_iter()
            .min_by_key(|candidate| {
                let footprint = probe.and_then(|probe| probe(candidate.pid, model, driver));
                let (class, cost) = match footprint {
                    Some(f) if f >= residual => (0u8, i64::from(f)),
                    Some(f) => (1, -i64::from(f)),
                    None => (2, 0),
                };
                (!candidate.idle, class, cost, u64::MAX - candidate.seq)
            })
            .map(|candidate| candidate.pid)
            .expect("candidates are non-empty")
    }

    /// D7 — the breach ladder: destruction before failure, youngest-
    /// necessary first, never the head or its elders. Prefer an already-
    /// selected victim that never reached its safe point (provably
    /// uncooperative), else the D6 cost pick over the running set.
    /// Suspended/quiescing processes are not targets — they already yielded
    /// (or are yielding) their pages. Fail-loud survives only as the final
    /// rung, when no younger work remains.
    fn escalate_breach(&self, key: EntryKey, error: &ContentionError) -> BreachAction {
        let Some(kill) = self.kill.get() else {
            return BreachAction::FailLoud;
        };
        let Some((residual, uncooperative, running)) = self.with_inner(|inner| {
            let (head_key, entry) = inner.queue.head()?;
            if *head_key != key {
                return None; // the breach went stale under us
            }
            let head_seq = head_key.0;
            let mut uncooperative: Vec<VictimCandidate> = Vec::new();
            let mut running: Vec<VictimCandidate> = Vec::new();
            for (pid, proc) in inner.procs.iter() {
                if proc.killed || proc.seq <= head_seq {
                    continue;
                }
                let candidate = VictimCandidate {
                    pid: *pid,
                    seq: proc.seq,
                    idle: proc.idle,
                };
                match proc.state {
                    ProcState::ParkRequested => uncooperative.push(candidate),
                    ProcState::Running => running.push(candidate),
                    _ => {}
                }
            }
            Some((entry.kv_missing(), uncooperative, running))
        }) else {
            return BreachAction::FailLoud;
        };
        let victim = if let Some(worst) = uncooperative
            .into_iter()
            .max_by_key(|candidate| candidate.seq)
        {
            Some(worst.pid)
        } else if running.is_empty() {
            None
        } else {
            Some(self.select_costed(residual, running))
        };
        let Some(victim) = victim else {
            return BreachAction::FailLoud;
        };
        let marked = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&victim) else {
                // It unregistered under us — its frees are the progress.
                inner.progress = None;
                return true;
            };
            if proc.killed {
                return false;
            }
            proc.killed = true;
            // Destruction is the progress action here: the teardown's frees
            // get a fresh deadline window.
            inner.progress = None;
            true
        });
        if !marked {
            return BreachAction::FailLoud;
        }
        self.stats.deadline_kills.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            victim = %victim,
            "KV contention progress deadline breached: killing the \
             youngest-necessary victim through the runtime abort path (D7)"
        );
        kill(
            victim,
            format!("KV contention progress deadline expired: {error}"),
        );
        BreachAction::Killed
    }

    /// Pressure relief, not timers: when nothing is waiting for pages any
    /// more, pending park requests are withdrawn.
    fn withdraw_stale_park_requests(&self) {
        if (self.summary.load(Ordering::Acquire) >> 32) & 0xFFFF == 0 {
            return;
        }
        let signals = self.with_inner(|inner| {
            if inner.queue.head().is_some() {
                return Vec::new();
            }
            let mut signals = Vec::new();
            for proc in inner.procs.values_mut() {
                if proc.state == ProcState::ParkRequested {
                    if let Ok(next) = proc.state.apply(ProcEvent::DeclinePark) {
                        proc.state = next;
                        signals.push(proc.signal.clone());
                    }
                }
            }
            signals
        });
        for signal in signals {
            signal.notify_waiters();
        }
    }

    // =========================================================================
    // The single progress deadline (D4)
    // =========================================================================

    /// If `key` is the current queue head, arm (or read) the progress clock
    /// and return its deadline. Non-heads run no clock — their rank only
    /// improves.
    fn head_deadline(&self, key: EntryKey) -> Option<Instant> {
        self.with_inner(|inner| {
            let (head_key, _) = inner.queue.head()?;
            if *head_key != key {
                return None;
            }
            let head_key = *head_key;
            let since = match inner.progress {
                Some((armed_for, since)) if armed_for == head_key => since,
                _ => {
                    let now = Instant::now();
                    inner.progress = Some((head_key, now));
                    now
                }
            };
            Some(since + self.config.exhaustion)
        })
    }

    /// Validate a deadline wake: still the head, still armed, really expired.
    fn progress_breach(&self, key: EntryKey) -> Option<ContentionError> {
        let (free, total) = self.port.device_stats();
        self.with_inner(|inner| {
            let (head_key, head) = inner.queue.head()?;
            if *head_key != key {
                return None;
            }
            let (armed_for, since) = inner.progress?;
            if armed_for != *head_key || since.elapsed() < self.config.exhaustion {
                return None;
            }
            Some(ContentionError::Exhausted {
                need: head.demand.kv_pages,
                // Pages accumulated toward the failing head return to the
                // pool with it — report them as free.
                free: free + head.kv_accum.len() as u32,
                total,
            })
        })
    }

    // =========================================================================
    // Self-suspend protocol — the victim-side surface.
    //
    // A running victim's working set is process-local (wasmtime
    // ResourceTable); no cross-process access exists by design. An active
    // preempt is a handshake driven by the VICTIM's own task at its next
    // safe point:
    //
    //   1. `should_park(pid)` — cheap check at every safe point.
    //   2. `begin_quiesce(pid)` — the commit point; then the victim saves
    //      its own state and calls `report_suspended(pid, freed)`.
    //   3. `park_until_restore_granted(pid).await` — waits for concrete
    //      device ids, restores H2D, republishes, then `report_restored`.
    // =========================================================================

    /// Victim-side step 1: does `pid` have a pending park request?
    pub fn should_park(&self, pid: ProcessId) -> bool {
        if (self.summary.load(Ordering::Acquire) >> 32) & 0xFFFF == 0 {
            return false;
        }
        matches!(
            self.inner.lock().unwrap().procs.get(&pid).map(|p| p.state),
            Some(ProcState::ParkRequested)
        )
    }

    /// The process's state-change signal, raced against long host awaits so
    /// an idle process honors a park request without another guest call.
    pub fn park_signal(&self, pid: ProcessId) -> Option<Arc<Notify>> {
        self.inner
            .lock()
            .unwrap()
            .procs
            .get(&pid)
            .map(|proc| proc.signal.clone())
    }

    /// Victim-side step 2 — the commit point: `ParkRequested → Quiescing`.
    /// Re-checks the pressure that justified the request (D4): the park is
    /// honored only while an OLDER unmet head still exists; a stale request
    /// is withdrawn right here — pressure relief, not timers.
    pub fn begin_quiesce(&self, pid: ProcessId) -> bool {
        let (accepted, withdrawn) = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get(&pid) else {
                return (false, None);
            };
            if proc.state != ProcState::ParkRequested {
                return (false, None);
            }
            let seq = proc.seq;
            let still_needed = inner.queue.head().is_some_and(|(key, _)| key.0 < seq);
            let proc = inner.procs.get_mut(&pid).expect("checked above");
            if !still_needed {
                if let Ok(next) = proc.state.apply(ProcEvent::DeclinePark) {
                    proc.state = next;
                }
                return (false, Some(proc.signal.clone()));
            }
            match proc.state.apply(ProcEvent::BeginQuiesce) {
                Ok(next) => {
                    proc.state = next;
                    (true, None)
                }
                Err(_) => (false, None),
            }
        });
        if let Some(signal) = withdrawn {
            signal.notify_waiters();
        }
        accepted
    }

    pub fn leave_for_suspend(&self, pid: ProcessId) {
        notify_pipeline_leave(pid, LeaveKind::Suspend);
    }

    /// Victim-side DECLINE: the victim reached its safe point but has
    /// nothing to yield right now. Clears the park request so the state
    /// machine doesn't leak a permanently-flagged process; it stays a future
    /// candidate.
    pub fn decline_park(&self, pid: ProcessId) {
        let signal = self.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid)?;
            match proc.state.apply(ProcEvent::DeclinePark) {
                Ok(next) => {
                    proc.state = next;
                    Some(proc.signal.clone())
                }
                Err(_) => None,
            }
        });
        if let Some(signal) = signal {
            signal.notify_waiters();
        }
    }

    /// Victim-side step 2b: the process saved its own state and freed
    /// `freed_now` pages. Marks it suspended, queues its restore entry at
    /// its spawn position, and drains (its freed pages flow to the head).
    pub fn report_suspended(&self, pid: ProcessId, freed_now: u32) {
        let accepted = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return false;
            };
            let Ok(next) = proc.state.apply(ProcEvent::ReportSuspended) else {
                return false;
            };
            proc.state = next;
            proc.restore_demand = freed_now;
            let seq = proc.seq;
            if inner.queue.find_restore(pid).is_none() {
                let id = inner.next_entry_id;
                inner.next_entry_id += 1;
                inner.queue.insert(
                    (seq, id),
                    Entry {
                        pid,
                        demand: Demand::kv(freed_now),
                        kind: EntryKind::Restore {
                            suspended_at: Instant::now(),
                        },
                        kv_accum: DevicePageReservation::empty(),
                    },
                );
            }
            inner.progress = None; // a suspend IS progress toward the head
            true
        });
        if accepted {
            self.stats.suspends.fetch_add(1, Ordering::Relaxed);
            self.stats
                .d2h_pages
                .fetch_add(u64::from(freed_now), Ordering::Relaxed);
        }
        self.drain();
    }

    /// Victim-side step 3: park until the drain issues this process's
    /// restore grant. No fallback poll: the only timer is the progress
    /// deadline, armed only while this restore entry is the queue head.
    pub async fn park_until_restore_granted(
        &self,
        pid: ProcessId,
    ) -> Result<AllocationGrant, ContentionError> {
        loop {
            self.drain();
            enum Wait {
                Ready(Box<AllocationGrant>),
                Gone,
                Park(Arc<Notify>, Option<EntryKey>),
            }
            let wait = self.with_inner(|inner| {
                if let Some(issued) = inner.restore_grants.remove(&pid) {
                    inner.restoring_since.insert(pid, issued.issued_at);
                    return Wait::Ready(Box::new(issued.grant));
                }
                let Some(proc) = inner.procs.get(&pid) else {
                    return Wait::Gone;
                };
                let key = inner.queue.find_restore(pid).map(|(key, _)| *key);
                Wait::Park(proc.signal.clone(), key)
            });
            let (signal, key) = match wait {
                Wait::Ready(grant) => return Ok(*grant),
                Wait::Gone => return Err(ContentionError::Cancelled),
                Wait::Park(signal, key) => (signal, key),
            };
            if self.port.reclaim_idle() > 0 {
                continue; // rung 0 freed pages; re-drain
            }
            self.escalate();
            let notified = signal.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Lost-wakeup guard: the grant may have landed after the check.
            if self.restore_ready(pid) {
                continue;
            }
            match key.and_then(|key| self.head_deadline(key)) {
                None => notified.await,
                Some(deadline) => {
                    let sleep =
                        tokio::time::sleep(deadline.saturating_duration_since(Instant::now()));
                    tokio::pin!(sleep);
                    tokio::select! {
                        _ = &mut notified => {}
                        _ = &mut sleep => {
                            let key = key.expect("deadline implies key");
                            if let Some(error) = self.progress_breach(key) {
                                match self.escalate_breach(key, &error) {
                                    // A kill reset the clock; keep waiting
                                    // for the teardown's frees.
                                    BreachAction::Killed => {}
                                    BreachAction::FailLoud => {
                                        let removed = self.with_inner(|inner| {
                                            inner.queue.remove(&key)
                                        });
                                        drop(removed);
                                        self.stats
                                            .exhaustion_timeouts
                                            .fetch_add(1, Ordering::Relaxed);
                                        return Err(error);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn restore_ready(&self, pid: ProcessId) -> bool {
        self.inner.lock().unwrap().restore_grants.contains_key(&pid)
    }

    /// Restore completed: H2D done, mapping republished, process runnable.
    pub fn report_restored(&self, pid: ProcessId, restored_pages: u32) {
        let (signal, turnaround) = self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return (None, None);
            };
            let Ok(next) = proc.state.apply(ProcEvent::ReportRestored) else {
                return (None, None);
            };
            proc.state = next;
            proc.restore_demand = 0;
            inner.progress = None; // a restore IS progress
            (
                Some(proc.signal.clone()),
                inner.restoring_since.remove(&pid),
            )
        });
        let Some(signal) = signal else { return };
        signal.notify_waiters();
        self.stats.restores.fetch_add(1, Ordering::Relaxed);
        self.stats
            .h2d_pages
            .fetch_add(u64::from(restored_pages), Ordering::Relaxed);
        if let Some(since) = turnaround {
            self.stats.restore_turnaround_us.fetch_add(
                since.elapsed().as_micros().min(u128::from(u64::MAX)) as u64,
                Ordering::Relaxed,
            );
            self.stats
                .restore_turnarounds
                .fetch_add(1, Ordering::Relaxed);
        }
        self.drain();
    }

    /// A restore attempt failed; the process re-queues as suspended at its
    /// spawn position (FCFS is authoritative for restores too).
    pub fn report_restore_failed(&self, pid: ProcessId) {
        self.with_inner(|inner| {
            let Some(proc) = inner.procs.get_mut(&pid) else {
                return;
            };
            let Ok(next) = proc.state.apply(ProcEvent::RestoreFailed) else {
                return;
            };
            proc.state = next;
            inner.restoring_since.remove(&pid);
            let seq = proc.seq;
            let demand = Demand::kv(proc.restore_demand);
            if inner.queue.find_restore(pid).is_none() {
                let id = inner.next_entry_id;
                inner.next_entry_id += 1;
                inner.queue.insert(
                    (seq, id),
                    Entry {
                        pid,
                        demand,
                        kind: EntryKind::Restore {
                            suspended_at: Instant::now(),
                        },
                        kv_accum: DevicePageReservation::empty(),
                    },
                );
            }
        });
        self.drain();
    }

    // =========================================================================
    // Telemetry
    // =========================================================================

    pub fn record_d2h_copy(&self, elapsed: Duration) {
        self.stats.d2h_copy_us.fetch_add(
            elapsed.as_micros().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }

    pub fn record_h2d_copy(&self, elapsed: Duration) {
        self.stats.h2d_copy_us.fetch_add(
            elapsed.as_micros().min(u128::from(u64::MAX)) as u64,
            Ordering::Relaxed,
        );
    }

    pub fn record_grace_deferred(&self) {
        self.stats.grace_deferred.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_suspend_rollback(&self) {
        self.stats.suspend_rollbacks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_restore_rollback(&self) {
        self.stats.restore_rollbacks.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_host_swap_exhaustion(&self) {
        self.stats
            .host_swap_exhaustions
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_restore_prepared(&self) {
        self.stats.restore_prepared.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_h2d_submitted(&self) {
        self.stats.h2d_submitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn diagnostics(&self) -> ContentionDiagnostics {
        let (device_pages_free, device_pages_total) = self.port.device_stats();
        let (host_slots_free, host_slots_total) = self.port.host_stats();
        let inner = self.inner.lock().unwrap();
        let mut processes: Vec<_> = inner
            .procs
            .iter()
            .map(|(pid, proc)| ContentionQueueEntry {
                process_id: pid.to_string(),
                submit_seq: proc.seq,
                state: match proc.state {
                    ProcState::Running => "running",
                    ProcState::ParkRequested => "park-requested",
                    ProcState::Quiescing => "quiescing",
                    ProcState::Suspended => "suspended",
                    ProcState::Restoring => "restoring",
                },
                pages: proc.restore_demand,
                granted: inner.restore_grants.contains_key(pid),
            })
            .collect();
        processes.sort_by_key(|entry| entry.submit_seq);
        let mut waiters: Vec<_> = inner
            .queue
            .iter()
            .filter_map(|(key, entry)| match &entry.kind {
                EntryKind::Allocation { grant, .. } => Some(ContentionQueueEntry {
                    process_id: entry.pid.to_string(),
                    submit_seq: key.0,
                    state: "allocation",
                    pages: entry.demand.kv_pages,
                    granted: grant.is_some(),
                }),
                EntryKind::Restore { .. } => None,
            })
            .collect();
        waiters.sort_by_key(|entry| entry.submit_seq);
        let mut suspended: Vec<_> = inner
            .procs
            .iter()
            .filter(|(_, proc)| {
                matches!(proc.state, ProcState::Suspended | ProcState::Restoring)
            })
            .map(|(pid, proc)| ContentionQueueEntry {
                process_id: pid.to_string(),
                submit_seq: proc.seq,
                state: if proc.state == ProcState::Restoring {
                    "restoring"
                } else {
                    "suspended"
                },
                pages: proc.restore_demand,
                granted: inner.restore_grants.contains_key(pid),
            })
            .collect();
        suspended.sort_by_key(|entry| entry.submit_seq);
        let device_pages_reserved = inner
            .queue
            .iter()
            .map(|(_, entry)| {
                entry.kv_accum.len() as u32
                    + match &entry.kind {
                        EntryKind::Allocation {
                            grant: Some(grant), ..
                        } => grant.remaining_kv() as u32,
                        _ => 0,
                    }
            })
            .chain(
                inner
                    .restore_grants
                    .values()
                    .map(|issued| issued.grant.remaining_kv() as u32),
            )
            .sum();
        let park_requests_current = inner
            .procs
            .values()
            .filter(|proc| proc.state == ProcState::ParkRequested)
            .count();
        use Ordering::Relaxed;
        ContentionDiagnostics {
            processes,
            waiters,
            suspended,
            device_pages_free,
            device_pages_total,
            device_pages_reserved,
            host_slots_free,
            host_slots_total,
            park_requests_current,
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
            restore_prepared_total: self.stats.restore_prepared.load(Relaxed),
            h2d_submitted_total: self.stats.h2d_submitted.load(Relaxed),
            restore_wait_us_total: self.stats.restore_wait_us.load(Relaxed),
            restore_turnaround_us_total: self.stats.restore_turnaround_us.load(Relaxed),
            restore_turnarounds_total: self.stats.restore_turnarounds.load(Relaxed),
            deadline_kills_total: self.stats.deadline_kills.load(Relaxed),
        }
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
}

enum FinishOutcome {
    Granted(Arc<Notify>),
    RestoreIssued {
        signal: Arc<Notify>,
        waited: Duration,
    },
    /// The entry vanished or went stale under the race; the drain drops the
    /// returned reservation (and any stale entry) OUTSIDE the lock.
    Vanished(RsSlotReservation, Option<Entry>),
}

// =============================================================================
// Registry-owned instances, per (model, driver)
// =============================================================================

static PRIMARY: OnceLock<Arc<ContentionOrchestrator>> = OnceLock::new();
static ORCHESTRATORS: OnceLock<RwLock<HashMap<(usize, usize), Arc<ContentionOrchestrator>>>> =
    OnceLock::new();

/// Install the orchestrator for `(model, driver)` (bootstrap, once per pair).
pub fn init_contention(model: usize, driver: usize, orchestrator: ContentionOrchestrator) {
    let orchestrator = Arc::new(orchestrator);
    if model == 0 && driver == 0 {
        let _ = PRIMARY.set(orchestrator.clone());
    }
    ORCHESTRATORS
        .get_or_init(Default::default)
        .write()
        .unwrap()
        .insert((model, driver), orchestrator);
}

/// The orchestrator for `(model, driver)`.
pub fn contention_for(model: usize, driver: usize) -> Option<Arc<ContentionOrchestrator>> {
    if model == 0 && driver == 0 {
        return PRIMARY.get().cloned();
    }
    ORCHESTRATORS
        .get()?
        .read()
        .unwrap()
        .get(&(model, driver))
        .cloned()
}

/// The (0, 0) orchestrator — the hot-path shorthand while process-side call
/// sites are still hardwired to driver 0 (parametrized in B5). `None` only
/// before bootstrap has run.
pub fn contention() -> Option<&'static Arc<ContentionOrchestrator>> {
    PRIMARY.get()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// A pool with `free` pages live plus `idle` pages reclaimable at zero
    /// cost (rung 0), and an RS pool.
    struct MockPool {
        free: AtomicU32,
        idle: AtomicU32,
        total: u32,
        rs_free: AtomicU32,
        rs_total: u32,
        suspend: bool,
    }

    impl MockPool {
        fn new(free: u32, idle: u32, total: u32) -> Self {
            Self {
                free: AtomicU32::new(free),
                idle: AtomicU32::new(idle),
                total,
                rs_free: AtomicU32::new(64),
                rs_total: 64,
                suspend: true,
            }
        }
    }

    impl PoolPort for MockPool {
        fn device_stats(&self) -> (u32, u32) {
            (self.free.load(Ordering::SeqCst), self.total)
        }
        fn host_stats(&self) -> (u32, u32) {
            (0, 0)
        }
        fn rs_stats(&self) -> (u32, u32) {
            (self.rs_free.load(Ordering::SeqCst), self.rs_total)
        }
        fn reclaim_idle(&self) -> u32 {
            let n = self.idle.swap(0, Ordering::SeqCst);
            self.free.fetch_add(n, Ordering::SeqCst);
            n
        }
        fn suspend_capable(&self) -> bool {
            self.suspend
        }
        fn locus(&self) -> (usize, usize) {
            (0, 0)
        }
        fn reserve_device(
            &self,
            count: u32,
        ) -> Option<Vec<crate::store::kv::page_table::PhysicalKvPageId>> {
            self.free
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |free| {
                    (free >= count).then_some(free - count)
                })
                .ok()?;
            Some(
                (0..count)
                    .map(crate::store::kv::page_table::PhysicalKvPageId)
                    .collect(),
            )
        }
        fn release_device(&self, pages: Vec<crate::store::kv::page_table::PhysicalKvPageId>) {
            self.free.fetch_add(pages.len() as u32, Ordering::SeqCst);
        }
        fn reserve_rs(&self, count: u32) -> Option<Vec<crate::store::rs::RsSlotId>> {
            self.rs_free
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |free| {
                    (free >= count).then_some(free - count)
                })
                .ok()?;
            Some((0..count).map(crate::store::rs::RsSlotId).collect())
        }
        fn release_rs(&self, slots: Vec<crate::store::rs::RsSlotId>) {
            self.rs_free.fetch_add(slots.len() as u32, Ordering::SeqCst);
        }
    }

    fn orch(free: u32, idle: u32, total: u32) -> (Arc<ContentionOrchestrator>, ProcessId) {
        let (orch, pid, _) = orch_with_pool(free, idle, total);
        (orch, pid)
    }

    fn orch_with_pool(
        free: u32,
        idle: u32,
        total: u32,
    ) -> (Arc<ContentionOrchestrator>, ProcessId, Arc<MockPool>) {
        let pool = Arc::new(MockPool::new(free, idle, total));
        let orch = Arc::new(ContentionOrchestrator::new(
            pool.clone(),
            ContentionConfig {
                exhaustion: Duration::from_millis(200),
                ..ContentionConfig::default()
            },
        ));
        let pid = ProcessId::new_v4();
        orch.register(pid);
        (orch, pid, pool)
    }

    fn orch_with_config(
        free: u32,
        idle: u32,
        total: u32,
        exhaustion: Duration,
    ) -> (Arc<ContentionOrchestrator>, ProcessId, Arc<MockPool>) {
        let pool = Arc::new(MockPool::new(free, idle, total));
        let orch = Arc::new(ContentionOrchestrator::new(
            pool.clone(),
            ContentionConfig {
                exhaustion,
                ..ContentionConfig::default()
            },
        ));
        let pid = ProcessId::new_v4();
        orch.register(pid);
        (orch, pid, pool)
    }

    /// Test-side acquire: plain, no pipeline aggregation.
    async fn acquire(
        orch: &ContentionOrchestrator,
        pid: ProcessId,
        need: u32,
    ) -> Result<AllocationGrant, ContentionError> {
        orch.acquire(pid, pid, Demand::kv(need)).await
    }

    /// Drive the park protocol for a test victim straight to Quiescing —
    /// tests forge suspensions without real pressure, and the production
    /// `begin_quiesce` commit point would (correctly) withdraw them.
    fn suspend_for_test(orch: &ContentionOrchestrator, pid: ProcessId, pages: u32) {
        orch.with_inner(|inner| {
            let proc = inner.procs.get_mut(&pid).unwrap();
            proc.state = ProcState::Quiescing;
        });
        orch.report_suspended(pid, pages);
    }

    #[tokio::test]
    async fn ladder_rung_0_reclaims_idle_leases_before_parking() {
        // 2 free + 6 idle-reclaimable, need 5: rung 0 must satisfy the
        // request without the requester ever parking.
        let (orch, pid) = orch(2, 6, 16);
        let got = acquire(&orch, pid, 5).await;
        assert!(got.is_ok());
        assert_eq!(orch.stats().waiters_parked.load(Ordering::Relaxed), 0);
        assert_eq!(orch.port.device_stats().0, 3); // five pages grant-reserved
        drop(got);
        assert_eq!(orch.port.device_stats().0, 8);
    }

    #[tokio::test]
    async fn keystone_with_nothing_reclaimable_parks_then_fails_loud() {
        // 1 free, nothing idle, no peers: the head parks, accumulates the
        // one free page, and once the progress deadline lapses with no
        // event, fails LOUD instead of wedging — returning its accumulation.
        let (orch, pid) = orch(1, 0, 16);
        let waiter = {
            let orch = orch.clone();
            tokio::spawn(async move { acquire(&orch, pid, 4).await })
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
        // The head's partial accumulation returned to the pool.
        assert_eq!(orch.port.device_stats().0, 1);
    }

    #[test]
    fn restores_follow_spawn_order_not_suspend_completion_order() {
        let (orch, _a, pool) = orch_with_pool(0, 0, 10);
        let b = ProcessId::new_v4();
        let c = ProcessId::new_v4();
        orch.register(b);
        orch.register(c);
        suspend_for_test(&orch, c, 2);
        suspend_for_test(&orch, b, 2);

        pool.free.store(2, Ordering::SeqCst);
        orch.on_blocks_freed();
        let inner = orch.inner.lock().unwrap();
        assert!(inner.restore_grants.contains_key(&b));
        assert!(!inner.restore_grants.contains_key(&c));
    }

    #[test]
    fn older_restore_outranks_younger_allocation_waiter() {
        // D1: one queue, one clock. The suspended elder is the head; a
        // younger allocation waiter is served only from surplus.
        let (orch, older, pool) = orch_with_pool(0, 0, 10);
        let younger = ProcessId::new_v4();
        orch.register(younger);
        suspend_for_test(&orch, older, 2);
        let (key, _) = orch
            .register_waiter(younger, younger, Demand::kv(2))
            .unwrap();

        pool.free.store(2, Ordering::SeqCst);
        orch.on_blocks_freed();
        let inner = orch.inner.lock().unwrap();
        assert!(inner.restore_grants.contains_key(&older));
        assert!(matches!(
            &inner.queue.get(&key).unwrap().kind,
            EntryKind::Allocation { grant: None, .. }
        ));
    }

    #[test]
    fn head_first_claim_accumulates_before_younger_are_served() {
        let (orch, head, pool) = orch_with_pool(3, 0, 16);
        let younger = ProcessId::new_v4();
        orch.register(younger);
        let (head_key, _) = orch.register_waiter(head, head, Demand::kv(5)).unwrap();
        let (younger_key, _) = orch
            .register_waiter(younger, younger, Demand::kv(2))
            .unwrap();
        orch.drain();
        // All 3 free pages accumulate toward the head; the younger waiter
        // fits but must not overtake.
        {
            let inner = orch.inner.lock().unwrap();
            assert_eq!(inner.queue.get(&head_key).unwrap().kv_accum.len(), 3);
            assert!(matches!(
                &inner.queue.get(&younger_key).unwrap().kind,
                EntryKind::Allocation { grant: None, .. }
            ));
        }
        // Four more pages: head completes (5), surplus (2) serves younger.
        pool.free.fetch_add(4, Ordering::SeqCst);
        orch.on_blocks_freed();
        let inner = orch.inner.lock().unwrap();
        assert!(matches!(
            &inner.queue.get(&head_key).unwrap().kind,
            EntryKind::Allocation { grant: Some(_), .. }
        ));
        assert!(matches!(
            &inner.queue.get(&younger_key).unwrap().kind,
            EntryKind::Allocation { grant: Some(_), .. }
        ));
        assert_eq!(pool.free.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn escalation_picks_a_victim_younger_than_the_head() {
        let (orch, head, _) = orch_with_pool(0, 0, 10);
        let younger = ProcessId::new_v4();
        orch.register(younger);
        let _ = orch.register_waiter(head, head, Demand::kv(4)).unwrap();
        let _ = orch
            .register_waiter(younger, younger, Demand::kv(2))
            .unwrap();
        orch.escalate();
        assert!(orch.should_park(younger));
        assert!(!orch.should_park(head));
        // The victim's own entry was displaced.
        let inner = orch.inner.lock().unwrap();
        assert!(
            inner
                .queue
                .iter()
                .all(|(_, entry)| entry.pid != younger)
        );
    }

    #[test]
    fn stale_park_request_is_withdrawn_once_nothing_waits() {
        let (orch, pid) = orch(1, 0, 1);
        orch.with_inner(|inner| {
            inner.procs.get_mut(&pid).unwrap().state = ProcState::ParkRequested;
        });
        orch.drain();
        assert_eq!(
            orch.inner.lock().unwrap().procs.get(&pid).unwrap().state,
            ProcState::Running
        );
        assert!(!orch.should_park(pid));
    }

    #[test]
    fn grants_are_reserved_and_returned_on_unregister() {
        let (orch, first, pool) = orch_with_pool(2, 0, 10);
        let second = ProcessId::new_v4();
        orch.register(second);
        let (first_key, _) = orch.register_waiter(first, first, Demand::kv(2)).unwrap();
        let (second_key, _) = orch
            .register_waiter(second, second, Demand::kv(2))
            .unwrap();
        orch.drain();
        assert_eq!(pool.free.load(Ordering::SeqCst), 0);
        assert!(orch.grant_ready(first_key));
        assert!(!orch.grant_ready(second_key));

        orch.unregister(first);
        assert!(orch.grant_ready(second_key));
        assert_eq!(pool.free.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn duplicate_pipeline_waits_share_one_priority_slot() {
        let (orch, pid) = orch(0, 0, 10);
        let (first, _) = orch.register_waiter(pid, pid, Demand::kv(2)).unwrap();
        let (second, _) = orch.register_waiter(pid, pid, Demand::kv(4)).unwrap();
        assert_eq!(first, second);
        let inner = orch.inner.lock().unwrap();
        let entry = inner.queue.get(&first).unwrap();
        assert_eq!(entry.demand.kv_pages, 4);
        assert!(matches!(
            &entry.kind,
            EntryKind::Allocation { waiters: 2, .. }
        ));
    }

    #[test]
    fn sibling_pipeline_waits_keep_independent_quorum_slots() {
        let (orch, pid) = orch(0, 0, 10);
        let first_pipeline = ProcessId::new_v4();
        let second_pipeline = ProcessId::new_v4();
        let (first, _) = orch
            .register_waiter(pid, first_pipeline, Demand::kv(2))
            .unwrap();
        let (second, _) = orch
            .register_waiter(pid, second_pipeline, Demand::kv(4))
            .unwrap();
        assert_ne!(first, second);
        let inner = orch.inner.lock().unwrap();
        assert_eq!(inner.queue.counts().0, 2);
    }

    #[test]
    fn larger_duplicate_cannot_consume_an_issued_smaller_grant() {
        let (orch, pid, _) = orch_with_pool(2, 0, 10);
        let (key, _) = orch.register_waiter(pid, pid, Demand::kv(2)).unwrap();
        orch.drain();
        let (same_key, _) = orch.register_waiter(pid, pid, Demand::kv(4)).unwrap();
        assert_eq!(key, same_key);
        assert!(orch.take_allocation_grant(pid, key, Demand::kv(4)).is_none());
        assert!(orch.take_allocation_grant(pid, key, Demand::kv(2)).is_some());
    }

    #[tokio::test]
    async fn unsatisfiable_suspended_keystone_times_out_without_an_allocator() {
        let (orch, pid, _) = orch_with_config(0, 0, 10, Duration::from_millis(40));
        suspend_for_test(&orch, pid, 2);
        let result =
            tokio::time::timeout(Duration::from_secs(1), orch.park_until_restore_granted(pid))
                .await
                .expect("the progress deadline must wake the suspended keystone");
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
    async fn victim_that_never_reaches_a_safe_point_does_not_reset_the_deadline() {
        // A posted park request alone is not progress; only report_suspended
        // (pages actually freed) resets the clock. The head must still fail
        // loud when its victim never complies.
        let (orch, oldest, _) = orch_with_config(0, 0, 10, Duration::from_millis(40));
        let younger = ProcessId::new_v4();
        orch.register(younger);
        let _ = orch
            .register_waiter(younger, younger, Demand::kv(1))
            .unwrap();
        let task = {
            let orch = orch.clone();
            tokio::spawn(async move { acquire(&orch, oldest, 2).await })
        };
        // The ladder engages: the younger peer receives a park request.
        tokio::time::timeout(Duration::from_secs(1), async {
            while !orch.should_park(younger) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("escalation must select the younger victim");
        let result = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("an uncooperative victim must not stall the deadline")
            .unwrap();
        assert!(matches!(result, Err(ContentionError::Exhausted { .. })));
        // Pressure relief withdrew the now-pointless park request when the
        // failed head left the queue.
        assert!(!orch.should_park(younger));
    }

    #[tokio::test]
    async fn dropping_an_acquisition_future_removes_its_waiter() {
        let (orch, pid, _) = orch_with_config(0, 0, 10, Duration::from_secs(5));
        let task = {
            let orch = orch.clone();
            tokio::spawn(async move { acquire(&orch, pid, 2).await })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while orch.inner.lock().unwrap().queue.counts().0 == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(orch.contended());
        task.abort();
        let _ = task.await;
        tokio::task::yield_now().await;
        assert_eq!(orch.inner.lock().unwrap().queue.counts().0, 0);
        assert!(!orch.contended());
        assert_eq!(orch.stats.cancelled_waits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn idle_hot_path_does_not_lock_inner() {
        let (orch, _pid, _) = orch_with_pool(10, 0, 10);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = orch.inner.lock().unwrap();
            panic!("poison inner");
        }));
        assert!(!orch.contended());
        orch.on_blocks_freed();
    }

    #[tokio::test]
    async fn grant_carries_rs_slots_and_returns_them_on_drop() {
        let (orch, pid, pool) = orch_with_pool(4, 0, 16);
        let grant = orch
            .acquire(
                pid,
                pid,
                Demand {
                    kv_pages: 2,
                    rs_slots: 3,
                },
            )
            .await
            .unwrap();
        assert_eq!(grant.remaining_kv(), 2);
        assert_eq!(grant.remaining_rs(), 3);
        assert_eq!(pool.rs_free.load(Ordering::SeqCst), 61);
        drop(grant);
        assert_eq!(pool.rs_free.load(Ordering::SeqCst), 64);
        assert_eq!(pool.free.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn hog_holding_pages_while_running_is_park_requested() {
        // The behavioral gap this project exists to close: a young process
        // with NO allocation entry — running steadily, holding pages — is a
        // victim like any other once an older head starves.
        let (orch, head, _) = orch_with_pool(0, 0, 10);
        let hog = ProcessId::new_v4();
        orch.register(hog);
        let _ = orch.register_waiter(head, head, Demand::kv(4)).unwrap();
        orch.escalate();
        assert!(orch.should_park(hog));
    }

    #[test]
    fn begin_quiesce_withdraws_a_stale_park_request() {
        let (orch, older, _) = orch_with_pool(0, 0, 10);
        let younger = ProcessId::new_v4();
        orch.register(younger);
        // No older unmet head exists: the commit point withdraws.
        orch.with_inner(|inner| {
            inner.procs.get_mut(&younger).unwrap().state = ProcState::ParkRequested;
        });
        assert!(!orch.begin_quiesce(younger));
        assert!(!orch.should_park(younger));
        // With an older unmet head, the same request commits.
        let _ = orch.register_waiter(older, older, Demand::kv(4)).unwrap();
        orch.with_inner(|inner| {
            inner.procs.get_mut(&younger).unwrap().state = ProcState::ParkRequested;
        });
        assert!(orch.begin_quiesce(younger));
    }

    #[test]
    fn idle_holder_is_preferred_over_a_running_victim() {
        let (orch, head, _) = orch_with_pool(0, 0, 10);
        let running = ProcessId::new_v4();
        let idler = ProcessId::new_v4();
        orch.register(running);
        orch.register(idler);
        orch.note_idle_await(idler, true);
        let _ = orch.register_waiter(head, head, Demand::kv(4)).unwrap();
        orch.escalate();
        // `running` is younger (spawned later), but the idle holder costs
        // no running lane (D6) and is selected first.
        assert!(orch.should_park(idler));
        assert!(!orch.should_park(running));
    }

    #[test]
    fn smallest_covering_footprint_wins_within_a_class() {
        let (orch, head, _) = orch_with_pool(0, 0, 32);
        let small = ProcessId::new_v4();
        let big = ProcessId::new_v4();
        orch.register(small);
        orch.register(big);
        let footprints = std::sync::Mutex::new(std::collections::HashMap::from([
            (small, 4u32),
            (big, 20u32),
        ]));
        orch.set_footprint_probe(move |pid, _, _| footprints.lock().unwrap().get(&pid).copied());
        let _ = orch.register_waiter(head, head, Demand::kv(3)).unwrap();
        orch.escalate();
        // Both cover the residual demand of 3; the 4-page victim wastes 1
        // page of D2H, the 20-page victim wastes 17 (D6 smallest-cover).
        assert!(orch.should_park(small));
        assert!(!orch.should_park(big));
    }

    #[tokio::test]
    async fn deadline_breach_kills_the_uncooperative_victim_before_failing_loud() {
        let (orch, head, pool) = orch_with_config(0, 0, 10, Duration::from_millis(40));
        let hog = ProcessId::new_v4();
        orch.register(hog);
        let kills: Arc<Mutex<Vec<ProcessId>>> = Arc::default();
        {
            let kills = kills.clone();
            orch.set_kill_hook(move |pid, _reason| kills.lock().unwrap().push(pid));
        }
        let task = {
            let orch = orch.clone();
            tokio::spawn(async move { acquire(&orch, head, 2).await })
        };
        // The ladder first posts a park request to the hog; at the breach
        // the still-uncooperative hog is the kill target (D7), and the head
        // KEEPS WAITING on a fresh deadline window instead of failing.
        tokio::time::timeout(Duration::from_secs(1), async {
            while kills.lock().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the breach must escalate to a kill");
        assert_eq!(kills.lock().unwrap().as_slice(), &[hog]);
        assert_eq!(orch.stats().deadline_kills.load(Ordering::Relaxed), 1);
        assert!(!task.is_finished(), "the head must outlive the kill");
        // The kill's teardown frees the hog's pages; the head completes.
        orch.unregister(hog);
        pool.free.store(2, Ordering::SeqCst);
        orch.on_blocks_freed();
        let got = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("the head must complete after the kill frees pages")
            .unwrap();
        assert!(got.is_ok());
        assert_eq!(orch.stats().exhaustion_timeouts.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn requester_selected_as_victim_parks_until_the_caller_yields() {
        // An older head exists; the younger requester is the only candidate
        // — the ladder picks it, its acquire parks (a standing safe point),
        // and the caller observes the park signal, cancels, and yields
        // (victim and requester are the same code, D3).
        let (orch, older, _) = orch_with_config(0, 0, 10, Duration::from_secs(5));
        let younger = ProcessId::new_v4();
        orch.register(younger);
        let _ = orch.register_waiter(older, older, Demand::kv(4)).unwrap();
        let task = {
            let orch = orch.clone();
            tokio::spawn(async move { orch.acquire(younger, younger, Demand::kv(2)).await })
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            while !orch.should_park(younger) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("escalation must select the younger requester");
        // The caller's safe point reacts: cancel the parked acquire and run
        // the park protocol (here: teardown, the harshest cancellation).
        task.abort();
        let _ = task.await;
        orch.unregister(younger);
        assert!(!orch.is_registered(younger));
    }
}
