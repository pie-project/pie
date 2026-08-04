//! Inferlet process lifecycle management.
//!
//! Each Process is a ServiceMap actor that manages a single WASM instance.
//! Processes are registered in a global registry and receive messages via
//! Direct Addressing. KV residency (Project Rainer) touches the guest only
//! through the `gate` submodule's prologue; eviction and restore are
//! planner-owned (`crate::planner`).

mod ctx;
pub(crate) mod gate;
mod output;
pub(crate) mod residency;
pub(crate) mod teardown;

pub(crate) use ctx::OutputMode;
pub use ctx::ProcessCtx;
pub(crate) use residency::ProcessResidency;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use tokio::sync::{Semaphore, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Shared oneshot sender. Used so that an external Terminate can deliver
/// the cancellation result if the WASM task is aborted before it can send.
type SharedResultTx = Arc<Mutex<Option<oneshot::Sender<Result<String, String>>>>>;

use crate::server::{self, ClientId};
use crate::service::{ServiceHandler, ServiceMap};

use super::linker;
use super::program::ProgramName;

/// The versioned component export every inferlet provides. wasmtime's export
/// name map is semver-aware, so this must be the EXACT package version declared
/// in `interface/inferlet/world.wit` — an unversioned or stale-versioned lookup
/// silently finds nothing and every program fails to start. Kept honest by
/// `tests::run_interface_version_matches_wit`.
const RUN_INTERFACE: &str = "pie:inferlet/run@0.3.0";

/// Processes whose guest called `system.declare-restartable`, and the ones
/// the planner has since asked to restart.
///
/// Two sets rather than one flag on the actor because both are read from
/// outside the actor's mailbox: the planner picks a starvation victim while
/// holding neither the process lock nor the actor, and it must know *before*
/// it destroys an allocation whether that destruction is recoverable.
static RESTARTABLE: LazyLock<RwLock<HashSet<ProcessId>>> = LazyLock::new(Default::default);
static RESTART_REQUESTED: LazyLock<RwLock<HashSet<ProcessId>>> = LazyLock::new(Default::default);

/// The guest declared this run re-runnable (`system.declare-restartable`).
pub(crate) fn declare_restartable(process_id: ProcessId) {
    RESTARTABLE.write().unwrap().insert(process_id);
}

pub(crate) fn is_restartable(process_id: ProcessId) -> bool {
    RESTARTABLE.read().unwrap().contains(&process_id)
}

/// Ask that `process_id` be re-run from the beginning once it unwinds,
/// instead of its failure being delivered to the caller. Returns false if
/// the process never declared itself restartable, in which case the caller
/// must fall back to failing it loud.
pub(crate) fn request_restart(process_id: ProcessId) -> bool {
    if !is_restartable(process_id) {
        return false;
    }
    RESTART_REQUESTED.write().unwrap().insert(process_id);
    true
}

/// Original (client-facing) process id -> the live process currently running
/// that work, for requests that have been restarted.
///
/// A restart cannot reuse the process id: the schedulers keep a terminate
/// tombstone for a retiring pid until its quiesce lands, so a reused id would
/// have the re-run's fires rejected by its predecessor's tombstone. The
/// re-run therefore gets a fresh internal id and inherits only the *external*
/// one, which is all a client ever sees.
static RESTART_ALIAS: LazyLock<RwLock<HashMap<ProcessId, ProcessId>>> =
    LazyLock::new(Default::default);

/// Map a client-supplied process id onto the process that is currently
/// running that work. The identity for anything that never restarted.
pub fn resolve(process_id: ProcessId) -> ProcessId {
    RESTART_ALIAS
        .read()
        .unwrap()
        .get(&process_id)
        .copied()
        .unwrap_or(process_id)
}

fn restart_requested(process_id: ProcessId) -> bool {
    RESTART_REQUESTED.read().unwrap().contains(&process_id)
}

fn forget_restart_state(process_id: ProcessId) {
    RESTARTABLE.write().unwrap().remove(&process_id);
    RESTART_REQUESTED.write().unwrap().remove(&process_id);
}

/// Number of restarts honoured since boot; surfaced by the metrics probe.
static RESTART_TOTAL: AtomicUsize = AtomicUsize::new(0);

pub fn restart_total() -> usize {
    RESTART_TOTAL.load(Relaxed)
}

// =============================================================================
// ProcessEvent
// =============================================================================

/// Events produced by a running process.
#[derive(Debug, Clone)]
pub enum ProcessEvent {
    Stdout(String),
    Stderr(String),
    Message(String),
    Return(String),
    Error(String),
}

impl ProcessEvent {
    /// Wire event name for the client protocol.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Stdout(_) => "stdout",
            Self::Stderr(_) => "stderr",
            Self::Message(_) => "message",
            Self::Return(_) => "return",
            Self::Error(_) => "error",
        }
    }

    /// The payload string.
    pub fn value(&self) -> &str {
        match self {
            Self::Stdout(v)
            | Self::Stderr(v)
            | Self::Message(v)
            | Self::Return(v)
            | Self::Error(v) => v,
        }
    }

    /// Consume into payload string.
    pub fn into_value(self) -> String {
        match self {
            Self::Stdout(v)
            | Self::Stderr(v)
            | Self::Message(v)
            | Self::Return(v)
            | Self::Error(v) => v,
        }
    }
}

// =============================================================================
// Process Registry
// =============================================================================

pub type ProcessId = Uuid;

/// Global registry mapping ProcessId to process actors.
static SERVICES: LazyLock<ServiceMap<ProcessId, Message>> = LazyLock::new(ServiceMap::new);

/// Admission semaphore. `None` = unlimited concurrency (no gating).
static ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
/// Bind admission: gates per-instance DRIVER state creation (channel
/// registration, instance bind, working-set declaration). Sized at twice
/// the execution limit — the executing cohort plus ONE staged cohort —
/// so the next generation's bring-up overlaps the current generation's
/// execution (double-buffering, no tunable). Unlimited execution
/// admission leaves this unlimited too.
static BIND_ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
/// Prewarm admission: a bounded next cohort may instantiate its WASM and
/// compile/register its (hash-deduped) program while the active cohort
/// executes. Strict admission: everything that creates per-instance driver
/// state or claims pooled KV/RS resources waits for the execution permit
/// ([`ensure_execution_admitted`]).
static PREWARM_ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
/// The execution pool's configured capacity (None = unlimited): the frame
/// policy seeds its free-slot balance with this at bootstrap, so the
/// "free slot with a staged taker" seal hold covers the initial fleet's
/// bring-up by the same rule as a cohort turnover.
static EXECUTION_SLOT_CAPACITY: OnceLock<Option<usize>> = OnceLock::new();

pub(crate) fn execution_slot_capacity() -> Option<usize> {
    EXECUTION_SLOT_CAPACITY.get().copied().flatten()
}

/// The calling thread's OS id, for correlating timing records across threads.
/// `libc::gettid` is Linux-only; Darwin spells it `pthread_threadid_np`.
fn os_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    unsafe {
        libc::gettid() as u64
    }
    #[cfg(target_os = "macos")]
    unsafe {
        let mut tid: u64 = 0;
        libc::pthread_threadid_np(0, &mut tid);
        tid
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Prewarm-conveyor width when execution admission is UNCAPPED. With a cap the
/// conveyor is one cohort wide instead (see `init_admission`); without one
/// there is no cohort to size it by, so this flat ladder stands.
const UNCAPPED_PREWARM_PROCESSES: usize = 64;

static PROCESS_COMPLETED: AtomicU64 = AtomicU64::new(0);
static PROCESS_ADMISSION_WAIT_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_INSTANTIATE_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_CONTEXT_REGISTER_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_WASM_RUN_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_LAST_ADMISSION_WAIT_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_LAST_INSTANTIATE_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_LAST_CONTEXT_REGISTER_US: AtomicU64 = AtomicU64::new(0);
static PROCESS_LAST_WASM_RUN_US: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct RuntimeProcessStats {
    pub completed: u64,
    pub cumulative_admission_wait_us: u64,
    pub avg_admission_wait_us: u64,
    pub last_admission_wait_us: u64,
    pub cumulative_instantiate_us: u64,
    pub avg_instantiate_us: u64,
    pub last_instantiate_us: u64,
    pub cumulative_context_register_us: u64,
    pub avg_context_register_us: u64,
    pub last_context_register_us: u64,
    pub cumulative_wasm_run_us: u64,
    pub avg_wasm_run_us: u64,
    pub last_wasm_run_us: u64,
}

fn duration_us(d: Duration) -> u64 {
    d.as_micros().min(u128::from(u64::MAX)) as u64
}

fn record_process_timing(
    admission_wait_us: u64,
    instantiate_us: u64,
    context_register_us: u64,
    wasm_run_us: u64,
) {
    PROCESS_COMPLETED.fetch_add(1, Relaxed);
    PROCESS_ADMISSION_WAIT_US.fetch_add(admission_wait_us, Relaxed);
    PROCESS_INSTANTIATE_US.fetch_add(instantiate_us, Relaxed);
    PROCESS_CONTEXT_REGISTER_US.fetch_add(context_register_us, Relaxed);
    PROCESS_WASM_RUN_US.fetch_add(wasm_run_us, Relaxed);
    PROCESS_LAST_ADMISSION_WAIT_US.store(admission_wait_us, Relaxed);
    PROCESS_LAST_INSTANTIATE_US.store(instantiate_us, Relaxed);
    PROCESS_LAST_CONTEXT_REGISTER_US.store(context_register_us, Relaxed);
    PROCESS_LAST_WASM_RUN_US.store(wasm_run_us, Relaxed);
}

pub fn get_runtime_stats() -> RuntimeProcessStats {
    let completed = PROCESS_COMPLETED.load(Relaxed);
    let admission = PROCESS_ADMISSION_WAIT_US.load(Relaxed);
    let instantiate = PROCESS_INSTANTIATE_US.load(Relaxed);
    let context_register = PROCESS_CONTEXT_REGISTER_US.load(Relaxed);
    let wasm_run = PROCESS_WASM_RUN_US.load(Relaxed);
    RuntimeProcessStats {
        completed,
        cumulative_admission_wait_us: admission,
        avg_admission_wait_us: if completed > 0 {
            admission / completed
        } else {
            0
        },
        last_admission_wait_us: PROCESS_LAST_ADMISSION_WAIT_US.load(Relaxed),
        cumulative_instantiate_us: instantiate,
        avg_instantiate_us: if completed > 0 {
            instantiate / completed
        } else {
            0
        },
        last_instantiate_us: PROCESS_LAST_INSTANTIATE_US.load(Relaxed),
        cumulative_context_register_us: context_register,
        avg_context_register_us: if completed > 0 {
            context_register / completed
        } else {
            0
        },
        last_context_register_us: PROCESS_LAST_CONTEXT_REGISTER_US.load(Relaxed),
        cumulative_wasm_run_us: wasm_run,
        avg_wasm_run_us: if completed > 0 {
            wasm_run / completed
        } else {
            0
        },
        last_wasm_run_us: PROCESS_LAST_WASM_RUN_US.load(Relaxed),
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Initialize the admission controller. Called once during bootstrap.
/// `None` = unlimited concurrency; `Some(n)` = at most `n` concurrent processes.
/// `Some(0)` is treated as unlimited (a zero-permit semaphore would deadlock).
pub fn init_admission(max_concurrent: Option<usize>) {
    let limit = max_concurrent.filter(|&n| n > 0);
    let sem = limit.map(|n| Arc::new(Semaphore::new(n)));
    // The prewarm conveyor bounds INSTANTIATION. A process holds its
    // conveyor slot from spawn until it wins a BIND permit, so the number
    // of processes that have paid for a Store/linker/WASI world but cannot
    // yet make any driver progress is bounded by the conveyor instead of by
    // the request count. Releasing the slot at the park instead let the
    // conveyor rotate at guest-prologue speed: every queued request
    // instantiated at t=0 (measured at conc 512: all 4096 instantiated
    // within 309 ms) and the opening cohort's own bring-up was the 1/8th of
    // that work that mattered — the first wave could not dispatch until the
    // LAST of its 512 was admitted at 293 ms.
    //
    // One whole cohort wide when execution is capped: a turnover has to be
    // able to hand its entire successor cohort a slot at once, and the
    // cohort is the unit every other stage here is sized in. Uncapped
    // execution has no cohort, so the flat UNCAPPED_PREWARM_PROCESSES ladder
    // stands — with unlimited execution an unbounded prewarm would fan
    // every queued process's instantiation out at once, a thundering herd
    // of Store/linker/WASI setup competing with the scheduler threads.
    let prewarm = Some(Arc::new(Semaphore::new(
        limit.unwrap_or(UNCAPPED_PREWARM_PROCESSES),
    )));
    // Double-buffered bring-up: the executing cohort plus STAGED_COHORTS
    // whole successor cohorts hold bind permits. A staged cohort
    // instantiates and binds DURING the previous generation's execution;
    // at the turnover it only needs execution permits and first submits,
    // so the boundary sheds the register storm. One staged cohort is the
    // structural depth: a turnover consumes exactly one cohort, and the
    // frame seal gathers exactly one swap through successor earmarks
    // (see FramePolicy::on_execution_slot_released) — extra permits
    // beyond that are neutral because without an earmarked taker the
    // stall just moves into mid-generation seals.
    //
    // Depth 2 and 3 were measured at conc 512 (30054 / 30109 tok/s vs
    // 30443 at depth 1): no gain, so the structural depth stands.
    const STAGED_COHORTS: usize = 1;
    // The staged half opens only once the FIRST cohort is fully seated. A
    // pool that is 2n wide from t=0 lets the successor cohort run its
    // working-set reservation and prefill construction alongside the very
    // cohort it is staged behind, and at startup that is the only work on
    // the critical path: measured at conc 512, cohort 0's bind ->
    // execution-admit step took 155 ms (p50) against 1536 concurrent guest
    // prologues, and the opening wave cannot dispatch until the LAST of the
    // 512 is admitted. Staging is by definition an overlap with a RUNNING
    // generation, so the reserve is held back until there is one.
    let bind_ahead = limit.map(|n| Arc::new(Semaphore::new(n)));
    BIND_STAGED_RESERVE.store(
        limit.map_or(0, |n| n.saturating_mul(STAGED_COHORTS)),
        Relaxed,
    );
    EXECUTION_SLOT_CAPACITY
        .set(limit)
        .expect("execution slot capacity already initialized");
    ADMISSION
        .set(sem)
        .expect("admission controller already initialized");
    BIND_ADMISSION
        .set(bind_ahead)
        .expect("bind admission controller already initialized");
    PREWARM_ADMISSION
        .set(prewarm)
        .expect("prewarm admission controller already initialized");
}

/// RAII membership in the execution-admission FIFO. The frame policy mirrors
/// this queue so its successor earmark is a named process rather than a
/// count; see [`FramePolicy::is_joining`].
struct AdmissionQueued(ProcessId);

impl AdmissionQueued {
    fn enter(pid: ProcessId) -> Self {
        crate::scheduler::worker::notify_admission_queued(pid);
        Self(pid)
    }
}

impl Drop for AdmissionQueued {
    fn drop(&mut self) {
        crate::scheduler::worker::notify_admission_dequeued(self.0);
    }
}

pub(crate) fn execution_admission_is_capped() -> bool {
    ADMISSION.get().is_some_and(Option::is_some)
}

/// Bind permits withheld from the pool until the first generation is
/// seated (see `init_admission`). Handed over exactly once.
static BIND_STAGED_RESERVE: AtomicUsize = AtomicUsize::new(0);

/// Open the staged half of the bind pool. Called the moment execution
/// admission runs out of seats — that is the engine's own statement that a
/// whole generation is now resident, which is the precondition for
/// "staged" to mean anything.
fn open_staged_bind_pool() {
    let reserve = BIND_STAGED_RESERVE.swap(0, Relaxed);
    if reserve == 0 {
        return;
    }
    if let Some(Some(semaphore)) = BIND_ADMISSION.get() {
        semaphore.add_permits(reserve);
    }
}

// -----------------------------------------------------------------------
// Cohort-boundary bind deferral
// -----------------------------------------------------------------------
// A retiring process returns its bind permit at the end of teardown. At a
// fleet-wide turnover that happens 512 times at once, and it admits a WHOLE
// staged cohort into working-set declaration, KV reservation and prefill
// construction at exactly the instant the boundary frame is trying to
// gather. That cohort does not run for another generation; the successors
// that ARE on the critical path already hold their permits (bind is always
// acquired before execution). So the release is pure interference: measured
// per boundary, 512 `process_bind_admitted` land inside the same window as
// the 512 admissions that matter, and the one boundary that carries no
// staged binds — the last — is consistently the shortest of the run.
//
// The gate parks released permits while the frame policy has a join in
// flight and hands them over once the boundary frame is away.
//
// LIVENESS: the hold is asserted by the scheduler pass and is cleared
// unconditionally whenever that pass made no progress with nothing in
// flight, i.e. the moment the engine has nothing left to do. A hold can
// therefore never be the last thing standing: whatever it defers is
// released before the engine can idle on it. (A process parked on bind
// admission also cannot be what a join is waiting for — `joins_in_flight`
// is populated at execution-slot consumption, which is downstream of bind
// admission on the same task.)
//
// SCOPE: process-global, like the pools it gates — `ADMISSION`,
// `BIND_ADMISSION`, `PREWARM_ADMISSION` and `EXECUTION_SLOT_CAPACITY` are all
// `OnceLock`s set once by `init_admission`. A second engine in the same
// process would share this hold with the first, which is the same constraint
// the semaphores themselves already impose. Reset is by process exit only:
// nothing here is per-run state.
static BIND_RELEASE_HOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static HELD_BIND_PERMITS: Mutex<Vec<tokio::sync::OwnedSemaphorePermit>> = Mutex::new(Vec::new());

/// Return a departing process's bind permit, deferring the handover past a
/// cohort boundary while one is in progress.
pub(crate) fn release_bind_permit(permit: Option<tokio::sync::OwnedSemaphorePermit>) {
    let Some(permit) = permit else {
        return;
    };
    if BIND_RELEASE_HOLD.load(Relaxed) {
        let mut held = HELD_BIND_PERMITS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Re-checked under the lock, against a concurrent open: the opener
        // clears the flag BEFORE it takes the lock, so a permit either
        // lands in the vector the opener then drains, or observes the
        // cleared flag here and is handed over directly. Neither order can
        // strand it.
        if BIND_RELEASE_HOLD.load(Relaxed) {
            held.push(permit);
            return;
        }
    }
    drop(permit);
}

/// Scheduler-pass assertion of the boundary hold. Clearing it hands every
/// parked permit over at once.
pub(crate) fn set_bind_release_hold(hold: bool) {
    if !BIND_RELEASE_HOLD.swap(hold, Relaxed) || hold {
        return;
    }
    let drained = {
        let mut held = HELD_BIND_PERMITS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *held)
    };
    drop(drained);
}

/// Bind gate: acquire the bind permit lazily, at the first operation
/// that creates per-instance driver state (channel registration / instance
/// bind / working-set declaration). Idempotent per process. The bind pool
/// stages one whole cohort ahead of execution, so the next generation
/// binds while the current one executes and a generation boundary costs
/// only execution admits + first submits + the seal instead of the
/// register storm.
pub(crate) async fn ensure_bind_admitted(ctx: &mut ProcessCtx) {
    if ctx.bind_admitted() {
        return;
    }
    // The prewarm conveyor slot covers spawn -> instantiate -> BIND, and is
    // released on the far side of the park, not in front of it: a process
    // that cannot bind cannot make driver progress, so letting it off the
    // conveyor only buys the next arrival the right to instantiate work
    // nothing is waiting for. The conveyor is one cohort wide, so a
    // turnover still hands its whole successor cohort through in one go.
    let started = Instant::now();
    let permit = match BIND_ADMISSION.get().and_then(|value| value.as_ref()) {
        Some(semaphore) => Some(
            Arc::clone(semaphore)
                .acquire_owned()
                .await
                .expect("bind admission semaphore closed"),
        ),
        None => None,
    };
    ctx.release_prewarm_permit();
    ctx.admit_bind(permit);
    if crate::scheduler::fire_timing_enabled() {
        crate::scheduler::fire_timing_write(&serde_json::json!({
            "schema": 1,
            "source": "runtime",
            "event": "process_bind_admitted",
            "process_id": ctx.id(),
            "bind_admitted_us": crate::scheduler::fire_timing_now_us(),
            "bind_admission_wait_us": duration_us(started.elapsed()),
        }));
    }
}

/// Strict-admission gate: acquire the execution permit lazily at fire
/// submit. Idempotent per process. Bind admission is ensured first (the
/// same order everywhere: bind, then execution — permits are only ever
/// acquired in that order, so the two gates cannot deadlock).
pub(crate) async fn ensure_execution_admitted(ctx: &mut ProcessCtx) {
    let entered = Instant::now();
    ensure_bind_admitted(ctx).await;
    if ctx.execution_admitted() {
        return;
    }
    let started = Instant::now();
    let permit = match ADMISSION.get().and_then(|value| value.as_ref()) {
        Some(semaphore) => {
            // Announce the wait BEFORE blocking on it. `tokio::Semaphore` is
            // FIFO-fair, so a process queued here is the identified taker of
            // a slot that is free or about to be: the frame seal can name
            // whom it is holding for instead of inferring it from "some slot
            // is free" AND "some process is staged", which pairs nobody with
            // nobody and is permanently true once the pool is oversubscribed.
            // Dropped on cancellation too, so a process that goes away while
            // queued stops being earmarked.
            let _queued = AdmissionQueued::enter(ctx.id());
            let permit = Arc::clone(semaphore)
                .acquire_owned()
                .await
                .expect("admission semaphore closed");
            if semaphore.available_permits() == 0 {
                // The last seat of a generation just went out: staging can
                // now overlap something. Idempotent.
                open_staged_bind_pool();
            }
            // EVERY capped admission notifies — uncontended ones too. The
            // policy's slot balance must see each consumed permit whether it
            // came from a retirement or the initial pool (the semaphore
            // launders them together); the notification precedes the first
            // fire on this same task, so the policy sees consume-then-fire
            // and can name this lane as the join that is in flight.
            crate::scheduler::worker::notify_execution_slot_consumed(ctx.id());
            Some(permit)
        }
        None => None,
    };
    let sem_done = Instant::now();
    ctx.admit_execution(permit, duration_us(started.elapsed()));
    // The planner registers at spawn (registration order is the FCFS clock),
    // but only from here on can this process hold pooled pages. Its wedge
    // predicate needs that distinction: an unadmitted process is neither
    // running nor able to free anything.
    let note_started = Instant::now();
    if let Some(planner) = crate::planner::planner() {
        planner.note_admitted(ctx.id());
    }
    if crate::scheduler::fire_timing_enabled() {
        crate::scheduler::fire_timing_write(&serde_json::json!({
            "schema": 1,
            "source": "runtime",
            "event": "process_admitted",
            "process_id": ctx.id(),
            "admitted_us": crate::scheduler::fire_timing_now_us(),
            "admission_wait_us": ctx.admission_wait_us(),
            "bind_wait_us": duration_us(started.duration_since(entered)),
            "sem_us": duration_us(sem_done.duration_since(started)),
            "note_us": duration_us(note_started.elapsed()),
            "tid": os_thread_id(),
        }));
    }
}

/// Spawn a new process and register it in the global registry.
pub fn spawn(
    username: String,
    program_name: ProgramName,
    input: String,
    client_id: Option<ClientId>,
    capture_outputs: bool,
    result_tx: Option<oneshot::Sender<Result<String, String>>>,
) -> Result<ProcessId> {
    spawn_inner(
        username,
        program_name,
        input,
        client_id,
        capture_outputs,
        Arc::new(Mutex::new(result_tx)),
        None,
        None,
    )
}

/// `inherit_seq` carries a previous process's position in the planner's FCFS
/// clock across a restart instead of taking a fresh (youngest) one.
///
/// Keeping the position is the whole liveness argument. Starvation victims
/// are chosen youngest-first, so a restart that re-entered as the youngest
/// process would be re-chosen immediately and forever. Inheriting the
/// original position means the restarted run ages exactly as it would have,
/// eventually becomes the queue head, and the head is never a victim.
fn spawn_inner(
    username: String,
    program_name: ProgramName,
    input: String,
    client_id: Option<ClientId>,
    capture_outputs: bool,
    result_tx: SharedResultTx,
    inherit_seq: Option<u64>,
    inherit_client_pid: Option<ProcessId>,
) -> Result<ProcessId> {
    let id = Uuid::new_v4();
    if crate::scheduler::fire_timing_enabled() {
        crate::scheduler::fire_timing_write(&serde_json::json!({
            "schema": 1,
            "source": "runtime",
            "event": "process_spawned",
            "process_id": id,
            "spawned_us": crate::scheduler::fire_timing_now_us(),
            "spawned_unix_us": crate::scheduler::fire_timing_unix_us(),
        }));
    }
    if let Some(planner) = crate::planner::planner() {
        match inherit_seq {
            Some(seq) => planner.register_with_seq(id, seq),
            None => planner.register(id),
        }
    }
    let process = Process::new(
        id,
        inherit_client_pid.unwrap_or(id),
        username,
        program_name,
        input,
        client_id,
        capture_outputs,
        result_tx,
    );
    if let Err(error) = SERVICES.spawn(id, || process) {
        if let Some(planner) = crate::planner::planner() {
            planner.unregister(id);
        }
        return Err(error);
    }

    Ok(id)
}

/// Attach a client to a process.
pub async fn attach(process_id: ProcessId, client_id: ClientId) -> Result<()> {
    let process_id = resolve(process_id);
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        &process_id,
        Message::AttachClient {
            client_id,
            response: tx,
        },
    )?;
    rx.await?
}

/// Detach the current client from a process (fire-and-forget).
pub fn detach(process_id: ProcessId) {
    let _ = SERVICES.send(&resolve(process_id), Message::DetachClient);
}

/// Terminate a process (fire-and-forget).
pub fn terminate(process_id: ProcessId, result: Result<String, String>) {
    let process_id = resolve(process_id);
    // Early wait-set drop for a LIVE process: the scheduler stops holding
    // waves for this pid immediately instead of at the teardown's own
    // leave. Guarded on registry delivery so a terminate aimed at an
    // already-quiesced pid cannot mint a fresh tombstone after
    // ProcessQuiesced retired it.
    if SERVICES
        .send(&process_id, Message::Terminate { result })
        .is_ok()
    {
        crate::scheduler::worker::post_process_terminate(process_id);
    }
}

/// Send stdout output from a WASM instance to its process (fire-and-forget).
pub fn stdout(process_id: ProcessId, content: String) {
    let _ = SERVICES.send(&process_id, Message::Stdout { content });
}

/// Send stderr output from a WASM instance to its process (fire-and-forget).
pub fn stderr(process_id: ProcessId, content: String) {
    let _ = SERVICES.send(&process_id, Message::Stderr { content });
}

/// Get the username of a process.
pub async fn get_username(process_id: ProcessId) -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetUsername { response: tx })?;
    Ok(rx.await??)
}

/// Get the client ID attached to a process, if any.
pub async fn get_client_id(process_id: ProcessId) -> Result<Option<ClientId>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetClientId { response: tx })?;
    Ok(rx.await??)
}

/// Returns stats/metadata for a single process.
pub async fn get_stats(process_id: ProcessId) -> Result<ProcessStats> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetStats { response: tx })?;
    rx.await?
}

/// List all registered process IDs.
pub fn list() -> Vec<ProcessId> {
    SERVICES.keys()
}

/// Stats snapshot for a single process (serialized in list_processes responses).
#[derive(Debug, serde::Serialize)]
pub struct ProcessStats {
    pub id: String,
    pub username: String,
    pub program: String,
    pub input: String,
    pub elapsed_secs: u64,
}

// =============================================================================
// Messages
// =============================================================================

/// Messages that can be sent directly to a Process.
enum Message {
    /// Attach a client to this process
    AttachClient {
        client_id: ClientId,
        response: oneshot::Sender<Result<()>>,
    },
    /// Detach the current client
    DetachClient,
    /// Terminate this process (Ok = return value, Err = exception)
    Terminate { result: Result<String, String> },

    /// Stdout output from the WASM instance
    Stdout { content: String },
    /// Query the process username
    GetUsername {
        response: oneshot::Sender<Result<String>>,
    },
    /// Stderr output from the WASM instance
    Stderr { content: String },
    /// Query the attached client ID
    GetClientId {
        response: oneshot::Sender<Result<Option<ClientId>>>,
    },
    /// Query process stats/metadata
    GetStats {
        response: oneshot::Sender<Result<ProcessStats>>,
    },
}

// =============================================================================
// Process
// =============================================================================

/// Maximum number of output entries kept in the ring buffer.
const OUTPUT_BUFFER_CAP: usize = 4096;

/// Actor managing a single WASM instance lifecycle.
struct Process {
    process_id: ProcessId,
    /// The id this work is known by OUTSIDE the engine. Equal to
    /// `process_id` except after a restart, where the re-run inherits the
    /// original so a client's handle keeps resolving to it.
    client_pid: ProcessId,
    username: String,
    program: ProgramName,
    input: String,
    start_time: Instant,
    handle: JoinHandle<()>,
    client_id: Option<ClientId>,
    capture_outputs: bool,
    output_buffer: VecDeque<ProcessEvent>,
    /// Shared with the WASM task. Whoever takes it first (the run loop on
    /// normal completion, or an external terminate) delivers the result.
    result_tx: SharedResultTx,
}

impl Process {
    /// Creates a new Process, generating a UUID, and spawns its WASM execution task.
    fn new(
        process_id: ProcessId,
        client_pid: ProcessId,
        username: String,
        program: ProgramName,
        input: String,
        client_id: Option<ClientId>,
        capture_outputs: bool,
        result_tx: SharedResultTx,
    ) -> Self {
        let task = Self::run(
            process_id,
            username.clone(),
            program.clone(),
            input.clone(),
            capture_outputs,
            result_tx.clone(),
        );
        let handle = if crate::scheduler::fire_timing_enabled() {
            tokio::spawn(crate::scheduler::CpuMetered::new(
                crate::scheduler::CpuClass::Process,
                task,
            ))
        } else {
            tokio::spawn(task)
        };

        Process {
            process_id,
            client_pid,
            username,
            program,
            input,
            start_time: Instant::now(),
            handle,
            client_id,
            capture_outputs,
            output_buffer: VecDeque::new(),
            result_tx,
        }
    }

    /// Deliver an event to the attached client and/or the parent workflow.
    fn deliver_event(&mut self, event: ProcessEvent) {
        // Deliver to attached client
        if let Some(client_id) = self.client_id {
            if server::send_event(client_id, self.client_pid, &event).is_err() {
                self.client_id = None;
                self.buffer_event(event);
            }
        } else if self.capture_outputs {
            self.buffer_event(event);
        }
    }

    /// Push an event into the ring buffer, evicting the oldest entry if full.
    fn buffer_event(&mut self, event: ProcessEvent) {
        if self.output_buffer.len() >= OUTPUT_BUFFER_CAP {
            self.output_buffer.pop_front();
        }
        self.output_buffer.push_back(event);
    }

    /// Flush buffered events to the attached client.
    /// On failure, detaches the client and retains undelivered entries.
    fn flush_output_buffer(&mut self) {
        let Some(client_id) = self.client_id else {
            return;
        };
        while let Some(event) = self.output_buffer.pop_front() {
            if server::send_event(client_id, self.client_pid, &event).is_err() {
                self.client_id = None;
                self.output_buffer.push_front(event);
                break;
            }
        }
    }

    /// Runs the WASM component: instantiate, find the `run` export, and call it.
    async fn run(
        process_id: ProcessId,
        username: String,
        program: ProgramName,
        input: String,
        capture_outputs: bool,
        result_tx: SharedResultTx,
    ) {
        // Prewarm admission: a bounded next cohort instantiates (and may
        // compile/register its hash-deduped program) while the active cohort
        // executes. The REAL concurrency permit is acquired lazily by
        // `ensure_execution_admitted` at the first per-instance driver or
        // pooled-resource operation, and held for the rest of the run.
        let launch_timing = crate::scheduler::fire_timing_enabled().then(|| {
            (
                crate::scheduler::fire_timing_now_us(),
                crate::scheduler::fire_timing_unix_us(),
            )
        });
        let prewarm_permit = match PREWARM_ADMISSION.get().and_then(|s| s.as_ref()) {
            Some(sem) => Some(
                Arc::clone(sem)
                    .acquire_owned()
                    .await
                    .expect("prewarm admission semaphore closed"),
            ),
            None => None,
        };
        if let Some((launched_us, launched_unix_us)) = launch_timing {
            let acquired_us = crate::scheduler::fire_timing_now_us();
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "runtime",
                "event": "process_launch",
                "process_id": process_id,
                "launched_us": launched_us,
                "launched_unix_us": launched_unix_us,
                "prewarm_admitted_us": acquired_us,
                "prewarm_wait_us": acquired_us.saturating_sub(launched_us),
            }));
        }
        let mut admission_wait_us = 0u64;
        let mut instantiate_us = 0u64;
        let context_register_us = 0u64;
        let mut wasm_run_us = 0u64;
        let result: Result<String, String> = async {
            let instantiate_start = Instant::now();
            let output = if capture_outputs {
                OutputMode::Stream
            } else {
                OutputMode::Discard
            };
            let (mut store, instance) = linker::instantiate(process_id, username, &program, output)
                .await
                .map_err(|e| e.to_string())?;
            instantiate_us = duration_us(instantiate_start.elapsed());
            store.data_mut().install_prewarm_permit(prewarm_permit);

            // (KV admission via the context actor removed — Phase 5; physical
            // admission is now the unified arena's concern.)

            // Every inferlet now exports the same stock `pie:inferlet/run`
            // (WIT-refactor Phase 2 — the per-package synthesized export is
            // gone). Program identity comes from `program.name` metadata, not
            // the export interface name. The name is version-qualified: an
            // unversioned lookup does NOT match a versioned component export in
            // wasmtime's semver-aware name map, so this must track the
            // `pie:inferlet@<version>` package version declared in world.wit.
            // `run_interface_version_matches_wit` pins the two together — the
            // 0.2.0 -> 0.3.0 bump was missed once and made EVERY inferlet
            // unloadable, a break only an e2e run could surface.
            let run_interface = RUN_INTERFACE;

            let (_, run_export) = instance
                .get_export(&mut store, None, run_interface)
                .ok_or_else(|| "No 'run' interface found".to_string())?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| "No 'run' function found".to_string())?;

            let run_func = instance
                .get_typed_func::<(&str,), (Result<String, String>,)>(&mut store, &run_func_export)
                .map_err(|e| format!("Failed to get 'run' function: {e:?}"))?;

            if crate::scheduler::fire_timing_enabled() {
                crate::scheduler::fire_timing_write(&serde_json::json!({
                    "schema": 1,
                    "source": "runtime",
                    "event": "guest_main_entered",
                    "process_id": process_id,
                    "entered_us": crate::scheduler::fire_timing_now_us(),
                }));
            }
            let wasm_run_start = Instant::now();
            let call = run_func.call_async(&mut store, (&input,));
            let called = if crate::scheduler::fire_timing_enabled() {
                crate::scheduler::CpuMetered::new(crate::scheduler::CpuClass::Guest, call).await
            } else {
                call.await
            };
            let result = match called {
                Ok((Ok(output),)) => {
                    wasm_run_us = duration_us(wasm_run_start.elapsed());
                    Ok(output)
                }
                Ok((Err(runtime_err),)) => {
                    wasm_run_us = duration_us(wasm_run_start.elapsed());
                    Err(runtime_err)
                }
                Err(call_err) => {
                    wasm_run_us = duration_us(wasm_run_start.elapsed());
                    Err(format!("Call error: {call_err}"))
                }
            };
            if crate::scheduler::fire_timing_enabled() {
                crate::scheduler::fire_timing_write(&serde_json::json!({
                    "schema": 1,
                    "source": "runtime",
                    "event": "guest_main_returned",
                    "process_id": process_id,
                    "returned_us": crate::scheduler::fire_timing_now_us(),
                }));
            }
            admission_wait_us = store.data().admission_wait_us();
            // Drop the store HERE rather than at the end of this block, so
            // the wasmtime teardown it triggers (`ProcessCtx::drop` and the
            // instance's own memory) is visible: at a cohort boundary 512 of
            // these run at once and the record is the only way to see them.
            if crate::scheduler::fire_timing_enabled() {
                let store_drop_started_us = crate::scheduler::fire_timing_now_us();
                drop(store);
                crate::scheduler::fire_timing_write(&serde_json::json!({
                    "schema": 1,
                    "source": "runtime",
                    "event": "process_store_drop",
                    "process_id": process_id,
                    "started_us": store_drop_started_us,
                    "store_drop_us":
                        crate::scheduler::fire_timing_now_us() - store_drop_started_us,
                    "tid": os_thread_id(),
                }));
            } else {
                drop(store);
            }
            result
        }
        .await;
        record_process_timing(
            admission_wait_us,
            instantiate_us,
            context_register_us,
            wasm_run_us,
        );

        if let Err(ref err) = result {
            tracing::info!("Process {process_id} failed: {err}");
        }

        // Fire result channel if a parent is waiting (and an external
        // terminate hasn't already claimed it). A process the planner has
        // asked to restart leaves the channel in place: its failure is not
        // this request's outcome, and the actor's `terminate` hands the
        // channel to the re-run.
        if !restart_requested(process_id)
            && let Some(tx) = result_tx.lock().unwrap().take()
        {
            let _ = tx.send(result.clone());
        }

        let terminate_started_us =
            crate::scheduler::fire_timing_enabled().then(crate::scheduler::fire_timing_now_us);
        terminate(process_id, result);
        if let Some(started_us) = terminate_started_us {
            crate::scheduler::fire_timing_write(&serde_json::json!({
                "schema": 1,
                "source": "runtime",
                "event": "process_terminate",
                "process_id": process_id,
                "started_us": started_us,
                "terminate_us": crate::scheduler::fire_timing_now_us() - started_us,
            }));
        }
    }

    /// Re-run this program from the beginning, carrying the caller's reply
    /// channel, the client-facing process id and the planner's FCFS position
    /// across to the new process.
    ///
    /// Nothing outside the engine observes the handover: an attached client
    /// keeps receiving events under the id it launched, and a parent waiting
    /// on the launch handle still gets exactly one result, because the reply
    /// cell is shared rather than moved. If the spawn fails the sender is
    /// still in that cell, so a failed restart degrades to today's fail-loud
    /// behaviour instead of losing the request.
    fn restart(&mut self) -> bool {
        let seq = crate::planner::planner().and_then(|planner| planner.spawn_seq(self.process_id));
        let spawned = spawn_inner(
            self.username.clone(),
            self.program.clone(),
            self.input.clone(),
            self.client_id,
            self.capture_outputs,
            self.result_tx.clone(),
            seq,
            Some(self.client_pid),
        );
        match spawned {
            Ok(new_id) => {
                // Published before this process finishes tearing down, so a
                // client message aimed at the original id in the handover
                // window reaches the re-run rather than the corpse.
                RESTART_ALIAS
                    .write()
                    .unwrap()
                    .insert(self.client_pid, new_id);
                RESTART_TOTAL.fetch_add(1, Relaxed);
                tracing::info!(
                    old = %self.process_id,
                    new = %new_id,
                    client_pid = %self.client_pid,
                    "process restarted after KV reclaim",
                );
                true
            }
            Err(error) => {
                tracing::error!(pid = %self.process_id, %error, "process restart failed");
                false
            }
        }
    }

    /// Abort the WASM execution task, notify any attached client, and unregister.
    fn terminate(&mut self, result: Result<String, String>) {
        self.handle.abort();

        // The planner reclaimed this process's pages and asked for a re-run
        // rather than a failure. A fresh process takes over the caller's
        // reply channel and this one's FCFS position; nothing is delivered
        // for the abandoned attempt. Teardown below is unchanged and
        // unconditional — the restart is only worth anything if this
        // process's pages actually go back to the pool.
        let restarted = restart_requested(self.process_id) && self.restart();
        forget_restart_state(self.process_id);

        if !restarted {
            // Deliver `result` to any parent waiting on the launch handle, if
            // the run loop didn't already send it (e.g., external Terminate
            // fires before the WASM task finished). First taker wins.
            if let Some(tx) = self.result_tx.lock().unwrap().take() {
                let _ = tx.send(result.clone());
            }

            // Notify attached client / workflow
            match result {
                Ok(output) => self.deliver_event(ProcessEvent::Return(output)),
                Err(msg) => self.deliver_event(ProcessEvent::Error(msg)),
            }
        }

        let _ = server::inbox::clear(self.process_id.to_string());
        SERVICES.remove(&self.process_id);

        // (No leave broadcast here: natural completion's run loop already
        // sent the early leave via the free `terminate` fn above, and the
        // deferred teardown sends the fenced one — a third copy from this
        // actor could land after the teardown's ProcessQuiesced and mint a
        // tombstone nothing retires.)

        // Residency: unregister from the planner (purges its queue entries,
        // wakes gate waiters for teardown, and re-plans — the exiting
        // process's KV frees follow via the WS-drop hook). Single exit
        // funnel: covers natural completion AND external terminate.
        if !restarted {
            RESTART_ALIAS.write().unwrap().remove(&self.client_pid);
        }
        if let Some(planner) = crate::planner::planner() {
            planner.unregister(self.process_id);
        }
        residency::unregister_residency(self.process_id);
    }
}

impl ServiceHandler for Process {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::AttachClient {
                client_id,
                response,
            } => {
                if self.client_id.is_some() {
                    let _ = response.send(Err(anyhow!("already attached")));
                } else {
                    self.client_id = Some(client_id);
                    self.flush_output_buffer();
                    let _ = response.send(Ok(()));
                }
            }

            Message::DetachClient => {
                self.client_id = None;
            }

            Message::Terminate { result } => {
                self.terminate(result);
            }

            Message::Stdout { content } => self.deliver_event(ProcessEvent::Stdout(content)),
            Message::Stderr { content } => self.deliver_event(ProcessEvent::Stderr(content)),

            Message::GetUsername { response } => {
                let _ = response.send(Ok(self.username.clone()));
            }

            Message::GetClientId { response } => {
                let _ = response.send(Ok(self.client_id));
            }

            Message::GetStats { response } => {
                let _ = response.send(Ok(ProcessStats {
                    id: self.process_id.to_string(),
                    username: self.username.clone(),
                    program: self.program.to_string(),
                    input: self.input.clone(),
                    elapsed_secs: self.start_time.elapsed().as_secs(),
                }));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RUN_INTERFACE;

    /// `RUN_INTERFACE` is a hand-written string that must track the WIT package
    /// version. When the package went 0.2.0 -> 0.3.0 this constant was left
    /// behind, and because the lookup fails at component-instantiation time the
    /// only symptom was every program dying with "No 'run' interface found" —
    /// invisible to `cargo test` and visible only on a box with real weights.
    #[test]
    fn run_interface_version_matches_wit() {
        let wit = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../interface/inferlet/world.wit"),
        )
        .expect("read interface/inferlet/world.wit");

        let declared = wit
            .lines()
            .find_map(|l| l.trim().strip_prefix("package pie:inferlet@"))
            .map(|v| v.trim_end_matches(';').trim().to_string())
            .expect("world.wit declares `package pie:inferlet@<version>;`");

        let expected = format!("pie:inferlet/run@{declared}");
        assert_eq!(
            RUN_INTERFACE, expected,
            "process.rs RUN_INTERFACE is stale: world.wit declares \
             pie:inferlet@{declared}. wasmtime's export lookup is semver-exact, \
             so a mismatch makes EVERY inferlet fail to start."
        );
    }
}
