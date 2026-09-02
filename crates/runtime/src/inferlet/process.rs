//! Inferlet process lifecycle: each `Process` is a `ServiceMap` actor
//! managing one WASM instance. KV eviction/restore is planner-owned
//! (`crate::planner`); this module only runs the `gate` prologue.

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

/// The versioned component export every inferlet provides. Must exactly match
/// the package version in `crates/inferlet/wit/world.wit` — wasmtime's export
/// lookup is semver-exact, so a stale version here silently fails every
/// program. Checked by `tests::run_interface_version_matches_wit`.
const RUN_INTERFACE: &str = "pie:inferlet/run@0.3.0";

/// Processes whose guest called `system.declare-restartable`, and the ones
/// the planner has since asked to restart. Two sets rather than one flag on
/// the actor because both are read from outside the actor's mailbox, before
/// the planner destroys an allocation.
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
/// instead of its failure being delivered to the caller. Returns false if it
/// never declared itself restartable, in which case the caller must fail it
/// loud instead.
pub(crate) fn request_restart(process_id: ProcessId) -> bool {
    if !is_restartable(process_id) {
        return false;
    }
    RESTART_REQUESTED.write().unwrap().insert(process_id);
    true
}

/// Original (client-facing) process id -> the live process currently running
/// that work, for requests that have been restarted. A restart cannot reuse
/// the process id (the scheduler keeps a terminate tombstone for the
/// retiring pid until its quiesce lands), so the re-run gets a fresh internal
/// id and inherits only the client-facing one.
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
/// Bind admission: gates per-instance engine state creation (channel
/// registration, instance bind, working-set declaration). Sized at twice the
/// execution limit (executing cohort + one staged cohort) so the next
/// generation's bring-up overlaps current execution.
static BIND_ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
/// Prewarm admission: a bounded next cohort may instantiate its WASM and
/// compile/register its program while the active cohort executes. Anything
/// that creates per-instance engine state or claims pooled resources still
/// waits for [`ensure_execution_admitted`].
static PREWARM_ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
/// The execution pool's configured capacity (None = unlimited): seeds the
/// frame policy's free-slot balance at bootstrap.
static EXECUTION_SLOT_CAPACITY: OnceLock<Option<usize>> = OnceLock::new();

pub(crate) fn execution_slot_capacity() -> Option<usize> {
    EXECUTION_SLOT_CAPACITY.get().copied().flatten()
}

/// Processes currently registered — i.e. guests that may still submit. The
/// quiesce test for `crate::scheduler::reconfigure`: batching knobs a guest
/// has already been told (`model.frame-size()`) can only move once none
/// remain.
pub fn live_count() -> usize {
    SERVICES.len()
}

/// Prewarm-conveyor width when execution admission is uncapped (with a cap
/// the conveyor is one cohort wide instead; see `init_admission`).
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
        avg_admission_wait_us: admission.checked_div(completed).unwrap_or(0),
        last_admission_wait_us: PROCESS_LAST_ADMISSION_WAIT_US.load(Relaxed),
        cumulative_instantiate_us: instantiate,
        avg_instantiate_us: instantiate.checked_div(completed).unwrap_or(0),
        last_instantiate_us: PROCESS_LAST_INSTANTIATE_US.load(Relaxed),
        cumulative_context_register_us: context_register,
        avg_context_register_us: context_register.checked_div(completed).unwrap_or(0),
        last_context_register_us: PROCESS_LAST_CONTEXT_REGISTER_US.load(Relaxed),
        cumulative_wasm_run_us: wasm_run,
        avg_wasm_run_us: wasm_run.checked_div(completed).unwrap_or(0),
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
    // The prewarm conveyor bounds instantiation: a process holds its slot
    // from spawn until it wins a bind permit. One cohort wide when execution
    // is capped, so a turnover can hand its whole successor cohort a slot at
    // once; the flat UNCAPPED_PREWARM_PROCESSES ladder stands when execution
    // is uncapped (no cohort to size it by), to avoid a thundering herd of
    // Store/linker/WASI setup.
    let prewarm = Some(Arc::new(Semaphore::new(
        limit.unwrap_or(UNCAPPED_PREWARM_PROCESSES),
    )));
    // Double-buffered bring-up: the executing cohort plus STAGED_COHORTS
    // successor cohorts hold bind permits, so a staged cohort instantiates
    // and binds during the previous generation's execution. Depth 1 is the
    // structural depth (a turnover consumes exactly one cohort); measured
    // depths 2/3 showed no throughput gain.
    const STAGED_COHORTS: usize = 1;
    // The staged half opens only once the first cohort is fully seated, so
    // the extra bind capacity overlaps a running generation rather than
    // competing with initial bring-up.
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

/// Bind permits withheld from the pool until the first generation is
/// seated (see `init_admission`). Handed over exactly once.
static BIND_STAGED_RESERVE: AtomicUsize = AtomicUsize::new(0);

/// Open the staged half of the bind pool. Called the moment execution
/// admission runs out of seats — that is the runtime's own statement that a
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

// Cohort-boundary bind deferral: a retiring process returns its bind permit
// at teardown, but releasing it immediately at a fleet-wide turnover would
// admit a staged cohort into working-set declaration right as the boundary
// frame is trying to gather. The gate parks released permits while a join
// is in flight and hands them over once the boundary frame is away.
// LIVENESS: cleared whenever the scheduler pass makes no progress with
// nothing in flight, so it can never be the last thing standing. SCOPE:
// process-global, reset only by process exit.
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
        // Re-checked under the lock: the opener clears the flag before
        // taking the lock, so no ordering can strand a permit.
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

/// Bind gate: acquire the bind permit lazily, at the first operation that
/// creates per-instance engine state (channel registration / instance bind /
/// working-set declaration). Idempotent per process.
pub(crate) async fn ensure_bind_admitted(ctx: &mut ProcessCtx) {
    if ctx.bind_admitted() {
        return;
    }
    // The prewarm conveyor slot is released after the bind wait, not before:
    // a process that cannot bind cannot make engine progress, so releasing
    // it earlier would only let the next arrival instantiate work nothing is
    // waiting for.
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
}

/// Strict-admission gate: acquire the execution permit lazily at fire
/// submit. Idempotent per process. Bind admission is always acquired first,
/// so the two gates cannot deadlock.
pub(crate) async fn ensure_execution_admitted(ctx: &mut ProcessCtx) {
    ensure_bind_admitted(ctx).await;
    if ctx.execution_admitted() {
        return;
    }
    let started = Instant::now();
    let permit = match ADMISSION.get().and_then(|value| value.as_ref()) {
        Some(semaphore) => {
            // Announce the wait before blocking on it (FIFO-fair semaphore),
            // so the frame seal can name whom it's holding for. Dropped on
            // cancellation too.
            let _queued = AdmissionQueued::enter(ctx.id());
            let permit = Arc::clone(semaphore)
                .acquire_owned()
                .await
                .expect("admission semaphore closed");
            if semaphore.available_permits() == 0 {
                // Last seat of a generation went out: staging can overlap now.
                open_staged_bind_pool();
            }
            // Every capped admission notifies, even uncontended ones, so the
            // policy's slot balance sees each consumed permit before the
            // first fire on this task.
            crate::scheduler::worker::notify_execution_slot_consumed(ctx.id());
            Some(permit)
        }
        None => None,
    };
    ctx.admit_execution(permit, duration_us(started.elapsed()));
    // Only from here on can this process hold pooled pages, distinct from
    // planner registration at spawn.
    if let Some(planner) = crate::planner::planner() {
        planner.note_admitted(ctx.id());
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
/// clock across a restart instead of taking a fresh (youngest) one: victims
/// are chosen youngest-first, so a restart re-entering as youngest would be
/// re-chosen forever.
#[allow(
    clippy::too_many_arguments,
    reason = "one spawn request in full: who is asking, what program on what input, \
              which client to report to, whether to capture output, where the result \
              goes, and the two inheritance fields (`inherit_seq`, \
              `inherit_client_pid`) that are set only when a process spawns a child. \
              The last two are exactly why a struct would not help — they are \
              `None` for every top-level spawn"
)]
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
    // Early wait-set drop for a live process, guarded on registry delivery
    // so an already-quiesced pid cannot mint a fresh tombstone.
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
    rx.await?
}

/// Get the client ID attached to a process, if any.
pub async fn get_client_id(process_id: ProcessId) -> Result<Option<ClientId>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetClientId { response: tx })?;
    rx.await?
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
    /// The id this work is known by outside the runtime. Equal to
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
    #[allow(
        clippy::too_many_arguments,
        reason = "the resolved form of `spawn_inner`'s argument list, one level down: \
                  both process ids are now known, and the rest is carried through \
                  unchanged. Introducing a struct here would only move the same \
                  fields across one call"
    )]
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
        let handle = tokio::spawn(task);

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
        // The real concurrency permit is acquired lazily by
        // `ensure_execution_admitted`, held for the rest of the run.
        let prewarm_permit = match PREWARM_ADMISSION.get().and_then(|s| s.as_ref()) {
            Some(sem) => Some(
                Arc::clone(sem)
                    .acquire_owned()
                    .await
                    .expect("prewarm admission semaphore closed"),
            ),
            None => None,
        };
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

            // Program identity comes from `program.name` metadata, not this
            // export interface name; see `RUN_INTERFACE`.
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

            let wasm_run_start = Instant::now();
            let call = run_func.call_async(&mut store, (&input,));
            let called = call.await;
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
            admission_wait_us = store.data().admission_wait_us();
            // Drop here, not at block end, so the wasmtime teardown it
            // triggers is visible in the timing record.
            drop(store);
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

        // A process the planner asked to restart leaves the channel in
        // place; `terminate` hands it to the re-run instead.
        if !restart_requested(process_id)
            && let Some(tx) = result_tx.lock().unwrap().take()
        {
            let _ = tx.send(result.clone());
        }

        terminate(process_id, result);
    }

    /// Re-run this program from the beginning, carrying the caller's reply
    /// channel, the client-facing process id and the planner's FCFS position
    /// across to the new process. The reply cell is shared rather than moved,
    /// so a failed spawn still degrades to fail-loud instead of losing the
    /// request.
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
                // client message in the handover window reaches the re-run.
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

        // If the planner asked for a re-run, a fresh process takes over the
        // reply channel and FCFS position; nothing is delivered for this
        // abandoned attempt. Teardown below still runs unconditionally.
        let restarted = restart_requested(self.process_id) && self.restart();
        forget_restart_state(self.process_id);

        if !restarted {
            // First taker wins: the run loop may already have sent this.
            if let Some(tx) = self.result_tx.lock().unwrap().take() {
                let _ = tx.send(result.clone());
            }

            match result {
                Ok(output) => self.deliver_event(ProcessEvent::Return(output)),
                Err(msg) => self.deliver_event(ProcessEvent::Error(msg)),
            }
        }

        let _ = server::inbox::clear(self.process_id.to_string());
        SERVICES.remove(&self.process_id);

        // No leave broadcast here: the free `terminate` fn and the deferred
        // teardown already send it; a third copy could mint a stray
        // tombstone.

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

