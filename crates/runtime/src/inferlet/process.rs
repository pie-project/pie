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
use tokio::sync::{Semaphore, oneshot, watch};
use tokio::task::JoinHandle;
use uuid::Uuid;

type SharedResultTx = Arc<Mutex<Option<oneshot::Sender<Result<String, String>>>>>;

use crate::server::{self, ClientId};
use crate::service::{ServiceHandler, ServiceMap};

use super::linker;
use super::program::ProgramName;

const RUN_INTERFACE: &str = "pie:inferlet/run@0.3.0";

/// Counts process capacity releases, so a load reporter can sample on the
/// release instead of on its own timer. A watch counter rather than a
/// `Notify` because a reporter that was busy when the release happened still
/// sees it on its next `changed()`. Only teardown moves it: the per-fire free
/// paths are too hot to wake a reporter on.
static RELEASES: LazyLock<watch::Sender<u64>> = LazyLock::new(|| watch::Sender::new(0));

pub(crate) fn note_release() {
    RELEASES.send_modify(|count| *count += 1);
}

/// Observes process capacity releases; the value only ever increases.
pub fn releases() -> watch::Receiver<u64> {
    RELEASES.subscribe()
}

static RESTARTABLE: LazyLock<RwLock<HashSet<ProcessId>>> = LazyLock::new(Default::default);
static RESTART_REQUESTED: LazyLock<RwLock<HashSet<ProcessId>>> = LazyLock::new(Default::default);

pub(crate) fn declare_restartable(process_id: ProcessId) {
    RESTARTABLE.write().unwrap().insert(process_id);
}

pub(crate) fn is_restartable(process_id: ProcessId) -> bool {
    RESTARTABLE.read().unwrap().contains(&process_id)
}

pub(crate) fn request_restart(process_id: ProcessId) -> bool {
    if !is_restartable(process_id) {
        return false;
    }
    RESTART_REQUESTED.write().unwrap().insert(process_id);
    true
}

static RESTART_ALIAS: LazyLock<RwLock<HashMap<ProcessId, ProcessId>>> =
    LazyLock::new(Default::default);

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

static RESTART_TOTAL: AtomicUsize = AtomicUsize::new(0);

pub fn restart_total() -> usize {
    RESTART_TOTAL.load(Relaxed)
}

#[derive(Debug, Clone)]
pub enum ProcessEvent {
    Stdout(String),
    Stderr(String),
    Message(String),
    Return(String),
    Error(String),
}

impl ProcessEvent {
    /// The two events a process ends with. Named because a session that only
    /// has the wire form still has to recognise "this process is over".
    pub const RETURN: &'static str = "return";
    pub const ERROR: &'static str = "error";

    pub fn name(&self) -> &'static str {
        match self {
            Self::Stdout(_) => "stdout",
            Self::Stderr(_) => "stderr",
            Self::Message(_) => "message",
            Self::Return(_) => Self::RETURN,
            Self::Error(_) => Self::ERROR,
        }
    }

    pub fn value(&self) -> &str {
        match self {
            Self::Stdout(v)
            | Self::Stderr(v)
            | Self::Message(v)
            | Self::Return(v)
            | Self::Error(v) => v,
        }
    }

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

pub type ProcessId = Uuid;

static SERVICES: LazyLock<ServiceMap<ProcessId, Message>> = LazyLock::new(ServiceMap::new);

static ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
static BIND_ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
static PREWARM_ADMISSION: OnceLock<Option<Arc<Semaphore>>> = OnceLock::new();
static EXECUTION_SLOT_CAPACITY: OnceLock<Option<usize>> = OnceLock::new();

pub(crate) fn execution_slot_capacity() -> Option<usize> {
    EXECUTION_SLOT_CAPACITY.get().copied().flatten()
}

pub fn live_count() -> usize {
    SERVICES.len()
}

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

pub fn init_admission(max_concurrent: Option<usize>) {
    let limit = max_concurrent.filter(|&n| n > 0);
    let sem = limit.map(|n| Arc::new(Semaphore::new(n)));
    let prewarm = Some(Arc::new(Semaphore::new(
        limit.unwrap_or(UNCAPPED_PREWARM_PROCESSES),
    )));
    const STAGED_COHORTS: usize = 1;
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

static BIND_STAGED_RESERVE: AtomicUsize = AtomicUsize::new(0);

fn open_staged_bind_pool() {
    let reserve = BIND_STAGED_RESERVE.swap(0, Relaxed);
    if reserve == 0 {
        return;
    }
    if let Some(Some(semaphore)) = BIND_ADMISSION.get() {
        semaphore.add_permits(reserve);
    }
}

static BIND_RELEASE_HOLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static HELD_BIND_PERMITS: Mutex<Vec<tokio::sync::OwnedSemaphorePermit>> = Mutex::new(Vec::new());

pub(crate) fn release_bind_permit(permit: Option<tokio::sync::OwnedSemaphorePermit>) {
    let Some(permit) = permit else {
        return;
    };
    if BIND_RELEASE_HOLD.load(Relaxed) {
        let mut held = HELD_BIND_PERMITS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if BIND_RELEASE_HOLD.load(Relaxed) {
            held.push(permit);
            return;
        }
    }
    drop(permit);
}

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

pub(crate) async fn ensure_bind_admitted(ctx: &mut ProcessCtx) {
    if ctx.bind_admitted() {
        return;
    }
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

pub(crate) async fn ensure_execution_admitted(ctx: &mut ProcessCtx) {
    ensure_bind_admitted(ctx).await;
    if ctx.execution_admitted() {
        return;
    }
    let started = Instant::now();
    let permit = match ADMISSION.get().and_then(|value| value.as_ref()) {
        Some(semaphore) => {
            let _queued = AdmissionQueued::enter(ctx.id());
            let permit = Arc::clone(semaphore)
                .acquire_owned()
                .await
                .expect("admission semaphore closed");
            if semaphore.available_permits() == 0 {
                open_staged_bind_pool();
            }
            crate::scheduler::worker::notify_execution_slot_consumed(ctx.id());
            Some(permit)
        }
        None => None,
    };
    ctx.admit_execution(permit, duration_us(started.elapsed()));
    if let Some(planner) = crate::planner::planner() {
        planner.note_admitted(ctx.id());
    }
}

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

pub fn detach(process_id: ProcessId) {
    let _ = SERVICES.send(&resolve(process_id), Message::DetachClient);
}

/// Terminate a process (fire-and-forget). The caller is an owner — a closing
/// session, a client, or a fail-loud path — so this never restarts.
pub fn terminate(process_id: ProcessId, result: Result<String, String>) {
    terminate_with(process_id, result, Origin::Owner);
}

/// The run task reporting its own end: the one termination a parked restart
/// request may replace.
fn terminate_from_run(process_id: ProcessId, result: Result<String, String>) {
    terminate_with(process_id, result, Origin::Run);
}

fn terminate_with(process_id: ProcessId, result: Result<String, String>, origin: Origin) {
    let target = resolve(process_id);
    // The result is cloned for the first attempt because `ServiceMap::send`
    // consumes the message and does not hand it back on failure; one string
    // clone per termination is cheaper than widening that signature.
    if deliver_terminate(target, result.clone(), origin) {
        return;
    }
    // The other side of the handover window described on `Process::successor`:
    // the resolve above can read the alias a moment before the restart
    // publishes it, and the old actor is gone by the time the send lands.
    let republished = resolve(process_id);
    if republished != target {
        deliver_terminate(republished, result, origin);
    }
}

/// Post a Terminate to one live actor; false if it is no longer registered.
fn deliver_terminate(
    process_id: ProcessId,
    result: Result<String, String>,
    origin: Origin,
) -> bool {
    // Early wait-set drop for a live process, guarded on registry delivery
    // so an already-quiesced pid cannot mint a fresh tombstone.
    if SERVICES
        .send(&process_id, Message::Terminate { result, origin })
        .is_err()
    {
        return false;
    }
    crate::scheduler::worker::post_process_terminate(process_id);
    true
}

pub fn stdout(process_id: ProcessId, content: String) {
    let _ = SERVICES.send(&process_id, Message::Stdout { content });
}

pub fn stderr(process_id: ProcessId, content: String) {
    let _ = SERVICES.send(&process_id, Message::Stderr { content });
}

pub async fn get_username(process_id: ProcessId) -> Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetUsername { response: tx })?;
    rx.await?
}

pub async fn get_client_id(process_id: ProcessId) -> Result<Option<ClientId>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetClientId { response: tx })?;
    rx.await?
}

pub async fn get_stats(process_id: ProcessId) -> Result<ProcessStats> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::GetStats { response: tx })?;
    rx.await?
}

pub fn list() -> Vec<ProcessId> {
    SERVICES.keys()
}

#[derive(Debug, serde::Serialize)]
pub struct ProcessStats {
    pub id: String,
    pub username: String,
    pub program: String,
    pub input: String,
    pub elapsed_secs: u64,
}

/// Who asked for a process to end. Carried on the message rather than
/// inferred from the result, because the two are indistinguishable there: an
/// owner's kill and a guest failure both arrive as an `Err` string.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Origin {
    /// The run task reporting its own completion.
    Run,
    /// A session closing, a client kill, a signal, or a worker cancel.
    Owner,
}

enum Message {
    AttachClient {
        client_id: ClientId,
        response: oneshot::Sender<Result<()>>,
    },
    DetachClient,
    Terminate {
        result: Result<String, String>,
        origin: Origin,
    },

    Stdout {
        content: String,
    },
    GetUsername {
        response: oneshot::Sender<Result<String>>,
    },
    Stderr {
        content: String,
    },
    GetClientId {
        response: oneshot::Sender<Result<Option<ClientId>>>,
    },
    GetStats {
        response: oneshot::Sender<Result<ProcessStats>>,
    },
}

const OUTPUT_BUFFER_CAP: usize = 4096;

struct Process {
    process_id: ProcessId,
    client_pid: ProcessId,
    username: String,
    program: ProgramName,
    input: String,
    start_time: Instant,
    handle: JoinHandle<()>,
    client_id: Option<ClientId>,
    capture_outputs: bool,
    output_buffer: VecDeque<ProcessEvent>,
    result_tx: SharedResultTx,
    /// Set by the first `Terminate` this actor handles.
    terminated: bool,
    /// The replacement a restart spawned, if any.
    ///
    /// The handover window: a restart registers its successor in `SERVICES`
    /// before it publishes the alias naming it, so for that moment a
    /// Terminate for the client's pid still resolves here and lands in this
    /// mailbox after the end has been claimed. Whoever holds the successor
    /// forwards such a Terminate on, which is what keeps a re-run from
    /// outliving the session that closed it.
    successor: Option<ProcessId>,
}

impl Process {
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
            terminated: false,
            successor: None,
        }
    }

    fn deliver_event(&mut self, event: ProcessEvent) {
        if let Some(client_id) = self.client_id {
            if server::send_event(client_id, self.client_pid, &event).is_err() {
                self.client_id = None;
                self.buffer_event(event);
            }
        } else if self.capture_outputs {
            self.buffer_event(event);
        }
    }

    fn buffer_event(&mut self, event: ProcessEvent) {
        if self.output_buffer.len() >= OUTPUT_BUFFER_CAP {
            self.output_buffer.pop_front();
        }
        self.output_buffer.push_back(event);
    }

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

    async fn run(
        process_id: ProcessId,
        username: String,
        program: ProgramName,
        input: String,
        capture_outputs: bool,
        result_tx: SharedResultTx,
    ) {
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
        if crate::planner::trace_enabled() {
            println!(
                "[process t_us={} pid={}] guest finished ok={} restart_requested={}",
                crate::scheduler::fire_timing_now_us(),
                process_id,
                result.is_ok(),
                restart_requested(process_id)
            );
        }

        if !restart_requested(process_id)
            && let Some(tx) = result_tx.lock().unwrap().take()
        {
            let _ = tx.send(result.clone());
        }

        terminate_from_run(process_id, result);
    }

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
                self.successor = Some(new_id);
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

    fn terminate(&mut self, result: Result<String, String>, origin: Origin) {
        // Natural completion and the owning session are both entitled to end
        // a process, and `ServiceMap::remove` does not discard the messages
        // already in the mailbox, so two Terminates can arrive. Claim the end
        // before any of it happens — including the restart branch, whose
        // replacement a second pass would disown by clearing its alias.
        if self.terminated {
            // Relayed, not dropped, when a restart left a replacement behind:
            // see the handover window on `successor`. Resolve from the stable
            // client pid because the immediate successor may itself have
            // restarted and aliases are only published for that stable id.
            if let Some(successor) = self.successor {
                let target = resolve(self.client_pid);
                terminate_with(
                    if target == self.process_id {
                        successor
                    } else {
                        self.client_pid
                    },
                    result,
                    origin,
                );
            }
            return;
        }
        self.terminated = true;

        self.handle.abort();

        // If the planner asked for a re-run, a fresh process takes over the
        // reply channel and FCFS position; nothing is delivered for this
        // abandoned attempt. Teardown below still runs unconditionally.
        //
        // Only the run task's own end may be replaced. An owner's Terminate
        // aborts the run before the guest can finish, so honouring the
        // planner's parked request there would spawn a replacement the owner
        // never asked for, does not know about, and cannot reach to close.
        // The request is forgotten either way, so it cannot be honoured later.
        let restarted =
            origin == Origin::Run && restart_requested(self.process_id) && self.restart();
        forget_restart_state(self.process_id);

        if !restarted {
            if let Some(tx) = self.result_tx.lock().unwrap().take() {
                let _ = tx.send(result.clone());
            }

            if crate::planner::trace_enabled() {
                println!(
                    "[process t_us={} pid={}] delivering {}",
                    crate::scheduler::fire_timing_now_us(),
                    self.process_id,
                    if result.is_ok() { "return" } else { "error" }
                );
            }
            match result {
                Ok(output) => self.deliver_event(ProcessEvent::Return(output)),
                Err(msg) => self.deliver_event(ProcessEvent::Error(msg)),
            }
        }

        let _ = server::inbox::clear(self.process_id.to_string());
        SERVICES.remove(&self.process_id);
        note_release();

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

            Message::Terminate { result, origin } => {
                self.terminate(result, origin);
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

// =============================================================================
// Test probe
// =============================================================================

/// A process-shaped mailbox with no WASM task behind it.
///
/// A real `Process` cannot serve a model-free test: with no program loader
/// running, its run task fails on its first poll and terminates the actor
/// before an assertion can look at it. The probe registers in the same
/// `SERVICES` map, so an owner still reaches it through the ordinary
/// `terminate`/`detach` path.
#[cfg(test)]
pub(crate) mod probe {
    use super::{Message, ProcessId, SERVICES};
    use crate::service::ServiceHandler;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    /// What an owner asked a probe process to do.
    #[derive(Debug, PartialEq, Eq)]
    pub(crate) enum Saw {
        Terminated,
        Detached,
    }

    struct Probe {
        id: ProcessId,
        saw: mpsc::UnboundedSender<Saw>,
    }

    impl ServiceHandler for Probe {
        type Message = Message;

        async fn handle(&mut self, msg: Message) {
            let saw = match msg {
                Message::Terminate { .. } => Saw::Terminated,
                Message::DetachClient => Saw::Detached,
                _ => return,
            };
            let ended = saw == Saw::Terminated;
            let _ = self.saw.send(saw);
            // A real process unregisters when it ends, and only then — a
            // detached process is still live. Without this every probe any
            // test ever made would stay in the global map for the life of the
            // test binary. Reported first: the report rides an unbounded
            // channel, so the receiver still sees it after the removal drops
            // this actor.
            if ended {
                SERVICES.remove(&self.id);
            }
        }
    }

    /// Register a probe process; returns its id and its record of what it saw.
    pub(crate) fn register() -> (ProcessId, mpsc::UnboundedReceiver<Saw>) {
        let id = Uuid::new_v4();
        let (saw, rx) = mpsc::unbounded_channel();
        SERVICES
            .spawn(id, || Probe { id, saw })
            .expect("fresh process id");
        (id, rx)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// The real actor, built without `Process::new`. `new` spawns the run
    /// task, which cannot get past `linker::instantiate` with no program
    /// loader spawned — the error comes back before the first await point, so
    /// the task terminates the actor on its first poll, before any assertion
    /// could look at it. The join handle here is therefore an inert task and
    /// `Process::run` never executes: what follows covers `Process::terminate`
    /// and not the run path into it.
    fn idle_process() -> Process {
        let id = Uuid::new_v4();
        Process {
            process_id: id,
            client_pid: id,
            username: "test".to_string(),
            program: ProgramName {
                name: "test".to_string(),
                version: "0.0.0".to_string(),
            },
            input: String::new(),
            start_time: Instant::now(),
            handle: tokio::spawn(std::future::pending::<()>()),
            client_id: None,
            // No client attached, so the end event lands in the output buffer
            // instead, which is where this test counts it.
            capture_outputs: true,
            output_buffer: VecDeque::new(),
            result_tx: Arc::new(Mutex::new(None)),
            terminated: false,
            successor: None,
        }
    }

    /// Everything a probe saw, up to it unregistering and dropping its
    /// sender. Bounded, so a probe that never reports fails the test instead
    /// of hanging the suite.
    async fn drained(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<probe::Saw>,
    ) -> Result<Vec<probe::Saw>, tokio::time::error::Elapsed> {
        tokio::time::timeout(Duration::from_secs(5), async {
            let mut seen = Vec::new();
            while let Some(saw) = rx.recv().await {
                seen.push(saw);
            }
            seen
        })
        .await
    }

    /// The buffer is the observable rather than `RELEASES` or the planner:
    /// it belongs to this one actor, so a test running beside this one cannot
    /// move it.
    #[tokio::test]
    async fn second_terminate_is_ignored() {
        let mut process = idle_process();

        process
            .handle(Message::Terminate {
                result: Ok("first".to_string()),
                origin: Origin::Run,
            })
            .await;
        process
            .handle(Message::Terminate {
                result: Err("second".to_string()),
                origin: Origin::Owner,
            })
            .await;

        let delivered: Vec<_> = process
            .output_buffer
            .iter()
            .map(|event| (event.name(), event.value().to_string()))
            .collect();
        assert_eq!(delivered, vec![(ProcessEvent::RETURN, "first".to_string())]);
    }

    /// The successor is a probe and the recorded link is set by hand:
    /// `restart` itself cannot run here, because the process it spawns dies
    /// in `linker::instantiate` with no program loader. So this covers the
    /// relay decision in `Process::terminate`, not the restart that records
    /// the successor.
    #[tokio::test]
    async fn terminate_after_restart_is_relayed_to_the_successor() {
        let (successor, mut saw) = probe::register();
        let mut process = idle_process();
        process.terminated = true;
        process.successor = Some(successor);

        process
            .handle(Message::Terminate {
                result: Err("session closed".to_string()),
                origin: Origin::Owner,
            })
            .await;

        // The probe unregisters on Terminate, so the channel closing is the
        // proof that exactly one relay arrived.
        assert_eq!(
            drained(&mut saw).await.expect("successor saw nothing"),
            vec![probe::Saw::Terminated]
        );
    }

    #[tokio::test]
    async fn terminate_after_two_restarts_reaches_the_latest_successor() {
        let (latest, mut saw) = probe::register();
        let mut process = idle_process();
        let client_pid = process.client_pid;
        process.terminated = true;
        process.successor = Some(Uuid::new_v4());
        RESTART_ALIAS.write().unwrap().insert(client_pid, latest);

        process
            .handle(Message::Terminate {
                result: Err("session closed".to_string()),
                origin: Origin::Owner,
            })
            .await;

        let seen = drained(&mut saw).await;
        RESTART_ALIAS.write().unwrap().remove(&client_pid);
        SERVICES.remove(&latest);
        assert_eq!(
            seen.expect("latest successor saw nothing"),
            vec![probe::Saw::Terminated]
        );
    }

    /// The blocker this rule closes: the planner parks a restart request and
    /// only then wakes the victim, so an owner's Terminate can land in that
    /// window. Honouring the request there spawns a replacement outside the
    /// closing session, which nothing would ever end.
    ///
    /// The request is parked through `declare_restartable` + `request_restart`
    /// rather than by writing the static, so the test fails if that path
    /// changes shape.
    #[tokio::test]
    async fn owner_terminate_does_not_restart() {
        let mut process = idle_process();
        let pid = process.process_id;
        declare_restartable(pid);
        // Held, not asserted here: `declare_restartable` has already written a
        // global, so failing before the cleanup below would leave it behind.
        let parked = request_restart(pid);

        process
            .handle(Message::Terminate {
                result: Err("session closed".to_string()),
                origin: Origin::Owner,
            })
            .await;

        let alias = RESTART_ALIAS.read().unwrap().get(&pid).copied();
        let cleared = !is_restartable(pid) && !restart_requested(pid);
        let delivered: Vec<_> = process
            .output_buffer
            .iter()
            .map(|event| (event.name(), event.value().to_string()))
            .collect();
        // Global maps: this test's entries go before any assertion, so a
        // failure cannot leave them behind for the rest of the binary.
        if let Some(successor) = process.successor {
            SERVICES.remove(&successor);
            RESTART_ALIAS.write().unwrap().remove(&pid);
        }
        forget_restart_state(pid);

        assert!(parked, "a declared process must park");
        assert_eq!(process.successor, None, "no replacement may be spawned");
        assert_eq!(alias, None, "no alias may be published for the client pid");
        assert!(cleared, "the parked request must be forgotten either way");
        assert_eq!(
            delivered,
            vec![(ProcessEvent::ERROR, "session closed".to_string())]
        );
    }

    /// A Terminate for a retired pid reaches the process the alias names.
    /// Delivered by the first resolve, since the alias is published before
    /// the call: the re-resolve after a failed send needs the alias to change
    /// mid-call, and no test task can change it there — the two resolves have
    /// only a synchronous failed send between them.
    #[tokio::test]
    async fn terminate_follows_the_restart_alias() {
        let (live, mut saw) = probe::register();
        // Never registered, so only the alias can lead anywhere.
        let retired = Uuid::new_v4();
        RESTART_ALIAS.write().unwrap().insert(retired, live);

        terminate(retired, Err("session closed".to_string()));

        let seen = drained(&mut saw).await;
        // Global map: cleared before the assertion so a failure cannot leave
        // the entry behind for another test in this binary.
        RESTART_ALIAS.write().unwrap().remove(&retired);
        assert_eq!(
            seen.expect("aliased process saw nothing"),
            vec![probe::Saw::Terminated]
        );
    }
}
