mod ctx;
pub(crate) mod gate;
mod ordered;
mod output;
pub(crate) mod residency;
pub(crate) mod teardown;

pub(crate) use ctx::OutputMode;
pub use ctx::ProcessCtx;
pub(crate) use residency::ProcessResidency;

use crate::rt::Instant;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};
use std::time::Duration;

use crate::rt::JoinHandle;
use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use uuid::Uuid;

type SharedResultTx = Arc<Mutex<Option<oneshot::Sender<Result<String, String>>>>>;

use crate::policy::{Arrival, By, Outcome};
use crate::server::{self, ClientId};
use crate::service::{ServiceHandler, ServiceMap};

use self::ordered::{OrderedPermit, OrderedSemaphore};
use super::linker;
use super::program::ProgramName;

const RUN_INTERFACE: &str = "pie:inferlet/run@0.3.0";

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

#[derive(Debug, Clone)]
pub enum ProcessEvent {
    Stdout(String),
    Stderr(String),
    Message(String),
    Return(String),
    Error(String),
}

impl ProcessEvent {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Stdout(_) => "stdout",
            Self::Stderr(_) => "stderr",
            Self::Message(_) => "message",
            Self::Return(_) => "return",
            Self::Error(_) => "error",
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
}

pub type ProcessId = Uuid;

static SERVICES: LazyLock<ServiceMap<ProcessId, Message>> = LazyLock::new(ServiceMap::new);

static ADMISSION: OnceLock<Option<Arc<OrderedSemaphore>>> = OnceLock::new();
static BIND_ADMISSION: OnceLock<Option<Arc<OrderedSemaphore>>> = OnceLock::new();
static PREWARM_ADMISSION: OnceLock<Option<Arc<OrderedSemaphore>>> = OnceLock::new();
static EXECUTION_SLOT_CAPACITY: OnceLock<Option<usize>> = OnceLock::new();
/// A process's bind permit, then its execution seat; a process the schedule
/// pauses gives both back.
#[derive(Default)]
pub(crate) struct Seat {
    bind: Option<OrderedPermit>,
    exec: Option<OrderedPermit>,
}

impl Drop for Seat {
    fn drop(&mut self) {
        release_bind_permit(self.bind.take());
    }
}

static SEATS: LazyLock<Mutex<HashMap<ProcessId, Seat>>> = LazyLock::new(Default::default);

fn seats_held() -> std::sync::MutexGuard<'static, HashMap<ProcessId, Seat>> {
    SEATS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn seats_capped() -> bool {
    ADMISSION.get().is_some_and(Option::is_some)
        || BIND_ADMISSION.get().is_some_and(Option::is_some)
}

fn take_permits(pid: ProcessId) -> Option<Seat> {
    seats_held().remove(&pid)
}

pub(crate) fn vacate_seat(pid: ProcessId) -> bool {
    let Some(seat) = take_permits(pid) else {
        return false;
    };
    if seat.exec.is_some() {
        crate::scheduler::worker::notify_execution_slot_released(pid);
    }
    true
}

pub(crate) async fn take_seat(pid: ProcessId) -> Seat {
    let mut seat = Seat::default();
    if let Some(semaphore) = BIND_ADMISSION.get().and_then(Option::as_ref) {
        seat.bind = Some(Arc::clone(semaphore).acquire_owned(pid).await);
    }
    if let Some(semaphore) = ADMISSION.get().and_then(Option::as_ref) {
        seat.exec = Some(Arc::clone(semaphore).acquire_owned(pid).await);
    }
    seat
}

pub(crate) fn reseat(pid: ProcessId, seat: Seat) {
    let seated = seat.exec.is_some();
    let replaced = seats_held().insert(pid, seat);
    drop(replaced);
    if seated {
        crate::scheduler::worker::notify_execution_slot_consumed(pid);
    }
}

pub(crate) fn execution_slot_capacity() -> Option<usize> {
    EXECUTION_SLOT_CAPACITY.get().copied().flatten()
}

pub fn live_count() -> usize {
    SERVICES.len()
}

pub fn seats() -> (usize, usize) {
    let live = live_count();
    let running = match (
        ADMISSION.get().and_then(Option::as_ref),
        execution_slot_capacity(),
    ) {
        (Some(semaphore), Some(capacity)) => capacity.saturating_sub(semaphore.available_permits()),
        _ => live,
    }
    .min(live);
    (running, live - running)
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
    // Only the first gate, prewarm, holds below the floor.
    let sem = limit.map(|n| OrderedSemaphore::new(n, false));
    let prewarm = Some(OrderedSemaphore::new(
        limit.unwrap_or(UNCAPPED_PREWARM_PROCESSES),
        true,
    ));
    const STAGED_COHORTS: usize = 1;
    let bind_ahead = limit.map(|n| OrderedSemaphore::new(n, false));
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
static HELD_BIND_PERMITS: Mutex<Vec<OrderedPermit>> = Mutex::new(Vec::new());

pub(crate) fn release_bind_permit(permit: Option<OrderedPermit>) {
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

#[cfg_attr(
    target_arch = "wasm32",
    allow(dead_code, reason = "no policy runs in the browser")
)]
pub(crate) fn regrant_admission() {
    for gate in [&PREWARM_ADMISSION, &BIND_ADMISSION, &ADMISSION] {
        if let Some(Some(semaphore)) = gate.get() {
            semaphore.regrant();
        }
    }
}

pub(crate) async fn ensure_bind_admitted(ctx: &mut ProcessCtx) {
    if ctx.bind_admitted() {
        return;
    }
    let permit = match BIND_ADMISSION.get().and_then(|value| value.as_ref()) {
        Some(semaphore) => Some(Arc::clone(semaphore).acquire_owned(ctx.id()).await),
        None => None,
    };
    ctx.release_prewarm_permit();
    if let Some(permit) = permit {
        seats_held().entry(ctx.id()).or_default().bind = Some(permit);
    }
    ctx.admit_bind();
}

pub(crate) async fn ensure_execution_admitted(ctx: &mut ProcessCtx) {
    ensure_bind_admitted(ctx).await;
    if ctx.execution_admitted() {
        return;
    }
    let started = Instant::now();
    let seated = match ADMISSION.get().and_then(|value| value.as_ref()) {
        Some(semaphore) => {
            let _queued = AdmissionQueued::enter(ctx.id());
            let permit = Arc::clone(semaphore).acquire_owned(ctx.id()).await;
            if semaphore.available_permits() == 0 {
                open_staged_bind_pool();
            }
            seats_held().entry(ctx.id()).or_default().exec = Some(permit);
            crate::scheduler::worker::notify_execution_slot_consumed(ctx.id());
            true
        }
        None => false,
    };
    ctx.admit_execution(seated, duration_us(started.elapsed()));
    if let Some(planner) = crate::planner::planner() {
        planner.note_admitted(ctx.id());
    }
    crate::policy::report(crate::policy::Event::Started(ctx.id()));
}

#[allow(
    clippy::too_many_arguments,
    reason = "one spawn request, as its callers have it"
)]
pub fn spawn(
    username: String,
    program_name: ProgramName,
    input: String,
    client_id: Option<ClientId>,
    capture_outputs: bool,
    result_tx: Option<oneshot::Sender<Result<String, String>>>,
    arrival: Arrival,
    executors: Vec<String>,
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
        arrival,
        executors,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "one spawn request, as its callers have it"
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
    arrival: Arrival,
    executors: Vec<String>,
) -> Result<ProcessId> {
    let id = Uuid::new_v4();
    // The workers the cluster names to run its passes, best first: the
    // offload's alone, before the process can run one.
    crate::offload::executors::set(id, executors.clone());
    crate::policy::schedule::alias(id, inherit_client_pid.unwrap_or(id));
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
        arrival,
        executors,
    );
    if let Err(error) = SERVICES.spawn(id, || process) {
        if let Some(planner) = crate::planner::planner() {
            planner.unregister(id);
        }
        crate::offload::executors::forget(id);
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

/// Ends a process on its owner's behalf, never re-running it. A send that
/// misses because a restart retired the resolved id follows the new alias.
pub fn terminate(process_id: ProcessId, result: Result<String, String>) {
    terminate_by(process_id, result, By::Client);
}

pub fn terminate_by(process_id: ProcessId, result: Result<String, String>, by: By) {
    let mut target = resolve(process_id);
    while !send_terminate(target, result.clone(), false, by) {
        let next = resolve(process_id);
        if next == target {
            return;
        }
        target = next;
    }
}

fn send_terminate(
    process_id: ProcessId,
    result: Result<String, String>,
    from_run: bool,
    by: By,
) -> bool {
    let sent = SERVICES
        .send(
            &process_id,
            Message::Terminate {
                result,
                from_run,
                by,
            },
        )
        .is_ok();
    if sent {
        crate::scheduler::worker::post_process_terminate(process_id);
    }
    sent
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

enum Message {
    AttachClient {
        client_id: ClientId,
        response: oneshot::Sender<Result<()>>,
    },
    DetachClient,
    Terminate {
        result: Result<String, String>,
        /// Only the run task's own end may be replaced by a parked restart.
        from_run: bool,
        by: By,
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
    arrival: Arrival,
    /// Kept for a restart: the new process runs its passes where this one did.
    executors: Vec<String>,
}

impl Process {
    #[allow(
        clippy::too_many_arguments,
        reason = "`spawn_inner`'s arguments, with both process ids known"
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
        arrival: Arrival,
        executors: Vec<String>,
    ) -> Self {
        let task = Self::run(
            process_id,
            username.clone(),
            program.clone(),
            input.clone(),
            capture_outputs,
            result_tx.clone(),
            arrival.clone(),
        );
        let handle = crate::rt::spawn(task);

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
            arrival,
            executors,
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
        arrival: Arrival,
    ) {
        // A process arrives, as a policy sees it, before it waits: every gate
        // below orders by its key.
        let props = arrival.props.clone();
        crate::policy::report(crate::policy::Event::Arrived(process_id, arrival));
        // Held ahead of every admission permit: a preempted process coming
        // back must not queue behind permits the held-back processes own.
        if let Some(planner) = crate::planner::planner() {
            planner.admit(process_id).await;
        }
        let prewarm_permit = match PREWARM_ADMISSION.get().and_then(|s| s.as_ref()) {
            Some(sem) => Some(Arc::clone(sem).acquire_owned(process_id).await),
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
            store.data_mut().set_props(props);

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

            // A script's component is its language component; the script
            // itself rides in the launch input.
            let input = match store.data().script() {
                Some(script) => script.envelope(&input),
                None => input,
            };

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
                "[process t_us={} pid={}] guest finished ok={} restart_requested={} \
                 admission_wait_us={admission_wait_us} instantiate_us={instantiate_us} \
                 wasm_run_us={wasm_run_us}",
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

        send_terminate(process_id, result, true, By::Host);
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
            self.arrival.clone(),
            self.executors.clone(),
        );
        match spawned {
            Ok(new_id) => {
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

    fn terminate(&mut self, result: Result<String, String>, from_run: bool, by: By) {
        // Queued here before a restart handed the client's pid over.
        let current = resolve(self.client_pid);
        if current != self.process_id {
            terminate_by(self.client_pid, result, by);
            return;
        }
        self.handle.abort();

        let restarted = from_run && restart_requested(self.process_id) && self.restart();
        forget_restart_state(self.process_id);

        if !restarted {
            let outcome = match (&result, from_run, by) {
                (Ok(_), true, _) => Outcome::Completed,
                (Err(_), true, _) | (_, false, By::Host) => Outcome::Failed,
                (_, false, By::Client | By::Policy) => Outcome::Cancelled,
            };
            // Before the client hears of the exit, so whoever it tells
            // can read the process's last counts.
            crate::policy::tokens::exit(self.client_pid);
            crate::policy::report(crate::policy::Event::Exited(self.client_pid, outcome));
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
        crate::policy::schedule::forget(self.process_id);

        if !restarted {
            RESTART_ALIAS.write().unwrap().remove(&self.client_pid);
        }
        if let Some(planner) = crate::planner::planner() {
            planner.unregister(self.process_id);
        }
        residency::unregister_residency(self.process_id);
        crate::offload::executors::forget(self.process_id);
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

            Message::Terminate {
                result,
                from_run,
                by,
            } => {
                self.terminate(result, from_run, by);
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

/// A stand-in registered under a process id, so a test can see what the
/// registry delivers to a process without instantiating a guest.
#[cfg(test)]
pub(crate) mod probe {
    use super::*;

    #[derive(Debug)]
    pub(crate) enum Seen {
        Terminate(Result<String, String>),
        Detach,
        Other,
    }

    struct Probe {
        process_id: ProcessId,
        report: Option<oneshot::Sender<Seen>>,
    }

    impl ServiceHandler for Probe {
        type Message = Message;

        async fn handle(&mut self, msg: Message) {
            let seen = match msg {
                Message::Terminate { result, .. } => {
                    SERVICES.remove(&self.process_id);
                    Seen::Terminate(result)
                }
                Message::DetachClient => Seen::Detach,
                _ => Seen::Other,
            };
            if let Some(report) = self.report.take() {
                let _ = report.send(seen);
            }
        }
    }

    pub(crate) fn spawn(process_id: ProcessId) -> oneshot::Receiver<Seen> {
        let (report, seen) = oneshot::channel();
        SERVICES
            .spawn(process_id, || Probe {
                process_id,
                report: Some(report),
            })
            .expect("a fresh process id");
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idle(process_id: ProcessId, client_pid: ProcessId, result_tx: SharedResultTx) -> Process {
        Process {
            process_id,
            client_pid,
            username: "test".to_string(),
            program: ProgramName {
                name: "test".to_string(),
                version: "0.0.0".to_string(),
            },
            input: String::new(),
            start_time: Instant::now(),
            handle: crate::rt::spawn(std::future::pending::<()>()),
            client_id: None,
            capture_outputs: true,
            output_buffer: VecDeque::new(),
            result_tx,
            arrival: Arrival::default(),
            executors: Vec::new(),
        }
    }

    #[tokio::test]
    async fn a_terminate_queued_behind_restarts_reaches_the_latest_process() {
        let (client_pid, latest) = (ProcessId::new_v4(), ProcessId::new_v4());
        let seen = probe::spawn(latest);
        RESTART_ALIAS.write().unwrap().insert(client_pid, latest);
        let mut superseded = idle(ProcessId::new_v4(), client_pid, Default::default());

        superseded
            .handle(Message::Terminate {
                result: Err("closed".to_string()),
                from_run: false,
                by: By::Client,
            })
            .await;

        let seen = tokio::time::timeout(Duration::from_secs(5), seen).await;
        RESTART_ALIAS.write().unwrap().remove(&client_pid);
        assert!(matches!(seen, Ok(Ok(probe::Seen::Terminate(Err(_))))));
        assert!(superseded.output_buffer.is_empty());
    }

    #[tokio::test]
    async fn an_owner_terminate_is_not_replaced_by_a_parked_restart() {
        let pid = ProcessId::new_v4();
        let (tx, rx) = oneshot::channel();
        let process = idle(pid, pid, Arc::new(Mutex::new(Some(tx))));
        SERVICES.spawn(pid, || process).unwrap();
        declare_restartable(pid);
        assert!(request_restart(pid));

        terminate(pid, Err("closed".to_string()));

        let result = tokio::time::timeout(Duration::from_secs(5), rx).await;
        assert_eq!(result.unwrap().unwrap(), Err("closed".to_string()));
        assert!(!RESTART_ALIAS.read().unwrap().contains_key(&pid));
    }
}
