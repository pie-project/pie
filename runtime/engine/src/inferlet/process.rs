//! Inferlet process lifecycle management.
//!
//! Each Process is a ServiceMap actor that manages a single WASM instance.
//! Processes are registered in a global registry and receive messages via
//! Direct Addressing. Process-owned KV quiesce/suspend/restore lives in the
//! `preemption` submodule; WIT host modules only delegate into it.

mod ctx;
mod output;
pub(crate) mod preemption;
mod residency;

pub(crate) use ctx::OutputMode;
pub use ctx::ProcessCtx;
pub(crate) use residency::ProcessResidency;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
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
    let sem = max_concurrent
        .filter(|&n| n > 0)
        .map(|n| Arc::new(Semaphore::new(n)));
    ADMISSION
        .set(sem)
        .expect("admission controller already initialized");
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
    let id = Uuid::new_v4();
    if let Some(orchestrator) = crate::store::reclaim::contention() {
        orchestrator.register(id);
    }
    let process = Process::new(
        id,
        username,
        program_name,
        input,
        client_id,
        capture_outputs,
        result_tx,
    );
    if let Err(error) = SERVICES.spawn(id, || process) {
        if let Some(orchestrator) = crate::store::reclaim::contention() {
            orchestrator.unregister(id);
        }
        return Err(error);
    }

    Ok(id)
}

/// Attach a client to a process.
pub async fn attach(process_id: ProcessId, client_id: ClientId) -> Result<()> {
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
    let _ = SERVICES.send(&process_id, Message::DetachClient);
}

/// Terminate a process (fire-and-forget).
pub fn terminate(process_id: ProcessId, result: Result<String, String>) {
    // M-A1 Stage 2: broadcast a wait-for-all pipeline `Leave` so the scheduler
    // drops this pid from its wave wait-set immediately (rather than after the
    // miss-counter backstop). No-op unless a waitall scheduler is registered.
    crate::scheduler::worker::notify_pipeline_leave(
        process_id,
        crate::scheduler::worker::LeaveKind::Terminate,
    );
    let _ = SERVICES.send(&process_id, Message::Terminate { result });
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
        username: String,
        program: ProgramName,
        input: String,
        client_id: Option<ClientId>,
        capture_outputs: bool,
        result_tx: Option<oneshot::Sender<Result<String, String>>>,
    ) -> Self {
        let result_tx: SharedResultTx = Arc::new(Mutex::new(result_tx));

        let handle = tokio::spawn(Self::run(
            process_id,
            username.clone(),
            program.clone(),
            input.clone(),
            capture_outputs,
            result_tx.clone(),
        ));

        Process {
            process_id,
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
            if server::send_event(client_id, self.process_id, &event).is_err() {
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
            if server::send_event(client_id, self.process_id, &event).is_err() {
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
        // Admission control: wait for a permit before instantiating.
        // The permit is held for the entire WASM execution lifetime
        // and auto-released on completion, error, or task abort.
        let admission_start = Instant::now();
        let _permit = match ADMISSION.get().and_then(|s| s.as_ref()) {
            Some(sem) => Some(sem.acquire().await.expect("admission semaphore closed")),
            None => None,
        };
        let admission_wait_us = duration_us(admission_start.elapsed());

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

            // (KV admission via the context actor removed — Phase 5; physical
            // admission is now the unified arena's concern.)

            // Every inferlet now exports the same stock `pie:inferlet/run`
            // (WIT-refactor Phase 2 — the per-package synthesized export is
            // gone). Program identity comes from `program.name` metadata, not
            // the export interface name. The name is version-qualified: an
            // unversioned lookup does NOT match a versioned component export in
            // wasmtime's semver-aware name map, so this must track the
            // `pie:inferlet@<version>` package version declared in world.wit.
            let run_interface = "pie:inferlet/run@0.2.0";

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
            match run_func.call_async(&mut store, (&input,)).await {
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
            }
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
        // terminate hasn't already claimed it).
        if let Some(tx) = result_tx.lock().unwrap().take() {
            let _ = tx.send(result.clone());
        }

        terminate(process_id, result);
    }

    /// Abort the WASM execution task, notify any attached client, and unregister.
    fn terminate(&mut self, result: Result<String, String>) {
        self.handle.abort();

        // Deliver `result` to any parent waiting on the launch handle, if the
        // run loop didn't already send it (e.g., external Terminate fires
        // before the WASM task finished). First taker wins.
        if let Some(tx) = self.result_tx.lock().unwrap().take() {
            let _ = tx.send(result.clone());
        }

        // Notify attached client / workflow
        match result {
            Ok(output) => self.deliver_event(ProcessEvent::Return(output)),
            Err(msg) => self.deliver_event(ProcessEvent::Error(msg)),
        }

        let _ = server::inbox::clear(self.process_id.to_string());
        SERVICES.remove(&self.process_id);

        // M-A1 wait-for-all: drop this pid from the scheduler's wave wait-set as
        // an explicit step of the SINGLE exit funnel — so NATURAL completion (not
        // only the external-terminate free fn) promptly stops holding the wave,
        // well before the miss-limit backstop would demote it. Idempotent with the
        // free fn's early `Leave{Terminate}`; no-op unless a waitall scheduler is
        // registered.
        crate::scheduler::worker::notify_pipeline_leave(
            self.process_id,
            crate::scheduler::worker::LeaveKind::Terminate,
        );

        // Task-B contention: unregister from the preempt/restore orchestrator
        // (purges its waiters/restore-queue entries, wakes a parked task for
        // teardown, and drains — the exiting process's KV frees follow via the
        // WS-drop hook). Single exit funnel: covers natural completion AND
        // external terminate. No-op unless PIE_KV_CONTENTION=preempt.
        if let Some(o) = crate::store::reclaim::contention() {
            o.unregister(self.process_id);
        }
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
