//! Process - Per-instance lifecycle management
//!
//! Each Process is a ServiceMap actor that manages a single WASM instance.
//! Processes are registered in a global registry and receive messages via
//! Direct Addressing.

use std::collections::VecDeque;
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{anyhow, Result};
use uuid::Uuid;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::linker;
use crate::program::ProgramName;
use crate::server::{self, ClientId};
use crate::service::{ServiceMap, ServiceHandler};
use crate::workflow::WorkflowId;

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
            Self::Stdout(v) | Self::Stderr(v) | Self::Message(v)
            | Self::Return(v) | Self::Error(v) => v,
        }
    }

    /// Consume into payload string.
    pub fn into_value(self) -> String {
        match self {
            Self::Stdout(v) | Self::Stderr(v) | Self::Message(v)
            | Self::Return(v) | Self::Error(v) => v,
        }
    }
}

// =============================================================================
// Process Registry
// =============================================================================

pub type ProcessId = Uuid;

/// Global registry mapping ProcessId to process actors.
static SERVICES: LazyLock<ServiceMap<ProcessId, Message>> =
    LazyLock::new(ServiceMap::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawn a new process and register it in the global registry.
pub fn spawn(
    username: String,
    program_name: ProgramName,
    input: String,
    client_id: Option<ClientId>,
    capture_outputs: bool,
    result_tx: Option<oneshot::Sender<Result<String, String>>>,
    workflow_id: Option<WorkflowId>,
) -> Result<ProcessId> {
    let process = Process::new(username, program_name, input, client_id, capture_outputs, result_tx, workflow_id);
    let id = process.process_id;

    SERVICES.spawn(id, || process)?;

    Ok(id)
}

/// Attach a client to a process.
pub async fn attach(process_id: ProcessId, client_id: ClientId) -> Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&process_id, Message::AttachClient { client_id, response: tx })?;
    rx.await?
}

/// Detach the current client from a process (fire-and-forget).
pub fn detach(process_id: ProcessId) {
    let _ = SERVICES.send(&process_id, Message::DetachClient);
}

/// Terminate a process (fire-and-forget).
pub fn terminate(process_id: ProcessId, result: Result<String, String>) {
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
    Terminate {
        result: Result<String, String>,
    },

    /// Stdout output from the WASM instance
    Stdout {
        content: String,
    },
    /// Query the process username
    GetUsername {
        response: oneshot::Sender<Result<String>>,
    },
    /// Stderr output from the WASM instance
    Stderr {
        content: String,
    },
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
    /// Optional link to the workflow that spawned this process.
    workflow_id: Option<WorkflowId>,
}

impl Process {
    /// Creates a new Process, generating a UUID, and spawns its WASM execution task.
    fn new(
        username: String,
        program: ProgramName,
        input: String,
        client_id: Option<ClientId>,
        capture_outputs: bool,
        result_tx: Option<oneshot::Sender<Result<String, String>>>,
        workflow_id: Option<WorkflowId>,
    ) -> Self {
        let process_id = Uuid::new_v4();

        let handle = tokio::spawn(Self::run(
            process_id,
            username.clone(),
            program.clone(),
            input.clone(),
            capture_outputs,
            result_tx,
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
            workflow_id,
        }
    }

    /// Deliver an event to the attached client and/or the parent workflow.
    fn deliver_event(&mut self, event: ProcessEvent) {
        // Forward to parent workflow (if any)
        if let Some(wf_id) = self.workflow_id {
            let _ = crate::workflow::forward_event(wf_id, self.process_id, event.clone());
        }

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
        let Some(client_id) = self.client_id else { return };
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
        result_tx: Option<oneshot::Sender<Result<String, String>>>,
    ) {

        let result: Result<String, String> = async {
            let (mut store, instance) = linker::instantiate(process_id, username, &program, capture_outputs)
                .await
                .map_err(|e| e.to_string())?;

            let run_interface = format!("pie:{}/run", program.name);

            let (_, run_export) = instance
                .get_export(&mut store, None, &run_interface)
                .ok_or_else(|| "No 'run' interface found".to_string())?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| "No 'run' function found".to_string())?;

            let run_func = instance
                .get_typed_func::<(&str,), (Result<String, String>,)>(&mut store, &run_func_export)
                .map_err(|e| format!("Failed to get 'run' function: {e}"))?;

            match run_func.call_async(&mut store, (&input,)).await {
                Ok((Ok(output),)) => Ok(output),
                Ok((Err(runtime_err),)) => Err(runtime_err),
                Err(call_err) => Err(format!("Call error: {call_err}")),
            }
        }.await;

        if let Err(ref err) = result {
            tracing::info!("Process {process_id} failed: {err}");
        }

        // Fire result channel if a parent is waiting
        if let Some(tx) = result_tx {
            let _ = tx.send(result.clone());
        }

        terminate(process_id, result);
    }

    /// Abort the WASM execution task, notify any attached client, and unregister.
    fn terminate(&mut self, result: Result<String, String>) {

        self.handle.abort();

        // Notify attached client / workflow
        match result {
            Ok(output) => self.deliver_event(ProcessEvent::Return(output)),
            Err(msg) => self.deliver_event(ProcessEvent::Error(msg)),
        }

        SERVICES.remove(&self.process_id);
    }
}

impl ServiceHandler for Process {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::AttachClient { client_id, response } => {
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
