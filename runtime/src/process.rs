//! Process - Per-instance lifecycle management
//!
//! Each Process is a ServiceMap actor that manages a single WASM instance.
//! Processes are registered in a global registry and receive messages via
//! Direct Addressing.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::linker;
use crate::program::ProgramName;
use crate::server::{self, ClientId};
use crate::service::{ServiceMap, ServiceHandler};

// =============================================================================
// Process Registry
// =============================================================================

type ProcessId = usize;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

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
    arguments: Vec<String>,
    client_id: Option<ClientId>,
    parent_id: Option<ProcessId>,
    capture_outputs: bool,
) -> Result<ProcessId> {
    let process = Process::new(username, program_name, arguments, client_id, parent_id, capture_outputs);
    let id = process.process_id;
    SERVICES.spawn(id, || process)?;

    // Register as child of parent so it terminates with the parent
    if let Some(parent_id) = parent_id {
        add_child(parent_id, id);
    }

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
pub fn terminate(process_id: ProcessId, exception: Option<String>) {
    let _ = SERVICES.send(&process_id, Message::Terminate { exception });
}

/// Send stdout output from a WASM instance to its process (fire-and-forget).
pub fn stdout(process_id: ProcessId, content: String) {
    let _ = SERVICES.send(&process_id, Message::Stdout { content });
}

/// Send stderr output from a WASM instance to its process (fire-and-forget).
pub fn stderr(process_id: ProcessId, content: String) {
    let _ = SERVICES.send(&process_id, Message::Stderr { content });
}

/// Register a child process under a parent. The child will be terminated
/// when the parent terminates or finishes.
fn add_child(parent_id: ProcessId, child_id: ProcessId) {
    let _ = SERVICES.send(&parent_id, Message::AddChild { child_id });
}

/// List all registered process IDs.
pub fn list() -> Vec<ProcessId> {
    SERVICES.keys()
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
    /// Terminate this process
    Terminate {
        exception: Option<String>,
    },
    /// Internal: WASM execution has finished
    ExecutionFinished {
        exception: Option<String>,
    },
    /// Register a child process
    AddChild {
        child_id: ProcessId,
    },
    /// Stdout output from the WASM instance
    Stdout {
        content: String,
    },
    /// Stderr output from the WASM instance
    Stderr {
        content: String,
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
    parent_id: Option<ProcessId>,
    username: String,
    program: ProgramName,
    arguments: Vec<String>,
    start_time: Instant,
    handle: JoinHandle<()>,
    client_id: Option<ClientId>,
    children: Vec<ProcessId>,
    capture_outputs: bool,
    output_buffer: VecDeque<String>,
}

impl Process {
    /// Creates a new Process and spawns its WASM execution task.
    fn new(
        username: String,
        program: ProgramName,
        arguments: Vec<String>,
        client_id: Option<ClientId>,
        parent_id: Option<ProcessId>,
        capture_outputs: bool,
    ) -> Self {
        let process_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);

        let handle = tokio::spawn(Self::run(
            process_id,
            username.clone(),
            program.clone(),
            arguments.clone(),
            capture_outputs,
        ));

        Process {
            process_id,
            parent_id,
            username,
            program,
            arguments,
            start_time: Instant::now(),
            handle,
            client_id,
            children: Vec::new(),
            capture_outputs,
            output_buffer: VecDeque::new(),
        }
    }

    /// Deliver output to the attached client, or buffer it if capturing.
    fn deliver_output(&mut self, content: String) {
        if let Some(client_id) = self.client_id {
            if server::sessions::streaming_output(client_id, self.process_id, content.clone()).is_err() {
                // Client gone — detach and fall back to buffering
                self.client_id = None;
                self.buffer_output(content);
            }
        } else if self.capture_outputs {
            self.buffer_output(content);
        }
    }

    /// Push content into the ring buffer, evicting the oldest entry if full.
    fn buffer_output(&mut self, content: String) {
        if self.output_buffer.len() >= OUTPUT_BUFFER_CAP {
            self.output_buffer.pop_front();
        }
        self.output_buffer.push_back(content);
    }

    /// Runs the WASM component: instantiate, find the `run` export, and call it.
    async fn run(
        process_id: ProcessId,
        username: String,
        program: ProgramName,
        arguments: Vec<String>,
        capture_outputs: bool,
    ) {
        let exception = match Self::run_inner(process_id, username, &program, &arguments, capture_outputs).await {
            Ok(_output) => None,
            Err(err) => {
                tracing::info!("Process {process_id} failed: {err}");
                Some(err.to_string())
            }
        };

        let _ = SERVICES.send(&process_id, Message::ExecutionFinished { exception });
    }

    /// Inner execution logic, returns the run result.
    async fn run_inner(
        process_id: ProcessId,
        username: String,
        program: &ProgramName,
        arguments: &[String],
        capture_outputs: bool,
    ) -> Result<String> {
        let (mut store, instance) = linker::instantiate(process_id, username, program, capture_outputs).await?;

        let run_interface = format!("pie:{}/run", program.name);

        let (_, run_export) = instance
            .get_export(&mut store, None, &run_interface)
            .ok_or_else(|| anyhow!("No 'run' interface found"))?;

        let (_, run_func_export) = instance
            .get_export(&mut store, Some(&run_export), "run")
            .ok_or_else(|| anyhow!("No 'run' function found"))?;

        let run_func = instance
            .get_typed_func::<(&[String],), (Result<String, String>,)>(&mut store, &run_func_export)
            .map_err(|e| anyhow!("Failed to get 'run' function: {e}"))?;

        match run_func.call_async(&mut store, (arguments,)).await {
            Ok((Ok(output),)) => Ok(output),
            Ok((Err(runtime_err),)) => Err(anyhow!(runtime_err)),
            Err(call_err) => Err(anyhow!("Call error: {call_err}")),
        }
    }

    /// Abort the WASM execution task, notify any attached client, terminate
    /// children, and unregister.
    fn cleanup(&mut self, exception: Option<String>) {
        self.handle.abort();

        // Cascade: terminate all children
        for child_id in self.children.drain(..) {
            terminate(child_id, None);
        }

        // Notify attached client
        if let Some(client_id) = self.client_id.take() {
            let process_id = self.process_id;
            let cause = match exception {
                Some(msg) => crate::runtime::TerminationCause::Exception(msg),
                None => crate::runtime::TerminationCause::Normal(String::new()),
            };
            let _ = server::sessions::terminate(client_id, process_id, cause);
            let _ = server::unregister_instance(process_id);
        }

        SERVICES.remove(&self.process_id);
    }
}

impl ServiceHandler for Process {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::AttachClient { client_id, response } => {
                self.client_id = Some(client_id);

                // Flush buffered output to the newly attached client
                while let Some(buffered) = self.output_buffer.pop_front() {
                    if server::sessions::streaming_output(client_id, self.process_id, buffered.clone()).is_err() {
                        self.client_id = None;
                        self.output_buffer.push_front(buffered);
                        break;
                    }
                }

                let _ = response.send(Ok(()));
            }

            Message::DetachClient => {
                self.client_id = None;
            }

            Message::Terminate { exception } => {
                self.cleanup(exception);
            }

            Message::ExecutionFinished { exception } => {
                self.cleanup(exception);
            }

            Message::AddChild { child_id } => {
                self.children.push(child_id);
            }

            Message::Stdout { content } | Message::Stderr { content } => {
                self.deliver_output(content);
            }
        }
    }
}
