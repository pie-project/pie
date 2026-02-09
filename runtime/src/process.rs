//! Process - Per-instance lifecycle management
//!
//! Each Process is a ServiceMap actor that manages a single WASM instance.
//! Processes are registered in a global registry and receive messages via
//! Direct Addressing.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{anyhow, Result};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::linker;
use crate::program::ProgramName;
use crate::runtime::instance::InstanceId;
use crate::runtime::TerminationCause;
use crate::server::{self, ClientId};
use crate::service::{ServiceMap, ServiceHandler};

// =============================================================================
// Process Registry
// =============================================================================

type ProcessId = usize;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Global registry mapping ProcessId to process actors.
static SERVICES: LazyLock<ServiceMap<ProcessId, Message>> =
    LazyLock::new(ServiceMap::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawn a new process and register it in the global registry.
pub async fn spawn(
    username: String,
    program_name: ProgramName,
    arguments: Vec<String>,
    client_id: Option<ClientId>,
    parent_id: Option<ProcessId>,
    _capture_outputs: bool,
) -> Result<ProcessId> {
    let process = Process::new(username, program_name, arguments, client_id);
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
}

// =============================================================================
// Process
// =============================================================================

/// Actor managing a single WASM instance lifecycle.
struct Process {
    process_id: ProcessId,
    instance_id: InstanceId,
    username: String,
    program: ProgramName,
    arguments: Vec<String>,
    start_time: Instant,
    handle: JoinHandle<()>,
    client_id: Option<ClientId>,
    children: Vec<ProcessId>,
}

impl Process {
    /// Creates a new Process and spawns its WASM execution task.
    fn new(
        username: String,
        program: ProgramName,
        arguments: Vec<String>,
        client_id: Option<ClientId>,
    ) -> Self {
        let process_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let instance_id = Uuid::new_v4();

        let handle = tokio::spawn(Self::run(
            process_id,
            instance_id,
            username.clone(),
            program.clone(),
            arguments.clone(),
        ));

        Process {
            process_id,
            instance_id,
            username,
            program,
            arguments,
            start_time: Instant::now(),
            handle,
            client_id,
            children: Vec::new(),
        }
    }

    /// Runs the WASM component: instantiate, find the `run` export, and call it.
    async fn run(
        process_id: ProcessId,
        instance_id: InstanceId,
        username: String,
        program: ProgramName,
        arguments: Vec<String>,
    ) {
        let exception = match Self::run_inner(instance_id, username, &program, &arguments).await {
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
        instance_id: InstanceId,
        username: String,
        program: &ProgramName,
        arguments: &[String],
    ) -> Result<String> {
        let (mut store, instance) = linker::instantiate(instance_id, username, program).await?;

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
    fn cleanup(&mut self, cause: TerminationCause) {
        self.handle.abort();

        // Cascade: terminate all children
        for child_id in self.children.drain(..) {
            terminate(child_id, None);
        }

        // Notify attached client
        if let Some(client_id) = self.client_id.take() {
            let instance_id = self.instance_id;
            tokio::spawn(async move {
                server::sessions::terminate(client_id, instance_id, cause).ok();
                server::unregister_instance(instance_id).ok();
            });
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
                let _ = response.send(Ok(()));
            }

            Message::DetachClient => {
                self.client_id = None;
            }

            Message::Terminate { exception } => {
                let cause = match exception {
                    Some(msg) => TerminationCause::Exception(msg),
                    None => TerminationCause::Signal,
                };
                self.cleanup(cause);
            }

            Message::ExecutionFinished { exception } => {
                let cause = match exception {
                    Some(msg) => TerminationCause::Exception(msg),
                    None => TerminationCause::Normal(String::new()),
                };
                self.cleanup(cause);
            }

            Message::AddChild { child_id } => {
                self.children.push(child_id);
            }
        }
    }
}
