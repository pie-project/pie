//! Instance Actor - Per-instance lifecycle management
//!
//! This module provides the Instance actor for managing individual WASM instances.
//! Following the Session pattern, instances are registered in a global registry
//! for Direct Addressing, allowing messages to bypass the RuntimeActor.

use std::sync::LazyLock;
use std::time::Instant;

use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use wasmtime::component::{Component, Linker};
use wasmtime::{Engine, Store};
use std::sync::Arc;

use crate::service::{ServiceMap, ServiceHandler};
use crate::{program, server};

use super::instance::{InstanceId, InstanceState};
use super::output::{OutputDelivery, OutputDeliveryCtrl};
use super::{AttachInstanceResult, InstanceRunningState, TerminationCause};

// =============================================================================
// Instance Registry (Direct Addressing)
// =============================================================================

/// Global registry mapping InstanceId to instance actors.
static SERVICES: LazyLock<ServiceMap<InstanceId, Message>> =
    LazyLock::new(ServiceMap::new);


/// Remove an instance from the registry.
fn unregister(inst_id: InstanceId) {
    SERVICES.remove(&inst_id);
}

// =============================================================================
// Public API
// =============================================================================

/// Spawn a new instance and register it in the global registry.
pub async fn spawn(
    inst_id: InstanceId,
    username: String,
    program_name: String,
    arguments: Vec<String>,
    capture_outputs: bool,
    component: Component,
    engine: Engine,
    linker: Arc<Linker<InstanceState>>,
) -> anyhow::Result<InstanceId> {
    let instance = Instance::new(
        inst_id, username, program_name, arguments,
        capture_outputs, component, engine, linker,
    ).await?;
    SERVICES.spawn(inst_id, || instance)?;
    Ok(inst_id)
}

/// Attach to an instance.
pub async fn attach(inst_id: InstanceId) -> AttachInstanceResult {
    let (tx, rx) = oneshot::channel();
    if SERVICES.send(&inst_id, Message::Attach { response: tx }).is_err() {
        return AttachInstanceResult::InstanceNotFound;
    }
    rx.await.unwrap_or(AttachInstanceResult::InstanceNotFound)
}

/// Detach from an instance (fire-and-forget).
pub fn detach(inst_id: InstanceId) {
    let _ = SERVICES.send(&inst_id, Message::Detach);
}

/// Allow output for an instance (fire-and-forget).
pub fn allow_output(inst_id: InstanceId) {
    let _ = SERVICES.send(&inst_id, Message::AllowOutput);
}

/// Set output delivery mode for an instance (fire-and-forget).
pub fn set_output_delivery(inst_id: InstanceId, mode: OutputDelivery) {
    let _ = SERVICES.send(&inst_id, Message::SetOutputDelivery { mode });
}

/// Terminate an instance (fire-and-forget).
pub fn terminate(inst_id: InstanceId, notification_to_client: Option<TerminationCause>) {
    let _ = SERVICES.send(&inst_id, Message::Terminate { notification_to_client });
}

/// Get instance info for listing.
pub async fn get_info(inst_id: InstanceId) -> Option<InstanceInfo> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&inst_id, Message::GetInfo { response: tx }).ok()?;
    rx.await.ok()
}

/// List all registered instance IDs.
pub fn list_instance_ids() -> Vec<InstanceId> {
    SERVICES.keys()
}

// =============================================================================
// Messages
// =============================================================================

/// Messages that can be sent directly to an Instance.
#[derive(Debug)]
enum Message {
    /// Attach a client to this instance
    Attach {
        response: oneshot::Sender<AttachInstanceResult>,
    },
    /// Detach the current client from this instance
    Detach,
    /// Allow output to start flowing
    AllowOutput,
    /// Set output delivery mode
    SetOutputDelivery { mode: OutputDelivery },
    /// Terminate this instance
    Terminate {
        notification_to_client: Option<TerminationCause>,
    },
    /// Internal: WASM execution has finished
    ExecutionFinished { cause: TerminationCause },
    /// Get instance info for listing
    GetInfo {
        response: oneshot::Sender<InstanceInfo>,
    },
}

/// Minimal info for instance listing (returned to RuntimeActor)
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub username: String,
    pub program_name: String,
    pub arguments: Vec<String>,
    pub elapsed_secs: u64,
    pub running_state: InstanceRunningState,
}

// =============================================================================
// Instance
// =============================================================================

/// Actor managing a single WASM instance lifecycle.
struct Instance {
    inst_id: InstanceId,
    username: String,
    program_name: String,
    arguments: Vec<String>,
    start_time: Instant,
    output_delivery_ctrl: OutputDeliveryCtrl,
    running_state: InstanceRunningState,
    execution_handle: Option<JoinHandle<()>>,
}

impl Instance {
    /// Creates a new Instance, starting its WASM execution task.
    async fn new(
        inst_id: InstanceId,
        username: String,
        program_name: String,
        arguments: Vec<String>,
        capture_outputs: bool,
        component: Component,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
    ) -> anyhow::Result<Self> {
        // Create channels for synchronization
        let (output_ctrl_tx, output_ctrl_rx) = oneshot::channel();

        // Spawn WASM execution task
        let execution_handle = tokio::spawn(Self::run_wasm(
            inst_id,
            username.clone(),
            component,
            arguments.clone(),
            capture_outputs,
            engine,
            linker,
            output_ctrl_tx,
        ));

        // Wait for the output delivery controller
        let output_delivery_ctrl = output_ctrl_rx
            .await
            .map_err(|_| anyhow::anyhow!("Failed to receive output controller"))?;

        let running_state = if capture_outputs {
            InstanceRunningState::Attached
        } else {
            InstanceRunningState::Detached
        };

        Ok(Instance {
            inst_id,
            username,
            program_name,
            arguments,
            start_time: Instant::now(),
            output_delivery_ctrl,
            running_state,
            execution_handle: Some(execution_handle),
        })
    }

    /// Runs the WASM component execution
    async fn run_wasm(
        instance_id: InstanceId,
        username: String,
        component: Component,
        arguments: Vec<String>,
        capture_outputs: bool,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        output_ctrl_tx: oneshot::Sender<OutputDeliveryCtrl>,
    ) {
        // Create instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username).await;

        let output_delivery = if capture_outputs {
            OutputDelivery::Streamed
        } else {
            OutputDelivery::Buffered
        };

        output_delivery_ctrl.set_output_delivery(output_delivery);

        // Send the controller back
        if output_ctrl_tx.send(output_delivery_ctrl).is_err() {
            tracing::error!("Failed to send output delivery controller for {}", instance_id);
            return;
        }

        let result = async {
            let mut store = Store::new(&engine, inst_state);

            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| anyhow::anyhow!("Instantiation error: {e}"))?;

            let (_, run_export) = instance
                .get_export(&mut store, None, "inferlet:core/run")
                .ok_or_else(|| anyhow::anyhow!("No 'run' function found"))?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| anyhow::anyhow!("No 'run' function found"))?;

            let run_func = instance
                .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                .map_err(|e| anyhow::anyhow!("Failed to get 'run' function: {e}"))?;

            match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => Ok(None),
                Ok((Err(runtime_err),)) => Err(anyhow::anyhow!(runtime_err)),
                Err(call_err) => Err(anyhow::anyhow!("Call error: {call_err}")),
            }
        }
        .await;

        // Notify the actor that execution finished
        let cause = match result {
            Ok(return_value) => TerminationCause::Normal(return_value.unwrap_or_default()),
            Err(err) => {
                tracing::info!("Instance {instance_id} failed: {err}");
                TerminationCause::Exception(err.to_string())
            }
        };

        let _ = SERVICES.send(&instance_id, Message::ExecutionFinished { cause });
    }

    /// Cleanup and unregister this instance
    fn cleanup(&mut self) {
        if let Some(handle) = self.execution_handle.take() {
            handle.abort();
        }
        unregister(self.inst_id);
    }

    /// Send termination notification to client
    fn notify_client_termination(&self, cause: TerminationCause) {
        let inst_id = self.inst_id;
        tokio::spawn(async move {
            if let Ok(Some(client_id)) = server::get_client_id(inst_id).await {
                server::sessions::terminate(client_id, inst_id, cause).ok();
            }
            server::unregister_instance(inst_id).ok();
        });
    }
}

impl ServiceHandler for Instance {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Attach { response } => {
                let result = match &self.running_state {
                    InstanceRunningState::Attached => AttachInstanceResult::AlreadyAttached,
                    InstanceRunningState::Detached => {
                        self.running_state = InstanceRunningState::Attached;
                        AttachInstanceResult::AttachedRunning
                    }
                    InstanceRunningState::Finished(cause) => {
                        let cause = cause.clone();
                        self.running_state = InstanceRunningState::Attached;
                        AttachInstanceResult::AttachedFinished(cause)
                    }
                };
                let _ = response.send(result);
            }

            Message::Detach => {
                if matches!(self.running_state, InstanceRunningState::Attached) {
                    self.running_state = InstanceRunningState::Detached;
                }
            }

            Message::AllowOutput => {
                self.output_delivery_ctrl.allow_output();
            }

            Message::SetOutputDelivery { mode } => {
                self.output_delivery_ctrl.set_output_delivery(mode);
            }

            Message::Terminate { notification_to_client } => {
                if let Some(cause) = notification_to_client {
                    self.notify_client_termination(cause);
                }
                self.cleanup();
            }

            Message::ExecutionFinished { cause } => {
                match &self.running_state {
                    InstanceRunningState::Attached => {
                        // Client is attached, notify them and cleanup
                        self.notify_client_termination(cause);
                        self.cleanup();
                    }
                    InstanceRunningState::Detached => {
                        // No client attached, transition to Finished state
                        self.running_state = InstanceRunningState::Finished(cause);
                        // Don't cleanup yet - client may attach later
                    }
                    InstanceRunningState::Finished(_) => {
                        // Already finished, ignore
                    }
                }
            }

            Message::GetInfo { response } => {
                let info = InstanceInfo {
                    username: self.username.clone(),
                    program_name: self.program_name.clone(),
                    arguments: self.arguments.clone(),
                    elapsed_secs: self.start_time.elapsed().as_secs(),
                    running_state: self.running_state.clone(),
                };
                let _ = response.send(info);
            }
        }
    }
}
