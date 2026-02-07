//! Instance Actor - Per-instance lifecycle management
//!
//! This module provides the InstanceActor for managing individual WASM instances.
//! Following the Session pattern, instances are registered in a global registry
//! for Direct Addressing, allowing messages to bypass the RuntimeActor.

use std::sync::LazyLock;
use std::time::Instant;

use dashmap::DashMap;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use wasmtime::component::{Component, Linker};
use wasmtime::{Engine, Store};
use std::sync::Arc;

use crate::service::{Service, ServiceHandler};
use crate::{program, server};

use super::instance::{InstanceId, InstanceState};
use super::output::{OutputDelivery, OutputDeliveryCtrl};
use super::{AttachInstanceResult, InstanceRunningState, TerminationCause};

// =============================================================================
// Instance Registry (Direct Addressing)
// =============================================================================

/// Global registry mapping InstanceId to instance actors.
/// Allows direct message delivery to instances without routing through Runtime.
static INSTANCE_REGISTRY: LazyLock<DashMap<InstanceId, Service<InstanceMessage>>> =
    LazyLock::new(DashMap::new);

/// Sends a message directly to an instance by InstanceId.
pub fn send(inst_id: InstanceId, msg: InstanceMessage) -> anyhow::Result<()> {
    INSTANCE_REGISTRY
        .get(&inst_id)
        .ok_or_else(|| anyhow::anyhow!("Instance not found"))?
        .send(msg)
}

/// Check if an instance exists for the given InstanceId.
pub fn exists(inst_id: InstanceId) -> bool {
    INSTANCE_REGISTRY.contains_key(&inst_id)
}

/// Remove an instance from the registry.
fn unregister(inst_id: InstanceId) {
    INSTANCE_REGISTRY.remove(&inst_id);
}

// =============================================================================
// Instance Messages
// =============================================================================

/// Messages that can be sent directly to an InstanceActor.
#[derive(Debug)]
pub enum InstanceMessage {
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
// InstanceActor
// =============================================================================

/// Actor managing a single WASM instance lifecycle.
pub struct InstanceActor {
    inst_id: InstanceId,
    username: String,
    program_name: String,
    arguments: Vec<String>,
    start_time: Instant,
    output_delivery_ctrl: OutputDeliveryCtrl,
    running_state: InstanceRunningState,
    execution_handle: Option<JoinHandle<()>>,
}

/// Configuration for spawning a new instance
pub struct InstanceConfig {
    pub inst_id: InstanceId,
    pub username: String,
    pub program_name: String,
    pub arguments: Vec<String>,
    pub detached: bool,
    pub component: Component,
    pub engine: Engine,
    pub linker: Arc<Linker<InstanceState>>,
}

impl InstanceActor {
    /// Spawns a new InstanceActor and registers it in the global registry.
    /// Returns the InstanceId on success.
    pub async fn spawn(config: InstanceConfig) -> anyhow::Result<InstanceId> {
        let inst_id = config.inst_id;
        let username = config.username.clone();
        let program_name = config.program_name.clone();
        let arguments = config.arguments.clone();
        let detached = config.detached;

        // Create the actor entry in the registry first
        let service = Service::new();
        INSTANCE_REGISTRY.insert(inst_id, service);

        // Get a reference to spawn with
        let actor_ref = INSTANCE_REGISTRY.get(&inst_id).unwrap();

        // Create channels for synchronization
        let (output_ctrl_tx, output_ctrl_rx) = oneshot::channel();

        // Spawn WASM execution task
        let execution_handle = tokio::spawn(Self::run_wasm(
            inst_id,
            config.username,
            config.component,
            config.arguments,
            detached,
            config.engine,
            config.linker,
            output_ctrl_tx,
        ));

        // Wait for the output delivery controller
        let output_delivery_ctrl = output_ctrl_rx
            .await
            .map_err(|_| anyhow::anyhow!("Failed to receive output controller"))?;

        let running_state = if detached {
            InstanceRunningState::Detached
        } else {
            InstanceRunningState::Attached
        };

        // Spawn the actor with the initialized state
        actor_ref.spawn(|| InstanceActor {
            inst_id,
            username,
            program_name,
            arguments,
            start_time: Instant::now(),
            output_delivery_ctrl,
            running_state,
            execution_handle: Some(execution_handle),
        })?;

        Ok(inst_id)
    }

    /// Runs the WASM component execution
    async fn run_wasm(
        instance_id: InstanceId,
        username: String,
        component: Component,
        arguments: Vec<String>,
        detached: bool,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        output_ctrl_tx: oneshot::Sender<OutputDeliveryCtrl>,
    ) {
        // Create instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username).await;

        let output_delivery = if detached {
            OutputDelivery::Buffered
        } else {
            OutputDelivery::Streamed
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

        let _ = send(instance_id, InstanceMessage::ExecutionFinished { cause });
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

impl ServiceHandler for InstanceActor {
    type Message = InstanceMessage;

    async fn handle(&mut self, msg: InstanceMessage) {
        match msg {
            InstanceMessage::Attach { response } => {
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

            InstanceMessage::Detach => {
                if matches!(self.running_state, InstanceRunningState::Attached) {
                    self.running_state = InstanceRunningState::Detached;
                }
            }

            InstanceMessage::AllowOutput => {
                self.output_delivery_ctrl.allow_output();
            }

            InstanceMessage::SetOutputDelivery { mode } => {
                self.output_delivery_ctrl.set_output_delivery(mode);
            }

            InstanceMessage::Terminate { notification_to_client } => {
                if let Some(cause) = notification_to_client {
                    self.notify_client_termination(cause);
                }
                self.cleanup();
            }

            InstanceMessage::ExecutionFinished { cause } => {
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

            InstanceMessage::GetInfo { response } => {
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

// =============================================================================
// Async Helper Functions (Ergonomic Wrappers)
// =============================================================================

/// Attach to an instance.
pub async fn attach(inst_id: InstanceId) -> AttachInstanceResult {
    let (tx, rx) = oneshot::channel();
    if send(inst_id, InstanceMessage::Attach { response: tx }).is_err() {
        return AttachInstanceResult::InstanceNotFound;
    }
    rx.await.unwrap_or(AttachInstanceResult::InstanceNotFound)
}

/// Detach from an instance (fire-and-forget).
pub fn detach(inst_id: InstanceId) {
    let _ = send(inst_id, InstanceMessage::Detach);
}

/// Allow output for an instance (fire-and-forget).
pub fn allow_output(inst_id: InstanceId) {
    let _ = send(inst_id, InstanceMessage::AllowOutput);
}

/// Set output delivery mode for an instance (fire-and-forget).
pub fn set_output_delivery(inst_id: InstanceId, mode: OutputDelivery) {
    let _ = send(inst_id, InstanceMessage::SetOutputDelivery { mode });
}

/// Terminate an instance (fire-and-forget).
pub fn terminate(inst_id: InstanceId, notification_to_client: Option<TerminationCause>) {
    let _ = send(inst_id, InstanceMessage::Terminate { notification_to_client });
}

/// Get instance info for listing.
pub async fn get_info(inst_id: InstanceId) -> Option<InstanceInfo> {
    let (tx, rx) = oneshot::channel();
    send(inst_id, InstanceMessage::GetInfo { response: tx }).ok()?;
    rx.await.ok()
}

/// List all registered instance IDs.
pub fn list_instance_ids() -> Vec<InstanceId> {
    INSTANCE_REGISTRY.iter().map(|r| *r.key()).collect()
}
