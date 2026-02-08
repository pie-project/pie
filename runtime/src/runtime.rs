//! Runtime Superactor - Instance lifecycle orchestration
//!
//! This module provides the Runtime "superactor" that:
//! - Spawns and coordinates InstanceActor and ServerInstanceActor children
//! - Maintains minimal state for listing (actual state is in child actors)
//! - Delegates per-instance operations to child actors via Direct Addressing
//!
//! Following the Server pattern, instances register themselves in global registries
//! and receive messages directly without routing through this actor.

use std::sync::{Arc, LazyLock};

use pie_client::message;
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::component::Linker;
use wasmtime::Engine;

use crate::service::{Service, ServiceHandler};
use crate::ffi::format::QueryResponse;
use crate::{api, program};
use anyhow::anyhow;

mod dynamic_linking;
pub mod instance;
pub mod instance_actor;
pub mod output;
pub mod server_instance;

pub use instance::InstanceId;
use instance::InstanceState;
use instance_actor::InstanceConfig;
use output::OutputDelivery;
use server_instance::ServerInstanceConfig;

// =============================================================================
// Shared Type Definitions
// =============================================================================

#[derive(Debug, Clone)]
pub enum TerminationCause {
    Normal(String),
    Signal,
    Exception(String),
    OutOfResources(String),
}

#[derive(Debug, Clone)]
pub enum InstanceRunningState {
    /// The instance is running and a client is attached to it.
    Attached,
    /// The instance is running and not attached to a client.
    Detached,
    /// The instance has finished execution.
    Finished(TerminationCause),
}

impl From<InstanceRunningState> for pie_client::message::InstanceStatus {
    fn from(state: InstanceRunningState) -> Self {
        match state {
            InstanceRunningState::Attached => pie_client::message::InstanceStatus::Attached,
            InstanceRunningState::Detached => pie_client::message::InstanceStatus::Detached,
            InstanceRunningState::Finished(_) => pie_client::message::InstanceStatus::Finished,
        }
    }
}

#[derive(Clone, Debug)]
pub enum AttachInstanceResult {
    /// The instance is running and the client has been attached to it successfully.
    AttachedRunning,
    /// The instance has finished execution and the client has been attached to it successfully.
    AttachedFinished(TerminationCause),
    /// The instance is not found.
    InstanceNotFound,
    /// Another client has already been attached to this instance.
    AlreadyAttached,
}

// =============================================================================
// Public API
// =============================================================================

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the runtime service.
pub fn spawn(engine: Engine) {
    SERVICE.spawn(|| Runtime::with_engine(engine)).expect("Runtime already spawned");
}

/// Gets the runtime version.
pub async fn get_version() -> anyhow::Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::GetVersion { response: tx })?;
    rx.await.map_err(|_| anyhow!("Runtime service channel closed"))
}

/// Launch an instance of a program.
pub async fn launch_instance(
    username: String,
    program_name: String,
    arguments: Vec<String>,
    capture_outputs: bool,
) -> anyhow::Result<InstanceId> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::LaunchInstance {
        username,
        program_name,
        arguments,
        detached: !capture_outputs,
        response: tx,
    })?;
    rx.await?
}

/// Launch a server instance (HTTP handler).
pub async fn launch_server_instance(
    username: String,
    program_name: String,
    port: u32,
    arguments: Vec<String>,
) -> anyhow::Result<()> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::LaunchServerInstance {
        username,
        program_name,
        port,
        arguments,
        response: tx,
    })?;
    rx.await?
}

/// List running instances for a user.
pub async fn list_instances(username: String) -> Vec<message::InstanceInfo> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::ListInstances {
        username,
        response: tx,
    }).ok();
    rx.await.unwrap_or_default()
}

/// Spawn a child inferlet.
pub async fn spawn_child(package_name: String, args: Vec<String>) -> anyhow::Result<String> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Spawn {
        package_name,
        args,
        result: tx,
    })?;
    rx.await.map_err(|_| anyhow!("Runtime service channel closed"))
}

/// Spawn a child inferlet, returning the raw receiver for lazy polling.
pub fn spawn_child_rx(package_name: String, args: Vec<String>) -> anyhow::Result<oneshot::Receiver<String>> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Spawn {
        package_name,
        args,
        result: tx,
    })?;
    Ok(rx)
}


/// Debug query for introspection.
pub async fn debug_query(query: String) -> QueryResponse {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::DebugQuery {
        query,
        response: tx,
    }).ok();
    rx.await.unwrap_or(QueryResponse { value: "error".to_string() })
}

/// Attach to an instance (delegates to InstanceActor via Direct Addressing).
pub async fn attach_instance(inst_id: InstanceId) -> AttachInstanceResult {
    instance_actor::attach(inst_id).await
}

/// Allow output for an instance (fire-and-forget, via Direct Addressing).
pub fn allow_output(inst_id: InstanceId) {
    instance_actor::allow_output(inst_id);
}

/// Set output delivery mode for an instance (fire-and-forget, via Direct Addressing).
pub fn set_output_delivery(inst_id: InstanceId, mode: OutputDelivery) {
    instance_actor::set_output_delivery(inst_id, mode);
}

/// Terminate an instance (fire-and-forget, via Direct Addressing).
pub fn terminate_instance(inst_id: InstanceId, notification_to_client: Option<TerminationCause>) {
    instance_actor::terminate(inst_id, notification_to_client);
}



// =============================================================================
// Runtime Service
// =============================================================================

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Runtime service: instance lifecycle orchestration.
///
/// Spawns and coordinates child actors (InstanceActor, ServerInstanceActor).
/// Delegates per-instance operations to child actors via Direct Addressing.
struct Runtime {
    engine: Engine,
    linker: Arc<Linker<InstanceState>>,
}

impl Runtime {
    fn with_engine(engine: Engine) -> Self {
        let mut linker = Linker::<InstanceState>::new(&engine);

        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .expect("Failed to link WASI");
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .expect("Failed to link WASI HTTP");

        api::add_to_linker(&mut linker).unwrap();

        Runtime {
            engine,
            linker: Arc::new(linker),
        }
    }

    fn get_version(&self) -> String {
        VERSION.to_string()
    }

    async fn launch_instance(
        &self,
        username: String,
        program_name: String,
        arguments: Vec<String>,
        capture_outputs: bool,
    ) -> anyhow::Result<InstanceId> {
        let component = program::get_wasm_component(&program::ProgramName::parse(&program_name))
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        let inst_id = Uuid::new_v4();

        let config = InstanceConfig {
            inst_id,
            username,
            program_name,
            arguments,
            detached: !capture_outputs,
            component,
            engine: self.engine.clone(),
            linker: self.linker.clone(),
        };

        instance_actor::InstanceActor::spawn(config).await
    }

    async fn launch_server_instance(
        &self,
        username: String,
        program_name: String,
        port: u32,
        arguments: Vec<String>,
    ) -> anyhow::Result<()> {
        let component = program::get_wasm_component(&program::ProgramName::parse(&program_name))
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        let inst_id = Uuid::new_v4();

        let config = ServerInstanceConfig {
            inst_id,
            username,
            program_name,
            port: port as u16,
            arguments,
            component,
            engine: self.engine.clone(),
            linker: self.linker.clone(),
        };

        server_instance::ServerInstanceActor::spawn(config).await?;
        Ok(())
    }

    async fn list_instances(&self, username: &str) -> Vec<message::InstanceInfo> {
        let show_all = username == "internal";
        let ids = instance_actor::list_instance_ids();

        let mut instances = Vec::new();
        for id in ids {
            if let Some(info) = instance_actor::get_info(id).await {
                if show_all || info.username == username {
                    instances.push(message::InstanceInfo {
                        id: id.to_string(),
                        arguments: info.arguments,
                        status: info.running_state.into(),
                        username: info.username,
                        elapsed_secs: info.elapsed_secs,
                        kv_pages_used: 0,
                    });
                }
            }
        }

        instances.sort_by(|a, b| a.elapsed_secs.cmp(&b.elapsed_secs));
        instances.truncate(50);
        instances
    }

    fn debug_query(&self, query: &str) -> QueryResponse {
        let value = match query {
            "ping" => "pong".to_string(),
            "get_instance_count" => format!("{}", instance_actor::list_instance_ids().len()),
            "get_server_instance_count" => format!("{}", server_instance::list_instance_ids().len()),
            "list_running_instances" => {
                let ids = instance_actor::list_instance_ids();
                ids.iter()
                    .map(|id| format!("Instance ID: {}", id))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => format!("Unknown query: {}", query),
        };
        QueryResponse { value }
    }

    fn spawn_child(&self, package_name: &str, args: Vec<String>) -> String {
        format!("spawn not yet implemented: {} {:?}", package_name, args)
    }
}

// =============================================================================
// ServiceHandler
// =============================================================================

#[derive(Debug)]
enum Message {
    GetVersion {
        response: oneshot::Sender<String>,
    },
    LaunchInstance {
        username: String,
        program_name: String,
        arguments: Vec<String>,
        detached: bool,
        response: oneshot::Sender<anyhow::Result<InstanceId>>,
    },
    LaunchServerInstance {
        username: String,
        program_name: String,
        port: u32,
        arguments: Vec<String>,
        response: oneshot::Sender<anyhow::Result<()>>,
    },
    ListInstances {
        username: String,
        response: oneshot::Sender<Vec<message::InstanceInfo>>,
    },
    DebugQuery {
        query: String,
        response: oneshot::Sender<QueryResponse>,
    },
    Spawn {
        package_name: String,
        args: Vec<String>,
        result: oneshot::Sender<String>,
    },
}

impl ServiceHandler for Runtime {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetVersion { response } => {
                let _ = response.send(self.get_version());
            }
            Message::LaunchInstance { username, program_name, arguments, detached, response } => {
                let _ = response.send(self.launch_instance(username, program_name, arguments, !detached).await);
            }
            Message::LaunchServerInstance { username, program_name, port, arguments, response } => {
                let _ = response.send(self.launch_server_instance(username, program_name, port, arguments).await);
            }
            Message::ListInstances { username, response } => {
                let _ = response.send(self.list_instances(&username).await);
            }
            Message::DebugQuery { query, response } => {
                let _ = response.send(self.debug_query(&query));
            }
            Message::Spawn { package_name, args, result } => {
                let _ = result.send(self.spawn_child(&package_name, args));
            }
        }
    }
}
