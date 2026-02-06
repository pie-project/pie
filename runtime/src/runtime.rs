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
// Runtime Actor (Superactor)
// =============================================================================

/// Global singleton Runtime actor.
static ACTOR: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the Runtime actor with the given engine.
pub fn spawn(engine: Engine) {
    ACTOR.spawn_with::<RuntimeActor, _>(|| RuntimeActor::with_engine(engine));
}

/// Check if the runtime actor is spawned.
pub fn is_spawned() -> bool {
    ACTOR.is_spawned()
}

/// Launch an instance of a program.
pub async fn launch_instance(
    username: String,
    program_name: String,
    arguments: Vec<String>,
    detached: bool,
) -> anyhow::Result<InstanceId> {
    let (tx, rx) = oneshot::channel();
    Message::LaunchInstance {
        username,
        program_name,
        arguments,
        detached,
        response: tx,
    }
    .send()
    .map_err(|_| anyhow!("Runtime actor not running"))?;
    rx.await.map_err(|_| anyhow!("Runtime actor did not respond"))?
}

/// Launch a server instance (HTTP handler).
pub async fn launch_server_instance(
    username: String,
    program_name: String,
    port: u32,
    arguments: Vec<String>,
) -> anyhow::Result<()> {
    let (tx, rx) = oneshot::channel();
    Message::LaunchServerInstance {
        username,
        program_name,
        port,
        arguments,
        response: tx,
    }
    .send()
    .map_err(|_| anyhow!("Runtime actor not running"))?;
    rx.await.map_err(|_| anyhow!("Runtime actor did not respond"))?
}

/// List running instances for a user.
pub async fn list_instances(username: String) -> Vec<message::InstanceInfo> {
    let (tx, rx) = oneshot::channel();
    let _ = Message::ListInstances {
        username,
        response: tx,
    }
    .send();
    rx.await.unwrap_or_default()
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
// Messages
// =============================================================================

/// Messages for the Runtime actor.
///
/// The Runtime actor now handles only lifecycle orchestration.
/// Per-instance messages go directly to InstanceActor via instance_actor::send().
#[derive(Debug)]
pub enum Message {
    /// Get the runtime version
    GetVersion {
        response: oneshot::Sender<String>,
    },

    /// Launch a program instance
    LaunchInstance {
        username: String,
        program_name: String,
        arguments: Vec<String>,
        detached: bool,
        response: oneshot::Sender<anyhow::Result<InstanceId>>,
    },

    /// Launch a server instance (HTTP handler)
    LaunchServerInstance {
        username: String,
        program_name: String,
        port: u32,
        arguments: Vec<String>,
        response: oneshot::Sender<anyhow::Result<()>>,
    },

    /// List running instances for a user
    ListInstances {
        username: String,
        response: oneshot::Sender<Vec<message::InstanceInfo>>,
    },

    /// Debug query for introspection
    DebugQuery {
        query: String,
        response: oneshot::Sender<QueryResponse>,
    },

    /// Spawn a child inferlet
    Spawn {
        package_name: String,
        args: Vec<String>,
        result: oneshot::Sender<String>,
    },
}

impl Message {
    /// Sends this message to the runtime actor.
    pub fn send(self) -> anyhow::Result<()> {
        ACTOR.send(self)
    }
}

// =============================================================================
// RuntimeActor (Superactor)
// =============================================================================

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Runtime superactor implementation.
///
/// Spawns and coordinates child actors (InstanceActor, ServerInstanceActor).
/// Delegates per-instance operations to child actors via Direct Addressing.
struct RuntimeActor {
    /// The Wasmtime engine (global)
    engine: Engine,
    /// Pre-configured linker with WASI and API bindings
    linker: Arc<Linker<InstanceState>>,
}

impl RuntimeActor {
    fn with_engine(engine: Engine) -> Self {
        let mut linker = Linker::<InstanceState>::new(&engine);

        // Add WASI and HTTP bindings
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .expect("Failed to link WASI");
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .expect("Failed to link WASI HTTP");

        // Add custom API bindings
        api::add_to_linker(&mut linker).unwrap();

        RuntimeActor {
            engine,
            linker: Arc::new(linker),
        }
    }

    /// Gets the runtime version.
    fn get_version(&self) -> String {
        VERSION.to_string()
    }

    /// Launches a program instance by spawning an InstanceActor.
    async fn launch_instance(
        &self,
        username: String,
        program_name: String,
        arguments: Vec<String>,
        detached: bool,
    ) -> anyhow::Result<InstanceId> {
        // Get the component from program manager
        let component = program::get_component(&program::ProgramName::parse(&program_name))
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        let inst_id = Uuid::new_v4();

        // Spawn the InstanceActor
        let config = InstanceConfig {
            inst_id,
            username,
            program_name,
            arguments,
            detached,
            component,
            engine: self.engine.clone(),
            linker: self.linker.clone(),
        };

        instance_actor::InstanceActor::spawn(config).await
    }

    /// Launches a server instance by spawning a ServerInstanceActor.
    async fn launch_server_instance(
        &self,
        username: String,
        program_name: String,
        port: u32,
        arguments: Vec<String>,
    ) -> anyhow::Result<()> {
        // Get the component from program manager
        let component = program::get_component(&program::ProgramName::parse(&program_name))
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        let inst_id = Uuid::new_v4();

        // Spawn the ServerInstanceActor
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

    /// Lists instances by querying all registered InstanceActors.
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

    /// Handles debug queries.
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

    /// Spawns a child inferlet (not yet implemented).
    fn spawn_child(&self, package_name: &str, args: Vec<String>) -> String {
        format!("spawn not yet implemented: {} {:?}", package_name, args)
    }
}

impl ServiceHandler for RuntimeActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetVersion { response } => {
                let _ = response.send(self.get_version());
            }
            Message::LaunchInstance {
                username,
                program_name,
                arguments,
                detached,
                response,
            } => {
                let result = self
                    .launch_instance(username, program_name, arguments, detached)
                    .await;
                let _ = response.send(result);
            }
            Message::LaunchServerInstance {
                username,
                program_name,
                port,
                arguments,
                response,
            } => {
                let result = self
                    .launch_server_instance(username, program_name, port, arguments)
                    .await;
                let _ = response.send(result);
            }
            Message::ListInstances { username, response } => {
                let _ = response.send(self.list_instances(&username).await);
            }
            Message::DebugQuery { query, response } => {
                let _ = response.send(self.debug_query(&query));
            }
            Message::Spawn {
                package_name,
                args,
                result,
            } => {
                let _ = result.send(self.spawn_child(&package_name, args));
            }
        }
    }
}
