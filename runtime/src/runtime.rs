//! Runtime Service - Instance lifecycle and program management
//!
//! This module provides actors for runtime management using the
//! modern actor model (Handle trait). It implements the Service-Actor pattern
//! where `Runtime` is the business logic and `RuntimeActor` is the async interface.

use std::net::SocketAddr;
use std::sync::{Arc, LazyLock};

use dashmap::DashMap;
use hyper::server::conn::http1;
use pie_client::message;
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::component::Resource;
use wasmtime::{Engine, Store, component::Component, component::Linker};
use wasmtime_wasi_http::WasiHttpView;
use wasmtime_wasi_http::bindings::exports::wasi::http::incoming_handler::{
    IncomingRequest, ResponseOutparam,
};
use wasmtime_wasi_http::bindings::http::types::Scheme;
use wasmtime_wasi_http::body::HyperOutgoingBody;
use wasmtime_wasi_http::io::TokioIo;

use super::instance::{InstanceId, InstanceState, OutputDelivery, OutputDeliveryCtrl};
use crate::actor::{Actor, Handle, SendError};
use crate::ffi::format::QueryResponse;
use crate::{api, program, server};
use thiserror::Error;

mod dynamic_linking;

// =============================================================================
// Shared Type Definitions
// =============================================================================

#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Wrap general I/O errors
    #[error("I/O error occurred: {0}")]
    Io(#[from] std::io::Error),

    /// Wrap Wasmtime errors
    #[error("Wasmtime error occurred: {0}")]
    Wasmtime(#[from] wasmtime::Error),

    /// No program found for the given hashes
    #[error("No such program with wasm_hash={0}, manifest_hash={1}")]
    MissingProgram(String, String),

    /// Failed to compile a WASM component from disk
    #[error("Failed to compile program at path {path:?}: {source}")]
    CompileWasm {
        path: std::path::PathBuf,
        #[source]
        source: wasmtime::Error,
    },

    /// Fallback for unexpected cases
    #[error("Runtime error: {0}")]
    Other(String),
}

/// A key identifying a compiled program by its WASM and manifest hashes.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProgramHash {
    pub wasm_hash: String,
    pub manifest_hash: String,
}

impl ProgramHash {
    pub fn new(wasm_hash: String, manifest_hash: String) -> Self {
        Self {
            wasm_hash,
            manifest_hash,
        }
    }
}

impl std::fmt::Display for ProgramHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.wasm_hash, self.manifest_hash)
    }
}

/// A compiled program with its component and dependency information.
#[derive(Clone)]
struct CompiledProgram {
    /// The compiled WASM component
    component: Component,
    /// Dependencies of this program, each specified by their hashes
    dependencies: Vec<ProgramHash>,
}

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

/// Cleanup resources for a finished instance.
/// This is a stub for now since the new model architecture doesn't require per-instance cleanup.
fn cleanup_instance(_inst_id: InstanceId) {
    // No-op: The new model/context/inference actors manage their own resources.
    // Instance-specific resources are cleaned up when the instance task terminates.
}

// =============================================================================
// Instance State Types (local to new Runtime)
// =============================================================================

/// Handle to a running instance, tracking its state and resources.
struct InstanceHandle {
    username: String,
    program_name: String,
    arguments: Vec<String>,
    start_time: std::time::Instant,
    output_delivery_ctrl: OutputDeliveryCtrl,
    running_state: InstanceRunningState,
    join_handle: tokio::task::JoinHandle<()>,
}

// =============================================================================
// Runtime Actor
// =============================================================================

/// Global singleton Runtime actor.
static ACTOR: LazyLock<Actor<Message>> = LazyLock::new(Actor::new);

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
) -> Result<InstanceId, RuntimeError> {
    let (tx, rx) = oneshot::channel();
    Message::LaunchInstance {
        username,
        program_name,
        arguments,
        detached,
        response: tx,
    }
    .send()
    .map_err(|_| RuntimeError::Other("Runtime actor not running".to_string()))?;
    rx.await.map_err(|_| RuntimeError::Other("Runtime actor did not respond".to_string()))?
}

/// Launch a server instance (HTTP handler).
pub async fn launch_server_instance(
    username: String,
    program_name: String,
    port: u32,
    arguments: Vec<String>,
) -> Result<(), RuntimeError> {
    let (tx, rx) = oneshot::channel();
    Message::LaunchServerInstance {
        username,
        program_name,
        port,
        arguments,
        response: tx,
    }
    .send()
    .map_err(|_| RuntimeError::Other("Runtime actor not running".to_string()))?;
    rx.await.map_err(|_| RuntimeError::Other("Runtime actor did not respond".to_string()))?
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

/// Attach to an instance.
pub async fn attach_instance(inst_id: InstanceId) -> AttachInstanceResult {
    let (tx, rx) = oneshot::channel();
    let _ = Message::AttachInstance {
        inst_id,
        response: tx,
    }
    .send();
    rx.await.unwrap_or(AttachInstanceResult::InstanceNotFound)
}

/// Allow output for an instance (fire-and-forget).
pub fn allow_output(inst_id: InstanceId) {
    let _ = Message::AllowOutput { inst_id }.send();
}

/// Set output delivery mode for an instance (fire-and-forget).
pub fn set_output_delivery(inst_id: InstanceId, mode: OutputDelivery) {
    let _ = Message::SetOutputDelivery { inst_id, mode }.send();
}

/// Terminate an instance (fire-and-forget).
pub fn terminate_instance(inst_id: InstanceId, notification_to_client: Option<TerminationCause>) {
    let _ = Message::TerminateInstance {
        inst_id,
        notification_to_client,
    }
    .send();
}

// =============================================================================
// Messages
// =============================================================================

/// Messages for the Runtime actor.
///
/// Note: No longer needs manual Debug impl since Component is not passed in messages.
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
        response: oneshot::Sender<Result<InstanceId, RuntimeError>>,
    },

    /// Attach to an instance
    AttachInstance {
        inst_id: InstanceId,
        response: oneshot::Sender<AttachInstanceResult>,
    },

    /// Detach from an instance
    DetachInstance {
        inst_id: InstanceId,
    },

    /// Allow output for an instance
    AllowOutput {
        inst_id: InstanceId,
    },

    /// Launch a server instance (HTTP handler)
    LaunchServerInstance {
        username: String,
        program_name: String,
        port: u32,
        arguments: Vec<String>,
        response: oneshot::Sender<Result<(), RuntimeError>>,
    },

    /// Terminate an instance
    TerminateInstance {
        inst_id: InstanceId,
        notification_to_client: Option<TerminationCause>,
    },

    /// Mark an instance as finished
    FinishInstance {
        inst_id: InstanceId,
        cause: TerminationCause,
    },

    /// Set output delivery mode for an instance
    SetOutputDelivery {
        inst_id: InstanceId,
        mode: OutputDelivery,
    },

    /// Debug query for introspection
    DebugQuery {
        query: String,
        response: oneshot::Sender<QueryResponse>,
    },

    /// List running instances for a user
    ListInstances {
        username: String,
        response: oneshot::Sender<Vec<message::InstanceInfo>>,
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
    pub fn send(self) -> Result<(), SendError> {
        ACTOR.send(self)
    }
}

// =============================================================================
// RuntimeActor
// =============================================================================

/// Runtime actor implementation.
struct RuntimeActor {
    service: Runtime,
}

impl RuntimeActor {
    fn with_engine(engine: Engine) -> Self {
        RuntimeActor {
            service: Runtime::new(engine),
        }
    }
}

impl Handle for RuntimeActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::GetVersion { response } => {
                let _ = response.send(self.service.get_version());
            }
            Message::LaunchInstance {
                username,
                program_name,
                arguments,
                detached,
                response,
            } => {
                let result = self
                    .service
                    .launch_instance(username, program_name, arguments, detached)
                    .await;
                let _ = response.send(result);
            }
            Message::AttachInstance { inst_id, response } => {
                let _ = response.send(self.service.attach_instance(inst_id));
            }
            Message::DetachInstance { inst_id } => {
                self.service.detach_instance(inst_id);
            }
            Message::AllowOutput { inst_id } => {
                self.service.allow_output(inst_id);
            }
            Message::LaunchServerInstance {
                username,
                program_name,
                port,
                arguments,
                response,
            } => {
                let result = self
                    .service
                    .launch_server_instance(username, program_name, port, arguments)
                    .await;
                let _ = response.send(result);
            }
            Message::TerminateInstance {
                inst_id,
                notification_to_client,
            } => {
                self.service.terminate_instance(inst_id, notification_to_client);
            }
            Message::FinishInstance { inst_id, cause } => {
                self.service.finish_instance(inst_id, cause);
            }
            Message::SetOutputDelivery { inst_id, mode } => {
                self.service.set_output_delivery(inst_id, mode);
            }
            Message::DebugQuery { query, response } => {
                let _ = response.send(self.service.debug_query(&query));
            }
            Message::ListInstances { username, response } => {
                let _ = response.send(self.service.list_instances(&username));
            }
            Message::Spawn {
                package_name,
                args,
                result,
            } => {
                let _ = result.send(self.service.spawn_child(&package_name, args));
            }
        }
    }
}

// =============================================================================
// Runtime - Business Logic
// =============================================================================

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The runtime service handles instance lifecycle and program management.
///
/// This is the core business logic, separate from the actor message handling.
pub struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,
    /// Pre-configured linker with WASI and API bindings
    linker: Arc<Linker<InstanceState>>,
    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,
    /// Finished instances (awaiting attachment)
    finished_instances: DashMap<InstanceId, InstanceHandle>,
    /// Running server instances
    running_server_instances: DashMap<InstanceId, InstanceHandle>,
}

impl Runtime {
    /// Creates a new runtime service with the given engine.
    pub fn new(engine: Engine) -> Self {
        let mut linker = Linker::<InstanceState>::new(&engine);

        // Add WASI and HTTP bindings
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI HTTP: {e}")))
            .unwrap();

        // Add custom API bindings
        api::add_to_linker(&mut linker).unwrap();

        Runtime {
            engine,
            linker: Arc::new(linker),
            running_instances: DashMap::new(),
            finished_instances: DashMap::new(),
            running_server_instances: DashMap::new(),
        }
    }

    /// Gets the runtime version.
    pub fn get_version(&self) -> String {
        VERSION.to_string()
    }

    /// Launches a program instance.
    pub async fn launch_instance(
        &self,
        username: String,
        program_name: String,
        arguments: Vec<String>,
        detached: bool,
    ) -> Result<InstanceId, RuntimeError> {
        // Get the component from program manager
        let component = program::get_component(&program::ProgramName::parse(&program_name))
            .await
            .ok_or_else(|| RuntimeError::Other(format!("Component not found for program: {}", program_name)))?;

        let instance_id = Uuid::new_v4();

        let engine = self.engine.clone();
        let linker = self.linker.clone();

        // Create channels for synchronization
        let (start_tx, start_rx) = oneshot::channel();
        let (output_delivery_ctrl_tx, output_delivery_ctrl_rx) = oneshot::channel();

        let join_handle = tokio::spawn(Self::launch(
            instance_id,
            username.clone(),
            component,
            arguments.clone(),
            detached,
            engine,
            linker,
            start_rx,
            output_delivery_ctrl_tx,
        ));

        // Wait for the output delivery controller
        let output_delivery_ctrl = output_delivery_ctrl_rx.await.unwrap();

        let running_state = if detached {
            InstanceRunningState::Detached
        } else {
            InstanceRunningState::Attached
        };

        // Record in running instances
        let instance_handle = InstanceHandle {
            username,
            program_name,
            arguments,
            start_time: std::time::Instant::now(),
            output_delivery_ctrl,
            running_state,
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        // Signal the task to start
        let _ = start_tx.send(());

        Ok(instance_id)
    }

    /// Attaches a client to an instance.
    pub fn attach_instance(&self, inst_id: InstanceId) -> AttachInstanceResult {
        // Check running instances
        if let Some(mut handle) = self.running_instances.get_mut(&inst_id) {
            if let InstanceRunningState::Attached = handle.running_state {
                return AttachInstanceResult::AlreadyAttached;
            }
            handle.running_state = InstanceRunningState::Attached;
            return AttachInstanceResult::AttachedRunning;
        }

        // Check finished instances
        if let Some(mut handle) = self.finished_instances.get_mut(&inst_id) {
            if matches!(&handle.running_state, InstanceRunningState::Finished(_)) {
                if let InstanceRunningState::Finished(cause) =
                    std::mem::replace(&mut handle.running_state, InstanceRunningState::Attached)
                {
                    return AttachInstanceResult::AttachedFinished(cause);
                }
            }
        }

        AttachInstanceResult::InstanceNotFound
    }

    /// Detaches a client from an instance.
    pub fn detach_instance(&self, inst_id: InstanceId) {
        if let Some(mut handle) = self.running_instances.get_mut(&inst_id) {
            handle.running_state = InstanceRunningState::Detached;
        }
    }

    /// Allows output for a running instance.
    pub fn allow_output(&self, inst_id: InstanceId) {
        if let Some(handle) = self.running_instances.get(&inst_id) {
            handle.output_delivery_ctrl.allow_output();
        }
    }

    /// Sets the output delivery mode for an instance.
    pub fn set_output_delivery(&self, instance_id: InstanceId, output_delivery: OutputDelivery) {
        if let Some(handle) = self.running_instances.get(&instance_id) {
            handle.output_delivery_ctrl.set_output_delivery(output_delivery);
        }
        if let Some(handle) = self.finished_instances.get(&instance_id) {
            handle.output_delivery_ctrl.set_output_delivery(output_delivery);
        }
    }

    /// Launches a server instance (HTTP handler).
    pub async fn launch_server_instance(
        &self,
        username: String,
        program_name: String,
        port: u32,
        arguments: Vec<String>,
    ) -> Result<(), RuntimeError> {
        // Get the component from program manager
        let component = program::get_component(&program::ProgramName::parse(&program_name))
            .await
            .ok_or_else(|| RuntimeError::Other(format!("Component not found for program: {}", program_name)))?;

        let instance_id = Uuid::new_v4();

        let engine = self.engine.clone();
        let linker = self.linker.clone();
        let addr = SocketAddr::from(([127, 0, 0, 1], port as u16));

        let (start_tx, start_rx) = oneshot::channel();

        let join_handle = tokio::spawn(Self::launch_server(
            addr,
            username.clone(),
            component,
            arguments.clone(),
            engine,
            linker,
            start_rx,
        ));

        // Create a dummy output delivery controller for server instances
        let (dummy_state, output_delivery_ctrl) =
            InstanceState::new(Uuid::new_v4(), username.clone(), vec![]).await;
        drop(dummy_state);

        let instance_handle = InstanceHandle {
            username,
            program_name,
            arguments,
            start_time: std::time::Instant::now(),
            output_delivery_ctrl,
            running_state: InstanceRunningState::Detached,
            join_handle,
        };
        self.running_server_instances.insert(instance_id, instance_handle);

        let _ = start_tx.send(());

        Ok(())
    }

    /// Terminates a running instance.
    pub fn terminate_instance(
        &self,
        instance_id: InstanceId,
        notification_to_client: Option<TerminationCause>,
    ) {
        let instance = self
            .running_instances
            .remove(&instance_id)
            .or(self.finished_instances.remove(&instance_id));

        if let Some((_, handle)) = instance {
            handle.join_handle.abort();
            cleanup_instance(instance_id);

            if let Some(cause) = notification_to_client {
                // Send termination notification directly to session
                tokio::spawn(async move {
                    if let Some(client_id) = server::get_client_id(instance_id).await {
                        server::session_send(
                            client_id,
                            server::SessionMessage::Terminate {
                                inst_id: instance_id,
                                cause,
                            },
                        )
                        .ok();
                    }
                    server::Message::UnregisterInstance { inst_id: instance_id }.send().ok();
                });
            }
        }
    }

    /// Marks an instance as finished.
    pub fn finish_instance(&self, instance_id: InstanceId, cause: TerminationCause) {
        if let Some((_, mut handle)) = self.running_instances.remove(&instance_id) {
            match handle.running_state {
                InstanceRunningState::Attached => {
                    handle.join_handle.abort();
                    cleanup_instance(instance_id);

                    // Send termination notification directly to session
                    tokio::spawn(async move {
                        if let Some(client_id) = server::get_client_id(instance_id).await {
                            server::session_send(
                                client_id,
                                server::SessionMessage::Terminate {
                                    inst_id: instance_id,
                                    cause,
                                },
                            )
                            .ok();
                        }
                        server::Message::UnregisterInstance { inst_id: instance_id }.send().ok();
                    });
                }
                InstanceRunningState::Detached => {
                    handle.running_state = InstanceRunningState::Finished(cause);
                    self.finished_instances.insert(instance_id, handle);
                }
                InstanceRunningState::Finished(_) => {
                    panic!("Instance {instance_id} is already finished and cannot be sealed again")
                }
            }
        }
    }

    /// Handles debug queries.
    pub fn debug_query(&self, query: &str) -> QueryResponse {
        let value = match query {
            "ping" => "pong".to_string(),
            "get_instance_count" => format!("{}", self.running_instances.len()),
            "get_server_instance_count" => format!("{}", self.running_server_instances.len()),
            "list_running_instances" => {
                let instances: Vec<String> = self
                    .running_instances
                    .iter()
                    .map(|item| {
                        format!(
                            "Instance ID: {}, Program: {}",
                            item.key(),
                            item.value().program_name
                        )
                    })
                    .collect();
                instances.join("\n")
            }
            _ => format!("Unknown query: {}", query),
        };
        QueryResponse { value }
    }

    /// Lists instances for a user.
    pub fn list_instances(&self, username: &str) -> Vec<message::InstanceInfo> {
        let show_all = username == "internal";
        let mut instances: Vec<message::InstanceInfo> = self
            .running_instances
            .iter()
            .chain(self.finished_instances.iter())
            .filter(|item| show_all || item.value().username == username)
            .map(|item| message::InstanceInfo {
                id: item.key().to_string(),
                arguments: item.value().arguments.clone(),
                status: item.value().running_state.clone().into(),
                username: item.value().username.clone(),
                elapsed_secs: item.value().start_time.elapsed().as_secs(),
                kv_pages_used: 0,
            })
            .collect();

        instances.sort_by(|a, b| a.elapsed_secs.cmp(&b.elapsed_secs));
        instances.truncate(50);

        instances
    }

    /// Spawns a child inferlet (not yet implemented).
    pub fn spawn_child(&self, package_name: &str, args: Vec<String>) -> String {
        format!("spawn not yet implemented: {} {:?}", package_name, args)
    }

    // =========================================================================
    // Instance Execution
    // =========================================================================

    async fn launch(
        instance_id: InstanceId,
        username: String,
        component: Component,
        arguments: Vec<String>,
        detached: bool,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        start_rx: oneshot::Receiver<()>,
        output_delivery_ctrl_tx: oneshot::Sender<OutputDeliveryCtrl>,
    ) {
        // Create instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username, arguments).await;

        let output_delivery = if detached {
            OutputDelivery::Buffered
        } else {
            OutputDelivery::Streamed
        };

        output_delivery_ctrl.set_output_delivery(output_delivery);

        // Send the controller back
        output_delivery_ctrl_tx
            .send(output_delivery_ctrl)
            .map_err(|_| "Failed to send output delivery controller")
            .unwrap();

        // Wait for start signal
        start_rx.await.unwrap();

        let result = async {
            let mut store = Store::new(&engine, inst_state);

            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

            let (_, run_export) = instance
                .get_export(&mut store, None, "inferlet:core/run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let run_func = instance
                .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                .map_err(|e| RuntimeError::Other(format!("Failed to get 'run' function: {e}")))?;

            match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    let return_value = store.data().return_value();
                    Ok(return_value)
                }
                Ok((Err(runtime_err),)) => Err(RuntimeError::Other(runtime_err)),
                Err(call_err) => Err(RuntimeError::Other(format!("Call error: {call_err}"))),
            }
        }
        .await;

        match result {
            Ok(return_value) => {
                let _ = Message::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Normal(return_value.unwrap_or_default()),
                }
                .send();
            }
            Err(err) => {
                tracing::info!("Instance {instance_id} failed: {err}");
                let _ = Message::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Exception(err.to_string()),
                }
                .send();
            }
        }
    }

    // =========================================================================
    // Server Instance Execution
    // =========================================================================

    async fn handle_server_request(
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        username: String,
        component: Component,
        arguments: Vec<String>,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> anyhow::Result<hyper::Response<HyperOutgoingBody>> {
        let inst_id = Uuid::new_v4();
        let (inst_state, _output_delivery_ctrl) =
            InstanceState::new(inst_id, username, arguments).await;

        let mut store = Store::new(&engine, inst_state);
        let (sender, receiver) = oneshot::channel();

        let req = store.data_mut().new_incoming_request(Scheme::Http, req)?;
        let out = store.data_mut().new_response_outparam(sender)?;

        let instance = linker
            .instantiate_async(&mut store, &component)
            .await
            .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

        let (_, serve_export) = instance
            .get_export(&mut store, None, "wasi:http/incoming-handler@0.2.4")
            .ok_or_else(|| RuntimeError::Other("No 'serve' function found".into()))?;

        let (_, handle_func_export) = instance
            .get_export(&mut store, Some(&serve_export), "handle")
            .ok_or_else(|| RuntimeError::Other("No 'handle' function found".into()))?;

        let handle_func = instance
            .get_typed_func::<(Resource<IncomingRequest>, Resource<ResponseOutparam>), ()>(
                &mut store,
                &handle_func_export,
            )
            .map_err(|e| RuntimeError::Other(format!("Failed to get 'handle' function: {e}")))?;

        let task = tokio::task::spawn(async move {
            if let Err(e) = handle_func.call_async(&mut store, (req, out)).await {
                eprintln!("error: {e:?}");
                return Err(e);
            }
            Ok(())
        });

        match receiver.await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => {
                let e = match task.await {
                    Ok(r) => r.expect_err("if the receiver has an error, the task must have failed"),
                    Err(e) => e.into(),
                };
                Err(e.context("guest never invoked `response-outparam::set` method"))
            }
        }
    }

    async fn launch_server(
        addr: SocketAddr,
        username: String,
        component: Component,
        arguments: Vec<String>,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        start_rx: oneshot::Receiver<()>,
    ) {
        let _ = start_rx.await;

        let result = async {
            let socket = tokio::net::TcpSocket::new_v4()?;
            socket.set_reuseaddr(!cfg!(windows))?;
            socket.bind(addr)?;
            let listener = socket.listen(100)?;
            eprintln!("Serving HTTP on http://{}/", listener.local_addr()?);

            tokio::task::spawn(async move {
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    let stream = TokioIo::new(stream);
                    let engine_ = engine.clone();
                    let linker_ = linker.clone();
                    let component_ = component.clone();
                    let arguments_ = arguments.clone();
                    let username_ = username.clone();
                    tokio::task::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(
                                stream,
                                hyper::service::service_fn(move |req| {
                                    Self::handle_server_request(
                                        engine_.clone(),
                                        linker_.clone(),
                                        username_.clone(),
                                        component_.clone(),
                                        arguments_.clone(),
                                        req,
                                    )
                                }),
                            )
                            .await
                        {
                            eprintln!("error: {e:?}");
                        }
                    });
                }
            });
            anyhow::Ok(())
        };
        if let Err(e) = result.await {
            eprintln!("error: {e}");
        }
    }
}


