use super::instance::{InstanceId, InstanceState, OutputDelivery, OutputDeliveryCtrl};
use super::service::{CommandDispatcher, Service};
use super::{api, server};
use crate::model;
use crate::model::request::QueryResponse;
use crate::service::ServiceCommand;
use dashmap::DashMap;
use hyper::server::conn::http1;
use pie_client::message;
use std::collections::HashSet;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::component::Resource;
use wasmtime::{Engine, Module, Store, component::Component, component::Linker};
use wasmtime_wasi_http::WasiHttpView;
use wasmtime_wasi_http::bindings::exports::wasi::http::incoming_handler::{
    IncomingRequest, ResponseOutparam,
};
use wasmtime_wasi_http::bindings::http::types::Scheme;
use wasmtime_wasi_http::body::HyperOutgoingBody;
use wasmtime_wasi_http::io::TokioIo;

mod dynamic_linking;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The sender of the command channel, which is used to send commands to the
/// handler task.
static COMMAND_DISPATCHER: OnceLock<CommandDispatcher<Command>> = OnceLock::new();

/// Starts the runtime service. A daemon task will be spawned to handle the
/// commands dispatched from other services.
pub fn start_service(engine: Engine) {
    let runtime = Runtime::new(engine);
    runtime.start(&COMMAND_DISPATCHER);
}

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

pub enum Command {
    GetVersion {
        event: oneshot::Sender<String>,
    },

    LoadProgram {
        program_hash: ProgramHash,
        component: Component,
        dependencies: Vec<ProgramHash>,
        event: oneshot::Sender<()>,
    },

    ProgramLoaded {
        program_hash: ProgramHash,
        event: oneshot::Sender<bool>,
    },

    LaunchInstance {
        username: String,
        program_name: String,
        program_hash: ProgramHash,
        arguments: Vec<String>,
        detached: bool,
        event: oneshot::Sender<Result<InstanceId, RuntimeError>>,
    },

    AttachInstance {
        inst_id: InstanceId,
        event: oneshot::Sender<AttachInstanceResult>,
    },

    DetachInstance {
        inst_id: InstanceId,
    },

    AllowOutput {
        inst_id: InstanceId,
    },

    LaunchServerInstance {
        username: String,
        program_hash: ProgramHash,
        port: u32,
        arguments: Vec<String>,
        event: oneshot::Sender<Result<(), RuntimeError>>,
    },

    TerminateInstance {
        inst_id: InstanceId,
        notification_to_client: Option<TerminationCause>,
    },

    FinishInstance {
        inst_id: InstanceId,
        cause: TerminationCause,
    },

    SetOutputDelivery {
        inst_id: InstanceId,
        mode: OutputDelivery,
    },

    DebugQuery {
        query: String,
        event: oneshot::Sender<QueryResponse>,
    },

    ListInstances {
        username: String,
        event: oneshot::Sender<Vec<message::InstanceInfo>>,
    },
}

impl ServiceCommand for Command {
    const DISPATCHER: &'static OnceLock<CommandDispatcher<Self>> = &COMMAND_DISPATCHER;
}

/// A key identifying a compiled program by its WASM and manifest hashes.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct ProgramHash {
    wasm_hash: String,
    manifest_hash: String,
}

impl ProgramHash {
    pub fn new(wasm_hash: String, manifest_hash: String) -> Self {
        Self {
            wasm_hash,
            manifest_hash,
        }
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

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,

    /// Pre-compiled WASM components, keyed by ProgramHash
    compiled_programs: DashMap<ProgramHash, CompiledProgram>,

    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,

    /// Finished instances
    finished_instances: DashMap<InstanceId, InstanceHandle>,

    /// Running server instances
    running_server_instances: DashMap<InstanceId, InstanceHandle>,

    /// Shared core modules (e.g. CPython interpreter) loaded from py-runtime/shared/
    shared_modules: Arc<Vec<(String, Module)>>,

    /// Path to the py-runtime directory (~/.pie/py-runtime), if it exists
    py_runtime_dir: Option<PathBuf>,
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
    /// The output will be streamed to the client.
    ///
    /// Note that this enum does not directly affect the output streaming behavior.
    /// The output streaming behavior is determined by the output delivery mode.
    /// The `Attached` state merely prevents other clients from attaching to the same
    /// instance. Once an instance is set to `Attached`, the output delivery mode must
    /// be set to `Streamed` via the `SetOutputDelivery` command.
    Attached,
    /// The instance is running and not attached to a client.
    /// The output will be buffered.
    ///
    /// Note that this enum does not directly affect the output streaming behavior.
    /// The output streaming behavior is determined by the output delivery mode.
    /// The `Detached` state merely indicates that the instance is not attached to a
    /// client and other clients can attach to it. Before setting the instance to
    /// `Detached`, the output delivery mode must be set to `Buffered` via the
    /// `SetOutputDelivery` command, so to prevent the output from being lost.
    Detached,
    /// The instance has finished execution.
    /// The output is buffered and waiting to be streamed to the client.
    Finished(TerminationCause),
}

impl From<InstanceRunningState> for message::InstanceStatus {
    fn from(state: InstanceRunningState) -> Self {
        match state {
            InstanceRunningState::Attached => message::InstanceStatus::Attached,
            InstanceRunningState::Detached => message::InstanceStatus::Detached,
            InstanceRunningState::Finished(_) => message::InstanceStatus::Finished,
        }
    }
}

#[derive(Clone)]
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

struct InstanceHandle {
    username: String,
    program_hash: ProgramHash,
    arguments: Vec<String>,
    start_time: std::time::Instant,
    output_delivery_ctrl: OutputDeliveryCtrl,
    running_state: InstanceRunningState,
    join_handle: tokio::task::JoinHandle<()>,
}

impl Service for Runtime {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::LoadProgram {
                program_hash,
                component,
                dependencies,
                event,
            } => {
                // Store the pre-compiled component with dependencies keyed by ProgramHash
                let compiled_program = CompiledProgram {
                    component,
                    dependencies,
                };
                self.compiled_programs
                    .insert(program_hash, compiled_program);
                event.send(()).unwrap();
            }

            Command::ProgramLoaded {
                program_hash,
                event,
            } => {
                let is_loaded = self.compiled_programs.contains_key(&program_hash);
                event.send(is_loaded).unwrap();
            }

            Command::LaunchInstance {
                username,
                program_name,
                program_hash,
                event,
                arguments,
                detached,
            } => {
                let res = self
                    .launch_instance(username, program_name, program_hash, arguments, detached)
                    .await;
                event
                    .send(res)
                    .map_err(|_| "Failed to send instance ID after launching instance")
                    .unwrap();
            }

            Command::AttachInstance { inst_id, event } => {
                let res = self.attach_instance(inst_id);
                event
                    .send(res)
                    .map_err(|_| "Failed to send attach instance result")
                    .unwrap();
            }

            Command::DetachInstance { inst_id } => {
                self.detach_instance(inst_id);
            }

            Command::AllowOutput { inst_id } => {
                self.allow_output(inst_id);
            }

            Command::LaunchServerInstance {
                username,
                program_hash,
                port,
                arguments,
                event,
            } => {
                let _ = self
                    .launch_server_instance(username, program_hash, port, arguments)
                    .await;
                event.send(Ok(())).unwrap();
            }

            Command::TerminateInstance {
                inst_id,
                notification_to_client,
            } => {
                self.terminate_instance(inst_id, notification_to_client);
            }

            Command::FinishInstance { inst_id, cause } => {
                self.finish_instance(inst_id, cause);
            }

            Command::SetOutputDelivery { inst_id, mode } => {
                self.set_output_delivery(inst_id, mode);
            }

            Command::GetVersion { event } => {
                event.send(VERSION.to_string()).unwrap();
            }

            Command::DebugQuery { query, event } => {
                let res = match query.as_str() {
                    "ping" => {
                        format!("pong")
                    }
                    "get_instance_count" => {
                        format!("{}", self.running_instances.len())
                    }
                    "get_server_instance_count" => {
                        format!("{}", self.running_server_instances.len())
                    }
                    // Add the new queries here
                    "list_running_instances" => {
                        let instances: Vec<String> = self
                            .running_instances
                            .iter()
                            .map(|item| {
                                let handle = item.value();
                                format!(
                                    "Instance ID: {}, wasm_hash: {}, manifest_hash: {}",
                                    item.key(),
                                    handle.program_hash.wasm_hash,
                                    handle.program_hash.manifest_hash
                                )
                            })
                            .collect();

                        format!("{}", instances.join("\n"))
                    }
                    "list_in_memory_programs" => {
                        let programs: Vec<String> = self
                            .compiled_programs
                            .iter()
                            .map(|item| {
                                let key = item.key();
                                format!(
                                    "wasm_hash: {}, manifest_hash: {}",
                                    key.wasm_hash, key.manifest_hash
                                )
                            })
                            .collect();

                        format!("{}", programs.join("\n"))
                    }

                    _ => {
                        format!("Unknown query: {}", query)
                    }
                };

                event.send(QueryResponse { value: res }).unwrap();
            }
            Command::ListInstances { username, event } => {
                // Internal users (from monitor) can see all instances
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
                        kv_pages_used: 0, // TODO: query from resource_manager
                    })
                    .collect();

                // Sort by elapsed time (most recent first) and limit to 50 for performance
                instances.sort_by(|a, b| a.elapsed_secs.cmp(&b.elapsed_secs));
                instances.truncate(50);

                event.send(instances).unwrap();
            }
        }
    }
}

/// Loads shared core modules (.wasm files) from a directory.
fn load_shared_modules(engine: &Engine, shared_dir: &Path) -> Vec<(String, Module)> {
    let mut modules = Vec::new();
    let entries = match fs::read_dir(shared_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                "Failed to read shared modules dir {}: {e}",
                shared_dir.display()
            );
            return modules;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to read shared module entry: {e}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "wasm") {
            let import_name = path.file_stem().unwrap().to_str().unwrap().to_string();
            tracing::info!(
                "Loading shared module: {} -> {}",
                path.display(),
                import_name
            );
            match Module::from_file(engine, &path) {
                Ok(module) => modules.push((import_name, module)),
                Err(e) => tracing::error!("Failed to load shared module {}: {e}", path.display()),
            }
        }
    }
    modules
}

/// Creates a new linker with WASI and API bindings configured.
fn create_linker(engine: &Engine, shared_modules: &[(String, Module)]) -> Linker<InstanceState> {
    let mut linker = Linker::<InstanceState>::new(engine);

    // Add WASI and HTTP bindings
    wasmtime_wasi::p2::add_to_linker_async(&mut linker)
        .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
        .unwrap();
    wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
        .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
        .unwrap();

    // Add custom API bindings
    api::add_to_linker(&mut linker).unwrap();

    // Register shared core modules (e.g. CPython interpreter)
    for (name, module) in shared_modules {
        linker
            .root()
            .module(name, module)
            .unwrap_or_else(|e| panic!("Failed to register shared module '{name}': {e}"));
    }

    linker
}

impl Runtime {
    fn new(engine: Engine) -> Self {
        let py_runtime_dir = {
            let dir = crate::path::get_py_runtime_dir();
            if dir.is_dir() {
                tracing::info!("Python runtime directory: {}", dir.display());
                Some(dir)
            } else {
                tracing::info!("No Python runtime directory found at {}", dir.display());
                None
            }
        };

        let shared_modules = if let Some(ref dir) = py_runtime_dir {
            let shared_dir = dir.join("shared");
            if shared_dir.is_dir() {
                load_shared_modules(&engine, &shared_dir)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        if !shared_modules.is_empty() {
            tracing::info!("Loaded {} shared core module(s)", shared_modules.len());
        }

        Self {
            engine,
            compiled_programs: DashMap::new(),
            running_instances: DashMap::new(),
            finished_instances: DashMap::new(),
            running_server_instances: DashMap::new(),
            shared_modules: Arc::new(shared_modules),
            py_runtime_dir,
        }
    }

    fn get_component(&self, program_hash: &ProgramHash) -> Result<Component, RuntimeError> {
        // Get the component from memory by ProgramHash
        match self.compiled_programs.get(program_hash) {
            Some(entry) => Ok(entry.value().component.clone()),
            None => Err(RuntimeError::MissingProgram(
                program_hash.wasm_hash.clone(),
                program_hash.manifest_hash.clone(),
            )),
        }
    }

    /// Collect all dependencies of a program in topological order (dependencies before dependents).
    /// This handles deduplication when a dependency appears multiple times in the dependency graph.
    fn collect_dependencies_topo_order(&self, program_hash: &ProgramHash) -> Vec<Component> {
        /// Recursively collect a dependency and all its transitive dependencies.
        /// Uses post-order DFS to ensure dependencies come before dependents.
        fn collect_recursive(
            compiled_programs: &DashMap<ProgramHash, CompiledProgram>,
            dep_hash: &ProgramHash,
            visited: &mut HashSet<ProgramHash>,
            result: &mut Vec<Component>,
        ) {
            // Skip if already visited (deduplication)
            if visited.contains(dep_hash) {
                return;
            }
            visited.insert(dep_hash.clone());

            if let Some(entry) = compiled_programs.get(dep_hash) {
                let compiled_program = entry.value();

                // First, recursively process all transitive dependencies (post-order DFS)
                for child_dep_hash in &compiled_program.dependencies {
                    collect_recursive(compiled_programs, child_dep_hash, visited, result);
                }

                // Then add this component (after all its dependencies are added)
                result.push(compiled_program.component.clone());
            }
        }

        let mut visited = HashSet::new();
        let mut result = Vec::new();

        // Get the direct dependencies of the main program and recursively collect them
        if let Some(entry) = self.compiled_programs.get(program_hash) {
            for dep_hash in &entry.value().dependencies {
                collect_recursive(&self.compiled_programs, dep_hash, &mut visited, &mut result);
            }
        }

        result
    }

    /// Actually start a program instance
    async fn launch_instance(
        &self,
        username: String,
        program_name: String,
        program_hash: ProgramHash,
        arguments: Vec<String>,
        detached: bool,
    ) -> Result<InstanceId, RuntimeError> {
        let component = self.get_component(&program_hash)?;
        let instance_id = Uuid::new_v4();

        // Collect dependencies in topological order (deduplication handled inside)
        let dependency_components = self.collect_dependencies_topo_order(&program_hash);

        // Instantiate and run in a task
        let engine = self.engine.clone();

        // Create a oneshot channel to signal when the task can start
        let (start_tx, start_rx) = oneshot::channel();
        // Create a oneshot channel to receive the output delivery controller
        let (output_delivery_ctrl_tx, output_delivery_ctrl_rx) = oneshot::channel();

        let join_handle = tokio::spawn(Self::launch(
            instance_id,
            username.clone(),
            program_name,
            component,
            dependency_components,
            arguments.clone(),
            detached,
            engine,
            self.shared_modules.clone(),
            self.py_runtime_dir.clone(),
            start_rx,
            output_delivery_ctrl_tx,
        ));

        // Wait for the output delivery controller to be sent back
        let output_delivery_ctrl = output_delivery_ctrl_rx.await.unwrap();

        let running_state = if detached {
            InstanceRunningState::Detached
        } else {
            InstanceRunningState::Attached
        };

        // Record in the "running_instances" so we can manage it later
        let instance_handle = InstanceHandle {
            username,
            program_hash,
            arguments,
            start_time: std::time::Instant::now(),
            output_delivery_ctrl,
            running_state,
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        // Signal the task to start now that the join_handle is in the map
        let _ = start_tx.send(());

        Ok(instance_id)
    }

    /// Set the instance as attached. Its output will be streamed to the client.
    fn attach_instance(&self, inst_id: InstanceId) -> AttachInstanceResult {
        // Check if the instance is still running.
        if let Some(mut handle) = self.running_instances.get_mut(&inst_id) {
            // An instance cannot be attached if it is already attached.
            if let InstanceRunningState::Attached = handle.running_state {
                return AttachInstanceResult::AlreadyAttached;
            }

            handle.running_state = InstanceRunningState::Attached;
            return AttachInstanceResult::AttachedRunning;
        }

        // Check if the instance has finished execution.
        if let Some(mut handle) = self.finished_instances.get_mut(&inst_id) {
            // Set the instance to attached to prevent other clients from attaching to it.
            // Take the termination cause from the finished state and return it. The code that
            // takes the termination cause should be responsible for later cleaning up the instance.
            if matches!(&handle.running_state, InstanceRunningState::Finished(_)) {
                if let InstanceRunningState::Finished(cause) =
                    std::mem::replace(&mut handle.running_state, InstanceRunningState::Attached)
                {
                    return AttachInstanceResult::AttachedFinished(cause);
                }
            }
        }

        return AttachInstanceResult::InstanceNotFound;
    }

    /// Set the instance as detached. Prior to calling this method, the instance output delivery
    /// mode must be set to buffered.
    fn detach_instance(&self, inst_id: InstanceId) {
        if let Some(mut handle) = self.running_instances.get_mut(&inst_id) {
            handle.running_state = InstanceRunningState::Detached;
        }
    }

    /// Allow output for a running instance
    fn allow_output(&self, inst_id: InstanceId) {
        if let Some(handle) = self.running_instances.get(&inst_id) {
            handle.output_delivery_ctrl.allow_output();
        }
    }

    /// Actually start a program instance
    async fn launch_server_instance(
        &self,
        username: String,
        program_hash: ProgramHash,
        port: u32,
        arguments: Vec<String>,
    ) -> Result<InstanceId, RuntimeError> {
        let instance_id = Uuid::new_v4();
        let component = self.get_component(&program_hash)?;

        // Collect dependencies in topological order (deduplication handled inside)
        let dependency_components = self.collect_dependencies_topo_order(&program_hash);

        // Instantiate and run in a task
        let engine = self.engine.clone();
        let addr = SocketAddr::from(([127, 0, 0, 1], port as u16));

        // Create a oneshot channel to signal when the task can start
        let (start_tx, start_rx) = oneshot::channel();

        let join_handle = tokio::spawn(Self::launch_server(
            addr,
            username.clone(),
            component,
            dependency_components,
            arguments.clone(),
            engine,
            self.shared_modules.clone(),
            self.py_runtime_dir.clone(),
            start_rx,
        ));

        // Create a dummy output delivery controller for server instances (not used since each request gets its own instance)
        let (dummy_state, output_delivery_ctrl) =
            InstanceState::new(Uuid::new_v4(), username.clone(), vec![], None).await;
        drop(dummy_state); // We don't actually use this

        // Record in the "running_instances" so we can manage it later
        let instance_handle = InstanceHandle {
            username,
            program_hash,
            arguments,
            start_time: std::time::Instant::now(),
            output_delivery_ctrl,
            running_state: InstanceRunningState::Detached,
            join_handle,
        };
        self.running_server_instances
            .insert(instance_id, instance_handle);

        // Signal the task to start now that the join_handle is in the map
        let _ = start_tx.send(());

        Ok(instance_id)
    }

    /// Terminate a running instance, and optionally notify the client.
    fn terminate_instance(
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

            model::cleanup_instance(instance_id.clone());

            if let Some(cause) = notification_to_client {
                server::InstanceEvent::Terminate {
                    inst_id: instance_id,
                    cause,
                }
                .dispatch();
            }
        }
    }

    /// Finish a running instance. If the instance is attached, it will notify the client and
    /// clean up the instance. If the instance is detached, it will mark the instance as finished
    /// and add it to the finished instances map. If the instance is already finished, it will
    /// panic.
    fn finish_instance(&self, instance_id: InstanceId, cause: TerminationCause) {
        if let Some((_, mut handle)) = self.running_instances.remove(&instance_id) {
            match handle.running_state {
                // For an attached instance, its output must have been streamed to the client,
                // so we can clean up the instance and notify the client about the termination.
                InstanceRunningState::Attached => {
                    handle.join_handle.abort();
                    model::cleanup_instance(instance_id.clone());

                    server::InstanceEvent::Terminate {
                        inst_id: instance_id,
                        cause,
                    }
                    .dispatch();
                }
                // For a detached instance, we can just mark it as finished and add it to the
                // finished instances map. Its output is buffered and waiting to be streamed to
                //the client.
                InstanceRunningState::Detached => {
                    handle.running_state = InstanceRunningState::Finished(cause);
                    self.finished_instances.insert(instance_id, handle);
                }
                // If the instance is already finished, we can't finish it again.
                // This should never happen.
                InstanceRunningState::Finished(_) => {
                    panic!("Instance {instance_id} is already finished and cannot be sealed again")
                }
            }
        }
    }

    /// Set the output delivery for a running instance
    fn set_output_delivery(&self, instance_id: InstanceId, output_delivery: OutputDelivery) {
        if let Some(handle) = self.running_instances.get(&instance_id) {
            handle
                .output_delivery_ctrl
                .set_output_delivery(output_delivery);
        }

        if let Some(handle) = self.finished_instances.get(&instance_id) {
            handle
                .output_delivery_ctrl
                .set_output_delivery(output_delivery);
        }
    }

    async fn handle_server_request(
        engine: Engine,
        username: String,
        component: Component,
        dependency_components: Vec<Component>,
        arguments: Vec<String>,
        shared_modules: Arc<Vec<(String, Module)>>,
        py_runtime_dir: Option<PathBuf>,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> anyhow::Result<hyper::Response<HyperOutgoingBody>> {
        let inst_id = Uuid::new_v4();
        let (inst_state, _output_delivery_ctrl) =
            InstanceState::new(inst_id, username, arguments, py_runtime_dir.as_deref()).await;

        let mut store = Store::new(&engine, inst_state);
        let (sender, receiver) = oneshot::channel();

        let req = store.data_mut().new_incoming_request(Scheme::Http, req)?;
        let out = store.data_mut().new_response_outparam(sender)?;

        let mut linker = create_linker(&engine, &shared_modules);

        // Instantiate dependencies and register their exports in the linker
        dynamic_linking::instantiate_libraries(
            &engine,
            &mut linker,
            &mut store,
            dependency_components,
        )
        .await
        .map_err(|e| anyhow::anyhow!("Failed to instantiate dependencies: {e}"))?;

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
                    Ok(r) => {
                        r.expect_err("if the receiver has an error, the task must have failed")
                    }
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
        dependency_components: Vec<Component>,
        arguments: Vec<String>,
        engine: Engine,
        shared_modules: Arc<Vec<(String, Module)>>,
        py_runtime_dir: Option<PathBuf>,
        start_rx: oneshot::Receiver<()>,
    ) {
        // Wait for the signal to start
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
                    let component_ = component.clone();
                    let dependency_components_ = dependency_components.clone();
                    let arguments_ = arguments.clone();
                    let username_ = username.clone();
                    let shared_modules_ = shared_modules.clone();
                    let py_runtime_dir_ = py_runtime_dir.clone();
                    tokio::task::spawn(async move {
                        if let Err(e) = http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(
                                stream,
                                hyper::service::service_fn(move |req| {
                                    Self::handle_server_request(
                                        engine_.clone(),
                                        username_.clone(),
                                        component_.clone(),
                                        dependency_components_.clone(),
                                        arguments_.clone(),
                                        shared_modules_.clone(),
                                        py_runtime_dir_.clone(),
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

    async fn launch(
        instance_id: InstanceId,
        username: String,
        program_name: String,
        component: Component,
        dependency_components: Vec<Component>,
        arguments: Vec<String>,
        detached: bool,
        engine: Engine,
        shared_modules: Arc<Vec<(String, Module)>>,
        py_runtime_dir: Option<PathBuf>,
        start_rx: oneshot::Receiver<()>,
        output_delivery_ctrl_tx: oneshot::Sender<OutputDeliveryCtrl>,
    ) {
        // Create the instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username, arguments, py_runtime_dir.as_deref()).await;

        let output_delivery = if detached {
            OutputDelivery::Buffered
        } else {
            OutputDelivery::Streamed
        };

        // Set the initial output delivery mode
        output_delivery_ctrl.set_output_delivery(output_delivery);

        // Send the output delivery controller back before starting
        output_delivery_ctrl_tx
            .send(output_delivery_ctrl)
            .map_err(|_| "Failed to send output delivery controller")
            .unwrap();

        // Wait for the signal to start
        start_rx.await.unwrap();

        // Wrap everything in a closure returning a Result,
        // so we can capture errors more systematically if desired:
        let result = async {
            let mut store = Store::new(&engine, inst_state);

            let mut linker = create_linker(&engine, &shared_modules);

            // Instantiate dependencies and register their exports in the linker
            dynamic_linking::instantiate_libraries(
                &engine,
                &mut linker,
                &mut store,
                dependency_components,
            )
            .await?;

            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

            // Attempt to call "run"
            let run_interface = format!("pie:{}/run", program_name);
            let (_, run_export) = instance
                .get_export(&mut store, None, &run_interface)
                .ok_or_else(|| {
                    RuntimeError::Other(format!("No '{}' export found", run_interface))
                })?;

            let (_, run_func_export) = instance
                .get_export(&mut store, Some(&run_export), "run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let run_func = instance
                .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                .map_err(|e| RuntimeError::Other(format!("Failed to get 'run' function: {e}")))?;

            return match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    let return_value = store.data().return_value();
                    //println!("Instance {instance_id} finished normally");
                    Ok(return_value)
                }
                Ok((Err(runtime_err),)) => {
                    //eprintln!("Instance {instance_id} returned an error");
                    Err(RuntimeError::Other(runtime_err))
                }
                Err(call_err) => {
                    //eprintln!("Instance {instance_id} call error: {call_err}");
                    Err(RuntimeError::Other(format!("Call error: {call_err}")))
                }
            };
        }
        .await;

        match result {
            Ok(return_value) => {
                Command::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Normal(return_value.unwrap_or_default()),
                }
                .dispatch();
            }
            Err(err) => {
                tracing::info!("Instance {instance_id} failed: {err}");
                Command::FinishInstance {
                    inst_id: instance_id,
                    cause: TerminationCause::Exception(err.to_string()),
                }
                .dispatch();
            }
        }
    }
}
