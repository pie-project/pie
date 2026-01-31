use super::instance::{InstanceId, InstanceState, OutputDelivery, OutputDeliveryCtrl};
use super::service::{CommandDispatcher, Service};
use super::{api, server};
use crate::model;
use crate::model::request::QueryResponse;
use crate::service::ServiceCommand;
use dashmap::DashMap;
use hyper::server::conn::http1;
use pie_client::message;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
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
    #[error("No such program with wasm_hash={0}, toml_hash={1}")]
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
        wasm_hash: String,
        toml_hash: String,
        component: Component,
        event: oneshot::Sender<()>,
    },

    ProgramLoaded {
        wasm_hash: String,
        toml_hash: String,
        event: oneshot::Sender<bool>,
    },

    LaunchInstance {
        username: String,
        wasm_hash: String,
        toml_hash: String,
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
        wasm_hash: String,
        toml_hash: String,
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

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,
    linker: Arc<Linker<InstanceState>>,

    /// Pre-compiled WASM components, keyed by (wasm_hash, toml_hash)
    compiled_programs: DashMap<(String, String), Component>,

    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,

    /// Finished instances
    finished_instances: DashMap<InstanceId, InstanceHandle>,

    /// Running server instances
    running_server_instances: DashMap<InstanceId, InstanceHandle>,
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
    wasm_hash: String,
    toml_hash: String,
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
                wasm_hash,
                toml_hash,
                component,
                event,
            } => {
                // Store the pre-compiled component keyed by (wasm_hash, toml_hash)
                self.compiled_programs
                    .insert((wasm_hash, toml_hash), component);
                event.send(()).unwrap();
            }

            Command::ProgramLoaded {
                wasm_hash,
                toml_hash,
                event,
            } => {
                let is_loaded = self.compiled_programs.contains_key(&(wasm_hash, toml_hash));
                event.send(is_loaded).unwrap();
            }

            Command::LaunchInstance {
                username,
                wasm_hash,
                toml_hash,
                event,
                arguments,
                detached,
            } => {
                let res = self
                    .launch_instance(username, wasm_hash, toml_hash, arguments, detached)
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
                wasm_hash,
                toml_hash,
                port,
                arguments,
                event,
            } => {
                let _ = self
                    .launch_server_instance(username, wasm_hash, toml_hash, port, arguments)
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
                                    "Instance ID: {}, wasm_hash: {}, toml_hash: {}",
                                    item.key(),
                                    handle.wasm_hash,
                                    handle.toml_hash
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
                                let (wasm_hash, toml_hash) = item.key();
                                format!("wasm_hash: {}, toml_hash: {}", wasm_hash, toml_hash)
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

impl Runtime {
    fn new(engine: Engine) -> Self {
        let mut linker = Linker::<InstanceState>::new(&engine);

        // Add to linker
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();

        api::add_to_linker(&mut linker).unwrap();

        Self {
            engine,
            linker: Arc::new(linker),
            compiled_programs: DashMap::new(),
            running_instances: DashMap::new(),
            finished_instances: DashMap::new(),
            running_server_instances: DashMap::new(),
        }
    }

    fn get_component(&self, wasm_hash: &str, toml_hash: &str) -> Result<Component, RuntimeError> {
        // Get the component from memory by (wasm_hash, toml_hash)
        let key = (wasm_hash.to_string(), toml_hash.to_string());
        match self.compiled_programs.get(&key) {
            Some(entry) => Ok(entry.value().clone()),
            None => Err(RuntimeError::MissingProgram(
                wasm_hash.to_string(),
                toml_hash.to_string(),
            )),
        }
    }

    /// Actually start a program instance
    async fn launch_instance(
        &self,
        username: String,
        wasm_hash: String,
        toml_hash: String,
        arguments: Vec<String>,
        detached: bool,
    ) -> Result<InstanceId, RuntimeError> {
        let component = self.get_component(&wasm_hash, &toml_hash)?;
        let instance_id = Uuid::new_v4();

        // Instantiate and run in a task
        let engine = self.engine.clone();
        let linker = self.linker.clone();

        // Create a oneshot channel to signal when the task can start
        let (start_tx, start_rx) = oneshot::channel();
        // Create a oneshot channel to receive the output delivery controller
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
            wasm_hash,
            toml_hash,
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
        wasm_hash: String,
        toml_hash: String,
        port: u32,
        arguments: Vec<String>,
    ) -> Result<InstanceId, RuntimeError> {
        let instance_id = Uuid::new_v4();
        let component = self.get_component(&wasm_hash, &toml_hash)?;

        // Instantiate and run in a task
        let engine = self.engine.clone();
        let linker = self.linker.clone();
        let addr = SocketAddr::from(([127, 0, 0, 1], port as u16));

        // Create a oneshot channel to signal when the task can start
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

        // Create a dummy output delivery controller for server instances (not used since each request gets its own instance)
        let (dummy_state, output_delivery_ctrl) =
            InstanceState::new(Uuid::new_v4(), username.clone(), vec![]).await;
        drop(dummy_state); // We don't actually use this

        // Record in the "running_instances" so we can manage it later
        let instance_handle = InstanceHandle {
            username,
            wasm_hash,
            toml_hash,
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
        arguments: Vec<String>,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
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
            //store.data_mut().w
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
        // Create the instance state and output delivery controller
        let (inst_state, output_delivery_ctrl) =
            InstanceState::new(instance_id, username, arguments).await;

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

            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

            // Attempt to call "run"
            let (_, run_export) = instance
                .get_export(&mut store, None, "inferlet:core/run")
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

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
