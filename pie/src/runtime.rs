use crate::instance::{InstanceId, InstanceState};
use crate::model::request::QueryResponse;
use crate::service::{Service, ServiceError};
use crate::{api, model, server, service};
use bytes::Bytes;
use dashmap::DashMap;
use hyper::server::conn::http1;
use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::sync::oneshot;
use uuid::Uuid;
use wasmtime::component::Resource;
use wasmtime::{
    Config, Engine, InstanceAllocationStrategy, PoolingAllocationConfig, Store,
    component::Component, component::Linker,
};
use wasmtime_wasi_http::WasiHttpView;
use wasmtime_wasi_http::bindings::exports::wasi::http::incoming_handler::{
    IncomingRequest, ResponseOutparam,
};
use wasmtime_wasi_http::bindings::http::types::Scheme;
use wasmtime_wasi_http::body::HyperOutgoingBody;
use wasmtime_wasi_http::io::TokioIo;

const VERSION: &str = env!("CARGO_PKG_VERSION");

static SERVICE_ID_RUNTIME: OnceLock<usize> = OnceLock::new();

pub fn trap(instance_id: InstanceId, cause: TerminationCause) {
    Command::Trap {
        inst_id: instance_id,
        cause,
    }
    .dispatch()
    .unwrap();
}

pub fn trap_exception<T>(instance_id: InstanceId, exception: T)
where
    T: ToString,
{
    Command::Trap {
        inst_id: instance_id,
        cause: TerminationCause::Exception(exception.to_string()),
    }
    .dispatch()
    .unwrap();
}

#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Wrap general I/O errors
    #[error("I/O error occurred: {0}")]
    Io(#[from] std::io::Error),

    /// Wrap Wasmtime errors
    #[error("Wasmtime error occurred: {0}")]
    Wasmtime(#[from] wasmtime::Error),

    /// No program found for a given hash
    #[error("No such program with hash={0}")]
    MissingProgram(String),

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

#[derive(Debug)]
pub enum Command {
    GetVersion {
        event: oneshot::Sender<String>,
    },

    ProgramExists {
        hash: String,
        event: oneshot::Sender<bool>,
    },

    UploadProgram {
        hash: String,
        raw: Vec<u8>,
        event: oneshot::Sender<Result<String, RuntimeError>>,
    },

    LaunchInstance {
        program_hash: String,
        arguments: Vec<String>,
        event: oneshot::Sender<Result<InstanceId, RuntimeError>>,
    },

    LaunchServerInstance {
        program_hash: String,
        port: u32,
        arguments: Vec<String>,
        event: oneshot::Sender<Result<(), RuntimeError>>,
    },

    Trap {
        inst_id: InstanceId,
        cause: TerminationCause,
    },

    Warn {
        inst_id: InstanceId,
        message: String,
    },

    DebugQuery {
        query: String,
        event: oneshot::Sender<QueryResponse>,
    },
}

impl Command {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        let service_id =
            *SERVICE_ID_RUNTIME.get_or_init(move || service::get_service_id("runtime").unwrap());

        service::dispatch(service_id, self)
    }
}

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
pub struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,
    linker: Arc<Linker<InstanceState>>,

    cache_dir: std::path::PathBuf,

    /// Pre-compiled WASM components, keyed by BLAKE3 hex string
    programs_in_memory: DashMap<String, Component>,

    /// Paths to compiled modules on disk
    programs_in_disk: DashMap<String, std::path::PathBuf>,

    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,

    /// Running server instances
    running_server_instances: DashMap<InstanceId, InstanceHandle>,
}

#[derive(Debug, Clone)]
pub enum TerminationCause {
    Normal,
    Signal,
    Exception(String),
    SystemError(String),
    OutOfResources(String),
}

pub struct InstanceHandle {
    pub hash: String,
    //pub to_origin: Sender<ServerMessage>,
    // pub evt_from_system: Sender<String>,
    // pub evt_from_origin: Sender<String>,
    // pub evt_from_peers: Sender<(String, String)>,
    pub join_handle: tokio::task::JoinHandle<()>,
}
//#[async_trait]
impl Service for Runtime {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::ProgramExists { hash, event } => {
                let exists = self.programs_in_memory.contains_key(&hash)
                    || self.programs_in_disk.contains_key(&hash);
                event.send(exists).unwrap();
            }

            Command::UploadProgram { hash, raw, event } => {
                if self.programs_in_memory.contains_key(&hash) {
                    event.send(Ok(hash)).unwrap();
                } else if let Ok(component) = Component::from_binary(&self.engine, raw.as_slice()) {
                    self.programs_in_memory.insert(hash.to_string(), component);

                    // Write to disk
                    let file_path = std::path::Path::new(&self.cache_dir).join(&hash);
                    std::fs::write(&file_path, &raw).unwrap();
                    self.programs_in_disk.insert(hash.clone(), file_path);
                    event.send(Ok(hash)).unwrap();
                } else {
                    event
                        .send(Err(RuntimeError::Other("Failed to compile".into())))
                        .unwrap();
                }
            }

            Command::LaunchInstance {
                program_hash: hash,
                event,
                arguments,
            } => {
                let instance_id = self.launch_instance(&hash, arguments).await.unwrap();
                event.send(Ok(instance_id)).unwrap();
            }

            Command::LaunchServerInstance {
                program_hash: hash,
                port,
                arguments,
                event,
            } => {
                let _ = self.launch_server_instance(&hash, port, arguments).await;
                event.send(Ok(())).unwrap();
            }

            Command::Trap { inst_id, cause } => {
                self.terminate_instance(inst_id, cause).await;
            }

            Command::Warn { inst_id, message } => server::Command::Send {
                inst_id,
                message: message.clone(),
            }
            .dispatch()
            .unwrap(),
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
                                format!(
                                    "Instance ID: {}, Program Hash: {}",
                                    item.key(),
                                    item.value().hash
                                )
                            })
                            .collect();

                        format!("{}", instances.join("\n"))
                    }
                    "list_in_memory_programs" => {
                        let keys: Vec<String> = self
                            .programs_in_memory
                            .iter()
                            .map(|item| item.key().clone())
                            .collect();

                        format!("{}", keys.join("\n"))
                    }

                    _ => {
                        format!("Unknown query: {}", query)
                    }
                };

                event.send(QueryResponse { value: res }).unwrap();
            }
        }
    }
}

impl Runtime {
    pub fn new<P: AsRef<std::path::Path>>(cache_dir: P) -> Self {
        // Configure Wasmtime engine
        let mut config = Config::default();
        config.async_support(true);

        let mut pooling_config = PoolingAllocationConfig::default();

        // TODO: Adjust settings later: https://docs.wasmtime.dev/api/wasmtime/struct.PoolingAllocationConfig.html

        //config.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling_config));

        let engine = Engine::new(&config).unwrap();

        let mut linker = Linker::<InstanceState>::new(&engine);

        // Add to linker
        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();

        api::add_to_linker(&mut linker).unwrap();

        let cache_dir = cache_dir.as_ref().join("programs");
        // Ensure the cache directory exists
        std::fs::create_dir_all(&cache_dir).expect("Failed to create cache directory");

        Self {
            engine,
            linker: Arc::new(linker),
            cache_dir,
            programs_in_memory: DashMap::new(),
            programs_in_disk: DashMap::new(),
            running_instances: DashMap::new(),
            running_server_instances: DashMap::new(),
        }
    }

    pub fn load_existing_programs(&self) -> Result<(), RuntimeError> {
        let entries = std::fs::read_dir(&self.cache_dir)?; // Will map to RuntimeError::Io automatically
        for entry in entries {
            let entry = entry?; // same here, auto-converted to RuntimeError::Io
            if entry.file_type()?.is_file() {
                let path = entry.path();
                let data = std::fs::read(&path)?; // also auto Io
                let hash = blake3::hash(&data).to_hex().to_string();
                self.programs_in_disk.insert(hash, path);
            }
        }
        Ok(())
    }

    fn get_component(&self, hash: &str) -> Result<Component, RuntimeError> {
        // 1) Make sure the `Component` is loaded in memory
        if self.programs_in_memory.get(hash).is_none() {
            // load from disk if possible
            if let Some(path_entry) = self.programs_in_disk.get(hash) {
                // Use a custom error variant for compile errors

                let component =
                    Component::from_file(&self.engine, path_entry.value()).map_err(|err| {
                        RuntimeError::CompileWasm {
                            path: path_entry.value().to_path_buf(),
                            source: err,
                        }
                    })?;
                self.programs_in_memory.insert(hash.to_string(), component);
            } else {
                // If not on disk either, return a custom error
                return Err(RuntimeError::MissingProgram(hash.to_string()));
            }
        }

        // 2) Now we have a compiled component
        let component = match self.programs_in_memory.get(hash) {
            Some(c) => c.clone(),
            None => {
                return Err(RuntimeError::Other(
                    "Failed to get component from memory".into(),
                ));
            }
        };

        Ok(component)
    }

    /// Actually start a program instance
    pub async fn launch_instance(
        &self,
        hash: &str,
        arguments: Vec<String>,
    ) -> Result<InstanceId, RuntimeError> {
        let component = self.get_component(hash)?;

        let instance_id = Uuid::new_v4();

        // Instantiate and run in a task
        let engine = self.engine.clone();
        let linker = self.linker.clone();

        let join_handle = tokio::spawn(Self::launch(
            instance_id,
            component,
            arguments,
            engine,
            linker,
        ));

        // Record in the “running_instances” so we can manage it later
        let instance_handle = InstanceHandle {
            hash: hash.to_string(),
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        Ok(instance_id)
    }

    /// Actually start a program instance
    pub async fn launch_server_instance(
        &self,
        hash: &str,
        port: u32,
        arguments: Vec<String>,
    ) -> Result<InstanceId, RuntimeError> {
        let instance_id = Uuid::new_v4();
        let component = self.get_component(hash)?;

        // Instantiate and run in a task
        let engine = self.engine.clone();
        let linker = self.linker.clone();
        let addr = SocketAddr::from(([127, 0, 0, 1], port as u16));

        let join_handle = tokio::spawn(Self::launch_server(
            addr, component, arguments, engine, linker,
        ));

        // Record in the “running_instances” so we can manage it later
        let instance_handle = InstanceHandle {
            hash: hash.to_string(),
            join_handle,
        };
        self.running_server_instances
            .insert(instance_id, instance_handle);

        Ok(instance_id)
    }

    /// Terminate (abort) a running instance
    pub async fn terminate_instance(&self, instance_id: InstanceId, cause: TerminationCause) {
        if let Some((_, handle)) = self.running_instances.remove(&instance_id) {
            handle.join_handle.abort();

            model::cleanup_instance(instance_id.clone());

            let (termination_code, message) = match cause {
                TerminationCause::Normal => (0, "Normal termination".to_string()),
                TerminationCause::Signal => (1, "Signal termination".to_string()),
                TerminationCause::Exception(message) => (2, message),
                TerminationCause::SystemError(message) => (3, message),
                TerminationCause::OutOfResources(message) => (4, message),
            };

            server::Command::DetachInstance {
                inst_id: instance_id.clone(),
                termination_code,
                message,
            }
            .dispatch()
            .ok();
        }
    }

    async fn handle_server_request(
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        component: Component,
        arguments: Vec<String>,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> anyhow::Result<hyper::Response<HyperOutgoingBody>> {
        let inst_id = Uuid::new_v4();
        let inst_state = InstanceState::new(inst_id, arguments).await;

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
        component: Component,
        arguments: Vec<String>,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
    ) {
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
                    tokio::task::spawn(async {
                        if let Err(e) = http1::Builder::new()
                            .keep_alive(true)
                            .serve_connection(
                                stream,
                                hyper::service::service_fn(move |req| {
                                    Self::handle_server_request(
                                        engine_.clone(),
                                        linker_.clone(),
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
        component: Component,
        arguments: Vec<String>,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
    ) {
        let inst_state = InstanceState::new(instance_id, arguments).await;

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
                server::Command::DetachInstance {
                    inst_id: instance_id.clone(),
                    termination_code: 0,
                    message: return_value.unwrap_or("".to_string()),
                }
                .dispatch()
                .ok();
            }
            Err(err) => {
                println!("Instance {instance_id} failed: {err}");
                server::Command::DetachInstance {
                    inst_id: instance_id.clone(),
                    termination_code: 2,
                    message: err.to_string(),
                }
                .dispatch()
                .ok();
            }
        }

        // force cleanup of the remaining resources
        model::cleanup_instance(instance_id);
    }
}
