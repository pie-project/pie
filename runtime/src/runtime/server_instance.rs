//! Server Instance Actor - HTTP server instance management
//!
//! This module provides the ServerInstanceActor for managing HTTP server instances.
//! Like InstanceActor, these are registered in a global registry for Direct Addressing.

use std::net::SocketAddr;
use std::sync::LazyLock;
use std::time::Instant;

use dashmap::DashMap;
use hyper::server::conn::http1;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use uuid::Uuid;
use wasmtime::component::{Component, Linker, Resource};
use wasmtime::{Engine, Store};
use wasmtime_wasi_http::WasiHttpView;
use wasmtime_wasi_http::bindings::exports::wasi::http::incoming_handler::{
    IncomingRequest, ResponseOutparam,
};
use wasmtime_wasi_http::bindings::http::types::Scheme;
use wasmtime_wasi_http::body::HyperOutgoingBody;
use wasmtime_wasi_http::io::TokioIo;
use std::sync::Arc;

use crate::service::{Service, ServiceHandler};

use super::instance::{InstanceId, InstanceState};

// =============================================================================
// Server Instance Registry (Direct Addressing)
// =============================================================================

/// Global registry mapping InstanceId to server instance actors.
static SERVER_INSTANCE_REGISTRY: LazyLock<DashMap<InstanceId, Service<ServerInstanceMessage>>> =
    LazyLock::new(DashMap::new);

/// Sends a message directly to a server instance by InstanceId.
pub fn send(inst_id: InstanceId, msg: ServerInstanceMessage) -> anyhow::Result<()> {
    SERVER_INSTANCE_REGISTRY
        .get(&inst_id)
        .ok_or_else(|| anyhow::anyhow!("Server instance not found"))?
        .send(msg)
}

/// Remove a server instance from the registry.
fn unregister(inst_id: InstanceId) {
    SERVER_INSTANCE_REGISTRY.remove(&inst_id);
}

// =============================================================================
// Server Instance Messages
// =============================================================================

/// Messages that can be sent to a ServerInstanceActor.
#[derive(Debug)]
pub enum ServerInstanceMessage {
    /// Terminate this server instance
    Terminate,
    /// Get info for listing
    GetInfo {
        response: oneshot::Sender<ServerInstanceInfo>,
    },
}

/// Info for server instance listing
#[derive(Debug, Clone)]
pub struct ServerInstanceInfo {
    pub username: String,
    pub program_name: String,
    pub port: u16,
    pub elapsed_secs: u64,
}

// =============================================================================
// ServerInstanceActor
// =============================================================================

/// Actor managing a single HTTP server instance.
pub struct ServerInstanceActor {
    inst_id: InstanceId,
    username: String,
    program_name: String,
    port: u16,
    start_time: Instant,
    listener_handle: Option<JoinHandle<()>>,
}

/// Configuration for spawning a new server instance
pub struct ServerInstanceConfig {
    pub inst_id: InstanceId,
    pub username: String,
    pub program_name: String,
    pub port: u16,
    pub arguments: Vec<String>,
    pub component: Component,
    pub engine: Engine,
    pub linker: Arc<Linker<InstanceState>>,
}

impl ServerInstanceActor {
    /// Spawns a new ServerInstanceActor and registers it in the global registry.
    pub async fn spawn(config: ServerInstanceConfig) -> anyhow::Result<InstanceId> {
        let inst_id = config.inst_id;
        let username = config.username.clone();
        let program_name = config.program_name.clone();
        let port = config.port;

        // Create the actor entry in the registry first
        let service = Service::new();
        SERVER_INSTANCE_REGISTRY.insert(inst_id, service);

        let actor_ref = SERVER_INSTANCE_REGISTRY.get(&inst_id).unwrap();

        let addr = SocketAddr::from(([127, 0, 0, 1], port));

        // Spawn the HTTP listener task
        let listener_handle = tokio::spawn(Self::run_server(
            addr,
            config.username,
            config.component,
            config.arguments,
            config.engine,
            config.linker,
        ));

        // Spawn the actor with initialized state
        actor_ref.spawn(|| ServerInstanceActor {
            inst_id,
            username,
            program_name,
            port,
            start_time: Instant::now(),
            listener_handle: Some(listener_handle),
        })?;

        Ok(inst_id)
    }

    /// Runs the HTTP server
    async fn run_server(
        addr: SocketAddr,
        username: String,
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

            loop {
                let (stream, _) = listener.accept().await?;
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
                                Self::handle_request(
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

            #[allow(unreachable_code)]
            anyhow::Ok(())
        };

        if let Err(e) = result.await {
            eprintln!("Server error: {e}");
        }
    }

    /// Handles a single HTTP request
    async fn handle_request(
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
        username: String,
        component: Component,
        _arguments: Vec<String>,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> anyhow::Result<hyper::Response<HyperOutgoingBody>> {
        let inst_id = Uuid::new_v4();
        let (inst_state, _output_delivery_ctrl) =
            InstanceState::new(inst_id, username).await;

        let mut store = Store::new(&engine, inst_state);
        let (sender, receiver) = oneshot::channel();

        let req = store.data_mut().new_incoming_request(Scheme::Http, req)?;
        let out = store.data_mut().new_response_outparam(sender)?;

        let instance = linker
            .instantiate_async(&mut store, &component)
            .await
            .map_err(|e| anyhow::anyhow!("Instantiation error: {e}"))?;

        let (_, serve_export) = instance
            .get_export(&mut store, None, "wasi:http/incoming-handler@0.2.4")
            .ok_or_else(|| anyhow::anyhow!("No 'serve' function found"))?;

        let (_, handle_func_export) = instance
            .get_export(&mut store, Some(&serve_export), "handle")
            .ok_or_else(|| anyhow::anyhow!("No 'handle' function found"))?;

        let handle_func = instance
            .get_typed_func::<(Resource<IncomingRequest>, Resource<ResponseOutparam>), ()>(
                &mut store,
                &handle_func_export,
            )
            .map_err(|e| anyhow::anyhow!("Failed to get 'handle' function: {e}"))?;

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

    /// Cleanup and unregister this server instance
    fn cleanup(&mut self) {
        if let Some(handle) = self.listener_handle.take() {
            handle.abort();
        }
        unregister(self.inst_id);
    }
}

impl ServiceHandler for ServerInstanceActor {
    type Message = ServerInstanceMessage;

    async fn handle(&mut self, msg: ServerInstanceMessage) {
        match msg {
            ServerInstanceMessage::Terminate => {
                self.cleanup();
            }
            ServerInstanceMessage::GetInfo { response } => {
                let info = ServerInstanceInfo {
                    username: self.username.clone(),
                    program_name: self.program_name.clone(),
                    port: self.port,
                    elapsed_secs: self.start_time.elapsed().as_secs(),
                };
                let _ = response.send(info);
            }
        }
    }
}

// =============================================================================
// Async Helper Functions
// =============================================================================

/// Terminate a server instance (fire-and-forget).
pub fn terminate(inst_id: InstanceId) {
    let _ = send(inst_id, ServerInstanceMessage::Terminate);
}

/// Get server instance info.
pub async fn get_info(inst_id: InstanceId) -> Option<ServerInstanceInfo> {
    let (tx, rx) = oneshot::channel();
    send(inst_id, ServerInstanceMessage::GetInfo { response: tx }).ok()?;
    rx.await.ok()
}

/// List all registered server instance IDs.
pub fn list_instance_ids() -> Vec<InstanceId> {
    SERVER_INSTANCE_REGISTRY.iter().map(|r| *r.key()).collect()
}
