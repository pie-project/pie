//! Daemon - Long-lived HTTP-serving WASM process
//!
//! Each Daemon is a ServiceMap actor that binds a TCP port and serves HTTP
//! requests by invoking a WASM component's `wasi:http/incoming-handler`.
//! Unlike a Process (one-shot execution), a Daemon runs indefinitely.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::time::Instant;

use anyhow::{anyhow, Result};
use hyper::server::conn::http1;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use wasmtime::component::Resource;
use wasmtime_wasi_http::WasiHttpView;
use wasmtime_wasi_http::bindings::exports::wasi::http::incoming_handler::{
    IncomingRequest, ResponseOutparam,
};
use wasmtime_wasi_http::bindings::http::types::Scheme;
use wasmtime_wasi_http::body::HyperOutgoingBody;
use wasmtime_wasi_http::io::TokioIo;

use crate::linker;
use crate::program::ProgramName;
use crate::service::{ServiceMap, ServiceHandler};

// =============================================================================
// Daemon Registry
// =============================================================================

type DaemonId = usize;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// Global registry mapping DaemonId to daemon actors.
static SERVICES: LazyLock<ServiceMap<DaemonId, Message>> =
    LazyLock::new(ServiceMap::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawn a new daemon and register it in the global registry.
pub fn spawn(
    username: String,
    program: ProgramName,
    port: u16,
    input: String,
) -> Result<DaemonId> {
    let daemon = Daemon::new(username, program, port, input);
    let id = daemon.daemon_id;
    SERVICES.spawn(id, || daemon)?;
    Ok(id)
}

/// Terminate a daemon (fire-and-forget).
pub fn terminate(daemon_id: DaemonId) {
    let _ = SERVICES.send(&daemon_id, Message::Terminate);
}

/// Get info about a running daemon.
pub async fn get_info(daemon_id: DaemonId) -> Option<DaemonInfo> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(&daemon_id, Message::GetInfo { response: tx }).ok()?;
    rx.await.ok()
}

/// List all registered daemon IDs.
pub fn list() -> Vec<DaemonId> {
    SERVICES.keys()
}

// =============================================================================
// Messages
// =============================================================================

/// Messages that can be sent directly to a Daemon.
enum Message {
    /// Terminate this daemon
    Terminate,
    /// Query daemon info
    GetInfo {
        response: oneshot::Sender<DaemonInfo>,
    },
}

/// Info returned for daemon listing.
#[derive(Debug, Clone)]
pub struct DaemonInfo {
    pub username: String,
    pub program: String,
    pub port: u16,
    pub elapsed_secs: u64,
}

// =============================================================================
// Daemon
// =============================================================================

/// Actor managing a long-lived HTTP-serving WASM instance.
struct Daemon {
    daemon_id: DaemonId,
    username: String,
    program: ProgramName,
    port: u16,
    start_time: Instant,
    listener_handle: JoinHandle<()>,
}

impl Daemon {
    /// Creates a new Daemon and spawns its HTTP listener task.
    fn new(
        username: String,
        program: ProgramName,
        port: u16,
        input: String,
    ) -> Self {
        let daemon_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let addr = SocketAddr::from(([127, 0, 0, 1], port));

        let listener_handle = tokio::spawn(Self::serve(
            addr,
            username.clone(),
            program.clone(),
            input,
        ));

        Daemon {
            daemon_id,
            username,
            program,
            port,
            start_time: Instant::now(),
            listener_handle,
        }
    }

    /// Binds the TCP port and serves HTTP requests indefinitely.
    async fn serve(
        addr: SocketAddr,
        username: String,
        program: ProgramName,
        input: String,
    ) {
        let result: Result<()> = async {
            let socket = tokio::net::TcpSocket::new_v4()?;
            socket.set_reuseaddr(!cfg!(windows))?;
            socket.bind(addr)?;
            let listener = socket.listen(100)?;
            tracing::info!("Daemon serving HTTP on http://{}/", listener.local_addr()?);

            loop {
                let (stream, _) = listener.accept().await?;
                let stream = TokioIo::new(stream);
                let username = username.clone();
                let program = program.clone();
                let input = input.clone();

                tokio::task::spawn(async move {
                    if let Err(e) = http1::Builder::new()
                        .keep_alive(true)
                        .serve_connection(
                            stream,
                            hyper::service::service_fn(move |req| {
                                Self::handle_request(
                                    username.clone(),
                                    program.clone(),
                                    input.clone(),
                                    req,
                                )
                            }),
                        )
                        .await
                    {
                        tracing::error!("HTTP connection error: {e:?}");
                    }
                });
            }

            #[allow(unreachable_code)]
            Ok(())
        }
        .await;

        if let Err(e) = result {
            tracing::error!("Daemon server error: {e}");
        }
    }

    /// Handles a single HTTP request by instantiating the WASM component.
    ///
    /// Each request gets a fresh Store and component instance. The WASM
    /// component must export `wasi:http/incoming-handler@0.2.4`.
    async fn handle_request(
        username: String,
        program: ProgramName,
        _input: String,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> Result<hyper::Response<HyperOutgoingBody>> {
        // Instantiate a fresh WASM component (store + instance) per request.
        // Daemons don't capture outputs — they serve HTTP responses directly.
        let (mut store, instance) =
            linker::instantiate(uuid::Uuid::new_v4(), username, &program, false).await?;

        // Convert the hyper request into WASI HTTP resources
        let (sender, receiver) = oneshot::channel();
        let req = store.data_mut().new_incoming_request(Scheme::Http, req)?;
        let out = store.data_mut().new_response_outparam(sender)?;

        // Find the incoming-handler export
        let (_, serve_export) = instance
            .get_export(&mut store, None, "wasi:http/incoming-handler@0.2.4")
            .ok_or_else(|| anyhow!("No 'wasi:http/incoming-handler' interface found"))?;

        let (_, handle_func_export) = instance
            .get_export(&mut store, Some(&serve_export), "handle")
            .ok_or_else(|| anyhow!("No 'handle' function found"))?;

        let handle_func = instance
            .get_typed_func::<(Resource<IncomingRequest>, Resource<ResponseOutparam>), ()>(
                &mut store,
                &handle_func_export,
            )
            .map_err(|e| anyhow!("Failed to get 'handle' function: {e}"))?;

        // Spawn the WASM handler — it writes the response via the outparam
        let task = tokio::task::spawn(async move {
            handle_func
                .call_async(&mut store, (req, out))
                .await
                .map_err(|e| anyhow!("Handler error: {e}"))
        });

        // Wait for the response from the outparam channel
        match receiver.await {
            Ok(Ok(resp)) => Ok(resp),
            Ok(Err(e)) => Err(e.into()),
            Err(_) => {
                // Outparam was never set — check the task for the real error
                let e = match task.await {
                    Ok(Err(e)) => e,
                    Err(e) => e.into(),
                    Ok(Ok(())) => anyhow!("handler completed without setting response"),
                };
                Err(e.context("guest never invoked `response-outparam::set` method"))
            }
        }
    }

    /// Cleanup: abort the listener and unregister.
    fn cleanup(&mut self) {
        self.listener_handle.abort();
        SERVICES.remove(&self.daemon_id);
    }
}

impl ServiceHandler for Daemon {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Terminate => {
                self.cleanup();
            }
            Message::GetInfo { response } => {
                let _ = response.send(DaemonInfo {
                    username: self.username.clone(),
                    program: self.program.to_string(),
                    port: self.port,
                    elapsed_secs: self.start_time.elapsed().as_secs(),
                });
            }
        }
    }
}
