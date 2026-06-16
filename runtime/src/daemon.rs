//! Daemon - Long-lived HTTP-serving WASM process
//!
//! Each Daemon is a ServiceMap actor that binds a TCP port and serves HTTP
//! requests by invoking a WASM component's `wasi:http/incoming-handler`.
//! Unlike a Process (one-shot execution), a Daemon runs indefinitely.

use std::net::{IpAddr, SocketAddr};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Result, anyhow};
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

use crate::instance::OutputMode;
use crate::linker;
use crate::program::ProgramName;
use crate::service::{ServiceHandler, ServiceMap};

// =============================================================================
// Daemon Registry
// =============================================================================

type DaemonId = usize;

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

/// Global registry mapping DaemonId to daemon actors.
static SERVICES: LazyLock<ServiceMap<DaemonId, Message>> = LazyLock::new(ServiceMap::new);

// =============================================================================
// Public API
// =============================================================================

/// Spawn a new daemon and register it in the global registry.
pub fn spawn(
    username: String,
    program: ProgramName,
    port: u16,
    host: String,
    input: String,
) -> Result<DaemonId> {
    let daemon = Daemon::new(username, program, port, host, input)?;
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
    SERVICES
        .send(&daemon_id, Message::GetInfo { response: tx })
        .ok()?;
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
// Fault classification for handle_request
// =============================================================================

/// Fault domain for a daemon HTTP request failure. Mapped to a status code at
/// the response boundary so the engine-vs-inferlet attribution stays visible
/// to clients (see `runtime/src/daemon.rs::Daemon::fault_response`).
#[derive(Clone, Copy, Debug)]
enum FaultClass {
    /// Pie-server failed to set up the guest call (body buffer, instantiate,
    /// missing wasi:http export, new-incoming-request, new-outparam). Status `500`.
    HostSetup,
    /// Pie-server set up correctly; the guest violated the WASI HTTP contract
    /// (set outparam to error, never set outparam, or trapped during `handle`).
    /// Status `502`.
    GuestFault,
    /// The spawned guest task panicked or was aborted in-flight — the engine
    /// is the immediate culprit from the client's POV. Status `503` with
    /// `Retry-After: 1`.
    InFlightCrash,
}

impl FaultClass {
    fn status(self) -> hyper::StatusCode {
        match self {
            FaultClass::HostSetup => hyper::StatusCode::INTERNAL_SERVER_ERROR,
            FaultClass::GuestFault => hyper::StatusCode::BAD_GATEWAY,
            FaultClass::InFlightCrash => hyper::StatusCode::SERVICE_UNAVAILABLE,
        }
    }
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
        host: String,
        input: String,
    ) -> Result<Self> {
        let daemon_id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let addr = daemon_addr(&host, port)?;

        let listener_handle =
            tokio::spawn(Self::serve(addr, username.clone(), program.clone(), input));

        Ok(Daemon {
            daemon_id,
            username,
            program,
            port,
            start_time: Instant::now(),
            listener_handle,
        })
    }

    /// Binds the TCP port and serves HTTP requests indefinitely.
    async fn serve(addr: SocketAddr, username: String, program: ProgramName, input: String) {
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
    ///
    /// Failures are classified into three fault domains and surfaced to the
    /// client as a status code plus a stable, short body tag (no anyhow chain,
    /// no internal paths or symbols — those stay in `tracing::error!`):
    ///   - `500` host setup fault (body buffer, instantiate, missing export,
    ///     new-incoming-request, new-outparam).
    ///   - `502` upstream/guest fault (guest set outparam to error, guest
    ///     never set outparam, handler trap).
    ///   - `503` in-flight engine crash (spawned handler task panicked or was
    ///     aborted); includes `Retry-After: 1`.
    /// Once the guest writes to the outparam, its response is returned
    /// verbatim — guest status codes are not rewritten.
    async fn handle_request(
        username: String,
        program: ProgramName,
        _input: String,
        req: hyper::Request<hyper::body::Incoming>,
    ) -> Result<hyper::Response<HyperOutgoingBody>> {
        type HandleErr = (FaultClass, &'static str, anyhow::Error);

        let result: std::result::Result<hyper::Response<HyperOutgoingBody>, HandleErr> = async {
            // Buffer the request body before the WASM handler runs in a spawned
            // task below. hyper's Incoming body uses a zero-capacity channel that
            // requires sender and receiver to poll in the same task; a spawned
            // handler deadlocks on bodies spanning multiple TCP segments (>~16KB).
            let (parts, body) = req.into_parts();
            let collected = http_body_util::BodyExt::collect(body).await.map_err(|e| {
                (
                    FaultClass::HostSetup,
                    "body-buffer-failed",
                    anyhow!("Failed to read request body: {e}"),
                )
            })?;
            let buffered_body = http_body_util::BodyExt::map_err(
                http_body_util::Full::new(collected.to_bytes()),
                |never: std::convert::Infallible| -> hyper::Error { match never {} },
            );
            let req = hyper::Request::from_parts(parts, buffered_body);

            // Instantiate a fresh WASM component (store + instance) per request.
            // Daemons serve HTTP responses directly, so there is no client to attach
            // their stdout/stderr to. Route guest output to pie-server's tracing log
            // (tagged with the program name) so inferlet diagnostics stay visible to
            // operators instead of falling through to wasmtime's default sink.
            let output = OutputMode::Log {
                program: program.to_string(),
            };
            // Fresh process id for this request. Register it with the context
            // manager *before* instantiating so any context operation the guest
            // makes (context.create -> ContextManager::create) finds a registered
            // process entry. One-shot Processes do this in `process::run`; the
            // daemon request path must do the same or `process_entry` panics on
            // an unregistered pid. Cleanup happens automatically when `store`
            // (and its InstanceState) is dropped at the end of the request, via
            // InstanceState::drop -> context::unregister_process.
            let process_id = uuid::Uuid::new_v4();
            crate::context::register_process(process_id, None)
                .await
                .map_err(|e| (FaultClass::HostSetup, "register-process-failed", e))?;
            let (mut store, instance) =
                match linker::instantiate(process_id, username, &program, output, None).await {
                    Ok(pair) => pair,
                    Err(e) => {
                        // No InstanceState was created, so its Drop won't run the
                        // automatic cleanup -- unregister the process explicitly to
                        // avoid leaking the registration.
                        crate::context::unregister_process(process_id);
                        return Err((FaultClass::HostSetup, "instantiate-failed", e));
                    }
                };

            // Convert the hyper request into WASI HTTP resources
            let (sender, receiver) = oneshot::channel();
            let req = store
                .data_mut()
                .new_incoming_request(Scheme::Http, req)
                .map_err(|e| (FaultClass::HostSetup, "new-incoming-request-failed", e))?;
            let out = store
                .data_mut()
                .new_response_outparam(sender)
                .map_err(|e| (FaultClass::HostSetup, "new-outparam-failed", e))?;

            // Find the incoming-handler export
            let (_, serve_export) = instance
                .get_export(&mut store, None, "wasi:http/incoming-handler@0.2.4")
                .ok_or_else(|| {
                    (
                        FaultClass::HostSetup,
                        "missing-export",
                        anyhow!("No 'wasi:http/incoming-handler' interface found"),
                    )
                })?;

            let (_, handle_func_export) = instance
                .get_export(&mut store, Some(&serve_export), "handle")
                .ok_or_else(|| {
                    (
                        FaultClass::HostSetup,
                        "missing-export",
                        anyhow!("No 'handle' function found"),
                    )
                })?;

            let handle_func = instance
                .get_typed_func::<(Resource<IncomingRequest>, Resource<ResponseOutparam>), ()>(
                    &mut store,
                    &handle_func_export,
                )
                .map_err(|e| {
                    (
                        FaultClass::HostSetup,
                        "missing-export",
                        anyhow!("Failed to get 'handle' function: {e}"),
                    )
                })?;

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
                Ok(Err(e)) => Err((FaultClass::GuestFault, "outparam-error", e.into())),
                Err(_) => {
                    // Outparam was never set — discriminate guest trap vs task panic.
                    match task.await {
                        Ok(Err(e)) => Err((
                            FaultClass::GuestFault,
                            "handler-trap",
                            e.context("guest never invoked `response-outparam::set` method"),
                        )),
                        Err(join_err) => Err((
                            FaultClass::InFlightCrash,
                            "handler-panic",
                            anyhow!("{join_err}")
                                .context("guest handler task panicked or was aborted"),
                        )),
                        Ok(Ok(())) => Err((
                            FaultClass::GuestFault,
                            "outparam-never-set",
                            anyhow!("handler completed without setting response"),
                        )),
                    }
                }
            }
        }
        .await;

        match result {
            Ok(resp) => Ok(resp),
            Err((class, tag, err)) => {
                let status = class.status().as_u16();
                tracing::error!(
                    fault_class = ?class, fault_tag = tag, fault_status = status,
                    "Daemon handle_request failed: {err:#}",
                );
                Ok(Self::fault_response(class, tag))
            }
        }
    }

    /// Build a fault response with a status from `class` and a short, stable
    /// plain-text body equal to `tag`. The anyhow chain is intentionally NOT
    /// placed in the body — it would leak internal topology / runtime versions
    /// / wasmtime symbols to any caller, and would also lock pie's host-error
    /// wire format to whatever `Display for anyhow::Error` happens to render.
    /// Callers wanting a coded shape can match on the tag string; the detail
    /// is recorded once via `tracing::error!`.
    ///
    /// Body tags currently in use:
    /// `body-buffer-failed`, `instantiate-failed`,
    /// `new-incoming-request-failed`, `new-outparam-failed`, `missing-export`
    /// (status 500); `outparam-error`, `handler-trap`, `outparam-never-set`
    /// (status 502); `handler-panic` (status 503, with `Retry-After: 1`).
    fn fault_response(class: FaultClass, tag: &'static str) -> hyper::Response<HyperOutgoingBody> {
        use http_body_util::{BodyExt, Full};
        let body: HyperOutgoingBody = Full::new(bytes::Bytes::from_static(tag.as_bytes()))
            .map_err(|never: std::convert::Infallible| match never {})
            .boxed_unsync();
        let mut builder = hyper::Response::builder()
            .status(class.status())
            .header(hyper::header::CONTENT_TYPE, "text/plain; charset=utf-8");
        if matches!(class, FaultClass::InFlightCrash) {
            builder = builder.header(hyper::header::RETRY_AFTER, "1");
        }
        builder
            .body(body)
            .expect("response builder never fails on static status + header + infallible body")
    }

    /// Cleanup: abort the listener and unregister.
    fn cleanup(&mut self) {
        self.listener_handle.abort();
        SERVICES.remove(&self.daemon_id);
    }
}

fn daemon_addr(host: &str, port: u16) -> Result<SocketAddr> {
    let ip: IpAddr = host
        .parse()
        .map_err(|_| anyhow!("invalid daemon bind host: {host}"))?;
    if !ip.is_ipv4() {
        return Err(anyhow!("daemon bind host must be IPv4: {host}"));
    }
    Ok(SocketAddr::new(ip, port))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_addr_accepts_loopback() {
        let addr = daemon_addr("127.0.0.1", 8123).unwrap();
        assert_eq!(addr.to_string(), "127.0.0.1:8123");
    }

    #[test]
    fn daemon_addr_accepts_all_interfaces() {
        let addr = daemon_addr("0.0.0.0", 8123).unwrap();
        assert_eq!(addr.to_string(), "0.0.0.0:8123");
    }

    #[test]
    fn daemon_addr_rejects_non_ip_hosts() {
        let err = daemon_addr("localhost", 8123).unwrap_err().to_string();
        assert!(err.contains("invalid daemon bind host"));
    }
}
