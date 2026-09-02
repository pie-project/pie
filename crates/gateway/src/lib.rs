//! `gateway`: Pie's client-facing edge plane (disaggregated serving).
//!
//! Terminates user protocols (REST/SSE, WebSocket), gates admission on
//! cluster resources, routes each turn to a worker, and pipes the token
//! stream back. Runs behind an edge proxy, replicated full-mesh, stateless
//! except for the lifetime of an in-flight session.
//!
//! [`bind`] wires the plane handles into one [`GatewayState`], binds both
//! listeners (client-facing + worker-facing), starts the worker accept loop
//! and control loops, and returns a [`Gateway`] handle exposing the bound
//! addrs. [`run`] / [`run_with`] are the serve-forever entrypoints over it.

pub mod admission;
pub mod blob;
pub mod controller;
pub mod ingress;
pub mod route;
pub mod session;
pub mod worker;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::Router;
use controller_api::GatewayInfo;
use ids::{ReqId, WorkerId};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{Notify, watch};
use worker_api::Request;

use crate::admission::AdmissionDecision;
use crate::blob::{BlobStore, GatewayOriginStore};
use crate::route::RoutingHandle;
use crate::session::{AdmitReject, DispatchFail, Sessions, TurnRouter};
use crate::worker::WorkerRegistry;

pub use crate::controller::GatewayControl;

/// Runtime configuration for the gateway. Parsed purely from a TOML string
/// by [`Config::parse`]; `bootstrap` sources the string and owns paths,
/// observability, and lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Client-facing edge (REST/SSE + WebSocket) address, behind the edge
    /// proxy.
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    /// Worker-facing data-plane address; workers dial in here.
    #[serde(default = "default_worker_listen")]
    pub worker_listen: SocketAddr,
    /// Controller's tarpc control endpoint: `tcp://host:port`, a bare
    /// `host:port`, or `unix:/path`.
    #[serde(default = "default_controller")]
    pub controller: String,
}

fn default_listen() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 8080))
}
fn default_worker_listen() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 8081))
}
fn default_controller() -> String {
    "127.0.0.1:7000".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            worker_listen: default_worker_listen(),
            controller: default_controller(),
        }
    }
}

impl Config {
    /// Parse a TOML config string into a validated [`Config`]. Pure: no IO,
    /// no env, no clap. An empty string yields all defaults.
    pub fn parse(s: &str) -> Result<Config> {
        toml::from_str(s).context("parse gateway config (TOML)")
    }
}

/// Back-compat alias used by the in-proc launcher during the build refactor;
/// prefer [`Config`].
pub type GatewayConfig = Config;

/// The shared, axum-`State`-injected gateway root. Cloneable; composes each
/// plane's handle so every route-provider and the worker server target one
/// type.
#[derive(Clone)]
pub struct GatewayState {
    /// Session construct / stream / lifecycle, plus the worker-facing
    /// `feed` / `redirect` producer end.
    pub sessions: Sessions,
    /// Worker selection + coarse cluster admission: `RoutingTable` cache
    /// behind `admit` / `select_worker` / `dispatch_with_retry`.
    /// Session-internal — ingress never touches it.
    pub routing: RoutingHandle,
    /// Live dialed-in worker connections: the `WorkerControlClient` registry
    /// and connected-set watch. Session-internal.
    pub workers: WorkerRegistry,
    /// Content-addressed blob store: gateway-origin tier behind a `dyn`
    /// boundary so the object-store graduation is a pure impl swap.
    pub blobs: Arc<dyn BlobStore>,
}

/// The composition-root adapter joining the session-internal [`TurnRouter`]
/// seam to [`RoutingHandle`] + [`WorkerRegistry`], so `session.rs` carries no
/// upward edge to `route.rs` / `worker.rs`.
struct RouteBackend {
    routing: RoutingHandle,
    workers: WorkerRegistry,
}

#[async_trait::async_trait]
impl TurnRouter for RouteBackend {
    async fn admit(&self, req: &Request) -> std::result::Result<(), AdmitReject> {
        match self.routing.admit(req) {
            AdmissionDecision::Admit => Ok(()),
            AdmissionDecision::Reject(reason) => Err(AdmitReject(reason.to_string())),
        }
    }

    async fn dispatch(
        &self,
        req: &Request,
        affinity: Option<u64>,
    ) -> std::result::Result<WorkerId, DispatchFail> {
        // The session layer chose the key (`None` for load-aware
        // power-of-two, `Some(session)` for HRW warm-KV); forward it.
        self.routing
            .dispatch_with_retry(&self.workers, req, affinity)
            .await
            .map(|d| d.worker_id)
            .map_err(|_| DispatchFail)
    }

    async fn cancel(&self, worker: WorkerId, req: ReqId) {
        // Immediate reverse-channel abort, best-effort: a dropped worker is
        // already gone.
        if let Some(client) = self.workers.client(worker) {
            let _ = client.cancel(tarpc::context::current(), req).await;
        }
    }

    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
        self.workers.connected_watch()
    }
}

/// A bound, assembled gateway. Holds both resolved listener addresses
/// (surfacing the real port for an ephemeral `:0` bind) and the shared
/// [`GatewayState`]. Call [`serve`](Gateway::serve) to run the client-facing
/// edge until shutdown; the worker accept loop + control loops already run.
pub struct Gateway {
    /// The resolved client-facing listen address.
    pub listen_addr: SocketAddr,
    /// The resolved worker-facing dial-in address (workers connect here).
    pub worker_addr: SocketAddr,
    /// The assembled shared state (test access without a live serve).
    pub state: GatewayState,
    listener: TcpListener,
    app: Router,
    _worker_task: tokio::task::JoinHandle<()>,
}

impl Gateway {
    /// Serve the client-facing edge until the listener errors or the process
    /// exits. The worker-facing data plane is already accepting dial-ins.
    pub async fn serve(self) -> Result<()> {
        axum::serve(self.listener, self.app)
            .await
            .context("gateway client-facing serve")?;
        Ok(())
    }

    /// Spawn the client-facing serve onto the runtime and return a
    /// [`GatewayHandle`] that owns it (and the worker accept loop) for clean
    /// lifecycle control, with graceful shutdown.
    pub fn into_handle(self) -> GatewayHandle {
        let shutdown = Arc::new(Notify::new());
        let listen_addr = self.listen_addr;
        let worker_addr = self.worker_addr;
        let worker_task = self._worker_task;
        let listener = self.listener;
        let app = self.app;
        let serve_shutdown = shutdown.clone();
        let serve_task = tokio::spawn(async move {
            let graceful = async move { serve_shutdown.notified().await };
            if let Err(e) = axum::serve(listener, app)
                .with_graceful_shutdown(graceful)
                .await
            {
                tracing::error!(error = %e, "gateway client-facing serve ended");
            }
        });
        GatewayHandle {
            listen_addr,
            worker_addr,
            shutdown,
            serve_task,
            worker_task,
        }
    }
}

/// A running gateway daemon. Owns the client-serve task, the worker accept
/// loop, and a shutdown signal; [`shutdown`](GatewayHandle::shutdown)
/// gracefully drains the client edge and stops accepting worker dial-ins.
/// Dropping without calling it detaches the tasks (the daemon keeps
/// running) — shutdown is always explicit.
pub struct GatewayHandle {
    /// The resolved client-facing listen address.
    pub listen_addr: SocketAddr,
    /// The resolved worker-facing dial-in address.
    pub worker_addr: SocketAddr,
    shutdown: Arc<Notify>,
    serve_task: tokio::task::JoinHandle<()>,
    worker_task: tokio::task::JoinHandle<()>,
}

impl GatewayHandle {
    /// Stop cleanly: drain in-flight requests and stop accepting (axum
    /// graceful shutdown), then stop accepting worker dial-ins.
    pub async fn shutdown(self) {
        self.shutdown.notify_one();
        let _ = self.serve_task.await;
        // In-flight turns are not rescued across a gateway shutdown.
        self.worker_task.abort();
        let _ = self.worker_task.await;
    }
}

/// Register with the controller, assemble the [`GatewayState`], bind both
/// listeners, and start the worker accept loop + control loops — returning a
/// [`Gateway`] handle (with the resolved bound addrs) ready to [`serve`].
///
/// Generic over the [`GatewayControl`] backend so the launcher injects either
/// the dialed [`ControlClient`](controller_api::ControlClient) (distributed)
/// or the in-proc embedded adapter (single-node).
pub async fn bind<C: GatewayControl>(config: Config, control: C) -> Result<Gateway> {
    // Subscribe to the routing table first, needed to assemble the state below.
    let routing_rx = control.routing_watch();

    // The worker registry's connected-set watch feeds the selector; the
    // `RouteBackend` adapter joins the `TurnRouter` seam to routing+workers.
    let workers = WorkerRegistry::new();
    let routing = RoutingHandle::new(routing_rx, workers.connected_watch());
    let sessions = Sessions::new(Arc::new(RouteBackend {
        routing: routing.clone(),
        workers: workers.clone(),
    }));
    let blobs: Arc<dyn BlobStore> =
        Arc::new(GatewayOriginStore::new(format!("http://{}", config.listen)));
    let state = GatewayState {
        sessions: sessions.clone(),
        routing,
        workers: workers.clone(),
        blobs: blobs.clone(),
    };

    // Bind the worker-facing listener first: its resolved address is the
    // endpoint we advertise (workers learn where to dial from the roster
    // the controller pushes).
    let worker_server = worker::serve(config.worker_listen, sessions, workers)
        .await
        .context("start worker-facing data-plane server")?;
    let worker_addr = worker_server.bound;
    tracing::info!(%worker_addr, "gateway worker-facing listener up (workers dial in)");

    // Register the worker-facing (not client) dial-in address, then
    // heartbeat. A `worker_listen` of `0.0.0.0` is unroutable for remote
    // workers; deployments must bind a routable interface.
    let info = GatewayInfo {
        addr: worker_addr.to_string(),
    };
    let gateway_id = control
        .register_gateway(info.clone())
        .await
        .context("register gateway with controller")?;
    tracing::info!(%gateway_id, %worker_addr, "gateway registered with controller");
    tokio::spawn(controller::heartbeat_loop(control, gateway_id, info));

    // Client plane: ingress + blob route-providers merge onto the one listener.
    let app = Router::new()
        .merge(ingress::router(state.clone()))
        .merge(blob::router(blobs));
    let listener = TcpListener::bind(config.listen)
        .await
        .with_context(|| format!("bind client-facing listener on {}", config.listen))?;
    let listen_addr = listener
        .local_addr()
        .context("client listener local_addr")?;
    tracing::info!(%listen_addr, "pie-gateway client-facing edge up");

    Ok(Gateway {
        listen_addr,
        worker_addr,
        state,
        listener,
        app,
        _worker_task: worker_server.task,
    })
}

/// Dial the controller and run the gateway as a daemon, returning a
/// [`GatewayHandle`]. A thin wrapper constructing a tarpc
/// [`ControlClient`](controller_api::ControlClient); the in-proc launcher
/// embeds via [`bind`] / [`run_with`] instead.
pub async fn run(config: Config) -> Result<GatewayHandle> {
    let control = controller::connect_controller(&config.controller).await?;
    run_with(config, control).await
}

/// [`bind`] then spawn the serve via [`into_handle`](Gateway::into_handle),
/// over an injected [`GatewayControl`] backend.
pub async fn run_with<C: GatewayControl>(config: Config, control: C) -> Result<GatewayHandle> {
    Ok(bind(config, control).await?.into_handle())
}

