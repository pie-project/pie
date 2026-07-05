//! `pie-gateway` — Pie's client-facing **edge plane** (disaggregated serving).
//!
//! The gateway terminates user protocols (REST/SSE, WebSocket), gates admission
//! on cluster *resources*, routes each turn to a worker, and pipes the resulting
//! token stream back. It runs behind an edge proxy, is replicated full-mesh, and
//! is stateless except for the lifetime of an in-flight session.
//!
//! # Three directions (design §4)
//!
//! - **user → gateway** (server): the `ingress` adapters terminate REST/SSE +
//!   WebSocket and converge onto charlie's one [`session::Sessions`].
//! - **gateway ↔ worker** (server; workers dial IN, 1:N fan-in — the M3
//!   inversion): [`worker`] serves [`GatewayInbound`](pie_worker_rpc::GatewayInbound)
//!   and holds a [`WorkerControl`](pie_worker_rpc::WorkerControl) client per worker.
//! - **gateway → controller** (client): [`controller`] registers, heartbeats, and
//!   long-polls `watch_gateway` for the [`RoutingTable`](pie_controller_rpc::RoutingTable).
//!
//! # Assembly
//!
//! [`bind`] wires the four plane handles into one [`GatewayState`] (the axum
//! `State`), binds both listeners (client-facing + worker-facing), starts the
//! worker accept loop + the control loops, and returns a [`Gateway`] handle that
//! exposes the bound addrs (so the single-node launcher can point its in-proc
//! worker at `worker_addr`, and the smoke harness can dial ephemeral `:0` binds).
//! [`run`] / [`run_with`] are the serve-forever entrypoints over it.

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
use pie_controller_rpc::GatewayInfo;
use pie_ids::{ReqId, WorkerId};
use pie_worker_rpc::Request;
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{Notify, watch};

use crate::admission::AdmissionDecision;
use crate::blob::{BlobStore, GatewayOriginStore};
use crate::route::RoutingHandle;
use crate::session::{AdmitReject, DispatchFail, Sessions, TurnRouter};
use crate::worker::WorkerRegistry;

pub use crate::controller::GatewayControl;

/// Runtime configuration for the gateway. Parsed *purely* from a TOML string by
/// [`Config::parse`]; the process skeleton (`bootstrap`) sources that string
/// (file locate + read + env merge) and owns paths / observability / lifecycle —
/// this crate only understands the gateway domain.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Address the client-facing edge (REST/SSE + WebSocket) listens on — the
    /// gateway's public address (behind the edge proxy).
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    /// Address the worker-facing data plane listens on. Post-inversion (M3)
    /// workers dial IN here; for the spine workers learn it from deploy-config.
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
    /// Parse a TOML config string into a validated [`Config`]. **Pure**: no IO,
    /// no env, no clap — `bootstrap` sources the string, all gateway config
    /// domain lives here. An empty string yields all defaults.
    pub fn parse(s: &str) -> Result<Config> {
        toml::from_str(s).context("parse gateway config (TOML)")
    }
}

/// Back-compat alias used by the in-proc launcher during the build refactor;
/// prefer [`Config`].
pub type GatewayConfig = Config;

/// The shared, axum-`State`-injected gateway root. Cloneable; composes each
/// plane's handle so every route-provider and the worker server target one type:
/// charlie's session registry, alpha's routing/admission, foxtrot's worker
/// registry, bravo's blob store.
#[derive(Clone)]
pub struct GatewayState {
    /// The internal session abstraction (charlie): construct / stream / lifecycle,
    /// plus the worker-facing `feed` / `redirect` producer end.
    pub sessions: Sessions,
    /// Worker selection + coarse cluster admission (alpha): `RoutingTable` cache
    /// (+ foxtrot's connected set) behind `admit` / `select_worker` /
    /// `dispatch_with_retry`. Session-internal — ingress never touches it.
    pub routing: RoutingHandle,
    /// Live dialed-in worker connections (foxtrot): the `WorkerControlClient`
    /// registry + the connected-set watch. Session-internal.
    pub workers: WorkerRegistry,
    /// Content-addressed blob store (bravo): gateway-origin tier behind a `dyn`
    /// boundary so the object-store graduation is a pure impl swap.
    pub blobs: Arc<dyn BlobStore>,
}

/// The composition-root adapter joining charlie's session-internal [`TurnRouter`]
/// seam to alpha's [`RoutingHandle`] + foxtrot's [`WorkerRegistry`]. It lives here
/// (the only place both are visible) so `session.rs` carries no upward edge to
/// `route.rs` / `worker.rs`.
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
        // The session layer chose the key: `None` (Ephemeral/one-shot → alpha's
        // load-aware power-of-two) or `Some(session)` (Sticky/WS → HRW warm-KV).
        // The adapter is pure mechanism — forward it to the retry loop.
        self.routing
            .dispatch_with_retry(&self.workers, req, affinity)
            .await
            .map(|d| d.worker_id)
            .map_err(|_| DispatchFail)
    }

    async fn cancel(&self, worker: WorkerId, req: ReqId) {
        // Immediate reverse-channel abort (when the worker isn't pushing, so the
        // piggybacked `Control::Abort` can't reach it promptly). Best-effort: a
        // dropped worker is already gone.
        if let Some(client) = self.workers.client(worker) {
            let _ = client.cancel(tarpc::context::current(), req).await;
        }
    }

    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
        self.workers.connected_watch()
    }
}

/// A bound, assembled gateway. Holds both resolved listener addresses (so an
/// ephemeral `:0` bind surfaces the real port — needed by the single-node
/// launcher and the smoke harness) and the shared [`GatewayState`]. Call
/// [`serve`](Gateway::serve) to run the client-facing edge until shutdown; the
/// worker accept loop + control loops are already running.
pub struct Gateway {
    /// The resolved client-facing listen address.
    pub listen_addr: SocketAddr,
    /// The resolved worker-facing dial-in address (workers connect here).
    pub worker_addr: SocketAddr,
    /// The assembled shared state (test access to sessions / routing / workers /
    /// blobs without a live serve).
    pub state: GatewayState,
    listener: TcpListener,
    app: Router,
    _worker_task: tokio::task::JoinHandle<()>,
}

impl Gateway {
    /// Serve the client-facing edge (ingress + blob routes) until the listener
    /// errors or the process exits. The worker-facing data plane is already
    /// accepting dial-ins. Used by the in-proc launcher (`bin/pie`) which drives
    /// the serve directly; the daemon path uses [`into_handle`](Gateway::into_handle).
    pub async fn serve(self) -> Result<()> {
        axum::serve(self.listener, self.app)
            .await
            .context("gateway client-facing serve")?;
        Ok(())
    }

    /// Spawn the client-facing serve onto the runtime and return a
    /// [`GatewayHandle`] that owns it (plus the already-running worker accept
    /// loop) for clean lifecycle control. The serve uses graceful shutdown driven
    /// by [`GatewayHandle::shutdown`].
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

/// A running gateway daemon. Owns the spawned client-serve task, the worker
/// accept loop, and a shutdown signal; [`shutdown`](GatewayHandle::shutdown)
/// gracefully drains the client edge and stops accepting worker dial-ins. The
/// resolved bound addresses are exposed for ephemeral `:0` binds, the in-proc
/// launcher, and tests.
///
/// Dropping the handle without calling [`shutdown`](GatewayHandle::shutdown)
/// detaches the tasks (the daemon keeps running) — shutdown is always explicit.
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
    /// Stop cleanly: signal the client edge to drain in-flight requests + stop
    /// accepting (axum graceful shutdown), then stop accepting worker dial-ins.
    /// Inherent `async` per the build-seam role-library boundary — no `bootstrap`
    /// trait (Ruling R1); the `bin` adapts this into `run_until_signal`.
    pub async fn shutdown(self) {
        self.shutdown.notify_one();
        let _ = self.serve_task.await;
        // In-flight turns are not rescued across a gateway shutdown — the gateway
        // is the session's single point (design §10).
        self.worker_task.abort();
        let _ = self.worker_task.await;
    }
}

/// Register with the controller, assemble the [`GatewayState`], bind both
/// listeners, and start the worker accept loop + control loops — returning a
/// [`Gateway`] handle (with the resolved bound addrs) ready to [`serve`].
///
/// Generic over the [`GatewayControl`] backend so the launcher injects either the
/// dialed [`ControlClient`](pie_controller_rpc::ControlClient) (distributed) or the
/// in-proc embedded adapter (single-node); juliet injects a stub yielding a
/// seeded [`RoutingTable`](pie_controller_rpc::RoutingTable) for the smoke.
pub async fn bind<C: GatewayControl>(config: Config, control: C) -> Result<Gateway> {
    // Subscribe to the routing table first — independent of our own
    // registration, and needed to assemble the state below.
    let routing_rx = control.routing_watch();

    // Assemble the four plane handles. The worker registry's connected-set watch
    // feeds alpha's selector (`RoutingTable.healthy ∩ connected`); the
    // `RouteBackend` adapter joins charlie's `TurnRouter` seam to routing+workers.
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

    // Data plane: bind the worker-facing listener FIRST (workers dial IN, M3) so
    // its resolved address is known — that is the endpoint we must advertise,
    // since workers learn where to dial from the roster the controller pushes
    // them (gateway.md).
    let worker_server = worker::serve(config.worker_listen, sessions, workers)
        .await
        .context("start worker-facing data-plane server")?;
    let worker_addr = worker_server.bound;
    tracing::info!(%worker_addr, "gateway worker-facing listener up (workers dial in)");

    // Control plane: register advertising the WORKER-FACING dial-in address (not
    // the client edge), then heartbeat + hold the routing subscription. NOTE: a
    // `worker_listen` bound to `0.0.0.0` resolves to an unroutable advertise
    // address for remote workers — deployments must bind a routable interface
    // (or front the gateways with a stable Service DNS name) so the pushed
    // roster is dialable.
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
/// [`GatewayHandle`] (the role-library `run(Config) -> Handle` boundary). A thin
/// wrapper that constructs a tarpc [`ControlClient`](pie_controller_rpc::ControlClient);
/// the in-proc launcher (`bin/pie`) embeds via [`bind`] / [`run_with`] with the
/// embedded adapter instead.
pub async fn run(config: Config) -> Result<GatewayHandle> {
    let control = controller::connect_controller(&config.controller).await?;
    run_with(config, control).await
}

/// [`bind`] then spawn the serve via [`into_handle`](Gateway::into_handle), over
/// an injected [`GatewayControl`] backend — returns a [`GatewayHandle`]. The
/// in-proc launcher calls this (or [`bind`] directly) with
/// `EmbeddedControl(handle)`; the distributed daemon path goes through [`run`].
pub async fn run_with<C: GatewayControl>(config: Config, control: C) -> Result<GatewayHandle> {
    Ok(bind(config, control).await?.into_handle())
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn parse_empty_yields_defaults() {
        assert_eq!(Config::parse("").unwrap(), Config::default());
    }

    #[test]
    fn parse_overrides_only_named_fields() {
        let cfg = Config::parse(
            r#"
            listen = "127.0.0.1:9000"
            controller = "unix:/tmp/ctl.sock"
            "#,
        )
        .unwrap();
        assert_eq!(cfg.listen, "127.0.0.1:9000".parse::<SocketAddr>().unwrap());
        // unspecified field keeps its default
        assert_eq!(cfg.worker_listen, default_worker_listen());
        assert_eq!(cfg.controller, "unix:/tmp/ctl.sock");
    }

    #[test]
    fn parse_rejects_unknown_field() {
        assert!(Config::parse("bogus = 1").is_err());
    }

    #[test]
    fn gateway_config_alias_is_config() {
        let _c: GatewayConfig = Config::default();
    }
}
