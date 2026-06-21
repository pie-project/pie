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
//!   inversion): [`worker`] serves [`GatewayInbound`](pie_dispatch::GatewayInbound)
//!   and holds a [`WorkerControl`](pie_dispatch::WorkerControl) client per worker.
//! - **gateway → controller** (client): [`controller`] registers, heartbeats, and
//!   long-polls `watch_gateway` for the [`RoutingTable`](pie_schema::control::RoutingTable).
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
use pie_schema::control::{GatewayInfo, WorkerId};
use pie_schema::gateway::{ReqId, Request};
use tokio::net::TcpListener;
use tokio::sync::watch;

use crate::admission::AdmissionDecision;
use crate::blob::{BlobStore, GatewayOriginStore};
use crate::route::RoutingHandle;
use crate::session::{AdmitReject, DispatchFail, Sessions, TurnRouter};
use crate::worker::WorkerRegistry;

pub use crate::controller::GatewayControl;

/// Runtime configuration for the gateway process.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Address the client-facing edge (REST/SSE + WebSocket) listens on — the
    /// gateway's public address (behind the edge proxy).
    pub listen: SocketAddr,
    /// Address the worker-facing data plane listens on. Post-inversion (M3)
    /// workers dial IN here; for the spine workers learn it from deploy-config.
    pub worker_listen: SocketAddr,
    /// Controller's tarpc control endpoint: `tcp://host:port`, a bare
    /// `host:port`, or `unix:/path`.
    pub controller: String,
}

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
    /// accepting dial-ins.
    pub async fn serve(self) -> Result<()> {
        axum::serve(self.listener, self.app)
            .await
            .context("gateway client-facing serve")?;
        Ok(())
    }
}

/// Register with the controller, assemble the [`GatewayState`], bind both
/// listeners, and start the worker accept loop + control loops — returning a
/// [`Gateway`] handle (with the resolved bound addrs) ready to [`serve`].
///
/// Generic over the [`GatewayControl`] backend so the launcher injects either the
/// dialed [`ControlClient`](pie_control::ControlClient) (distributed) or the
/// in-proc embedded adapter (single-node); juliet injects a stub yielding a
/// seeded [`RoutingTable`](pie_schema::control::RoutingTable) for the smoke.
pub async fn bind<C: GatewayControl>(config: GatewayConfig, control: C) -> Result<Gateway> {
    // Control plane: register + heartbeat + subscribe to the routing table.
    let info = GatewayInfo {
        addr: config.listen.to_string(),
    };
    let gateway_id = control
        .register_gateway(info.clone())
        .await
        .context("register gateway with controller")?;
    tracing::info!(%gateway_id, "gateway registered with controller");
    let routing_rx = control.routing_watch();
    tokio::spawn(controller::heartbeat_loop(control, gateway_id, info));

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
    // its resolved address is known before serving — the single-node launcher
    // points its in-proc worker here, and the smoke harness dials it.
    let worker_server = worker::serve(config.worker_listen, sessions, workers)
        .await
        .context("start worker-facing data-plane server")?;
    let worker_addr = worker_server.bound;
    tracing::info!(%worker_addr, "gateway worker-facing listener up (workers dial in)");

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

/// Dial the controller and serve the edge plane over the distributed transport.
/// A thin wrapper that constructs a tarpc [`ControlClient`](pie_control::ControlClient);
/// the single-node launcher calls [`bind`] / [`run_with`] with the embedded adapter.
pub async fn run(config: GatewayConfig) -> Result<()> {
    let control = controller::connect_controller(&config.controller).await?;
    run_with(config, control).await
}

/// [`bind`] then [`serve`](Gateway::serve), over an injected [`GatewayControl`]
/// backend. The single-node launcher calls this with `EmbeddedControl(handle)`;
/// the distributed path goes through [`run`].
pub async fn run_with<C: GatewayControl>(config: GatewayConfig, control: C) -> Result<()> {
    bind(config, control).await?.serve().await
}
