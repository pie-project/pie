//! P5a compose — the real in-proc standalone: an embedded controller actor, a
//! gateway, and a worker co-resident over loopback. golf's overlay of delta's
//! stub contract (`Mode`, `StandaloneHandle`, `run_standalone`).
//!
//! Topology is the same M3 dial-in as a real cluster, just collapsed into one
//! process: the controller actor is embedded and a single cloneable `Handle`
//! drives BOTH control planes through [`EmbeddedControl`] (no control sockets).
//! The gateway binds an ephemeral loopback worker-facing port; the embedded
//! worker dials INTO it via [`pie_worker::run_with`], exactly as a remote worker
//! would. The gateway's client edge is then served on `listen_addr`.

use std::net::{Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};
use pie_worker::ControlLink;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// Standalone run mode. `Local` and `Serve` boot the *same* in-proc cluster;
/// they differ only in client-facing exposure (loopback vs the configured bind).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `pie local` — loopback-only, developer one-shot/local use.
    Local,
    /// `pie serve` — bind the configured client address; persistent.
    Serve,
}

/// In-proc adapter over the embedded controller `Handle`, implementing BOTH the
/// worker [`ControlLink`] and the gateway [`pie_gateway::GatewayControl`] seams
/// against the same actor. Registration is infallible (no transport) and the
/// watches forward the controller's own receivers, so the co-resident roles
/// observe topology/routing updates with zero network hops.
#[derive(Clone)]
struct EmbeddedControl(pie_controller::Handle);

impl ControlLink for EmbeddedControl {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        Ok(self.0.register_worker(info).await)
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        Ok(self.0.heartbeat(id).await)
    }

    async fn report_worker(&self, id: WorkerId, status: WorkerStatus) -> Result<()> {
        self.0.report_worker(id, status).await;
        Ok(())
    }

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        self.0.worker_watch(id)
    }
}

impl pie_gateway::GatewayControl for EmbeddedControl {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        Ok(self.0.register_gateway(info).await)
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        Ok(self.0.heartbeat(id).await)
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.0.gateway_watch()
    }
}

/// A running in-proc standalone (embedded controller + gateway + worker over
/// loopback). Owns the three role handles; [`shutdown`](Self::shutdown) drains
/// all three.
pub struct StandaloneHandle {
    /// The resolved client-facing listen address (after an ephemeral bind).
    pub listen_addr: SocketAddr,
    /// The resolved worker dial-in address the embedded worker connected to.
    pub worker_addr: SocketAddr,
    /// Keeps the embedded controller actor alive; dropping the last `Handle`
    /// closes its command channel and the actor task winds down.
    _controller: pie_controller::Handle,
    worker: pie_worker::WorkerHandle,
    gateway: JoinHandle<()>,
}

impl StandaloneHandle {
    /// Drain + stop all three embedded planes cleanly: stop accepting client
    /// traffic first, then drain in-flight turns in the worker, then release the
    /// controller actor.
    pub async fn shutdown(self) {
        self.gateway.abort();
        self.worker.shutdown().await;
        // `_controller` drops here, retiring the actor.
    }
}

/// Boot the embedded controller + gateway + worker over loopback from the
/// pre-derived typed Configs (delta's `derive_standalone`) and return a handle.
pub async fn run_standalone(
    controller: pie_controller::Config,
    mut gateway: pie_gateway::Config,
    worker: pie_worker::Config,
    mode: Mode,
) -> Result<StandaloneHandle> {
    // Embed the controller actor; one cloneable Handle drives both planes.
    let handle = pie_controller::embed(controller);
    let control = EmbeddedControl(handle.clone());

    // The in-proc gateway binds its worker-facing socket on an ephemeral
    // loopback port so the embedded worker can dial in; `Local` also forces the
    // client edge to loopback (`Serve` keeps the configured bind).
    gateway.worker_listen = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
    if mode == Mode::Local {
        gateway.listen.set_ip(Ipv4Addr::LOCALHOST.into());
    }
    let gw = pie_gateway::bind(gateway, control.clone())
        .await
        .context("bind in-proc gateway")?;
    let listen_addr = gw.listen_addr;
    let worker_addr = gw.worker_addr;

    // Boot the embedded worker against the injected control link, dialing INTO
    // the in-proc gateway (M3 inversion — the same path a remote worker takes).
    let worker = pie_worker::run_with(worker, control, vec![format!("tcp://{worker_addr}")])
        .await
        .context("boot embedded worker")?;

    // Serve the gateway client edge (its worker-facing accept loop is already up).
    let gateway = tokio::spawn(async move {
        if let Err(e) = gw.serve().await {
            tracing::error!(error = %e, "in-proc gateway exited");
        }
    });

    Ok(StandaloneHandle {
        listen_addr,
        worker_addr,
        _controller: handle,
        worker,
        gateway,
    })
}
