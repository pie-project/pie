//! Single-node **composition root** (`single-node` feature).
//!
//! The worker binary is the one place that legitimately depends on *both*
//! `pie-controller` (the controller actor) and `pie-gateway` (the edge plane),
//! so the in-proc adapter that bridges them lives here — never in a library,
//! which would re-introduce the very `→ pie-controller` edge the `pie-control`
//! contract crate was carved out to forbid.
//!
//! [`assemble`] embeds the controller actor on the engine runtime, then wires
//! the three planes into one process over loopback:
//!
//! ```text
//! client ─ws→ in-proc gateway ←──dial-in── worker engine
//!                    │       (worker serves WorkerControl +     │
//!                    │        pushes tokens back; loopback)     │
//!                    └──── embedded controller Handle ──────────┘
//!                          (register / heartbeat / watch)
//! ```
//!
//! A single [`pie_controller::embed`] yields a cloneable `Handle`; the
//! [`EmbeddedControl`] newtype wraps it and implements both the worker-side
//! [`ControlLink`] and the gateway-side [`pie_gateway::GatewayControl`] seams
//! against the in-proc calls (registration is infallible in-proc; watches return
//! the controller's `watch::Receiver` directly, no epoch long-poll).

use std::net::{SocketAddr, ToSocketAddrs};

use anyhow::{Context, Result, anyhow};
use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, Role, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};
use tokio::sync::watch;

use super::control_link::{self, ControlLink};
use super::{ControlPlane, EdgeServer, gateway_link};
use crate::config;
use crate::embedded_driver::DriverCapabilities;

/// In-proc adapter over the controller [`Handle`](pie_controller::Handle),
/// implementing both control-plane seams so the worker loops and the in-proc
/// gateway talk to the embedded actor directly (no tarpc, no serialization).
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
        // Charlie's direct path: the controller's live global receiver — the
        // actor is the sole writer, so the gateway's per-session `borrow()` is
        // always current and no republish loop is needed in-proc.
        self.0.gateway_watch()
    }
}

/// Assemble the single-node in-proc cluster on the engine runtime.
///
/// Embeds the controller, binds the in-proc gateway (client-facing edge on the
/// user's `host:port`, worker-facing data plane on a loopback ephemeral port),
/// registers the worker, then **dials the worker INTO the gateway** (M3) and
/// serves `WorkerControl` over that link — the in-proc mirror of the distributed
/// path. Returns the worker's edge link(s), its control-loop tasks, the live
/// control-plane resources, and the advertised `ws://` URL — matching the
/// distributed arm of [`super::assemble_control_and_edge`].
pub(super) async fn assemble(
    user_cfg: &config::Config,
    model: String,
    caps: DriverCapabilities,
) -> Result<(
    EdgeServer,
    Vec<tokio::task::JoinHandle<()>>,
    ControlPlane,
    String,
)> {
    // 1. Embed the controller actor on the engine runtime (no socket). The
    //    Handle is cheaply cloneable: one clone backs the worker's control loops,
    //    another the in-proc gateway, both talking to the same single-writer actor.
    let handle = pie_controller::embed(pie_controller::Config::default());
    let embedded = EmbeddedControl(handle.clone());

    // 2. Bind the in-proc gateway: the client-facing edge on the user's
    //    `host:port` and the worker-facing data plane on an ephemeral loopback
    //    port. `bind` registers the gateway, binds BOTH sockets, and starts the
    //    worker accept loop before returning — so the worker can dial in
    //    immediately (no client-listen race; register-first lands first).
    let listen = resolve_listen(&user_cfg.server.host, user_cfg.server.port)?;
    let gateway_config = pie_gateway::GatewayConfig {
        listen,
        worker_listen: SocketAddr::from(([127, 0, 0, 1], 0)),
        controller: String::new(),
    };
    let gw = pie_gateway::bind(gateway_config, embedded.clone())
        .await
        .context("binding in-proc gateway")?;
    let worker_dial = format!("tcp://{}", gw.worker_addr);
    let listen_addr = gw.listen_addr;

    // 3. Register the worker with the embedded controller (control plane) so it
    //    appears Healthy in the gateway's routing table, then dial INTO the
    //    gateway's worker-facing socket (data plane, M3) and serve
    //    `WorkerControl`. A single-node worker serves all stages; routing doesn't
    //    filter by role yet, so `Decode` is an inert default (echo owns the
    //    future `Role::Monolithic`).
    let info = WorkerInfo {
        role: Role::Decode,
        model,
        // Vestigial post-inversion: the gateway dispatches via its dial-in
        // registry (keyed by WorkerId), not by dialing this address.
        addr: gw.worker_addr.to_string(),
        capability: caps,
    };
    let worker_id = ControlLink::register_worker(&embedded, info)
        .await
        .context("registering worker with in-proc controller")?;
    let control_tasks = control_link::spawn_control_tasks(embedded, worker_id);
    let link = gateway_link::connect_gateway(&worker_dial, worker_id)
        .await
        .context("dialing in-proc gateway")?;

    // 4. Serve the gateway's client-facing edge (its worker accept loop is
    //    already live from `bind`).
    let gateway_task = tokio::spawn(async move {
        if let Err(e) = gw.serve().await {
            tracing::error!(error = %e, "in-proc gateway exited");
        }
    });

    let url = format!("ws://{listen_addr}");
    Ok((
        EdgeServer::GatewayLinks(vec![link]),
        control_tasks,
        ControlPlane::Embedded {
            _handle: handle,
            worker_id,
            gateway_task,
        },
        url,
    ))
}

/// Resolve a `host`/`port` config pair into a bindable [`SocketAddr`] for the
/// in-proc gateway's client-facing listener.
fn resolve_listen(host: &str, port: u16) -> Result<SocketAddr> {
    (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolving gateway listen address {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow!("no socket address resolved for {host}:{port}"))
}
