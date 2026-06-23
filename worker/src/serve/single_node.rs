//! Single-node **composition root** (`single-node` feature).
//!
//! The worker binary is the one place that legitimately depends on *both*
//! `pie-controller` (the controller actor) and `pie-gateway` (the edge plane),
//! so the in-proc adapter that bridges them lives here — never in a library,
//! which would re-introduce the very `→ pie-controller` edge the `pie-control`
//! contract crate was carved out to forbid.
//!
//! [`assemble`] embeds the controller actor on the engine runtime, then wires
//! the three planes into one process with no sockets between them:
//!
//! ```text
//! client ─ws→ in-proc gateway ─edge-rpc(loopback)→ worker engine
//!                    │                    │
//!                    └──── embedded controller Handle ────┘
//!                         (register / heartbeat / watch)
//! ```
//!
//! A single [`pie_controller::embed`] yields a cloneable `Handle`; the
//! [`EmbeddedControl`] newtype wraps it and implements both the worker-side
//! [`ControlLink`] and the gateway-side [`pie_gateway::GatewayControl`] seams
//! against the in-proc calls (registration is infallible in-proc; watches return
//! the controller's `watch::Receiver` directly, no epoch long-poll).

use std::net::{SocketAddr, ToSocketAddrs};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, Role, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};
use tokio::sync::watch;

use super::control_link::{self, ControlLink};
use super::{ControlPlane, EdgeServer, edge_session};
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
/// Embeds the controller, binds the worker's edge-rpc on an ephemeral loopback
/// port, registers the worker, spawns its control loops, and runs the gateway
/// in this process (serving the client WebSocket on the user's `host:port`,
/// routing sessions to the loopback edge-rpc). Returns the worker's edge server,
/// its control-loop tasks, the live control-plane resources, and the advertised
/// `ws://` URL — matching the distributed arm of
/// [`super::assemble_control_and_edge`].
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
    //    Handle is cheaply cloneable: one clone drives the worker loops, another
    //    the in-proc gateway, both talking to the same single-writer actor.
    let handle = pie_controller::embed(pie_controller::Config::default());
    let embedded = EmbeddedControl(handle.clone());

    // 2. Bind the worker's edge-rpc on an ephemeral loopback port; the in-proc
    //    gateway dials it. The user's `host:port` is the gateway's WS listen, so
    //    the worker's internal endpoint must be a separate address.
    let edge = edge_session::spawn("127.0.0.1:0")
        .await
        .context("starting worker edge-rpc server")?;
    let worker_addr = edge.bound.clone(); // tcp://127.0.0.1:<port>

    // 3. Register the worker so it appears in the gateway's routing table. A
    //    single-node worker serves all stages; routing doesn't filter by role
    //    yet, so the role is inert for selection — register as Decode (the
    //    steady-state role that holds a session through generation).
    let info = WorkerInfo {
        role: Role::Decode,
        model,
        addr: worker_addr,
        capability: caps,
    };
    let worker_id = ControlLink::register_worker(&embedded, info)
        .await
        .context("registering worker with in-proc controller")?;
    let control_tasks = control_link::spawn_control_tasks(embedded.clone(), worker_id);

    // 4. Run the gateway in-proc: it serves the client WebSocket on the user's
    //    host:port and routes each session to the worker's loopback edge-rpc,
    //    reading the registered worker from the embedded controller's routing
    //    table. `controller` is unused — `run_with` takes the adapter directly.
    let listen = resolve_listen(&user_cfg.server.host, user_cfg.server.port)?;
    let gateway_config = pie_gateway::GatewayConfig {
        listen,
        controller: String::new(),
    };
    let gateway_embedded = embedded;
    let gateway_task = tokio::spawn(async move {
        if let Err(e) = pie_gateway::run_with(gateway_config, gateway_embedded).await {
            tracing::error!(error = %e, "in-proc gateway exited");
        }
    });

    // `run_with` binds its listener asynchronously (after registering + spawning
    // its loops), but one-shot clients (`pie run`) connect immediately once we
    // return. Wait until the gateway is accepting so that connect doesn't race
    // the bind.
    wait_until_listening(listen)
        .await
        .context("in-proc gateway failed to start listening")?;

    let url = format!("ws://{listen}");
    Ok((
        EdgeServer::WorkerListener(edge),
        control_tasks,
        ControlPlane::Embedded {
            _handle: handle,
            worker_id,
            gateway_task,
        },
        url,
    ))
}

/// Poll-connect to `addr` until the in-proc gateway's listener accepts, so we
/// only advertise its URL once it is reachable. Bounded so a gateway that fails
/// to bind (e.g. the port is in use) surfaces as an error rather than hanging.
async fn wait_until_listening(addr: SocketAddr) -> Result<()> {
    const PROBE_INTERVAL: Duration = Duration::from_millis(25);
    const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
    let deadline = tokio::time::Instant::now() + PROBE_TIMEOUT;
    loop {
        match tokio::net::TcpStream::connect(addr).await {
            Ok(_) => return Ok(()),
            Err(e) if tokio::time::Instant::now() >= deadline => {
                return Err(anyhow!(
                    "gateway not listening on {addr} after {PROBE_TIMEOUT:?}: {e}"
                ));
            }
            Err(_) => tokio::time::sleep(PROBE_INTERVAL).await,
        }
    }
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
