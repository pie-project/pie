//! `pie-controller` — Pie's cluster **control plane**.
//!
//! A registry of workers + gateways behind a **single-writer actor**. Workers
//! long-poll their `Neighbors` (who to coordinate with); gateways long-poll the
//! global `RoutingTable` (the worker roster + coarse load) to route locally.
//! Liveness is tracked from heartbeats (controller-side clock); a background
//! reaper evicts the silent. It is control plane only — tokens and KV never
//! transit it.
//!
//! # Shape
//!
//! ```text
//! service.rs  tarpc `Control` server  ─┐
//! Handle      in-proc front door      ─┼─► mpsc ─► actor.rs (sole writer)
//!                                       │            owns state.rs (Cluster)
//! reaper tick ── Command::Tick ────────┘            publishes 2 watch channels
//!                                                     topology.rs (pure planner)
//! ```
//!
//! Two deployment forms, one actor:
//! - **distributed**: [`run`] serves the `Control` RPC; workers/gateways dial it.
//! - **single-node**: [`embed`] returns a [`Handle`] the worker/gateway use
//!   in-proc — no socket, no serialization.
//!
//! # Invariants (§2/§3)
//!
//! One owner (no locks). Two **independent** epochs, each a version tag on a
//! **full** scoped snapshot (never a delta) → watches are idempotent and
//! self-healing. `role` is immutable; only join/leave moves topology. A coarse
//! load-bucket crossing re-versions only the gateway view (the load/membership
//! split that prevents watch storms).

mod actor;
mod service;
mod state;
mod store;
mod topology;

pub use store::{SoftState, StateStore};

use std::io;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};

use actor::{Actor, ActorConfig, Command};
use topology::{Topology, empty_routing, project};

/// Long-poll hold time. Must be **less than** the watch RPC deadline clients set
/// (~30s) so a no-change watch returns as a keepalive before the call times out.
const T_HANG: Duration = Duration::from_secs(20);

/// Controller configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Address the RPC server binds: `tcp://host:port`, a bare `host:port`, or
    /// `unix:/path`. Unused by [`embed`].
    pub listen_addr: String,
    /// Evict a member after this long without a liveness signal.
    pub heartbeat_timeout: Duration,
    /// How often the reaper scans for expired members.
    pub tick_interval: Duration,
    /// Command-channel buffer depth.
    pub command_buffer: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:7000".to_string(),
            // ≈4× the 2s client heartbeat: tolerates a couple of missed beats
            // (no false evictions) while detecting death in ~8–10s.
            heartbeat_timeout: Duration::from_secs(8),
            tick_interval: Duration::from_secs(2),
            command_buffer: 256,
        }
    }
}

/// In-process front door to the controller actor — cloneable and cheap. Mirrors
/// the `Control` RPC calls (minus the tarpc context) so a single-node worker /
/// gateway can embed the controller and talk to it directly. For watches it
/// offers a directly-subscribable [`watch::Receiver`] (no epoch cursor needed
/// in-proc), while the distributed RPC server reuses the epoch long-poll helpers.
#[derive(Clone)]
pub struct Handle {
    cmd: mpsc::Sender<Command>,
    worker_rx: watch::Receiver<Topology>,
    gateway_rx: watch::Receiver<RoutingTable>,
}

impl Handle {
    /// Register a worker; returns its controller-minted [`WorkerId`].
    pub async fn register_worker(&self, info: WorkerInfo) -> WorkerId {
        let (reply, rx) = oneshot::channel();
        let _ = self
            .cmd
            .send(Command::RegisterWorker {
                role: info.role,
                model: info.model,
                addr: info.addr,
                reply,
            })
            .await;
        rx.await.expect("controller actor stopped")
    }

    /// Register a gateway; returns its controller-minted [`GatewayId`].
    pub async fn register_gateway(&self, info: GatewayInfo) -> GatewayId {
        let (reply, rx) = oneshot::channel();
        let _ = self
            .cmd
            .send(Command::RegisterGateway {
                addr: info.addr,
                reply,
            })
            .await;
        rx.await.expect("controller actor stopped")
    }

    /// Liveness ping; [`Ack::ReRegister`] means the controller has no record of
    /// `id` (restart / timeout) and the node must re-register.
    pub async fn heartbeat(&self, id: NodeId) -> Ack {
        let (reply, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::Heartbeat { node: id, reply }).await;
        rx.await.expect("controller actor stopped")
    }

    /// Push a worker's coarse load (write-only).
    pub async fn report_worker(&self, id: WorkerId, status: WorkerStatus) {
        let _ = self.cmd.send(Command::ReportWorker { id, status }).await;
    }

    /// Directly subscribe to a worker's neighbor view (epoch-free single-node
    /// path). A small projector task forwards each membership change as the
    /// worker's scoped [`Neighbors`]; await `changed()` on the returned receiver.
    pub fn worker_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        let mut topo_rx = self.worker_rx.clone();
        let initial = project(&topo_rx.borrow(), id);
        let (tx, rx) = watch::channel(initial);
        tokio::spawn(async move {
            while topo_rx.changed().await.is_ok() {
                let view = project(&topo_rx.borrow(), id);
                if tx.send(view).is_err() {
                    break; // subscriber dropped
                }
            }
        });
        rx
    }

    /// Directly subscribe to the global routing table (epoch-free single-node
    /// path). The gateway view is already global, so this is the raw receiver.
    pub fn gateway_watch(&self) -> watch::Receiver<RoutingTable> {
        self.gateway_rx.clone()
    }

    /// Epoch long-poll for `watch_worker` (distributed read-path): block until
    /// the worker epoch passes `since`, then return the scoped view; on a hang
    /// timeout return the current view (same-epoch keepalive → client re-polls).
    pub(crate) async fn watch_worker_poll(&self, id: WorkerId, since: u64) -> Neighbors {
        let mut rx = self.worker_rx.clone();
        loop {
            if rx.borrow().epoch > since {
                return project(&rx.borrow(), id);
            }
            match tokio::time::timeout(T_HANG, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return project(&rx.borrow(), id),
            }
        }
    }

    /// Epoch long-poll for `watch_gateway`.
    pub(crate) async fn watch_gateway_poll(&self, since: u64) -> RoutingTable {
        let mut rx = self.gateway_rx.clone();
        loop {
            if rx.borrow().epoch > since {
                return rx.borrow().clone();
            }
            match tokio::time::timeout(T_HANG, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return rx.borrow().clone(),
            }
        }
    }
}

/// Spawn the actor + reaper tick and return the in-process [`Handle`]. No socket
/// (single-node embed). Must be called from within a Tokio runtime.
pub fn embed(config: Config) -> Handle {
    let (cmd_tx, cmd_rx) = mpsc::channel(config.command_buffer);
    let (worker_tx, worker_rx) = watch::channel(Topology::default());
    let (gateway_tx, gateway_rx) = watch::channel(empty_routing());

    let actor = Actor::new(
        cmd_rx,
        worker_tx,
        gateway_tx,
        ActorConfig {
            heartbeat_timeout: config.heartbeat_timeout,
        },
    );
    tokio::spawn(actor.run());

    // The reaper is just a timer feeding `Command::Tick` into the one actor.
    let tick_cmd = cmd_tx.clone();
    let interval = config.tick_interval;
    tokio::spawn(async move {
        let mut timer = tokio::time::interval(interval);
        loop {
            timer.tick().await;
            if tick_cmd.send(Command::Tick).await.is_err() {
                break; // actor stopped
            }
        }
    });

    Handle {
        cmd: cmd_tx,
        worker_rx,
        gateway_rx,
    }
}

/// Run the controller as a server: embed the actor and serve the `Control` RPC
/// over tarpc (tcp + unix). Returns the [`Handle`] (for any in-proc co-tenant)
/// and the spawned accept-loop task, which runs until the listener closes.
pub async fn run(config: Config) -> io::Result<(Handle, JoinHandle<()>)> {
    let handle = embed(config.clone());
    let serve = service::serve(&config.listen_addr, handle.clone()).await?;
    Ok((handle, serve))
}
