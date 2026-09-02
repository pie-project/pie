//! The actor — the single writer. It owns the [`Cluster`] outright and is the only task that mutates it: every RPC handler and the in-process [`Handle`](crate::Handle) send a [`Command`] and await a reply, so "mutate -> bump epoch -> publish" happens atomically in one place, with no locks and no shared map.
//!
//! Write-path rules: RegisterWorker bumps both epochs; RegisterGateway bumps only worker_epoch (workers learn the new gateway, the worker roster is unchanged); ReportWorker bumps gateway_epoch only if the load bucket crossed, never the worker epoch; Tick evicts expired members, bumping both epochs for a departed worker or only worker_epoch for a gateway-only removal.

use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, watch};

use controller_api::{Ack, Role, RoutingTable, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};

use crate::state::Cluster;
use crate::topology::{Topology, reassign, routing_only};

/// Messages the actor processes; the only way to touch cluster state.
pub enum Command {
    /// Register a worker; reply with its minted [`WorkerId`].
    RegisterWorker {
        role: Role,
        model: String,
        addr: String,
        reply: oneshot::Sender<WorkerId>,
    },
    /// Register a gateway; reply with its minted [`GatewayId`].
    RegisterGateway {
        addr: String,
        reply: oneshot::Sender<GatewayId>,
    },
    /// Liveness refresh; reply [`Ack::ReRegister`] for an unknown id.
    Heartbeat {
        node: NodeId,
        reply: oneshot::Sender<Ack>,
    },
    /// Write-only coarse load report (`report_worker` returns nothing).
    ReportWorker { id: WorkerId, status: WorkerStatus },
    /// Reaper pulse from the background timer.
    Tick,
}

/// Liveness knob. The reaper evicts a member whose last controller-side receipt
/// time is older than this.
#[derive(Debug, Clone, Copy)]
pub struct ActorConfig {
    pub heartbeat_timeout: Duration,
}

impl Default for ActorConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout: Duration::from_secs(8),
        }
    }
}

/// The sole owner of cluster state.
pub struct Actor {
    cluster: Cluster,
    worker_tx: watch::Sender<Topology>,
    gateway_tx: watch::Sender<RoutingTable>,
    cmd_rx: mpsc::Receiver<Command>,
    config: ActorConfig,
}

impl Actor {
    /// Build the actor over its command inbox and the two publish channels.
    pub fn new(
        cmd_rx: mpsc::Receiver<Command>,
        worker_tx: watch::Sender<Topology>,
        gateway_tx: watch::Sender<RoutingTable>,
        config: ActorConfig,
    ) -> Self {
        Self {
            cluster: Cluster::new(),
            worker_tx,
            gateway_tx,
            cmd_rx,
            config,
        }
    }

    /// Run until the command channel closes (all senders dropped).
    pub async fn run(mut self) {
        while let Some(cmd) = self.cmd_rx.recv().await {
            match cmd {
                Command::RegisterWorker {
                    role,
                    model,
                    addr,
                    reply,
                } => {
                    let _ = reply.send(self.register_worker(role, model, addr));
                }
                Command::RegisterGateway { addr, reply } => {
                    let _ = reply.send(self.register_gateway(addr));
                }
                Command::Heartbeat { node, reply } => {
                    let ack = if self.cluster.touch(node, Instant::now()) {
                        Ack::Ok
                    } else {
                        Ack::ReRegister
                    };
                    let _ = reply.send(ack);
                }
                Command::ReportWorker { id, status } => self.report_worker(id, status),
                Command::Tick => self.tick(),
            }
        }
    }

    fn register_worker(&mut self, role: Role, model: String, addr: String) -> WorkerId {
        let id = self
            .cluster
            .insert_worker(role, model, addr, Instant::now());
        // membership change: both views move.
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    fn register_gateway(&mut self, addr: String) -> GatewayId {
        let id = self.cluster.insert_gateway(addr, Instant::now());
        // every worker must dial the new gateway; only the worker-facing view moves.
        self.cluster.worker_epoch += 1;
        self.publish_worker_view();
        id
    }

    fn report_worker(&mut self, id: WorkerId, status: WorkerStatus) {
        if let Some(true) = self.cluster.report(id, status, Instant::now()) {
            // bucket crossed: re-version the gateway view only.
            self.cluster.gateway_epoch += 1;
            let _ = self.gateway_tx.send(routing_only(&self.cluster));
        }
    }

    fn tick(&mut self) {
        let (workers_removed, gateways_removed) = self
            .cluster
            .evict_expired(Instant::now(), self.config.heartbeat_timeout);
        if workers_removed > 0 {
            // a worker left: topology changed, both views move.
            self.cluster.worker_epoch += 1;
            self.cluster.gateway_epoch += 1;
            self.replan_and_publish();
        } else if gateways_removed > 0 {
            // only a gateway left: the worker-facing roster shrank, the RoutingTable is unchanged.
            self.cluster.worker_epoch += 1;
            self.publish_worker_view();
        }
    }

    /// Recompute both snapshots, cache each worker's neighbors, and publish both.
    fn replan_and_publish(&mut self) {
        let (topology, routing) = reassign(&self.cluster);
        self.cache_neighbors(&topology);
        let _ = self.worker_tx.send(topology);
        let _ = self.gateway_tx.send(routing);
    }

    /// Recompute and publish the **worker-facing** view only (used when the
    /// gateway roster moved but the worker roster/load did not). Caches
    /// neighbors like [`replan_and_publish`] but leaves the gateway watch alone.
    fn publish_worker_view(&mut self) {
        let (topology, _routing) = reassign(&self.cluster);
        self.cache_neighbors(&topology);
        let _ = self.worker_tx.send(topology);
    }

    /// Cache each worker's planner-assigned neighbor ids back into cluster state
    /// (the published topology remains the source of truth).
    fn cache_neighbors(&mut self, topology: &Topology) {
        for (id, peers) in &topology.peers {
            if let Some(w) = self.cluster.workers.get_mut(id) {
                w.neighbors = peers.iter().map(|p| p.id).collect();
            }
        }
    }
}

