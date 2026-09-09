use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, watch};

use controller_api::{Ack, Role, RoutingTable, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};

use crate::state::Cluster;
use crate::topology::{Topology, reassign, routing_only};

pub enum Command {
    RegisterWorker {
        role: Role,
        model: String,
        addr: String,
        reply: oneshot::Sender<WorkerId>,
    },
    RegisterGateway {
        addr: String,
        reply: oneshot::Sender<GatewayId>,
    },
    Heartbeat {
        node: NodeId,
        reply: oneshot::Sender<Ack>,
    },
    ReportWorker {
        id: WorkerId,
        status: WorkerStatus,
    },
    Tick,
}

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

pub struct Actor {
    cluster: Cluster,
    worker_tx: watch::Sender<Topology>,
    gateway_tx: watch::Sender<RoutingTable>,
    cmd_rx: mpsc::Receiver<Command>,
    config: ActorConfig,
}

impl Actor {
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
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    fn register_gateway(&mut self, addr: String) -> GatewayId {
        let id = self.cluster.insert_gateway(addr, Instant::now());
        self.cluster.worker_epoch += 1;
        self.publish_worker_view();
        id
    }

    fn report_worker(&mut self, id: WorkerId, status: WorkerStatus) {
        if let Some(true) = self.cluster.report(id, status, Instant::now()) {
            self.cluster.gateway_epoch += 1;
            let _ = self.gateway_tx.send(routing_only(&self.cluster));
        }
    }

    fn tick(&mut self) {
        let (workers_removed, gateways_removed) = self
            .cluster
            .evict_expired(Instant::now(), self.config.heartbeat_timeout);
        if workers_removed > 0 {
            self.cluster.worker_epoch += 1;
            self.cluster.gateway_epoch += 1;
            self.replan_and_publish();
        } else if gateways_removed > 0 {
            self.cluster.worker_epoch += 1;
            self.publish_worker_view();
        }
    }

    fn replan_and_publish(&mut self) {
        let (topology, routing) = reassign(&self.cluster);
        self.cache_neighbors(&topology);
        let _ = self.worker_tx.send(topology);
        let _ = self.gateway_tx.send(routing);
    }

    fn publish_worker_view(&mut self) {
        let (topology, _routing) = reassign(&self.cluster);
        self.cache_neighbors(&topology);
        let _ = self.worker_tx.send(topology);
    }

    fn cache_neighbors(&mut self, topology: &Topology) {
        for (id, peers) in &topology.peers {
            if let Some(w) = self.cluster.workers.get_mut(id) {
                w.neighbors = peers.iter().map(|p| p.id).collect();
            }
        }
    }
}
