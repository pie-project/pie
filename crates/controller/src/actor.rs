use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, watch};

use controller_api::{Ack, RoutingTable, ScaleEntry, ScaleStatus, WorkerInfo, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};

use crate::actuator::Actuator;
use crate::state::Cluster;
use crate::topology::{Topology, reassign, routing_only};

pub enum Command {
    RegisterWorker {
        info: WorkerInfo,
        reply: oneshot::Sender<WorkerId>,
    },
    RegisterGateway {
        addr: String,
        prior: Option<GatewayId>,
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
    DesireScale {
        entries: Vec<ScaleEntry>,
    },
    ScaleStatus {
        reply: oneshot::Sender<Vec<ScaleStatus>>,
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
    actuator: Actuator,
}

impl Actor {
    pub fn new(
        cmd_rx: mpsc::Receiver<Command>,
        worker_tx: watch::Sender<Topology>,
        gateway_tx: watch::Sender<RoutingTable>,
        config: ActorConfig,
        actuator: Actuator,
    ) -> Self {
        let mut cluster = Cluster::new();
        cluster.scales = actuator.scales();
        Self {
            cluster,
            worker_tx,
            gateway_tx,
            cmd_rx,
            config,
            actuator,
        }
    }

    pub async fn run(mut self) {
        while let Some(cmd) = self.cmd_rx.recv().await {
            match cmd {
                Command::RegisterWorker { info, reply } => {
                    let _ = reply.send(self.register_worker(info));
                }
                Command::RegisterGateway { addr, prior, reply } => {
                    let _ = reply.send(self.register_gateway(addr, prior));
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
                Command::DesireScale { entries } => {
                    self.actuator.desire(entries);
                    self.actuator.converge(&self.cluster.workers);
                }
                Command::ScaleStatus { reply } => {
                    let _ = reply.send(self.actuator.status(&self.cluster.workers));
                }
                Command::Tick => self.tick(),
            }
        }
    }

    fn register_worker(&mut self, info: WorkerInfo) -> WorkerId {
        let id = self.cluster.insert_worker_as(
            info.prior,
            info.role,
            info.model,
            info.addr,
            Instant::now(),
        );
        if let Some(worker) = self.cluster.workers.get_mut(&id) {
            worker.engine = info.engine;
            worker.launch = info.launch;
        }
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    fn register_gateway(&mut self, addr: String, prior: Option<GatewayId>) -> GatewayId {
        let id = self.cluster.insert_gateway_as(prior, addr, Instant::now());
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
        self.actuator.converge(&self.cluster.workers);
        let (workers_removed, gateways_removed) = self
            .cluster
            .evict_expired(Instant::now(), self.config.heartbeat_timeout);
        let regraded = self
            .cluster
            .grade(Instant::now(), self.config.heartbeat_timeout / 2);
        if workers_removed > 0 {
            self.cluster.worker_epoch += 1;
            self.cluster.gateway_epoch += 1;
            self.replan_and_publish();
        } else if regraded {
            self.cluster.gateway_epoch += 1;
            let _ = self.gateway_tx.send(routing_only(&self.cluster));
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
