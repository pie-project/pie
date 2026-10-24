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
    /// The cluster policies' scale table, whole.
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
                Command::DesireScale { entries } => self.desire_scale(entries),
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
            worker.pid = info.pid;
            self.actuator.registered(worker);
        }
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    /// The gateways changed: workers hear the roster, and gateways the
    /// routing table, which names the author.
    fn register_gateway(&mut self, addr: String, prior: Option<GatewayId>) -> GatewayId {
        let id = self.cluster.insert_gateway_as(prior, addr, Instant::now());
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    /// The cluster policies' scale table, from the host that runs them.
    fn desire_scale(&mut self, entries: Vec<ScaleEntry>) {
        self.actuator.desire(entries);
        self.actuator.converge(&self.cluster.workers);
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
        if workers_removed > 0 || gateways_removed > 0 {
            self.cluster.worker_epoch += 1;
            self.cluster.gateway_epoch += 1;
            self.replan_and_publish();
        } else if regraded {
            self.cluster.gateway_epoch += 1;
            let _ = self.gateway_tx.send(routing_only(&self.cluster));
        }
    }

    fn replan_and_publish(&mut self) {
        let (topology, routing) = reassign(&self.cluster);
        let _ = self.worker_tx.send(topology);
        let _ = self.gateway_tx.send(routing);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actor() -> (Actor, watch::Receiver<RoutingTable>) {
        let (_cmd, cmd_rx) = mpsc::channel(1);
        let (worker_tx, _) = watch::channel(Topology::default());
        let (gateway_tx, gateway_rx) = watch::channel(crate::topology::empty_routing());
        let actor = Actor::new(
            cmd_rx,
            worker_tx,
            gateway_tx,
            ActorConfig::default(),
            Actuator::new(Vec::new(), Duration::ZERO),
        );
        (actor, gateway_rx)
    }

    fn table(workers: u32) -> Vec<ScaleEntry> {
        vec![ScaleEntry {
            model: Some("m".into()),
            engine: "cuda".into(),
            workers,
        }]
    }

    fn desired(actor: &Actor) -> Vec<u32> {
        actor
            .actuator
            .status(&actor.cluster.workers)
            .iter()
            .map(|s| s.desired)
            .collect()
    }

    #[test]
    fn the_lowest_live_gateway_authors_and_the_next_takes_over_when_it_is_evicted() {
        let (mut actor, routing) = actor();
        assert_eq!(routing.borrow().author, None);
        let b = actor.register_gateway("b".into(), Some(GatewayId(4)));
        assert_eq!(routing.borrow().author, Some(b), "the only one");
        let a = actor.register_gateway("a".into(), Some(GatewayId(1)));
        assert_eq!(routing.borrow().author, Some(a), "a lower id takes it");
        let c = actor.register_gateway("c".into(), None);
        assert_eq!(routing.borrow().author, Some(a), "a higher one does not");

        // The author's heartbeats lapse: on the tick that evicts it, the next
        // lowest is named.
        let epoch = routing.borrow().epoch;
        actor.cluster.gateways.get_mut(&a).unwrap().last_hb =
            Instant::now() - 2 * actor.config.heartbeat_timeout;
        actor.tick();
        assert_eq!(routing.borrow().author, Some(b));
        assert!(routing.borrow().epoch > epoch, "gateways hear it");
        assert!(actor.cluster.gateways.contains_key(&c));
    }

    #[test]
    fn the_last_scale_table_is_the_one_converged_on() {
        let (mut actor, _routing) = actor();
        actor.desire_scale(table(2));
        assert_eq!(desired(&actor), [2]);
        actor.desire_scale(table(5));
        assert_eq!(desired(&actor), [5]);
    }
}
