//! The actor — the **single writer**. It owns the [`Cluster`] outright and is the
//! only task that mutates it: every RPC handler (and the in-process local path)
//! sends a [`Command`] and awaits a reply. So "mutate → bump epoch → publish"
//! happens atomically in one place, with no locks and no shared map.
//!
//! Write-path (§7), the load-bearing rules:
//! - **RegisterWorker**: insert → bump **both** epochs → replan → publish both.
//! - **RegisterGateway**: insert only (a new gateway changes neither topology
//!   nor roster; it gets the current view from its first `watch_gateway(0)`).
//! - **Heartbeat**: refresh `last_hb`; unknown id → reply `false` (→ `ReRegister`).
//! - **ReportWorker**: overwrite coarse load; **only if the bucket crossed**,
//!   bump **gateway_epoch** and republish the routing table — never the worker
//!   epoch (load never re-pairs anyone). This is the load/membership split that
//!   keeps the control plane from becoming a fan-out amplifier.
//! - **Tick** (reaper): evict expired members; a **worker** leaving bumps both +
//!   publishes; a gateway-only removal publishes nothing.

use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, watch};

use pie_schema::WorkerId;

use crate::state::{Cluster, GatewayId, GatewaySpec, NodeId, WorkerSpec, WorkerStatus};
use crate::topology::{RoutingTable, Topology, reassign, routing_only};

/// Messages the actor processes. The only way to touch cluster state.
pub enum Command {
    /// Register a worker; reply with its minted [`WorkerId`].
    RegisterWorker {
        spec: WorkerSpec,
        reply: oneshot::Sender<WorkerId>,
    },
    /// Register a gateway; reply with its minted [`GatewayId`].
    RegisterGateway {
        spec: GatewaySpec,
        reply: oneshot::Sender<GatewayId>,
    },
    /// Liveness refresh. `reply` = whether the id was known; the service maps
    /// `false` → `Ack::ReRegister`.
    Heartbeat {
        node: NodeId,
        reply: oneshot::Sender<bool>,
    },
    /// Write-only coarse load report (no reply — `report_worker` returns nothing).
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
            heartbeat_timeout: Duration::from_secs(15),
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
                Command::RegisterWorker { spec, reply } => {
                    let _ = reply.send(self.register_worker(spec));
                }
                Command::RegisterGateway { spec, reply } => {
                    let _ = reply.send(self.register_gateway(spec));
                }
                Command::Heartbeat { node, reply } => {
                    let known = self.cluster.touch(node, Instant::now());
                    let _ = reply.send(known);
                }
                Command::ReportWorker { id, status } => self.report_worker(id, status),
                Command::Tick => self.tick(),
            }
        }
    }

    fn register_worker(&mut self, spec: WorkerSpec) -> WorkerId {
        let id = self.cluster.insert_worker(spec, Instant::now());
        // Membership change → both views move.
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    fn register_gateway(&mut self, spec: GatewaySpec) -> GatewayId {
        // Insert only; the gateway's first `watch_gateway(0)` delivers the view.
        self.cluster.insert_gateway(spec, Instant::now())
    }

    fn report_worker(&mut self, id: WorkerId, status: WorkerStatus) {
        if let Some(true) = self.cluster.report(id, status, Instant::now()) {
            // Coarse bucket crossed → re-version the gateway view only.
            self.cluster.gateway_epoch += 1;
            let _ = self.gateway_tx.send(routing_only(&self.cluster));
        }
        // bucket unchanged or unknown id → no version change (unknown id learns
        // it is gone on its next heartbeat).
    }

    fn tick(&mut self) {
        let (workers_removed, _gateways_removed) = self
            .cluster
            .evict_expired(Instant::now(), self.config.heartbeat_timeout);
        if workers_removed > 0 {
            // A worker left → topology changed → both views move.
            self.cluster.worker_epoch += 1;
            self.cluster.gateway_epoch += 1;
            self.replan_and_publish();
        }
        // Gateway-only removal changes no published view → publish nothing.
    }

    /// Recompute both snapshots, cache each worker's neighbors, and publish.
    fn replan_and_publish(&mut self) {
        let (topology, routing) = reassign(&self.cluster);
        for (id, peers) in &topology.neighbors {
            if let Some(w) = self.cluster.workers.get_mut(id) {
                w.neighbors = peers.clone();
            }
        }
        let _ = self.worker_tx.send(topology);
        let _ = self.gateway_tx.send(routing);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_schema::Role;

    fn spawn() -> (
        mpsc::Sender<Command>,
        watch::Receiver<Topology>,
        watch::Receiver<RoutingTable>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let (worker_tx, worker_rx) = watch::channel(Topology::default());
        let (gateway_tx, gateway_rx) = watch::channel(RoutingTable::default());
        let actor = Actor::new(cmd_rx, worker_tx, gateway_tx, ActorConfig::default());
        tokio::spawn(actor.run());
        (cmd_tx, worker_rx, gateway_rx)
    }

    async fn register_worker(cmd: &mpsc::Sender<Command>, model: &str) -> WorkerId {
        let (reply, rx) = oneshot::channel();
        cmd.send(Command::RegisterWorker {
            spec: WorkerSpec {
                role: Role::Prefill,
                model: model.to_string(),
                addr: "127.0.0.1:0".to_string(),
            },
            reply,
        })
        .await
        .unwrap();
        rx.await.unwrap()
    }

    #[tokio::test]
    async fn register_worker_bumps_both_epochs_and_publishes() {
        let (cmd, mut wrx, mut grx) = spawn();
        let a = register_worker(&cmd, "m").await;
        assert_eq!(a, WorkerId(0));

        wrx.changed().await.unwrap();
        grx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);
        assert_eq!(grx.borrow().epoch, 1);

        let b = register_worker(&cmd, "m").await;
        assert_eq!(b, WorkerId(1));
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 2);
        // a's neighbors now include b
        assert_eq!(wrx.borrow().neighbors[&a], vec![b]);
    }

    #[tokio::test]
    async fn report_bumps_gateway_epoch_only_on_bucket_cross() {
        let (cmd, mut wrx, mut grx) = spawn();
        let a = register_worker(&cmd, "m").await;
        wrx.changed().await.unwrap();
        grx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);
        assert_eq!(grx.borrow().epoch, 1);

        // same bucket (0) → no version change
        cmd.send(Command::ReportWorker {
            id: a,
            status: WorkerStatus {
                kv_pressure_bucket: 0,
                inflight: 5,
            },
        })
        .await
        .unwrap();
        // bucket crossed (0 → 3) → gateway epoch bumps, worker epoch does not
        cmd.send(Command::ReportWorker {
            id: a,
            status: WorkerStatus {
                kv_pressure_bucket: 3,
                inflight: 9,
            },
        })
        .await
        .unwrap();
        grx.changed().await.unwrap();
        assert_eq!(grx.borrow().epoch, 2);
        assert_eq!(grx.borrow().workers[0].coarse_load, 3);
        assert_eq!(wrx.borrow().epoch, 1, "load never moves the worker epoch");
    }

    #[tokio::test]
    async fn unknown_heartbeat_reports_reregister() {
        let (cmd, _wrx, _grx) = spawn();
        let (reply, rx) = oneshot::channel();
        cmd.send(Command::Heartbeat {
            node: NodeId::Worker(WorkerId(42)),
            reply,
        })
        .await
        .unwrap();
        assert!(!rx.await.unwrap(), "unknown id → ReRegister");
    }

    #[tokio::test]
    async fn tick_evicts_expired_worker_and_republishes() {
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let (worker_tx, mut wrx) = watch::channel(Topology::default());
        let (gateway_tx, _grx) = watch::channel(RoutingTable::default());
        // zero timeout → any worker is immediately expired on the next tick
        let actor = Actor::new(
            cmd_rx,
            worker_tx,
            gateway_tx,
            ActorConfig {
                heartbeat_timeout: Duration::ZERO,
            },
        );
        tokio::spawn(actor.run());

        let _a = register_worker(&cmd_tx, "m").await;
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);

        // sleep a hair so `now - last_hb > 0`
        tokio::time::sleep(Duration::from_millis(2)).await;
        cmd_tx.send(Command::Tick).await.unwrap();
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 2);
        assert!(wrx.borrow().neighbors.is_empty(), "worker evicted");
    }
}
