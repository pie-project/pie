//! The actor — the **single writer**. It owns the [`Cluster`] outright and is the
//! only task that mutates it: every RPC handler and the in-process [`Handle`]
//! send a [`Command`] and await a reply. So "mutate → bump epoch → publish"
//! happens atomically in one place, with no locks and no shared map.
//!
//! [`Handle`]: crate::Handle
//!
//! Write-path (§7), the load-bearing rules:
//! - **RegisterWorker**: insert → bump **both** epochs → replan → publish both.
//! - **RegisterGateway**: insert → bump **worker_epoch** → publish the worker
//!   view so every worker learns the new gateway to dial INTO. The gateway's own
//!   `RoutingTable` (worker roster) is unchanged, so gateway_epoch stays put.
//! - **Heartbeat**: refresh `last_hb`; unknown id → `Ack::ReRegister`.
//! - **ReportWorker**: overwrite coarse load; **only if the bucket crossed**,
//!   bump **gateway_epoch** and republish routing — never the worker epoch
//!   (load never re-pairs anyone). This load/membership split keeps the control
//!   plane from becoming a fan-out amplifier.
//! - **Tick** (reaper): evict expired members; a **worker** leaving bumps both +
//!   publishes; a **gateway**-only removal bumps worker_epoch + publishes the
//!   worker view (workers must stop dialing the departed gateway).

use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, watch};

use pie_controller_rpc::{Ack, Role, RoutingTable, WorkerStatus};
use pie_ids::{GatewayId, NodeId, WorkerId};

use crate::state::Cluster;
use crate::topology::{Topology, reassign, routing_only};

/// Messages the actor processes. The only way to touch cluster state. The
/// service extracts these from the wire calls (e.g. `register_worker` drops the
/// unused `capability` field).
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
        // Membership change → both views move.
        self.cluster.worker_epoch += 1;
        self.cluster.gateway_epoch += 1;
        self.replan_and_publish();
        id
    }

    fn register_gateway(&mut self, addr: String) -> GatewayId {
        let id = self.cluster.insert_gateway(addr, Instant::now());
        // A new gateway joins the global roster → every worker must dial it.
        // Only the worker-facing view moves; the RoutingTable (worker roster)
        // is unchanged, so gateway_epoch stays put.
        self.cluster.worker_epoch += 1;
        self.publish_worker_view();
        id
    }

    fn report_worker(&mut self, id: WorkerId, status: WorkerStatus) {
        if let Some(true) = self.cluster.report(id, status, Instant::now()) {
            // Coarse bucket crossed → re-version the gateway view only.
            self.cluster.gateway_epoch += 1;
            let _ = self.gateway_tx.send(routing_only(&self.cluster));
        }
        // bucket unchanged or unknown id → no version change (an unknown id
        // learns it is gone on its next heartbeat).
    }

    fn tick(&mut self) {
        let (workers_removed, gateways_removed) = self
            .cluster
            .evict_expired(Instant::now(), self.config.heartbeat_timeout);
        if workers_removed > 0 {
            // A worker left → topology changed → both views move. `reassign`
            // reads the post-eviction cluster, so this also carries any gateway
            // that expired in the same tick.
            self.cluster.worker_epoch += 1;
            self.cluster.gateway_epoch += 1;
            self.replan_and_publish();
        } else if gateways_removed > 0 {
            // Only a gateway left → the worker-facing roster shrank (workers must
            // stop dialing it), but the RoutingTable (worker roster) is unchanged.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::empty_routing;

    fn spawn() -> (
        mpsc::Sender<Command>,
        watch::Receiver<Topology>,
        watch::Receiver<RoutingTable>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let (worker_tx, worker_rx) = watch::channel(Topology::default());
        let (gateway_tx, gateway_rx) = watch::channel(empty_routing());
        let actor = Actor::new(cmd_rx, worker_tx, gateway_tx, ActorConfig::default());
        tokio::spawn(actor.run());
        (cmd_tx, worker_rx, gateway_rx)
    }

    async fn register(cmd: &mpsc::Sender<Command>) -> WorkerId {
        let (reply, rx) = oneshot::channel();
        cmd.send(Command::RegisterWorker {
            role: Role::Prefill,
            model: "m".to_string(),
            addr: "127.0.0.1:0".to_string(),
            reply,
        })
        .await
        .unwrap();
        rx.await.unwrap()
    }

    async fn register_gateway(cmd: &mpsc::Sender<Command>, addr: &str) -> GatewayId {
        let (reply, rx) = oneshot::channel();
        cmd.send(Command::RegisterGateway {
            addr: addr.to_string(),
            reply,
        })
        .await
        .unwrap();
        rx.await.unwrap()
    }

    #[tokio::test]
    async fn register_worker_bumps_both_epochs_and_publishes() {
        let (cmd, mut wrx, mut grx) = spawn();
        let a = register(&cmd).await;
        assert_eq!(a, WorkerId(0));

        wrx.changed().await.unwrap();
        grx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);
        assert_eq!(grx.borrow().epoch, 1);

        let b = register(&cmd).await;
        assert_eq!(b, WorkerId(1));
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 2);
        // a's neighbor list now includes b
        assert_eq!(wrx.borrow().peers[&a].len(), 1);
        assert_eq!(wrx.borrow().peers[&a][0].id, b);
    }

    #[tokio::test]
    async fn report_bumps_gateway_epoch_only_on_bucket_cross() {
        let (cmd, mut wrx, mut grx) = spawn();
        let a = register(&cmd).await;
        wrx.changed().await.unwrap();
        grx.changed().await.unwrap();

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
        assert_eq!(grx.borrow().workers[0].coarse_load.kv_pressure_bucket, 3);
        assert_eq!(wrx.borrow().epoch, 1, "load never moves the worker epoch");
    }

    #[tokio::test]
    async fn unknown_heartbeat_acks_reregister() {
        let (cmd, _wrx, _grx) = spawn();
        let (reply, rx) = oneshot::channel();
        cmd.send(Command::Heartbeat {
            node: NodeId::Worker(WorkerId(42)),
            reply,
        })
        .await
        .unwrap();
        assert_eq!(rx.await.unwrap(), Ack::ReRegister);
    }

    #[tokio::test]
    async fn register_gateway_bumps_worker_epoch_and_publishes_roster() {
        let (cmd, mut wrx, mut grx) = spawn();
        // A worker first, so both views are at epoch 1 with an empty roster.
        let _w = register(&cmd).await;
        wrx.changed().await.unwrap();
        grx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);
        assert_eq!(grx.borrow().epoch, 1);
        assert!(wrx.borrow().gateways.is_empty());

        // Registering a gateway moves ONLY the worker-facing view: the roster
        // now carries the gateway, but the worker-roster (gateway) view is
        // untouched (no worker joined/left).
        let g = register_gateway(&cmd, "127.0.0.1:8000").await;
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 2);
        assert_eq!(wrx.borrow().gateways.len(), 1);
        assert_eq!(wrx.borrow().gateways[0].id, g);
        assert_eq!(wrx.borrow().gateways[0].addr, "127.0.0.1:8000");
        assert_eq!(
            grx.borrow().epoch,
            1,
            "gateway join never moves the worker roster"
        );
    }

    #[tokio::test]
    async fn tick_evicts_expired_gateway_and_republishes_worker_view() {
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let (worker_tx, mut wrx) = watch::channel(Topology::default());
        let (gateway_tx, grx) = watch::channel(empty_routing());
        let actor = Actor::new(
            cmd_rx,
            worker_tx,
            gateway_tx,
            ActorConfig {
                heartbeat_timeout: Duration::ZERO,
            },
        );
        tokio::spawn(actor.run());

        let _g = register_gateway(&cmd_tx, "127.0.0.1:8000").await;
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);
        assert_eq!(wrx.borrow().gateways.len(), 1);
        let gw_view_epoch = grx.borrow().epoch;

        tokio::time::sleep(Duration::from_millis(2)).await;
        cmd_tx.send(Command::Tick).await.unwrap();
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 2);
        assert!(
            wrx.borrow().gateways.is_empty(),
            "evicted gateway leaves the roster"
        );
        assert_eq!(
            grx.borrow().epoch,
            gw_view_epoch,
            "gateway-only eviction never moves the worker roster"
        );
    }

    #[tokio::test]
    async fn tick_evicts_expired_worker_and_republishes() {
        let (cmd_tx, cmd_rx) = mpsc::channel(16);
        let (worker_tx, mut wrx) = watch::channel(Topology::default());
        let (gateway_tx, _grx) = watch::channel(empty_routing());
        let actor = Actor::new(
            cmd_rx,
            worker_tx,
            gateway_tx,
            ActorConfig {
                heartbeat_timeout: Duration::ZERO,
            },
        );
        tokio::spawn(actor.run());

        let _a = register(&cmd_tx).await;
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 1);

        tokio::time::sleep(Duration::from_millis(2)).await;
        cmd_tx.send(Command::Tick).await.unwrap();
        wrx.changed().await.unwrap();
        assert_eq!(wrx.borrow().epoch, 2);
        assert!(wrx.borrow().peers.is_empty(), "worker evicted");
    }
}
