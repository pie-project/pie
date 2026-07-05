//! Cluster state — the single source of truth, owned **solely** by the actor
//! ([`crate::actor`]). No locks, no sharing: every mutation runs on the actor
//! task. These are **internal** types and never cross the wire — the actor
//! publishes the `pie_controller_rpc` wire views ([`Neighbors`], [`RoutingTable`])
//! derived from this state.
//!
//! Two independent monotonic epochs version two independent full snapshots:
//! `worker_epoch` tags the worker-facing topology, `gateway_epoch` the
//! gateway-facing routing table. Membership changes bump both; a coarse
//! load-bucket crossing bumps only the gateway epoch. An epoch is a version tag
//! on a *whole* snapshot, never a delta. `role` is immutable once registered.
//!
//! [`Neighbors`]: pie_controller_rpc::Neighbors
//! [`RoutingTable`]: pie_controller_rpc::RoutingTable

use std::collections::HashMap;
use std::time::{Duration, Instant};

use pie_controller_rpc::{Role, WorkerStatus};
use pie_ids::{GatewayId, NodeId, WorkerId};

/// A registered worker. `role`/`model`/`addr` are immutable after registration.
#[derive(Debug, Clone)]
pub struct Worker {
    pub role: Role,
    pub model: String,
    pub addr: String,
    /// Peers last assigned by the planner (cached; the published topology is the
    /// source of truth).
    pub neighbors: Vec<WorkerId>,
    /// Controller-side receipt time of the last liveness signal (no worker
    /// clocks → no skew).
    pub last_hb: Instant,
    /// Latest coarse load the worker reported. `kv_pressure_bucket` is the
    /// coalescing axis.
    pub load: WorkerStatus,
}

/// A registered gateway. Liveness only.
#[derive(Debug, Clone)]
pub struct Gateway {
    pub addr: String,
    pub last_hb: Instant,
}

/// The whole cluster. Sole owner: the actor task.
#[derive(Debug, Default)]
pub struct Cluster {
    /// Version of the worker-facing topology snapshot.
    pub worker_epoch: u64,
    /// Version of the gateway-facing routing snapshot.
    pub gateway_epoch: u64,
    pub workers: HashMap<WorkerId, Worker>,
    pub gateways: HashMap<GatewayId, Gateway>,
    next_worker_id: u64,
    next_gateway_id: u64,
}

impl Cluster {
    /// Empty cluster, both epochs at 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a worker (fields extracted from its wire `WorkerInfo` by the
    /// service; `capability` is not used by the trivial planner). Mints a
    /// [`WorkerId`].
    pub fn insert_worker(
        &mut self,
        role: Role,
        model: String,
        addr: String,
        now: Instant,
    ) -> WorkerId {
        let id = WorkerId(self.next_worker_id);
        self.next_worker_id += 1;
        self.workers.insert(
            id,
            Worker {
                role,
                model,
                addr,
                neighbors: Vec::new(),
                last_hb: now,
                load: WorkerStatus {
                    kv_pressure_bucket: 0,
                    inflight: 0,
                },
            },
        );
        id
    }

    /// Register a gateway, minting a [`GatewayId`].
    pub fn insert_gateway(&mut self, addr: String, now: Instant) -> GatewayId {
        let id = GatewayId(self.next_gateway_id);
        self.next_gateway_id += 1;
        self.gateways.insert(id, Gateway { addr, last_hb: now });
        id
    }

    /// Refresh a member's liveness. Returns `false` for an unknown id (the caller
    /// replies [`Ack::ReRegister`](pie_controller_rpc::Ack::ReRegister)).
    pub fn touch(&mut self, node: NodeId, now: Instant) -> bool {
        match node {
            NodeId::Worker(id) => match self.workers.get_mut(&id) {
                Some(w) => {
                    w.last_hb = now;
                    true
                }
                None => false,
            },
            NodeId::Gateway(id) => match self.gateways.get_mut(&id) {
                Some(g) => {
                    g.last_hb = now;
                    true
                }
                None => false,
            },
        }
    }

    /// Apply a worker load report (also refreshes liveness — a reporting worker
    /// is alive). Returns `Some(bucket_crossed)` for a known worker, `None` if
    /// the worker is unknown.
    pub fn report(&mut self, id: WorkerId, status: WorkerStatus, now: Instant) -> Option<bool> {
        let w = self.workers.get_mut(&id)?;
        let bucket_crossed = w.load.kv_pressure_bucket != status.kv_pressure_bucket;
        w.load = status;
        w.last_hb = now;
        Some(bucket_crossed)
    }

    /// Evict every member whose last liveness signal is older than `timeout`.
    /// Returns `(workers_removed, gateways_removed)`.
    pub fn evict_expired(&mut self, now: Instant, timeout: Duration) -> (usize, usize) {
        let (before_w, before_g) = (self.workers.len(), self.gateways.len());
        self.workers
            .retain(|_, w| now.duration_since(w.last_hb) <= timeout);
        self.gateways
            .retain(|_, g| now.duration_since(g.last_hb) <= timeout);
        (
            before_w - self.workers.len(),
            before_g - self.gateways.len(),
        )
    }
}
