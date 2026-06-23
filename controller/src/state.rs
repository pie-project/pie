//! Cluster state — the single source of truth, owned **solely** by the actor
//! ([`crate::actor`]). No locks, no sharing: every mutation happens on the actor
//! task. These are **internal** types and never cross the wire — `service.rs`
//! projects them to the `protocol/schema` wire views before publishing.
//!
//! Two independent monotonic epochs version two independent full snapshots:
//! `worker_epoch` tags the worker-facing topology (neighbor assignments),
//! `gateway_epoch` tags the gateway-facing routing table. Membership changes
//! (join/leave/death) bump both; a coarse load-bucket change bumps only the
//! gateway epoch. An epoch is a version tag on a *whole* snapshot, never a delta.

use std::collections::HashMap;
use std::time::Instant;

use pie_schema::{Role, WorkerId};

/// Cluster-unique gateway handle, minted by the controller at registration.
///
/// Internal mirror of the wire `GatewayId` (defined on the `pie-schema` floor by
/// the contract crate); newtype so it can't be confused with a [`WorkerId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GatewayId(pub u64);

impl std::fmt::Display for GatewayId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gateway#{}", self.0)
    }
}

/// A cluster member reference — the unit a heartbeat names. Internal mirror of
/// the wire `NodeId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeId {
    Worker(WorkerId),
    Gateway(GatewayId),
}

/// A registered worker. `role` is **immutable** after registration — only
/// join/leave changes the topology, never a role flip.
#[derive(Debug, Clone)]
pub struct Worker {
    /// Inference stage this worker serves. Immutable.
    pub role: Role,
    /// Model the worker is bound to.
    pub model: String,
    /// Control/edge address peers reach it at.
    pub addr: String,
    /// The peers this worker was last assigned (cached from the planner).
    pub neighbors: Vec<WorkerId>,
    /// Controller-side receipt time of the last liveness signal (no worker
    /// clocks → no skew).
    pub last_hb: Instant,
    /// Coarse KV-pressure bucket the worker last reported. The coalescing axis:
    /// a change here is the *only* load signal that re-versions the gateway view.
    pub kv_pressure_bucket: u8,
    /// In-flight request count the worker last reported (finer; rides along in
    /// the roster but does not by itself re-version).
    pub inflight: u32,
}

/// A registered gateway. Liveness only — gateways carry no topology weight.
#[derive(Debug, Clone)]
pub struct Gateway {
    /// Address the gateway is reachable at.
    pub addr: String,
    /// Controller-side receipt time of its last liveness signal.
    pub last_hb: Instant,
}

/// What a worker declares about itself at registration.
#[derive(Debug, Clone)]
pub struct WorkerSpec {
    pub role: Role,
    pub model: String,
    pub addr: String,
}

/// What a gateway declares about itself at registration.
#[derive(Debug, Clone)]
pub struct GatewaySpec {
    pub addr: String,
}

/// The coarse load a worker reports. Already bucketed by the worker so the
/// controller can coalesce (bump the gateway epoch only on a bucket crossing).
#[derive(Debug, Clone, Copy)]
pub struct WorkerStatus {
    pub kv_pressure_bucket: u8,
    pub inflight: u32,
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

    /// Register a worker, minting and returning its [`WorkerId`]. Neighbors start
    /// empty; the planner fills them in the same write transaction.
    pub fn insert_worker(&mut self, spec: WorkerSpec, now: Instant) -> WorkerId {
        let id = WorkerId(self.next_worker_id);
        self.next_worker_id += 1;
        self.workers.insert(
            id,
            Worker {
                role: spec.role,
                model: spec.model,
                addr: spec.addr,
                neighbors: Vec::new(),
                last_hb: now,
                kv_pressure_bucket: 0,
                inflight: 0,
            },
        );
        id
    }

    /// Register a gateway, minting and returning its [`GatewayId`].
    pub fn insert_gateway(&mut self, spec: GatewaySpec, now: Instant) -> GatewayId {
        let id = GatewayId(self.next_gateway_id);
        self.next_gateway_id += 1;
        self.gateways.insert(
            id,
            Gateway {
                addr: spec.addr,
                last_hb: now,
            },
        );
        id
    }

    /// Refresh a member's liveness. Returns `false` for an unknown id (the caller
    /// then replies `ReRegister`).
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

    /// Apply a worker load report. Refreshes liveness too (a reporting worker is
    /// alive). Returns `Some(bucket_changed)` for a known worker, `None` if the
    /// worker is unknown.
    pub fn report(&mut self, id: WorkerId, status: WorkerStatus, now: Instant) -> Option<bool> {
        let w = self.workers.get_mut(&id)?;
        let bucket_changed = w.kv_pressure_bucket != status.kv_pressure_bucket;
        w.kv_pressure_bucket = status.kv_pressure_bucket;
        w.inflight = status.inflight;
        w.last_hb = now;
        Some(bucket_changed)
    }

    /// Evict every member whose last liveness signal is older than `timeout`.
    /// Returns `(workers_removed, gateways_removed)` counts so the caller can
    /// decide whether to re-plan (a worker leaving changes topology).
    pub fn evict_expired(&mut self, now: Instant, timeout: std::time::Duration) -> (usize, usize) {
        let before_w = self.workers.len();
        let before_g = self.gateways.len();
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
