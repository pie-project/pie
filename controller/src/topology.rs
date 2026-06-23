//! The planner — a **pure** function from cluster membership to the two
//! published snapshots. This is the controller's "brain"; the neighbor policy is
//! deliberately trivial for now (every worker peers with all others) behind a
//! single seam ([`neighbors_for`]) so a real PD-pairing / least-loaded policy
//! drops in later without touching the actor or the wire.
//!
//! Both snapshots are **internal** and carry their epoch inside, mirroring the
//! wire views `Neighbors{ epoch, peers }` / `RoutingTable{ epoch, workers }`.
//! `service.rs` projects [`Topology`] down to one worker's [`wire Neighbors`] and
//! converts [`RoutingTable`] to its wire form before sending.

use std::collections::HashMap;

use pie_schema::{HealthStatus, Role, WorkerId};

use crate::state::Cluster;

/// The worker-facing plan: every worker's assigned peers, versioned by
/// `worker_epoch`. Projected per-worker before the wire.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    pub epoch: u64,
    pub neighbors: HashMap<WorkerId, Vec<WorkerId>>,
}

/// One worker as the gateway sees it.
#[derive(Debug, Clone)]
pub struct RoutingEntry {
    pub id: WorkerId,
    pub addr: String,
    pub role: Role,
    pub model: String,
    /// All present (non-evicted) workers are healthy in the minimal start.
    pub health: HealthStatus,
    /// Coarse KV-pressure bucket — the coalescing axis.
    pub coarse_load: u8,
}

/// The gateway-facing plan: the worker roster + status, versioned by
/// `gateway_epoch`.
#[derive(Debug, Clone, Default)]
pub struct RoutingTable {
    pub epoch: u64,
    pub workers: Vec<RoutingEntry>,
}

/// Trivial neighbor policy: a worker peers with every *other* registered worker.
/// The single seam a smarter policy (round-robin PD-pairing, least-loaded)
/// replaces later; the actor and wire stay unchanged.
fn neighbors_for(id: WorkerId, cluster: &Cluster) -> Vec<WorkerId> {
    let mut peers: Vec<WorkerId> = cluster
        .workers
        .keys()
        .copied()
        .filter(|&w| w != id)
        .collect();
    peers.sort_unstable(); // deterministic output → stable snapshots
    peers
}

/// Recompute both snapshots from current membership. Pure: reads `cluster`
/// (including its already-bumped epochs) and returns the versioned plans. The
/// actor bumps the epoch(s) *before* calling this so the snapshots carry the new
/// version.
pub fn reassign(cluster: &Cluster) -> (Topology, RoutingTable) {
    let neighbors = cluster
        .workers
        .keys()
        .map(|&id| (id, neighbors_for(id, cluster)))
        .collect();
    let topology = Topology {
        epoch: cluster.worker_epoch,
        neighbors,
    };

    let mut workers: Vec<RoutingEntry> = cluster
        .workers
        .iter()
        .map(|(&id, w)| RoutingEntry {
            id,
            addr: w.addr.clone(),
            role: w.role,
            model: w.model.clone(),
            health: HealthStatus::Healthy,
            coarse_load: w.kv_pressure_bucket,
        })
        .collect();
    workers.sort_unstable_by_key(|e| e.id.0);
    let routing = RoutingTable {
        epoch: cluster.gateway_epoch,
        workers,
    };

    (topology, routing)
}

/// Rebuild only the gateway-facing routing table (used on a coarse-load change,
/// which never alters worker topology). Reads the already-bumped `gateway_epoch`.
pub fn routing_only(cluster: &Cluster) -> RoutingTable {
    reassign(cluster).1
}
