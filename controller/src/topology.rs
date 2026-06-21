//! The planner — a **pure** function from cluster membership to the two
//! published snapshots. This is the controller's "brain"; the neighbor policy is
//! deliberately trivial for now (every worker peers with all others) behind a
//! single seam ([`neighbors_for`]) so a real PD-pairing / TP-grouping policy
//! drops in later without touching the actor or the wire.
//!
//! The gateway view is **global** (same for every gateway), so the wire
//! [`RoutingTable`] is produced directly. The worker view is **per-worker**, so
//! the internal [`Topology`] holds every worker's wire-ready peer list keyed by
//! id; the service projects one [`Neighbors`] from it per `watch_worker`.
//!
//! [`Neighbors`]: pie_schema::control::Neighbors

use std::collections::HashMap;

use pie_schema::control::{
    Health, NeighborPeer, Neighbors, RoutableWorker, RoutingTable, WorkerId,
};

use crate::state::Cluster;

/// Worker-facing plan: each worker's wire-ready peer list, versioned by
/// `worker_epoch`. Internal; the service projects one entry per `watch_worker`.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    pub epoch: u64,
    pub peers: HashMap<WorkerId, Vec<NeighborPeer>>,
}

/// Trivial neighbor policy: a worker peers with every *other* registered worker.
/// The single seam a smarter policy (PD-pairing, TP-grouping by role+model,
/// least-loaded) replaces later; the actor and wire stay unchanged.
fn neighbors_for(id: WorkerId, cluster: &Cluster) -> Vec<NeighborPeer> {
    let mut peers: Vec<NeighborPeer> = cluster
        .workers
        .iter()
        .filter(|(w, _)| **w != id)
        .map(|(&w, info)| NeighborPeer {
            id: w,
            addr: info.addr.clone(),
            role: info.role,
        })
        .collect();
    peers.sort_unstable_by_key(|p| p.id.0); // deterministic → stable snapshots
    peers
}

/// Recompute both snapshots from current membership. Pure: reads `cluster`
/// (including its already-bumped epochs) and stamps the new versions. The actor
/// bumps the epoch(s) **before** calling this.
pub fn reassign(cluster: &Cluster) -> (Topology, RoutingTable) {
    let peers = cluster
        .workers
        .keys()
        .map(|&id| (id, neighbors_for(id, cluster)))
        .collect();
    let topology = Topology {
        epoch: cluster.worker_epoch,
        peers,
    };

    let mut workers: Vec<RoutableWorker> = cluster
        .workers
        .iter()
        .map(|(&id, w)| RoutableWorker {
            id,
            addr: w.addr.clone(),
            role: w.role,
            model: w.model.clone(),
            health: Health::Healthy, // present (non-evicted) ⇒ healthy, minimal start
            coarse_load: w.load,
        })
        .collect();
    workers.sort_unstable_by_key(|r| r.id.0);
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

/// An empty routing table at epoch 0 — the initial watch-channel value (the wire
/// [`RoutingTable`] has no `Default`).
pub fn empty_routing() -> RoutingTable {
    RoutingTable {
        epoch: 0,
        workers: Vec::new(),
    }
}

/// Project the full [`Topology`] down to one worker's wire [`Neighbors`] view.
/// An absent id (e.g. evicted mid-watch) projects to an empty peer set at the
/// current epoch — the worker learns it is gone via its next `heartbeat` `Ack`.
pub fn project(topology: &Topology, id: WorkerId) -> Neighbors {
    Neighbors {
        epoch: topology.epoch,
        peers: topology.peers.get(&id).cloned().unwrap_or_default(),
    }
}
