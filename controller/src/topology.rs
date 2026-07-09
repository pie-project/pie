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
//! [`Neighbors`]: pie_controller_rpc::Neighbors

use std::collections::HashMap;

use pie_controller_rpc::{
    GatewayEndpoint, Health, NeighborPeer, Neighbors, RoutableWorker, RoutingTable,
};
use pie_ids::WorkerId;

use crate::state::Cluster;

/// Worker-facing plan: each worker's wire-ready peer list plus the global
/// gateway roster, versioned by `worker_epoch`. Internal; the service projects
/// one entry per `watch_worker`.
#[derive(Debug, Clone, Default)]
pub struct Topology {
    pub epoch: u64,
    pub peers: HashMap<WorkerId, Vec<NeighborPeer>>,
    /// The live gateway roster, identical for every worker (global full-mesh
    /// dial-in). Copied verbatim into each projected [`Neighbors`].
    pub gateways: Vec<GatewayEndpoint>,
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

/// The global gateway roster: every registered gateway as a dial-in target.
/// Identical for every worker (full-mesh dial-in), so it is computed once and
/// cloned into each projected [`Neighbors`]. Sorted for stable snapshots.
fn gateway_roster(cluster: &Cluster) -> Vec<GatewayEndpoint> {
    let mut gateways: Vec<GatewayEndpoint> = cluster
        .gateways
        .iter()
        .map(|(&id, g)| GatewayEndpoint {
            id,
            addr: g.addr.clone(),
        })
        .collect();
    gateways.sort_unstable_by_key(|g| g.id.0);
    gateways
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
        gateways: gateway_roster(cluster),
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
/// The global gateway roster is copied verbatim regardless of the worker id.
pub fn project(topology: &Topology, id: WorkerId) -> Neighbors {
    Neighbors {
        epoch: topology.epoch,
        peers: topology.peers.get(&id).cloned().unwrap_or_default(),
        gateways: topology.gateways.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use pie_controller_rpc::Role;

    #[test]
    fn roster_projects_into_every_neighbors() {
        let mut cluster = Cluster::new();
        let now = Instant::now();
        let w = cluster.insert_worker(Role::Decode, "m".into(), "10.0.0.1:7000".into(), now);
        let g = cluster.insert_gateway("10.0.0.9:8080".into(), now);

        let (topology, _routing) = reassign(&cluster);
        assert_eq!(topology.gateways.len(), 1);
        assert_eq!(topology.gateways[0].id, g);

        // The registered worker sees the global roster.
        let neighbors = project(&topology, w);
        assert_eq!(neighbors.gateways.len(), 1);
        assert_eq!(neighbors.gateways[0].addr, "10.0.0.9:8080");

        // An unknown (evicted mid-watch) worker still gets the global roster,
        // just an empty peer set.
        let unknown = project(&topology, WorkerId(999));
        assert_eq!(unknown.gateways.len(), 1);
        assert!(unknown.peers.is_empty());
    }
}
