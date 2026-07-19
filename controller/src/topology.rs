//! The planner — a **pure** function from cluster membership to the two
//! published snapshots. This is the controller's "brain"; the worker view pairs
//! decode clients with bounded, model-compatible prefill and encode executors.
//!
//! The gateway view is **global** (same for every gateway), so the wire
//! [`RoutingTable`] is produced directly. The worker view is **per-worker**, so
//! the internal [`Topology`] holds every worker's wire-ready peer list keyed by
//! id; the service projects one [`Neighbors`] from it per `watch_worker`.
//!
//! [`Neighbors`]: pie_controller_rpc::Neighbors

use std::collections::HashMap;

use pie_controller_rpc::{
    GatewayEndpoint, Health, NeighborPeer, Neighbors, Role, RoutableWorker, RoutingTable,
};
use pie_ids::WorkerId;

use crate::state::Cluster;

const PARTNERS_PER_ROLE: usize = 2;

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

/// Build deterministic D4 pairings. Decode workers are assigned at most
/// [`PARTNERS_PER_ROLE`] same-model executors of each role. Executors with the
/// lowest current decode fan-in win, with [`WorkerId`] breaking ties.
fn pairing_plan(cluster: &Cluster) -> HashMap<WorkerId, Vec<NeighborPeer>> {
    let mut peers: HashMap<WorkerId, Vec<NeighborPeer>> =
        cluster.workers.keys().map(|&id| (id, Vec::new())).collect();
    let mut fan_in = HashMap::<WorkerId, usize>::new();
    let mut decodes: Vec<WorkerId> = cluster
        .workers
        .iter()
        .filter_map(|(&id, worker)| (worker.role == Role::Decode).then_some(id))
        .collect();
    decodes.sort_unstable_by_key(|id| id.0);

    for decode_id in decodes {
        let decode = &cluster.workers[&decode_id];

        for executor_role in [Role::Prefill, Role::Encode] {
            let mut candidates: Vec<WorkerId> = cluster
                .workers
                .iter()
                .filter_map(|(&id, worker)| {
                    (worker.role == executor_role && worker.model == decode.model).then_some(id)
                })
                .collect();
            candidates
                .sort_unstable_by_key(|id| (fan_in.get(id).copied().unwrap_or_default(), id.0));

            for executor_id in candidates.into_iter().take(PARTNERS_PER_ROLE) {
                let executor = &cluster.workers[&executor_id];
                peers
                    .get_mut(&decode_id)
                    .expect("decode is present in the pairing plan")
                    .push(NeighborPeer {
                        id: executor_id,
                        addr: executor.addr.clone(),
                        role: executor.role,
                    });
                peers
                    .get_mut(&executor_id)
                    .expect("executor is present in the pairing plan")
                    .push(NeighborPeer {
                        id: decode_id,
                        addr: decode.addr.clone(),
                        role: decode.role,
                    });
                *fan_in.entry(executor_id).or_default() += 1;
            }
        }
    }

    for worker_peers in peers.values_mut() {
        worker_peers.sort_unstable_by_key(|peer| peer.id.0);
    }
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
    let topology = Topology {
        epoch: cluster.worker_epoch,
        peers: pairing_plan(cluster),
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

    fn add_worker(cluster: &mut Cluster, role: Role, model: &str) -> WorkerId {
        let ordinal = cluster.workers.len();
        cluster.insert_worker(
            role,
            model.into(),
            format!("10.0.0.{ordinal}:7000"),
            Instant::now(),
        )
    }

    fn peer_ids(topology: &Topology, id: WorkerId) -> Vec<WorkerId> {
        topology.peers[&id].iter().map(|peer| peer.id).collect()
    }

    fn peer_ids_for_role(topology: &Topology, id: WorkerId, role: Role) -> Vec<WorkerId> {
        topology.peers[&id]
            .iter()
            .filter_map(|peer| (peer.role == role).then_some(peer.id))
            .collect()
    }

    fn assert_reverse_symmetric(topology: &Topology) {
        for (&id, peers) in &topology.peers {
            for peer in peers {
                assert!(
                    topology.peers[&peer.id]
                        .iter()
                        .any(|reverse| reverse.id == id),
                    "missing reverse edge for {id:?} -> {:?}",
                    peer.id
                );
            }
        }
    }

    #[test]
    fn pairings_filter_same_roles_and_cross_model_workers() {
        let mut cluster = Cluster::new();
        let decode_a0 = add_worker(&mut cluster, Role::Decode, "model-a");
        let decode_a1 = add_worker(&mut cluster, Role::Decode, "model-a");
        let decode_b = add_worker(&mut cluster, Role::Decode, "model-b");
        let prefill_a = add_worker(&mut cluster, Role::Prefill, "model-a");
        let prefill_b = add_worker(&mut cluster, Role::Prefill, "model-b");
        let encode_a = add_worker(&mut cluster, Role::Encode, "model-a");
        let encode_b = add_worker(&mut cluster, Role::Encode, "model-b");

        let (topology, _) = reassign(&cluster);

        assert_eq!(peer_ids(&topology, decode_a0), vec![prefill_a, encode_a]);
        assert_eq!(peer_ids(&topology, decode_a1), vec![prefill_a, encode_a]);
        assert_eq!(peer_ids(&topology, decode_b), vec![prefill_b, encode_b]);
        assert_eq!(peer_ids(&topology, prefill_a), vec![decode_a0, decode_a1]);
        assert_eq!(peer_ids(&topology, encode_a), vec![decode_a0, decode_a1]);
        assert_eq!(peer_ids(&topology, prefill_b), vec![decode_b]);
        assert_eq!(peer_ids(&topology, encode_b), vec![decode_b]);

        for (&id, peers) in &topology.peers {
            let worker = &cluster.workers[&id];
            for peer in peers {
                let other = &cluster.workers[&peer.id];
                assert_eq!(worker.model, other.model);
                assert!(matches!(
                    (worker.role, other.role),
                    (Role::Decode, Role::Prefill | Role::Encode)
                        | (Role::Prefill | Role::Encode, Role::Decode)
                ));
            }
        }
    }

    #[test]
    fn decode_pairings_are_capped_at_two_per_executor_role() {
        let mut cluster = Cluster::new();
        let decode = add_worker(&mut cluster, Role::Decode, "model");
        let prefill0 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill1 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill2 = add_worker(&mut cluster, Role::Prefill, "model");
        let encode0 = add_worker(&mut cluster, Role::Encode, "model");
        let encode1 = add_worker(&mut cluster, Role::Encode, "model");
        let encode2 = add_worker(&mut cluster, Role::Encode, "model");

        let (topology, _) = reassign(&cluster);

        assert_eq!(
            peer_ids_for_role(&topology, decode, Role::Prefill),
            vec![prefill0, prefill1]
        );
        assert_eq!(
            peer_ids_for_role(&topology, decode, Role::Encode),
            vec![encode0, encode1]
        );
        assert_eq!(peer_ids(&topology, prefill0), vec![decode]);
        assert_eq!(peer_ids(&topology, prefill1), vec![decode]);
        assert!(peer_ids(&topology, prefill2).is_empty());
        assert_eq!(peer_ids(&topology, encode0), vec![decode]);
        assert_eq!(peer_ids(&topology, encode1), vec![decode]);
        assert!(peer_ids(&topology, encode2).is_empty());
    }

    #[test]
    fn executor_views_are_exact_reverse_pairings() {
        let mut cluster = Cluster::new();
        let decode0 = add_worker(&mut cluster, Role::Decode, "model");
        let decode1 = add_worker(&mut cluster, Role::Decode, "model");
        let decode2 = add_worker(&mut cluster, Role::Decode, "model");
        let prefill0 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill1 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill2 = add_worker(&mut cluster, Role::Prefill, "model");
        let encode0 = add_worker(&mut cluster, Role::Encode, "model");
        let encode1 = add_worker(&mut cluster, Role::Encode, "model");
        let encode2 = add_worker(&mut cluster, Role::Encode, "model");

        let (topology, _) = reassign(&cluster);

        assert_reverse_symmetric(&topology);
        assert_eq!(peer_ids(&topology, prefill0), vec![decode0, decode1]);
        assert_eq!(peer_ids(&topology, prefill1), vec![decode0, decode2]);
        assert_eq!(peer_ids(&topology, prefill2), vec![decode1, decode2]);
        assert_eq!(peer_ids(&topology, encode0), vec![decode0, decode1]);
        assert_eq!(peer_ids(&topology, encode1), vec![decode0, decode2]);
        assert_eq!(peer_ids(&topology, encode2), vec![decode1, decode2]);
    }

    #[test]
    fn fan_in_is_balanced_deterministically() {
        let mut cluster = Cluster::new();
        let prefill0 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill1 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill2 = add_worker(&mut cluster, Role::Prefill, "model");
        let decode0 = add_worker(&mut cluster, Role::Decode, "model");
        let decode1 = add_worker(&mut cluster, Role::Decode, "model");
        let decode2 = add_worker(&mut cluster, Role::Decode, "model");
        let decode3 = add_worker(&mut cluster, Role::Decode, "model");
        let decode4 = add_worker(&mut cluster, Role::Decode, "model");

        let (first, _) = reassign(&cluster);
        let (second, _) = reassign(&cluster);

        assert_eq!(first.peers, second.peers);
        assert_eq!(peer_ids(&first, decode0), vec![prefill0, prefill1]);
        assert_eq!(peer_ids(&first, decode1), vec![prefill0, prefill2]);
        assert_eq!(peer_ids(&first, decode2), vec![prefill1, prefill2]);
        assert_eq!(peer_ids(&first, decode3), vec![prefill0, prefill1]);
        assert_eq!(peer_ids(&first, decode4), vec![prefill0, prefill2]);
        assert_eq!(peer_ids(&first, prefill0).len(), 4);
        assert_eq!(peer_ids(&first, prefill1).len(), 3);
        assert_eq!(peer_ids(&first, prefill2).len(), 3);
    }

    #[test]
    fn reassign_updates_both_sides_after_membership_churn() {
        let mut cluster = Cluster::new();
        let prefill0 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill1 = add_worker(&mut cluster, Role::Prefill, "model");
        let prefill2 = add_worker(&mut cluster, Role::Prefill, "model");
        let decode0 = add_worker(&mut cluster, Role::Decode, "model");
        let decode1 = add_worker(&mut cluster, Role::Decode, "model");
        let decode2 = add_worker(&mut cluster, Role::Decode, "model");

        let (initial, _) = reassign(&cluster);
        assert_eq!(peer_ids(&initial, decode0), vec![prefill0, prefill1]);
        assert_eq!(peer_ids(&initial, decode1), vec![prefill0, prefill2]);
        assert_eq!(peer_ids(&initial, decode2), vec![prefill1, prefill2]);

        cluster.workers.remove(&prefill0);
        let (after_loss, _) = reassign(&cluster);
        assert!(!after_loss.peers.contains_key(&prefill0));
        assert_eq!(peer_ids(&after_loss, decode0), vec![prefill1, prefill2]);
        assert_eq!(peer_ids(&after_loss, decode1), vec![prefill1, prefill2]);
        assert_eq!(peer_ids(&after_loss, decode2), vec![prefill1, prefill2]);
        assert_eq!(
            peer_ids(&after_loss, prefill1),
            vec![decode0, decode1, decode2]
        );
        assert_eq!(
            peer_ids(&after_loss, prefill2),
            vec![decode0, decode1, decode2]
        );

        let prefill3 = add_worker(&mut cluster, Role::Prefill, "model");
        let (after_join, _) = reassign(&cluster);
        assert_eq!(peer_ids(&after_join, decode0), vec![prefill1, prefill2]);
        assert_eq!(peer_ids(&after_join, decode1), vec![prefill1, prefill3]);
        assert_eq!(peer_ids(&after_join, decode2), vec![prefill2, prefill3]);
        assert_eq!(peer_ids(&after_join, prefill1), vec![decode0, decode1]);
        assert_eq!(peer_ids(&after_join, prefill2), vec![decode0, decode2]);
        assert_eq!(peer_ids(&after_join, prefill3), vec![decode1, decode2]);
        assert_reverse_symmetric(&after_join);
    }

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
