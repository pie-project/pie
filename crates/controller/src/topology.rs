use std::collections::HashMap;

use controller_api::{
    GatewayEndpoint, Health, NeighborPeer, Neighbors, Role, RoutableWorker, RoutingTable,
};
use ids::WorkerId;

use crate::state::Cluster;

const PARTNERS_PER_ROLE: usize = 2;

#[derive(Debug, Clone, Default)]
pub struct Topology {
    pub epoch: u64,
    pub peers: HashMap<WorkerId, Vec<NeighborPeer>>,
    pub gateways: Vec<GatewayEndpoint>,
}

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
            health: Health::Healthy,
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

pub fn routing_only(cluster: &Cluster) -> RoutingTable {
    reassign(cluster).1
}

pub fn empty_routing() -> RoutingTable {
    RoutingTable {
        epoch: 0,
        workers: Vec::new(),
    }
}

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

    use controller_api::Role;

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

    #[test]
    fn topology_every_case() {
        pairings_filter_same_roles_and_cross_model_workers();
        fan_in_is_balanced_deterministically();
        roster_projects_into_every_neighbors();
    }

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

    fn roster_projects_into_every_neighbors() {
        let mut cluster = Cluster::new();
        let now = Instant::now();
        let w = cluster.insert_worker(Role::Decode, "m".into(), "10.0.0.1:7000".into(), now);
        let g = cluster.insert_gateway("10.0.0.9:8080".into(), now);

        let (topology, _routing) = reassign(&cluster);
        assert_eq!(topology.gateways.len(), 1);
        assert_eq!(topology.gateways[0].id, g);

        let neighbors = project(&topology, w);
        assert_eq!(neighbors.gateways.len(), 1);
        assert_eq!(neighbors.gateways[0].addr, "10.0.0.9:8080");

        let unknown = project(&topology, WorkerId(999));
        assert_eq!(unknown.gateways.len(), 1);
        assert!(unknown.peers.is_empty());
    }
}
