use controller_api::{GatewayEndpoint, Neighbors, RoutableWorker, RoutingTable};
use ids::WorkerId;

use crate::state::Cluster;

#[derive(Debug, Clone, Default)]
pub struct Topology {
    pub epoch: u64,
    pub gateways: Vec<GatewayEndpoint>,
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
            engine: w.engine.clone(),
            health: w.health,
            coarse_load: w.load,
        })
        .collect();
    workers.sort_unstable_by_key(|r| r.id.0);
    let routing = RoutingTable {
        epoch: cluster.gateway_epoch,
        workers,
        scales: cluster.scales,
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
        scales: false,
    }
}

pub fn project(topology: &Topology, _id: WorkerId) -> Neighbors {
    Neighbors {
        epoch: topology.epoch,
        gateways: topology.gateways.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use controller_api::Role;

    #[test]
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
        assert_eq!(project(&topology, WorkerId(999)).gateways.len(), 1);
    }
}
