use std::collections::HashMap;
use std::time::{Duration, Instant};

use controller_api::{Role, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};

#[derive(Debug, Clone)]
pub struct Worker {
    pub role: Role,
    pub model: String,
    pub addr: String,
    pub neighbors: Vec<WorkerId>,
    pub last_hb: Instant,
    pub load: WorkerStatus,
}

#[derive(Debug, Clone)]
pub struct Gateway {
    pub addr: String,
    pub last_hb: Instant,
}

#[derive(Debug, Default)]
pub struct Cluster {
    pub worker_epoch: u64,
    pub gateway_epoch: u64,
    pub workers: HashMap<WorkerId, Worker>,
    pub gateways: HashMap<GatewayId, Gateway>,
    next_worker_id: u64,
    next_gateway_id: u64,
}

impl Cluster {
    pub fn new() -> Self {
        Self::default()
    }

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
                load: WorkerStatus::default(),
            },
        );
        id
    }

    pub fn insert_gateway(&mut self, addr: String, now: Instant) -> GatewayId {
        let id = GatewayId(self.next_gateway_id);
        self.next_gateway_id += 1;
        self.gateways.insert(id, Gateway { addr, last_hb: now });
        id
    }

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

    pub fn report(&mut self, id: WorkerId, status: WorkerStatus, now: Instant) -> Option<bool> {
        let w = self.workers.get_mut(&id)?;
        let bucket_crossed = w.load.kv_pressure_bucket != status.kv_pressure_bucket;
        w.load = status;
        w.last_hb = now;
        Some(bucket_crossed)
    }

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
