use std::collections::HashMap;
use std::time::{Duration, Instant};

use controller_api::{Health, Role, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};

#[derive(Debug, Clone)]
pub struct Worker {
    pub role: Role,
    pub model: String,
    /// Its engine kind, as its config spells it.
    pub engine: String,
    pub addr: String,
    /// The token the actuator launched it with, if it did.
    pub launch: Option<String>,
    /// Its process id, as it told it.
    pub pid: Option<u32>,
    pub last_hb: Instant,
    pub load: WorkerStatus,
    /// Unreachable once its heartbeats have lagged for half the eviction
    /// timeout: routed around before it is evicted.
    pub health: Health,
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
    /// The actuator has slots to launch workers into.
    pub scales: bool,
    next_worker_id: u64,
    next_gateway_id: u64,
}

impl Cluster {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub fn insert_worker(
        &mut self,
        role: Role,
        model: String,
        addr: String,
        now: Instant,
    ) -> WorkerId {
        self.insert_worker_as(None, role, model, addr, now)
    }

    /// A worker registering: under `prior`, the id it had, if no one else
    /// holds it (a worker holding it at the same address is this one, whose
    /// registration lapsed); else a fresh one.
    pub fn insert_worker_as(
        &mut self,
        prior: Option<WorkerId>,
        role: Role,
        model: String,
        addr: String,
        now: Instant,
    ) -> WorkerId {
        let id = match prior {
            Some(id) if self.workers.get(&id).is_none_or(|w| w.addr == addr) => {
                self.next_worker_id = self.next_worker_id.max(id.0 + 1);
                id
            }
            _ => {
                let id = WorkerId(self.next_worker_id);
                self.next_worker_id += 1;
                id
            }
        };
        self.workers.insert(
            id,
            Worker {
                role,
                model,
                engine: String::new(),
                addr,
                launch: None,
                pid: None,
                last_hb: now,
                health: Health::Healthy,
                load: WorkerStatus::default(),
            },
        );
        id
    }

    #[cfg(test)]
    pub fn insert_gateway(&mut self, addr: String, now: Instant) -> GatewayId {
        self.insert_gateway_as(None, addr, now)
    }

    /// A gateway registering, under `prior` as a worker does.
    pub fn insert_gateway_as(
        &mut self,
        prior: Option<GatewayId>,
        addr: String,
        now: Instant,
    ) -> GatewayId {
        let id = match prior {
            Some(id) if self.gateways.get(&id).is_none_or(|g| g.addr == addr) => {
                self.next_gateway_id = self.next_gateway_id.max(id.0 + 1);
                id
            }
            _ => {
                let id = GatewayId(self.next_gateway_id);
                self.next_gateway_id += 1;
                id
            }
        };
        self.gateways.insert(id, Gateway { addr, last_hb: now });
        id
    }

    /// The gateway that authors the cluster-wide tables: the live one with
    /// the lowest id. One that is evicted hands it to the next.
    pub fn author(&self) -> Option<GatewayId> {
        self.gateways.keys().min().copied()
    }

    pub fn touch(&mut self, node: NodeId, now: Instant) -> bool {
        match node {
            NodeId::Worker(id) => match self.workers.get_mut(&id) {
                Some(w) => {
                    w.last_hb = now;
                    if w.health == Health::Unreachable {
                        w.health = if w.load.draining {
                            Health::Degraded
                        } else {
                            Health::Healthy
                        };
                    }
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

    /// Answers whether the routing table changes: the kv bucket crossed, the
    /// inflight count moved, or the worker began or stopped draining.
    pub fn report(&mut self, id: WorkerId, status: WorkerStatus, now: Instant) -> Option<bool> {
        let w = self.workers.get_mut(&id)?;
        let bucket_crossed = w.load.kv_pressure_bucket != status.kv_pressure_bucket
            || w.load.inflight != status.inflight;
        let drain_changed = w.load.draining != status.draining;
        w.load = status;
        w.last_hb = now;
        if status.draining {
            w.health = Health::Degraded;
        } else if w.health == Health::Degraded {
            w.health = Health::Healthy;
        }
        Some(bucket_crossed || drain_changed)
    }

    /// Marks each worker whose heartbeats lag past `lag` unreachable, and
    /// answers whether any worker's health changed.
    pub fn grade(&mut self, now: Instant, lag: Duration) -> bool {
        let mut changed = false;
        for w in self.workers.values_mut() {
            let health = if now.duration_since(w.last_hb) > lag {
                Health::Unreachable
            } else if w.load.draining {
                Health::Degraded
            } else {
                Health::Healthy
            };
            changed |= health != w.health;
            w.health = health;
        }
        changed
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

#[cfg(test)]
mod health {
    use super::*;

    #[test]
    fn a_node_registering_again_keeps_its_id_if_no_one_else_has_it() {
        let t0 = Instant::now();
        // A restarted controller: nothing registered, yet a worker says it
        // was 5 at its address.
        let mut cluster = Cluster::new();
        let w =
            cluster.insert_worker_as(Some(WorkerId(5)), Role::Decode, "m".into(), "a".into(), t0);
        assert_eq!(w, WorkerId(5));
        let fresh = cluster.insert_worker(Role::Decode, "m".into(), "b".into(), t0);
        assert_eq!(
            fresh,
            WorkerId(6),
            "fresh ids start past what was granted back"
        );
        // 5 again from 5's address is 5; from another, it is not.
        let again =
            cluster.insert_worker_as(Some(WorkerId(5)), Role::Decode, "m".into(), "a".into(), t0);
        assert_eq!(again, WorkerId(5));
        let other =
            cluster.insert_worker_as(Some(WorkerId(5)), Role::Decode, "m".into(), "c".into(), t0);
        assert_eq!(other, WorkerId(7));
        let g = cluster.insert_gateway_as(Some(GatewayId(3)), "g".into(), t0);
        assert_eq!(g, GatewayId(3));
        assert_eq!(cluster.insert_gateway("h".into(), t0), GatewayId(4));
    }

    #[test]
    fn a_worker_whose_heartbeats_lag_is_unreachable_until_it_beats_again() {
        let mut cluster = Cluster::new();
        let t0 = Instant::now();
        let id = cluster.insert_worker(Role::Decode, "m".into(), "a".into(), t0);
        let lag = Duration::from_secs(4);
        assert!(!cluster.grade(t0 + Duration::from_secs(3), lag));
        assert!(
            cluster.grade(t0 + Duration::from_secs(5), lag),
            "the change is told"
        );
        assert_eq!(cluster.workers[&id].health, Health::Unreachable);
        assert!(!cluster.grade(t0 + Duration::from_secs(6), lag), "once");
        cluster.touch(NodeId::Worker(id), t0 + Duration::from_secs(6));
        assert_eq!(cluster.workers[&id].health, Health::Healthy);
    }
}
