//! The control-plane [`Coordinator`]: the worker registry that backs the
//! standalone controller process.
//!
//! The coordinator owns the soft cluster state — registrations, pushed load,
//! roles, and heartbeat liveness — behind a mutex, and answers the control-RPC
//! surface ([`crate::ControlApi`]) that [`crate::run_as_process`] serves over
//! tarpc. The gateway and distributed workers dial it; single-node skips the
//! control plane entirely. It never touches tokens or KV — those ride the data
//! plane.
//!
//! # Minimal start (YAGNI)
//!
//! Per the ratified spec the coordinator is a **worker registry + round-robin
//! `route`** with **pushed** load (`report`); `pair` is the trivial same-node
//! decision. Real PD pairing, least-loaded routing, and placement scaling are
//! deferred.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::error::{ControllerError, Result};
use crate::health::HealthChecker;
use crate::role::RoleTable;
use pie_schema::{
    HealthStatus, LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo,
};

#[cfg(test)]
use pie_schema::Role;

/// Tunables for the coordinator.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Silent report-ticks before a worker is graded degraded. See
    /// [`HealthChecker`].
    pub degrade_after: u64,
    /// Silent report-ticks before a worker is graded unreachable (and dropped
    /// from routing).
    pub unreachable_after: u64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            degrade_after: 3,
            unreachable_after: 6,
        }
    }
}

/// The control-plane coordinator. Holds the soft cluster state behind a mutex so
/// it can be shared (`&self`) across the standalone process's connection-handler
/// threads and its background liveness ticker.
#[derive(Debug)]
pub(crate) struct Coordinator {
    inner: Mutex<Inner>,
}

#[derive(Debug)]
struct Inner {
    /// Static registration record per worker.
    workers: HashMap<WorkerId, WorkerInfo>,
    /// Latest pushed load per worker (soft state).
    load: HashMap<WorkerId, LoadState>,
    /// Assigned role per worker.
    roles: RoleTable,
    /// Heartbeat-based liveness.
    health: HealthChecker,
    /// Registration order — the rotation `route`/`pair` walk.
    order: Vec<WorkerId>,
    /// Round-robin cursor into `order`.
    rr_cursor: usize,
    /// Monotonic id source.
    next_worker_id: u64,
}

impl Coordinator {
    /// New coordinator.
    pub(crate) fn new(config: ControllerConfig) -> Self {
        Self {
            inner: Mutex::new(Inner {
                workers: HashMap::new(),
                load: HashMap::new(),
                roles: RoleTable::new(),
                health: HealthChecker::new(config.degrade_after, config.unreachable_after),
                order: Vec::new(),
                rr_cursor: 0,
                next_worker_id: 0,
            }),
        }
    }

    /// Advance the liveness clock by one tick. Driven by the standalone
    /// process's background timer; a worker that stops reporting ages out of
    /// routing.
    pub(crate) fn tick(&self) {
        self.lock().health.tick();
    }

    /// The role assigned to a worker, if any.
    #[cfg(test)]
    pub(crate) fn role_of(&self, worker: WorkerId) -> Option<Role> {
        self.lock().roles.role_of(worker)
    }

    /// The latest load reported by a worker, if any.
    #[cfg(test)]
    pub(crate) fn load_of(&self, worker: WorkerId) -> Option<LoadState> {
        self.lock().load.get(&worker).copied()
    }

    /// Number of registered workers.
    #[cfg(test)]
    pub(crate) fn worker_count(&self) -> usize {
        self.lock().workers.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().expect("coordinator mutex poisoned")
    }
}

impl Inner {
    fn register(&mut self, info: WorkerInfo) -> WorkerId {
        let id = WorkerId(self.next_worker_id);
        self.next_worker_id += 1;
        if let Some(role) = info.preferred_role {
            self.roles.assign(id, role);
        }
        self.health.track(id);
        self.load.insert(
            id,
            LoadState {
                active_requests: 0,
                kv_pages_free: 0,
            },
        );
        self.workers.insert(id, info);
        self.order.push(id);
        id
    }

    fn report(&mut self, worker: WorkerId, load: LoadState) -> Result<()> {
        if !self.workers.contains_key(&worker) {
            return Err(ControllerError::UnknownWorker(worker));
        }
        self.load.insert(worker, load);
        self.health.heartbeat(worker);
        Ok(())
    }

    /// Round-robin over registered workers, skipping any that have timed out.
    fn select_worker(&mut self) -> Result<WorkerId> {
        let n = self.order.len();
        if n == 0 {
            return Err(ControllerError::NoEligibleWorker);
        }
        for _ in 0..n {
            let id = self.order[self.rr_cursor % n];
            self.rr_cursor = (self.rr_cursor + 1) % n;
            if matches!(
                self.health.status(id),
                Some(HealthStatus::Healthy | HealthStatus::Degraded)
            ) {
                return Ok(id);
            }
        }
        Err(ControllerError::NoEligibleWorker)
    }
}

impl Coordinator {
    /// Register a worker on boot; mint and return its [`WorkerId`].
    pub(crate) fn register(&self, worker: WorkerInfo) -> Result<WorkerId> {
        Ok(self.lock().register(worker))
    }

    /// Accept a pushed load report (doubles as the liveness heartbeat).
    pub(crate) fn report(&self, worker: WorkerId, load: LoadState) -> Result<()> {
        self.lock().report(worker, load)
    }

    /// Decide which worker should serve a request.
    pub(crate) fn route(&self, req: &RequestMeta) -> Result<Placement> {
        let worker = self.lock().select_worker()?;
        tracing::trace!(request = %req.id, %worker, "routed request");
        Ok(Placement { worker })
    }

    /// Decide the `(prefill, decode)` worker pair for a request. Trivial
    /// same-node in the minimal start (both elements equal).
    pub(crate) fn pair(&self, req: RequestId) -> Result<(WorkerId, WorkerId)> {
        // Minimal start: monolithic worker serves both stages → same-node pair.
        let worker = self.lock().select_worker()?;
        tracing::trace!(request = %req, %worker, "paired request (same-node)");
        Ok((worker, worker))
    }

    /// Resolve a [`WorkerId`] to its registered [`WorkerInfo`] (control address
    /// + role) so a router can dial the placed worker.
    pub(crate) fn resolve(&self, worker: WorkerId) -> Result<WorkerInfo> {
        self.lock()
            .workers
            .get(&worker)
            .cloned()
            .ok_or(ControllerError::UnknownWorker(worker))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(role: Option<Role>) -> WorkerInfo {
        WorkerInfo {
            control_addr: "127.0.0.1:0".to_string(),
            preferred_role: role,
        }
    }

    fn meta(id: u64) -> RequestMeta {
        RequestMeta {
            id: RequestId(id),
            prompt_tokens: 8,
        }
    }

    #[test]
    fn register_mints_sequential_ids() {
        let c = Coordinator::new(ControllerConfig::default());
        assert_eq!(c.register(info(None)).unwrap(), WorkerId(0));
        assert_eq!(c.register(info(None)).unwrap(), WorkerId(1));
        assert_eq!(c.worker_count(), 2);
    }

    #[test]
    fn preferred_role_recorded() {
        let c = Coordinator::new(ControllerConfig::default());
        let w = c.register(info(Some(Role::Decode))).unwrap();
        assert_eq!(c.role_of(w), Some(Role::Decode));
    }

    #[test]
    fn route_round_robins() {
        let c = Coordinator::new(ControllerConfig::default());
        let a = c.register(info(None)).unwrap();
        let b = c.register(info(None)).unwrap();
        assert_eq!(c.route(&meta(1)).unwrap().worker, a);
        assert_eq!(c.route(&meta(2)).unwrap().worker, b);
        assert_eq!(c.route(&meta(3)).unwrap().worker, a);
    }

    #[test]
    fn route_without_workers_errors() {
        let c = Coordinator::new(ControllerConfig::default());
        assert!(matches!(
            c.route(&meta(1)).unwrap_err(),
            ControllerError::NoEligibleWorker
        ));
    }

    #[test]
    fn pair_is_same_node() {
        let c = Coordinator::new(ControllerConfig::default());
        let w = c.register(info(None)).unwrap();
        assert_eq!(c.pair(RequestId(1)).unwrap(), (w, w));
    }

    #[test]
    fn report_updates_load_and_unknown_errors() {
        let c = Coordinator::new(ControllerConfig::default());
        let w = c.register(info(None)).unwrap();
        let load = LoadState {
            active_requests: 3,
            kv_pages_free: 42,
        };
        c.report(w, load).unwrap();
        assert_eq!(c.load_of(w), Some(load));
        assert!(matches!(
            c.report(WorkerId(99), load).unwrap_err(),
            ControllerError::UnknownWorker(WorkerId(99))
        ));
    }

    #[test]
    fn route_skips_timed_out_workers() {
        let c = Coordinator::new(ControllerConfig {
            degrade_after: 1,
            unreachable_after: 2,
        });
        let _a = c.register(info(None)).unwrap();
        let b = c.register(info(None)).unwrap();
        c.tick();
        c.tick(); // both silent 2 ticks → unreachable
        assert!(matches!(
            c.route(&meta(1)).unwrap_err(),
            ControllerError::NoEligibleWorker
        ));
        // keep b alive; a stays dead → route always lands on b
        c.report(
            b,
            LoadState {
                active_requests: 0,
                kv_pages_free: 0,
            },
        )
        .unwrap();
        assert_eq!(c.route(&meta(2)).unwrap().worker, b);
        assert_eq!(c.route(&meta(3)).unwrap().worker, b);
    }
}
