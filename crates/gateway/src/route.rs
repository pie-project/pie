use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use controller_api::{Health, Role, RoutableWorker, RoutingTable};
use ids::WorkerId;
use tokio::sync::watch;
use worker_api::{Accepted, Request};

use crate::admission::{AdmissionConfig, AdmissionDecision, admit};

pub type AffinityKey = u64;

const MAX_DISPATCH_ATTEMPTS: usize = 8;

pub trait WorkerDispatch {
    type Err: std::fmt::Display;

    fn dispatch(
        &self,
        id: WorkerId,
        req: Request,
    ) -> impl std::future::Future<Output = Result<Accepted, Self::Err>> + Send;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dispatched {
    pub worker_id: WorkerId,
    pub accepted: Accepted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    NoCandidate,
    Exhausted { attempts: usize },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteError::NoCandidate => {
                f.write_str("no healthy, connected worker available to route")
            }
            RouteError::Exhausted { attempts } => {
                write!(
                    f,
                    "no worker accepted the turn after {attempts} dispatch attempt(s)"
                )
            }
        }
    }
}

impl std::error::Error for RouteError {}

#[derive(Clone)]
pub struct RoutingHandle {
    routing: watch::Receiver<RoutingTable>,
    connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
    admission: AdmissionConfig,
}

impl RoutingHandle {
    pub fn new(
        routing: watch::Receiver<RoutingTable>,
        connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
    ) -> Self {
        Self {
            routing,
            connected,
            admission: AdmissionConfig::default(),
        }
    }

    pub fn with_admission(mut self, admission: AdmissionConfig) -> Self {
        self.admission = admission;
        self
    }

    pub fn admit(&self, _req: &Request) -> AdmissionDecision {
        admit(&self.routing.borrow(), &self.admission)
    }

    pub fn select_worker(&self, affinity: Option<AffinityKey>) -> Vec<WorkerId> {
        let table = self.routing.borrow();
        let connected = self.connected.borrow();
        let mut rng = next_rand;
        select_candidates(&table, &connected, None, affinity, &mut rng)
    }

    pub async fn dispatch_with_retry<W: WorkerDispatch>(
        &self,
        workers: &W,
        req: &Request,
        affinity: Option<AffinityKey>,
    ) -> Result<Dispatched, RouteError> {
        let candidates = {
            let table = self.routing.borrow();
            let connected = self.connected.borrow();
            let mut rng = next_rand;
            select_candidates(&table, &connected, None, affinity, &mut rng)
        };
        if candidates.is_empty() {
            return Err(RouteError::NoCandidate);
        }

        let mut attempts = 0usize;
        for id in candidates.into_iter().take(MAX_DISPATCH_ATTEMPTS) {
            attempts += 1;
            match workers.dispatch(id, req.clone()).await {
                Ok(accepted @ Accepted::Ok { .. }) => {
                    return Ok(Dispatched {
                        worker_id: id,
                        accepted,
                    });
                }
                Ok(Accepted::Reject) | Ok(Accepted::Redirect { .. }) => {
                    tracing::debug!(%id, req_id = %req.req_id, "worker declined turn; trying next candidate");
                }
                Err(e) => {
                    tracing::debug!(%id, req_id = %req.req_id, error = %e, "dispatch failed; trying next candidate");
                }
            }
        }
        Err(RouteError::Exhausted { attempts })
    }
}

fn load_key(w: &RoutableWorker) -> (u8, u32, u64) {
    (
        w.coarse_load.kv_pressure_bucket,
        w.coarse_load.inflight,
        w.id.0,
    )
}

fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn hrw_score(key: u64, worker: WorkerId) -> u64 {
    mix64(key ^ mix64(worker.0))
}

fn next_rand() -> u64 {
    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut s = STATE.load(Ordering::Relaxed);
    if s == 0 {
        use std::hash::{BuildHasher, Hasher};
        let seed = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish()
            | 1;
        STATE.store(seed, Ordering::Relaxed);
        s = seed;
    }
    let prev = STATE.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    mix64(prev.wrapping_add(s))
}

fn select_candidates(
    table: &RoutingTable,
    connected: &HashSet<WorkerId>,
    want_model: Option<&str>,
    affinity: Option<AffinityKey>,
    rng: &mut dyn FnMut() -> u64,
) -> Vec<WorkerId> {
    let mut eligible: Vec<&RoutableWorker> = table
        .workers
        .iter()
        .filter(|w| w.health == Health::Healthy)
        .filter(|w| w.role == Role::Decode)
        .filter(|w| connected.contains(&w.id))
        .filter(|w| want_model.is_none_or(|m| w.model == m))
        .collect();

    if eligible.is_empty() {
        return Vec::new();
    }

    match affinity {
        Some(key) => {
            eligible.sort_by_key(|w| (std::cmp::Reverse(hrw_score(key, w.id)), w.id.0));
            eligible.iter().map(|w| w.id).collect()
        }
        None => {
            let primary_pos = p2c_pick(&eligible, rng);
            let primary = eligible.remove(primary_pos);
            eligible.sort_by_key(|w| load_key(w));
            std::iter::once(primary.id)
                .chain(eligible.iter().map(|w| w.id))
                .collect()
        }
    }
}

fn p2c_pick(eligible: &[&RoutableWorker], rng: &mut dyn FnMut() -> u64) -> usize {
    let n = eligible.len();
    if n == 1 {
        return 0;
    }
    let i = (rng() % n as u64) as usize;
    let mut j = (rng() % (n as u64 - 1)) as usize;
    if j >= i {
        j += 1;
    }
    if load_key(eligible[i]) <= load_key(eligible[j]) {
        i
    } else {
        j
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_api::ClientMessage;
    use controller_api::{Role, WorkerStatus};
    use ids::{ReqId, SessionId, TenantId};
    use std::sync::Mutex;
    use worker_api::Priority;

    fn worker(id: u64, model: &str, health: Health, kv: u8, inflight: u32) -> RoutableWorker {
        RoutableWorker {
            id: WorkerId(id),
            addr: format!("10.0.0.{id}:7000"),
            role: Role::Decode,
            model: model.to_string(),
            health,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: kv,
                inflight,
            },
        }
    }

    fn table(workers: Vec<RoutableWorker>) -> RoutingTable {
        RoutingTable { epoch: 1, workers }
    }

    fn connset(ids: &[u64]) -> HashSet<WorkerId> {
        ids.iter().map(|&i| WorkerId(i)).collect()
    }

    fn scripted(seq: Vec<u64>) -> impl FnMut() -> u64 {
        let mut it = seq.into_iter();
        move || it.next().unwrap_or(0)
    }

    #[test]
    fn route_every_case() {
        filters_unhealthy_disconnected_and_model();
        filters_executor_roles();
        empty_when_nothing_eligible();
    }

    fn filters_unhealthy_disconnected_and_model() {
        let t = table(vec![
            worker(1, "m", Health::Unreachable, 0, 0), // unhealthy
            worker(2, "m", Health::Healthy, 0, 0),     // not connected
            worker(3, "other", Health::Healthy, 0, 0), // wrong model
            worker(4, "m", Health::Healthy, 0, 0),     // ✓ only eligible
        ]);
        let conn = connset(&[1, 3, 4]);
        let mut rng = scripted(vec![]);
        assert_eq!(
            select_candidates(&t, &conn, Some("m"), Some(7), &mut rng),
            vec![WorkerId(4)]
        );
    }

    fn filters_executor_roles() {
        let mut prefill = worker(1, "m", Health::Healthy, 0, 0);
        prefill.role = Role::Prefill;
        let mut encode = worker(2, "m", Health::Healthy, 0, 0);
        encode.role = Role::Encode;
        let decode = worker(3, "m", Health::Healthy, 0, 0);
        let t = table(vec![prefill, encode, decode]);
        let conn = connset(&[1, 2, 3]);
        let mut rng = scripted(vec![]);
        assert_eq!(
            select_candidates(&t, &conn, Some("m"), None, &mut rng),
            vec![WorkerId(3)]
        );
    }

    fn empty_when_nothing_eligible() {
        let t = table(vec![worker(1, "m", Health::Unreachable, 0, 0)]);
        let conn = connset(&[1]);
        let mut rng = scripted(vec![]);
        assert!(select_candidates(&t, &conn, Some("m"), Some(1), &mut rng).is_empty());
        assert!(select_candidates(&t, &conn, None, None, &mut rng).is_empty());
    }

    fn req() -> Request {
        Request {
            req_id: ReqId(7),
            session: SessionId(1),
            tenant: TenantId("t".to_string()),
            priority: Priority::Normal,
            blobs: Vec::new(),
            message: ClientMessage::Query {
                corr_id: 1,
                subject: "s".to_string(),
                record: "r".to_string(),
            },
        }
    }

    struct StubRegistry {
        answers: std::collections::HashMap<WorkerId, Result<Accepted, String>>,
        calls: Mutex<Vec<WorkerId>>,
    }

    impl WorkerDispatch for StubRegistry {
        type Err = String;
        async fn dispatch(&self, id: WorkerId, _req: Request) -> Result<Accepted, String> {
            self.calls.lock().unwrap().push(id);
            self.answers
                .get(&id)
                .cloned()
                .unwrap_or(Err("not-connected".to_string()))
        }
    }

    fn handle_with(table_v: RoutingTable, connected: &[u64]) -> RoutingHandle {
        let (_rt, rr) = watch::channel(table_v);
        let (_ct, cr) = watch::channel(Arc::new(connset(connected)));
        RoutingHandle::new(rr, cr)
    }

    #[tokio::test]
    async fn dispatch_returns_dispatched_on_first_accept() {
        let t = table(vec![worker(1, "m", Health::Healthy, 0, 0)]);
        let h = handle_with(t, &[1]);
        let reg = StubRegistry {
            answers: [(
                WorkerId(1),
                Ok(Accepted::Ok {
                    worker: WorkerId(1),
                }),
            )]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let d = h.dispatch_with_retry(&reg, &req(), Some(42)).await.unwrap();
        assert_eq!(d.worker_id, WorkerId(1));
        assert_eq!(
            d.accepted,
            Accepted::Ok {
                worker: WorkerId(1)
            }
        );
    }

    #[tokio::test]
    async fn dispatch_advances_past_reject_and_transport_error() {
        let t = table(
            (1..=3)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let h = handle_with(t, &[1, 2, 3]);
        let order = h.select_worker(Some(99));
        assert_eq!(order.len(), 3);
        let (first, second, accepting) = (order[0], order[1], order[2]);
        let reg = StubRegistry {
            answers: [
                (first, Err("not-connected".to_string())),
                (second, Ok(Accepted::Reject)),
                (accepting, Ok(Accepted::Ok { worker: accepting })),
            ]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let d = h.dispatch_with_retry(&reg, &req(), Some(99)).await.unwrap();
        assert_eq!(d.worker_id, accepting);
        assert_eq!(*reg.calls.lock().unwrap(), vec![first, second, accepting]);
    }

    #[tokio::test]
    async fn dispatch_no_candidate_when_none_connected() {
        let t = table(vec![worker(1, "m", Health::Healthy, 0, 0)]);
        let h = handle_with(t, &[]);
        let reg = StubRegistry {
            answers: Default::default(),
            calls: Mutex::new(Vec::new()),
        };
        let err = h.dispatch_with_retry(&reg, &req(), None).await.unwrap_err();
        assert_eq!(err, RouteError::NoCandidate);
    }

    #[tokio::test]
    async fn dispatch_exhausted_when_all_reject() {
        let t = table(
            (1..=2)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let h = handle_with(t, &[1, 2]);
        let reg = StubRegistry {
            answers: [
                (WorkerId(1), Ok(Accepted::Reject)),
                (WorkerId(2), Ok(Accepted::Reject)),
            ]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let err = h
            .dispatch_with_retry(&reg, &req(), Some(5))
            .await
            .unwrap_err();
        assert_eq!(err, RouteError::Exhausted { attempts: 2 });
    }
}
