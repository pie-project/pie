use std::collections::HashSet;
use std::sync::Arc;

use controller_api::{Health, Role, RoutingTable};
use ids::WorkerId;
use tokio::sync::watch;
use worker_api::{Accepted, Request};

use crate::admission::{AdmissionDecision, admit};

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

/// The routing table as the controller last sent it, and the workers linked
/// to this gateway. It places nothing by load: a launch goes where its
/// placement row lists, else to the eligible workers in the table's order.
#[derive(Clone)]
pub struct RoutingHandle {
    routing: watch::Receiver<RoutingTable>,
    connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
}

impl RoutingHandle {
    pub fn new(
        routing: watch::Receiver<RoutingTable>,
        connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
    ) -> Self {
        Self { routing, connected }
    }

    pub fn table_watch(&self) -> watch::Receiver<RoutingTable> {
        self.routing.clone()
    }

    // Only a launch takes a seat on a worker: a ping, an upload, a query or
    // a terminate holds no KV and no lane, so it is never refused here.
    pub fn admit(&self, req: &Request) -> AdmissionDecision {
        if !matches!(
            &req.message,
            client_api::ClientMessage::LaunchProcess { .. }
        ) {
            return AdmissionDecision::Admit;
        }
        admit(&self.routing.borrow())
    }

    /// The eligible workers, in the table's order.
    pub fn select_worker(&self) -> Vec<WorkerId> {
        select_candidates(&self.routing.borrow(), &self.connected.borrow(), None)
    }

    /// Tries the eligible workers in order, `preferred` first when it is
    /// one, and only `pinned` when set, until one accepts.
    pub async fn dispatch_with_retry<W: WorkerDispatch>(
        &self,
        workers: &W,
        req: &Request,
        preferred: Option<WorkerId>,
        pinned: Option<WorkerId>,
    ) -> Result<Dispatched, RouteError> {
        let candidates = {
            let mut candidates = self.select_worker();
            if let Some(at) = preferred.and_then(|p| candidates.iter().position(|&w| w == p)) {
                let worker = candidates.remove(at);
                candidates.insert(0, worker);
            }
            if let Some(pinned) = pinned {
                candidates.retain(|&worker| worker == pinned);
            }
            candidates
        };
        if candidates.is_empty() {
            return Err(RouteError::NoCandidate);
        }

        let mut attempts = 0usize;
        for id in candidates {
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

/// The healthy, linked decode workers (of `want_model`, when named), in the
/// table's order.
fn select_candidates(
    table: &RoutingTable,
    connected: &HashSet<WorkerId>,
    want_model: Option<&str>,
) -> Vec<WorkerId> {
    table
        .workers
        .iter()
        .filter(|w| w.health == Health::Healthy)
        .filter(|w| w.role == Role::Decode)
        .filter(|w| connected.contains(&w.id))
        .filter(|w| want_model.is_none_or(|m| w.model == m))
        .map(|w| w.id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use client_api::ClientMessage;
    use controller_api::{RoutableWorker, WorkerStatus};
    use ids::{ReqId, SessionId, TenantId};
    use std::sync::Mutex;
    use worker_api::Priority;

    fn worker(id: u64, model: &str, health: Health, kv: u8, inflight: u32) -> RoutableWorker {
        RoutableWorker {
            id: WorkerId(id),
            addr: format!("10.0.0.{id}:7000"),
            role: Role::Decode,
            model: model.to_string(),
            engine: String::new(),
            health,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: kv,
                inflight,
                ..WorkerStatus::default()
            },
        }
    }

    fn table(workers: Vec<RoutableWorker>) -> RoutingTable {
        RoutingTable {
            epoch: 1,
            workers,
            author: None,
        }
    }

    fn connset(ids: &[u64]) -> HashSet<WorkerId> {
        ids.iter().map(|&i| WorkerId(i)).collect()
    }

    #[test]
    fn route_every_case() {
        filters_unhealthy_disconnected_and_model();
        filters_executor_roles();
        empty_when_nothing_eligible();
        the_table_order_holds_whatever_the_load();
    }

    fn filters_unhealthy_disconnected_and_model() {
        let t = table(vec![
            worker(1, "m", Health::Unreachable, 0, 0), // unhealthy
            worker(2, "m", Health::Healthy, 0, 0),     // not connected
            worker(3, "other", Health::Healthy, 0, 0), // wrong model
            worker(4, "m", Health::Healthy, 0, 0),     // ✓ only eligible
        ]);
        let conn = connset(&[1, 3, 4]);
        assert_eq!(select_candidates(&t, &conn, Some("m")), vec![WorkerId(4)]);
    }

    fn filters_executor_roles() {
        let mut prefill = worker(1, "m", Health::Healthy, 0, 0);
        prefill.role = Role::Prefill;
        let mut encode = worker(2, "m", Health::Healthy, 0, 0);
        encode.role = Role::Encode;
        let decode = worker(3, "m", Health::Healthy, 0, 0);
        let t = table(vec![prefill, encode, decode]);
        let conn = connset(&[1, 2, 3]);
        assert_eq!(select_candidates(&t, &conn, Some("m")), vec![WorkerId(3)]);
    }

    fn empty_when_nothing_eligible() {
        let t = table(vec![worker(1, "m", Health::Unreachable, 0, 0)]);
        let conn = connset(&[1]);
        assert!(select_candidates(&t, &conn, Some("m")).is_empty());
        assert!(select_candidates(&t, &conn, None).is_empty());
    }

    /// With no placement row, the eligible workers go in the table's order:
    /// load orders nothing here, the cluster policy's rows do.
    fn the_table_order_holds_whatever_the_load() {
        let t = table(vec![
            worker(3, "m", Health::Healthy, 255, 1000),
            worker(1, "m", Health::Healthy, 0, 0),
            worker(2, "m", Health::Healthy, 128, 10),
        ]);
        let conn = connset(&[1, 2, 3]);
        assert_eq!(
            select_candidates(&t, &conn, None),
            vec![WorkerId(3), WorkerId(1), WorkerId(2)]
        );
    }

    fn req() -> Request {
        Request {
            req_id: ReqId(7),
            session: SessionId(1),
            tenant: TenantId("t".to_string()),
            user: "u".to_string(),
            priority: Priority::Normal,
            blobs: Vec::new(),
            cluster_props: Default::default(),
            executors: Vec::new(),
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

    fn launch() -> Request {
        let mut request = req();
        request.message = ClientMessage::LaunchProcess {
            corr_id: 1,
            inferlet: "chat".into(),
            input: String::new(),
            capture_outputs: false,
            props: Default::default(),
            session: None,
            files: Default::default(),
            handles: Default::default(),
            returns: Default::default(),
            cluster_props: Default::default(),
            executors: Vec::new(),
        };
        request
    }

    /// The gateway holds no launch for load (the cluster policy does): a
    /// launch is refused only when no worker is healthy, and nothing else
    /// that takes no seat ever is.
    #[test]
    fn a_launch_is_refused_only_with_no_healthy_worker() {
        let full = handle_with(
            table(vec![worker(1, "m", Health::Healthy, 255, 4096)]),
            &[1],
        );
        assert_eq!(full.admit(&launch()), AdmissionDecision::Admit);
        let gone = handle_with(table(vec![worker(1, "m", Health::Unreachable, 0, 0)]), &[1]);
        assert_eq!(
            gone.admit(&launch()),
            AdmissionDecision::Reject(crate::admission::RejectReason::NoHealthyWorker)
        );
        let mut ping = req();
        ping.message = ClientMessage::Ping { corr_id: 1 };
        assert_eq!(gone.admit(&ping), AdmissionDecision::Admit);
    }

    #[tokio::test]
    async fn process_control_still_requires_a_healthy_connected_worker() {
        for (health, connected) in [(Health::Unreachable, vec![1]), (Health::Healthy, vec![])] {
            let h = handle_with(table(vec![worker(1, "m", health, 255, 256)]), &connected);
            let reg = StubRegistry {
                answers: Default::default(),
                calls: Mutex::new(Vec::new()),
            };
            let mut request = req();
            request.message = ClientMessage::ListProcesses { corr_id: 1 };
            assert_eq!(h.admit(&request), AdmissionDecision::Admit);
            assert_eq!(
                h.dispatch_with_retry(&reg, &request, None, None)
                    .await
                    .unwrap_err(),
                RouteError::NoCandidate
            );
            assert!(reg.calls.lock().unwrap().is_empty());
        }
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
        let d = h
            .dispatch_with_retry(&reg, &req(), None, None)
            .await
            .unwrap();
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
        let order = h.select_worker();
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
        let d = h
            .dispatch_with_retry(&reg, &req(), None, None)
            .await
            .unwrap();
        assert_eq!(d.worker_id, accepting);
        assert_eq!(*reg.calls.lock().unwrap(), vec![first, second, accepting]);
    }

    #[tokio::test]
    async fn a_preferred_worker_goes_first_when_it_is_eligible() {
        let t = table(
            (1..=3)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let h = handle_with(t, &[1, 2, 3]);
        let order = h.select_worker();
        let preferred = order[2];
        let accept_all = || StubRegistry {
            answers: (1..=3)
                .map(|i| {
                    (
                        WorkerId(i),
                        Ok(Accepted::Ok {
                            worker: WorkerId(i),
                        }),
                    )
                })
                .collect(),
            calls: Mutex::new(Vec::new()),
        };
        let d = h
            .dispatch_with_retry(&accept_all(), &req(), Some(preferred), None)
            .await
            .unwrap();
        assert_eq!(d.worker_id, preferred);

        // One that is not connected is no candidate: the default order holds.
        let d = h
            .dispatch_with_retry(&accept_all(), &req(), Some(WorkerId(9)), None)
            .await
            .unwrap();
        assert_eq!(d.worker_id, order[0]);
    }

    #[tokio::test]
    async fn dispatch_no_candidate_when_none_connected() {
        let t = table(vec![worker(1, "m", Health::Healthy, 0, 0)]);
        let h = handle_with(t, &[]);
        let reg = StubRegistry {
            answers: Default::default(),
            calls: Mutex::new(Vec::new()),
        };
        let err = h
            .dispatch_with_retry(&reg, &req(), None, None)
            .await
            .unwrap_err();
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
            .dispatch_with_retry(&reg, &req(), None, None)
            .await
            .unwrap_err();
        assert_eq!(err, RouteError::Exhausted { attempts: 2 });
    }
}
