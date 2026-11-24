use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::Result;
use controller_api::{
    Ack, ControllerLink, HEARTBEAT_INTERVAL, Neighbors, StateReport, WorkerInfo, WorkerStatus,
};
use ids::{NodeId, WorkerId};
use tokio::sync::watch;

use super::gateway::GatewayLinkManager;

/// An unchanged load is restated at the heartbeat's pace.
const REPORT_INTERVAL: Duration = Duration::from_secs(2);
/// How often the load is read for a band change, which is reported at once.
const REPORT_POLL: Duration = Duration::from_millis(100);
/// How often dead gateway links are reaped and missing ones dialed again:
/// the heartbeat's pace.
const LINK_HEAL_INTERVAL: Duration = Duration::from_secs(2);

fn restart_after_lost_registration() -> ! {
    #[cfg(test)]
    panic!("controller requested worker re-registration");
    #[cfg(not(test))]
    std::process::abort();
}

pub trait ControlLink: Clone + Send + Sync + 'static {
    fn register_worker(&self, info: WorkerInfo) -> impl Future<Output = Result<WorkerId>> + Send;

    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    fn report_worker(
        &self,
        id: WorkerId,
        status: WorkerStatus,
    ) -> impl Future<Output = Result<()>> + Send;

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors>;

    /// What this worker holds, gained and lost since its last report: the
    /// controller's directory.
    fn report_state(
        &self,
        id: WorkerId,
        report: StateReport,
    ) -> impl Future<Output = Result<()>> + Send;

    /// The workers other than `asker` that hold a run of `keys` from the
    /// first: id, bulk address and run length, longest first.
    fn locate(
        &self,
        asker: WorkerId,
        keys: Vec<[u8; 32]>,
    ) -> impl Future<Output = Result<Vec<(WorkerId, String, u32)>>> + Send;

    /// Where `worker`'s peer link listens.
    fn link(&self, worker: WorkerId) -> impl Future<Output = Result<Option<String>>> + Send;

    /// Which controller this link reaches: a new one after it was dialed
    /// again, which has heard none of this worker's reports.
    fn generation(&self) -> u64 {
        0
    }
}

impl ControlLink for ControllerLink {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        self.call("register_worker", |c, ctx| async move {
            c.register_worker(ctx, info).await
        })
        .await
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        self.call("heartbeat", |c, ctx| async move { c.heartbeat(ctx, id).await })
            .await
    }

    async fn report_worker(&self, id: WorkerId, status: WorkerStatus) -> Result<()> {
        self.call("report_worker", |c, ctx| async move {
            c.report_worker(ctx, id, status).await
        })
        .await
    }

    async fn report_state(&self, id: WorkerId, report: StateReport) -> Result<()> {
        self.call("report_state", |c, ctx| async move {
            c.report_state(ctx, id, report).await
        })
        .await
    }

    async fn locate(
        &self,
        asker: WorkerId,
        keys: Vec<[u8; 32]>,
    ) -> Result<Vec<(WorkerId, String, u32)>> {
        self.call("locate", |c, ctx| async move { c.locate(ctx, asker, keys).await })
            .await
    }

    async fn link(&self, worker: WorkerId) -> Result<Option<String>> {
        self.call("link", |c, ctx| async move { c.link(ctx, worker).await })
            .await
    }

    fn generation(&self) -> u64 {
        ControllerLink::generation(self)
    }

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        self.watch(
            "watch_worker",
            Neighbors {
                epoch: 0,
                gateways: Vec::new(),
            },
            |neighbors| neighbors.epoch,
            move |c, ctx, since| async move { c.watch_worker(ctx, id, since).await },
        )
    }
}

fn report_band(status: &WorkerStatus) -> (u8, bool, u32, u32, bool) {
    // The kv pressure by sixteenths, and the seats and the queue (the queue
    // by quarters, so a long one is not chattier): what a cluster policy
    // places by.
    (
        status.kv_pressure_bucket / 16,
        status.inflight == 0,
        status.running,
        status.waiting.div_ceil(4),
        status.draining,
    )
}

static DRAINING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// This worker takes no new turns from now on, and says so in its next
/// report.
pub fn set_draining() {
    DRAINING.store(true, std::sync::atomic::Ordering::Release);
}

pub fn draining() -> bool {
    DRAINING.load(std::sync::atomic::Ordering::Acquire)
}

pub fn spawn_control_tasks<C: ControlLink>(
    ctrl: C,
    worker_id: WorkerId,
    info: WorkerInfo,
    mut gateways: GatewayLinkManager,
) -> Vec<tokio::task::JoinHandle<()>> {
    let state_task = tokio::spawn(crate::link::peer::report_state(ctrl.clone(), worker_id));
    let heartbeat_ctrl = ctrl.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
        let again = WorkerInfo {
            prior: Some(worker_id),
            ..info
        };
        loop {
            ticker.tick().await;
            match heartbeat_ctrl.heartbeat(NodeId::Worker(worker_id)).await {
                Ok(Ack::Ok) => {}
                // The controller does not know this worker (it restarted, or
                // timed it out): register again as the same worker. Only if
                // the id is not granted back must the worker start over, as
                // its links and turns are under it.
                Ok(Ack::ReRegister) => match heartbeat_ctrl.register_worker(again.clone()).await {
                    Ok(id) if id == worker_id => {
                        tracing::info!(worker = %worker_id, "registered with the controller again");
                    }
                    Ok(id) => {
                        tracing::error!(
                            worker = %worker_id,
                            granted = %id,
                            "the controller gave this worker another id; restarting worker"
                        );
                        restart_after_lost_registration();
                    }
                    Err(e) => tracing::warn!(
                        worker = %worker_id,
                        error = %format!("{e:#}"),
                        "registering again failed; retrying next tick"
                    ),
                },
                Err(e) => {
                    tracing::warn!(
                        worker = %worker_id,
                        error = %e,
                        "controller heartbeat transport failed"
                    );
                }
            }
        }
    });

    let report_ctrl = ctrl.clone();
    let report_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(REPORT_POLL);
        let mut last: Option<(WorkerStatus, Instant)> = None;
        loop {
            ticker.tick().await;
            let (running, waiting) = runtime::inferlet::process::seats();
            let (capacity_tokens, used_tokens) = runtime::store::kv_tokens();
            let status = WorkerStatus {
                kv_pressure_bucket: runtime::store::kv_pressure_bucket(),
                inflight: runtime::inferlet::process::list()
                    .len()
                    .min(u32::MAX as usize) as u32,
                running: running.min(u32::MAX as usize) as u32,
                waiting: waiting.min(u32::MAX as usize) as u32,
                capacity_tokens,
                used_tokens,
                draining: draining(),
            };
            let due = match &last {
                None => true,
                Some((sent, at)) => {
                    at.elapsed() >= REPORT_INTERVAL || report_band(sent) != report_band(&status)
                }
            };
            if !due {
                continue;
            }
            last = Some((status, Instant::now()));
            if tracing::enabled!(tracing::Level::DEBUG) {
                let (free, idle, held, total) = runtime::store::kv_census();
                tracing::debug!(
                    kv_bucket = status.kv_pressure_bucket,
                    free,
                    idle,
                    held,
                    total,
                    inflight = status.inflight,
                    queue = %runtime::planner::planner()
                        .map(|p| p.debug_queue())
                        .unwrap_or_default(),
                    "worker report"
                );
            }
            if let Err(e) = report_ctrl.report_worker(worker_id, status).await {
                tracing::warn!(
                    worker = %worker_id,
                    error = %e,
                    "controller report_worker transport failed"
                );
            }
        }
    });

    let watch_task = tokio::spawn(async move {
        let mut rx = ctrl.neighbors_watch(worker_id);
        let mut last = rx.borrow_and_update().clone();
        loop {
            tracing::debug!(
                worker = %worker_id,
                gateways = last.gateways.len(),
                epoch = last.epoch,
                "neighbor view updated"
            );
            gateways.reconcile(&last.gateways).await;
            loop {
                tokio::select! {
                    changed = rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                        last = rx.borrow_and_update().clone();
                        break;
                    }
                    _ = tokio::time::sleep(LINK_HEAL_INTERVAL) => {
                        let reaped = !gateways.reap_dead().is_empty();
                        if reaped || gateways.missing(&last.gateways) {
                            break;
                        }
                    }
                }
            }
        }
    });

    vec![state_task, heartbeat_task, report_task, watch_task]
}
