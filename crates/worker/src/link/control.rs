use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, Neighbors, StateReport, WorkerInfo, WorkerStatus};
use ids::{NodeId, WorkerId};
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

use super::gateway::GatewayLinkManager;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
const REPORT_INTERVAL: Duration = Duration::from_secs(2);
const REPORT_POLL: Duration = Duration::from_millis(100);
const LINK_HEAL_INTERVAL: Duration = Duration::from_secs(2);
const WATCH_DEADLINE: Duration = Duration::from_secs(300);
const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

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

impl ControlLink for ControlClient {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        self.register_worker(tarpc::context::current(), info)
            .await
            .context("register_worker rpc")
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        self.heartbeat(tarpc::context::current(), id)
            .await
            .context("heartbeat rpc")
    }

    async fn report_worker(&self, id: WorkerId, status: WorkerStatus) -> Result<()> {
        self.report_worker(tarpc::context::current(), id, status)
            .await
            .context("report_worker rpc")
    }

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        let (tx, rx) = watch::channel(Neighbors {
            epoch: 0,
            gateways: Vec::new(),
        });
        tokio::spawn(watch_neighbors_loop(self.clone(), id, tx));
        rx
    }

    async fn report_state(&self, id: WorkerId, report: StateReport) -> Result<()> {
        ControlClient::report_state(self, tarpc::context::current(), id, report)
            .await
            .context("report_state rpc")
    }

    async fn locate(
        &self,
        asker: WorkerId,
        keys: Vec<[u8; 32]>,
    ) -> Result<Vec<(WorkerId, String, u32)>> {
        ControlClient::locate(self, tarpc::context::current(), asker, keys)
            .await
            .context("locate rpc")
    }

    async fn link(&self, worker: WorkerId) -> Result<Option<String>> {
        ControlClient::link(self, tarpc::context::current(), worker)
            .await
            .context("link rpc")
    }
}

/// A link to the controller that is dialed again when it goes, so a
/// controller restarted on its address is found again: the watch starts
/// over, and the next heartbeat hears it does not know this worker, which
/// registers again under its id.
#[derive(Clone)]
pub struct Controller {
    addr: std::sync::Arc<str>,
    client: std::sync::Arc<std::sync::Mutex<(ControlClient, u64)>>,
    dialing: std::sync::Arc<tokio::sync::Mutex<()>>,
}

impl Controller {
    pub async fn dial(addr: &str) -> Result<Self> {
        let client = dial_controller(addr).await?;
        Ok(Self {
            addr: addr.into(),
            client: std::sync::Arc::new(std::sync::Mutex::new((client, 0))),
            dialing: Default::default(),
        })
    }

    /// The client, and which dial it came from.
    fn client(&self) -> (ControlClient, u64) {
        self.client.lock().unwrap().clone()
    }

    /// After `error` on the link from dial `seen`: if the link is gone,
    /// dials again, unless another caller already has.
    async fn heal(&self, error: &tarpc::client::RpcError, seen: u64) {
        if !matches!(error, tarpc::client::RpcError::Shutdown) {
            return;
        }
        let _dialing = self.dialing.lock().await;
        if self.client().1 != seen {
            return;
        }
        match dial_controller(&self.addr).await {
            Ok(client) => {
                *self.client.lock().unwrap() = (client, seen + 1);
                tracing::info!(controller = %self.addr, "dialed the controller again");
            }
            Err(e) => tracing::debug!(error = %format!("{e:#}"), "the controller is not back yet"),
        }
    }
}

impl ControlLink for Controller {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        let (client, seen) = self.client();
        match client
            .register_worker(tarpc::context::current(), info)
            .await
        {
            Ok(id) => Ok(id),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("register_worker rpc")
            }
        }
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        let (client, seen) = self.client();
        match client.heartbeat(tarpc::context::current(), id).await {
            Ok(ack) => Ok(ack),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("heartbeat rpc")
            }
        }
    }

    async fn report_worker(&self, id: WorkerId, status: WorkerStatus) -> Result<()> {
        let (client, seen) = self.client();
        match client
            .report_worker(tarpc::context::current(), id, status)
            .await
        {
            Ok(()) => Ok(()),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("report_worker rpc")
            }
        }
    }

    async fn report_state(&self, id: WorkerId, report: StateReport) -> Result<()> {
        let (client, seen) = self.client();
        match client
            .report_state(tarpc::context::current(), id, report)
            .await
        {
            Ok(()) => Ok(()),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("report_state rpc")
            }
        }
    }

    async fn locate(
        &self,
        asker: WorkerId,
        keys: Vec<[u8; 32]>,
    ) -> Result<Vec<(WorkerId, String, u32)>> {
        let (client, seen) = self.client();
        match client.locate(tarpc::context::current(), asker, keys).await {
            Ok(found) => Ok(found),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("locate rpc")
            }
        }
    }

    async fn link(&self, worker: WorkerId) -> Result<Option<String>> {
        let (client, seen) = self.client();
        match client.link(tarpc::context::current(), worker).await {
            Ok(link) => Ok(link),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("link rpc")
            }
        }
    }

    fn generation(&self) -> u64 {
        self.client().1
    }

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        let (tx, rx) = watch::channel(Neighbors {
            epoch: 0,
            gateways: Vec::new(),
        });
        let control = self.clone();
        tokio::spawn(async move {
            let mut since = 0u64;
            loop {
                let (client, seen) = control.client();
                let mut ctx = tarpc::context::current();
                ctx.deadline = Instant::now() + WATCH_DEADLINE;
                match client.watch_worker(ctx, id, since).await {
                    Ok(neighbors) => {
                        since = neighbors.epoch;
                        if tx.send(neighbors).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(worker = %id, error = %e, "controller watch_worker transport failed");
                        control.heal(&e, seen).await;
                        if control.client().1 != seen {
                            // A new controller counts its epochs afresh.
                            since = 0;
                        }
                        tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
                    }
                }
            }
        });
        rx
    }
}

async fn watch_neighbors_loop(
    client: ControlClient,
    worker_id: WorkerId,
    tx: watch::Sender<Neighbors>,
) {
    let mut since = 0u64;
    loop {
        let mut ctx = tarpc::context::current();
        ctx.deadline = Instant::now() + WATCH_DEADLINE;
        match client.watch_worker(ctx, worker_id, since).await {
            Ok(neighbors) => {
                since = neighbors.epoch;
                if tx.send(neighbors).is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(
                    worker = %worker_id,
                    error = %e,
                    "controller watch_worker transport failed"
                );
                tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
            }
        }
    }
}

pub async fn dial_controller(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        #[cfg(unix)]
        {
            let conn = unix::connect(path, Bincode::default)
                .await
                .with_context(|| format!("dialing controller at {addr}"))?;
            Ok(ControlClient::new(cfg, conn).spawn())
        }
        #[cfg(not(unix))]
        {
            let _ = (path, cfg);
            anyhow::bail!(
                "{addr}: a `unix://` control address is distributed serving, which needs a unix-domain socket; this build is single-node and speaks `tcp://`"
            )
        }
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        let _ = conn.get_ref().set_nodelay(true);
        Ok(ControlClient::new(cfg, conn).spawn())
    }
}

fn report_band(status: &WorkerStatus) -> (u8, bool, u32, u32, bool) {
    let bucket = status.kv_pressure_bucket;
    let band = if bucket >= 240 {
        u8::MAX
    } else if bucket >= 224 {
        u8::MAX - 1
    } else {
        bucket / 16
    };
    // Seats and the queue move the router's load attrs: report a change
    // in either (the queue by quarters, so a long one is not chattier).
    (
        band,
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
            if runtime::planner::trace_enabled() {
                let (free, idle, held, total) = runtime::store::kv_census();
                println!(
                    "[report] kv_bucket={} pages free={free} idle={idle} held={held} of {total} inflight={} queue=[{}]",
                    status.kv_pressure_bucket,
                    status.inflight,
                    runtime::planner::planner()
                        .map(|p| p.debug_queue())
                        .unwrap_or_default()
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
