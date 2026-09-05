//! Worker control-plane seam: the [`ControlLink`] trait the worker's
//! register + heartbeat/report/watch loops run against, plus the distributed
//! [`controller_api::ControlClient`] implementation. Keeps `worker` depending
//! only on the `controller-api` contract, never the controller
//! implementation; a single-node build injects an in-proc adapter behind the
//! same trait so [`spawn_control_tasks`]'s loops stay transport-agnostic.

use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, Neighbors, WorkerInfo, WorkerStatus};
use ids::{NodeId, WorkerId};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

use super::gateway::GatewayLinkManager;
use super::partner::PartnerLinkManager;

/// Worker→controller heartbeat cadence; well under the controller's liveness
/// timeout so a few dropped beats never trip a false eviction.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
/// Coarse-load report cadence; the controller coalesces these per epoch.
const REPORT_INTERVAL: Duration = Duration::from_secs(2);
/// How often the dial-in links are checked for death when the roster is quiet.
/// Cheap (a `JoinHandle::is_finished` per link) and only ever leads to work
/// when a link has actually ended, so this can be brisk.
const LINK_HEAL_INTERVAL: Duration = Duration::from_secs(2);
/// `watch_worker` long-poll client deadline; must exceed the controller's
/// `T_HANG` keepalive so its same-epoch return always lands before we time out.
const WATCH_DEADLINE: Duration = Duration::from_secs(300);
/// Backoff before re-polling `watch_worker` after a transport error.
const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

fn restart_after_lost_registration(kind: &str) -> ! {
    #[cfg(test)]
    panic!("controller requested {kind} re-registration");
    #[cfg(not(test))]
    {
        let _ = kind;
        std::process::abort();
    }
}

/// Control-plane operations the worker's loops need, abstracted over the
/// transport. `Clone` so each of the three loops can hold its own cheap copy.
pub trait ControlLink: Clone + Send + Sync + 'static {
    /// Register this worker; returns its controller-minted [`WorkerId`].
    fn register_worker(&self, info: WorkerInfo) -> impl Future<Output = Result<WorkerId>> + Send;

    /// Liveness ping. [`Ack::ReRegister`] ⇒ the controller lost our record and
    /// the worker must re-register.
    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    /// Push this worker's coarse load (write-only).
    fn report_worker(
        &self,
        id: WorkerId,
        status: WorkerStatus,
    ) -> impl Future<Output = Result<()>> + Send;

    /// A receiver of this worker's latest neighbor view.
    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors>;
}

impl ControlLink for ControlClient {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        // The tarpc-generated inherent method shadows this trait method, so
        // this dispatches to the RPC, not back into itself.
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
            peers: Vec::new(),
            gateways: Vec::new(),
        });
        tokio::spawn(watch_neighbors_loop(self.clone(), id, tx));
        rx
    }
}

/// Long-poll `watch_worker`, republishing each new [`Neighbors`] view into the
/// shared channel; re-polls with the returned epoch, backs off on transport
/// error, exits when all receivers drop.
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
                    break; // all subscribers dropped
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

/// Dial the controller's tarpc endpoint and spawn the request dispatcher.
/// `addr` is `tcp://host:port`, a bare `host:port`, or `unix:/path`.
pub async fn dial_controller(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        let conn = unix::connect(path, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlClient::new(cfg, conn).spawn())
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlClient::new(cfg, conn).spawn())
    }
}

/// Spawn the worker's three control-plane loops against `ctrl` and return
/// their join handles.
///
/// - heartbeat every [`HEARTBEAT_INTERVAL`]; [`Ack::ReRegister`] is fatal since
///   gateway/partner state is keyed by the old worker id.
/// - report coarse load every [`REPORT_INTERVAL`].
/// - watch the neighbor view and reconcile the [`GatewayLinkManager`]'s
///   dial-in links against each update.
pub fn spawn_control_tasks<C: ControlLink>(
    ctrl: C,
    worker_id: WorkerId,
    mut gateways: GatewayLinkManager,
    partners: Option<std::sync::Arc<tokio::sync::Mutex<PartnerLinkManager>>>,
) -> Vec<tokio::task::JoinHandle<()>> {
    let heartbeat_ctrl = ctrl.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
        loop {
            ticker.tick().await;
            match heartbeat_ctrl.heartbeat(NodeId::Worker(worker_id)).await {
                Ok(Ack::Ok) => {}
                Ok(Ack::ReRegister) => {
                    tracing::error!(
                        worker = %worker_id,
                        "controller lost our registration; restarting worker"
                    );
                    restart_after_lost_registration("worker");
                }
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
        let mut ticker = tokio::time::interval(REPORT_INTERVAL);
        loop {
            ticker.tick().await;
            let status = WorkerStatus {
                kv_pressure_bucket: runtime::store::kv_pressure_bucket(),
                inflight: runtime::inferlet::process::list()
                    .len()
                    .min(u32::MAX as usize) as u32,
            };
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
                peers = last.peers.len(),
                gateways = last.gateways.len(),
                epoch = last.epoch,
                "neighbor view updated"
            );
            gateways.reconcile(&last.gateways).await;
            if let Some(partners) = partners.as_ref() {
                partners.lock().await.reconcile(&last.peers).await;
            }
            // A roster change is not the only reason a link needs attention:
            // one can simply die. Waiting only on `changed()` meant a worker
            // whose gateway link broke stayed dead forever in any deployment
            // where the roster is static — which is every standalone one.
            loop {
                tokio::select! {
                    changed = rx.changed() => {
                        if changed.is_err() {
                            return; // controller gone → shutdown
                        }
                        last = rx.borrow_and_update().clone();
                        break;
                    }
                    _ = tokio::time::sleep(LINK_HEAL_INTERVAL) => {
                        if !gateways.reap_dead().is_empty() {
                            break; // re-dial on the next pass through
                        }
                    }
                }
            }
        }
    });

    vec![heartbeat_task, report_task, watch_task]
}

/// Spawn controller liveness loops for an executor. Executors do not dial
/// gateways or query runtime/store globals.
pub fn spawn_executor_control_tasks<C: ControlLink>(
    ctrl: C,
    worker_id: WorkerId,
    stats: std::sync::Arc<crate::executor::ExecutorStats>,
    total_pages: u32,
) -> Vec<tokio::task::JoinHandle<()>> {
    let heartbeat_ctrl = ctrl.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
        loop {
            ticker.tick().await;
            match heartbeat_ctrl.heartbeat(NodeId::Worker(worker_id)).await {
                Ok(Ack::Ok) => {}
                Ok(Ack::ReRegister) => {
                    tracing::error!(
                        worker = %worker_id,
                        "controller lost executor registration; restarting executor"
                    );
                    restart_after_lost_registration("executor");
                }
                Err(error) => {
                    tracing::warn!(
                        worker = %worker_id,
                        %error,
                        "executor heartbeat transport failed"
                    );
                }
            }
        }
    });

    let report_ctrl = ctrl.clone();
    let report_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(REPORT_INTERVAL);
        loop {
            ticker.tick().await;
            let status = WorkerStatus {
                kv_pressure_bucket: stats.kv_pressure_bucket(total_pages),
                inflight: stats.inflight(),
            };
            if let Err(error) = report_ctrl.report_worker(worker_id, status).await {
                tracing::warn!(
                    worker = %worker_id,
                    %error,
                    "executor report_worker transport failed"
                );
            }
        }
    });

    let watch_task = tokio::spawn(async move {
        let mut rx = ctrl.neighbors_watch(worker_id);
        loop {
            let neighbors = rx.borrow_and_update().clone();
            tracing::debug!(
                worker = %worker_id,
                peers = neighbors.peers.len(),
                epoch = neighbors.epoch,
                "executor neighbor view updated"
            );
            if rx.changed().await.is_err() {
                break;
            }
        }
    });

    vec![heartbeat_task, report_task, watch_task]
}
