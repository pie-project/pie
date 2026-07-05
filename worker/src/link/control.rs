//! Worker control-plane **seam**: the [`ControlLink`] trait the worker's
//! register + heartbeat/report/watch loops run against, plus the distributed
//! [`pie_controller_rpc::ControlClient`] implementation.
//!
//! The seam is what keeps `pie-worker` depending only on the *contract*
//! (`pie-controller-rpc`) and never on the controller *implementation* (`pie-controller`):
//!
//! - **distributed** (always linked): [`ControlLink`] for [`ControlClient`]
//!   dials the standalone controller over tarpc; [`neighbors_watch`] spawns the
//!   `watch_worker` long-poll loop and republishes each view into a local
//!   `watch` channel.
//! - **single-node** (`single-node` feature): an `EmbeddedControl(Handle)`
//!   newtype at the composition root (the worker bin's `single_node` module)
//!   implements the same trait against the in-proc controller `Handle` — no
//!   sockets, `neighbors_watch` returns the controller's `worker_watch(id)`
//!   receiver directly.
//!
//! Either backend is injected into [`spawn_control_tasks`], so the three loops
//! are transport-agnostic.

use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use pie_controller_rpc::{Ack, ControlClient, Neighbors, WorkerInfo, WorkerStatus};
use pie_ids::{NodeId, WorkerId};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

use super::gateway::GatewayLinkManager;

/// Worker→controller heartbeat cadence. Matches the gateway's interval and sits
/// well under the controller's liveness timeout (8s) → ~4× margin, so a few
/// dropped beats never trip a false eviction.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
/// Coarse-load report cadence. The controller coalesces these per epoch (only a
/// `kv_pressure_bucket` cross advances the gateway epoch), so the exact period
/// is a liveness floor, not a wire-churn source.
const REPORT_INTERVAL: Duration = Duration::from_secs(2);
/// `watch_worker` long-poll client deadline. Must exceed the controller's
/// `T_HANG` (20s) so the server's same-epoch keepalive return always lands
/// before we time out; on error we back off and re-poll.
const WATCH_DEADLINE: Duration = Duration::from_secs(300);
/// Backoff before re-polling `watch_worker` after a transport error, so a wedged
/// or restarting controller doesn't spin the loop.
const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// The control-plane operations the worker's loops need, abstracted over the
/// transport (distributed tarpc client vs in-proc controller handle).
///
/// Mirrors the relevant `pie_controller_rpc::Control` calls minus the tarpc context.
/// `Clone` so each of the three loops can hold its own cheap copy.
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

    /// A receiver of this worker's latest neighbor view. The distributed impl
    /// spawns the `watch_worker` long-poll loop and republishes into the
    /// returned channel; the in-proc impl returns the controller's
    /// `worker_watch(id)` receiver directly (epoch-free).
    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors>;
}

impl ControlLink for ControlClient {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        // The inherent (tarpc-generated) method shadows the trait method for
        // method-call syntax, so this dispatches to the RPC, not back into us.
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
/// shared channel. The controller blocks until the worker epoch advances past
/// `since` (or its `T_HANG` keepalive fires); we re-poll with the returned
/// epoch. On a transport error we back off and retry. Exits when all receivers
/// drop (the worker is shutting down).
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

/// Dial the controller's tarpc endpoint and spawn the request dispatcher,
/// returning a [`ControlClient`]. `addr` is `tcp://host:port`, a bare
/// `host:port`, or `unix:/path`. Control messages are tiny → tarpc's default
/// frame cap is fine.
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

/// Spawn the worker's three control-plane loops against `ctrl` (the dialed
/// client or the in-proc adapter) and return their join handles, aborted on
/// engine shutdown.
///
/// - **heartbeat** every [`HEARTBEAT_INTERVAL`]; logs (but does not yet act on)
///   an [`Ack::ReRegister`].
/// - **report** coarse load every [`REPORT_INTERVAL`].
/// - **watch** the neighbor view: read-before-wait over the
///   [`ControlLink::neighbors_watch`] receiver (full snapshots, coalesced). Each
///   view also carries the global gateway roster; the watch loop owns the
///   [`GatewayLinkManager`] and reconciles its dial-in links against every
///   update (dial newly-added gateways, drop removed ones). Aborting this task
///   on shutdown drops the manager, tearing down every dial-in link.
pub fn spawn_control_tasks<C: ControlLink>(
    ctrl: C,
    worker_id: WorkerId,
    mut gateways: GatewayLinkManager,
) -> Vec<tokio::task::JoinHandle<()>> {
    let heartbeat_ctrl = ctrl.clone();
    let heartbeat_task = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
        loop {
            ticker.tick().await;
            match heartbeat_ctrl.heartbeat(NodeId::Worker(worker_id)).await {
                Ok(Ack::Ok) => {}
                Ok(Ack::ReRegister) => {
                    tracing::warn!(
                        worker = %worker_id,
                        "controller lost our registration; re-register needed"
                    );
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
                kv_pressure_bucket: 0,
                inflight: 0,
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
        loop {
            // Read-before-wait: take the current snapshot, then block for the
            // next change. A fresh subscriber's first borrow is the seeded view.
            let neighbors = rx.borrow_and_update().clone();
            tracing::debug!(
                worker = %worker_id,
                peers = neighbors.peers.len(),
                gateways = neighbors.gateways.len(),
                epoch = neighbors.epoch,
                "neighbor view updated"
            );
            // Reconcile dial-in links against the freshly-pushed gateway roster.
            gateways.reconcile(&neighbors.gateways).await;
            if rx.changed().await.is_err() {
                break; // controller gone → shutdown
            }
        }
    });

    vec![heartbeat_task, report_task, watch_task]
}
