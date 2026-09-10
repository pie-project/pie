use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, Neighbors, WorkerInfo, WorkerStatus};
use ids::{NodeId, WorkerId};
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

use super::gateway::GatewayLinkManager;
use super::partner::PartnerLinkManager;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);
const REPORT_INTERVAL: Duration = Duration::from_secs(2);
const REPORT_POLL: Duration = Duration::from_millis(100);
const LINK_HEAL_INTERVAL: Duration = Duration::from_secs(2);
const WATCH_DEADLINE: Duration = Duration::from_secs(300);
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

pub trait ControlLink: Clone + Send + Sync + 'static {
    fn register_worker(&self, info: WorkerInfo) -> impl Future<Output = Result<WorkerId>> + Send;

    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    fn report_worker(
        &self,
        id: WorkerId,
        status: WorkerStatus,
    ) -> impl Future<Output = Result<()>> + Send;

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors>;
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
            peers: Vec::new(),
            gateways: Vec::new(),
        });
        tokio::spawn(watch_neighbors_loop(self.clone(), id, tx));
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

fn report_band(status: &WorkerStatus) -> (u8, bool) {
    let bucket = status.kv_pressure_bucket;
    let band = if bucket >= 240 {
        u8::MAX
    } else if bucket >= 224 {
        u8::MAX - 1
    } else {
        bucket / 16
    };
    (band, status.inflight == 0)
}

async fn report_loop<C: ControlLink>(
    ctrl: C,
    worker_id: WorkerId,
    mut sample: impl FnMut() -> WorkerStatus,
    mut releases: watch::Receiver<u64>,
) {
    let mut ticker = tokio::time::interval(REPORT_POLL);
    let mut last: Option<(WorkerStatus, tokio::time::Instant)> = None;
    let mut releases_live = true;
    loop {
        let sender_gone = tokio::select! {
            _ = ticker.tick() => false,
            changed = releases.changed(), if releases_live => changed.is_err(),
        };
        if sender_gone {
            // `changed()` stays Err once the sender is gone, so leaving the
            // branch armed would spin the loop; the ticker still reports.
            releases_live = false;
            continue;
        }
        let status = sample();
        let due = match &last {
            None => true,
            Some((sent, at)) => {
                at.elapsed() >= REPORT_INTERVAL || report_band(sent) != report_band(&status)
            }
        };
        if !due {
            continue;
        }
        last = Some((status, tokio::time::Instant::now()));
        if runtime::planner::trace_enabled() {
            println!(
                "[report] kv_bucket={} inflight={} queue=[{}]",
                status.kv_pressure_bucket,
                status.inflight,
                runtime::planner::planner()
                    .map(|p| p.debug_queue())
                    .unwrap_or_default()
            );
        }
        if let Err(e) = ctrl.report_worker(worker_id, status).await {
            tracing::warn!(
                worker = %worker_id,
                error = %e,
                "controller report_worker transport failed"
            );
        }
    }
}

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
    let report_task = tokio::spawn(report_loop(
        report_ctrl,
        worker_id,
        || WorkerStatus {
            kv_pressure_bucket: runtime::store::kv_pressure_bucket(),
            inflight: runtime::inferlet::process::list()
                .len()
                .min(u32::MAX as usize) as u32,
        },
        runtime::inferlet::process::releases(),
    ));

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
                        if !gateways.reap_dead().is_empty() {
                            break;
                        }
                    }
                }
            }
        }
    });

    vec![heartbeat_task, report_task, watch_task]
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::sync::{mpsc, oneshot};
    use tokio::time::Instant;

    const WORKER: WorkerId = WorkerId(7);

    /// One `report_worker` call, parked until the test hands back `resume`.
    /// Holding the loop inside the RPC is what makes the cells deterministic:
    /// nothing else runs while a test sets up the next sample.
    struct Report {
        status: WorkerStatus,
        resume: oneshot::Sender<()>,
    }

    #[derive(Clone)]
    struct MockLink {
        reports: mpsc::UnboundedSender<Report>,
    }

    impl MockLink {
        fn new() -> (Self, mpsc::UnboundedReceiver<Report>) {
            let (reports, rx) = mpsc::unbounded_channel();
            (Self { reports }, rx)
        }
    }

    impl ControlLink for MockLink {
        async fn register_worker(&self, _info: WorkerInfo) -> Result<WorkerId> {
            unreachable!()
        }

        async fn heartbeat(&self, _id: NodeId) -> Result<Ack> {
            unreachable!()
        }

        async fn report_worker(&self, _id: WorkerId, status: WorkerStatus) -> Result<()> {
            let (resume, wait) = oneshot::channel();
            let _ = self.reports.send(Report { status, resume });
            let _ = wait.await;
            Ok(())
        }

        fn neighbors_watch(&self, _id: WorkerId) -> watch::Receiver<Neighbors> {
            unreachable!()
        }
    }

    struct Probe {
        status: Arc<Mutex<WorkerStatus>>,
        calls: Arc<AtomicUsize>,
    }

    impl Probe {
        fn new(kv_pressure_bucket: u8, inflight: u32) -> Self {
            Self {
                status: Arc::new(Mutex::new(WorkerStatus {
                    kv_pressure_bucket,
                    inflight,
                })),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn sampler(&self) -> Box<dyn FnMut() -> WorkerStatus + Send> {
            let status = self.status.clone();
            let calls = self.calls.clone();
            Box::new(move || {
                calls.fetch_add(1, Ordering::Relaxed);
                *status.lock().unwrap()
            })
        }

        fn set(&self, kv_pressure_bucket: u8, inflight: u32) {
            *self.status.lock().unwrap() = WorkerStatus {
                kv_pressure_bucket,
                inflight,
            };
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    /// Run the loop until it parks again. Keeps a task runnable throughout, so
    /// the paused clock cannot auto-advance underneath the cell.
    async fn settle() {
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
    }

    #[tokio::test(start_paused = true)]
    async fn release_wake_crossing_a_band_reports_without_advancing_time() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(0, 0);
        let (releases, rx) = watch::channel(0u64);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        let first = reports.recv().await.unwrap();
        assert_eq!(first.status.kv_pressure_bucket, 0);
        first.resume.send(()).unwrap();
        settle().await;

        probe.set(250, 0);
        let at = Instant::now();
        releases.send_modify(|count| *count += 1);

        let second = reports.recv().await.unwrap();
        assert_eq!(second.status.kv_pressure_bucket, 250);
        assert_eq!(at.elapsed(), Duration::ZERO);

        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn release_wake_inside_the_same_band_does_not_report() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(0, 0);
        let (releases, rx) = watch::channel(0u64);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        reports.recv().await.unwrap().resume.send(()).unwrap();
        settle().await;
        let sampled = probe.calls();

        probe.set(15, 0);
        let at = Instant::now();
        releases.send_modify(|count| *count += 1);
        settle().await;

        assert!(probe.calls() > sampled);
        assert!(reports.try_recv().is_err());
        assert_eq!(at.elapsed(), Duration::ZERO);

        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn release_wake_sent_before_the_first_poll_is_not_lost() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(0, 0);
        let (releases, rx) = watch::channel(0u64);
        releases.send_modify(|count| *count += 1);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        let first = reports.recv().await.unwrap();
        assert_eq!(first.status.kv_pressure_bucket, 0);

        probe.set(250, 0);
        let at = Instant::now();
        first.resume.send(()).unwrap();

        let second = reports.recv().await.unwrap();
        assert_eq!(second.status.kv_pressure_bucket, 250);
        assert_eq!(at.elapsed(), Duration::ZERO);

        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn periodic_report_still_arrives_with_no_release_wakes() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(0, 0);
        let (_releases, rx) = watch::channel(0u64);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        reports.recv().await.unwrap().resume.send(()).unwrap();
        settle().await;

        tokio::time::advance(REPORT_INTERVAL).await;

        let periodic = reports.recv().await.unwrap();
        assert_eq!(periodic.status.kv_pressure_bucket, 0);
        periodic.resume.send(()).unwrap();
        settle().await;
        assert!(reports.try_recv().is_err());

        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn idle_transition_on_a_tick_reports_before_the_interval() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(100, 1);
        let (_releases, rx) = watch::channel(0u64);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        let first = reports.recv().await.unwrap();
        assert_eq!(first.status.inflight, 1);
        first.resume.send(()).unwrap();
        settle().await;

        probe.set(100, 0);
        let at = Instant::now();
        tokio::time::advance(REPORT_POLL).await;

        let second = reports.recv().await.unwrap();
        assert_eq!(second.status.inflight, 0);
        assert_eq!(second.status.kv_pressure_bucket, 100);
        assert!(at.elapsed() < REPORT_INTERVAL);

        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn release_wake_during_the_report_rpc_resamples_after_it() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(0, 0);
        let (releases, rx) = watch::channel(0u64);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        let first = reports.recv().await.unwrap();
        assert_eq!(first.status.kv_pressure_bucket, 0);

        probe.set(250, 0);
        releases.send_modify(|count| *count += 1);
        let at = Instant::now();
        first.resume.send(()).unwrap();

        let second = reports.recv().await.unwrap();
        assert_eq!(second.status.kv_pressure_bucket, 250);
        assert_eq!(at.elapsed(), Duration::ZERO);

        task.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn dropped_release_sender_keeps_the_periodic_report_without_spinning() {
        let (link, mut reports) = MockLink::new();
        let probe = Probe::new(0, 0);
        let (releases, rx) = watch::channel(0u64);
        drop(releases);
        let task = tokio::spawn(report_loop(link, WORKER, probe.sampler(), rx));

        reports.recv().await.unwrap().resume.send(()).unwrap();
        settle().await;

        tokio::time::advance(REPORT_INTERVAL).await;

        let periodic = reports.recv().await.unwrap();
        assert_eq!(periodic.status.kv_pressure_bucket, 0);
        periodic.resume.send(()).unwrap();
        settle().await;
        assert!(reports.try_recv().is_err());
        // One interval holds about twenty polls; a spinning select would sample
        // orders of magnitude more.
        assert!(probe.calls() < 64);

        task.abort();
    }
}
