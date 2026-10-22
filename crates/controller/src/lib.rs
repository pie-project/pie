mod actor;
mod actuator;
pub mod cluster;
pub mod directory;
mod service;
mod state;
mod store;
mod topology;

pub use actuator::Slot;
pub use store::{SoftState, StateStore};

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use controller_api::{
    Ack, DirectoryView, GatewayDecisions, GatewayInfo, Key, Neighbors, ProcessEvent, RoutingTable,
    ScaleStatus, StateReport, WorkerInfo, WorkerStatus,
};
use ids::{GatewayId, NodeId, WorkerId};

use actor::{Actor, ActorConfig, Command};
use topology::{Topology, empty_routing, project};

const T_HANG: Duration = Duration::from_secs(20);
/// How often a changed directory is published to the gateways: reports
/// within it go out as one view.
const DIRECTORY_EVERY: Duration = Duration::from_millis(200);

/// The controller's actor queue: commands from every worker and gateway
/// link wait here.
const COMMAND_BUFFER: usize = 256;

#[derive(Debug, Clone)]
pub struct Config {
    /// Where the Control endpoint listens: `host:port`, `tcp://host:port`
    /// or `unix:/path`.
    pub listen: String,
    pub heartbeat_timeout: Duration,
    pub tick_interval: Duration,
    /// The workers the actuator may launch to meet a scale table; none,
    /// and the controller does not scale (`cluster.scale` is absent).
    pub slots: Vec<Slot>,
    /// The cluster policies: each a built-in policy's name or a `.wasm`
    /// path, no two granted the same write. None: every gateway goes by its
    /// defaults.
    pub cluster_policies: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: "0.0.0.0:7000".to_string(),
            heartbeat_timeout: Duration::from_secs(8),
            tick_interval: Duration::from_secs(2),
            slots: Vec::new(),
            cluster_policies: Vec::new(),
        }
    }
}

/// The controller's file: `listen`, `heartbeat_timeout` and `tick_interval`
/// (written with their unit, `"8s"`), `cluster_policies`, and
/// `[[scale.slot]]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigToml {
    listen: String,
    heartbeat_timeout: units::Duration,
    tick_interval: units::Duration,
    scale: ScaleToml,
    cluster_policies: Vec<String>,
}

/// `[scale]`: `[[scale.slot]]` entries, each a worker the actuator may
/// launch (`model`, `engine`, `command`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ScaleToml {
    slot: Vec<Slot>,
}

impl Default for ConfigToml {
    fn default() -> Self {
        let d = Config::default();
        Self {
            listen: d.listen,
            heartbeat_timeout: units::Duration::from_secs(d.heartbeat_timeout.as_secs()),
            tick_interval: units::Duration::from_secs(d.tick_interval.as_secs()),
            scale: ScaleToml::default(),
            cluster_policies: d.cluster_policies,
        }
    }
}

impl From<ConfigToml> for Config {
    fn from(t: ConfigToml) -> Self {
        Self {
            listen: t.listen,
            heartbeat_timeout: t.heartbeat_timeout.into(),
            tick_interval: t.tick_interval.into(),
            slots: t.scale.slot,
            cluster_policies: t.cluster_policies,
        }
    }
}

impl Config {
    pub fn parse(s: &str) -> Result<Config> {
        let raw: ConfigToml = toml::from_str(s).context("parse controller config (TOML)")?;
        let config = Config::from(raw);
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.listen.trim().is_empty(),
            "controller config: `listen` must not be empty"
        );
        ensure!(
            !self.heartbeat_timeout.is_zero(),
            "controller config: `heartbeat_timeout` must be > 0"
        );
        ensure!(
            !self.tick_interval.is_zero(),
            "controller config: `tick_interval` must be > 0"
        );
        for (at, slot) in self.slots.iter().enumerate() {
            ensure!(
                !slot.command.is_empty(),
                "controller config: `[[scale.slot]]` {at} has an empty `command`"
            );
            ensure!(
                !slot.engine.trim().is_empty(),
                "controller config: `[[scale.slot]]` {at} names no `engine`"
            );
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct ControllerHandle {
    cmd: mpsc::Sender<Command>,
    worker_rx: watch::Receiver<Topology>,
    gateway_rx: watch::Receiver<RoutingTable>,
    directory: directory::Directory,
    /// Set by a report that changed something, cleared as the view goes out.
    directory_dirty: Arc<std::sync::atomic::AtomicBool>,
    directory_rx: watch::Receiver<DirectoryView>,
    /// The cluster policies, when any run.
    cluster: Arc<Mutex<Option<Arc<cluster::Host>>>>,
    shutdown: CancellationToken,
    tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

pub type Handle = ControllerHandle;

impl ControllerHandle {
    pub async fn register_worker(&self, info: WorkerInfo) -> WorkerId {
        let (reply, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::RegisterWorker { info, reply }).await;
        rx.await.expect("controller actor stopped")
    }

    pub async fn register_gateway(&self, info: GatewayInfo) -> GatewayId {
        let (reply, rx) = oneshot::channel();
        let _ = self
            .cmd
            .send(Command::RegisterGateway {
                addr: info.addr,
                prior: info.prior,
                reply,
            })
            .await;
        rx.await.expect("controller actor stopped")
    }

    pub async fn heartbeat(&self, id: NodeId) -> Ack {
        let (reply, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::Heartbeat { node: id, reply }).await;
        rx.await.expect("controller actor stopped")
    }

    pub async fn report_worker(&self, id: WorkerId, status: WorkerStatus) {
        let _ = self.cmd.send(Command::ReportWorker { id, status }).await;
    }

    /// Each shape the scale table names, desired against current; none
    /// once the controller's actor has stopped.
    pub async fn scale_status(&self) -> Vec<ScaleStatus> {
        let (reply, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::ScaleStatus { reply }).await;
        rx.await.unwrap_or_default()
    }

    pub fn worker_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        let mut topo_rx = self.worker_rx.clone();
        let initial = project(&topo_rx.borrow(), id);
        let (tx, rx) = watch::channel(initial);
        tokio::spawn(async move {
            while topo_rx.changed().await.is_ok() {
                let view = project(&topo_rx.borrow(), id);
                if tx.send(view).is_err() {
                    break;
                }
            }
        });
        rx
    }

    pub fn gateway_watch(&self) -> watch::Receiver<RoutingTable> {
        self.gateway_rx.clone()
    }

    /// What `worker` holds, gained and lost since its last report.
    pub fn report_state(&self, worker: WorkerId, report: StateReport) {
        let quiet = !report.full
            && report.added.is_empty()
            && report.removed.is_empty()
            && report.runs_added.is_empty()
            && report.runs_removed.is_empty()
            && report.published.is_none()
            && report.disk_added.is_empty()
            && report.disk_removed.is_empty()
            && report.persist_hints.is_none()
            && report.disk_tier.is_none()
            && report.link.is_none();
        self.directory.report_state(worker, report);
        if !quiet {
            self.directory_changed();
        }
    }

    /// `worker` holds nothing any more.
    pub fn forget_state(&self, worker: WorkerId) {
        self.directory.forget(worker);
        self.directory_changed();
    }

    fn directory_changed(&self) {
        self.directory_dirty
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub fn locate(&self, asker: WorkerId, keys: &[Key]) -> Vec<(WorkerId, String, u32)> {
        self.directory.holders(asker, keys)
    }

    pub fn link(&self, worker: WorkerId) -> Option<String> {
        self.directory.link(worker)
    }

    pub fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
        self.directory_rx.clone()
    }

    /// What happened to `gateway`'s processes, for the cluster policies.
    pub fn cluster_events(&self, gateway: GatewayId, events: Vec<ProcessEvent>) {
        if let Some(host) = self.cluster.lock().unwrap().as_ref() {
            host.events(gateway, events);
        }
    }

    /// The cluster policies' decisions about `gateway`'s processes, as they
    /// change; with none running, every launch goes by its gateway's
    /// default.
    pub fn decisions_watch(&self, gateway: GatewayId) -> watch::Receiver<GatewayDecisions> {
        match self.cluster.lock().unwrap().as_ref() {
            Some(host) => host.decisions(gateway),
            None => {
                watch::channel(GatewayDecisions {
                    epoch: 1,
                    heard: u64::MAX,
                    ..GatewayDecisions::default()
                })
                .1
            }
        }
    }

    pub(crate) async fn watch_decisions_poll(
        &self,
        gateway: GatewayId,
        since: u64,
    ) -> GatewayDecisions {
        let mut rx = self.decisions_watch(gateway);
        loop {
            if rx.borrow().epoch > since {
                return rx.borrow().clone();
            }
            match tokio::time::timeout(T_HANG, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return rx.borrow().clone(),
            }
        }
    }

    /// Swaps the cluster policies for those `specs` names (SIGHUP): the new
    /// set loads whole, or the old one runs on. With none running, starts
    /// them.
    pub async fn reload_policies(&self, specs: &[String]) -> Result<()> {
        let programs = policy_components(specs)?;
        let host = self.cluster.lock().unwrap().clone();
        match host {
            Some(host) => host
                .reload(programs)
                .await
                .context("reload the cluster policies"),
            None => {
                let host = start_cluster(
                    programs,
                    self.gateway_rx.clone(),
                    self.directory_rx.clone(),
                    self.cmd.clone(),
                )?;
                *self.cluster.lock().unwrap() = Some(host);
                Ok(())
            }
        }
    }

    pub(crate) async fn watch_directory_poll(&self, since: u64) -> DirectoryView {
        let mut rx = self.directory_rx.clone();
        loop {
            if rx.borrow().epoch > since {
                return rx.borrow().clone();
            }
            match tokio::time::timeout(T_HANG, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return rx.borrow().clone(),
            }
        }
    }

    pub(crate) async fn watch_worker_poll(&self, id: WorkerId, since: u64) -> Neighbors {
        let mut rx = self.worker_rx.clone();
        loop {
            if rx.borrow().epoch > since {
                return project(&rx.borrow(), id);
            }
            match tokio::time::timeout(T_HANG, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return project(&rx.borrow(), id),
            }
        }
    }

    pub(crate) async fn watch_gateway_poll(&self, since: u64) -> RoutingTable {
        let mut rx = self.gateway_rx.clone();
        loop {
            if rx.borrow().epoch > since {
                return rx.borrow().clone();
            }
            match tokio::time::timeout(T_HANG, rx.changed()).await {
                Ok(Ok(())) => continue,
                Ok(Err(_)) | Err(_) => return rx.borrow().clone(),
            }
        }
    }

    pub async fn shutdown(self) {
        self.shutdown.cancel();
        let tasks =
            std::mem::take(&mut *self.tasks.lock().expect("controller tasks lock poisoned"));
        for task in tasks {
            let _ = task.await;
        }
    }
}

pub fn embed(config: Config) -> ControllerHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(COMMAND_BUFFER);
    let (worker_tx, worker_rx) = watch::channel(Topology::default());
    let (gateway_tx, gateway_rx) = watch::channel(empty_routing());
    let shutdown = CancellationToken::new();

    let actor = Actor::new(
        cmd_rx,
        worker_tx,
        gateway_tx,
        ActorConfig {
            heartbeat_timeout: config.heartbeat_timeout,
        },
        actuator::Actuator::new(config.slots.clone(), config.heartbeat_timeout),
    );
    let actor_token = shutdown.clone();
    let actor_task = tokio::spawn(async move {
        tokio::select! {
            _ = actor.run() => {}
            _ = actor_token.cancelled() => {}
        }
    });

    let tick_cmd = cmd_tx.clone();
    let interval = config.tick_interval;
    let reaper_token = shutdown.clone();
    let reaper_task = tokio::spawn(async move {
        let mut timer = tokio::time::interval(interval);
        loop {
            tokio::select! {
                _ = timer.tick() => {
                    if tick_cmd.send(Command::Tick).await.is_err() {
                        break;
                    }
                }
                _ = reaper_token.cancelled() => break,
            }
        }
    });

    let directory = directory::Directory::default();
    let directory_dirty = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (directory_tx, directory_rx) = watch::channel(DirectoryView::default());
    let directory_task = tokio::spawn(publish_directory(
        directory.clone(),
        directory_dirty.clone(),
        directory_tx,
        gateway_rx.clone(),
        shutdown.clone(),
    ));

    // A policy that does not load is the operator's to fix; the cluster
    // runs on by its defaults meanwhile.
    let cluster = if config.cluster_policies.is_empty() {
        None
    } else {
        match policy_components(&config.cluster_policies).and_then(|programs| {
            start_cluster(
                programs,
                gateway_rx.clone(),
                directory_rx.clone(),
                cmd_tx.clone(),
            )
        }) {
            Ok(host) => Some(host),
            Err(error) => {
                tracing::error!(error = %format!("{error:#}"), "cluster policies not started");
                None
            }
        }
    };

    ControllerHandle {
        cmd: cmd_tx,
        worker_rx,
        gateway_rx,
        directory,
        directory_dirty,
        directory_rx,
        cluster: Arc::new(Mutex::new(cluster)),
        shutdown,
        tasks: Arc::new(Mutex::new(vec![actor_task, reaper_task, directory_task])),
    }
}

/// The components `specs` names, each a built-in policy's name or a file.
fn policy_components(specs: &[String]) -> Result<Vec<(String, Vec<u8>)>> {
    specs
        .iter()
        .map(|spec| Ok((spec.clone(), builtins::policy_component(spec)?.into_owned())))
        .collect()
}

/// Starts the cluster policies `programs`, their scale table sent to the
/// actuator as it changes.
fn start_cluster(
    programs: Vec<(String, Vec<u8>)>,
    routing: watch::Receiver<RoutingTable>,
    directory: watch::Receiver<DirectoryView>,
    cmd: mpsc::Sender<Command>,
) -> Result<Arc<cluster::Host>> {
    let host =
        cluster::Host::start(programs, routing, directory).context("load the cluster policies")?;
    let mut scale = host.scale_watch();
    tokio::spawn(async move {
        while scale.changed().await.is_ok() {
            let entries = scale.borrow_and_update().clone();
            if let Some(entries) = entries
                && cmd.send(Command::DesireScale { entries }).await.is_err()
            {
                return;
            }
        }
    });
    Ok(host)
}

/// Publishes the directory whenever a report changed it, at most every
/// [`DIRECTORY_EVERY`]; and forgets what a worker held once the controller
/// lets it go. Pools are not the controller's: their gateway forgets them.
async fn publish_directory(
    directory: directory::Directory,
    dirty: Arc<std::sync::atomic::AtomicBool>,
    tx: watch::Sender<DirectoryView>,
    mut routing: watch::Receiver<RoutingTable>,
    shutdown: CancellationToken,
) {
    use std::sync::atomic::Ordering;
    let mut epoch = 0u64;
    let mut known: std::collections::HashSet<WorkerId> = std::collections::HashSet::new();
    let mut timer = tokio::time::interval(DIRECTORY_EVERY);
    loop {
        tokio::select! {
            _ = timer.tick() => {}
            changed = routing.changed() => {
                if changed.is_err() {
                    return;
                }
                let now: std::collections::HashSet<WorkerId> =
                    routing.borrow().workers.iter().map(|w| w.id).collect();
                for gone in known.difference(&now) {
                    directory.forget(*gone);
                    dirty.store(true, Ordering::Release);
                }
                known = now;
            }
            _ = shutdown.cancelled() => return,
        }
        if dirty.swap(false, Ordering::AcqRel) {
            epoch += 1;
            if tx.send(directory.view(epoch)).is_err() {
                return;
            }
        }
    }
}

pub async fn run(config: Config) -> Result<ControllerHandle> {
    let handle = embed(config.clone());
    let serve = service::serve(&config.listen, handle.clone(), handle.shutdown.clone())
        .await
        .with_context(|| format!("bind controller Control endpoint on {}", config.listen))?;
    handle
        .tasks
        .lock()
        .expect("controller tasks lock poisoned")
        .push(serve);
    Ok(handle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lib_every_case() {
        parse_empty_is_default();
        parse_overrides_fields();
        parse_scale_slots();
    }

    fn parse_scale_slots() {
        let cfg = Config::parse(
            r#"
            [[scale.slot]]
            model = "Qwen/Qwen3-14B"
            engine = "cuda"
            command = ["pie-worker", "-c", "wa.toml"]

            [[scale.slot]]
            engine = "cuda"
            command = ["pie-pool"]
            "#,
        )
        .expect("scale slots parse");
        assert_eq!(cfg.slots.len(), 2);
        assert_eq!(cfg.slots[0].model.as_deref(), Some("Qwen/Qwen3-14B"));
        assert_eq!(cfg.slots[1].model, None);
        assert!(Config::parse("[[scale.slot]]\nengine = \"cuda\"\ncommand = []\n").is_err());
    }

    fn parse_empty_is_default() {
        let cfg = Config::parse("").expect("empty config parses to defaults");
        let d = Config::default();
        assert_eq!(cfg.listen, d.listen);
        assert_eq!(cfg.heartbeat_timeout, d.heartbeat_timeout);
        assert_eq!(cfg.tick_interval, d.tick_interval);
    }

    fn parse_overrides_fields() {
        let cfg = Config::parse(
            r#"
            listen = "127.0.0.1:9000"
            heartbeat_timeout = "12s"
            tick_interval = "500ms"
            cluster_policies = ["persist-prefix"]
            "#,
        )
        .expect("valid config parses");
        assert_eq!(cfg.listen, "127.0.0.1:9000");
        assert_eq!(cfg.heartbeat_timeout, Duration::from_secs(12));
        assert_eq!(cfg.tick_interval, Duration::from_millis(500));
        assert_eq!(cfg.cluster_policies, ["persist-prefix"]);
        assert!(
            Config::parse("heartbeat_timeout = 12").is_err(),
            "a duration is written with its unit"
        );
    }

    #[tokio::test]
    async fn a_report_goes_out_as_a_directory_view_and_quiet_ones_do_not() {
        let handle = embed(Config::parse("").unwrap());
        let mut views = handle.directory_watch();
        handle.report_state(
            WorkerId(4),
            StateReport {
                peer: Some("w:4".into()),
                added: vec![[1; 32]],
                full: true,
                runs_added: vec![vec![[1; 32]]],
                ..StateReport::default()
            },
        );
        tokio::time::timeout(Duration::from_secs(2), views.changed())
            .await
            .expect("published")
            .unwrap();
        let view = views.borrow_and_update().clone();
        assert_eq!(view.epoch, 1);
        assert_eq!(view.groups[0].1.sites, vec![WorkerId(4)]);
        assert_eq!(handle.link(WorkerId(4)), None);
        assert_eq!(
            handle.locate(WorkerId(9), &[[1; 32]]),
            vec![(WorkerId(4), "w:4".to_string(), 1)]
        );
        assert_eq!(handle.watch_directory_poll(0).await.epoch, 1);

        handle.report_state(WorkerId(4), StateReport::default());
        assert!(
            tokio::time::timeout(DIRECTORY_EVERY * 3, views.changed())
                .await
                .is_err(),
            "a quiet report publishes nothing"
        );
        handle.forget_state(WorkerId(4));
        tokio::time::timeout(Duration::from_secs(2), views.changed())
            .await
            .expect("published")
            .unwrap();
        assert!(views.borrow().groups.is_empty());
        handle.shutdown().await;
    }
}
