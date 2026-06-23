//! `pie-controller` — Pie's cluster **control plane**.
//!
//! A registry of workers + gateways behind a **single-writer actor**. Workers
//! long-poll their `Neighbors` (who to coordinate with); gateways long-poll the
//! global `RoutingTable` (the worker roster + coarse load) to route locally.
//! Liveness is tracked from heartbeats (controller-side clock); a background
//! reaper evicts the silent. It is control plane only — tokens and KV never
//! transit it.
//!
//! # Shape
//!
//! ```text
//! service.rs  tarpc `Control` server  ─┐
//! Handle      in-proc front door      ─┼─► mpsc ─► actor.rs (sole writer)
//!                                       │            owns state.rs (Cluster)
//! reaper tick ── Command::Tick ────────┘            publishes 2 watch channels
//!                                                     topology.rs (pure planner)
//! ```
//!
//! Two deployment forms, one actor:
//! - **distributed**: [`run`] serves the `Control` RPC; workers/gateways dial it.
//! - **single-node**: [`embed`] returns a [`Handle`] the worker/gateway use
//!   in-proc — no socket, no serialization.
//!
//! # Invariants (§2/§3)
//!
//! One owner (no locks). Two **independent** epochs, each a version tag on a
//! **full** scoped snapshot (never a delta) → watches are idempotent and
//! self-healing. `role` is immutable; only join/leave moves topology. A coarse
//! load-bucket crossing re-versions only the gateway view (the load/membership
//! split that prevents watch storms).

mod actor;
mod service;
mod state;
mod store;
mod topology;

pub use store::{SoftState, StateStore};

use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tokio::sync::{mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};

use actor::{Actor, ActorConfig, Command};
use topology::{Topology, empty_routing, project};

/// Long-poll hold time. Must be **less than** the watch RPC deadline clients set
/// (~30s) so a no-change watch returns as a keepalive before the call times out.
const T_HANG: Duration = Duration::from_secs(20);

/// Controller configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Address the RPC server binds: `tcp://host:port`, a bare `host:port`, or
    /// `unix:/path`. Unused by [`embed`].
    pub listen_addr: String,
    /// Evict a member after this long without a liveness signal.
    pub heartbeat_timeout: Duration,
    /// How often the reaper scans for expired members.
    pub tick_interval: Duration,
    /// Command-channel buffer depth.
    pub command_buffer: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:7000".to_string(),
            // ≈4× the 2s client heartbeat: tolerates a couple of missed beats
            // (no false evictions) while detecting death in ~8–10s.
            heartbeat_timeout: Duration::from_secs(8),
            tick_interval: Duration::from_secs(2),
            command_buffer: 256,
        }
    }
}

/// TOML schema mirror of [`Config`] — durations as whole seconds (TOML/serde has
/// no native `Duration`). Kept private; [`Config::parse`] deserializes this and
/// converts. `#[serde(default)]` makes every field optional so a partial or empty
/// config string still yields a valid controller (fields fall back to defaults).
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigToml {
    listen_addr: String,
    heartbeat_timeout_secs: u64,
    tick_interval_secs: u64,
    command_buffer: usize,
}

impl Default for ConfigToml {
    fn default() -> Self {
        let d = Config::default();
        Self {
            listen_addr: d.listen_addr,
            heartbeat_timeout_secs: d.heartbeat_timeout.as_secs(),
            tick_interval_secs: d.tick_interval.as_secs(),
            command_buffer: d.command_buffer,
        }
    }
}

impl From<ConfigToml> for Config {
    fn from(t: ConfigToml) -> Self {
        Self {
            listen_addr: t.listen_addr,
            heartbeat_timeout: Duration::from_secs(t.heartbeat_timeout_secs),
            tick_interval: Duration::from_secs(t.tick_interval_secs),
            command_buffer: t.command_buffer,
        }
    }
}

impl Config {
    /// Parse a controller [`Config`] from a TOML string. **Pure** (Seam 1): no
    /// IO, no env, no clap — the `bootstrap` skeleton sources the string; this
    /// turns it into typed config and validates it. An empty string is valid and
    /// yields [`Config::default`].
    pub fn parse(s: &str) -> Result<Config> {
        let raw: ConfigToml = toml::from_str(s).context("parse controller config (TOML)")?;
        let config = Config::from(raw);
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.listen_addr.trim().is_empty(),
            "controller config: `listen_addr` must not be empty"
        );
        ensure!(
            self.command_buffer > 0,
            "controller config: `command_buffer` must be > 0"
        );
        ensure!(
            !self.tick_interval.is_zero(),
            "controller config: `tick_interval_secs` must be > 0"
        );
        Ok(())
    }
}

/// In-process front door to the controller actor — cloneable and cheap. Mirrors
/// the `Control` RPC calls (minus the tarpc context) so a single-node worker /
/// gateway can embed the controller and talk to it directly. For watches it
/// offers a directly-subscribable [`watch::Receiver`] (no epoch cursor needed
/// in-proc), while the distributed RPC server reuses the epoch long-poll helpers.
#[derive(Clone)]
pub struct ControllerHandle {
    cmd: mpsc::Sender<Command>,
    worker_rx: watch::Receiver<Topology>,
    gateway_rx: watch::Receiver<RoutingTable>,
    /// Cooperative shutdown signal observed by the actor, the reaper, and — for
    /// [`run`] — the serve loop. Cancelled by [`ControllerHandle::shutdown`].
    shutdown: CancellationToken,
    /// Background tasks (actor, reaper, and the serve loop when started via
    /// [`run`]) joined on shutdown so the process drains before exit. Shared so
    /// the cloneable handle stays cheap; `shutdown` drains them once.
    tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

/// Back-compat alias for [`ControllerHandle`]. The in-proc composition root
/// (`bin/pie`'s `EmbeddedControl`) and the single-node worker path refer to the
/// controller's handle as `pie_controller::Handle`.
pub type Handle = ControllerHandle;

impl ControllerHandle {
    /// Register a worker; returns its controller-minted [`WorkerId`].
    pub async fn register_worker(&self, info: WorkerInfo) -> WorkerId {
        let (reply, rx) = oneshot::channel();
        let _ = self
            .cmd
            .send(Command::RegisterWorker {
                role: info.role,
                model: info.model,
                addr: info.addr,
                reply,
            })
            .await;
        rx.await.expect("controller actor stopped")
    }

    /// Register a gateway; returns its controller-minted [`GatewayId`].
    pub async fn register_gateway(&self, info: GatewayInfo) -> GatewayId {
        let (reply, rx) = oneshot::channel();
        let _ = self
            .cmd
            .send(Command::RegisterGateway {
                addr: info.addr,
                reply,
            })
            .await;
        rx.await.expect("controller actor stopped")
    }

    /// Liveness ping; [`Ack::ReRegister`] means the controller has no record of
    /// `id` (restart / timeout) and the node must re-register.
    pub async fn heartbeat(&self, id: NodeId) -> Ack {
        let (reply, rx) = oneshot::channel();
        let _ = self.cmd.send(Command::Heartbeat { node: id, reply }).await;
        rx.await.expect("controller actor stopped")
    }

    /// Push a worker's coarse load (write-only).
    pub async fn report_worker(&self, id: WorkerId, status: WorkerStatus) {
        let _ = self.cmd.send(Command::ReportWorker { id, status }).await;
    }

    /// Directly subscribe to a worker's neighbor view (epoch-free single-node
    /// path). A small projector task forwards each membership change as the
    /// worker's scoped [`Neighbors`]; await `changed()` on the returned receiver.
    pub fn worker_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        let mut topo_rx = self.worker_rx.clone();
        let initial = project(&topo_rx.borrow(), id);
        let (tx, rx) = watch::channel(initial);
        tokio::spawn(async move {
            while topo_rx.changed().await.is_ok() {
                let view = project(&topo_rx.borrow(), id);
                if tx.send(view).is_err() {
                    break; // subscriber dropped
                }
            }
        });
        rx
    }

    /// Directly subscribe to the global routing table (epoch-free single-node
    /// path). The gateway view is already global, so this is the raw receiver.
    pub fn gateway_watch(&self) -> watch::Receiver<RoutingTable> {
        self.gateway_rx.clone()
    }

    /// Epoch long-poll for `watch_worker` (distributed read-path): block until
    /// the worker epoch passes `since`, then return the scoped view; on a hang
    /// timeout return the current view (same-epoch keepalive → client re-polls).
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

    /// Epoch long-poll for `watch_gateway`.
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

    /// Shut the controller down cleanly (Seam 1). Cancels the shared shutdown
    /// token — the actor, the reaper, and (for [`run`]) the serve loop observe it
    /// and stop — then joins those background tasks so in-flight work drains
    /// before the process exits. Idempotent across clones: the first call drains
    /// the task set; later calls find it empty and simply re-assert the cancel.
    pub async fn shutdown(self) {
        self.shutdown.cancel();
        let tasks =
            std::mem::take(&mut *self.tasks.lock().expect("controller tasks lock poisoned"));
        for task in tasks {
            let _ = task.await;
        }
    }
}

/// Spawn the actor + reaper tick and return the in-process [`ControllerHandle`].
/// No socket (single-node embed). Must be called from within a Tokio runtime.
/// Sync because nothing here awaits (Model A: the caller owns the runtime).
pub fn embed(config: Config) -> ControllerHandle {
    let (cmd_tx, cmd_rx) = mpsc::channel(config.command_buffer);
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
    );
    let actor_token = shutdown.clone();
    let actor_task = tokio::spawn(async move {
        tokio::select! {
            _ = actor.run() => {}
            _ = actor_token.cancelled() => {}
        }
    });

    // The reaper is just a timer feeding `Command::Tick` into the one actor.
    let tick_cmd = cmd_tx.clone();
    let interval = config.tick_interval;
    let reaper_token = shutdown.clone();
    let reaper_task = tokio::spawn(async move {
        let mut timer = tokio::time::interval(interval);
        loop {
            tokio::select! {
                _ = timer.tick() => {
                    if tick_cmd.send(Command::Tick).await.is_err() {
                        break; // actor stopped
                    }
                }
                _ = reaper_token.cancelled() => break,
            }
        }
    });

    ControllerHandle {
        cmd: cmd_tx,
        worker_rx,
        gateway_rx,
        shutdown,
        tasks: Arc::new(Mutex::new(vec![actor_task, reaper_task])),
    }
}

/// Run the controller as a daemon (Seam 1): embed the actor and serve the
/// `Control` RPC over tarpc (tcp + unix), then return the [`ControllerHandle`].
/// The accept loop runs in the background, owned by the handle, and stops on
/// [`ControllerHandle::shutdown`]. Async because binding the listener awaits — a
/// bind failure (e.g. address in use) surfaces here, before the handle returns.
pub async fn run(config: Config) -> Result<ControllerHandle> {
    let handle = embed(config.clone());
    let serve = service::serve(&config.listen_addr, handle.clone(), handle.shutdown.clone())
        .await
        .with_context(|| format!("bind controller Control endpoint on {}", config.listen_addr))?;
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
    fn parse_empty_is_default() {
        let cfg = Config::parse("").expect("empty config parses to defaults");
        let d = Config::default();
        assert_eq!(cfg.listen_addr, d.listen_addr);
        assert_eq!(cfg.heartbeat_timeout, d.heartbeat_timeout);
        assert_eq!(cfg.tick_interval, d.tick_interval);
        assert_eq!(cfg.command_buffer, d.command_buffer);
    }

    #[test]
    fn parse_overrides_fields() {
        let cfg = Config::parse(
            r#"
            listen_addr = "127.0.0.1:9000"
            heartbeat_timeout_secs = 12
            tick_interval_secs = 3
            command_buffer = 64
            "#,
        )
        .expect("valid config parses");
        assert_eq!(cfg.listen_addr, "127.0.0.1:9000");
        assert_eq!(cfg.heartbeat_timeout, Duration::from_secs(12));
        assert_eq!(cfg.tick_interval, Duration::from_secs(3));
        assert_eq!(cfg.command_buffer, 64);
    }

    #[test]
    fn parse_rejects_unknown_field() {
        assert!(Config::parse("bogus = 1").is_err(), "deny_unknown_fields");
    }

    #[test]
    fn parse_rejects_zero_tick_interval() {
        assert!(Config::parse("tick_interval_secs = 0").is_err());
    }

    #[tokio::test]
    async fn embed_serves_in_proc_then_shuts_down() {
        let handle = embed(Config::default());
        // An in-proc call returns ⇒ the actor is live behind the handle.
        let _id = handle
            .register_gateway(GatewayInfo {
                addr: "127.0.0.1:0".to_string(),
            })
            .await;
        // shutdown drains actor + reaper without hanging.
        handle.shutdown().await;
    }

    #[tokio::test]
    async fn run_binds_ephemeral_then_shuts_down() {
        let config = Config {
            listen_addr: "127.0.0.1:0".to_string(),
            ..Config::default()
        };
        let handle = run(config).await.expect("controller daemon starts");
        // shutdown cancels + joins the serve loop, actor, and reaper cleanly.
        handle.shutdown().await;
    }
}
