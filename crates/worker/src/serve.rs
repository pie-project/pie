mod banner;
use banner::StartupBanner;

use anyhow::{Context, Result, bail};
use controller_api::{Role, WorkerInfo};
use ids::WorkerId;
use std::path::Path;

use crate::config;
use crate::embedded::{Embedded, Engine, Summary};
use crate::link::client;
use crate::link::control::{self, ControlLink};
use crate::link::{gateway, kv, peer, topology};

pub use crate::link::topology::{Coordinator, TopologyMode, connect};

enum EdgeServer {
    Standalone(client::ClientServerHandle),
    GatewayLinks(Vec<String>),
}

impl EdgeServer {
    fn url(&self) -> String {
        match self {
            EdgeServer::Standalone(h) => h.bound.clone(),
            EdgeServer::GatewayLinks(addrs) => {
                if addrs.is_empty() {
                    "gateway://<none>".to_string()
                } else {
                    format!("gateway://{}", addrs.join(","))
                }
            }
        }
    }

    fn abort(&self) {
        match self {
            EdgeServer::Standalone(h) => h.task.abort(),
            EdgeServer::GatewayLinks(_) => {}
        }
    }
}

pub struct RuntimeHandle {
    runtime: Option<runtime::bootstrap::BootstrapHandle>,
    edge_server: EdgeServer,
    control_tasks: Vec<tokio::task::JoinHandle<()>>,
    control_plane: ClusterControl,
    pub url: String,
    pub summary: Summary,
}

enum ClusterControl {
    None,
    Distributed {
        _client: control::Controller,
        worker_id: WorkerId,
    },
    Embedded {
        worker_id: WorkerId,
    },
}

impl ClusterControl {
    fn worker_id(&self) -> Option<WorkerId> {
        match self {
            ClusterControl::None => None,
            ClusterControl::Distributed { worker_id, .. }
            | ClusterControl::Embedded { worker_id } => Some(*worker_id),
        }
    }
}

impl RuntimeHandle {
    pub async fn wait_then_shutdown(self) -> Result<()> {
        let shutdown_reason = tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = wait_for_sigterm() => "SIGTERM",
        };
        eprintln!("\nshutting down ({shutdown_reason})...");
        self.shutdown().await;
        Ok(())
    }

    pub async fn shutdown(mut self) {
        self.edge_server.abort();
        for task in &self.control_tasks {
            task.abort();
        }
        for task in self.control_tasks {
            let _ = task.await;
        }
        tracing::info!(worker = ?self.control_plane.worker_id(), "leaving control plane");
        drop(self.control_plane);
        if let Some(runtime) = self.runtime.take()
            && let Err(err) = runtime.shutdown().await
        {
            tracing::error!(?err, "runtime shutdown failed");
        }
    }
}

pub struct WorkerHandle {
    inner: RuntimeHandle,
}

impl WorkerHandle {
    pub fn url(&self) -> &str {
        &self.inner.url
    }

    /// What booted.
    pub fn summary(&self) -> &Summary {
        &self.inner.summary
    }

    /// Takes no new turns, and waits up to `within` for what runs here to
    /// end or be moved elsewhere by the cluster policy.
    pub async fn drain(&self, within: std::time::Duration) {
        crate::link::control::set_draining();
        let deadline = std::time::Instant::now() + within;
        tracing::info!("draining: no new turns; waiting for running ones");
        while !runtime::inferlet::process::list().is_empty() {
            if std::time::Instant::now() >= deadline {
                tracing::warn!(
                    left = runtime::inferlet::process::list().len(),
                    "drained as far as the deadline allows"
                );
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        }
        tracing::info!("drained");
    }

    pub async fn shutdown(self) {
        self.inner.shutdown().await;
    }
}

/// Boots the worker `cfg` describes, keeping its files under `home`.
pub async fn run(cfg: config::Config, home: &Path) -> Result<WorkerHandle> {
    let mode = match (&cfg.cluster.controller, cfg.cluster.role) {
        (Some(controller), Some(role)) => {
            TopologyMode::distributed(role, controller.clone(), cfg.cluster.gateways.clone())?
        }
        (Some(_), None) => bail!("[cluster] role is required when controller is set"),
        (None, _) => TopologyMode::SingleNode,
    };
    let control_addr = topology::addr_from_host_port(&cfg.server.host, cfg.server.port);
    let coordinator = topology::connect(&mode, control_addr)?;
    // A prefill worker runs a runtime like any other: it takes no client
    // turns (the gateway routes to decode workers) and computes peers'
    // passes over the peer link.
    let engine = start_runtime(cfg, home, coordinator).await?;
    Ok(WorkerHandle { inner: engine })
}

pub async fn run_with<C: ControlLink>(
    cfg: config::Config,
    home: &Path,
    control: C,
    gateways: Vec<String>,
    client_edge: Option<String>,
) -> Result<WorkerHandle> {
    let engine = start_runtime_embedded(cfg, home, control, gateways, client_edge).await?;
    Ok(WorkerHandle { inner: engine })
}

pub async fn start_runtime(
    user_cfg: config::Config,
    home: &Path,
    coordinator: Coordinator,
) -> Result<RuntimeHandle> {
    let (model, booted) = boot_decode(&user_cfg, home).await?;
    let (edge_server, control_tasks, control_plane, url) =
        assemble_control_and_edge(coordinator, &user_cfg, model).await?;
    log_serving(&user_cfg, &url);
    Ok(RuntimeHandle {
        url,
        edge_server,
        control_tasks,
        control_plane,
        runtime: Some(booted.runtime),
        summary: booted.summary,
    })
}

pub async fn start_runtime_embedded<C: ControlLink>(
    user_cfg: config::Config,
    home: &Path,
    control: C,
    gateways: Vec<String>,
    client_edge: Option<String>,
) -> Result<RuntimeHandle> {
    let (model, booted) = boot_decode(&user_cfg, home).await?;
    let addr = topology::addr_from_host_port(&user_cfg.server.host, user_cfg.server.port);
    let (edge_server, control_tasks, worker_id) = assemble_distributed(
        control,
        &gateways,
        Role::Decode,
        model,
        engine_of(&user_cfg),
        addr,
    )
    .await?;
    let url = client_edge.unwrap_or_else(|| edge_server.url());
    log_serving(&user_cfg, &url);
    Ok(RuntimeHandle {
        url,
        edge_server,
        control_tasks,
        control_plane: ClusterControl::Embedded { worker_id },
        runtime: Some(booted.runtime),
        summary: booted.summary,
    })
}

/// Boots a worker on its configured engine, with offload set up.
async fn boot_decode(user_cfg: &config::Config, home: &Path) -> Result<(String, Embedded)> {
    let mut loaded = Embedded::load(
        user_cfg,
        home,
        Engine::Configured,
        crate::translate::builtins(),
    )?;
    let model = std::mem::take(&mut loaded.model);
    let kv_handle = loaded.kv_handle.take();
    let booted = loaded.start().await?;
    runtime::offload::configure(user_cfg.offload.prefill_min_suffix_tokens);
    if let Some(handle) = kv_handle {
        runtime::offload::set_home_kv_handle(handle);
    }
    start_peers(user_cfg).await?;
    Ok((model, booted))
}

/// Serves other workers' passes, and dials the peers this worker's passes
/// may be prefilled on, retrying until one takes them.
async fn start_peers(user_cfg: &config::Config) -> Result<()> {
    let engine = engine_of(user_cfg);
    // Its kv agent first: the hello says whether peers can read it in place.
    if let Some(handle) = runtime::offload::home_kv_handle() {
        let name = match &user_cfg.offload.listen {
            Some(listen) => peer::bulk_addr(listen)?,
            None => format!("pie-worker-{}", std::process::id()),
        };
        kv::open(&name, handle)?;
    }
    let hello = peer::hello(WorkerId(0), &engine);
    peer::set_hello(hello.clone());
    if let Some(listen) = &user_cfg.offload.listen {
        peer::serve(listen, hello.clone()).await?;
        peer::set_listen(peer::bulk_addr(listen)?);
        peer::set_link(listen.clone());
    }
    peer::clients();
    if !user_cfg.offload.pools.is_empty() {
        tokio::spawn(peer::push_to_pools(user_cfg.offload.pools.clone()));
    }
    let peers = user_cfg.offload.peers.clone();
    let listen = peer::listen();
    if !peers.is_empty() {
        tokio::spawn(async move {
            loop {
                for addr in &peers {
                    match peer::dial(addr, &hello, listen.clone()).await {
                        Ok(()) => return,
                        Err(error) => tracing::debug!(peer = %addr, %error, "peer not taken"),
                    }
                }
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            }
        });
    }
    Ok(())
}

/// The verbose banner goes to stderr; the ready line is the CLI's to print,
/// so a process embedding the server (the Node addon, the Python wheel)
/// keeps its stdout.
fn log_serving(cfg: &config::Config, url: &str) {
    if cfg.server.verbose {
        eprintln!("{}", StartupBanner::from_config(cfg).render(url));
    }
}

async fn assemble_control_and_edge(
    coordinator: Coordinator,
    user_cfg: &config::Config,
    model: String,
) -> Result<(
    EdgeServer,
    Vec<tokio::task::JoinHandle<()>>,
    ClusterControl,
    String,
)> {
    match coordinator.mode {
        TopologyMode::Distributed {
            role,
            controller,
            gateways,
        } => {
            let client = control::Controller::dial(&controller)
                .await
                .with_context(|| format!("dialing controller at {controller}"))?;
            let (edge, control_tasks, worker_id) = assemble_distributed(
                client.clone(),
                &gateways,
                role,
                model,
                engine_of(user_cfg),
                coordinator.control_addr.clone(),
            )
            .await?;
            let url = edge.url();
            Ok((
                edge,
                control_tasks,
                ClusterControl::Distributed {
                    _client: client,
                    worker_id,
                },
                url,
            ))
        }
        TopologyMode::SingleNode => {
            let _ = model;
            let listen = format!("{}:{}", user_cfg.server.host, user_cfg.server.port);
            let edge = EdgeServer::Standalone(
                client::spawn(&listen)
                    .await
                    .context("starting standalone client server")?,
            );
            let url = edge.url();
            Ok((edge, Vec::new(), ClusterControl::None, url))
        }
    }
}

/// The engine kind a worker registers its shape with, as its config spells it.
fn engine_of(cfg: &config::Config) -> String {
    cfg.model.engine.kind.as_str().to_string()
}

async fn assemble_distributed<C: ControlLink>(
    control: C,
    gateways: &[String],
    role: Role,
    model: String,
    engine: String,
    addr: String,
) -> Result<(EdgeServer, Vec<tokio::task::JoinHandle<()>>, WorkerId)> {
    let info = WorkerInfo {
        role,
        model,
        engine,
        addr,
        // Launched by the controller's actuator, the token it was given.
        launch: std::env::var(controller_api::LAUNCH_ENV).ok(),
        prior: None,
    };
    let worker_id = ControlLink::register_worker(&control, info.clone())
        .await
        .context("registering worker with controller")?;

    let mut manager = gateway::GatewayLinkManager::new(worker_id, gateways.to_vec());
    manager
        .dial_pinned()
        .await
        .context("dialing pinned gateways")?;
    let dialed = manager.addrs();
    let control_tasks = control::spawn_control_tasks(control, worker_id, info, manager);

    Ok((EdgeServer::GatewayLinks(dialed), control_tasks, worker_id))
}

async fn wait_for_sigterm() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut stream = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("could not install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
                return;
            }
        };
        stream.recv().await;
    }

    #[cfg(windows)]
    {
        std::future::pending::<()>().await;
    }
}
