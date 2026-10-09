mod banner;
use banner::StartupBanner;

use anyhow::{Context, Result, bail};
use controller_api::{ControlClient, Role, WorkerInfo};
use ids::WorkerId;

use crate::boot::{self, Booted, LoadedPartnerMetadata, Summary};
use crate::config;
use crate::executor::ExecutorServer;
use crate::link::client;
use crate::link::control::{self, ControlLink};
use crate::link::{gateway, partner, topology};

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
    partners: Option<std::sync::Arc<tokio::sync::Mutex<partner::PartnerLinkManager>>>,
    control_plane: ClusterControl,
    pub url: String,
    pub summary: Summary,
}

enum ClusterControl {
    None,
    Distributed {
        _client: ControlClient,
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
        if let Some(partners) = self.partners.take() {
            partners.lock().await.shutdown().await;
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
    inner: WorkerKind,
}

enum WorkerKind {
    Decode(RuntimeHandle),
    Executor(ExecutorHandle),
}

struct ExecutorHandle {
    server: ExecutorServer,
    control_tasks: Vec<tokio::task::JoinHandle<()>>,
    _client: ControlClient,
    worker_id: WorkerId,
}

impl WorkerHandle {
    pub fn url(&self) -> &str {
        match &self.inner {
            WorkerKind::Decode(engine) => &engine.url,
            WorkerKind::Executor(executor) => executor.server.endpoint(),
        }
    }

    /// What booted, for a decode worker.
    pub fn summary(&self) -> Option<&Summary> {
        match &self.inner {
            WorkerKind::Decode(engine) => Some(&engine.summary),
            WorkerKind::Executor(_) => None,
        }
    }

    pub async fn shutdown(self) {
        match self.inner {
            WorkerKind::Decode(engine) => engine.shutdown().await,
            WorkerKind::Executor(executor) => executor.shutdown().await,
        }
    }
}

impl ExecutorHandle {
    async fn shutdown(self) {
        for task in &self.control_tasks {
            task.abort();
        }
        for task in self.control_tasks {
            let _ = task.await;
        }
        tracing::info!(worker = %self.worker_id, "leaving executor control plane");
        self.server.shutdown().await;
    }
}

pub async fn run(cfg: config::Config) -> Result<WorkerHandle> {
    let mode = match (&cfg.cluster.controller, cfg.cluster.role) {
        (Some(controller), Some(role)) => {
            TopologyMode::distributed(role, controller.clone(), cfg.cluster.gateways.clone())?
        }
        (Some(_), None) => bail!("[cluster] role is required when controller is set"),
        (None, _) => TopologyMode::SingleNode,
    };
    let control_addr = topology::addr_from_host_port(&cfg.server.host, cfg.server.port);
    let coordinator = topology::connect(&mode, control_addr)?;
    if matches!(coordinator.role(), Some(Role::Prefill | Role::Encode)) {
        let executor = boot_executor(&cfg, &coordinator).await?;
        Ok(WorkerHandle {
            inner: WorkerKind::Executor(executor),
        })
    } else {
        let engine = start_runtime(cfg, coordinator).await?;
        Ok(WorkerHandle {
            inner: WorkerKind::Decode(engine),
        })
    }
}

pub async fn run_with<C: ControlLink>(
    cfg: config::Config,
    control: C,
    gateways: Vec<String>,
    client_edge: Option<String>,
) -> Result<WorkerHandle> {
    let engine = start_runtime_embedded(cfg, control, gateways, client_edge).await?;
    Ok(WorkerHandle {
        inner: WorkerKind::Decode(engine),
    })
}

pub fn build_runtime(user_cfg: &config::Config) -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(user_cfg.server.worker_threads)
        .enable_all()
        .build()
        .context("building tokio runtime")
}

async fn boot_executor(
    user_cfg: &config::Config,
    coordinator: &Coordinator,
) -> Result<ExecutorHandle> {
    let role = coordinator
        .role()
        .context("executor boot requires a distributed role")?;
    anyhow::ensure!(
        matches!(role, Role::Prefill | Role::Encode),
        "executor boot requires prefill or encode role"
    );
    let controller = coordinator
        .controller_addr()
        .context("executor boot requires a controller")?;
    let component = if role == Role::Encode {
        crate::executor::ModelComponent::Encode
    } else {
        crate::executor::ModelComponent::Full
    };
    let loaded = boot::load_model_engines(user_cfg, component, None)?;
    let model_identity = if role == Role::Encode {
        loaded.encode_identity.clone()
    } else {
        loaded.full_identity.clone()
    };
    let server = ExecutorServer::bind_with_transfer(
        &coordinator.control_addr,
        loaded.engines,
        model_identity,
        user_cfg.executor.max_clients,
        user_cfg.offload.transfer,
    )
    .await?;
    let client = match control::dial_controller(controller).await {
        Ok(client) => client,
        Err(error) => {
            server.shutdown().await;
            return Err(error).with_context(|| format!("dialing controller at {controller}"));
        }
    };
    let worker_id = match ControlLink::register_worker(
        &client,
        WorkerInfo {
            role,
            model: loaded.model,
            addr: server.endpoint().to_string(),
        },
    )
    .await
    {
        Ok(worker_id) => worker_id,
        Err(error) => {
            server.shutdown().await;
            return Err(error).context("registering executor with controller");
        }
    };
    let control_tasks = control::spawn_executor_control_tasks(
        client.clone(),
        worker_id,
        server.stats(),
        server.total_pages(),
    );
    tracing::info!(
        worker = %worker_id,
        %role,
        endpoint = server.endpoint(),
        "executor ready"
    );
    Ok(ExecutorHandle {
        server,
        control_tasks,
        _client: client,
        worker_id,
    })
}

pub async fn start_runtime(
    user_cfg: config::Config,
    coordinator: Coordinator,
) -> Result<RuntimeHandle> {
    let (booted, partner_bootstrap) = boot_decode(&user_cfg).await?;
    let (edge_server, control_tasks, control_plane, partners, url) =
        assemble_control_and_edge(coordinator, &user_cfg, booted.model, partner_bootstrap).await?;
    log_serving(&user_cfg, &url);
    Ok(RuntimeHandle {
        url,
        edge_server,
        control_tasks,
        partners,
        control_plane,
        runtime: Some(booted.runtime),
        summary: booted.summary,
    })
}

pub async fn start_runtime_embedded<C: ControlLink>(
    user_cfg: config::Config,
    control: C,
    gateways: Vec<String>,
    client_edge: Option<String>,
) -> Result<RuntimeHandle> {
    let (booted, partner_bootstrap) = boot_decode(&user_cfg).await?;
    let addr = topology::addr_from_host_port(&user_cfg.server.host, user_cfg.server.port);
    let (edge_server, control_tasks, worker_id, partners) = assemble_distributed(
        control,
        &gateways,
        Role::Decode,
        booted.model,
        addr,
        partner_bootstrap,
    )
    .await?;
    let url = client_edge.unwrap_or_else(|| edge_server.url());
    log_serving(&user_cfg, &url);
    Ok(RuntimeHandle {
        url,
        edge_server,
        control_tasks,
        partners,
        control_plane: ClusterControl::Embedded { worker_id },
        runtime: Some(booted.runtime),
        summary: booted.summary,
    })
}

/// Boots a decode worker on its configured engine, with offload set up.
async fn boot_decode(
    user_cfg: &config::Config,
) -> Result<(Booted, Option<partner::PartnerBootstrap>)> {
    let mut booted = boot::boot(
        user_cfg,
        boot::Engine::Configured,
        crate::translate::builtins(),
    )
    .await?;
    let metadata = booted
        .partner
        .take()
        .context("a configured engine reports its partner metadata")?;
    let partner = build_partner_bootstrap(user_cfg, metadata, booted.runtime.model_idx);
    Ok((booted, partner))
}

fn build_partner_bootstrap(
    user_cfg: &config::Config,
    metadata: LoadedPartnerMetadata,
    model_idx: usize,
) -> Option<partner::PartnerBootstrap> {
    runtime::offload::configure(
        user_cfg.offload.enabled,
        user_cfg.offload.prefill_min_suffix_tokens,
    );
    runtime::offload::configure_encode_injection(
        user_cfg.offload.enabled && metadata.supports_media_encode,
        if metadata.supports_media_encode {
            metadata.hidden_size
        } else {
            0
        },
    );
    if !user_cfg.offload.enabled {
        return None;
    }
    let Some(kv_handle) = metadata.kv_handle else {
        tracing::warn!(
            "offload is enabled but the home backend has no KV export layout; using local fallback"
        );
        return None;
    };
    runtime::offload::set_home_kv_handle(kv_handle.clone());
    Some(partner::PartnerBootstrap {
        full_identity: metadata.full_identity,
        encode_identity: metadata.encode_identity,
        kv_layout: kv_handle.layout.clone(),
        home_kv_handle: kv_handle,
        transfer: user_cfg.offload.transfer,
        model_idx,
        page_size: metadata.page_size,
        max_outstanding: user_cfg.offload.max_outstanding_per_partner,
    })
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
    partner_bootstrap: Option<partner::PartnerBootstrap>,
) -> Result<(
    EdgeServer,
    Vec<tokio::task::JoinHandle<()>>,
    ClusterControl,
    Option<std::sync::Arc<tokio::sync::Mutex<partner::PartnerLinkManager>>>,
    String,
)> {
    match coordinator.mode {
        TopologyMode::Distributed {
            role,
            controller,
            gateways,
        } => {
            let client = control::dial_controller(&controller)
                .await
                .with_context(|| format!("dialing controller at {controller}"))?;
            let (edge, control_tasks, worker_id, partners) = assemble_distributed(
                client.clone(),
                &gateways,
                role,
                model,
                coordinator.control_addr.clone(),
                partner_bootstrap,
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
                partners,
                url,
            ))
        }
        TopologyMode::SingleNode => {
            let _ = (model, partner_bootstrap);
            let listen = format!("{}:{}", user_cfg.server.host, user_cfg.server.port);
            let edge = EdgeServer::Standalone(
                client::spawn(&listen)
                    .await
                    .context("starting standalone client server")?,
            );
            let url = edge.url();
            Ok((edge, Vec::new(), ClusterControl::None, None, url))
        }
    }
}

async fn assemble_distributed<C: ControlLink>(
    control: C,
    gateways: &[String],
    role: Role,
    model: String,
    addr: String,
    partner_bootstrap: Option<partner::PartnerBootstrap>,
) -> Result<(
    EdgeServer,
    Vec<tokio::task::JoinHandle<()>>,
    WorkerId,
    Option<std::sync::Arc<tokio::sync::Mutex<partner::PartnerLinkManager>>>,
)> {
    let info = WorkerInfo { role, model, addr };
    let worker_id = ControlLink::register_worker(&control, info)
        .await
        .context("registering worker with controller")?;

    let mut manager = gateway::GatewayLinkManager::new(worker_id, gateways.to_vec());
    manager
        .dial_pinned()
        .await
        .context("dialing pinned gateways")?;
    let dialed = manager.addrs();
    let partners = partner_bootstrap
        .map(|config| partner::PartnerLinkManager::new(worker_id, config))
        .transpose()?
        .map(|manager| std::sync::Arc::new(tokio::sync::Mutex::new(manager)));
    let control_tasks = control::spawn_control_tasks(control, worker_id, manager, partners.clone());

    Ok((
        EdgeServer::GatewayLinks(dialed),
        control_tasks,
        worker_id,
        partners,
    ))
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
