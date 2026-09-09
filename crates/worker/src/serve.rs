mod banner;
use banner::StartupBanner;

use anyhow::{Context, Result, anyhow, bail};
use controller_api::{ControlClient, Role, WorkerInfo};
use ids::WorkerId;
use std::path::Path;

use crate::backend::flavor::Flavor;
use crate::backend::{EngineCapabilities, EngineOptions};
use crate::backend::{GroupEngine, ModelEngines};
use crate::config;
use crate::executor::ExecutorServer;
use crate::link::client;
use crate::link::control::{self, ControlLink};
use crate::link::{gateway, partner, topology};
use crate::translate;
use crate::weights;

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

struct LoadedModelEngines {
    model: String,
    caps: EngineCapabilities,
    full_identity: crate::executor::ModelIdentity,
    encode_identity: crate::executor::ModelIdentity,
    kv_handle: Option<engine::KvHandle>,
    engines: ModelEngines,
    metadata: runtime::model::ModelMetadata,
}

struct LoadedPartnerMetadata {
    full_identity: crate::executor::ModelIdentity,
    encode_identity: crate::executor::ModelIdentity,
    kv_handle: Option<engine::KvHandle>,
    page_size: u32,
    supports_media_encode: bool,
    hidden_size: u32,
}

fn load_model_engines(
    user_cfg: &config::Config,
    component: crate::executor::ModelComponent,
) -> Result<LoadedModelEngines> {
    let engine_cache_dir = crate::disk::engine_cache_dir();

    let (engine_groups, snapshot_dir, metadata) = {
        let m = &user_cfg.model;
        let flavor = crate::backend::flavor::resolve(m.engine.kind, &m.name)?;

        let world_size = m.engine.device.len();
        let tp_degree = if m.engine.tensor_parallel_size == 0 {
            world_size
        } else {
            m.engine.tensor_parallel_size as usize
        };
        let topology = crate::backend::calculate_topology(world_size, tp_degree)
            .with_context(|| format!("model {:?} topology", m.name))?;

        #[allow(unreachable_patterns)]
        if tp_degree > 1 {
            match flavor {
                #[cfg(feature = "cuda")]
                Flavor::Cuda => {}
                _ => anyhow::bail!(
                    "model {:?}: tensor_parallel_size={tp_degree} is only \
                     supported for cuda_native",
                    m.name,
                ),
            }
        }

        let mut embedded_base_opts = crate::backend::build_options(m, flavor)?;
        apply_embedded_verbose(&mut embedded_base_opts, user_cfg.server.verbose);
        let resolved_model = weights::resolve(
            &m.model,
            weights::Want {
                backend: Some(flavor.as_str()),
                sku: m.sku.as_deref(),
            },
        )
        .with_context(|| format!("resolving the model for {:?}", m.name))?;
        let lifted = resolved_model
            .metadata()
            .with_context(|| format!("reading the model metadata for {:?}", m.name))?;
        let snapshot_dir = resolved_model.path().to_path_buf();
        let mut group_engines: Vec<GroupEngine> = Vec::with_capacity(topology.len());
        for (group_idx, group) in topology.iter().enumerate() {
            group_engines.push(create_engine_group(
                m,
                group_idx,
                group,
                flavor,
                &embedded_base_opts,
                &snapshot_dir,
                &engine_cache_dir,
                tp_degree,
                component,
                u8::try_from(user_cfg.runtime.frame_dispatch_depth).unwrap_or(u8::MAX),
            )?);
        }
        (
            ModelEngines {
                groups: group_engines,
            },
            snapshot_dir,
            lifted,
        )
    };

    let caps = engine_groups
        .groups
        .first()
        .map(|group| group.caps.clone())
        .context("no engine capabilities available for control-plane registration")?;
    let kv_handle = engine_groups
        .groups
        .first()
        .and_then(|group| group.backend.export_kv_handle());
    let artifact_digest = if user_cfg.cluster.controller.is_some() || user_cfg.offload.enabled {
        weights::model_artifact_digest(&snapshot_dir)?
    } else {
        *blake3::hash(user_cfg.model.model.as_bytes()).as_bytes()
    };
    Ok(LoadedModelEngines {
        metadata,
        model: user_cfg.model.name.clone(),
        full_identity: weights::model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            crate::executor::ModelComponent::Full,
        )?,
        encode_identity: weights::model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            crate::executor::ModelComponent::Encode,
        )?,
        caps,
        kv_handle,
        engines: engine_groups,
    })
}

async fn boot_engine(
    user_cfg: &config::Config,
) -> Result<(
    String,
    LoadedPartnerMetadata,
    runtime::bootstrap::BootstrapHandle,
)> {
    let LoadedModelEngines {
        model,
        caps,
        full_identity,
        encode_identity,
        kv_handle,
        engines,
        metadata,
    } = load_model_engines(user_cfg, crate::executor::ModelComponent::Full)?;
    let metadata_hidden_size: u32 = serde_json::from_slice::<serde_json::Value>(&metadata.config)
        .ok()
        .and_then(|config| config.get("hidden_size")?.as_u64())
        .and_then(|size| u32::try_from(size).ok())
        .unwrap_or(0);

    let boot_cfg = translate::build(user_cfg, engines, metadata)
        .context("translating to bootstrap::Config")?;

    let boot = runtime::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("runtime::bootstrap::bootstrap: {e}"))?;
    let page_size = caps.pools.kv_page_size;
    let supports_media_encode = caps.media_encode;
    let hidden_size = metadata_hidden_size;
    Ok((
        model,
        LoadedPartnerMetadata {
            full_identity,
            encode_identity,
            kv_handle,
            page_size,
            supports_media_encode,
            hidden_size,
        },
        boot,
    ))
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
    let loaded = load_model_engines(user_cfg, component)?;
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
    let (model, partner_metadata, runtime) = boot_engine(&user_cfg).await?;
    let partner_bootstrap = build_partner_bootstrap(&user_cfg, partner_metadata, runtime.model_idx);
    let (edge_server, control_tasks, control_plane, partners, url) =
        assemble_control_and_edge(coordinator, &user_cfg, model, partner_bootstrap).await?;
    log_serving(&user_cfg, &url);
    Ok(RuntimeHandle {
        url,
        edge_server,
        control_tasks,
        partners,
        control_plane,
        runtime: Some(runtime),
    })
}

pub async fn start_runtime_embedded<C: ControlLink>(
    user_cfg: config::Config,
    control: C,
    gateways: Vec<String>,
    client_edge: Option<String>,
) -> Result<RuntimeHandle> {
    let (model, partner_metadata, runtime) = boot_engine(&user_cfg).await?;
    let partner_bootstrap = build_partner_bootstrap(&user_cfg, partner_metadata, runtime.model_idx);
    let addr = topology::addr_from_host_port(&user_cfg.server.host, user_cfg.server.port);
    let (edge_server, control_tasks, worker_id, partners) = assemble_distributed(
        control,
        &gateways,
        Role::Decode,
        model,
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
        runtime: Some(runtime),
    })
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
        request_timeout_secs: user_cfg.runtime.request_timeout.as_secs(),
        max_outstanding: user_cfg.offload.max_outstanding_per_partner,
    })
}

fn log_serving(cfg: &config::Config, url: &str) {
    use std::io::Write;

    if cfg.server.verbose {
        eprintln!("{}", StartupBanner::from_config(cfg).render(url));
    }
    println!("{}", banner::ready_line(url));
    let _ = std::io::stdout().flush();
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

#[allow(
    clippy::too_many_arguments,
    reason = "independent inputs to one engine launch; a struct here \
              would be a parameter list with a name"
)]
#[cfg_attr(
    not(feature = "cuda"),
    allow(
        unused_variables,
        unreachable_code,
        reason = "with no `engine-*` feature `EngineOptions` is uninhabited, so \
                  every path that takes one diverges"
    )
)]
fn create_engine_group(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &EngineOptions,
    snapshot_dir: &Path,
    cache_dir: &Path,
    tp_degree: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
) -> Result<GroupEngine> {
    #[cfg(feature = "cuda")]
    {
        if flavor == Flavor::Cuda && tp_degree > 1 {
            let rank_opts = cuda_rank_options(m, group_idx, group, base_opts)?;
            return crate::backend::create_engine_backend_group(
                &rank_opts,
                snapshot_dir,
                cache_dir,
                m.adapter_mount().as_deref(),
                group_idx,
                component,
                frames_in_flight,
                &m.adapters,
                m.residency(),
                m.patch_ceilings(),
                m.voxel_ceilings(),
                m.sku.as_deref(),
            )
            .with_context(|| {
                format!(
                    "creating cuda TP engine group for model {:?} group {group_idx}",
                    m.name,
                )
            });
        }
    }

    #[cfg(not(feature = "cuda"))]
    let _ = (flavor, tp_degree);

    let first_engine_idx = group.first().copied().ok_or_else(|| {
        anyhow!(
            "model {:?}: group {group_idx} is empty; topology calculation produced no ranks",
            m.name,
        )
    })?;
    let device = group_engine(m, group_idx, first_engine_idx)?;
    let opts = embedded_opts_for_device(base_opts, device);

    crate::backend::create_engine_backend(
        &opts,
        snapshot_dir,
        cache_dir,
        m.adapter_mount().as_deref(),
        group_idx,
        component,
        frames_in_flight,
        &m.adapters,
        m.residency(),
        m.patch_ceilings(),
        m.voxel_ceilings(),
        m.sku.as_deref(),
    )
    .with_context(|| format!("creating engine for model {:?} group {group_idx}", m.name,))
}

fn embedded_opts_for_device(base_opts: &EngineOptions, device: String) -> EngineOptions {
    #[cfg(not(feature = "cuda"))]
    let _ = &device;

    #[allow(unreachable_patterns)]
    match base_opts {
        #[cfg(feature = "cuda")]
        EngineOptions::CudaNative(opts) => {
            let mut opts = opts.clone();
            opts.device = device;
            EngineOptions::CudaNative(opts)
        }
        other => other.clone(),
    }
}

fn apply_embedded_verbose(options: &mut EngineOptions, verbose: bool) {
    #[cfg(feature = "cuda")]
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    if let EngineOptions::CudaNative(opts) = options {
        opts.verbose = verbose;
    }

    #[cfg(not(feature = "cuda"))]
    let _ = (options, verbose);
}

#[cfg(feature = "cuda")]
fn cuda_rank_options(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    base_opts: &EngineOptions,
) -> Result<Vec<EngineOptions>> {
    let mut rank_opts = Vec::with_capacity(group.len());
    for &rank_engine_idx in group {
        let rank_engine = group_engine(m, group_idx, rank_engine_idx)?;
        #[allow(
            unreachable_patterns,
            reason = "`EngineOptions` has one variant in a CUDA-only build"
        )]
        match base_opts {
            EngineOptions::CudaNative(opts) => {
                let mut o = opts.clone();
                o.device = rank_engine;
                rank_opts.push(EngineOptions::CudaNative(o));
            }
            _ => unreachable!("flavor checked before building cuda rank options"),
        }
    }
    Ok(rank_opts)
}

fn group_engine(m: &config::ModelConfig, group_idx: usize, engine_idx: usize) -> Result<String> {
    m.engine
        .device
        .get(engine_idx)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "model {:?}: group {group_idx} references device index {} but only {} devices configured",
                m.name,
                engine_idx,
                m.engine.device.len(),
            )
        })
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
