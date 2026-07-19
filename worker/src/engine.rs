//! `pie serve` core: boot drivers, wire RPC, hand off to the runtime,
//! and surface an [`EngineHandle`] the caller drives.
//!
//! Wires the standalone's pieces in dependency order:
//!   1. Translate user TOML to per-driver options.
//!   2. For the `[model]`, partition devices into DP groups and create
//!      native drivers directly, collecting their capabilities.
//!   3. Translate the resulting native drivers → [`pie_engine::bootstrap::Config`]
//!      and call [`pie_engine::bootstrap::bootstrap`]. The runtime now owns
//!      the runtime services + scheduler; the worker dials into the
//!      gateway and serves `pie_worker_rpc::WorkerControl`.
//!   4. Caller decides what to do with the [`EngineHandle`]:
//!        * `pie serve`: [`EngineHandle::wait_then_shutdown`] blocks
//!          on SIGINT/SIGTERM/watchdog and tears down.
//!        * `pie serve --monitor`: TUI runs concurrently and calls
//!          [`EngineHandle::shutdown`] when the user quits.

use anyhow::{Context, Result, anyhow, bail};
use pie_controller_rpc::{ControlClient, Role, WorkerInfo};
use pie_ids::WorkerId;
use std::path::Path;

use crate::config;
use crate::driver_ffi::Flavor;
use crate::embedded_driver::{DriverCapabilities, DriverOptions};
use crate::executor::ExecutorServer;
use crate::link::control::{self, ControlLink};
use crate::link::{gateway, partner, topology};
use crate::preflight::{self, ResolvedFlavor};
use crate::translate::{self, GroupDriver, ModelDrivers};
use crate::{client_server, lifecycle, weights};

pub use crate::link::topology::{Coordinator, TopologyMode, connect};
pub use crate::preflight::calculate_topology;

/// Live engine — drivers, RPC dispatch threads, and enough state to perform an
/// orderly shutdown.
/// Returned from [`start_engine`]; consumed by either
/// [`EngineHandle::wait_then_shutdown`] (the `pie serve` path) or
/// [`EngineHandle::shutdown`] (the `pie serve --monitor` path, where
/// the TUI owns the wait loop).
/// The worker's data-plane edge, selected by topology: a direct WebSocket
/// terminator in the single-node default build (gateway-free local inference),
/// or the dial-in link(s) the worker serves `WorkerControl` over after dialing
/// INTO the gateway(s) (distributed + single-node feature; M3 inversion).
enum EdgeServer {
    Standalone(client_server::ClientServerHandle),
    /// Post-inversion (M3): the worker dials INTO the gateway(s). The live links
    /// are owned by the control-plane watch task, which reconciles them against
    /// the controller-pushed gateway roster (`gateway.md`); this holds only the
    /// addresses dialed at boot, for the advertised URL. Aborting the control
    /// tasks (and dropping the manager) tears the links down.
    GatewayLinks(Vec<String>),
}

impl EdgeServer {
    /// The advertised URL: `ws://…` for the direct client server, or
    /// `gateway://addr[,…]` listing the gateway endpoint(s) the worker dialed
    /// into (the worker is not client-facing in distributed mode).
    fn url(&self) -> String {
        match self {
            EdgeServer::Standalone(h) => h.bound.clone(),
            EdgeServer::GatewayLinks(addrs) => {
                // The worker is not client-facing in distributed mode — the
                // gateway is. Report the gateway endpoint(s) it dialed into.
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
            // Links live in the control-plane watch task; aborting the control
            // tasks (which drops the GatewayLinkManager) tears them down.
            EdgeServer::GatewayLinks(_) => {}
        }
    }
}

pub struct EngineHandle {
    runtime: Option<pie_engine::bootstrap::BootstrapHandle>,
    edge_server: EdgeServer,
    /// Controller heartbeat/report/watch tasks. Empty when there is no control
    /// plane (single-node without the `single-node` feature).
    control_tasks: Vec<tokio::task::JoinHandle<()>>,
    partners: Option<std::sync::Arc<tokio::sync::Mutex<partner::PartnerLinkManager>>>,
    /// Live control-plane state kept alive for the engine's lifetime: the dialed
    /// client (distributed) or the embedded controller handle + in-proc gateway
    /// task (single-node feature). `None` in gateway-free single-node.
    control_plane: ClusterControl,
    /// Client endpoint this worker advertises: `ws://host:port` in single-node
    /// (direct client server, or the in-proc gateway), or `gateway://addr[,…]`
    /// in distributed (the gateway endpoint(s) the worker dialed into — clients
    /// hit the gateway, not the worker).
    pub url: String,
}

/// Live control-plane resources held for the engine's lifetime, by topology.
enum ClusterControl {
    /// No control plane (single-node): the worker terminates clients directly
    /// and never registers.
    None,
    /// Distributed: the dialed control client (its dispatch task) stays alive
    /// until shutdown, when dropping it closes the connection so the controller
    /// ages this worker out of routing.
    Distributed {
        _client: ControlClient,
        worker_id: WorkerId,
    },
    /// In-proc embed (`bin/pie`): the injected control link is owned by the
    /// composition root; the worker holds only its id (the spawned control-loop
    /// tasks keep their own clones of the injected link alive).
    Embedded { worker_id: WorkerId },
}

impl ClusterControl {
    /// The controller-minted worker id, if this worker registered.
    fn worker_id(&self) -> Option<WorkerId> {
        match self {
            ClusterControl::None => None,
            ClusterControl::Distributed { worker_id, .. }
            | ClusterControl::Embedded { worker_id } => Some(*worker_id),
        }
    }
}

impl EngineHandle {
    /// Block on SIGINT / SIGTERM, then run the
    /// shutdown sequence. The original `run_with_config` flow.
    pub async fn wait_then_shutdown(self) -> Result<()> {
        let shutdown_reason = tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = lifecycle::wait_for_sigterm() => "SIGTERM",
        };
        eprintln!("\nshutting down ({shutdown_reason})...");
        self.shutdown().await;
        Ok(())
    }

    /// Tear down the engine without waiting for a signal. Used by the
    /// monitor TUI, which owns its own input loop and decides when to
    /// quit.
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
        // Stop the in-proc gateway (single-node) and drop the control-plane
        // resources so the dialed control connection is closed (distributed) or
        // the embedded controller handle is released (single-node). The
        // controller then ages this worker out of routing on the next missed
        // report.
        tracing::info!(worker = ?self.control_plane.worker_id(), "leaving control plane");
        drop(self.control_plane);
        if let Some(runtime) = self.runtime.take() {
            if let Err(err) = runtime.shutdown().await {
                tracing::error!(?err, "runtime shutdown failed");
            }
        }
    }
}

/// A running worker: the engine plus an async drain-and-stop. Returned by
/// [`run`] (daemon) and [`run_with`] (in-proc embed). The bin owns the runtime
/// (Model A) and drives [`shutdown`](WorkerHandle::shutdown) on signal.
pub struct WorkerHandle {
    inner: WorkerKind,
}

enum WorkerKind {
    Decode(EngineHandle),
    Executor(ExecutorHandle),
}

struct ExecutorHandle {
    server: ExecutorServer,
    control_tasks: Vec<tokio::task::JoinHandle<()>>,
    _client: ControlClient,
    worker_id: WorkerId,
}

impl WorkerHandle {
    /// The client endpoint this worker advertises (`ws://…` in single-node, or
    /// the `gateway://…` endpoint(s) it dialed into in distributed mode).
    pub fn url(&self) -> &str {
        match &self.inner {
            WorkerKind::Decode(engine) => &engine.url,
            WorkerKind::Executor(executor) => executor.server.endpoint(),
        }
    }

    /// Drain in-flight work and stop the engine (runtime, control loops, edge).
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

/// Daemon entry (`bin/worker`): derive the topology from `cfg.cluster`, boot the
/// engine, dial the cluster (distributed) or terminate clients directly
/// (single-node), and return a [`WorkerHandle`]. Async (Model A) — the bin owns
/// the runtime and drives `shutdown` on signal via `bootstrap`.
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
        let engine = start_engine(cfg, coordinator).await?;
        Ok(WorkerHandle {
            inner: WorkerKind::Decode(engine),
        })
    }
}

/// In-proc embed entry for the composition root (`bin/pie`): run the worker
/// against an **injected** [`ControlLink`] (the root's `EmbeddedControl`) plus
/// the in-proc gateway address(es), instead of dialing a real controller — the
/// counterpart of the gateway's `run_with`.
pub async fn run_with<C: ControlLink>(
    cfg: config::Config,
    control: C,
    gateways: Vec<String>,
) -> Result<WorkerHandle> {
    let engine = start_engine_embedded(cfg, control, gateways).await?;
    Ok(WorkerHandle {
        inner: WorkerKind::Decode(engine),
    })
}

struct StartupBanner {
    model: String,
    driver: String,
    device: String,
}

impl StartupBanner {
    fn from_config(cfg: &config::Config) -> Self {
        let m = &cfg.model;
        let model = format!("{} ({})", m.name, m.hf_repo);
        let driver = m.driver.kind.as_str().to_string();
        let device = {
            let device = m.driver.device.join(", ");
            if device.is_empty() {
                "-".to_string()
            } else {
                device
            }
        };

        Self {
            model,
            driver,
            device,
        }
    }

    fn render(&self, url: &str) -> String {
        let host = url
            .strip_prefix("ws://")
            .or_else(|| url.strip_prefix("edge://"))
            .unwrap_or(url);
        let rows = [
            ("Host", host),
            ("Model", self.model.as_str()),
            ("Driver", self.driver.as_str()),
            ("Device", self.device.as_str()),
        ];
        let label_width = 12;
        let header = "─ Pie Engine ";
        let content_width = rows
            .iter()
            .map(|(_, value)| label_width + 1 + value.len())
            .max()
            .unwrap_or(0)
            .max(header.len() - 2);
        let inner_width = content_width + 2;
        let mut out = String::new();

        out.push_str(&format!(
            "╭{}{}╮\n",
            header,
            "─".repeat(inner_width - header.len())
        ));
        for (label, value) in rows {
            let content = format!("{label:<label_width$} {value}");
            out.push_str(&format!(
                "│ {:<content_width$} │\n",
                content,
                content_width = content_width
            ));
        }
        out.push_str(&format!("╰{}╯\n\n", "─".repeat(inner_width)));
        out.push_str(&format!("✓ Server ready at {url}"));
        out
    }
}

/// Build the multi-threaded tokio runtime sized by the user's config.
/// Exposed because the monitor command reuses it (it has to spawn the
/// engine + the provider's polling task on the same runtime).
pub fn build_runtime(user_cfg: &config::Config) -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(user_cfg.runtime.worker_threads)
        .enable_all()
        .build()
        .context("building tokio runtime")
}

/// Create native drivers, bootstrap the runtime, and return the registration
/// caps plus the runtime handle. Shared by every engine entry point.
struct LoadedModelDrivers {
    model: String,
    caps: DriverCapabilities,
    full_identity: pie_driver_abi::ModelIdentity,
    encode_identity: pie_driver_abi::ModelIdentity,
    kv_handle: Option<pie_driver_abi::KvHandle>,
    drivers: ModelDrivers,
}

struct LoadedPartnerMetadata {
    full_identity: pie_driver_abi::ModelIdentity,
    encode_identity: pie_driver_abi::ModelIdentity,
    kv_handle: Option<pie_driver_abi::KvHandle>,
    page_size: u32,
    supports_media_encode: bool,
    hidden_size: u32,
}

fn model_identity(
    user_cfg: &config::Config,
    caps: &DriverCapabilities,
    artifact_digest: &[u8; 32],
    component: pie_driver_abi::ModelComponent,
) -> Result<pie_driver_abi::ModelIdentity> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(user_cfg.model.name.as_bytes());
    hasher.update(artifact_digest);
    hasher.update(caps.arch_name.as_bytes());
    hasher.update(&caps.vocab_size.to_le_bytes());
    hasher.update(&caps.max_model_len.to_le_bytes());
    hasher.update(caps.activation_dtype.as_bytes());
    hasher.update(&caps.hidden_size.to_le_bytes());
    hasher.update(format!("{:?}", user_cfg.model.driver.kind).as_bytes());
    hasher.update(user_cfg.model.driver.activation_dtype.as_bytes());
    match user_cfg.model.driver.kind {
        config::DriverKind::CudaNative => {
            let options: config::CudaNativeDriverOptions =
                toml::Value::Table(user_cfg.model.driver.options.clone())
                    .try_into()
                    .context("normalizing CUDA options for model identity")?;
            hasher.update(options.runtime_quant.as_bytes());
            hasher.update(options.mxfp4_moe.as_bytes());
            hasher.update(options.weight_dtype.as_bytes());
        }
        config::DriverKind::Dummy => {
            hasher.update(&user_cfg.model.driver.random_seed.to_le_bytes());
        }
        config::DriverKind::Metal => {}
    }
    Ok(pie_driver_abi::ModelIdentity {
        hash: *hasher.finalize().as_bytes(),
        component,
    })
}

fn model_artifact_digest(snapshot_dir: &Path) -> Result<[u8; 32]> {
    let components = snapshot_dir.components().collect::<Vec<_>>();
    for pair in components.windows(2) {
        if pair[0].as_os_str() == "snapshots" {
            let revision = pair[1].as_os_str().to_string_lossy();
            if !revision.is_empty() {
                return Ok(*blake3::hash(revision.as_bytes()).as_bytes());
            }
        }
    }

    fn collect_files(current: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<()> {
        if current.is_file() {
            files.push(current.to_path_buf());
            return Ok(());
        }
        let mut entries = std::fs::read_dir(current)
            .with_context(|| format!("reading model artifact directory {current:?}"))?
            .collect::<std::io::Result<Vec<_>>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                let target = std::fs::canonicalize(&path)?;
                if target.is_file() {
                    files.push(path);
                }
            } else if metadata.is_dir() {
                collect_files(&path, files)?;
            } else if metadata.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    collect_files(snapshot_dir, &mut files)?;
    files.sort();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    for path in files {
        use std::io::Read;

        let relative = path.strip_prefix(snapshot_dir).unwrap_or(&path);
        hasher.update(relative.to_string_lossy().as_bytes());
        let mut file = std::fs::File::open(&path)
            .with_context(|| format!("opening model artifact {path:?}"))?;
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn load_model_drivers(
    user_cfg: &config::Config,
    component: pie_driver_abi::ModelComponent,
) -> Result<LoadedModelDrivers> {
    let (driver_groups, snapshot_dir) = {
        let m = &user_cfg.model;
        let resolved = preflight::resolve_flavor(m.driver.kind, &m.name)?;

        let world_size = m.driver.device.len();
        let tp_degree = if m.driver.tensor_parallel_size == 0 {
            world_size
        } else {
            m.driver.tensor_parallel_size as usize
        };
        let topology = calculate_topology(world_size, tp_degree)
            .with_context(|| format!("model {:?} topology", m.name))?;

        #[allow(unreachable_patterns)]
        if tp_degree > 1 {
            match resolved {
                #[cfg(feature = "driver-cuda")]
                ResolvedFlavor::Embedded(Flavor::Cuda) => {}
                _ => anyhow::bail!(
                    "model {:?}: tensor_parallel_size={tp_degree} is only \
                     supported for cuda_native",
                    m.name,
                ),
            }
        }

        let ResolvedFlavor::Embedded(flavor) = resolved;
        let mut embedded_base_opts = preflight::build_embedded_options(m, flavor)?;
        apply_embedded_verbose(&mut embedded_base_opts, user_cfg.server.verbose);
        let snapshot_dir = weights::resolve(&m.hf_repo)
            .with_context(|| format!("resolving hf_repo for model {:?}", m.name))?;
        let mut group_drivers: Vec<GroupDriver> = Vec::with_capacity(topology.len());
        for (group_idx, group) in topology.iter().enumerate() {
            group_drivers.push(create_driver_group(
                m,
                group_idx,
                group,
                flavor,
                &embedded_base_opts,
                &snapshot_dir,
                tp_degree,
                component,
            )?);
        }
        (
            ModelDrivers {
                groups: group_drivers,
            },
            snapshot_dir,
        )
    };

    let caps = driver_groups
        .groups
        .first()
        .map(|group| group.caps.clone())
        .context("no driver capabilities available for control-plane registration")?;
    let kv_handle = driver_groups
        .groups
        .first()
        .and_then(|group| group.backend.export_kv_handle());
    let artifact_digest = if user_cfg.cluster.controller.is_some() || user_cfg.offload.enabled {
        model_artifact_digest(&snapshot_dir)?
    } else {
        *blake3::hash(user_cfg.model.hf_repo.as_bytes()).as_bytes()
    };
    Ok(LoadedModelDrivers {
        model: user_cfg.model.name.clone(),
        full_identity: model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            pie_driver_abi::ModelComponent::Full,
        )?,
        encode_identity: model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            pie_driver_abi::ModelComponent::Encode,
        )?,
        caps,
        kv_handle,
        drivers: driver_groups,
    })
}

async fn boot_engine(
    user_cfg: &config::Config,
) -> Result<(
    String,
    DriverCapabilities,
    LoadedPartnerMetadata,
    pie_engine::bootstrap::BootstrapHandle,
)> {
    let LoadedModelDrivers {
        model,
        caps,
        full_identity,
        encode_identity,
        kv_handle,
        drivers,
    } = load_model_drivers(user_cfg, pie_driver_abi::ModelComponent::Full)?;

    let boot_cfg =
        translate::build(user_cfg, drivers).context("translating to bootstrap::Config")?;

    let boot = pie_engine::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("pie_engine::bootstrap::bootstrap: {e}"))?;
    let page_size = caps.kv_page_size;
    let supports_media_encode = caps.supports_media_encode;
    let hidden_size = caps.hidden_size;
    Ok((
        model,
        caps,
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
        pie_driver_abi::ModelComponent::Encode
    } else {
        pie_driver_abi::ModelComponent::Full
    };
    let loaded = load_model_drivers(user_cfg, component)?;
    let model_identity = if role == Role::Encode {
        loaded.encode_identity.clone()
    } else {
        loaded.full_identity.clone()
    };
    let server = ExecutorServer::bind_with_transfer(
        &coordinator.control_addr,
        loaded.drivers,
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
            capability: loaded.caps,
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

/// Boot the engine + assemble the control/edge plane for the resolved topology
/// ([`Coordinator`]). The in-proc engine-embedding entry (Seam 1b) used by
/// `bin/worker` (via [`run`]) and the `pie-server-py` wheel (single-node
/// direct-WS).
pub async fn start_engine(
    user_cfg: config::Config,
    coordinator: Coordinator,
) -> Result<EngineHandle> {
    let (model, caps, partner_metadata, runtime) = boot_engine(&user_cfg).await?;
    let partner_bootstrap = build_partner_bootstrap(&user_cfg, partner_metadata, runtime.model_idx);
    let (edge_server, control_tasks, control_plane, partners, url) =
        assemble_control_and_edge(coordinator, &user_cfg, model, caps, partner_bootstrap).await?;
    log_serving(&user_cfg, &url);
    Ok(EngineHandle {
        url,
        edge_server,
        control_tasks,
        partners,
        control_plane,
        runtime: Some(runtime),
    })
}

/// In-proc engine-embedding entry for the composition root (`bin/pie`): boot the
/// engine, then assemble the distributed edge/control plane against an
/// **injected** [`ControlLink`] (the root's `EmbeddedControl`) + the in-proc
/// gateway address(es), instead of dialing a real controller.
pub async fn start_engine_embedded<C: ControlLink>(
    user_cfg: config::Config,
    control: C,
    gateways: Vec<String>,
) -> Result<EngineHandle> {
    let (model, caps, partner_metadata, runtime) = boot_engine(&user_cfg).await?;
    let partner_bootstrap = build_partner_bootstrap(&user_cfg, partner_metadata, runtime.model_idx);
    let addr = topology::addr_from_host_port(&user_cfg.server.host, user_cfg.server.port);
    // A single-node-monolithic worker serves all stages; routing doesn't filter
    // by role yet, so Decode is an inert default (echo owns Role::Monolithic).
    let (edge_server, control_tasks, worker_id, partners) = assemble_distributed(
        control,
        &gateways,
        Role::Decode,
        model,
        addr,
        caps,
        partner_bootstrap,
    )
    .await?;
    let url = edge_server.url();
    log_serving(&user_cfg, &url);
    Ok(EngineHandle {
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
    pie_engine::offload::configure(
        user_cfg.offload.enabled,
        user_cfg.offload.prefill_min_suffix_tokens,
    );
    pie_engine::offload::configure_encode_injection(
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
    pie_engine::offload::set_home_kv_handle(kv_handle.clone());
    Some(partner::PartnerBootstrap {
        full_identity: metadata.full_identity,
        encode_identity: metadata.encode_identity,
        kv_layout: kv_handle.layout.clone(),
        home_kv_handle: kv_handle,
        transfer: user_cfg.offload.transfer,
        model_idx,
        page_size: metadata.page_size,
        request_timeout_secs: user_cfg.model.scheduler.request_timeout_secs,
        max_outstanding: user_cfg.offload.max_outstanding_per_partner,
    })
}

/// Print the startup banner when `server.verbose` is set.
fn log_serving(cfg: &config::Config, url: &str) {
    if cfg.server.verbose {
        eprintln!("{}", StartupBanner::from_config(cfg).render(url));
    }
}

/// Build the client-facing edge server + control plane for the resolved
/// topology, after the runtime is bootstrapped and driver capabilities are
/// known. Returns the edge server, the worker's control-loop tasks, the live
/// control-plane resources to hold for the engine's lifetime, and the URL to
/// advertise.
///
/// - **distributed:** dial the controller, register, spawn the
///   heartbeat/report/watch loops, then dial INTO each configured gateway and
///   serve `WorkerControl` over the link (M3 — the worker is the client, the
///   gateway the listening server).
/// - **single-node:** terminate client WebSockets directly; no control plane.
async fn assemble_control_and_edge(
    coordinator: Coordinator,
    user_cfg: &config::Config,
    model: String,
    caps: DriverCapabilities,
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
            // Dial the controller (the daemon's control link), then register +
            // spawn loops + dial INTO the gateways via the shared assembly.
            let client = control::dial_controller(&controller)
                .await
                .with_context(|| format!("dialing controller at {controller}"))?;
            let (edge, control_tasks, worker_id, partners) = assemble_distributed(
                client.clone(),
                &gateways,
                role,
                model,
                coordinator.control_addr.clone(),
                caps,
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
            // Gateway-free local inference: the worker terminates client
            // WebSockets itself and never registers, so the model name and
            // capabilities have no controller to be registered with.
            let _ = (model, caps, partner_bootstrap);
            let listen = format!("{}:{}", user_cfg.server.host, user_cfg.server.port);
            let edge = EdgeServer::Standalone(
                client_server::spawn(&listen)
                    .await
                    .context("starting standalone client server")?,
            );
            let url = edge.url();
            Ok((edge, Vec::new(), ClusterControl::None, None, url))
        }
    }
}

/// Register the worker over `control`, spawn its three control loops, and dial
/// INTO the gateways, serving `WorkerControl` over the links. Generic over the
/// [`ControlLink`] backend so the daemon injects a dialed [`ControlClient`] and
/// the composition root (`bin/pie`) injects its in-proc `EmbeddedControl`.
///
/// `register` happens BEFORE dialing the gateways, so the worker presents its
/// controller-minted id on each gateway dial-in `register` (the join key for
/// `routing ∩ connected`). The static `gateways` are pinned (dialed eagerly for
/// boot readiness); the control-plane watch loop then reconciles the dial-in set
/// against the controller-pushed gateway roster (`gateway.md`), so an empty list
/// means fully dynamic discovery.
async fn assemble_distributed<C: ControlLink>(
    control: C,
    gateways: &[String],
    role: Role,
    model: String,
    addr: String,
    caps: DriverCapabilities,
    partner_bootstrap: Option<partner::PartnerBootstrap>,
) -> Result<(
    EdgeServer,
    Vec<tokio::task::JoinHandle<()>>,
    WorkerId,
    Option<std::sync::Arc<tokio::sync::Mutex<partner::PartnerLinkManager>>>,
)> {
    let info = WorkerInfo {
        role,
        model,
        addr,
        capability: caps,
    };
    let worker_id = ControlLink::register_worker(&control, info)
        .await
        .context("registering worker with controller")?;

    // The static `gateways` are a pin/override: always kept dialed. Dial them
    // eagerly for boot readiness, then hand the manager to the watch loop, which
    // reconciles dial-in links against the controller-pushed roster (gateway.md).
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

fn create_driver_group(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &DriverOptions,
    snapshot_dir: &Path,
    tp_degree: usize,
    component: pie_driver_abi::ModelComponent,
) -> Result<GroupDriver> {
    #[cfg(feature = "driver-cuda")]
    {
        if flavor == Flavor::Cuda && tp_degree > 1 {
            let rank_opts = cuda_rank_options(m, group_idx, group, base_opts)?;
            let tp_launches = crate::embedded_driver::tp_launches(rank_opts.len())?;
            return crate::embedded_driver::create_driver_backend_group(
                &rank_opts,
                snapshot_dir,
                group_idx,
                &tp_launches,
                component,
            )
            .with_context(|| {
                format!(
                    "creating cuda TP driver group for model {:?} group {group_idx}",
                    m.name,
                )
            });
        }
    }

    #[cfg(not(feature = "driver-cuda"))]
    let _ = (flavor, tp_degree);

    let first_driver_idx = group.first().copied().ok_or_else(|| {
        anyhow!(
            "model {:?}: group {group_idx} is empty; topology calculation produced no ranks",
            m.name,
        )
    })?;
    let device = group_driver(m, group_idx, first_driver_idx)?;
    let opts = embedded_opts_for_device(base_opts, device);

    crate::embedded_driver::create_driver_backend(&opts, snapshot_dir, group_idx, None, component)
        .with_context(|| format!("creating driver for model {:?} group {group_idx}", m.name,))
}

fn embedded_opts_for_device(base_opts: &DriverOptions, device: String) -> DriverOptions {
    #[cfg(not(feature = "driver-cuda"))]
    let _ = &device;

    #[allow(unreachable_patterns)]
    match base_opts {
        #[cfg(feature = "driver-cuda")]
        DriverOptions::CudaNative(opts) => {
            let mut opts = opts.clone();
            opts.device = device;
            DriverOptions::CudaNative(opts)
        }
        other => other.clone(),
    }
}

fn apply_embedded_verbose(options: &mut DriverOptions, verbose: bool) {
    #[cfg(feature = "driver-cuda")]
    if let DriverOptions::CudaNative(opts) = options {
        opts.verbose = verbose;
    }

    #[cfg(feature = "driver-metal")]
    if let DriverOptions::Metal(opts) = options {
        opts.verbose = verbose;
    }

    #[cfg(not(any(feature = "driver-cuda", feature = "driver-metal")))]
    let _ = (options, verbose);
}

#[cfg(feature = "driver-cuda")]
fn cuda_rank_options(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    base_opts: &DriverOptions,
) -> Result<Vec<DriverOptions>> {
    let mut rank_opts = Vec::with_capacity(group.len());
    for &rank_driver_idx in group {
        let rank_driver = group_driver(m, group_idx, rank_driver_idx)?;
        match base_opts {
            DriverOptions::CudaNative(opts) => {
                let mut o = opts.clone();
                o.device = rank_driver;
                rank_opts.push(DriverOptions::CudaNative(o));
            }
            _ => unreachable!("flavor checked before building cuda rank options"),
        }
    }
    Ok(rank_opts)
}

fn group_driver(m: &config::ModelConfig, group_idx: usize, driver_idx: usize) -> Result<String> {
    m.driver
        .device
        .get(driver_idx)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "model {:?}: group {group_idx} references device index {} but only {} devices configured",
                m.name,
                driver_idx,
                m.driver.device.len(),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::{StartupBanner, model_artifact_digest, model_identity};

    #[test]
    fn startup_banner_render_includes_public_startup_fields_only() {
        let banner = StartupBanner {
            model: "default (Qwen/Qwen3-0.6B)".to_string(),
            driver: "dummy".to_string(),
            device: "cpu".to_string(),
        };

        let rendered = banner.render("ws://127.0.0.1:8080");

        assert!(rendered.contains("╭─ Pie Engine"));
        assert!(rendered.contains("Host"));
        assert!(rendered.contains("Model"));
        assert!(rendered.contains("Driver"));
        assert!(rendered.contains("Device"));
        assert!(rendered.contains("✓ Server ready at ws://127.0.0.1:8080"));
        assert!(!rendered.contains("internal token"));
    }

    #[test]
    fn model_identity_uses_snapshot_revision_or_file_contents() {
        let root = tempfile::tempdir().unwrap();
        let revision = "0123456789abcdef";
        let snapshot = root.path().join("snapshots").join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        assert_eq!(
            model_artifact_digest(&snapshot).unwrap(),
            *blake3::hash(revision.as_bytes()).as_bytes()
        );

        let local = root.path().join("local");
        std::fs::create_dir_all(&local).unwrap();
        std::fs::write(local.join("weights.bin"), b"first").unwrap();
        let first = model_artifact_digest(&local).unwrap();
        std::fs::write(local.join("weights.bin"), b"second").unwrap();
        let second = model_artifact_digest(&local).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn component_identity_ignores_kv_capacity_but_covers_weight_semantics() {
        let config = crate::config::Config::parse(
            r#"
            [model]
            name = "test"
            hf_repo = "local"

            [model.driver]
            type = "dummy"
            device = ["cpu"]

            [model.driver.options]
            vocab_size = 32
            arch_name = "dummy"
            "#,
        )
        .unwrap();
        let driver = pie_engine::driver::DummyDriver::new(
            pie_driver_dummy_lib::DummyDriverOptions::default(),
        );
        let full_caps = driver.capabilities().clone();
        let mut encode_caps = full_caps.clone();
        encode_caps.total_pages = 0;
        encode_caps.kv_page_size = 0;
        encode_caps.kv_handle = None;
        let artifact = [7; 32];
        let full = model_identity(
            &config,
            &full_caps,
            &artifact,
            pie_driver_abi::ModelComponent::Full,
        )
        .unwrap();
        let encode = model_identity(
            &config,
            &encode_caps,
            &artifact,
            pie_driver_abi::ModelComponent::Encode,
        )
        .unwrap();
        assert_eq!(full.hash, encode.hash);
        assert_ne!(full.component, encode.component);

        encode_caps.activation_dtype = "f16".to_string();
        let incompatible = model_identity(
            &config,
            &encode_caps,
            &artifact,
            pie_driver_abi::ModelComponent::Encode,
        )
        .unwrap();
        assert_ne!(encode.hash, incompatible.hash);
    }
}
