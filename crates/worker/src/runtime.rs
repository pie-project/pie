//! `pie serve` core: boot engines, wire RPC, hand off to the runtime,
//! and surface an [`RuntimeHandle`] the caller drives.
//!
//! Wires the standalone's pieces in dependency order:
//!   1. Translate user TOML to per-engine options.
//!   2. For the `[model]`, partition devices into DP groups and create
//!      native engines directly, collecting their capabilities.
//!   3. Translate the resulting native engines → [`::runtime::bootstrap::Config`]
//!      and call [`::runtime::bootstrap::bootstrap`]. The runtime now owns
//!      the runtime services + scheduler; the worker dials into the
//!      gateway and serves `worker_api::WorkerControl`.
//!   4. Caller decides what to do with the [`RuntimeHandle`]:
//!        * `pie serve`: [`RuntimeHandle::wait_then_shutdown`] blocks
//!          on SIGINT/SIGTERM/watchdog and tears down.
//!        * `pie serve --monitor`: TUI runs concurrently and calls
//!          [`RuntimeHandle::shutdown`] when the user quits.

use anyhow::{Context, Result, anyhow, bail};
use controller_api::{ControlClient, Role, WorkerInfo};
use ids::WorkerId;
use std::path::Path;

use crate::config;
use crate::embedded_engine::{EngineCapabilities, EngineOptions};
use crate::engine_ffi::Flavor;
use crate::executor::ExecutorServer;
use crate::link::control::{self, ControlLink};
use crate::link::{gateway, partner, topology};
use crate::preflight;
use crate::translate::{self, GroupEngine, ModelEngines};
use crate::{client_server, lifecycle, weights};

pub use crate::link::topology::{Coordinator, TopologyMode, connect};
pub use crate::preflight::calculate_topology;

/// Live server — engines, RPC dispatch threads, and enough state to perform an
/// orderly shutdown.
/// Returned from [`start_runtime`]; consumed by either
/// [`RuntimeHandle::wait_then_shutdown`] (the `pie serve` path) or
/// [`RuntimeHandle::shutdown`] (the `pie serve --monitor` path, where
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

pub struct RuntimeHandle {
    runtime: Option<::runtime::bootstrap::BootstrapHandle>,
    edge_server: EdgeServer,
    /// Controller heartbeat/report/watch tasks. Empty when there is no control
    /// plane (single-node without the `single-node` feature).
    control_tasks: Vec<tokio::task::JoinHandle<()>>,
    partners: Option<std::sync::Arc<tokio::sync::Mutex<partner::PartnerLinkManager>>>,
    /// Live control-plane state kept alive for the server's lifetime: the dialed
    /// client (distributed) or the embedded controller handle + in-proc gateway
    /// task (single-node feature). `None` in gateway-free single-node.
    control_plane: ClusterControl,
    /// Client endpoint this worker advertises: `ws://host:port` in single-node
    /// (direct client server, or the in-proc gateway), or `gateway://addr[,…]`
    /// in distributed (the gateway endpoint(s) the worker dialed into — clients
    /// hit the gateway, not the worker).
    pub url: String,
}

/// Live control-plane resources held for the server's lifetime, by topology.
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

impl RuntimeHandle {
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

    /// Tear down the server without waiting for a signal. Used by the
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
        if let Some(runtime) = self.runtime.take()
            && let Err(err) = runtime.shutdown().await
        {
            tracing::error!(?err, "runtime shutdown failed");
        }
    }
}

/// A running worker: the server plus an async drain-and-stop. Returned by
/// [`run`] (daemon) and [`run_with`] (in-proc embed). The bin owns the runtime
/// (Model A) and drives [`shutdown`](WorkerHandle::shutdown) on signal.
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
    /// The client endpoint this worker advertises (`ws://…` in single-node, or
    /// the `gateway://…` endpoint(s) it dialed into in distributed mode).
    pub fn url(&self) -> &str {
        match &self.inner {
            WorkerKind::Decode(engine) => &engine.url,
            WorkerKind::Executor(executor) => executor.server.endpoint(),
        }
    }

    /// Drain in-flight work and stop the server (runtime, control loops, edge).
    pub async fn shutdown(self) {
        match self.inner {
            WorkerKind::Decode(engine) => engine.shutdown().await,
            WorkerKind::Executor(executor) => executor.shutdown().await,
        }
        // The bootstrap TOMLs under `$PIE_HOME/standalone/<pid>` are read once at
        // engine creation and never again, so they are dead the moment the
        // engines are down. The boot sweep covers the unclean exits.
        crate::embedded_engine::remove_launch_state();
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
/// the runtime and drives `shutdown` on signal via the bin layer's `bootstrap` skeleton.
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

/// In-proc embed entry for the composition root (`bin/pie`): run the worker
/// against an **injected** [`ControlLink`] (the root's `EmbeddedControl`) plus
/// the in-proc gateway address(es), instead of dialing a real controller — the
/// counterpart of the gateway's `run_with`.
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

struct StartupBanner {
    model: String,
    /// The execution shell the box's third row names. Called `backend` and not
    /// `engine` because the box is headed `Pie Engine` -- the product -- and a
    /// field one line from that header sharing its word is the same collision
    /// the row label had. It is the backend axis everywhere else in the tree
    /// (`register_engine_backend`, `runtime::engine::backend`).
    backend: String,
    device: String,
}

impl StartupBanner {
    fn from_config(cfg: &config::Config) -> Self {
        let m = &cfg.model;
        let model = format!("{} ({})", m.name, m.model);
        let backend = m.engine.kind.as_str().to_string();
        let device = {
            let device = m.engine.device.join(", ");
            if device.is_empty() {
                "-".to_string()
            } else {
                device
            }
        };

        Self {
            model,
            backend,
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
            ("Backend", self.backend.as_str()),
            ("Device", self.device.as_str()),
        ];
        let label_width = 12;
        let header = "─ Pie Engine ";
        // Character counts, not `str::len()`. `header` opens with `─` (U+2500,
        // three bytes), so its byte length is 15 against 13 columns -- the top
        // border came out two dashes short of every other line. Rust's format
        // width counts characters, so byte lengths cannot be mixed with it.
        let header_cols = header.chars().count();
        let content_width = rows
            .iter()
            .map(|(_, value)| label_width + 1 + value.chars().count())
            .max()
            .unwrap_or(0)
            .max(header_cols - 2);
        let inner_width = content_width + 2;
        let mut out = String::new();

        out.push_str(&format!(
            "╭{}{}╮\n",
            header,
            "─".repeat(inner_width - header_cols)
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
        .worker_threads(user_cfg.server.worker_threads)
        .enable_all()
        .build()
        .context("building tokio runtime")
}

/// Create native engines, bootstrap the runtime, and return the registration
/// caps plus the runtime handle. Shared by every server entry point.
struct LoadedModelEngines {
    model: String,
    caps: EngineCapabilities,
    full_identity: crate::executor::ModelIdentity,
    encode_identity: crate::executor::ModelIdentity,
    kv_handle: Option<engine_api::KvHandle>,
    engines: ModelEngines,
    /// The model's compiled metadata, read once while resolving it. Present
    /// for either input form: an artifact carries the config, a snapshot's
    /// `config.json` is normalized into one.
    metadata: ::runtime::model::ModelMetadata,
}

struct LoadedPartnerMetadata {
    full_identity: crate::executor::ModelIdentity,
    encode_identity: crate::executor::ModelIdentity,
    kv_handle: Option<engine_api::KvHandle>,
    page_size: u32,
    supports_media_encode: bool,
    hidden_size: u32,
}

/// THE ENGINE'S SELF-REPORT, DELIBERATELY, WHERE `register` READS THE ROW.
///
/// `arch_name` and `vocab_size` are stated on `::runtime::model::ROWS`, and the
/// engine's own `register` reads them there — but this is not a statement of
/// what the model IS. It is the token two workers compare before they trade
/// KV pages, and what has to agree is what the two ENGINES loaded, not what
/// the two catalogs say. Substituting the row would fold in a table both
/// binaries already share, which discriminates nothing, and would make the one
/// divergence worth catching — an engine reporting a width its row does not
/// state — invisible. `max_model_len`, `activation_dtype` and `hidden_size` in
/// the same hash have no row column at all, which is the same fact from the
/// other side: this is the engine's answer, whole.
fn model_identity(
    user_cfg: &config::Config,
    caps: &EngineCapabilities,
    artifact_digest: &[u8; 32],
    component: crate::executor::ModelComponent,
) -> Result<crate::executor::ModelIdentity> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(user_cfg.model.name.as_bytes());
    hasher.update(artifact_digest);
    // THE ENGINE'S ANSWER, WHOLE — still, and read out of the three records
    // that replaced the flat struct. `arch_name` and `hidden_size` are gone
    // from it: the first says which catalog row the CALLER resolved (so it
    // discriminated nothing two peers did not already share), and the second
    // has no seat in `ModelProfile`. What is left is what the two engines
    // actually loaded.
    hasher.update(caps.device.backend.as_bytes());
    hasher.update(&caps.profile.vocab.to_le_bytes());
    hasher.update(&caps.profile.num_layers.to_le_bytes());
    hasher.update(&caps.limits.max_context.to_le_bytes());
    hasher.update(caps.profile.activation.name().as_bytes());
    hasher.update(&caps.pools.kv_page_size.to_le_bytes());
    hasher.update(format!("{:?}", user_cfg.model.engine.kind).as_bytes());
    hasher.update(user_cfg.model.engine.activation_dtype.as_bytes());
    // A `[model]` key rather than an engine option: what the checkpoint holds
    // is a fact about the weights, so it discriminates for every kind and is
    // read off the model rather than out of the options bag below.
    hasher.update(user_cfg.model.weight_dtype.as_bytes());
    match user_cfg.model.engine.kind {
        config::EngineKind::CudaNative => {
            let options: config::CudaNativeEngineOptions =
                toml::Value::Table(user_cfg.model.engine.options.clone())
                    .try_into()
                    .context("normalizing CUDA options for model identity")?;
            hasher.update(options.runtime_quant.as_bytes());
            hasher.update(options.mxfp4_moe.as_bytes());
        }
        // Nothing to add for any of the three. The identity already carries
        // the kind itself, and none of them has an option that changes what
        // the weights ARE -- no requantization, no expert lowering choice, and
        // wgpu's one knob is a page count that sizes a pool. An option folded
        // in here would make two runs that serve the same bytes trade no
        // cached layout.
        config::EngineKind::Metal | config::EngineKind::Vulkan | config::EngineKind::Wgpu => {}
    }
    Ok(crate::executor::ModelIdentity {
        hash: *hasher.finalize().as_bytes(),
        component,
    })
}

/// The identity of a `.zt` artifact, or `None` for anything else.
///
/// The loader answers what identifies a checkpoint; this only folds its answer
/// into the 32-byte shape the identity plumbing expects.
fn manifest_digest(path: &Path) -> Result<Option<[u8; 32]>> {
    let identity = model_loader::checkpoint::zt::artifact_identity(path)
        .map_err(|err| anyhow!("reading the identity of {path:?}: {err}"))?;
    Ok(identity.map(|bytes| *blake3::hash(&bytes).as_bytes()))
}

fn model_artifact_digest(snapshot_dir: &Path) -> Result<[u8; 32]> {
    // A `.zt` artifact already has an identity, and it is a better one than
    // anything derivable here: the manifest digest covers every tensor, the
    // compiled tokenizer and the checkpoint config together, and for a sharded
    // artifact it reaches the shards through their entries in the shard table.
    // It also survives the file being moved, which the path-derived answer
    // below does not.
    if let Some(digest) = manifest_digest(snapshot_dir)? {
        return Ok(digest);
    }

    // Legacy snapshots. The revision in `snapshots/<rev>/` is HF's own content
    // identity, so it beats re-hashing gigabytes; falling through to the walk
    // is the last resort.
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

fn load_model_engines(
    user_cfg: &config::Config,
    component: crate::executor::ModelComponent,
) -> Result<LoadedModelEngines> {
    // Process housekeeping, once per boot and before anything writes under
    // `$PIE_HOME/standalone/<pid>`: reclaim the directories left by launches
    // that did not exit cleanly.
    crate::embedded_engine::sweep_stale_launch_state();

    // Every engine-side disk cache derives from this. Resolved here because
    // `$PIE_HOME` is the worker layer's to know; the engine has never been
    // told it, which is the only reason those caches used to sit under XDG.
    crate::embedded_engine::set_cache_dir(
        crate::state::engine_cache_dir()
            .to_string_lossy()
            .into_owned(),
    );

    // Resolve the weight-artifact directory here, before any engine bootstrap
    // TOML is written. `$PIE_HOME` is this layer's to know: the engine has
    // never been told it, which is why the old env-var form fell back to XDG.
    let weight_cache_dir = if user_cfg.model.weight_cache_dir.is_empty() {
        // Under `cache/`, not `models/`. `models/` is the artifact store now,
        // and these are the opposite kind of thing: device bytes for one
        // engine, one TP layout and one ABI version, rebuilt by a single cold
        // load. Sharing a directory left `.weights` files sitting in a store
        // that scans for `.zt` and silently ignored them, while `pie cache`
        // reported their size under the store's name.
        crate::state::weight_cache_dir()
            .to_string_lossy()
            .into_owned()
    } else {
        user_cfg.model.weight_cache_dir.clone()
    };
    crate::embedded_engine::set_weight_cache_dir(weight_cache_dir);

    let (engine_groups, snapshot_dir, metadata) = {
        let m = &user_cfg.model;
        let flavor = preflight::resolve_flavor(m.engine.kind, &m.name)?;

        let world_size = m.engine.device.len();
        let tp_degree = if m.engine.tensor_parallel_size == 0 {
            world_size
        } else {
            m.engine.tensor_parallel_size as usize
        };
        let topology = calculate_topology(world_size, tp_degree)
            .with_context(|| format!("model {:?} topology", m.name))?;

        #[allow(unreachable_patterns)]
        if tp_degree > 1 {
            match flavor {
                #[cfg(feature = "_engine-cuda")]
                Flavor::Cuda => {}
                _ => anyhow::bail!(
                    "model {:?}: tensor_parallel_size={tp_degree} is only \
                     supported for cuda_native",
                    m.name,
                ),
            }
        }

        let mut embedded_base_opts = preflight::build_embedded_options(m, flavor)?;
        apply_embedded_verbose(&mut embedded_base_opts, user_cfg.server.verbose);
        apply_embedded_calibration(&mut embedded_base_opts, user_cfg.server.calibrate_planner);
        let resolved_model = weights::resolve(&m.model)
            .with_context(|| format!("resolving the model for {:?}", m.name))?;
        // A SECOND STEP STOOD HERE -- weights::prefer_runtime, which asked
        // whether the store already held a `<name>/runtime/` artifact built for
        // this exact boot and bound it instead of the archive. Nothing in this
        // build writes that directory: its sole producer was the offline
        // `pie model build`, which R3 retired with the load contract its
        // transforms authored, so the lookup could only ever miss. It is deleted
        // rather than left as a no-op that reads like a live cache. The archive
        // is general form and is servable as it stands; what a runtime bought
        // was moving the family transforms offline, and that returns when a
        // command writes one again.
        //
        // Lifted once, here, in one open. The engines get the compiled model
        // config beside their bootstrap TOML; the runtime gets the whole of it.
        // Nobody downstream re-opens the artifact or re-decides what it is —
        // and for a snapshot, nobody re-parses `config.json`, which is what
        // this one call replaced on both sides.
        let lifted = resolved_model
            .metadata()
            .with_context(|| format!("reading the model metadata for {:?}", m.name))?;
        let config = lifted.config.clone();
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
                &config,
                tp_degree,
                component,
                // **THE ONE RUN-AHEAD NUMBER, HANDED OVER AT THE LOAD** — the
                // same `[runtime] frame_dispatch_depth` the scheduler is given
                // below, so the engine's staging ring and the scheduler's
                // in-flight bound derive from one statement rather than two
                // (article 8).
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
        model_artifact_digest(&snapshot_dir)?
    } else {
        *blake3::hash(user_cfg.model.model.as_bytes()).as_bytes()
    };
    Ok(LoadedModelEngines {
        metadata,
        model: user_cfg.model.name.clone(),
        full_identity: model_identity(
            user_cfg,
            &caps,
            &artifact_digest,
            crate::executor::ModelComponent::Full,
        )?,
        encode_identity: model_identity(
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
    EngineCapabilities,
    LoadedPartnerMetadata,
    ::runtime::bootstrap::BootstrapHandle,
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
    // The checkpoint's own `config.json`, read for the one number the engine
    // no longer answers. A checkpoint that does not state it means an encode
    // partner cannot be sized, which `configure_encode_injection` reads as
    // "off" — the honest answer, and the same one an engine that reported
    // zero gave.
    let metadata_hidden_size: u32 = serde_json::from_slice::<serde_json::Value>(&metadata.config)
        .ok()
        .and_then(|config| config.get("hidden_size")?.as_u64())
        .and_then(|size| u32::try_from(size).ok())
        .unwrap_or(0);

    let boot_cfg = translate::build(user_cfg, engines, metadata)
        .context("translating to bootstrap::Config")?;

    let boot = ::runtime::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("::runtime::bootstrap::bootstrap: {e}"))?;
    let page_size = caps.pools.kv_page_size;
    let supports_media_encode = caps.media_encode;
    // OFF THE MODEL'S OWN METADATA, not off the engine. `hidden_size` was a
    // `EngineCapabilities` field and `ModelProfile` has no seat for it; the
    // checkpoint's config carries it, and this crate lifted that config a
    // moment ago.
    let hidden_size = metadata_hidden_size;
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
pub async fn start_runtime(
    user_cfg: config::Config,
    coordinator: Coordinator,
) -> Result<RuntimeHandle> {
    let (model, caps, partner_metadata, runtime) = boot_engine(&user_cfg).await?;
    let partner_bootstrap = build_partner_bootstrap(&user_cfg, partner_metadata, runtime.model_idx);
    let (edge_server, control_tasks, control_plane, partners, url) =
        assemble_control_and_edge(coordinator, &user_cfg, model, caps, partner_bootstrap).await?;
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

/// In-proc engine-embedding entry for the composition root (`bin/pie`): boot the
/// engine, then assemble the distributed edge/control plane against an
/// **injected** [`ControlLink`] (the root's `EmbeddedControl`) + the in-proc
/// gateway address(es), instead of dialing a real controller.
pub async fn start_runtime_embedded<C: ControlLink>(
    user_cfg: config::Config,
    control: C,
    gateways: Vec<String>,
    client_edge: Option<String>,
) -> Result<RuntimeHandle> {
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
    // `edge_server.url()` reports the address the worker DIALED -- the
    // gateway's worker-facing listener. In standalone that is an ephemeral port
    // that speaks the worker protocol, so advertising it told the user to point
    // their client at a socket no client can use, and at a different (random)
    // port on every boot. The composition root knows the real client edge, so it
    // passes it; the dial-in listing is only right when nobody knows better.
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
    ::runtime::offload::configure(
        user_cfg.offload.enabled,
        user_cfg.offload.prefill_min_suffix_tokens,
    );
    ::runtime::offload::configure_encode_injection(
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
    ::runtime::offload::set_home_kv_handle(kv_handle.clone());
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

/// Print the bootstrap banner when `server.verbose` is set.
fn log_serving(cfg: &config::Config, url: &str) {
    if cfg.server.verbose {
        eprintln!("{}", StartupBanner::from_config(cfg).render(url));
    }
}

/// Build the client-facing edge server + control plane for the resolved
/// topology, after the runtime is bootstrapped and engine capabilities are
/// known. Returns the edge server, the worker's control-loop tasks, the live
/// control-plane resources to hold for the server's lifetime, and the URL to
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
    caps: EngineCapabilities,
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
    caps: EngineCapabilities,
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

#[allow(
    clippy::too_many_arguments,
    reason = "nine independent inputs to one engine launch; a struct here \
              would be a parameter list with a name"
)]
#[cfg_attr(
    not(feature = "_engine-cuda"),
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
    config: &[u8],
    tp_degree: usize,
    component: crate::executor::ModelComponent,
    frames_in_flight: u8,
) -> Result<GroupEngine> {
    #[cfg(feature = "_engine-cuda")]
    {
        if flavor == Flavor::Cuda && tp_degree > 1 {
            let rank_opts = cuda_rank_options(m, group_idx, group, base_opts)?;
            let tp_launches = crate::embedded_engine::tp_launches(rank_opts.len())?;
            return crate::embedded_engine::create_engine_backend_group(
                &rank_opts,
                &m.weight_dtype,
                snapshot_dir,
                config,
                group_idx,
                &tp_launches,
                component,
                frames_in_flight,
                &m.adapters,
            )
            .with_context(|| {
                format!(
                    "creating cuda TP engine group for model {:?} group {group_idx}",
                    m.name,
                )
            });
        }
    }

    #[cfg(not(feature = "_engine-cuda"))]
    let _ = (flavor, tp_degree);

    let first_engine_idx = group.first().copied().ok_or_else(|| {
        anyhow!(
            "model {:?}: group {group_idx} is empty; topology calculation produced no ranks",
            m.name,
        )
    })?;
    let device = group_engine(m, group_idx, first_engine_idx)?;
    let opts = embedded_opts_for_device(base_opts, device);

    crate::embedded_engine::create_engine_backend(
        &opts,
        &m.weight_dtype,
        snapshot_dir,
        config,
        group_idx,
        None,
        component,
        frames_in_flight,
        &m.adapters,
    )
    .with_context(|| format!("creating engine for model {:?} group {group_idx}", m.name,))
}

fn embedded_opts_for_device(base_opts: &EngineOptions, device: String) -> EngineOptions {
    #[cfg(not(feature = "_engine-cuda"))]
    let _ = &device;

    #[allow(unreachable_patterns)]
    match base_opts {
        #[cfg(feature = "_engine-cuda")]
        EngineOptions::CudaNative(opts) => {
            let mut opts = opts.clone();
            opts.device = device;
            EngineOptions::CudaNative(opts)
        }
        other => other.clone(),
    }
}

/// Carry a calibration request from the in-memory config onto the engine
/// options, the same way `verbose` travels.
///
/// CUDA only, because it is the only engine with a memory planner to calibrate.
/// A request against any other engine is silently nothing rather than an error:
/// the caller asked for a measurement this backend does not have, and refusing
/// the boot over it would be worse than doing the ordinary thing.
fn apply_embedded_calibration(options: &mut EngineOptions, calibrate: bool) {
    // ONE VARIANT OR SEVERAL, depending on the feature list. `if let` is
    // how this reads "the CUDA options, if these are them", and in a build
    // whose only engine is CUDA there is nothing else it could be -- so the
    // pattern is irrefutable there and refutable everywhere else. The `if`
    // is kept because the other builds need it.
    #[cfg(feature = "_engine-cuda")]
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    if let EngineOptions::CudaNative(opts) = options {
        opts.calibrate_planner = calibrate;
    }

    #[cfg(not(feature = "_engine-cuda"))]
    let _ = (options, calibrate);
}

fn apply_embedded_verbose(options: &mut EngineOptions, verbose: bool) {
    #[cfg(feature = "_engine-cuda")]
    #[allow(
        irrefutable_let_patterns,
        reason = "`EngineOptions` has one variant in a CUDA-only build"
    )]
    if let EngineOptions::CudaNative(opts) = options {
        opts.verbose = verbose;
    }

    #[cfg(not(feature = "_engine-cuda"))]
    let _ = (options, verbose);
}

#[cfg(feature = "_engine-cuda")]
fn cuda_rank_options(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    base_opts: &EngineOptions,
) -> Result<Vec<EngineOptions>> {
    let mut rank_opts = Vec::with_capacity(group.len());
    for &rank_engine_idx in group {
        let rank_engine = group_engine(m, group_idx, rank_engine_idx)?;
        // The wildcard is unreachable in a CUDA-only build and the only
        // arm that catches a metal or vulkan option set in any other, which
        // is the same asymmetry `apply_embedded_calibration` explains.
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

#[cfg(test)]
mod tests {
    use super::{StartupBanner, model_artifact_digest};

    #[test]
    fn startup_banner_render_includes_public_startup_fields_only() {
        let banner = StartupBanner {
            model: "default (Qwen/Qwen3-0.6B)".to_string(),
            backend: "metal".to_string(),
            device: "cpu".to_string(),
        };

        let rendered = banner.render("ws://127.0.0.1:8080");

        assert!(rendered.contains("╭─ Pie Engine"));
        assert!(rendered.contains("Host"));
        assert!(rendered.contains("Model"));
        // The LABEL PADDED, not the bare word. The row names the execution
        // shell and the header names the product, and while `Backend` no
        // longer collides with `Pie Engine` the way `Engine` did, only the
        // padded form proves the ROW rendered rather than the word appearing
        // somewhere in the box. `label_width` is 12 and `Backend` is seven
        // characters, so the row is the label, five columns of padding, the
        // separator space, and the value.
        assert!(rendered.contains("Backend      metal"));
        assert!(rendered.contains("Device"));
        assert!(rendered.contains("✓ Server ready at ws://127.0.0.1:8080"));
        assert!(!rendered.contains("internal token"));
    }

    /// The box lines up.
    ///
    /// It did not: `content_width` and the top border's dash count were taken
    /// from `str::len()` (bytes) while `format!`'s width pads by characters,
    /// and `"─ Pie Engine "` opens with a three-byte `─`. The top border came
    /// out two columns short of the rows beneath it. A short model name hides
    /// it (the header is not the widest line), so the case that matters is a
    /// long one.
    #[test]
    fn startup_banner_box_is_aligned() {
        for model in [
            "default (Qwen/Qwen3-0.6B)",
            "default (mlx-community/Qwen3.5-0.8B-4bit)",
            "d",
        ] {
            let banner = StartupBanner {
                model: model.to_string(),
                backend: "metal".to_string(),
                device: "metal:0".to_string(),
            };
            let rendered = banner.render("ws://127.0.0.1:8080");
            let widths: Vec<usize> = rendered
                .lines()
                .take_while(|l| !l.is_empty())
                .map(|l| l.chars().count())
                .collect();
            assert!(
                widths.windows(2).all(|w| w[0] == w[1]),
                "unaligned for {model:?}: {widths:?}\n{rendered}"
            );
        }
    }

    #[test]
    fn an_artifacts_identity_is_its_contents_and_survives_a_move() {
        use model_loader::checkpoint::write::CheckpointWriter;
        use model_loader::types::{DType, Encoding, TensorDecl, TensorId};

        let dir = tempfile::tempdir().unwrap();
        let write = |path: &std::path::Path, bytes: &[u8]| {
            let mut writer = CheckpointWriter::create(path, &Default::default()).unwrap();
            writer
                .add_tensor(
                    &TensorDecl {
                        id: TensorId(0),
                        name: "w".to_string(),
                        shape: vec![bytes.len() as i64],
                        encoding: Encoding::Raw(DType::U8),
                        alignment: 1,
                        visibility: Default::default(),
                    },
                    bytes,
                )
                .unwrap();
            writer.finish().unwrap();
        };

        let a = dir.path().join("a.zt");
        let b = dir.path().join("nested").join("b.zt");
        std::fs::create_dir_all(b.parent().unwrap()).unwrap();
        write(&a, &[1u8, 2, 3, 4]);
        write(&b, &[1u8, 2, 3, 4]);

        // Same contents, different names and directories — same identity. The
        // path-derived answer this replaced could not say that, which is what
        // made a relocated artifact look like a different model.
        assert_eq!(
            model_artifact_digest(&a).unwrap(),
            model_artifact_digest(&b).unwrap()
        );

        // Different contents, same name shape — different identity.
        let c = dir.path().join("c.zt");
        write(&c, &[9u8, 9, 9, 9]);
        assert_ne!(
            model_artifact_digest(&a).unwrap(),
            model_artifact_digest(&c).unwrap()
        );

        // A non-artifact still goes down the legacy path rather than erroring.
        assert!(super::manifest_digest(dir.path()).unwrap().is_none());
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
}
