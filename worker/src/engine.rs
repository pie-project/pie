//! `pie serve` core: boot drivers, wire RPC, hand off to the runtime,
//! and surface an [`EngineHandle`] the caller drives.
//!
//! Wires the standalone's pieces in dependency order:
//!   1. Translate user TOML to per-driver options.
//!   2. For each `[[model]]`, partition devices into DP groups; for
//!      each group spawn an [`EmbeddedDriver`] thread, attach an
//!      a unified `DriverChannel` (one channel per driver carries
//!   3. Translate the resulting handshakes → [`pie::bootstrap::Config`]
//!      and call [`pie::bootstrap::bootstrap`]. The runtime now owns
//!      the runtime services + scheduler; the worker dials into the
//!      gateway and serves `pie_worker_rpc::WorkerControl`.
//!   4. Caller decides what to do with the [`EngineHandle`]:
//!        * `pie serve`: [`EngineHandle::wait_then_shutdown`] blocks
//!          on SIGINT/SIGTERM/watchdog and tears down.
//!        * `pie serve --monitor`: TUI runs concurrently and calls
//!          [`EngineHandle::shutdown`] when the user quits.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use pie_controller_rpc::{ControlClient, Role, WorkerInfo};
use pie_ids::WorkerId;

use crate::config;
use crate::driver_ffi::Flavor;
use crate::embedded_driver::{DriverCapabilities, DriverOptions, EmbeddedDriver};
use crate::link::control::{self, ControlLink};
use crate::link::{gateway, topology};
use crate::preflight::{self, ResolvedFlavor};
use crate::translate::{self, GroupHandshake, ModelHandshake};
use crate::{client_server, lifecycle, weights};

pub use crate::link::topology::{Coordinator, TopologyMode, connect};
pub use crate::preflight::calculate_topology;

/// Embedded driver supervisor.
pub struct DriverHandle(EmbeddedDriver);

impl DriverHandle {
    /// Returns the driver's shmem region name, if any. `None` for
    /// embedded cuda/portable drivers (no shmem region opened).
    pub fn shmem_name(&self) -> Option<&str> {
        self.0.shmem_name.as_deref()
    }

    pub fn caps(&self) -> &DriverCapabilities {
        &self.0.caps
    }

    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }

    pub fn request_stop(&self) {
        self.0.request_stop();
    }

    pub fn join(self) -> i32 {
        self.0.join()
    }
}

/// Live engine — drivers, RPC dispatch threads, the bootstrapped
/// runtime token, and enough state to perform an orderly shutdown.
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
    /// Post-inversion (M3): the worker dials INTO the gateway(s) — distributed
    /// (real gateways from `--gateway`) or single-node (the in-proc gateway).
    /// The live dial-in links serving `WorkerControl`.
    GatewayLinks(Vec<gateway::GatewayLink>),
}

impl EdgeServer {
    /// The advertised URL: `ws://…` for the direct client server, or
    /// `gateway://addr[,…]` listing the gateway endpoint(s) the worker dialed
    /// into (the worker is not client-facing in distributed mode).
    fn url(&self) -> String {
        match self {
            EdgeServer::Standalone(h) => h.bound.clone(),
            EdgeServer::GatewayLinks(links) => {
                // The worker is not client-facing in distributed mode — the
                // gateway is. Report the gateway endpoint(s) it dialed into.
                if links.is_empty() {
                    "gateway://<none>".to_string()
                } else {
                    let addrs: Vec<&str> = links.iter().map(|l| l.addr.as_str()).collect();
                    format!("gateway://{}", addrs.join(","))
                }
            }
        }
    }

    fn abort(&self) {
        match self {
            EdgeServer::Standalone(h) => h.task.abort(),
            EdgeServer::GatewayLinks(links) => {
                for link in links {
                    link.abort();
                }
            }
        }
    }
}

pub struct EngineHandle {
    drivers: Vec<DriverHandle>,
    shmem_names: Vec<String>,
    edge_server: EdgeServer,
    /// Controller heartbeat/report/watch tasks. Empty when there is no control
    /// plane (single-node without the `single-node` feature).
    control_tasks: Vec<tokio::task::JoinHandle<()>>,
    /// Live control-plane state kept alive for the engine's lifetime: the dialed
    /// client (distributed) or the embedded controller handle + in-proc gateway
    /// task (single-node feature). `None` in gateway-free single-node.
    control_plane: ControlPlane,
    /// Bootstrapped engine internal auth token.
    pub token: String,
    /// Client endpoint this worker advertises: `ws://host:port` in single-node
    /// (direct client server, or the in-proc gateway), or `gateway://addr[,…]`
    /// in distributed (the gateway endpoint(s) the worker dialed into — clients
    /// hit the gateway, not the worker).
    pub url: String,
}

/// Live control-plane resources held for the engine's lifetime, by topology.
enum ControlPlane {
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

impl ControlPlane {
    /// The controller-minted worker id, if this worker registered.
    fn worker_id(&self) -> Option<WorkerId> {
        match self {
            ControlPlane::None => None,
            ControlPlane::Distributed { worker_id, .. } | ControlPlane::Embedded { worker_id } => {
                Some(*worker_id)
            }
        }
    }
}

impl EngineHandle {
    /// Block on SIGINT / SIGTERM / driver-watchdog, then run the
    /// shutdown sequence. The original `run_with_config` flow.
    pub async fn wait_then_shutdown(self) -> Result<()> {
        let shutdown_reason = tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => "SIGINT",
            _ = lifecycle::wait_for_sigterm() => "SIGTERM",
            reason = lifecycle::watchdog(&self.drivers) => reason,
        };
        eprintln!("\nshutting down ({shutdown_reason})...");
        self.shutdown();
        Ok(())
    }

    /// Tear down the engine without waiting for a signal. Used by the
    /// monitor TUI, which owns its own input loop and decides when to
    /// quit.
    pub fn shutdown(self) {
        self.edge_server.abort();
        for task in self.control_tasks {
            task.abort();
        }
        // Stop the in-proc gateway (single-node) and drop the control-plane
        // resources so the dialed control connection is closed (distributed) or
        // the embedded controller handle is released (single-node). The
        // controller then ages this worker out of routing on the next missed
        // report.
        tracing::info!(worker = ?self.control_plane.worker_id(), "leaving control plane");
        drop(self.control_plane);
        // Signal each driver's serve loop, wake any transport-side
        // waiters/recv loops, then join the threads.
        for d in &self.drivers {
            d.request_stop();
        }
        pie::driver::abort_all_driver_channels();
        for d in self.drivers {
            let rc = d.join();
            if rc != 0 {
                tracing::warn!("driver thread exited with rc={rc}");
            }
        }
        // Best-effort shmem cleanup — see `unlink_shmem`.
        for name in &self.shmem_names {
            lifecycle::unlink_shmem(name);
        }
    }
}

/// A running worker: the engine plus an async drain-and-stop. Returned by
/// [`run`] (daemon) and [`run_with`] (in-proc embed). The bin owns the runtime
/// (Model A) and drives [`shutdown`](WorkerHandle::shutdown) on signal.
pub struct WorkerHandle {
    engine: EngineHandle,
}

impl WorkerHandle {
    /// The client endpoint this worker advertises (`ws://…` in single-node, or
    /// the `gateway://…` endpoint(s) it dialed into in distributed mode).
    pub fn url(&self) -> &str {
        &self.engine.url
    }

    /// The bootstrapped engine auth token.
    pub fn token(&self) -> &str {
        &self.engine.token
    }

    /// Drain in-flight work and stop the engine (drivers, control loops, edge).
    pub async fn shutdown(self) {
        // `EngineHandle::shutdown` joins native driver threads (blocking), so
        // run it off the async runtime.
        let engine = self.engine;
        let _ = tokio::task::spawn_blocking(move || engine.shutdown()).await;
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
    let engine = start_engine(cfg, coordinator).await?;
    Ok(WorkerHandle { engine })
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
    Ok(WorkerHandle { engine })
}

struct StartupBanner {
    model: String,
    driver: String,
    device: String,
}

impl StartupBanner {
    fn from_config(cfg: &config::Config) -> Self {
        let model = match cfg.models.as_slice() {
            [m] => format!("{} ({})", m.name, m.hf_repo),
            models => format!("{} models", models.len()),
        };
        let driver = match cfg.models.as_slice() {
            [m] => m.driver.kind.as_str().to_string(),
            models => {
                let mut drivers = models
                    .iter()
                    .map(|m| m.driver.kind.as_str())
                    .collect::<Vec<_>>();
                drivers.sort_unstable();
                drivers.dedup();
                if drivers.len() == 1 {
                    drivers[0].to_string()
                } else {
                    "mixed".to_string()
                }
            }
        };
        let device = match cfg.models.as_slice() {
            [m] => {
                let device = m.driver.device.join(", ");
                if device.is_empty() {
                    "-".to_string()
                } else {
                    device
                }
            }
            models => {
                let count = models.iter().map(|m| m.driver.device.len()).sum::<usize>();
                if count == 0 {
                    "-".to_string()
                } else {
                    format!("{count} devices")
                }
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

/// Boot the drivers + bootstrap the runtime for `user_cfg`, returning the live
/// driver handles, the registration model/capabilities, the runtime auth token,
/// and the driver shmem region names. Shared by every engine entry point
/// ([`start_engine`] daemon/wheel, [`start_engine_embedded`] in-proc root).
async fn boot_engine(
    user_cfg: &config::Config,
) -> Result<(
    Vec<DriverHandle>,
    String,
    DriverCapabilities,
    String,
    Vec<String>,
)> {
    let mut handshakes: Vec<ModelHandshake> = Vec::with_capacity(user_cfg.models.len());
    let mut drivers: Vec<DriverHandle> = Vec::new();

    // Global device index. The runtime's `driver::spawn` returns
    // indices in call order; the driver-side shmem region is named
    // `/pie_shmem_g{driver_idx}` (`runtime/src/device.rs::shmem_name`).
    // Pass this counter as the driver's `group_id` so the names line
    // up across all models, including DP > 1.
    let mut next_global_driver_idx: usize = 0;

    for m in &user_cfg.models {
        let resolved = preflight::resolve_flavor(m.driver.kind, &m.name)?;

        // Determine TP/DP topology. For multi-DP, we spawn one driver
        // per group; each group gets its own snapshot+startup TOML.
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

        // Resolve snapshot once per model — every group serves the same
        // weights against the same on-disk path.
        let snapshot_dir = weights::resolve(&m.hf_repo)
            .with_context(|| format!("resolving hf_repo for model {:?}", m.name))?;

        let mut group_handshakes: Vec<GroupHandshake> = Vec::with_capacity(topology.len());

        for (group_idx, group) in topology.iter().enumerate() {
            let driver_idx = next_global_driver_idx;
            next_global_driver_idx += 1;

            let started = start_embedded_group(
                m,
                group_idx,
                group,
                flavor,
                &embedded_base_opts,
                &snapshot_dir,
                driver_idx,
                tp_degree,
            )?;
            group_handshakes.push(started.handshake);
            drivers.extend(started.drivers);
        }

        handshakes.push(ModelHandshake {
            groups: group_handshakes,
        });
    }

    let registration_caps = handshakes
        .first()
        .and_then(|model| model.groups.first())
        .map(|group| group.caps.clone())
        .context("no driver capabilities available for control-plane registration")?;
    let registration_model = user_cfg
        .models
        .first()
        .map(|model| model.name.clone())
        .context("no model configured for control-plane registration")?;

    let boot_cfg =
        translate::build(user_cfg, &handshakes).context("translating to bootstrap::Config")?;

    let boot = pie::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("pie::bootstrap::bootstrap: {e}"))?;
    let token = boot.token;

    let shmem_names: Vec<String> = drivers
        .iter()
        .filter_map(|d| d.shmem_name().map(|s| s.to_string()))
        .collect();

    Ok((
        drivers,
        registration_model,
        registration_caps,
        token,
        shmem_names,
    ))
}

/// Boot the engine + assemble the control/edge plane for the resolved topology
/// ([`Coordinator`]). The in-proc engine-embedding entry (Seam 1b) used by
/// `bin/worker` (via [`run`]) and the `pie-server-py` wheel (single-node
/// direct-WS).
pub async fn start_engine(
    user_cfg: config::Config,
    coordinator: Coordinator,
) -> Result<EngineHandle> {
    let (drivers, model, caps, token, shmem_names) = boot_engine(&user_cfg).await?;
    let (edge_server, control_tasks, control_plane, url) =
        assemble_control_and_edge(coordinator, &user_cfg, model, caps).await?;
    log_serving(&user_cfg, &url, &token);
    Ok(EngineHandle {
        drivers,
        shmem_names,
        url,
        edge_server,
        control_tasks,
        control_plane,
        token,
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
    let (drivers, model, caps, token, shmem_names) = boot_engine(&user_cfg).await?;
    let addr = topology::addr_from_host_port(&user_cfg.server.host, user_cfg.server.port);
    // A single-node-monolithic worker serves all stages; routing doesn't filter
    // by role yet, so Decode is an inert default (echo owns Role::Monolithic).
    let (edge_server, control_tasks, worker_id) =
        assemble_distributed(control, &gateways, Role::Decode, model, addr, caps).await?;
    let url = edge_server.url();
    log_serving(&user_cfg, &url, &token);
    Ok(EngineHandle {
        drivers,
        shmem_names,
        url,
        edge_server,
        control_tasks,
        control_plane: ControlPlane::Embedded { worker_id },
        token,
    })
}

/// Print the startup banner + token when `server.verbose` is set.
fn log_serving(cfg: &config::Config, url: &str, token: &str) {
    if cfg.server.verbose {
        eprintln!("{}", StartupBanner::from_config(cfg).render(url));
        eprintln!("internal token: {token}");
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
) -> Result<(
    EdgeServer,
    Vec<tokio::task::JoinHandle<()>>,
    ControlPlane,
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
            let (edge, control_tasks, worker_id) = assemble_distributed(
                client.clone(),
                &gateways,
                role,
                model,
                coordinator.control_addr.clone(),
                caps,
            )
            .await?;
            let url = edge.url();
            Ok((
                edge,
                control_tasks,
                ControlPlane::Distributed {
                    _client: client,
                    worker_id,
                },
                url,
            ))
        }
        TopologyMode::SingleNode => {
            // Gateway-free local inference: the worker terminates client
            // WebSockets itself and never registers, so the model name and
            // capabilities have no controller to be registered with.
            let _ = (model, caps);
            let listen = format!("{}:{}", user_cfg.server.host, user_cfg.server.port);
            let edge = EdgeServer::Standalone(
                client_server::spawn(&listen)
                    .await
                    .context("starting standalone client server")?,
            );
            let url = edge.url();
            Ok((edge, Vec::new(), ControlPlane::None, url))
        }
    }
}

/// Register the worker over `control`, spawn its three control loops, then dial
/// INTO each gateway, serving `WorkerControl` over the links. Generic over the
/// [`ControlLink`] backend so the daemon injects a dialed [`ControlClient`] and
/// the composition root (`bin/pie`) injects its in-proc `EmbeddedControl`.
///
/// `register` happens BEFORE dialing the gateways, so the worker presents its
/// controller-minted id on each gateway dial-in `register` (the join key for
/// `routing ∩ connected`).
async fn assemble_distributed<C: ControlLink>(
    control: C,
    gateways: &[String],
    role: Role,
    model: String,
    addr: String,
    caps: DriverCapabilities,
) -> Result<(EdgeServer, Vec<tokio::task::JoinHandle<()>>, WorkerId)> {
    let info = WorkerInfo {
        role,
        model,
        addr,
        capability: caps,
    };
    let worker_id = ControlLink::register_worker(&control, info)
        .await
        .context("registering worker with controller")?;
    let control_tasks = control::spawn_control_tasks(control, worker_id);

    let mut links = Vec::with_capacity(gateways.len());
    for gw in gateways {
        let link = gateway::connect_gateway(gw, worker_id)
            .await
            .with_context(|| format!("dialing gateway at {gw}"))?;
        links.push(link);
    }
    Ok((EdgeServer::GatewayLinks(links), control_tasks, worker_id))
}

struct StartedEmbeddedGroup {
    handshake: GroupHandshake,
    drivers: Vec<DriverHandle>,
}

fn start_embedded_group(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &DriverOptions,
    snapshot_dir: &Path,
    driver_idx: usize,
    tp_degree: usize,
) -> Result<StartedEmbeddedGroup> {
    let group_drivers = start_embedded_drivers(
        m,
        group_idx,
        group,
        flavor,
        base_opts,
        snapshot_dir,
        driver_idx,
        tp_degree,
    )?;

    let primary = group_drivers.first().ok_or_else(|| {
        anyhow!(
            "starting driver for model {:?} group {group_idx} returned no ranks",
            m.name,
        )
    })?;
    let caps = primary.caps.clone();
    if let Some(shmem_name) = primary.shmem_name.as_deref() {
        let channel =
            pie::driver::ShmemChannel::open(shmem_name, m.driver.effective_spin_budget_us())
                .with_context(|| {
                    format!(
                        "opening shmem channel for embedded driver ({}) group {group_idx}",
                        flavor.as_str(),
                    )
                })?;
        pie::driver::install_channel(driver_idx, Arc::new(channel));
    }
    let handshake = GroupHandshake { caps };
    let drivers = group_drivers.into_iter().map(DriverHandle).collect();

    Ok(StartedEmbeddedGroup { handshake, drivers })
}

fn start_embedded_drivers(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &DriverOptions,
    snapshot_dir: &Path,
    driver_idx: usize,
    tp_degree: usize,
) -> Result<Vec<EmbeddedDriver>> {
    let spin_budget_us = m.driver.effective_spin_budget_us();
    let use_inproc_polling_channel = m.driver.use_inproc_polling_channel();

    #[cfg(feature = "driver-cuda")]
    {
        if flavor == Flavor::Cuda && tp_degree > 1 {
            let rank_opts = cuda_rank_options(m, group_idx, group, base_opts)?;
            return EmbeddedDriver::start_cuda_tp_group(
                &rank_opts,
                snapshot_dir,
                driver_idx,
                use_inproc_polling_channel,
                spin_budget_us,
            )
            .with_context(|| {
                format!(
                    "starting cuda TP driver group for model {:?} group {group_idx}",
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

    Ok(vec![
        EmbeddedDriver::start(
            &opts,
            snapshot_dir,
            driver_idx,
            use_inproc_polling_channel,
            spin_budget_us,
        )
        .with_context(|| format!("starting driver for model {:?} group {group_idx}", m.name,))?,
    ])
}

fn embedded_opts_for_device(base_opts: &DriverOptions, device: String) -> DriverOptions {
    #[cfg(not(any(feature = "driver-portable", feature = "driver-cuda")))]
    let _ = &device;

    #[allow(unreachable_patterns)]
    match base_opts {
        #[cfg(feature = "driver-portable")]
        DriverOptions::Portable(opts) => {
            let mut opts = opts.clone();
            opts.device = device;
            DriverOptions::Portable(opts)
        }
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
    #[cfg(feature = "driver-portable")]
    if let DriverOptions::Portable(p) = options {
        p.verbose = verbose;
    }

    #[cfg(feature = "driver-cuda")]
    if let DriverOptions::CudaNative(opts) = options {
        opts.verbose = verbose;
    }

    #[cfg(not(any(feature = "driver-portable", feature = "driver-cuda")))]
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
    use super::StartupBanner;

    #[test]
    fn startup_banner_render_includes_public_startup_fields_only() {
        let banner = StartupBanner {
            model: "default (Qwen/Qwen3-0.6B)".to_string(),
            driver: "portable".to_string(),
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
}
