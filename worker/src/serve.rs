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
//!      the runtime services + scheduler; the worker exposes edge-rpc
//!      (`WorkerSessionApi`) for the gateway.
//!   4. Caller decides what to do with the [`EngineHandle`]:
//!        * `pie serve`: [`EngineHandle::wait_then_shutdown`] blocks
//!          on SIGINT/SIGTERM/watchdog and tears down.
//!        * `pie serve --monitor`: TUI runs concurrently and calls
//!          [`EngineHandle::shutdown`] when the user quits.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use pie_control::ControlClient;
use pie_schema::control::{WorkerId, WorkerInfo};

use crate::bootstrap_translate::{self, GroupHandshake, ModelHandshake};
use crate::config;
use crate::driver_ffi::Flavor;
use crate::embedded_driver::{DriverCapabilities, DriverOptions, EmbeddedDriver};
use crate::hf;
use control::WorkerControl;

#[cfg(not(feature = "single-node"))]
mod client_server;
mod control;
pub mod coordination;
mod edge_session;
mod lifecycle;
#[cfg(feature = "single-node")]
mod single_node;
mod topology;

pub use coordination::{CoordinationArgs, Coordinator, TopologyMode};

use topology::ResolvedFlavor;
pub use topology::calculate_topology;

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
/// The client-facing server, selected by topology: a direct WebSocket terminator
/// in single-node (gateway-free local inference), or the tarpc edge-rpc a gateway
/// proxies in distributed mode.
enum EdgeServer {
    #[cfg(not(feature = "single-node"))]
    Standalone(client_server::ClientServerHandle),
    Distributed(edge_session::EdgeSessionServerHandle),
}

impl EdgeServer {
    /// The advertised URL: `ws://…` for the direct client server, `edge://…` for
    /// the gateway-fronted tarpc endpoint.
    fn url(&self) -> String {
        match self {
            #[cfg(not(feature = "single-node"))]
            EdgeServer::Standalone(h) => h.bound.clone(),
            EdgeServer::Distributed(h) => h.bound.replacen("tcp://", "edge://", 1),
        }
    }

    fn abort(&self) {
        match self {
            #[cfg(not(feature = "single-node"))]
            EdgeServer::Standalone(h) => h.task.abort(),
            EdgeServer::Distributed(h) => h.task.abort(),
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
    /// (direct client server, or the in-proc gateway), `edge://host:port` in
    /// distributed (gateway-fronted).
    pub url: String,
}

/// Live control-plane resources held for the engine's lifetime, by topology.
enum ControlPlane {
    /// No control plane (single-node, default build): the worker terminates
    /// clients directly and never registers.
    #[cfg(not(feature = "single-node"))]
    None,
    /// Distributed: the dialed control client (its dispatch task) stays alive
    /// until shutdown, when dropping it closes the connection so the controller
    /// ages this worker out of routing.
    Distributed {
        _client: ControlClient,
        worker_id: WorkerId,
    },
    /// Single-node embed: the controller actor `Handle` and the in-proc gateway
    /// task both live in this process and are torn down on shutdown.
    #[cfg(feature = "single-node")]
    Embedded {
        _handle: pie_controller::Handle,
        worker_id: WorkerId,
        gateway_task: tokio::task::JoinHandle<()>,
    },
}

impl ControlPlane {
    /// The controller-minted worker id, if this worker registered.
    fn worker_id(&self) -> Option<WorkerId> {
        match self {
            #[cfg(not(feature = "single-node"))]
            ControlPlane::None => None,
            ControlPlane::Distributed { worker_id, .. } => Some(*worker_id),
            #[cfg(feature = "single-node")]
            ControlPlane::Embedded { worker_id, .. } => Some(*worker_id),
        }
    }

    /// Stop any owned background tasks (the in-proc gateway). Dropping `self`
    /// afterwards releases the client / controller handle.
    fn abort(&self) {
        #[cfg(feature = "single-node")]
        if let ControlPlane::Embedded { gateway_task, .. } = self {
            gateway_task.abort();
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
        self.control_plane.abort();
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

/// Boot the engine from an already-loaded + validated config. The CLI
/// layer in [`crate::cli::serve_cmd`] is the only caller; it loads
/// from TOML, applies `--host` / `--port` / `--no-auth` / `--debug`
/// / `--no-snapshot` overrides, then invokes us.
pub fn run_with_config(user_cfg: config::Config, coordinator: Coordinator) -> Result<()> {
    // Best-effort install of the Python WASM runtime tarball.
    // Python inferlets fail to instantiate without
    // `$PIE_HOME/py-runtime/shared/componentize-py-runtime.wasm`;
    // pre-fetching here mirrors `pie/src/pie/server.py::Server.__aenter__`.
    // Failures (offline, registry down) just log a warning — the engine
    // still boots, and non-Python inferlets work normally.
    crate::py_runtime::ensure_installed_best_effort();

    let runtime = build_runtime(&user_cfg)?;
    let banner = StartupBanner::from_config(&user_cfg);
    let verbose = user_cfg.server.verbose;

    runtime.block_on(async move {
        let engine = start_engine(user_cfg, coordinator).await?;
        eprintln!("{}", banner.render(&engine.url));
        if verbose {
            eprintln!("internal token: {}", engine.token);
            eprintln!("press Ctrl-C to shut down");
        }
        engine.wait_then_shutdown().await
    })
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

/// Boot drivers + RPC + runtime bootstrap; return an [`EngineHandle`]
/// the caller can drive (wait-and-shutdown for plain serve, or hand
/// off to a TUI for the monitor path).
pub async fn start_engine(
    user_cfg: config::Config,
    coordinator: Coordinator,
) -> Result<EngineHandle> {
    let mut handshakes: Vec<ModelHandshake> = Vec::with_capacity(user_cfg.models.len());
    let mut drivers: Vec<DriverHandle> = Vec::new();

    // Global device index. The runtime's `driver::spawn` returns
    // indices in call order; the driver-side shmem region is named
    // `/pie_shmem_g{driver_idx}` (`runtime/src/device.rs::shmem_name`).
    // Pass this counter as the driver's `group_id` so the names line
    // up across all models, including DP > 1.
    let mut next_global_driver_idx: usize = 0;

    for m in &user_cfg.models {
        let resolved = topology::resolve_flavor(m.driver.kind, &m.name)?;

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
        let mut embedded_base_opts = topology::build_embedded_options(m, flavor)?;
        apply_embedded_verbose(&mut embedded_base_opts, user_cfg.server.verbose);

        // Resolve snapshot once per model — every group serves the same
        // weights against the same on-disk path.
        let snapshot_dir = hf::resolve_or_download(&m.hf_repo)
            .await
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

    let boot_cfg = bootstrap_translate::build(&user_cfg, &handshakes)
        .context("translating to bootstrap::Config")?;

    let boot = pie::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("pie::bootstrap::bootstrap: {e}"))?;
    let token = boot.token;

    // Build the client-facing edge + control plane for this topology, now that
    // the runtime is up and driver capabilities are known: dial + register
    // (distributed), embed the controller + run the in-proc gateway (single-node
    // feature), or terminate clients directly (single-node default build).
    let (edge_server, control_tasks, control_plane, url) = assemble_control_and_edge(
        coordinator,
        &user_cfg,
        registration_model,
        registration_caps,
    )
    .await?;

    if user_cfg.server.verbose {
        eprintln!(
            "pie-worker serving on {} ({} model(s))",
            url,
            user_cfg.models.len(),
        );
        eprintln!("internal token: {token}");
        eprintln!("press Ctrl-C to shut down");
    }

    let shmem_names: Vec<String> = drivers
        .iter()
        .filter_map(|d| d.shmem_name().map(|s| s.to_string()))
        .collect();

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

/// Build the client-facing edge server + control plane for the resolved
/// topology, after the runtime is bootstrapped and driver capabilities are
/// known. Returns the edge server, the worker's control-loop tasks, the live
/// control-plane resources to hold for the engine's lifetime, and the URL to
/// advertise.
///
/// - **distributed:** bind the worker's edge-rpc, dial the controller, register
///   (now that it can serve), and spawn the heartbeat/report/watch loops. A
///   remote gateway proxies client sessions to the edge-rpc.
/// - **single-node, `single-node` feature:** delegate to [`single_node::assemble`]
///   — embed the controller in-proc, run the gateway in this process, and bind
///   the worker's edge-rpc on a loopback port the in-proc gateway dials.
/// - **single-node, default build:** terminate client WebSockets directly; no
///   control plane.
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
        TopologyMode::Distributed { role, controller } => {
            let endpoint = coordinator.control_addr.clone();
            let edge = EdgeServer::Distributed(
                edge_session::spawn(&endpoint)
                    .await
                    .context("starting worker edge-rpc server")?,
            );
            let client = control::dial_controller(&controller)
                .await
                .with_context(|| format!("dialing controller at {controller}"))?;
            let info = WorkerInfo {
                role,
                model,
                addr: endpoint,
                capability: caps,
            };
            let worker_id = WorkerControl::register_worker(&client, info)
                .await
                .context("registering worker with controller")?;
            let control_tasks = control::spawn_control_tasks(client.clone(), worker_id);
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
            #[cfg(feature = "single-node")]
            {
                single_node::assemble(user_cfg, model, caps).await
            }
            #[cfg(not(feature = "single-node"))]
            {
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
