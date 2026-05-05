//! `pie serve` core: boot drivers, wire RPC, hand off to the runtime,
//! and surface an [`EngineHandle`] the caller drives.
//!
//! Wires the standalone's pieces in dependency order:
//!   1. Translate user TOML to per-driver options.
//!   2. For each `[[model]]`, partition devices into DP groups; for
//!      each group spawn an [`EmbeddedDriver`] thread, attach an
//!      [`AuxIpcClient`] (portable today) + a cold-path RPC dispatcher.
//!   3. Translate the resulting handshakes → [`pie::bootstrap::Config`]
//!      and call [`pie::bootstrap::bootstrap`]. The runtime now owns
//!      the websocket server + scheduler.
//!   4. Caller decides what to do with the [`EngineHandle`]:
//!        * `pie serve`: [`EngineHandle::wait_then_shutdown`] blocks
//!          on SIGINT/SIGTERM/watchdog and tears down.
//!        * `pie serve --monitor`: TUI runs concurrently and calls
//!          [`EngineHandle::shutdown`] when the user quits.

use std::path::Path;
use std::sync::Arc;
use std::thread::JoinHandle;

use anyhow::{Context, Result, anyhow};

use crate::aux_ipc::AuxIpcClient;
use crate::bootstrap_translate::{self, GroupHandshake, ModelHandshake};
use crate::config;
use crate::driver_ffi::Flavor;
use crate::embedded_driver::{DriverCapabilities, DriverOptions, EmbeddedDriver};
use crate::hf;
use crate::python_resolve::{self, DriversConfig};
use crate::rpc_loop;
use crate::subprocess_driver::SubprocessDriver;

mod lifecycle;
mod topology;

use topology::ResolvedFlavor;
pub use topology::calculate_topology;

/// Heterogeneous driver supervisor — embedded threads (C++/Rust static
/// libs) and out-of-process Python subprocesses share a unified
/// lifecycle surface so [`EngineHandle`] can manage them in one list.
pub enum DriverHandle {
    Embedded(EmbeddedDriver),
    Subprocess(SubprocessDriver),
}

impl DriverHandle {
    pub fn shmem_name(&self) -> &str {
        match self {
            DriverHandle::Embedded(d) => &d.shmem_name,
            DriverHandle::Subprocess(d) => &d.shmem_name,
        }
    }

    pub fn caps(&self) -> &DriverCapabilities {
        match self {
            DriverHandle::Embedded(d) => &d.caps,
            DriverHandle::Subprocess(d) => &d.caps,
        }
    }

    pub fn is_finished(&self) -> bool {
        match self {
            DriverHandle::Embedded(d) => d.is_finished(),
            DriverHandle::Subprocess(d) => d.is_finished(),
        }
    }

    pub fn request_stop(&self) {
        match self {
            DriverHandle::Embedded(d) => d.request_stop(),
            DriverHandle::Subprocess(d) => d.request_stop(),
        }
    }

    pub fn join(self) -> i32 {
        match self {
            DriverHandle::Embedded(d) => d.join(),
            DriverHandle::Subprocess(d) => d.join(),
        }
    }
}

/// Live engine — drivers, RPC dispatch threads, the bootstrapped
/// runtime token, and enough state to perform an orderly shutdown.
/// Returned from [`start_engine`]; consumed by either
/// [`EngineHandle::wait_then_shutdown`] (the `pie serve` path) or
/// [`EngineHandle::shutdown`] (the `pie serve --monitor` path, where
/// the TUI owns the wait loop).
pub struct EngineHandle {
    drivers: Vec<DriverHandle>,
    /// Cold-path RPC servers — only populated for embedded drivers.
    /// Subprocess drivers host their own `RpcServer` inside the Python
    /// launcher (via the `pie-rpc` wheel), so the standalone has
    /// nothing to spawn on this side.
    rpc_servers: Vec<Arc<pie::device::RpcServer>>,
    rpc_threads: Vec<std::thread::JoinHandle<()>>,
    shmem_names: Vec<String>,
    /// Bootstrapped engine's WS auth token — handed to the monitor
    /// provider so it can `auth_by_token`.
    pub token: String,
    /// `ws://host:port` the engine is listening on.
    pub url: String,
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
        // Close cold-path channels first so any in-flight RPCs bail.
        for s in &self.rpc_servers {
            s.close();
        }
        for t in self.rpc_threads {
            let _ = t.join();
        }
        // Signal each driver's serve loop, then join the threads.
        for d in &self.drivers {
            d.request_stop();
        }
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
pub fn run_with_config(user_cfg: config::Config) -> Result<()> {
    // Best-effort install of the Python WASM runtime tarball.
    // Python inferlets fail to instantiate without
    // `$PIE_HOME/py-runtime/shared/componentize-py-runtime.wasm`;
    // pre-fetching here mirrors `pie/src/pie/server.py::Server.__aenter__`.
    // Failures (offline, registry down) just log a warning — the engine
    // still boots, and non-Python inferlets work normally.
    crate::py_runtime::ensure_installed_best_effort();

    let runtime = build_runtime(&user_cfg)?;

    runtime.block_on(async move {
        let engine = start_engine(user_cfg).await?;
        engine.wait_then_shutdown().await
    })
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
pub async fn start_engine(user_cfg: config::Config) -> Result<EngineHandle> {
    let listener = pie::server::bind(&user_cfg.server.host, user_cfg.server.port).await?;
    let mut handshakes: Vec<ModelHandshake> = Vec::with_capacity(user_cfg.models.len());
    let mut drivers: Vec<DriverHandle> = Vec::new();
    let mut rpc_servers: Vec<Arc<pie::device::RpcServer>> = Vec::new();
    let mut rpc_threads = Vec::new();

    // Global device index. The runtime's `device::spawn` returns
    // indices in call order; the driver-side shmem region is named
    // `/pie_shmem_g{device_idx}` (`runtime/src/device.rs::shmem_name`).
    // Pass this counter as the driver's `group_id` so the names line
    // up across all models, including DP > 1.
    let mut next_global_device_idx: usize = 0;

    // Per-model master-port assignment for the Python launchers'
    // torch.distributed FileStore rendezvous. Mirrors the legacy
    // Python server's `base_master_port + model_idx * 100` stride.
    // PID-mod-1000 spreads concurrent `pie serve` invocations across
    // the port space without pulling in `rand`.
    let base_master_port: u16 = 29500u16.saturating_add((std::process::id() % 1000) as u16);

    // Read `~/.pie/drivers.toml` once. Missing file → empty config;
    // resolve_python falls through to env vars / `which python3`.
    let drivers_config = DriversConfig::load().context("loading ~/.pie/drivers.toml")?;

    for (model_idx, m) in user_cfg.models.iter().enumerate() {
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
                ResolvedFlavor::Subprocess(_) => {}
                _ => anyhow::bail!(
                    "model {:?}: tensor_parallel_size={tp_degree} is only \
                     supported for cuda_native and Python subprocess drivers",
                    m.name,
                ),
            }
        }

        // Embedded options are typed; subprocess uses raw passthrough,
        // so it's only built when we know we're on the embedded path.
        let mut embedded_base_opts = match resolved {
            ResolvedFlavor::Embedded(f) => Some(topology::build_embedded_options(m, f)?),
            ResolvedFlavor::Subprocess(_) => None,
        };
        apply_embedded_verbose(&mut embedded_base_opts, user_cfg.server.verbose);

        // Resolve snapshot once per model — every group serves the same
        // weights against the same on-disk path.
        let snapshot_dir = hf::resolve_or_download(&m.hf_repo)
            .await
            .with_context(|| format!("resolving hf_repo for model {:?}", m.name))?;

        let mut group_handshakes: Vec<GroupHandshake> = Vec::with_capacity(topology.len());

        let model_master_port = base_master_port.saturating_add((model_idx as u16) * 100);

        for (group_idx, group) in topology.iter().enumerate() {
            let device_idx = next_global_device_idx;
            next_global_device_idx += 1;

            match resolved {
                ResolvedFlavor::Embedded(flavor) => {
                    let started = start_embedded_group(
                        m,
                        group_idx,
                        group,
                        flavor,
                        embedded_base_opts.as_ref().expect("embedded => Some"),
                        &snapshot_dir,
                        device_idx,
                        tp_degree,
                    )?;
                    group_handshakes.push(started.handshake);
                    drivers.extend(started.drivers);
                    rpc_servers.push(started.rpc_server);
                    rpc_threads.push(started.rpc_thread);
                }
                ResolvedFlavor::Subprocess(sub_flavor) => {
                    let group_devices = group
                        .iter()
                        .map(|&idx| {
                            m.driver.device.get(idx).cloned().ok_or_else(|| {
                                anyhow!(
                                    "model {:?}: group {group_idx} references device index {} but only {} devices configured",
                                    m.name,
                                    idx,
                                    m.driver.device.len(),
                                )
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let group_master_port =
                        model_master_port.saturating_add((group_idx as u16) * 10);
                    let started = start_subprocess_group(
                        m,
                        group_idx,
                        sub_flavor,
                        &drivers_config,
                        &snapshot_dir,
                        device_idx,
                        &group_devices,
                        tp_degree,
                        group_master_port,
                        user_cfg.server.verbose,
                    )?;
                    group_handshakes.push(started.handshake);
                    drivers.push(started.driver);
                }
            }
        }

        handshakes.push(ModelHandshake {
            groups: group_handshakes,
        });
    }

    let boot_cfg = bootstrap_translate::build(&user_cfg, &handshakes)
        .context("translating to bootstrap::Config")?;

    let boot = pie::bootstrap::bootstrap_with_listener(boot_cfg, listener)
        .await
        .map_err(|e| anyhow!("pie::bootstrap::bootstrap: {e}"))?;
    let bound_port = boot.port;
    let token = boot.token;

    if user_cfg.server.verbose {
        eprintln!(
            "pie-server serving on {}:{} ({} model(s))",
            user_cfg.server.host,
            bound_port,
            user_cfg.models.len(),
        );
        eprintln!("internal token: {token}");
        eprintln!("press Ctrl-C to shut down");
    }

    let shmem_names: Vec<String> = drivers.iter().map(|d| d.shmem_name().to_string()).collect();

    Ok(EngineHandle {
        drivers,
        rpc_servers,
        rpc_threads,
        shmem_names,
        token,
        url: format!("ws://{}:{}", user_cfg.server.host, bound_port),
    })
}

struct StartedEmbeddedGroup {
    handshake: GroupHandshake,
    drivers: Vec<DriverHandle>,
    rpc_server: Arc<pie::device::RpcServer>,
    rpc_thread: JoinHandle<()>,
}

struct StartedSubprocessGroup {
    handshake: GroupHandshake,
    driver: DriverHandle,
}

fn start_embedded_group(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &DriverOptions,
    snapshot_dir: &Path,
    device_idx: usize,
    tp_degree: usize,
) -> Result<StartedEmbeddedGroup> {
    let rpc_server = Arc::new(pie::device::RpcServer::create().map_err(|e| {
        anyhow!(
            "RpcServer::create for model {:?} group {group_idx}: {e}",
            m.name,
        )
    })?);
    let rpc_server_name = rpc_server.server_name().to_owned();

    let group_drivers = start_embedded_drivers(
        m,
        group_idx,
        group,
        flavor,
        base_opts,
        snapshot_dir,
        device_idx,
        tp_degree,
    )?;

    let primary = group_drivers.first().ok_or_else(|| {
        anyhow!(
            "starting driver for model {:?} group {group_idx} returned no ranks",
            m.name,
        )
    })?;
    let primary_aux_socket = primary.aux_socket_path.clone();
    let caps = primary.caps.clone();

    #[cfg(not(feature = "driver-portable"))]
    let _ = primary_aux_socket;

    let aux_client: Option<Arc<AuxIpcClient>> = match flavor {
        #[cfg(feature = "driver-portable")]
        Flavor::Portable => Some(Arc::new(
            AuxIpcClient::connect(primary_aux_socket).with_context(|| {
                format!(
                    "connecting aux-ipc socket for model {:?} group {group_idx}",
                    m.name,
                )
            })?,
        )),
        #[allow(unreachable_patterns)]
        _ => None,
    };

    let rpc_thread = rpc_loop::spawn(flavor, Arc::clone(&rpc_server), aux_client);
    let handshake = GroupHandshake {
        rpc_server_name,
        caps,
    };
    let drivers = group_drivers
        .into_iter()
        .map(DriverHandle::Embedded)
        .collect();

    Ok(StartedEmbeddedGroup {
        handshake,
        drivers,
        rpc_server,
        rpc_thread,
    })
}

fn start_embedded_drivers(
    m: &config::ModelConfig,
    group_idx: usize,
    group: &[usize],
    flavor: Flavor,
    base_opts: &DriverOptions,
    snapshot_dir: &Path,
    device_idx: usize,
    tp_degree: usize,
) -> Result<Vec<EmbeddedDriver>> {
    #[cfg(feature = "driver-cuda")]
    {
        if flavor == Flavor::Cuda && tp_degree > 1 {
            let rank_opts = cuda_rank_options(m, group_idx, group, base_opts)?;
            return EmbeddedDriver::start_cuda_tp_group(&rank_opts, snapshot_dir, device_idx)
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

    let first_device_idx = group.first().copied().ok_or_else(|| {
        anyhow!(
            "model {:?}: group {group_idx} is empty; topology calculation produced no ranks",
            m.name,
        )
    })?;
    let device = group_device(m, group_idx, first_device_idx)?;
    let opts = embedded_opts_for_device(base_opts, device);

    Ok(vec![
        EmbeddedDriver::start(&opts, snapshot_dir, device_idx).with_context(|| {
            format!("starting driver for model {:?} group {group_idx}", m.name,)
        })?,
    ])
}

fn embedded_opts_for_device(base_opts: &DriverOptions, device: String) -> DriverOptions {
    #[cfg(not(feature = "driver-cuda"))]
    let _ = &device;
    #[allow(unreachable_patterns)]
    match base_opts {
        #[cfg(feature = "driver-cuda")]
        DriverOptions::CudaNative { opts, hf_repo, .. } => DriverOptions::CudaNative {
            opts: opts.clone(),
            device,
            hf_repo: hf_repo.clone(),
        },
        other => other.clone(),
    }
}

fn apply_embedded_verbose(options: &mut Option<DriverOptions>, verbose: bool) {
    #[cfg(feature = "driver-portable")]
    if let Some(DriverOptions::Portable(p)) = options.as_mut() {
        p.verbose = verbose;
    }

    #[cfg(feature = "driver-cuda")]
    if let Some(DriverOptions::CudaNative { opts, .. }) = options.as_mut() {
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
    for &rank_device_idx in group {
        let rank_device = group_device(m, group_idx, rank_device_idx)?;
        match base_opts {
            DriverOptions::CudaNative { opts, hf_repo, .. } => {
                rank_opts.push(DriverOptions::CudaNative {
                    opts: opts.clone(),
                    device: rank_device,
                    hf_repo: hf_repo.clone(),
                });
            }
            _ => unreachable!("flavor checked before building cuda rank options"),
        }
    }
    Ok(rank_opts)
}

fn group_device(m: &config::ModelConfig, group_idx: usize, device_idx: usize) -> Result<String> {
    m.driver
        .device
        .get(device_idx)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "model {:?}: group {group_idx} references device index {} but only {} devices configured",
                m.name,
                device_idx,
                m.driver.device.len(),
            )
        })
}

fn start_subprocess_group(
    m: &config::ModelConfig,
    group_idx: usize,
    sub_flavor: crate::subprocess_driver::SubprocessFlavor,
    drivers_config: &DriversConfig,
    snapshot_dir: &Path,
    device_idx: usize,
    devices: &[String],
    tp_degree: usize,
    master_port: u16,
    verbose: bool,
) -> Result<StartedSubprocessGroup> {
    let resolved =
        python_resolve::resolve_python(sub_flavor, &m.driver.options, Some(drivers_config))
            .with_context(|| format!("resolving Python interpreter for model {:?}", m.name,))?;
    if verbose {
        eprintln!(
            "  [{}] python: {} (from {})",
            m.name,
            resolved.path.display(),
            resolved.source,
        );
    }

    let driver = SubprocessDriver::start(
        sub_flavor,
        &resolved.path,
        m,
        snapshot_dir,
        device_idx,
        devices,
        tp_degree,
        master_port,
    )
    .with_context(|| {
        format!(
            "starting subprocess driver ({}) for model {:?} group {group_idx}",
            sub_flavor.as_str(),
            m.name,
        )
    })?;

    let handshake = GroupHandshake {
        rpc_server_name: driver.server_name.clone(),
        caps: driver.caps.clone(),
    };
    Ok(StartedSubprocessGroup {
        handshake,
        driver: DriverHandle::Subprocess(driver),
    })
}
