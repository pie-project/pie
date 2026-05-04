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

use std::sync::Arc;

use anyhow::{Context, Result, anyhow};

use crate::aux_ipc::AuxIpcClient;
use crate::bootstrap_translate::{self, GroupHandshake, ModelHandshake};
use crate::config;
use crate::driver_ffi::Flavor;
use crate::embedded_driver::EmbeddedDriver;
#[cfg(feature = "driver-cuda")]
use crate::embedded_driver::DriverOptions;
use crate::embedded_driver::DriverCapabilities;
use crate::hf;
use crate::python_resolve::{self, DriversConfig};
use crate::rpc_loop;
use crate::subprocess_driver::SubprocessDriver;

mod lifecycle;
mod topology;

pub use topology::calculate_topology;
use topology::ResolvedFlavor;

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
        println!("\nshutting down ({shutdown_reason})...");
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
/// from TOML, applies `--host` / `--port` / `--no-auth` / `--verbose`
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
    let base_master_port: u16 = 29500u16
        .saturating_add((std::process::id() % 1000) as u16);

    // Read `~/.pie/drivers.toml` once. Missing file → empty config;
    // resolve_python falls through to env vars / `which python3`.
    let drivers_config = DriversConfig::load()
        .context("loading ~/.pie/drivers.toml")?;

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

        if tp_degree > 1 {
            anyhow::bail!(
                "model {:?}: tensor_parallel_size={tp_degree} is not yet \
                 supported in server — only DP > 1 (one driver \
                 per replica, tp=1) works today. NCCL rendezvous for \
                 in-process TP lands in M4.5.",
                m.name,
            );
        }

        // Embedded options are typed; subprocess uses raw passthrough,
        // so it's only built when we know we're on the embedded path.
        let embedded_base_opts = match resolved {
            ResolvedFlavor::Embedded(f) => Some(topology::build_embedded_options(m, f)?),
            ResolvedFlavor::Subprocess(_) => None,
        };

        // Resolve snapshot once per model — every group serves the same
        // weights against the same on-disk path.
        let snapshot_dir = hf::resolve_or_download(&m.hf_repo)
            .await
            .with_context(|| format!("resolving hf_repo for model {:?}", m.name))?;

        let mut group_handshakes: Vec<GroupHandshake> = Vec::with_capacity(topology.len());

        let model_master_port = base_master_port.saturating_add((model_idx as u16) * 100);

        for (group_idx, group) in topology.iter().enumerate() {
            // For tp=1 the per-group device is just `device[group_idx]`.
            // For tp>1 (gated above), it would be `device[group[0]]`.
            let _device_for_group = m
                .driver
                .device
                .get(group[0])
                .ok_or_else(|| {
                    anyhow!(
                        "model {:?}: group {group_idx} references device \
                         index {} but only {} devices configured",
                        m.name,
                        group[0],
                        m.driver.device.len(),
                    )
                })?
                .clone();

            let device_idx = next_global_device_idx;
            next_global_device_idx += 1;

            match resolved {
                ResolvedFlavor::Embedded(flavor) => {
                    // Cuda needs the device pinned per group (`model.device`
                    // in its TOML); other flavors derive everything else from
                    // the snapshot dir alone.
                    let opts = match embedded_base_opts.as_ref().expect("embedded => Some") {
                        #[cfg(feature = "driver-cuda")]
                        DriverOptions::CudaNative { opts, hf_repo, .. } => {
                            DriverOptions::CudaNative {
                                opts: opts.clone(),
                                device: _device_for_group.clone(),
                                hf_repo: hf_repo.clone(),
                            }
                        }
                        other => other.clone(),
                    };

                    // Cold-path RPC server first; its server_name goes into
                    // the handshake bundle so bootstrap can wire one
                    // DeviceConfig per group.
                    let rpc_server = Arc::new(
                        pie::device::RpcServer::create()
                            .map_err(|e| anyhow!(
                                "RpcServer::create for model {:?} group {group_idx}: {e}",
                                m.name,
                            ))?,
                    );
                    let rpc_server_name = rpc_server.server_name().to_owned();

                    // Driver thread first: its `AuxServer` is constructed
                    // *before* ready_cb fires, so by the time `start()`
                    // returns the aux socket is accepting. Spawning the
                    // cold-path dispatcher before the driver would race
                    // the runtime's first call against an absent client.
                    let driver = EmbeddedDriver::start(&opts, &snapshot_dir, device_idx)
                        .with_context(|| format!(
                            "starting driver for model {:?} group {group_idx}",
                            m.name,
                        ))?;

                    // Connect the aux-IPC client only for drivers that listen
                    // (portable today). Cuda's `[aux_ipc]` listener and
                    // dummy's no-aux design both leave this `None` —
                    // `dispatch_copy` handles both cases (dummy: stub `()`;
                    // cuda: explicit error).
                    let aux_client: Option<Arc<AuxIpcClient>> = match flavor {
                        #[cfg(feature = "driver-portable")]
                        Flavor::Portable => Some(Arc::new(
                            AuxIpcClient::connect(driver.aux_socket_path.clone())
                                .with_context(|| format!(
                                    "connecting aux-ipc socket for model {:?} group {group_idx}",
                                    m.name,
                                ))?,
                        )),
                        #[allow(unreachable_patterns)]
                        _ => None,
                    };

                    let rpc_thread = rpc_loop::spawn(flavor, Arc::clone(&rpc_server), aux_client);

                    group_handshakes.push(GroupHandshake {
                        rpc_server_name,
                        caps: driver.caps.clone(),
                    });
                    drivers.push(DriverHandle::Embedded(driver));
                    rpc_servers.push(rpc_server);
                    rpc_threads.push(rpc_thread);
                }
                ResolvedFlavor::Subprocess(sub_flavor) => {
                    let resolved = python_resolve::resolve_python(
                        sub_flavor,
                        &m.driver.options,
                        Some(&drivers_config),
                    )
                    .with_context(|| format!(
                        "resolving Python interpreter for model {:?}",
                        m.name,
                    ))?;
                    if user_cfg.server.verbose {
                        println!(
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
                        &snapshot_dir,
                        device_idx,
                        model_master_port,
                    )
                    .with_context(|| format!(
                        "starting subprocess driver ({}) for model {:?} group {group_idx}",
                        sub_flavor.as_str(),
                        m.name,
                    ))?;

                    group_handshakes.push(GroupHandshake {
                        rpc_server_name: driver.server_name.clone(),
                        caps: driver.caps.clone(),
                    });
                    drivers.push(DriverHandle::Subprocess(driver));
                    // No rpc_server / rpc_thread on the standalone side —
                    // the Python launcher hosts its own RpcServer.
                }
            }
        }

        handshakes.push(ModelHandshake { groups: group_handshakes });
    }

    let boot_cfg = bootstrap_translate::build(&user_cfg, &handshakes)
        .context("translating to bootstrap::Config")?;

    let token = pie::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("pie::bootstrap::bootstrap: {e}"))?;

    println!(
        "pie-server serving on {}:{} ({} model(s))",
        user_cfg.server.host,
        user_cfg.server.port,
        user_cfg.models.len(),
    );
    println!("internal token: {token}");
    println!("press Ctrl-C to shut down");

    let shmem_names: Vec<String> = drivers
        .iter()
        .map(|d| d.shmem_name().to_string())
        .collect();

    Ok(EngineHandle {
        drivers,
        rpc_servers,
        rpc_threads,
        shmem_names,
        token,
        url: format!("ws://{}:{}", user_cfg.server.host, user_cfg.server.port),
    })
}
