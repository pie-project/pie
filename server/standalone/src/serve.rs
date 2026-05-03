//! `pie --config <toml>` — the main serve path.
//!
//! Wires together the standalone's pieces in dependency order:
//!   1. Load + validate the user TOML.
//!   2. Build a tokio runtime (worker_threads pinned by user config).
//!   3. For each `[[model]]`:
//!      a. Create the cold-path `RpcServer` and spawn its dispatch loop.
//!      b. Start the [`EmbeddedDriver`] thread; block until the driver
//!         emits caps via the `ready_cb` callback.
//!   4. Translate (user TOML, handshakes) → [`pie::bootstrap::Config`]
//!      and call [`pie::bootstrap::bootstrap`]. The runtime now owns
//!      the websocket server + scheduler.
//!   5. Wait for SIGINT. On ctrl-c, close the cold-path channels and
//!      exit. (Driver threads run to OS-level reap for now; a graceful
//!      stop API lands post-M2 alongside the aux-IPC client.)

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};

use crate::bootstrap_translate::{self, ModelHandshake};
use crate::config::{self, DriverKind, DummyDriverOptions, PortableDriverOptions};
use crate::embedded_driver::{DriverOptions, EmbeddedDriver};
use crate::rpc_loop;

pub fn run(config_path: &Path) -> Result<()> {
    let user_cfg =
        config::Config::from_toml_file(config_path).context("loading TOML config")?;

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(user_cfg.runtime.worker_threads)
        .enable_all()
        .build()
        .context("building tokio runtime")?;

    runtime.block_on(run_async(user_cfg))
}

async fn run_async(user_cfg: config::Config) -> Result<()> {
    let mut handshakes = Vec::with_capacity(user_cfg.models.len());
    let mut drivers: Vec<EmbeddedDriver> = Vec::with_capacity(user_cfg.models.len());
    let mut rpc_servers: Vec<Arc<pie::device::RpcServer>> =
        Vec::with_capacity(user_cfg.models.len());
    let mut rpc_threads = Vec::with_capacity(user_cfg.models.len());

    for (i, m) in user_cfg.models.iter().enumerate() {
        let opts = match m.driver.kind {
            DriverKind::Portable => {
                let p: PortableDriverOptions = m
                    .driver
                    .options
                    .clone()
                    .try_into()
                    .map_err(|e| anyhow!("[model.driver.options] for {:?}: {e}", m.name))?;
                DriverOptions::Portable(p)
            }
            DriverKind::Dummy => {
                let d: DummyDriverOptions = m
                    .driver
                    .options
                    .clone()
                    .try_into()
                    .map_err(|e| anyhow!("[model.driver.options] for {:?}: {e}", m.name))?;
                DriverOptions::Dummy {
                    opts: d,
                    random_seed: m.driver.random_seed,
                    activation_dtype: m.driver.activation_dtype.clone(),
                }
            }
            other => {
                anyhow::bail!(
                    "model {:?}: driver type {other:?} is not implemented in \
                     server/standalone yet (cuda_native lands in M3/M4)",
                    m.name,
                );
            }
        };

        // v0: `hf_repo` is a local snapshot dir. HF download support is
        // a separate piece of work — for now, error clearly if it
        // doesn't exist locally.
        let snapshot_dir = PathBuf::from(&m.hf_repo);
        ensure!(
            snapshot_dir.is_dir(),
            "model {:?}: hf_repo {snapshot_dir:?} is not a local directory. \
             Standalone v0 does not download from HF — point hf_repo at a \
             local snapshot (e.g. via huggingface-cli download).",
            m.name,
        );

        // Cold-path RPC server first — its server_name is what the
        // runtime uses to connect via device::spawn.
        let rpc_server = Arc::new(
            pie::device::RpcServer::create()
                .map_err(|e| anyhow!("RpcServer::create for model {:?}: {e}", m.name))?,
        );
        let rpc_thread = rpc_loop::spawn(Arc::clone(&rpc_server));
        let rpc_server_name = rpc_server.server_name().to_owned();

        let driver = EmbeddedDriver::start(&opts, &snapshot_dir, i)
            .with_context(|| format!("starting driver for model {:?}", m.name))?;

        handshakes.push(ModelHandshake {
            rpc_server_name,
            caps: driver.caps.clone(),
        });
        drivers.push(driver);
        rpc_servers.push(rpc_server);
        rpc_threads.push(rpc_thread);
    }

    let boot_cfg = bootstrap_translate::build(&user_cfg, &handshakes)
        .context("translating to bootstrap::Config")?;

    let token = pie::bootstrap::bootstrap(boot_cfg)
        .await
        .map_err(|e| anyhow!("pie::bootstrap::bootstrap: {e}"))?;

    println!(
        "pie-standalone serving on {}:{} ({} model(s))",
        user_cfg.server.host,
        user_cfg.server.port,
        user_cfg.models.len(),
    );
    println!("internal token: {token}");
    println!("press Ctrl-C to shut down");

    tokio::signal::ctrl_c()
        .await
        .context("listening for SIGINT")?;
    println!("\nshutting down...");

    // Close the cold-path channels first — the runtime's pending RPCs
    // (if any) bail out cleanly.
    for s in &rpc_servers {
        s.close();
    }
    for t in rpc_threads {
        let _ = t.join();
    }

    // Signal each driver's serve loop to exit, then join the threads
    // so we exit only after the C++ side has had a chance to release
    // its shmem segments.
    for d in &drivers {
        d.request_stop();
    }
    for d in drivers {
        let rc = d.join();
        if rc != 0 {
            tracing::warn!("driver thread exited with rc={rc}");
        }
    }

    Ok(())
}
