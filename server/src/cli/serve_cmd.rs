//! `pie serve` — boot the engine + serve.
//!
//! Mirrors `pie/src/pie_cli/commands/serve.py`. Loads config from
//! `--config` (or the default `~/.pie/config.toml`), applies CLI flag
//! overrides, then hands off to [`crate::serve::run_with_config`].

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Args;

use crate::cli::monitor;
use crate::cli::monitor::data::DisplayConfig;
use crate::{config, paths, serve};

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Path to TOML configuration file. Defaults to `~/.pie/config.toml`.
    #[arg(short = 'c', long)]
    pub config: Option<PathBuf>,

    /// Override `[server].host`.
    #[arg(long)]
    pub host: Option<String>,

    /// Override `[server].port`.
    #[arg(long)]
    pub port: Option<u16>,

    /// Disable authentication for this run (overrides `[auth].enabled`).
    #[arg(long)]
    pub no_auth: bool,

    /// Show engine, driver, and server diagnostics.
    #[arg(long)]
    pub debug: bool,

    /// Disable the host-side Python snapshot optimization for this run
    /// (overrides `[server].python_snapshot`).
    #[arg(long)]
    pub no_snapshot: bool,

    /// Launch the live TUI monitor alongside the engine. Quits with
    /// `q` / `Esc`; engine shuts down on TUI exit.
    #[arg(short = 'm', long)]
    pub monitor: bool,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let config_path = args.config.unwrap_or_else(paths::default_config_path);

    let mut cfg = config::Config::from_toml_file(&config_path)
        .with_context(|| format!("loading TOML config from {config_path:?}"))?;

    // Apply overrides. Match the Python `load_config` precedence:
    // CLI > TOML > dataclass defaults.
    if let Some(h) = args.host {
        cfg.server.host = h;
    }
    if let Some(p) = args.port {
        cfg.server.port = p;
    }
    if args.no_auth {
        cfg.auth.enabled = false;
    }
    if args.debug {
        cfg.server.verbose = true;
    }
    if args.no_snapshot {
        cfg.server.python_snapshot = false;
    }
    if args.monitor {
        run_with_monitor(cfg)
    } else {
        serve::run_with_config(cfg)
    }
}

/// Boot the engine + run the TUI in the same process. Mirrors
/// `pie/src/pie_cli/commands/serve.py`'s `--monitor` branch.
fn run_with_monitor(cfg: config::Config) -> Result<()> {
    crate::py_runtime::ensure_installed_best_effort();

    let display_cfg = display_config_from(&cfg)?;
    let runtime = serve::build_runtime(&cfg)?;

    let engine = runtime
        .block_on(async { serve::start_engine(cfg).await })
        .context("starting engine for monitor")?;

    // The TUI takes over the terminal; capture token + url before
    // moving the engine into the wait task.
    let url = engine.url.clone();
    let token = engine.token.clone();

    // Run the TUI on the calling thread (ratatui owns stdin/stdout).
    // Provider polling lives on the engine's runtime so it shares
    // worker threads with the engine itself.
    let tui_result = monitor::run_tui(runtime.handle(), url, token, display_cfg);

    // TUI exited (user quit, or it errored). Either way, shut the
    // engine down cleanly. We block on the runtime's shutdown so all
    // driver threads finish their teardown before we return.
    let shutdown_handle = runtime.spawn_blocking(move || engine.shutdown());
    let _ = runtime.block_on(shutdown_handle);

    tui_result
}

/// Project a [`config::Config`] into the trim subset the TUI's
/// configuration panel renders. Falls through cleanly when the
/// config has no models (the validate step would have rejected
/// that, but we return an explicit error for paranoia).
fn display_config_from(cfg: &config::Config) -> Result<DisplayConfig> {
    let m = cfg
        .models
        .first()
        .ok_or_else(|| anyhow!("no [[model]] sections to monitor"))?;
    Ok(DisplayConfig {
        host: cfg.server.host.clone(),
        port: cfg.server.port,
        auth_enabled: cfg.auth.enabled,
        hf_repo: m.hf_repo.clone(),
        device: m.driver.device.clone(),
        tensor_parallel_size: m.driver.tensor_parallel_size,
        activation_dtype: m.driver.activation_dtype.clone(),
        memory_profile: extract_memory_profile(m).unwrap_or_else(|| "balanced".to_string()),
    })
}

fn extract_memory_profile(m: &config::ModelConfig) -> Option<String> {
    m.driver
        .options
        .get("memory_profile")?
        .as_str()
        .map(str::to_string)
}
