//! `pie-worker` daemon — runs the inference runtime: boots drivers, serves the
//! engine, and (distributed) dials into the gateway + registers with the
//! controller. A thin bin shell (Seam 3): the `startup` process skeleton
//! composed with the `pie-worker` role library — only the two domain lines
//! (`Config::parse` + `run`) and the role-specific flags differ from the other
//! role bins.
//!
//! Model A: this bin owns the tokio runtime (`#[tokio::main]`); `startup` is
//! runtime-agnostic; `run` / `run_until_signal` / `shutdown` are async, awaited
//! on this runtime.

use std::process::ExitCode;

use clap::Parser;

// The allocator is not set here. `pie-worker` declares it, because that lib is
// the one crate this bin, the `pie` CLI and the pyo3 wheel all link, and the
// wheel links nothing else in common with us. Declaring a second one here is a
// link error, not an override.

/// Pie worker daemon. Global flags (`--config` / `--log-level` / `--metrics-addr`)
/// come from `startup`'s [`GlobalArgs`](startup::GlobalArgs); the worker adds
/// optional overrides of its config-file values.
#[derive(Parser)]
#[command(name = "pie-worker", version)]
struct Cli {
    #[command(flatten)]
    global: startup::GlobalArgs,

    /// Override the client-facing server host from config.
    #[arg(long)]
    host: Option<String>,

    /// Override the client-facing server port from config.
    #[arg(long)]
    port: Option<u16>,

    /// Override the controller endpoint from config (joins a distributed cluster).
    #[arg(long)]
    controller: Option<String>,

    /// Override this node's cluster role: decode, prefill, or encode.
    #[arg(long)]
    role: Option<pie_worker::Role>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = startup::init(
        startup::BootSpec::worker().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    // The role lib owns the domain: parse the sourced config string, then apply
    // any CLI overrides (which win over the config file).
    let mut cfg = pie_worker::Config::parse(ctx.config_str())?;
    if let Some(host) = cli.host {
        cfg.server.host = host;
    }
    if let Some(port) = cli.port {
        cfg.server.port = port;
    }
    if let Some(controller) = cli.controller {
        cfg.cluster.controller = Some(controller);
    }
    if let Some(role) = cli.role {
        cfg.cluster.role = Some(role);
    }
    cfg.validate()?;

    let handle = pie_worker::run(cfg).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
