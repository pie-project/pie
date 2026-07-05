//! `pie-gateway` daemon — the client-facing edge plane (REST/SSE + WebSocket);
//! workers dial IN. A thin bin shell (Seam 3): the `bootstrap` process skeleton
//! composed with the `pie-gateway` role library — only the two domain lines
//! (`Config::parse` + `run`) and the role-specific flags differ from the other
//! role bins.
//!
//! Model A: this bin owns the tokio runtime (`#[tokio::main]`); `bootstrap` is
//! runtime-agnostic; `run` / `run_until_signal` / `shutdown` are async, awaited
//! on this runtime.

use std::net::SocketAddr;
use std::process::ExitCode;

use clap::Parser;

/// Pie gateway daemon. Global flags (`--config` / `--log-level` / `--metrics-addr`)
/// come from `bootstrap`'s [`GlobalArgs`](bootstrap::GlobalArgs); the gateway adds
/// optional overrides of its config-file addresses.
#[derive(Parser)]
#[command(name = "pie-gateway", version)]
struct Cli {
    #[command(flatten)]
    global: bootstrap::GlobalArgs,

    /// Override the client-facing listen address from config.
    #[arg(long)]
    listen: Option<SocketAddr>,

    /// Override the worker-facing dial-in listen address from config.
    #[arg(long)]
    worker_listen: Option<SocketAddr>,

    /// Override the controller endpoint from config.
    #[arg(long)]
    controller: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = bootstrap::init(
        bootstrap::BootSpec::gateway().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    // The role lib owns the domain: parse the sourced config string, then apply
    // any CLI overrides (which win over the config file).
    let mut cfg = pie_gateway::Config::parse(ctx.config_str())?;
    if let Some(listen) = cli.listen {
        cfg.listen = listen;
    }
    if let Some(worker_listen) = cli.worker_listen {
        cfg.worker_listen = worker_listen;
    }
    if let Some(controller) = cli.controller {
        cfg.controller = controller;
    }

    let handle = pie_gateway::run(cfg).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
