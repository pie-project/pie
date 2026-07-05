//! `pie-worker` daemon — runs the inference runtime: boots drivers, serves the
//! engine, and (distributed) dials into the gateway + registers with the
//! controller. A thin bin shell (Seam 3): the `bootstrap` process skeleton
//! composed with the `pie-worker` role library — only the two domain lines
//! (`Config::parse` + `run`) and the role-specific flags differ from the other
//! role bins.
//!
//! Model A: this bin owns the tokio runtime (`#[tokio::main]`); `bootstrap` is
//! runtime-agnostic; `run` / `run_until_signal` / `shutdown` are async, awaited
//! on this runtime.

use std::process::ExitCode;

use clap::Parser;

// The worker runs the inference runtime (scheduler + chain-extender pool), whose
// burst-allocation pattern benefits from mimalloc — as the pre-refactor worker
// binary used. A `#[global_allocator]` must live in the final binary crate.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Pie worker daemon. Global flags (`--config` / `--log-level` / `--metrics-addr`)
/// come from `bootstrap`'s [`GlobalArgs`](bootstrap::GlobalArgs); the worker adds
/// optional overrides of its config-file values.
#[derive(Parser)]
#[command(name = "pie-worker", version)]
struct Cli {
    #[command(flatten)]
    global: bootstrap::GlobalArgs,

    /// Override the client-facing server host from config.
    #[arg(long)]
    host: Option<String>,

    /// Override the client-facing server port from config.
    #[arg(long)]
    port: Option<u16>,

    /// Override the controller endpoint from config (joins a distributed cluster).
    #[arg(long)]
    controller: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = bootstrap::init(
        bootstrap::BootSpec::worker().version(env!("CARGO_PKG_VERSION")),
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

    let handle = pie_worker::run(cfg).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
