//! `pie-controller` — standalone control-plane daemon (Seam 3 thin shell).
//!
//! Composes the shared [`bootstrap`] process skeleton (global flags, config
//! sourcing, tracing, `/metrics`, signal lifecycle) with the [`pie_controller`]
//! role library. Only the two middle lines (`Config::parse` + `run`) are
//! controller-specific; every pie daemon bin is otherwise identical.
//!
//! Single-node deployments do not use this binary — they embed the controller
//! in-proc via [`pie_controller::embed`] at the `bin/pie` composition root.

use std::process::ExitCode;

use clap::Parser;

/// Pie cluster controller — control-plane coordination (registry, neighbor
/// assignment, routing-table push, liveness). Never handles tensor data.
#[derive(Debug, Parser)]
#[command(name = "pie-controller", version)]
struct Cli {
    #[command(flatten)]
    global: bootstrap::GlobalArgs,

    /// Control endpoint to bind: `tcp://host:port`, a bare `host:port`, or
    /// `unix:/path`. Overrides `listen_addr` from the config file.
    #[arg(long, value_name = "ADDR")]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();

    let ctx = bootstrap::init(
        bootstrap::BootSpec::controller().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    let mut config = pie_controller::Config::parse(ctx.config_str())?;
    if let Some(listen) = cli.listen {
        config.listen_addr = listen;
    }

    let handle = pie_controller::run(config).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
