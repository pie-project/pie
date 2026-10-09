use std::net::SocketAddr;
use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(name = "pie-gateway", version)]
struct Cli {
    #[command(flatten)]
    global: pie::args::GlobalArgs,

    #[arg(long)]
    listen: Option<SocketAddr>,

    #[arg(long)]
    worker_listen: Option<SocketAddr>,

    #[arg(long)]
    controller: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = pie::daemon::init(
        pie::daemon::BootSpec::gateway().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    let mut cfg = gateway::Config::parse(ctx.config_str())?;
    if let Some(listen) = cli.listen {
        cfg.listen = listen;
    }
    if let Some(worker_listen) = cli.worker_listen {
        cfg.worker_listen = worker_listen;
    }
    if let Some(controller) = cli.controller {
        cfg.controller = controller;
    }

    let handle = gateway::run(cfg).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
