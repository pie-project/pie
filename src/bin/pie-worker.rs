use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(name = "pie-worker", version)]
struct Cli {
    #[command(flatten)]
    global: pie::process::GlobalArgs,

    #[arg(long)]
    host: Option<String>,

    #[arg(long)]
    port: Option<u16>,

    #[arg(long)]
    controller: Option<String>,

    #[arg(long)]
    role: Option<worker::Role>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = pie::process::init(
        pie::process::BootSpec::worker().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    let mut cfg = worker::Config::parse(ctx.config_str())?;
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

    let handle = worker::run(cfg, &pie::paths::pie_home()).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
