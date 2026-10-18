//! A pool worker: keeps sealed pages for the cluster's workers in host
//! memory, by key, and serves them back over the bulk link. It loads no
//! model.

use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(name = "pie-pool", version)]
struct Cli {
    #[command(flatten)]
    global: pie::args::GlobalArgs,

    /// Where the pool serves pages (`host:port`).
    #[arg(long, default_value = "127.0.0.1:7200")]
    listen: String,

    /// The gateway's worker port.
    #[arg(long, default_value = "127.0.0.1:8081")]
    gateway: String,

    /// Bytes of pages the pool keeps, in GiB.
    #[arg(long, default_value_t = 16)]
    capacity_gib: usize,

    /// Keep every page sent on disk here too, and come back with them.
    #[arg(long)]
    dir: Option<std::path::PathBuf>,

    /// The worker id the pool reports under; apart from the controller's.
    #[arg(long, default_value_t = 1_000_000)]
    id: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = pie::daemon::init(
        pie::daemon::BootSpec::worker().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;
    let task = tokio::spawn(async move {
        if let Err(error) = worker::run_pool(
            &cli.listen,
            &cli.gateway,
            cli.capacity_gib << 30,
            cli.dir,
            ids::WorkerId(cli.id),
        )
        .await
        {
            tracing::error!(%error, "pool stopped");
        }
    });
    Ok(ctx.run_until_signal(async move { task.abort() }).await)
}
