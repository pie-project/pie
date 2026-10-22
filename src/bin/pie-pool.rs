//! A pool worker: keeps sealed pages for the cluster's workers in host
//! memory, by key, and beyond it in a directory or an object store, and
//! serves them back over the bulk link. It loads no model.

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
    #[arg(long, conflicts_with = "objects")]
    dir: Option<std::path::PathBuf>,

    /// Keep every page sent in an object store, through this NIXL storage
    /// plugin (`OBJ` for S3), and come back with them (needs the `nixl`
    /// build).
    #[arg(long)]
    objects: Option<String>,

    /// One of the plugin's params, `key=value`; again for each. For `OBJ`:
    /// `bucket`, `endpoint_override`, `scheme`, `region`; credentials come
    /// from the environment (`AWS_ACCESS_KEY_ID`, ...) unless given.
    #[arg(long = "object-param", value_parser = param, requires = "objects")]
    object_params: Vec<(String, String)>,

    /// The names the pool's objects go under; pools sharing a store each
    /// need their own.
    #[arg(long, requires = "objects")]
    objects_prefix: Option<String>,

    /// The worker id the pool reports under; apart from the controller's.
    #[arg(long, default_value_t = 1_000_000)]
    id: u64,
}

fn param(given: &str) -> Result<(String, String), String> {
    let (key, value) = given
        .split_once('=')
        .ok_or_else(|| format!("`{given}` is not `key=value`"))?;
    Ok((key.to_string(), value.to_string()))
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = pie::daemon::init(
        pie::daemon::BootSpec::worker().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;
    let store = match (cli.dir, cli.objects) {
        (Some(dir), _) => worker::PoolStore::Dir(dir),
        (None, Some(plugin)) => worker::PoolStore::Objects {
            plugin,
            params: cli.object_params,
            prefix: cli
                .objects_prefix
                .unwrap_or_else(|| format!("pie-pool/{}", cli.id)),
        },
        (None, None) => worker::PoolStore::Memory,
    };
    let task = tokio::spawn(async move {
        if let Err(error) = worker::run_pool(
            &cli.listen,
            &cli.gateway,
            cli.capacity_gib << 30,
            store,
            ids::WorkerId(cli.id),
        )
        .await
        {
            tracing::error!(%error, "pool stopped");
        }
    });
    Ok(ctx.run_until_signal(async move { task.abort() }).await)
}
