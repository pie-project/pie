use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(name = "pie-worker", version)]
struct Cli {
    #[command(flatten)]
    global: pie::args::GlobalArgs,

    #[arg(long)]
    listen: Option<std::net::SocketAddr>,

    #[arg(long)]
    controller: Option<String>,

    #[arg(long)]
    role: Option<worker::Role>,
}

// Every work queue a CUDA context may have, set before CUDA's first call
// reads the count and before any thread (the runtime's) runs: an
// expert-parallel group of four or eight ranks on one device needs more
// than the default eight, and refuses to wire without them.
#[cfg(all(target_os = "linux", feature = "engine-cuda"))]
#[used]
#[allow(unsafe_code)]
#[unsafe(link_section = ".init_array")]
static CUDA_WORK_QUEUES: extern "C" fn() = {
    extern "C" fn queues() {
        if std::env::var_os("CUDA_DEVICE_MAX_CONNECTIONS").is_none() {
            // SAFETY: a constructor, before main and before any thread.
            unsafe { std::env::set_var("CUDA_DEVICE_MAX_CONNECTIONS", "32") };
        }
    }
    queues
};

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();
    let ctx = pie::daemon::init(
        pie::daemon::BootSpec::worker().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    let mut cfg = worker::Config::parse(ctx.config_str())?;
    if let Some(listen) = cli.listen {
        cfg.server.listen = listen;
    }
    if let Some(controller) = cli.controller {
        cfg.cluster.controller = Some(controller);
    }
    if let Some(role) = cli.role {
        cfg.cluster.role = Some(role);
    }
    cfg.validate()?;

    let handle = worker::run(cfg, &pie::paths::pie_home()).await?;
    // SIGHUP swaps the worker policy for the file's `[runtime] worker_policy`;
    // nothing else in it is read again, and the worker serves on.
    ctx.on_hangup(|config| async move {
        match worker::Config::parse(&config) {
            Ok(config) => {
                if let Err(e) = worker::reload_policy(&config).await {
                    tracing::error!("SIGHUP: {e:#}");
                }
            }
            Err(e) => {
                tracing::error!("SIGHUP: the config does not parse, so the policy stands: {e:#}")
            }
        }
    });
    Ok(ctx
        .run_until_signal(async move {
            // A signal drains first: no new turns, and the cluster policy may
            // move what runs here; then the worker leaves.
            handle.drain(std::time::Duration::from_secs(60)).await;
            handle.shutdown().await
        })
        .await)
}
