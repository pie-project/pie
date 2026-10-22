use std::process::ExitCode;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "pie-controller", version)]
struct Cli {
    #[command(flatten)]
    global: pie::args::GlobalArgs,

    #[arg(long, value_name = "ADDR")]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();

    let ctx = pie::daemon::init(
        pie::daemon::BootSpec::controller().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    let mut config = controller::Config::parse(ctx.config_str())?;
    if let Some(listen) = cli.listen {
        config.listen_addr = listen;
    }

    let handle = controller::run(config).await?;
    // SIGHUP swaps the cluster policies for the file's `policies`; nothing
    // else in it is read again.
    let policies = handle.clone();
    ctx.on_hangup(move |config| {
        let policies = policies.clone();
        async move {
            let specs = match controller::Config::parse(&config) {
                Ok(config) => config.policies,
                Err(e) => {
                    tracing::error!(
                        "SIGHUP: the config does not parse, so the policies stand: {e:#}"
                    );
                    return;
                }
            };
            if let Err(e) = policies.reload_policies(&specs).await {
                tracing::error!("SIGHUP: {e:#}");
            }
        }
    });
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
