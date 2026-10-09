use std::process::ExitCode;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "pie-controller", version)]
struct Cli {
    #[command(flatten)]
    global: pie::process::GlobalArgs,

    #[arg(long, value_name = "ADDR")]
    listen: Option<String>,
}

#[tokio::main]
async fn main() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();

    let ctx = pie::process::init(
        pie::process::BootSpec::controller().version(env!("CARGO_PKG_VERSION")),
        cli.global,
    )?;

    let mut config = controller::Config::parse(ctx.config_str())?;
    if let Some(listen) = cli.listen {
        config.listen_addr = listen;
    }

    let handle = controller::run(config).await?;
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
