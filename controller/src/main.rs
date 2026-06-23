//! `pie-controller` standalone process entry.
//!
//! A thin argv shell over [`pie_controller::run`]: binds the `Control` endpoint
//! and serves workers + gateways over tarpc (tcp/unix). Single-node deployments
//! skip this binary and embed the controller in-proc via
//! [`pie_controller::embed`].

use std::error::Error;
use std::time::Duration;

use clap::Parser;

use pie_controller::{Config, run};

/// Pie cluster controller — control-plane coordination (registry, neighbor
/// assignment, routing-table push, liveness). Never handles tensor data.
#[derive(Debug, Parser)]
#[command(name = "pie-controller", version, about)]
struct Cli {
    /// Control endpoint to bind: `tcp://host:port`, a bare `host:port`, or
    /// `unix:/path`.
    #[arg(long, default_value = "0.0.0.0:7000")]
    listen: String,

    /// Evict a member after this many seconds without a liveness signal.
    #[arg(long, default_value_t = 8)]
    heartbeat_timeout_secs: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let config = Config {
        listen_addr: cli.listen,
        heartbeat_timeout: Duration::from_secs(cli.heartbeat_timeout_secs),
        ..Config::default()
    };

    let (_handle, serve) = run(config).await?;
    serve.await?;
    Ok(())
}
