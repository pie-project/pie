//! `pie-gateway` — standalone edge-plane process entry (`src/main.rs`).
//!
//! The single global host clients dial. A thin argv shell over
//! [`pie_gateway::run`]: it dials the controller for placement and terminates
//! client WebSockets, relaying each session to the routed worker. Never handles
//! tensor data.

use std::net::SocketAddr;

use clap::Parser;
use pie_gateway::{GatewayConfig, run};

/// Pie gateway — client-facing edge plane (terminates WebSockets, routes
/// sessions to workers via the controller). Never handles tensor data.
#[derive(Debug, Parser)]
#[command(name = "pie-gateway", version, about)]
struct Cli {
    /// Address the client-facing WebSocket listens on (the single global host).
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: SocketAddr,

    /// Controller's tarpc control endpoint: `tcp://host:port`, a bare
    /// `host:port`, or `unix:/path/to.sock`.
    #[arg(long, default_value = "127.0.0.1:7000")]
    controller: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run(GatewayConfig {
        listen: cli.listen,
        controller: cli.controller,
    })
    .await
}
