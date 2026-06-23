//! `pie-gateway` — standalone edge-plane process entry (`src/main.rs`).
//!
//! The client-facing host users dial. A thin argv shell over [`pie_gateway::run`]:
//! it dials the controller for the routing table, terminates user protocols
//! (REST/SSE + WebSocket), and pipes each turn's token stream back. Post-inversion
//! (M3) it also binds a worker-facing listener that workers dial INTO. Never
//! handles tensor data.

use std::net::SocketAddr;

use clap::Parser;
use pie_gateway::{GatewayConfig, run};

/// Pie gateway — client-facing edge plane (terminates REST/SSE + WebSocket,
/// routes turns to workers that dial in, streams tokens back). Never handles
/// tensor data.
#[derive(Debug, Parser)]
#[command(name = "pie-gateway", version, about)]
struct Cli {
    /// Address the client-facing edge (REST/SSE + WebSocket) listens on.
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: SocketAddr,

    /// Address the worker-facing data plane listens on (workers dial IN, M3).
    #[arg(long, default_value = "0.0.0.0:8081")]
    worker_listen: SocketAddr,

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
        worker_listen: cli.worker_listen,
        controller: cli.controller,
    })
    .await
}
