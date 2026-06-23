//! `pie-controller` — standalone controller process entry (`src/main.rs`).
//!
//! A thin argv shell over [`pie_controller::run_as_process`]: it binds the
//! control endpoint and serves the control-RPC surface that the gateway and
//! distributed workers dial. Single-node deployments run controller-free and
//! never spawn this binary.

use std::error::Error;

use clap::Parser;

use pie_controller::{ControllerConfig, ProcessConfig, run_as_process};

/// Pie cluster controller — control-plane coordination (pairing, role
/// assignment, health). Never handles tensor data.
#[derive(Debug, Parser)]
#[command(name = "pie-controller", version, about)]
struct Cli {
    /// Address the control endpoint listens on.
    #[arg(long, default_value = "0.0.0.0:7000")]
    listen: String,

    /// Silent heartbeat ticks before a node is graded degraded.
    #[arg(long, default_value_t = 3)]
    degrade_after: u64,

    /// Silent heartbeat ticks before a node is graded unreachable.
    #[arg(long, default_value_t = 6)]
    unreachable_after: u64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let config = ProcessConfig {
        listen_addr: cli.listen,
        controller: ControllerConfig {
            degrade_after: cli.degrade_after,
            unreachable_after: cli.unreachable_after,
        },
        ..ProcessConfig::default()
    };

    run_as_process(config)?;
    Ok(())
}
