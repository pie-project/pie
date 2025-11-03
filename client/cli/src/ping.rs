//! Ping command implementation for the Pie CLI.
//!
//! This module implements the `pie-cli ping` subcommand for checking the liveness
//! of a running Pie engine instance.

use crate::engine;
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;
use std::time::Instant;

/// Arguments for the `pie-cli ping` command.
#[derive(Args, Debug)]
pub struct PingArgs {
    /// Path to a custom TOML configuration file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// The network host to connect to.
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to connect to.
    #[arg(long)]
    pub port: Option<u16>,
    /// The username to use for authentication.
    #[arg(long)]
    pub username: Option<String>,
    /// Path to the private key file to use for authentication.
    #[arg(long)]
    pub private_key_path: Option<PathBuf>,
}

/// Handles the `pie-cli ping` command.
///
/// This function:
/// 1. Reads configuration from the specified config file or default config
/// 2. Creates a client configuration from config and command-line arguments
/// 3. Attempts to connect to the Pie engine server
/// 4. Reports success if the connection and authentication succeed, or failure otherwise
pub async fn handle_ping_command(
    config_path: Option<PathBuf>,
    host: Option<String>,
    port: Option<u16>,
    username: Option<String>,
    private_key_path: Option<PathBuf>,
) -> Result<()> {
    let client_config =
        engine::ClientConfig::new(config_path, host, port, username, private_key_path)?;

    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    println!("🔍 Pinging Pie engine at {}", url);

    let client = engine::connect_and_authenticate(&client_config)
        .await
        .context("Failed to connect to Pie engine")?;

    let start_time = Instant::now();
    client.ping().await.context("Failed to ping Pie engine")?;
    let duration = start_time.elapsed();

    println!(
        "✅ Pie engine is alive and responsive! (latency: {:.3}ms)",
        duration.as_secs_f64() * 1000.0
    );
    Ok(())
}
