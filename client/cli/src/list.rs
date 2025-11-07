//! List command implementation for the Pie CLI.
//!
//! This module implements the `pie-cli list` subcommand for querying
//! all running inferlet instances from a Pie engine.

use crate::engine;
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

/// Arguments for the `pie-cli list` command.
#[derive(Args, Debug)]
pub struct ListArgs {
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

/// Handles the `pie-cli list` command.
///
/// This function:
/// 1. Reads configuration from the specified config file or default config
/// 2. Creates a client configuration from config and command-line arguments
/// 3. Attempts to connect to the Pie engine server
/// 4. Queries for all live instances
/// 5. Displays the list of running inferlet instances
pub async fn handle_list_command(
    config_path: Option<PathBuf>,
    host: Option<String>,
    port: Option<u16>,
    username: Option<String>,
    private_key_path: Option<PathBuf>,
) -> Result<()> {
    let client_config =
        engine::ClientConfig::new(config_path, host, port, username, private_key_path)?;

    let client = engine::connect_and_authenticate(&client_config)
        .await
        .context("Failed to connect to Pie engine")?;

    let instances = client
        .list_instances()
        .await
        .context("Failed to list instances")?;

    if instances.is_empty() {
        println!("✅ No running instances found.");
    } else {
        println!(
            "✅ Found {} running instance{}:",
            instances.len(),
            if instances.len() == 1 { "" } else { "s" }
        );
        for (idx, instance) in instances.iter().enumerate() {
            println!("  {}. {}", idx + 1, instance.id);
        }
    }

    Ok(())
}
