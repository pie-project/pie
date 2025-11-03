//! Submit command implementation for the Pie CLI.
//!
//! This module implements the `pie submit` subcommand for submitting inferlets
//! to an existing running Pie engine instance.

use crate::engine;
use crate::path;
use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

/// Arguments for the `pie submit` command.
#[derive(Args, Debug)]
pub struct SubmitArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = path::expand_tilde)]
    pub inferlet: PathBuf,
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
    /// Arguments to pass to the inferlet after `--`.
    #[arg(last = true)]
    pub arguments: Vec<String>,
}

/// Handles the `pie submit` command.
///
/// This function:
/// 1. Reads configuration from the specified config file or default config
/// 2. Creates a client configuration from config and command-line arguments
/// 3. Connects to the existing Pie engine server
/// 4. Submits the specified inferlet with the provided arguments
/// 5. Waits for the inferlet to finish execution and prints the result
pub async fn handle_submit_command(
    config_path: Option<PathBuf>,
    host: Option<String>,
    port: Option<u16>,
    username: Option<String>,
    private_key_path: Option<PathBuf>,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<()> {
    let client_config =
        engine::ClientConfig::new(config_path, host, port, username, private_key_path)?;

    engine::submit_inferlet_and_wait(&client_config, inferlet_path, arguments).await
}
