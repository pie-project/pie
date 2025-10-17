//! Submit command implementation for the Pie CLI.
//!
//! This module implements the `pie submit` subcommand for submitting inferlets
//! to an existing running Pie engine instance.

use crate::{engine, path};
use anyhow::Context;
use anyhow::Result;
use clap::Args;
use libpie::auth;
use std::{fs, path::PathBuf};

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
    /// Authentication secret for connecting to the server.
    #[arg(long)]
    pub auth_secret: Option<String>,
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
    auth_secret: Option<String>,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<()> {
    // Read config file only if when any parameter is missing
    let config_file = if host.is_none() || port.is_none() || auth_secret.is_none() {
        let config_str = match config_path {
            Some(path) => fs::read_to_string(&path)
                .with_context(|| format!("Failed to read config file at {:?}", path))?,
            None => fs::read_to_string(&crate::path::get_default_config_path()?).context(
                "Failed to read default config file. Try running `pie config init` first.",
            )?,
        };
        Some(toml::from_str::<crate::config::ConfigFile>(&config_str)?)
    } else {
        None
    };

    // Prefer command-line arguments and use config file values if not provided
    let host = host
        .or_else(|| config_file.as_ref().and_then(|cfg| cfg.host.clone()))
        .unwrap_or_else(|| "127.0.0.1".to_string());

    let port = port
        .or_else(|| config_file.as_ref().and_then(|cfg| cfg.port))
        .unwrap_or(8080);

    let auth_secret = auth_secret
        .or_else(|| config_file.as_ref().and_then(|cfg| cfg.auth_secret.clone()))
        .unwrap_or_else(engine::generate_random_auth_secret);

    // Initialize the JWT secret for authentication
    auth::init_secret(&auth_secret);

    // Create client configuration
    let client_config = engine::ClientConfig { host, port };

    // Submit the inferlet to the existing server
    engine::submit_inferlet_and_wait(&client_config, inferlet_path, arguments).await?;

    Ok(())
}
