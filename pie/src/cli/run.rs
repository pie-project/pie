//! Run command implementation for the Pie CLI.
//!
//! This module implements the `pie run` subcommand for running inferlets
//! with a one-shot Pie engine instance.

use super::{path, service};
use crate::engine::Config as EngineConfig;
use anyhow::Result;
use clap::Args;
use std::path::PathBuf;

/// Arguments for the `pie run` command.
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Path to the .wasm inferlet file.
    #[arg(value_parser = path::expand_tilde)]
    pub inferlet: PathBuf,
    /// Path to a custom TOML configuration file.
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    /// A log file to write to.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Arguments to pass to the inferlet after `--`.
    #[arg(last = true)]
    pub arguments: Vec<String>,
}

/// Handles the `pie run` command.
///
/// This function:
/// 1. Starts the Pie engine and backend services
/// 2. Runs the specified inferlet with the provided arguments
/// 3. Waits for the inferlet to finish execution
/// 4. Terminates the engine and backend services
pub async fn handle_run_command(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<()> {
    // Start the engine and backend services
    let (shutdown_tx, server_handle, backend_processes, client_config) =
        service::start_engine_and_backend(engine_config, backend_configs, None).await?;

    // Run the inferlet
    service::submit_inferlet_and_wait(&client_config, inferlet_path, arguments, None).await?;

    // Terminate the engine and backend services
    service::terminate_engine_and_backend(backend_processes, shutdown_tx, server_handle).await?;
    Ok(())
}
