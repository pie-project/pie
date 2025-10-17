//! Run command implementation for the Pie CLI.
//!
//! This module implements the `pie run` subcommand for running inferlets
//! with a one-shot Pie engine instance.

use crate::{engine, output, path};
use anyhow::Result;
use clap::Args;
use libpie::Config as EngineConfig;
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
/// 1. Creates an editor and printer for output handling
/// 2. Starts the Pie engine and backend services
/// 3. Runs the specified inferlet with the provided arguments
/// 4. Waits for the inferlet to finish execution
/// 5. Terminates the engine and backend services
pub async fn handle_run_command(
    engine_config: EngineConfig,
    backend_configs: Vec<toml::Value>,
    inferlet_path: PathBuf,
    arguments: Vec<String>,
) -> Result<()> {
    let (_rl, printer) = output::create_editor_and_printer_with_history().await?;

    // Start the engine and backend services
    let (shutdown_tx, server_handle, backend_processes, client_config) =
        engine::start_engine_and_backend(engine_config, backend_configs, printer.clone()).await?;

    // Run the inferlet
    engine::submit_inferlet_and_wait(&client_config, inferlet_path, arguments, printer.clone())
        .await?;

    // Terminate the engine and backend services
    engine::terminate_engine_and_backend(
        &client_config,
        backend_processes,
        shutdown_tx,
        server_handle,
    )
    .await?;
    Ok(())
}
