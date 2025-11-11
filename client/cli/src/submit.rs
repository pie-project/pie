//! Submit command implementation for the Pie CLI.
//!
//! This module implements the `pie submit` subcommand for submitting inferlets
//! to an existing running Pie engine instance.

use crate::engine;
use crate::path;
use anyhow::Context;
use anyhow::Result;
use clap::Args;
use pie_client::client;
use std::fs;
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
    /// Run the inferlet in detached mode.
    #[arg(short, long, default_value = "false")]
    pub detached: bool,
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
/// 5. In non-detached mode, streams the inferlet output with signal handling:
///    - Ctrl-C (SIGINT): Terminates the inferlet on the server
///    - Ctrl-D (EOF): Detaches from the inferlet (continues running on server)
pub async fn handle_submit_command(args: SubmitArgs) -> Result<()> {
    let client_config = engine::ClientConfig::new(
        args.config,
        args.host,
        args.port,
        args.username,
        args.private_key_path,
    )?;

    let client = engine::connect_and_authenticate(&client_config).await?;

    let inferlet_blob = fs::read(&args.inferlet)
        .context(format!("Failed to read Wasm file at {:?}", args.inferlet))?;
    let hash = client::hash_blob(&inferlet_blob);
    println!("Inferlet hash: {}", hash);

    if !client.program_exists(&hash).await? {
        client.upload_program(&inferlet_blob).await?;
        println!("✅ Inferlet upload successful.");
    }

    let cmd_name = args
        .inferlet
        .file_stem()
        .context("Inferlet path must have a valid file name")?
        .to_string_lossy()
        .to_string();
    let instance = client
        .launch_instance(hash, cmd_name, args.arguments, args.detached)
        .await?;

    println!("✅ Inferlet launched with ID: {}", instance.id());

    if !args.detached {
        engine::stream_inferlet_output(instance, client).await?;
    }

    Ok(())
}
