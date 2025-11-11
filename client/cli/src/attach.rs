//! Attach command implementation for the Pie CLI.
//!
//! This module implements the `pie-cli attach` subcommand for attaching
//! to a running inferlet instance on a Pie engine and streaming its output.

use crate::engine::{self, ClientConfig};
use anyhow::{Context, Result};
use clap::Args;
use std::path::PathBuf;

/// Arguments for the `pie-cli attach` command.
#[derive(Args, Debug)]
pub struct AttachArgs {
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
    /// Prefix or full UUID of the instance to attach to.
    pub instance_id_prefix: String,
}

/// Handles the `pie-cli attach` command.
///
/// This function:
/// 1. Reads configuration from the specified config file or default config
/// 2. Creates a client configuration from config and command-line arguments
/// 3. Attempts to connect to the Pie engine server
/// 4. Queries for all live instances
/// 5. Matches the UUID prefix to find the target instance
/// 6. Attaches to the instance if a unique match is found
/// 7. Streams the inferlet output with signal handling:
///    - Ctrl-C (SIGINT): Terminates the inferlet on the server
///    - Ctrl-D (EOF): Detaches from the inferlet (continues running on server)
/// 8. Reports errors if no match or multiple matches are found
pub async fn handle_attach_command(args: AttachArgs) -> Result<()> {
    let client_config = ClientConfig::new(
        args.config,
        args.host,
        args.port,
        args.username,
        args.private_key_path,
    )?;

    let client = engine::connect_and_authenticate(&client_config)
        .await
        .context("Failed to connect to Pie engine")?;

    // Query all running instances
    let instances = client
        .list_instances()
        .await
        .context("Failed to list instances")?;

    // Find all instances matching the prefix
    let instance_id_prefix = &args.instance_id_prefix;
    let matching_instances: Vec<_> = instances
        .iter()
        .filter(|instance| instance.id.starts_with(instance_id_prefix))
        .collect();

    match matching_instances.len() {
        0 => {
            anyhow::bail!(
                "No instance found with ID prefix '{}'. Use `pie-cli list` to see running instances.",
                instance_id_prefix
            );
        }
        1 => {
            let instance_info = matching_instances[0];
            let instance_id = &instance_info.id;

            // Attach to the instance
            let instance = client
                .attach_instance(instance_id)
                .await
                .context("Failed to attach to instance")?;

            println!(
                "✅ Attached to instance {} ({})",
                instance_id, instance_info.cmd_name
            );

            engine::stream_inferlet_output(instance, client).await?;
            Ok(())
        }
        _ => {
            println!(
                "❌ The prefix '{}' is ambiguous. Multiple instances match:",
                instance_id_prefix
            );
            println!();
            for instance in &matching_instances {
                println!("  {} ({})", instance.id, instance.cmd_name);
            }
            println!();
            println!("Please provide a more specific prefix to uniquely identify the instance.");
            anyhow::bail!("Ambiguous instance ID prefix");
        }
    }
}
