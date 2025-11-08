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
pub async fn handle_list_command(args: ListArgs) -> Result<()> {
    let client_config = engine::ClientConfig::new(
        args.config,
        args.host,
        args.port,
        args.username,
        args.private_key_path,
    )?;

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
        for instance in instances.iter() {
            // Extract first 4 characters of UUID
            let uuid_prefix = instance.id.chars().take(4).collect::<String>();

            // Format arguments as a space-separated string
            let args_str = instance.arguments.join(" ");

            // Truncate command and arguments if too long
            const MAX_CMD_LEN: usize = 40;
            const MAX_ARGS_LEN: usize = 60;
            let cmd_display = truncate_with_ellipsis(instance.cmd_name.clone(), MAX_CMD_LEN);
            let args_display = truncate_with_ellipsis(args_str, MAX_ARGS_LEN);

            println!("  {} {} {}", uuid_prefix, cmd_display, args_display);
        }
    }

    Ok(())
}

/// Truncates a string to a maximum length, appending "..." if truncated.
fn truncate_with_ellipsis(mut s: String, max_display_chars: usize) -> String {
    let mut chars_scanned = 0;
    let mut truncation_byte_pos = None;
    let chars_before_ellipsis = max_display_chars.saturating_sub(3);
    
    for (byte_pos, _) in s.char_indices() {
        if chars_scanned == chars_before_ellipsis {
            truncation_byte_pos = Some(byte_pos);
        }
        chars_scanned += 1;
        if chars_scanned > max_display_chars {
            break;
        }
    }
    
    // If string fits within the limit, no truncation needed
    if chars_scanned <= max_display_chars {
        return s;
    }
    
    // Edge case: ellipsis itself is too long to fit
    if max_display_chars < 3 {
        return s.chars().take(max_display_chars).collect();
    }
    
    // Truncate and append ellipsis
    if let Some(byte_pos) = truncation_byte_pos {
        s.truncate(byte_pos);
        s.push_str("...");
    }
    s
}
