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
    /// Display the full UUID.
    #[arg(long)]
    pub full: bool,
    /// Display the first 8 characters of the UUID.
    #[arg(long)]
    pub long: bool,
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
        println!();

        // Column widths for the table
        // UUID is typically 36 characters (with hyphens)
        let id_width = if args.full {
            36
        } else if args.long {
            8
        } else {
            4
        };
        const STATUS_WIDTH: usize = 8;
        const CMD_WIDTH: usize = 24;
        let args_width = if args.full {
            32
        } else if args.long {
            60
        } else {
            64
        };

        // Print table header
        println!(
            "{:<id_width$}  {:<status_width$}  {:<cmd_width$}  {:<args_width$}",
            "ID",
            "STATUS",
            "COMMAND",
            "ARGUMENTS",
            id_width = id_width,
            status_width = STATUS_WIDTH,
            cmd_width = CMD_WIDTH,
            args_width = args_width
        );

        // Print separator line
        println!(
            "{}  {}  {}  {}",
            "-".repeat(id_width),
            "-".repeat(STATUS_WIDTH),
            "-".repeat(CMD_WIDTH),
            "-".repeat(args_width)
        );

        // Print each instance
        for instance in instances.iter() {
            // Use full UUID, first 8 characters, or first 4 characters based on the flag
            let uuid_display = if args.full {
                instance.id.clone()
            } else if args.long {
                instance.id.chars().take(8).collect::<String>()
            } else {
                instance.id.chars().take(4).collect::<String>()
            };

            // Format status
            let status_display = format!("{:?}", instance.status);

            // Format arguments with proper quoting to show grouping
            let args_str = format_arguments(&instance.arguments);

            // Truncate command and arguments if too long
            let cmd_display = truncate_with_ellipsis(instance.cmd_name.clone(), CMD_WIDTH);
            let args_display = truncate_with_ellipsis(args_str, args_width);

            // Print row with proper padding
            println!(
                "{:<id_width$}  {:<status_width$}  {:<cmd_width$}  {:<args_width$}",
                uuid_display,
                status_display,
                cmd_display,
                args_display,
                id_width = id_width,
                status_width = STATUS_WIDTH,
                cmd_width = CMD_WIDTH,
                args_width = args_width
            );
        }
    }

    Ok(())
}

/// Formats arguments with proper quoting to preserve grouping information.
///
/// Arguments containing spaces or special characters are quoted to show
/// they are a single argument. For example: ["-p", "Hello World"] becomes
/// `-p "Hello World"` instead of `-p Hello World`.
fn format_arguments(args: &[String]) -> String {
    args.iter()
        .map(|arg| {
            // Quote if the argument contains spaces or is empty
            if arg.is_empty() || arg.contains(char::is_whitespace) {
                format!("\"{}\"", arg.replace('"', "\\\""))
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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
