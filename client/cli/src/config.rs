//! Configuration management commands for the Pie CLI.
//!
//! This module implements the `pie-cli config` subcommands for managing Pie CLI configuration files,
//! including initializing, updating, and displaying configuration settings.

use super::path;
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

#[derive(Subcommand, Debug)]
pub enum ConfigCommands {
    /// Create a default config file.
    Init,
    /// Update the entries of the default config file.
    Update(ConfigUpdateArgs),
    /// Show the content of the default config file.
    Show,
}

#[derive(Args, Debug)]
pub struct ConfigUpdateArgs {
    /// Host to connect to
    #[arg(long)]
    pub host: Option<String>,
    /// Port to connect to
    #[arg(long)]
    pub port: Option<u16>,
    /// Username for authentication
    #[arg(long)]
    pub username: Option<String>,
    /// Path to private key file
    #[arg(long, value_parser = path::expand_tilde)]
    pub private_key_path: Option<PathBuf>,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Serialize, Debug)]
pub struct ConfigFile {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub username: Option<String>,
    pub private_key_path: Option<PathBuf>,
}

/// Handles the `pie-cli config` command.
pub async fn handle_config_command(command: ConfigCommands) -> Result<()> {
    match command {
        ConfigCommands::Init => handle_config_init_subcommand().await,
        ConfigCommands::Update(args) => handle_config_update_subcommand(args).await,
        ConfigCommands::Show => handle_config_show_subcommand().await,
    }
}

/// Create a default config file.
async fn handle_config_init_subcommand() -> Result<()> {
    let config_path = path::get_default_config_path()?;

    // Check if config file already exists
    if config_path.exists() {
        print!(
            "Configuration file already exists at {:?}. Overwrite? (y/N): ",
            config_path
        );
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborting. Configuration file was not overwritten.");
            return Ok(());
        }
    }

    // Create parent directories if they don't exist
    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory {:?}", parent))?;
    }

    // Create the config file with default content
    let default_content = create_default_config_content();
    fs::write(&config_path, &default_content)
        .with_context(|| format!("Failed to write config file at {:?}", config_path))?;

    println!("✅ Created default configuration file at {:?}", config_path);
    print_default_config_content(&config_path, &default_content);
    Ok(())
}

/// Update the specified entries of the default config file.
async fn handle_config_update_subcommand(args: ConfigUpdateArgs) -> Result<()> {
    let config_path = path::get_default_config_path()?;

    // Check if config file exists
    if !config_path.exists() {
        anyhow::bail!(
            "Configuration file not found at {:?}. Run `pie-cli config init` first.",
            config_path
        );
    }

    // Read the existing config file
    let config_str = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file at {:?}", config_path))?;

    // Parse the existing config
    let mut config: ConfigFile = toml::from_str(&config_str)
        .with_context(|| format!("Failed to parse config file at {:?}", config_path))?;

    // Track which fields were updated
    let mut updated = Vec::new();

    // Update fields if provided
    if let Some(host) = args.host {
        updated.push(format!("host = \"{}\"", host));
        config.host = Some(host);
    }
    if let Some(port) = args.port {
        updated.push(format!("port = {}", port));
        config.port = Some(port);
    }
    if let Some(username) = args.username {
        updated.push(format!("username = \"{}\"", username));
        config.username = Some(username);
    }
    if let Some(private_key_path) = args.private_key_path {
        let path_str = private_key_path.to_string_lossy().to_string();
        updated.push(format!("private_key_path = \"{}\"", path_str));
        config.private_key_path = Some(private_key_path);
    }

    if updated.is_empty() {
        println!("⚠️  No fields provided to update.");
        return Ok(());
    }

    // Serialize the updated config
    let updated_config_str =
        toml::to_string_pretty(&config).with_context(|| "Failed to serialize updated config")?;

    // Write the updated config back to the file
    fs::write(&config_path, updated_config_str)
        .with_context(|| format!("Failed to write updated config to {:?}", config_path))?;

    println!("✅ Updated configuration file at {:?}", config_path);
    println!("   Updated fields:");
    for field in updated {
        println!("   - {}", field);
    }

    Ok(())
}

/// Show the content of the default config file.
async fn handle_config_show_subcommand() -> Result<()> {
    let config_path = path::get_default_config_path()?;

    // Check if config file exists
    if !config_path.exists() {
        anyhow::bail!(
            "Configuration file not found at {:?}. Run `pie-cli config init` first.",
            config_path
        );
    }

    // Read and display the config file content
    let config_content = fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read config file at {:?}", config_path))?;

    print_default_config_content(&config_path, &config_content);

    Ok(())
}

fn print_default_config_content(config_path: &PathBuf, config_content: &str) {
    println!("📄 Configuration file at {:?}:", config_path);
    println!();
    println!("{}", config_content);
}

/// Create the default content of the config file.
fn create_default_config_content() -> String {
    format!(
        r#"host = "127.0.0.1"
port = 8080
username = "{username}"
private_key_path = "~/.ssh/id_rsa"
"#,
        username = whoami::username()
    )
}
