//! Configuration management commands for the Pie CLI.
//!
//! This module implements the `pie-cli config` subcommands for managing Pie CLI configuration files,
//! including initializing, updating, and displaying configuration settings.

use super::path;
use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use pie_client::crypto::ParsedPrivateKey;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

#[derive(Subcommand, Debug)]
pub enum ConfigCommands {
    /// Create a default config file.
    Init {
        /// Enable authentication
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        enable_auth: bool,
    },
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
    /// Enable authentication
    #[arg(long)]
    pub enable_auth: Option<bool>,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Serialize, Debug)]
pub struct ConfigFile {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub username: Option<String>,
    pub private_key_path: Option<PathBuf>,
    pub enable_auth: Option<bool>,
}

/// Handles the `pie-cli config` command.
pub async fn handle_config_command(command: ConfigCommands) -> Result<()> {
    match command {
        ConfigCommands::Init { enable_auth } => handle_config_init_subcommand(enable_auth).await,
        ConfigCommands::Update(args) => handle_config_update_subcommand(args).await,
        ConfigCommands::Show => handle_config_show_subcommand().await,
    }
}

/// Create a default config file.
async fn handle_config_init_subcommand(enable_auth: bool) -> Result<()> {
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
        fs::create_dir_all(parent).context(format!("Failed to create directory {:?}", parent))?;
    }

    // Find the SSH key or use default
    let found_key_path = find_ssh_key().context("Failed when searching for SSH key")?;
    const DEFAULT_KEY_PATH: &str = "~/.ssh/id_ed25519";

    // Create the config file with default content
    let default_content = create_default_config_content(
        found_key_path.as_deref().unwrap_or(DEFAULT_KEY_PATH),
        enable_auth,
    );
    fs::write(&config_path, &default_content)
        .context(format!("Failed to write config file at {:?}", config_path))?;

    println!("✅ Created default configuration file at {:?}", config_path);
    print_default_config_content(&default_content);

    // Print messages about the private key if authentication is enabled
    if enable_auth {
        // If the key exists, print it and validate it
        if let Some(found_key_path) = found_key_path {
            println!("✅ Using private key found at {:?}", found_key_path);
            println!("   You can update the key path in the config file:");
            println!("      `pie-cli config update --private-key-path <path>`");

            if !validate_private_key(&found_key_path) {
                println!(
                    "   The configuration has been saved, but you'll need to \
                    provide a valid key to connect."
                );
            }
        // Otherwise, warn if the key doesn't exist
        } else {
            println!();
            println!(
                "⚠️ Warning: Private key not found in '~/.ssh', using default path: '{}'",
                DEFAULT_KEY_PATH
            );
            println!("   Please take either of the following actions when using authentication:");
            println!("   1. Generate an SSH key pair by running `ssh-keygen`");
            println!("   2. Update the key path in the config file:");
            println!("      `pie-cli config update --private-key-path <path>`");
        }
    }

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
        .context(format!("Failed to read config file at {:?}", config_path))?;

    // Parse the existing config
    let mut config: ConfigFile = toml::from_str(&config_str)
        .context(format!("Failed to parse config file at {:?}", config_path))?;

    // Track which fields were updated
    let mut updated = Vec::new();
    let mut updated_key_path: Option<String> = None;

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
        updated_key_path = Some(path_str.clone());
        config.private_key_path = Some(private_key_path);
    }
    if let Some(enable_auth) = args.enable_auth {
        updated.push(format!("enable_auth = {}", enable_auth));
        config.enable_auth = Some(enable_auth);
    }

    if updated.is_empty() {
        println!("⚠️ No fields provided to update.");
        return Ok(());
    }

    // Serialize the updated config
    let updated_config_str =
        toml::to_string_pretty(&config).context("Failed to serialize updated config")?;

    // Write the updated config back to the file
    fs::write(&config_path, updated_config_str).context(format!(
        "Failed to write updated config to {:?}",
        config_path
    ))?;

    println!("✅ Updated configuration file at {:?}", config_path);
    println!("   Updated fields:");
    for field in updated {
        println!("   - {}", field);
    }

    // If the private key path was updated, validate it
    if let Some(key_path) = updated_key_path {
        if !validate_private_key(&key_path) {
            println!(
                "   The configuration has been saved, but you'll need to \
                provide a valid key to connect."
            );
        }
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
        .context(format!("Failed to read config file at {:?}", config_path))?;

    println!("📄 Configuration file at {:?}:", config_path);
    print_default_config_content(&config_content);

    Ok(())
}

fn print_default_config_content(config_content: &str) {
    println!("{}", config_content);
}

/// Find the SSH key in the user's `~/.ssh` directory.
/// Searches for keys in order: `id_ed25519`, `id_rsa`, `id_ecdsa`.
/// Returns `Some(path)` if a key is found, `None` otherwise.
fn find_ssh_key() -> Result<Option<String>> {
    let home = dirs::home_dir().context("Failed to find home directory")?;
    let ssh_dir = home.join(".ssh");

    // Search for keys in order of preference
    for key_name in &["id_ed25519", "id_rsa", "id_ecdsa"] {
        let key_path = ssh_dir.join(key_name);
        if key_path.exists() {
            return Ok(Some(format!("~/.ssh/{}", key_name)));
        }
    }

    Ok(None)
}

/// Create the default content of the config file.
fn create_default_config_content(private_key_path: &str, enable_auth: bool) -> String {
    let private_key_path = if enable_auth { private_key_path } else { "" };

    format!(
        r#"host = "127.0.0.1"
port = 8080
username = "{username}"
private_key_path = "{private_key_path}"
enable_auth = {enable_auth}
"#,
        username = whoami::username(),
        private_key_path = private_key_path,
        enable_auth = enable_auth
    )
}

/// Validate that a private key at the given path can be parsed.
/// Prints a warning if the key cannot be parsed, but does not return an error.
fn validate_private_key(key_path: &str) -> bool {
    // Expand tilde in the path
    let path = PathBuf::from(shellexpand::tilde(key_path).as_ref());

    if !path.exists() {
        println!();
        println!("⚠️ Warning: Private key file not found at {:?}", path);
        return false;
    }

    // Check file permissions (Unix only)
    #[cfg(unix)]
    if let Err(e) = path::check_private_key_permissions(&path) {
        println!();
        println!("⚠️ Warning: {}", e);
        return false;
    }

    match fs::read_to_string(&path) {
        Ok(key_content) => match ParsedPrivateKey::parse(&key_content) {
            Ok(_) => return true,
            Err(e) => {
                println!();
                println!("⚠️ Warning: Failed to parse private key at {:?}", path);
                println!("   Error: {}", e);
            }
        },
        Err(e) => {
            println!();
            println!("⚠️ Warning: Failed to read private key file at {:?}", path);
            println!("   Error: {}", e);
        }
    }

    false
}
