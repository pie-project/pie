//! Authorization management commands for Pie.
//!
//! This module implements the `pie auth` commands for managing authorized clients
//! by adding or removing their public keys in the `authorized_clients.toml` file.

use super::path;
use crate::auth::{
    AuthorizedClients, InsertKeyResult, InsertUserResult, PublicKey, RemoveKeyResult,
    RemoveUserResult,
};
use anyhow::{Context, Result, bail};
use chrono::Local;
use clap::Subcommand;
use std::{
    fs,
    io::{self, IsTerminal, Read, Write},
};

#[derive(Subcommand, Debug)]
pub enum AuthCommands {
    /// Add an authorized client and its public key.
    /// The public key is read from stdin and can be in OpenSSH, PKCS#8 PEM, or PKCS#1 PEM format.
    /// If key_name is not provided, the current timestamp will be used.
    Add {
        /// Username of the client
        username: String,
        /// Optional name for this key (e.g., 'laptop', 'desktop'). Defaults to current timestamp.
        key_name: Option<String>,
    },
    /// Remove an authorized client or a specific key.
    /// If key_name is provided, only that key is removed.
    /// If key_name is not provided, the entire user entry is removed.
    Remove {
        /// Username of the client
        username: String,
        /// Optional name of the specific key to remove. If omitted, removes the entire user.
        key_name: Option<String>,
    },
    /// List all authorized clients and their keys.
    List,
}

/// Handles the `pie auth` command.
/// Public keys are read from stdin to avoid exposing them in shell history.
pub async fn handle_auth_command(command: AuthCommands) -> Result<()> {
    match command {
        AuthCommands::Add { username, key_name } => {
            handle_auth_add_subcommand(username, key_name).await
        }
        AuthCommands::Remove { username, key_name } => {
            handle_auth_remove_subcommand(username, key_name).await
        }
        AuthCommands::List => handle_auth_list_subcommand().await,
    }
}

/// Handles the `pie auth add` subcommand.
async fn handle_auth_add_subcommand(username: String, key_name: Option<String>) -> Result<()> {
    // Generate key name if not provided (using current timestamp)
    let key_name = key_name.unwrap_or(Local::now().format("%Y-%m-%d-%H:%M:%S").to_string());

    // Only show prompts if stdin is a terminal (interactive mode)
    if io::stdin().is_terminal() {
        println!("🔐 Adding authorized client...");
        println!("   Username: {}", username);
        println!("   Key name: {}", key_name);
        println!();

        // Prompt for public key
        println!("Enter public key (paste, then press Ctrl-D on a new line):");
        println!("  Supported algorithms:");
        println!("  - RSA (2048-8192 bits)");
        println!("  - ED25519 (256 bits)");
        println!("  - ECDSA (256, 384 bits)");
        println!("  Supported formats:");
        println!("  - OpenSSH (single line)");
        println!("  - PKCS#8 PEM (multi-line)");
        println!("  - PKCS#1 PEM (multi-line)");
        print!("> ");
        io::stdout().flush().context("Failed to flush stdout")?;
    }

    let mut public_key = String::new();
    io::stdin()
        .read_to_string(&mut public_key)
        .context("Failed to read public key")?;
    let public_key = public_key.trim().to_string();

    // Warn but still create the user without keys if no public key is provided
    if public_key.is_empty() {
        println!();
        println!("⚠️ Warning: No public key provided");

        let result = add_authorized_client(&username)?;

        match result {
            InsertUserResult::CreatedUser => {
                println!("✅ Created user '{}' without any keys", username);
            }
            InsertUserResult::UserExists => {
                println!("📋 User '{}' already exists", username);
            }
        }

        return Ok(());
    }

    // Validates that the string is a valid public key by parsing it.
    let public_key = PublicKey::parse(&public_key).context("Failed to parse public key")?;

    // Ensure the client exists first, then add the key
    let user_result = add_authorized_client(&username)?;
    let key_result = add_key_for_authorized_client(&username, key_name.clone(), public_key)?;

    println!();
    match key_result {
        InsertKeyResult::AddedKey => {
            if user_result == InsertUserResult::CreatedUser {
                println!(
                    "✅ Created user '{}' and added key '{}'",
                    username, key_name
                );
            } else {
                println!("✅ Added key '{}' to user '{}'", key_name, username);
            }
        }
        InsertKeyResult::KeyNameExists => {
            bail!(
                "Key with name '{}' already exists for user '{}'",
                key_name,
                username
            );
        }
        InsertKeyResult::UserNotFound => {
            bail!("User '{}' not found", username);
        }
    }

    Ok(())
}

/// Handles the `pie auth remove` subcommand.
async fn handle_auth_remove_subcommand(username: String, key_name: Option<String>) -> Result<()> {
    match key_name {
        Some(key_name) => {
            // Remove a specific key
            let result = remove_authorized_client_key(&username, &key_name)?;

            match result {
                RemoveKeyResult::RemovedKey => {
                    println!("✅ Removed key '{}' from user '{}'", key_name, username);
                }
                RemoveKeyResult::KeyNotFound => {
                    bail!("Key '{}' not found for user '{}'", key_name, username);
                }
                RemoveKeyResult::UserNotFound => {
                    bail!("User '{}' not found", username);
                }
            }
        }
        None => {
            // Remove entire user
            let result = remove_authorized_client(&username)?;

            match result {
                RemoveUserResult::RemovedUser => {
                    println!("✅ Removed user '{}' and all associated keys", username);
                }
                RemoveUserResult::UserNotFound => {
                    bail!("User '{}' not found", username);
                }
            }
        }
    }
    Ok(())
}

/// Handles the `pie auth list` subcommand.
async fn handle_auth_list_subcommand() -> Result<()> {
    let auth_path = path::get_authorized_clients_path()?;

    // Check if the file exists
    if !auth_path.exists() {
        println!("📋 No authorized clients found.");
        println!("    (File not found at {:?})", auth_path);
        return Ok(());
    }

    // Read and parse the authorized clients file
    let authorized_clients = AuthorizedClients::load(&auth_path)?;

    // Check if there are any clients
    if authorized_clients.is_empty() {
        println!("📋 No authorized clients found.");
        return Ok(());
    }

    // Print the list of authorized clients
    println!("📋 Authorized clients:");
    println!("    File: {:?}", auth_path);
    println!();

    // Collect and sort usernames for consistent output
    let mut usernames: Vec<&String> = authorized_clients
        .iter()
        .map(|(username, _)| username)
        .collect();
    usernames.sort();

    for username in usernames {
        if let Some(client_keys) = authorized_clients.get(username) {
            let key_count = client_keys.len();
            let key_word = if key_count == 1 { "key" } else { "keys" };
            println!("  {} ({} {}):", username, key_count, key_word);

            // List individual keys with their names
            // Collect and sort key names for consistent output
            let mut key_names: Vec<&String> = client_keys.iter().map(|(name, _)| name).collect();
            key_names.sort();
            for key_name in key_names {
                println!("    - {}", key_name);
            }
        }
    }

    println!();
    println!("Total: {} authorized client(s)", authorized_clients.len());

    Ok(())
}

/// Creates an authorized client entry in the `authorized_clients.toml` file.
/// Does nothing if the client already exists.
fn add_authorized_client(username: &str) -> Result<InsertUserResult> {
    let auth_path = path::get_authorized_clients_path()?;

    // Create the directory if it doesn't exist
    if let Some(parent) = auth_path.parent() {
        fs::create_dir_all(parent).context(format!(
            "Failed to create authorized clients directory at {:?}",
            parent
        ))?;
    }

    // Read existing authorized clients or create a new structure
    let mut authorized_clients = if auth_path.exists() {
        AuthorizedClients::load(&auth_path)?
    } else {
        AuthorizedClients::default()
    };

    // Add the user without keys
    let result = authorized_clients.insert_user(username);

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    Ok(result)
}

/// Adds a key to an authorized client in the `authorized_clients.toml` file.
fn add_key_for_authorized_client(
    username: &str,
    key_name: String,
    public_key: PublicKey,
) -> Result<InsertKeyResult> {
    let auth_path = path::get_authorized_clients_path()?;

    // Read existing authorized clients
    let mut authorized_clients =
        AuthorizedClients::load(&auth_path).context("Failed to load authorized clients file")?;

    // Add the key to the user
    let result = authorized_clients.insert_key_for_user(username, key_name, public_key);

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    Ok(result)
}

/// Removes an authorized client from the `authorized_clients.toml` file.
fn remove_authorized_client(username: &str) -> Result<RemoveUserResult> {
    let auth_path = path::get_authorized_clients_path()?;

    // Check if the file exists
    if !auth_path.exists() {
        bail!(
            "Authorized clients file not found at {:?}. No clients to remove.",
            auth_path
        );
    }

    // Read existing authorized clients
    let mut authorized_clients = AuthorizedClients::load(&auth_path)?;

    // Check if user exists and get key count for confirmation prompt
    let client_keys = authorized_clients.get(username);
    if client_keys.is_none() {
        return Ok(RemoveUserResult::UserNotFound);
    }

    // Get the number of keys for the user
    let key_count = client_keys.unwrap().len();

    // Prompt for confirmation if stdin is a terminal
    if io::stdin().is_terminal() {
        print!(
            "⚠️  This will remove user '{}' and all {} key(s). Continue? (y/N): ",
            username, key_count
        );
        io::stdout().flush().context("Failed to flush stdout")?;

        // Read confirmation from stdin
        let mut response = String::new();
        io::stdin()
            .read_line(&mut response)
            .context("Failed to read confirmation")?;

        let response = response.trim().to_lowercase();
        if response != "y" {
            bail!("Operation cancelled.");
        }
    }

    // Remove the user
    let result = authorized_clients.remove_user(username);

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    Ok(result)
}

/// Removes a specific key from an authorized client in the `authorized_clients.toml` file.
fn remove_authorized_client_key(username: &str, key_name: &str) -> Result<RemoveKeyResult> {
    let auth_path = path::get_authorized_clients_path()?;

    // Check if the file exists
    if !auth_path.exists() {
        bail!(
            "Authorized clients file not found at {:?}. No clients to remove.",
            auth_path
        );
    }

    // Read existing authorized clients
    let mut authorized_clients = AuthorizedClients::load(&auth_path)?;

    // Remove the specific key
    let result = authorized_clients.remove_key(username, key_name);

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    Ok(result)
}
