//! Authorization management commands for Pie.
//!
//! This module implements the `pie auth` commands for managing authorized clients
//! by adding or removing their public keys in the `authorized_clients.toml` file.

use super::path;
use crate::auth::{AuthorizedClients, InsertKeyResult, PublicKey, RemoveKeyResult};
use anyhow::{Context, Result, bail};
use clap::Subcommand;
use std::{
    fs,
    io::{self, Read, Write},
};

#[derive(Subcommand, Debug)]
pub enum AuthCommands {
    /// Add an authorized client and its public key.
    /// The public key can be in OpenSSH, PKCS#8 PEM, or PKCS#1 PEM format.
    /// A user can have multiple public keys.
    Add,
    /// Remove an authorized client and all its public keys.
    Remove,
    /// Remove a specific key from an authorized client.
    RemoveKey,
    /// List all authorized clients and their keys.
    List,
}

/// Handles the `pie auth` command.
/// These commands take input from stdin to avoid exposing sensitive information
/// to the shell history.
pub async fn handle_auth_command(command: AuthCommands) -> Result<()> {
    match command {
        AuthCommands::Add => handle_auth_add_subcommand().await,
        AuthCommands::Remove => handle_auth_remove_subcommand().await,
        AuthCommands::RemoveKey => handle_auth_remove_key_subcommand().await,
        AuthCommands::List => handle_auth_list_subcommand().await,
    }
}

/// Handles the `pie auth add` subcommand.
async fn handle_auth_add_subcommand() -> Result<()> {
    println!("🔐 Adding authorized client...");
    println!();

    // Prompt for username
    print!("Enter username: ");
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut username = String::new();
    io::stdin()
        .read_line(&mut username)
        .context("Failed to read username")?;
    let username = username.trim().to_string();

    if username.is_empty() {
        bail!("Username cannot be empty");
    }

    // Prompt for key name
    print!("Enter key name: ");
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut key_name = String::new();
    io::stdin()
        .read_line(&mut key_name)
        .context("Failed to read key name")?;
    let key_name = key_name.trim().to_string();

    if key_name.is_empty() {
        bail!("Key name cannot be empty");
    }

    // Prompt for public key
    println!();
    println!("Enter public key (paste multi-line PEM, then press Ctrl-D on a new line):");
    println!("  Supported formats:");
    println!("  - OpenSSH (single line)");
    println!("  - PKCS#8 PEM (multi-line)");
    println!("  - PKCS#1 PEM (multi-line)");
    print!("> ");
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut public_key = String::new();
    io::stdin()
        .read_to_string(&mut public_key)
        .context("Failed to read public key")?;
    let public_key = public_key.trim().to_string();

    if public_key.is_empty() {
        bail!("Public key cannot be empty");
    }

    // Validates that the string is a valid public key by parsing it.
    let public_key = PublicKey::parse(&public_key).context("Failed to parse public key")?;

    // Add the key to `authorized_clients.toml`
    add_authorized_client(&username, key_name.clone(), public_key)?;

    println!();
    println!(
        "✅ Successfully added public key '{}' for user '{}'",
        key_name, username
    );

    Ok(())
}

/// Handles the `pie auth remove` subcommand.
async fn handle_auth_remove_subcommand() -> Result<()> {
    println!("🔐 Removing authorized client...");
    println!();

    // Prompt for username
    print!("Enter username to remove: ");
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut username = String::new();
    io::stdin()
        .read_line(&mut username)
        .context("Failed to read username")?;
    let username = username.trim().to_string();

    if username.is_empty() {
        bail!("Username cannot be empty");
    }

    // Remove the client from authorized_clients.toml
    remove_authorized_client(&username)?;

    println!("✅ Successfully removed user '{}'", username);
    Ok(())
}

/// Handles the `pie auth remove-key` subcommand.
async fn handle_auth_remove_key_subcommand() -> Result<()> {
    println!("🔐 Removing specific key from authorized client...");
    println!();

    // Prompt for username
    print!("Enter username: ");
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut username = String::new();
    io::stdin()
        .read_line(&mut username)
        .context("Failed to read username")?;
    let username = username.trim().to_string();

    if username.is_empty() {
        bail!("Username cannot be empty");
    }

    // Prompt for key name
    print!("Enter key name to remove: ");
    io::stdout().flush().context("Failed to flush stdout")?;

    let mut key_name = String::new();
    io::stdin()
        .read_line(&mut key_name)
        .context("Failed to read key name")?;
    let key_name = key_name.trim().to_string();

    if key_name.is_empty() {
        bail!("Key name cannot be empty");
    }

    // Remove the specific key from authorized_clients.toml
    remove_authorized_client_key(&username, &key_name)?;

    println!(
        "✅ Successfully removed key '{}' from user '{}'",
        key_name, username
    );
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

/// Adds an authorized client to the `authorized_clients.toml` file.
fn add_authorized_client(username: &str, key_name: String, public_key: PublicKey) -> Result<()> {
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

    // Add or update the user's keys
    let result = authorized_clients.insert(username, key_name.clone(), public_key);

    match result {
        InsertKeyResult::CreatedUser => {
            println!("Created new user '{}' with key '{}'", username, key_name);
        }
        InsertKeyResult::AddedKey => {
            println!(
                "Added new key '{}' to existing user '{}'",
                key_name, username
            );
        }
        InsertKeyResult::KeyNameExists => {
            bail!(
                "Key with name '{}' already exists for user '{}'",
                key_name,
                username
            );
        }
    }

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    println!("Authorized clients file updated at {:?}", auth_path);
    Ok(())
}

/// Removes an authorized client from the `authorized_clients.toml` file.
fn remove_authorized_client(username: &str) -> Result<()> {
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

    // Check if user exists
    let client_keys = authorized_clients.get(username);
    if client_keys.is_none() {
        bail!("User '{}' not found in authorized clients", username);
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
    }

    // Read confirmation from stdin
    let mut response = String::new();
    io::stdin()
        .read_line(&mut response)
        .context("Failed to read confirmation")?;

    let response = response.trim().to_lowercase();
    if response != "y" {
        bail!("Operation cancelled.");
    }

    // Remove the user
    if authorized_clients.remove(username).is_some() {
        println!("Removed user '{}' and all associated keys", username);
    } else {
        bail!("User '{}' not found in authorized clients", username);
    }

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    println!("Authorized clients file updated at {:?}", auth_path);
    Ok(())
}

/// Removes a specific key from an authorized client in the `authorized_clients.toml` file.
fn remove_authorized_client_key(username: &str, key_name: &str) -> Result<()> {
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

    match result {
        RemoveKeyResult::RemovedLastKey => {
            println!(
                "Removed last key '{}' from user '{}', user entry removed",
                key_name, username
            );
        }
        RemoveKeyResult::RemovedKey => {
            println!("Removed key '{}' from user '{}'", key_name, username);
        }
        RemoveKeyResult::KeyNotFound => {
            bail!("Key '{}' not found for user '{}'", key_name, username);
        }
        RemoveKeyResult::UserNotFound => {
            bail!("User '{}' not found in authorized clients", username);
        }
    }

    // Serialize and write back to file
    authorized_clients.save(&auth_path)?;

    println!("Authorized clients file updated at {:?}", auth_path);
    Ok(())
}
