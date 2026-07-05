//! `pie auth { add | remove | list }` — manage `~/.pie/authorized_users.toml`.
//!
//! File format matches the Rust runtime's [`AuthorizedUsers`] schema:
//!
//! ```toml
//! [users.<username>.keys]
//! <key_name> = "<openssh-or-pem-public-key>"
//! ```
//!
//! Same shape `pie/src/pie_cli/commands/auth.py` reads + writes.

use std::collections::BTreeMap;
use std::io::{IsTerminal, Read};
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use chrono::Local;
use clap::Subcommand;
use serde::{Deserialize, Serialize};

use crate::paths;

/// Top-level [users] table for the TOML file. Each user has a nested
/// `keys` map (key name → OpenSSH/PEM public key string).
#[derive(Default, Deserialize, Serialize)]
struct AuthorizedUsersFile {
    #[serde(default)]
    users: BTreeMap<String, UserEntry>,
}

#[derive(Default, Deserialize, Serialize)]
struct UserEntry {
    #[serde(default)]
    keys: BTreeMap<String, String>,
}

#[derive(Subcommand, Debug)]
pub enum AuthCmd {
    /// Add a public key for a user. Reads the key from stdin (one
    /// OpenSSH or PEM blob). If `key_name` is omitted, a timestamp
    /// is used.
    Add {
        username: String,
        key_name: Option<String>,
    },
    /// Remove either a single key or the whole user entry.
    Remove {
        username: String,
        /// If omitted, remove the entire user entry.
        key_name: Option<String>,
    },
    /// List authorized users + key names.
    List,
}

pub fn run(cmd: AuthCmd) -> Result<()> {
    let path = paths::authorized_users_path();
    match cmd {
        AuthCmd::Add { username, key_name } => add(&path, username, key_name),
        AuthCmd::Remove { username, key_name } => remove(&path, username, key_name),
        AuthCmd::List => list(&path),
    }
}

fn load(path: &Path) -> Result<AuthorizedUsersFile> {
    if !path.exists() {
        return Ok(AuthorizedUsersFile::default());
    }
    let content = std::fs::read_to_string(path).map_err(|e| anyhow!("read {path:?}: {e}"))?;
    toml::from_str(&content).map_err(|e| anyhow!("parse {path:?}: {e}"))
}

fn save(path: &Path, file: &AuthorizedUsersFile) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow!("create parent dir {parent:?}: {e}"))?;
    }
    let serialized = toml::to_string(file).map_err(|e| anyhow!("serialize TOML: {e}"))?;
    std::fs::write(path, serialized).map_err(|e| anyhow!("write {path:?}: {e}"))?;
    // Tighten permissions on unix — mirrors what the runtime's
    // `check_file_permissions` expects (no group/world access).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(path, perms).map_err(|e| anyhow!("chmod 600 {path:?}: {e}"))?;
    }
    Ok(())
}

fn add(path: &Path, username: String, key_name: Option<String>) -> Result<()> {
    let key_name = key_name.unwrap_or_else(default_key_name);

    let mut stdin = String::new();
    if std::io::stdin().is_terminal() {
        eprintln!("Adding authorized user: {username}");
        eprintln!("Key name: {key_name}");
        eprintln!("Paste public key (OpenSSH or PEM), then EOF (Ctrl-D):");
    }
    std::io::stdin()
        .read_to_string(&mut stdin)
        .map_err(|e| anyhow!("read public key from stdin: {e}"))?;
    let pubkey = stdin.trim();

    let mut file = load(path)?;
    let user_existed = file.users.contains_key(&username);
    let user = file.users.entry(username.clone()).or_default();

    if pubkey.is_empty() {
        let had_keys = !user.keys.is_empty();
        // Drop the mutable borrow before save().
        let _ = user;
        save(path, &file)?;
        if !user_existed && !had_keys {
            println!("✓ Created user '{username}' without keys");
        } else {
            println!("(no key provided; user '{username}' unchanged)");
        }
        return Ok(());
    }

    if user.keys.contains_key(&key_name) {
        bail!(
            "key '{key_name}' already exists for user '{username}' \
             (use `pie auth remove {username} {key_name}` first)"
        );
    }
    user.keys.insert(key_name.clone(), pubkey.to_string());
    let _ = user;
    save(path, &file)?;
    println!("✓ Added key '{key_name}' to '{username}'");
    Ok(())
}

fn remove(path: &Path, username: String, key_name: Option<String>) -> Result<()> {
    if !path.exists() {
        bail!("no authorized users file at {path:?}");
    }
    let mut file = load(path)?;
    let Some(user) = file.users.get_mut(&username) else {
        bail!("user '{username}' not found");
    };

    if let Some(name) = key_name {
        if user.keys.remove(&name).is_none() {
            bail!("key '{name}' not found for user '{username}'");
        }
        save(path, &file)?;
        println!("✓ Removed key '{name}' from '{username}'");
    } else {
        let count = user.keys.len();
        file.users.remove(&username);
        save(path, &file)?;
        println!("✓ Removed user '{username}' and {count} key(s)");
    }
    Ok(())
}

fn list(path: &Path) -> Result<()> {
    let file = load(path)?;
    if file.users.is_empty() {
        println!("(no authorized users)");
        return Ok(());
    }
    for (user, entry) in &file.users {
        let names: Vec<&str> = entry.keys.keys().map(|s| s.as_str()).collect();
        let summary = if names.is_empty() {
            "no keys".to_string()
        } else {
            names.join(", ")
        };
        println!("{user:<20} {} key(s): {summary}", entry.keys.len());
    }
    println!("\n{}", path.display());
    Ok(())
}

fn default_key_name() -> String {
    Local::now().format("%Y-%m-%d-%H:%M:%S").to_string()
}
