//! Path utilities for the Pie CLI.
//!
//! This module provides functions for working with Pie-specific paths and directories.

use anyhow::{Context, Result, anyhow, bail};
use std::env;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

pub fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_CLI_HOME") {
        Ok(PathBuf::from(path))
    } else {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Failed to find home directory"))?;
        Ok(home.join(".pie_cli"))
    }
}

pub fn get_default_config_path() -> Result<PathBuf> {
    let pie_home = get_pie_home()?;
    let config_path = pie_home.join("config.toml");
    Ok(config_path)
}

/// Helper for clap to expand `~` in path arguments.
pub fn expand_tilde(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(shellexpand::tilde(s).as_ref()))
}

/// Check file permissions and bail if they're not 0o600 (Unix only).
#[cfg(unix)]
pub fn check_private_key_permissions(path: &Path) -> Result<()> {
    use std::fs;

    let metadata = fs::metadata(path).context(format!(
        "Failed to read metadata for file at '{}'",
        path.display()
    ))?;
    let permissions = metadata.permissions();
    let mode = permissions.mode() & 0o777;

    // Check if permissions are too permissive (should be 0o600)
    if mode != 0o600 {
        bail!(
            "Private key file at '{}' has insecure permissions: {:o}. \
            Run: `chmod 600 '{}'`",
            path.display(),
            mode,
            path.display()
        );
    }
    Ok(())
}
