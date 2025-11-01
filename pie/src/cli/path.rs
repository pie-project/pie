//! Path utilities for the Pie CLI.
//!
//! This module provides functions for working with Pie-specific paths and directories.

use anyhow::{Result, anyhow};
use std::{env, path::PathBuf};

pub fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Failed to find home directory"))?;
        Ok(home.join(".pie"))
    }
}

pub fn get_pie_cache_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Failed to find home directory"))?;
        Ok(home.join(".cache").join("pie"))
    }
}

pub fn get_default_config_path() -> Result<PathBuf> {
    let pie_home = get_pie_home()?;
    let config_path = pie_home.join("config.toml");
    Ok(config_path)
}

pub fn get_shell_history_path() -> Result<PathBuf> {
    let pie_home = get_pie_home()?;
    let history_path = pie_home.join(".pie_history");
    Ok(history_path)
}

pub fn get_authorized_users_path() -> Result<PathBuf> {
    let pie_home = get_pie_home()?;
    let authorized_users_path = pie_home.join("authorized_users.toml");
    Ok(authorized_users_path)
}

/// Helper for clap to expand `~` in path arguments.
pub fn expand_tilde(s: &str) -> Result<PathBuf, std::convert::Infallible> {
    Ok(PathBuf::from(shellexpand::tilde(s).as_ref()))
}
