//! Path utilities for the Pie CLI.
//!
//! This module provides functions for working with Pie-specific paths and directories.

use anyhow::{Result, anyhow};
use std::{env, path::PathBuf};

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
