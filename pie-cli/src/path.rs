//! Path utilities for the Pie CLI.
//!
//! This module provides functions for working with Pie-specific paths and directories.

use anyhow::{Result, anyhow};
use std::{env, path::PathBuf};

pub fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        dirs::cache_dir()
            .map(|p| p.join("pie"))
            .ok_or_else(|| anyhow!("Failed to find home dir"))
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
