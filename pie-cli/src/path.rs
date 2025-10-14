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
