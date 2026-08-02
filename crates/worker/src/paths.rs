//! `$PIE_HOME` resolution for the worker role library.
//!
//! Re-implemented here (rather than depending on the bin-layer process
//! skeleton) so the role library stays a pure leaf: nothing under `bin/` may be
//! a dependency of a role lib. Mirrors `bootstrap::paths` — empty `$PIE_HOME` is
//! ignored so both layers resolve identically.

use std::path::PathBuf;

/// `$PIE_HOME` if set and non-empty, else `~/.pie` (falling back to `.pie` in
/// the cwd if the home directory can't be resolved).
pub fn pie_home() -> PathBuf {
    if let Ok(dir) = std::env::var("PIE_HOME")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pie")
}
