//! `$PIE_HOME` and the well-known files under it.
//!
//! Re-implemented here (rather than reaching into the runtime crate) so
//! `bootstrap` depends on no role/runtime library — it is the dependency floor
//! for the bins. Empty `$PIE_HOME` is ignored so callers get the same fallback
//! semantics everywhere (`$PIE_HOME` else `~/.pie`).

use std::path::PathBuf;

/// `$PIE_HOME` if set, else `~/.pie` (falling back to `.pie` in the cwd if the
/// home directory can't be resolved).
pub fn pie_home() -> PathBuf {
    if let Ok(dir) = std::env::var("PIE_HOME") {
        if !dir.trim().is_empty() {
            return PathBuf::from(dir);
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pie")
}

/// A file directly under `$PIE_HOME` (e.g. `config.toml`).
pub fn pie_home_file(name: &str) -> PathBuf {
    pie_home().join(name)
}
