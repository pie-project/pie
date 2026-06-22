//! `~/.pie/...` path helpers, mirroring `pie/src/pie/path.py`.
//!
//! `pie::path::get_pie_home()` already lives in the runtime crate and
//! is reused here. This module adds the standalone-side conveniences
//! (`config.toml`, `authorized_users.toml`, etc.) without forcing the
//! runtime crate to know about them.

use std::path::PathBuf;

/// `$PIE_HOME` (default `~/.pie`).
pub fn pie_home() -> PathBuf {
    pie::path::get_pie_home()
}

/// `~/.pie/config.toml` — default config file.
pub fn default_config_path() -> PathBuf {
    pie_home().join("config.toml")
}

/// `~/.pie/authorized_users.toml` — auth keys file.
pub fn authorized_users_path() -> PathBuf {
    pie_home().join("authorized_users.toml")
}
