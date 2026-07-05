//! `~/.pie/...` path helpers for the standalone CLI, layered on bootstrap's
//! `pie_home`. These are worker-domain conveniences (`config.toml`,
//! `authorized_users.toml`) that `bootstrap::paths` deliberately doesn't carry —
//! they belong with the CLI ops that own them (config / auth).

use std::path::PathBuf;

/// `$PIE_HOME/config.toml` — the default standalone config file.
pub fn default_config_path() -> PathBuf {
    bootstrap::paths::pie_home().join("config.toml")
}

/// `$PIE_HOME/authorized_users.toml` — the auth backend's keys file.
pub fn authorized_users_path() -> PathBuf {
    bootstrap::paths::pie_home().join("authorized_users.toml")
}
