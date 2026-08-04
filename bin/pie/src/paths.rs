//! `~/.pie/...` path helpers for the standalone CLI, layered on `startup`'s
//! `pie_home`. These are worker-domain conveniences (`config.toml`,
//! `authorized_users.toml`) that `startup::paths` deliberately doesn't carry —
//! they belong with the CLI ops that own them (auth).

use std::path::PathBuf;

// No `default_config_path` here. It was a fourth implementation of "which
// config file?", one that knew only the $PIE_HOME leg and so silently ignored
// --config and $PIE_CONFIG. `startup::cli_config_path` is the one that answers
// it, and every caller now asks that.

/// `$PIE_HOME/authorized_users.toml` — the auth backend's keys file.
pub fn authorized_users_path() -> PathBuf {
    startup::paths::pie_home().join("authorized_users.toml")
}
