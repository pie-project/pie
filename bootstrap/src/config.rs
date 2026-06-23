//! Config sourcing — locate and read the config file into a `String`.
//!
//! `bootstrap` deliberately stays format-agnostic: it produces the config
//! *string*; the role lib's `Config::parse(&str)` owns all domain parsing and
//! validation. Resolution order for the path: `--config` flag → `$PIE_CONFIG`
//! env → `$PIE_HOME/<default_config_filename>`. A missing default file is not an
//! error (the role lib applies its own defaults from an empty string); a missing
//! *explicitly requested* file is.

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::{BootSpec, GlobalArgs, paths};

/// Resolve the config path and read it to a string (see module docs).
pub(crate) fn source(spec: &BootSpec, global: &GlobalArgs) -> Result<String> {
    if let Some(flag) = global.config.as_deref() {
        let path = PathBuf::from(flag);
        return std::fs::read_to_string(&path)
            .with_context(|| format!("reading --config {}", path.display()));
    }
    if let Ok(env) = std::env::var("PIE_CONFIG") {
        let path = PathBuf::from(env);
        return std::fs::read_to_string(&path)
            .with_context(|| format!("reading $PIE_CONFIG {}", path.display()));
    }
    let default = paths::pie_home_file(spec.default_config_filename);
    match std::fs::read_to_string(&default) {
        Ok(s) => Ok(s),
        // No config present at the default location: hand the role lib an empty
        // string so it applies its own defaults.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::debug!(path = %default.display(), "no config file; using role defaults");
            Ok(String::new())
        }
        Err(e) => Err(e).with_context(|| format!("reading {}", default.display())),
    }
}
