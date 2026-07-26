//! Config sourcing — locate and read the config file into a `String`.
//!
//! `startup` deliberately stays format-agnostic: it produces the config
//! *string*; the role lib's `Config::parse(&str)` owns all domain parsing and
//! validation. Resolution order for the path: `--config` flag → `$PIE_CONFIG`
//! env → `$PIE_HOME/<default_config_filename>`. A missing default file is not an
//! error (the role lib applies its own defaults from an empty string); a missing
//! *explicitly requested* file is.
//!
//! [`resolve_path`] is the single source of truth for that order. Both the boot
//! path ([`source`], called by every daemon) and the `pie-env` command go
//! through it, so the tool can never report a path the daemon wouldn't read.

use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::{BootSpec, GlobalArgs, paths};

/// Where a resolved config path came from — which also decides whether a
/// missing file is fatal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// The `--config` flag. Explicitly requested, so a missing file is an error.
    Flag,
    /// The `$PIE_CONFIG` environment variable. Also explicit, also fatal.
    Env,
    /// `$PIE_HOME/<default_config_filename>`. Absent is normal: the role lib
    /// falls back to its own defaults.
    Default,
}

impl Origin {
    /// Whether a file at this path was explicitly asked for.
    pub fn is_explicit(self) -> bool {
        matches!(self, Self::Flag | Self::Env)
    }

    /// Human-readable provenance, for `pie-env` and error messages.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Flag => "--config flag",
            Self::Env => "$PIE_CONFIG",
            Self::Default => "$PIE_HOME default",
        }
    }
}

/// Resolve *which* config file this role would read, without touching the disk.
pub(crate) fn resolve_path(spec: &BootSpec, global: &GlobalArgs) -> (PathBuf, Origin) {
    if let Some(flag) = global.config.as_deref() {
        return (PathBuf::from(flag), Origin::Flag);
    }
    if let Ok(env) = std::env::var("PIE_CONFIG") {
        return (PathBuf::from(env), Origin::Env);
    }
    (
        paths::pie_home_file(spec.default_config_filename),
        Origin::Default,
    )
}

/// Resolve the config path and read it to a string (see module docs).
pub(crate) fn source(spec: &BootSpec, global: &GlobalArgs) -> Result<String> {
    let (path, origin) = resolve_path(spec, global);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s),
        // No config present at the default location: hand the role lib an empty
        // string so it applies its own defaults.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && !origin.is_explicit() => {
            tracing::debug!(path = %path.display(), "no config file; using role defaults");
            Ok(String::new())
        }
        Err(e) => Err(e).with_context(|| match origin {
            Origin::Flag => format!("reading --config {}", path.display()),
            Origin::Env => format!("reading $PIE_CONFIG {}", path.display()),
            Origin::Default => format!("reading {}", path.display()),
        }),
    }
}
