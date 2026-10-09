use std::path::PathBuf;

use anyhow::{Context, Result};

use super::{BootSpec, GlobalArgs};
use crate::paths;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Flag,
    Env,
    Default,
}

impl Origin {
    pub fn is_explicit(self) -> bool {
        matches!(self, Self::Flag | Self::Env)
    }

    pub fn describe(self) -> &'static str {
        match self {
            Self::Flag => "--config flag",
            Self::Env => "$PIE_CONFIG",
            Self::Default => "$PIE_HOME default",
        }
    }
}

pub fn cli_config_path(global: &GlobalArgs) -> (PathBuf, Origin) {
    if let Some(flag) = global.config.as_deref() {
        return (PathBuf::from(flag), Origin::Flag);
    }
    if let Ok(env) = std::env::var("PIE_CONFIG")
        && !env.trim().is_empty()
    {
        return (PathBuf::from(env), Origin::Env);
    }
    (paths::pie_home_file("config.toml"), Origin::Default)
}

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

pub(crate) fn source(spec: &BootSpec, global: &GlobalArgs) -> Result<String> {
    let (path, origin) = resolve_path(spec, global);
    match std::fs::read_to_string(&path) {
        Ok(s) => Ok(s),
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
