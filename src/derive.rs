use anyhow::{Context, Result};

pub fn read_config_file(path: &std::path::Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("reading config file {}", path.display()))
}

pub use worker::standalone::derive_standalone;
