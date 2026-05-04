//! Startup TOML for the dummy driver.
//!
//! Mirrors `driver/portable`'s startup-TOML shape where it overlaps
//! (`[shmem]`) and adds a `[dummy]` section for the knobs the dummy
//! needs in lieu of model introspection. The standalone server emits
//! this file via `embedded_driver::write_dummy_startup_toml`.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub shmem: ShmemConfig,
    pub dummy: DummyConfig,
}

#[derive(Debug, Deserialize)]
pub struct ShmemConfig {
    pub name: String,
    pub num_slots: usize,
    pub req_buf: usize,
    pub resp_buf: usize,
    #[serde(default)]
    pub spin_us: u64,
}

#[derive(Debug, Deserialize)]
pub struct DummyConfig {
    pub kv_page_size: u32,
    pub max_num_kv_pages: u32,
    pub max_batch_tokens: u32,
    pub max_batch_size: u32,
    pub vocab_size: u32,
    pub arch_name: String,
    pub max_model_len: u32,
    #[serde(default = "default_activation_dtype")]
    pub activation_dtype: String,
    #[serde(default)]
    pub random_seed: u64,
    /// Snapshot dir advertised to the runtime. The dummy doesn't read
    /// it, but `bootstrap_translate.rs` joins it with `tokenizer.json`
    /// to find the tokenizer.
    #[serde(default)]
    pub snapshot_dir: String,
}

fn default_activation_dtype() -> String {
    "bfloat16".to_string()
}

pub fn load(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read startup TOML {path:?}"))?;
    let cfg: Config =
        toml::from_str(&text).with_context(|| format!("parse startup TOML {path:?}"))?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_minimal() {
        let txt = r#"
[shmem]
name = "/pie_shmem_g0"
num_slots = 8
req_buf = 4194304
resp_buf = 4194304
spin_us = 0

[dummy]
kv_page_size = 16
max_num_kv_pages = 256
max_batch_tokens = 4096
max_batch_size = 128
vocab_size = 32000
arch_name = "qwen3"
max_model_len = 4096
snapshot_dir = "/tmp/snap"
"#;
        let cfg: Config = toml::from_str(txt).unwrap();
        assert_eq!(cfg.shmem.name, "/pie_shmem_g0");
        assert_eq!(cfg.dummy.vocab_size, 32000);
        assert_eq!(cfg.dummy.arch_name, "qwen3");
    }
}
