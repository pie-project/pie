//! Startup TOML for the dummy driver.
//!
//! Mirrors `driver/portable`'s startup-TOML shape where it overlaps
//! (`[shmem]`) and adds a `[dummy]` section for the knobs the dummy
//! needs in lieu of model introspection. The standalone server emits
//! this file via `embedded_driver::write_dummy_startup_toml`.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Deserializer};
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
    #[serde(
        default = "default_spin_budget_us",
        deserialize_with = "deserialize_u64_or_string"
    )]
    pub spin_budget_us: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DummyConfig {
    pub kv_page_size: u32,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
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

impl DummyConfig {
    pub fn derived_total_pages(&self) -> u32 {
        let forward_pages = pages_for_tokens(self.max_forward_tokens, self.kv_page_size);
        let context_pages = pages_for_tokens(self.max_model_len, self.kv_page_size);
        let request_pages = self.max_forward_requests.saturating_mul(2);
        256.max(forward_pages).max(context_pages).max(request_pages)
    }
}

fn pages_for_tokens(tokens: u32, page_size: u32) -> u32 {
    tokens.saturating_add(page_size - 1) / page_size
}

fn default_activation_dtype() -> String {
    "bfloat16".to_string()
}

fn default_spin_budget_us() -> u64 {
    1_000
}

fn deserialize_u64_or_string<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct U64Visitor;

    impl Visitor<'_> for U64Visitor {
        type Value = u64;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a non-negative integer or decimal integer string")
        }

        fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(value)
        }

        fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            u64::try_from(value).map_err(|_| E::custom("integer must be non-negative"))
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            value.parse::<u64>().map_err(E::custom)
        }
    }

    deserializer.deserialize_any(U64Visitor)
}

pub fn load(path: &Path) -> Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read startup TOML {path:?}"))?;
    let cfg: Config =
        toml::from_str(&text).with_context(|| format!("parse startup TOML {path:?}"))?;
    ensure!(cfg.dummy.kv_page_size > 0, "dummy.kv_page_size must be > 0");
    ensure!(
        cfg.dummy.max_forward_tokens > 0,
        "dummy.max_forward_tokens must be > 0"
    );
    ensure!(
        cfg.dummy.max_forward_requests > 0,
        "dummy.max_forward_requests must be > 0"
    );
    ensure!(
        cfg.dummy.max_model_len > 0,
        "dummy.max_model_len must be > 0"
    );
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

[dummy]
kv_page_size = 16
max_forward_tokens = 4096
max_forward_requests = 128
vocab_size = 32000
arch_name = "qwen3"
max_model_len = 4096
snapshot_dir = "/tmp/snap"
"#;
        let cfg: Config = toml::from_str(txt).unwrap();
        assert_eq!(cfg.shmem.name, "/pie_shmem_g0");
        assert_eq!(cfg.shmem.spin_budget_us, 1_000);
        assert_eq!(cfg.dummy.vocab_size, 32000);
        assert_eq!(cfg.dummy.arch_name, "qwen3");
        assert_eq!(cfg.dummy.derived_total_pages(), 256);
    }

    #[test]
    fn shmem_spin_budget_accepts_string_u64() {
        let txt = r#"
[shmem]
name = "/pie_shmem_g0"
num_slots = 8
req_buf = 4194304
resp_buf = 4194304
spin_budget_us = "18446744073709551615"

[dummy]
kv_page_size = 16
max_forward_tokens = 4096
max_forward_requests = 128
vocab_size = 32000
arch_name = "qwen3"
max_model_len = 4096
"#;
        let cfg: Config = toml::from_str(txt).unwrap();
        assert_eq!(cfg.shmem.spin_budget_us, u64::MAX);
    }

    #[test]
    fn rejects_legacy_page_count_key() {
        let txt = r#"
[shmem]
name = "/pie_shmem_g0"
num_slots = 8
req_buf = 4194304
resp_buf = 4194304

[dummy]
kv_page_size = 16
max_num_kv_pages = 256
max_forward_tokens = 4096
max_forward_requests = 128
vocab_size = 32000
arch_name = "qwen3"
max_model_len = 4096
"#;
        let err = toml::from_str::<Config>(txt).unwrap_err().to_string();
        assert!(err.contains("max_num_kv_pages"), "got: {err}");
    }
}
