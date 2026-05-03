//! Pie standalone server config — TOML schema mirror of `pie.config`.
//!
//! Same TOML the Python `server/torch` path consumes, with two differences:
//!   * [`DriverKind::reject_torch_only`] errors out on torch-hosted drivers
//!     (`native` / `vllm` / `sglang`) — those need a Python interpreter.
//!     Use `server/torch` if you need them.
//!   * Python-only fields are absent from validation here (e.g. nothing
//!     checks `huggingface_hub`-resolved paths until the driver actually
//!     loads the model).
//!
//! The Rust [`Config`] type below is the user-facing TOML schema. The
//! conversion to `pie::bootstrap::Config` (the runtime's own config) is
//! a later step (M2.4 wires it up).

use std::path::PathBuf;

use anyhow::{Result, bail, ensure};
use serde::Deserialize;

// -----------------------------------------------------------------------------
// Top-level
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    /// `[[model]]` array — at least one entry required. The first entry
    /// is the implicit default for inferlets that don't pin a model.
    #[serde(default, rename = "model")]
    pub models: Vec<ModelConfig>,
}

impl Config {
    pub fn from_toml_file(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("read {path:?}: {e}"))?;
        let cfg: Config = toml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("parse {path:?}: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(!self.models.is_empty(), "at least one [[model]] section is required");

        let mut seen = std::collections::HashSet::new();
        for m in &self.models {
            ensure!(
                seen.insert(m.name.clone()),
                "duplicate [[model]] name {:?}",
                m.name
            );
            m.validate()?;
        }

        // Disjoint device check — same constraint as pie/config.py.
        let mut owner: std::collections::HashMap<String, String> = Default::default();
        for m in &self.models {
            for d in &m.driver.device {
                if let Some(prev) = owner.insert(d.clone(), m.name.clone()) {
                    bail!(
                        "device {d:?} claimed by both model {prev:?} and {:?}",
                        m.name
                    );
                }
            }
        }

        self.server.validate()?;
        self.runtime.validate()?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// [server]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default)]
    pub verbose: bool,
    #[serde(default = "default_registry")]
    pub registry: String,
    #[serde(default)]
    pub max_concurrent_processes: Option<usize>,
    #[serde(default = "default_true")]
    pub python_snapshot: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            verbose: false,
            registry: default_registry(),
            max_concurrent_processes: None,
            python_snapshot: true,
        }
    }
}

impl ServerConfig {
    fn validate(&self) -> Result<()> {
        if let Some(n) = self.max_concurrent_processes {
            ensure!(n > 0, "server.max_concurrent_processes must be > 0 if set");
        }
        Ok(())
    }
}

fn default_host() -> String { "127.0.0.1".to_string() }
fn default_port() -> u16 { 8080 }
fn default_registry() -> String { "https://registry.pie-project.org/".to_string() }
fn default_true() -> bool { true }

// -----------------------------------------------------------------------------
// [auth]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self { enabled: true }
    }
}

// -----------------------------------------------------------------------------
// [telemetry]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_otlp_endpoint")]
    pub endpoint: String,
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: default_otlp_endpoint(),
            service_name: default_service_name(),
        }
    }
}

fn default_otlp_endpoint() -> String { "http://localhost:4317".to_string() }
fn default_service_name() -> String { "pie".to_string() }

// -----------------------------------------------------------------------------
// [runtime]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    #[serde(default = "default_wasm_max_instances")]
    pub wasm_max_instances: u32,
    #[serde(default = "default_wasm_max_memory_mb")]
    pub wasm_max_memory_mb: usize,
    #[serde(default)]
    pub wasm_warm_memory_mb: usize,
    #[serde(default = "default_wasm_warm_slots")]
    pub wasm_warm_slots: u32,
    #[serde(default)]
    pub allow_fs: bool,
    #[serde(default = "default_fs_scratch_dir")]
    pub fs_scratch_dir: PathBuf,
    #[serde(default = "default_true")]
    pub allow_network: bool,
    #[serde(default = "default_network_allowed_hosts")]
    pub network_allowed_hosts: Vec<String>,
    #[serde(default = "default_max_upload_mb")]
    pub max_upload_mb: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: default_worker_threads(),
            wasm_max_instances: default_wasm_max_instances(),
            wasm_max_memory_mb: default_wasm_max_memory_mb(),
            wasm_warm_memory_mb: 0,
            wasm_warm_slots: default_wasm_warm_slots(),
            allow_fs: false,
            fs_scratch_dir: default_fs_scratch_dir(),
            allow_network: true,
            network_allowed_hosts: default_network_allowed_hosts(),
            max_upload_mb: default_max_upload_mb(),
        }
    }
}

impl RuntimeConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.worker_threads > 0, "runtime.worker_threads must be > 0");
        ensure!(self.wasm_max_instances > 0, "runtime.wasm_max_instances must be > 0");
        ensure!(self.wasm_max_memory_mb > 0, "runtime.wasm_max_memory_mb must be > 0");
        ensure!(self.max_upload_mb > 0, "runtime.max_upload_mb must be > 0");
        Ok(())
    }
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
fn default_wasm_max_instances() -> u32 { 1000 }
fn default_wasm_max_memory_mb() -> usize { 4096 }
fn default_wasm_warm_slots() -> u32 { 100 }
fn default_fs_scratch_dir() -> PathBuf { std::env::temp_dir().join("pie") }
fn default_network_allowed_hosts() -> Vec<String> { vec!["*".to_string()] }
fn default_max_upload_mb() -> usize { 256 }

// -----------------------------------------------------------------------------
// [[model]]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub hf_repo: String,
    pub driver: DriverConfig,
    #[serde(default)]
    pub scheduler: SchedulerConfig,
}

impl ModelConfig {
    fn validate(&self) -> Result<()> {
        ensure!(!self.name.is_empty(), "model.name must be a non-empty string");
        self.driver.validate()?;
        self.scheduler.validate()?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// [model.scheduler]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    #[serde(default = "default_batch_policy")]
    pub batch_policy: String,
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    #[serde(default)]
    pub default_token_limit: Option<usize>,
    #[serde(default = "default_endowment_pages")]
    pub default_endowment_pages: usize,
    #[serde(default = "default_oversubscription_factor")]
    pub admission_oversubscription_factor: f64,
    #[serde(default = "default_restore_pause_at_utilization")]
    pub restore_pause_at_utilization: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            batch_policy: default_batch_policy(),
            request_timeout_secs: default_request_timeout_secs(),
            default_token_limit: None,
            default_endowment_pages: default_endowment_pages(),
            admission_oversubscription_factor: default_oversubscription_factor(),
            restore_pause_at_utilization: default_restore_pause_at_utilization(),
        }
    }
}

impl SchedulerConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.batch_policy.as_str(), "adaptive" | "eager" | "greedy"),
            "scheduler.batch_policy must be one of 'adaptive' | 'eager' | 'greedy' (got {:?})",
            self.batch_policy
        );
        ensure!(self.request_timeout_secs > 0, "scheduler.request_timeout_secs must be > 0");
        if let Some(n) = self.default_token_limit {
            ensure!(n > 0, "scheduler.default_token_limit must be > 0 if set");
        }
        ensure!(self.default_endowment_pages > 0, "scheduler.default_endowment_pages must be > 0");
        ensure!(
            self.admission_oversubscription_factor > 0.0
                && self.admission_oversubscription_factor.is_finite(),
            "scheduler.admission_oversubscription_factor must be finite > 0"
        );
        ensure!(
            self.restore_pause_at_utilization > 0.0
                && self.restore_pause_at_utilization <= 1.0,
            "scheduler.restore_pause_at_utilization must be in (0.0, 1.0]"
        );
        Ok(())
    }
}

fn default_batch_policy() -> String { "adaptive".to_string() }
fn default_request_timeout_secs() -> u64 { 120 }
fn default_endowment_pages() -> usize { 64 }
fn default_oversubscription_factor() -> f64 { 4.0 }
fn default_restore_pause_at_utilization() -> f64 { 0.85 }

// -----------------------------------------------------------------------------
// [model.driver]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // forwarded to the embedded driver via TOML; not all
// fields are read on the Rust side yet.
pub struct DriverConfig {
    /// Driver discriminator. `server/standalone` only accepts the
    /// out-of-process native drivers (`portable`, `cuda_native`); the
    /// torch-hosted drivers (`native`, `vllm`, `sglang`) require Python.
    #[serde(rename = "type")]
    pub kind: DriverKind,
    /// Single string or list of strings — both accepted on input.
    #[serde(deserialize_with = "deserialize_string_or_list")]
    pub device: Vec<String>,
    #[serde(default = "default_tp_size")]
    pub tensor_parallel_size: u32,
    #[serde(default = "default_activation_dtype")]
    pub activation_dtype: String,
    #[serde(default = "default_random_seed")]
    pub random_seed: u64,
    /// Driver-specific knobs. The standalone server interprets this as
    /// [`PortableDriverOptions`] / [`CudaNativeDriverOptions`] depending
    /// on `kind`. Stored as a generic table here so unknown drivers
    /// don't break parsing.
    #[serde(default)]
    pub options: toml::Table,
}

impl DriverConfig {
    fn validate(&self) -> Result<()> {
        ensure!(!self.device.is_empty(), "model.driver.device must be non-empty");
        self.kind.reject_torch_only()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverKind {
    /// Portable ggml driver — embedded as a static lib in `pie-standalone`.
    Portable,
    /// Native CUDA driver — embedded as a static lib in `pie-standalone`
    /// (requires `--features driver-cuda`).
    CudaNative,
    /// Rust dummy driver — random tokens, no model load. Embedded as a
    /// Rust staticlib in `pie-standalone` (requires `--features driver-dummy`).
    Dummy,
    /// Torch-hosted CUDA driver. Requires Python; not supported here.
    Native,
    /// Torch-hosted vLLM driver. Requires Python; not supported here.
    Vllm,
    /// Torch-hosted SGLang driver. Requires Python; not supported here.
    Sglang,
}

impl DriverKind {
    fn reject_torch_only(&self) -> Result<()> {
        match self {
            DriverKind::Portable | DriverKind::CudaNative | DriverKind::Dummy => Ok(()),
            DriverKind::Native | DriverKind::Vllm | DriverKind::Sglang => bail!(
                "model.driver.type = {self:?} is hosted by Python (`server/torch`) \
                 and is not available in `server/standalone`. Use `pie-standalone` \
                 with `type = \"portable\"`, `type = \"cuda_native\"`, or \
                 `type = \"dummy\"`, or run `server/torch` (Python) for the \
                 torch-based drivers."
            ),
        }
    }
}

fn default_tp_size() -> u32 { 1 }
fn default_activation_dtype() -> String { "bfloat16".to_string() }
fn default_random_seed() -> u64 { 42 }

/// Accept either a single string or a list of strings, matching
/// `pie/config.py::_parse_driver`'s `device` handling.
fn deserialize_string_or_list<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::{self, Visitor};
    use std::fmt;

    struct V;
    impl<'de> Visitor<'de> for V {
        type Value = Vec<String>;
        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a string or list of strings")
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(vec![v.to_string()])
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(vec![v])
        }
        fn visit_seq<A: de::SeqAccess<'de>>(self, mut s: A) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some(v) = s.next_element::<String>()? {
                out.push(v);
            }
            Ok(out)
        }
    }
    d.deserialize_any(V)
}

// -----------------------------------------------------------------------------
// Driver-specific options (typed views over `DriverConfig::options`)
// -----------------------------------------------------------------------------

/// `[model.driver.options]` for `type = "portable"`.
/// Mirrors `pie/src/pie_driver_portable/config.py::PortableDriverConfig`.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PortableDriverOptions {
    pub kv_page_size: u32,
    pub max_num_kv_pages: u32,
    pub max_batch_tokens: u32,
    pub max_batch_size: u32,
    pub cpu_pages: u32,
    pub n_ctx: u32,
    pub n_gpu_layers: i32,
    pub ready_timeout_s: f64,
    pub shutdown_timeout_s: f64,
    /// Ignored in standalone (binary is statically linked); accepted
    /// for config compatibility with the Python wrapper path.
    pub binary_path: String,
}

impl Default for PortableDriverOptions {
    fn default() -> Self {
        Self {
            kv_page_size: 32,
            max_num_kv_pages: 1024,
            max_batch_tokens: 10240,
            max_batch_size: 512,
            cpu_pages: 0,
            n_ctx: 4096,
            n_gpu_layers: 0,
            ready_timeout_s: 120.0,
            shutdown_timeout_s: 5.0,
            binary_path: String::new(),
        }
    }
}

/// `[model.driver.options]` for `type = "dummy"`. The dummy driver
/// fabricates everything the portable driver would otherwise read from
/// model weights — `vocab_size` and `arch_name` are required because no
/// safe default exists. Page geometry and timeouts have generic defaults.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DummyDriverOptions {
    /// Vocabulary size advertised in the caps handshake. Required —
    /// must match the tokenizer at `hf_repo/tokenizer.json`.
    pub vocab_size: u32,
    /// Architecture name advertised in the caps handshake (e.g.
    /// `"qwen3"`, `"llama3"`). Required — the runtime uses it to
    /// look up the matching chat template.
    pub arch_name: String,

    #[serde(default = "default_kv_page_size")]
    pub kv_page_size: u32,
    #[serde(default = "default_max_num_kv_pages")]
    pub max_num_kv_pages: u32,
    #[serde(default = "default_max_batch_tokens")]
    pub max_batch_tokens: u32,
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    #[serde(default = "default_max_model_len")]
    pub max_model_len: u32,
    #[serde(default = "default_dummy_ready_timeout_s")]
    pub ready_timeout_s: f64,
}

fn default_kv_page_size() -> u32 { 16 }
fn default_max_num_kv_pages() -> u32 { 256 }
fn default_max_batch_tokens() -> u32 { 4096 }
fn default_max_batch_size() -> u32 { 128 }
fn default_max_model_len() -> u32 { 4096 }
fn default_dummy_ready_timeout_s() -> f64 { 5.0 }

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_PORTABLE: &str = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "portable"
device = ["cpu"]
"#;

    #[test]
    fn parses_minimal_portable_config() {
        let cfg: Config = toml::from_str(MINIMAL_PORTABLE).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.models.len(), 1);
        assert_eq!(cfg.models[0].driver.kind, DriverKind::Portable);
        assert_eq!(cfg.models[0].driver.device, vec!["cpu".to_string()]);
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn device_string_or_list() {
        let one = r#"
[[model]]
name = "m"
hf_repo = "x"
[model.driver]
type = "portable"
device = "cuda:0"
"#;
        let cfg: Config = toml::from_str(one).unwrap();
        assert_eq!(cfg.models[0].driver.device, vec!["cuda:0".to_string()]);
    }

    #[test]
    fn rejects_torch_drivers() {
        let torch = r#"
[[model]]
name = "m"
hf_repo = "x"
[model.driver]
type = "native"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(torch).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("server/torch"), "got: {err}");
    }

    #[test]
    fn rejects_duplicate_devices() {
        let dup = r#"
[[model]]
name = "a"
hf_repo = "x"
[model.driver]
type = "portable"
device = ["cuda:0"]

[[model]]
name = "b"
hf_repo = "y"
[model.driver]
type = "portable"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(dup).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("claimed by both"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_top_level_keys() {
        let bad = r#"
nonsense = true

[[model]]
name = "m"
hf_repo = "x"
[model.driver]
type = "portable"
device = ["cpu"]
"#;
        assert!(toml::from_str::<Config>(bad).is_err());
    }
}
