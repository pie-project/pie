//! Pie standalone server config — TOML schema mirror of `pie.config`.
//!
//! Same TOML the legacy Python server consumed. Both embedded
//! ([`DriverKind::Portable`] / [`DriverKind::CudaNative`] / [`DriverKind::Dummy`])
//! and subprocess-hosted ([`DriverKind::Dev`] / [`DriverKind::Vllm`] /
//! [`DriverKind::Sglang`]) drivers are valid; the dispatch happens in
//! [`crate::serve::start_engine`] via [`crate::serve::topology::resolve_flavor`].
//! Python-only fields are absent from validation here (e.g. nothing
//! checks `huggingface_hub`-resolved paths until the driver actually
//! loads the model).
//!
//! The Rust [`Config`] type below is the user-facing TOML schema; the
//! conversion to `pie::bootstrap::Config` (the runtime's own config)
//! happens in [`crate::bootstrap_translate`].

use std::path::PathBuf;

use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// Top-level
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
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
        let text =
            std::fs::read_to_string(path).map_err(|e| anyhow::anyhow!("read {path:?}: {e}"))?;
        let cfg: Config =
            toml::from_str(&text).map_err(|e| anyhow::anyhow!("parse {path:?}: {e}"))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.models.is_empty(),
            "at least one [[model]] section is required"
        );

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

#[derive(Debug, Clone, Deserialize, Serialize)]
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

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_registry() -> String {
    "https://registry.pie-project.org/".to_string()
}
fn default_true() -> bool {
    true
}

// -----------------------------------------------------------------------------
// [auth]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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

fn default_otlp_endpoint() -> String {
    "http://localhost:4317".to_string()
}
fn default_service_name() -> String {
    "pie".to_string()
}

// -----------------------------------------------------------------------------
// [runtime]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
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
        ensure!(
            self.worker_threads > 0,
            "runtime.worker_threads must be > 0"
        );
        ensure!(
            self.wasm_max_instances > 0,
            "runtime.wasm_max_instances must be > 0"
        );
        ensure!(
            self.wasm_max_memory_mb > 0,
            "runtime.wasm_max_memory_mb must be > 0"
        );
        ensure!(self.max_upload_mb > 0, "runtime.max_upload_mb must be > 0");
        Ok(())
    }
}

fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
fn default_wasm_max_instances() -> u32 {
    1000
}
fn default_wasm_max_memory_mb() -> usize {
    4096
}
fn default_wasm_warm_slots() -> u32 {
    100
}
fn default_fs_scratch_dir() -> PathBuf {
    std::env::temp_dir().join("pie")
}
fn default_network_allowed_hosts() -> Vec<String> {
    vec!["*".to_string()]
}
fn default_max_upload_mb() -> usize {
    256
}

// -----------------------------------------------------------------------------
// [[model]]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
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
        ensure!(
            !self.name.is_empty(),
            "model.name must be a non-empty string"
        );
        self.driver.validate()?;
        self.scheduler.validate()?;
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// [model.scheduler]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
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
    /// Per-context depth of pass-level speculative execution.
    /// `0` disables speculation entirely (every submit goes
    /// through the cold path). `1` is the piggyback path —
    /// one staged pass pre-fired per real pass. Higher values
    /// let chain firing overlap with the inferlet's WASM time
    /// (see SPECULATIVE_EXECUTION_DESIGN.md phase B4b.3). The
    /// eventual ceiling is page-boundary-limited. Valid range:
    /// 0..=64. Default 1.
    #[serde(default = "default_speculation_depth")]
    pub speculation_depth: u32,
}

fn default_speculation_depth() -> u32 {
    1
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
            speculation_depth: default_speculation_depth(),
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
        ensure!(
            self.request_timeout_secs > 0,
            "scheduler.request_timeout_secs must be > 0"
        );
        if let Some(n) = self.default_token_limit {
            ensure!(n > 0, "scheduler.default_token_limit must be > 0 if set");
        }
        ensure!(
            self.default_endowment_pages > 0,
            "scheduler.default_endowment_pages must be > 0"
        );
        ensure!(
            self.admission_oversubscription_factor > 0.0
                && self.admission_oversubscription_factor.is_finite(),
            "scheduler.admission_oversubscription_factor must be finite > 0"
        );
        ensure!(
            self.restore_pause_at_utilization > 0.0 && self.restore_pause_at_utilization <= 1.0,
            "scheduler.restore_pause_at_utilization must be in (0.0, 1.0]"
        );
        ensure!(
            self.speculation_depth <= 64,
            "scheduler.speculation_depth must be in 0..=64 (got {}); 0 disables speculation",
            self.speculation_depth
        );
        Ok(())
    }
}

fn default_batch_policy() -> String {
    "adaptive".to_string()
}
fn default_request_timeout_secs() -> u64 {
    120
}
fn default_endowment_pages() -> usize {
    64
}
fn default_oversubscription_factor() -> f64 {
    4.0
}
fn default_restore_pause_at_utilization() -> f64 {
    0.85
}

// -----------------------------------------------------------------------------
// [model.driver]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // forwarded to the embedded driver via TOML; not all
// fields are read on the Rust side yet.
pub struct DriverConfig {
    /// Driver discriminator. Embedded drivers (`portable`,
    /// `cuda_native`, `dummy`) run in-process; Python drivers (`dev`,
    /// `vllm`, `sglang`) are supervised as subprocesses.
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
    /// IPC wait profile. When omitted, CUDA defaults to `latency`;
    /// other drivers default to the hybrid `balanced` path.
    /// `power` parks immediately whenever no work is ready.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ipc_profile: Option<IpcProfile>,
    /// Expert override for the profile's busy-spin window, in µs.
    ///
    /// Leave unset for the profile default. `0` parks immediately;
    /// larger values trade CPU for lower wake latency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spin_budget_us: Option<u64>,
    /// Driver-specific knobs. Embedded drivers parse this into typed
    /// option structs; Python drivers receive the raw table after
    /// standalone-only `venv` / `python` keys are stripped.
    #[serde(default)]
    pub options: toml::Table,
}

impl DriverConfig {
    pub fn effective_ipc_profile(&self) -> IpcProfile {
        self.ipc_profile.unwrap_or(match self.kind {
            DriverKind::CudaNative => IpcProfile::Latency,
            _ => IpcProfile::Balanced,
        })
    }

    pub fn effective_spin_budget_us(&self) -> u64 {
        self.spin_budget_us
            .unwrap_or_else(|| self.effective_ipc_profile().default_spin_budget_us())
    }

    pub fn use_inproc_polling_channel(&self) -> bool {
        self.effective_ipc_profile() == IpcProfile::Latency
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.device.is_empty(),
            "model.driver.device must be non-empty"
        );
        // All `DriverKind`s are valid here: embedded flavors run in-process
        // through static libs, while dev/vllm/sglang are supervised through
        // `crate::subprocess_driver`.
        match self.kind {
            DriverKind::Portable => {
                let _: PortableDriverOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [model.driver.options] for driver type {:?}: {e}",
                            self.kind,
                        )
                    })?;
            }
            DriverKind::CudaNative => {
                let opts: CudaNativeDriverOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [model.driver.options] for driver type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                opts.validate()?;
            }
            DriverKind::Dummy => {
                let _: DummyDriverOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [model.driver.options] for driver type {:?}: {e}",
                            self.kind,
                        )
                    })?;
            }
            DriverKind::Dev | DriverKind::Vllm | DriverKind::Sglang => {
                validate_subprocess_driver_options(&self.options, self.kind)?;
            }
        }
        Ok(())
    }
}

fn validate_subprocess_driver_options(options: &toml::Table, kind: DriverKind) -> Result<()> {
    for key in [
        "max_num_kv_pages",
        "max_batch_tokens",
        "max_batch_size",
        "linear_attn_max_slots",
        "total_pages",
        "cpu_pages",
        "max_forward_tokens",
        "max_forward_requests",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_running_requests",
        "max_total_tokens",
    ] {
        ensure!(
            !options.contains_key(key),
            "invalid [model.driver.options] for driver type {:?}: \
             `{key}` is not accepted; drivers compute KV pages and scheduling \
             capacity in capabilities",
            kind,
        );
    }
    for key in ["venv", "python"] {
        if let Some(value) = options.get(key) {
            ensure!(
                value.as_str().is_some(),
                "invalid [model.driver.options] for driver type {:?}: `{key}` must be a string",
                kind,
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IpcProfile {
    /// Lowest wake latency. Uses the polling in-process channel for
    /// embedded drivers and unbounded busy-spin for shmem drivers
    /// unless `spin_budget_us` overrides it.
    Latency,
    /// Hybrid spin-then-park path. Good default for GPU-bound work.
    #[default]
    Balanced,
    /// Park immediately after an empty poll. Minimizes idle CPU.
    Power,
}

impl IpcProfile {
    pub fn default_spin_budget_us(self) -> u64 {
        match self {
            Self::Latency => u64::MAX,
            Self::Balanced => default_spin_budget_us(),
            Self::Power => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverKind {
    /// Portable ggml driver — embedded as a static lib in `pie-server`.
    Portable,
    /// Native CUDA driver — embedded as a static lib in `pie-server`
    /// (requires `--features driver-cuda`).
    CudaNative,
    /// Rust dummy driver — random tokens, no model load. Always
    /// embedded in `pie-server`.
    Dummy,
    /// Torch-backed reference Python driver (`pie_driver_dev`). Hosted
    /// out-of-process by [`crate::subprocess_driver::SubprocessDriver`].
    Dev,
    /// vLLM-backed Python driver. Subprocess-hosted.
    Vllm,
    /// SGLang-backed Python driver. Subprocess-hosted.
    Sglang,
}

impl DriverKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DriverKind::Portable => "portable",
            DriverKind::CudaNative => "cuda_native",
            DriverKind::Dummy => "dummy",
            DriverKind::Dev => "dev",
            DriverKind::Vllm => "vllm",
            DriverKind::Sglang => "sglang",
        }
    }
}

fn default_tp_size() -> u32 {
    1
}
fn default_activation_dtype() -> String {
    "bfloat16".to_string()
}
fn default_random_seed() -> u64 {
    42
}
/// Default busy-spin budget (µs) before the driver-side channel falls
/// back to parking. Matches `pie::driver::InProcChannel::new()`.
fn default_spin_budget_us() -> u64 {
    1_000
}

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
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct PortableDriverOptions {
    #[serde(skip)]
    pub device: String,
    #[serde(skip)]
    pub verbose: bool,
    pub ready_timeout_s: f64,
    pub shutdown_timeout_s: f64,
    /// Ignored in standalone (binary is statically linked); accepted
    /// for config compatibility with the Python wrapper path.
    pub binary_path: String,
}

impl Default for PortableDriverOptions {
    fn default() -> Self {
        Self {
            device: "auto".to_string(),
            verbose: false,
            ready_timeout_s: 120.0,
            shutdown_timeout_s: 5.0,
            binary_path: String::new(),
        }
    }
}

/// `[model.driver.options]` for `type = "dummy"`. The dummy driver
/// fabricates everything the portable driver would otherwise read from
/// model weights — `vocab_size` and `arch_name` are required because no
/// safe default exists. Page geometry and timeouts have generic defaults;
/// the driver derives its synthetic KV page pool from these limits.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DummyDriverOptions {
    /// Vocabulary size advertised in the caps handshake. Should match
    /// the tokenizer at `hf_repo/tokenizer.json`. When `None` the
    /// standalone reads `vocab_size` from `<snapshot_dir>/config.json`
    /// before launching the driver.
    #[serde(default)]
    pub vocab_size: Option<u32>,
    /// Architecture name advertised in the caps handshake (e.g.
    /// `"qwen3"`, `"llama3"`). When `None` the standalone derives it
    /// from `<snapshot_dir>/config.json` (lowercased first
    /// `architectures[0]` with a `forcausallm` suffix stripped).
    #[serde(default)]
    pub arch_name: Option<String>,

    #[serde(default = "default_dummy_ready_timeout_s")]
    pub ready_timeout_s: f64,
}

fn default_dummy_ready_timeout_s() -> f64 {
    5.0
}

/// `[model.driver.options]` for `type = "cuda_native"`.
/// Mirrors `pie/src/pie_driver_cuda_native/config.py::CudaNativeDriverConfig`.
///
/// `binary_path` is accepted for config compatibility with the Python
/// wrapper path but ignored — the standalone embeds the cuda driver as
/// a static library, so there is no separate executable to discover.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaNativeDriverOptions {
    pub binary_path: String,

    pub gpu_mem_utilization: f64,
    pub memory_profile: CudaMemoryProfile,
    pub swap_pool_size: u32,
    pub weight_dtype: String,
    /// CUDA device string, e.g. `"cuda:0"`. Populated by the caller
    /// from `model.driver.device`; set on the C++ side via
    /// `cudaSetDevice` (see `driver/cuda/src/engine.cpp`).
    #[serde(skip)]
    pub device: String,
    #[serde(skip)]
    pub verbose: bool,
    /// Runtime quantization mode applied after weight load. Empty = none;
    /// `"fp8"` enables per-tensor symmetric FP8_E4M3 on every llama-like
    /// projection weight. Currently only honored for model_type=qwen3 by
    /// the C++ side.
    pub runtime_quant: String,

    pub ready_timeout_s: f64,
    pub shutdown_timeout_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CudaMemoryProfile {
    #[default]
    Auto,
    Latency,
    Balanced,
    Throughput,
    Capacity,
}

impl Default for CudaNativeDriverOptions {
    fn default() -> Self {
        Self {
            binary_path: String::new(),
            gpu_mem_utilization: 0.90,
            memory_profile: CudaMemoryProfile::Auto,
            swap_pool_size: 0,
            weight_dtype: "bfloat16".to_string(),
            device: String::new(),
            verbose: false,
            runtime_quant: String::new(),
            ready_timeout_s: 600.0,
            shutdown_timeout_s: 5.0,
        }
    }
}

impl CudaNativeDriverOptions {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "model.driver.options.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        Ok(())
    }
}

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
        assert_eq!(
            cfg.models[0].driver.effective_ipc_profile(),
            IpcProfile::Balanced
        );
        assert_eq!(cfg.models[0].driver.effective_spin_budget_us(), 1_000);
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn cuda_tp_defaults_to_latency_ipc() {
        let text = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]
tensor_parallel_size = 2

[model.driver.options]
gpu_mem_utilization = 0.90
memory_profile = "balanced"
"#;
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            cfg.models[0].driver.effective_ipc_profile(),
            IpcProfile::Latency
        );
        assert!(cfg.models[0].driver.use_inproc_polling_channel());
    }

    #[test]
    fn cuda_single_rank_defaults_to_latency_ipc() {
        let text = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
gpu_mem_utilization = 0.90
memory_profile = "balanced"
"#;
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            cfg.models[0].driver.effective_ipc_profile(),
            IpcProfile::Latency
        );
        assert!(cfg.models[0].driver.use_inproc_polling_channel());
    }

    #[test]
    fn cuda_latency_profile_defaults_to_latency_ipc() {
        let text = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
gpu_mem_utilization = 0.90
memory_profile = "latency"
"#;
        let cfg: Config = toml::from_str(text).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            cfg.models[0].driver.effective_ipc_profile(),
            IpcProfile::Latency
        );
        assert!(cfg.models[0].driver.use_inproc_polling_channel());
    }

    #[test]
    fn rejects_legacy_portable_kv_page_knob() {
        let stale = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "portable"
device = ["cpu"]

[model.driver.options]
max_num_kv_pages = 1024
"#;
        let cfg: Config = toml::from_str(stale).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("max_num_kv_pages"), "got: {err}");
    }

    #[test]
    fn rejects_legacy_dummy_kv_page_knob() {
        let stale = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
max_num_kv_pages = 1024
"#;
        let cfg: Config = toml::from_str(stale).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("max_num_kv_pages"), "got: {err}");
    }

    #[test]
    fn rejects_legacy_subprocess_kv_page_knob() {
        let stale = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "vllm"
device = ["cuda:0"]

[model.driver.options]
max_num_kv_pages = 1024
"#;
        let cfg: Config = toml::from_str(stale).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("max_num_kv_pages"), "got: {err}");
        assert!(err.contains("capabilities"), "got: {err}");
    }

    #[test]
    fn rejects_public_driver_capacity_knobs() {
        for (ty, key) in [
            ("portable", "total_pages"),
            ("portable", "cpu_pages"),
            ("portable", "max_forward_tokens"),
            ("portable", "max_forward_requests"),
            ("dummy", "max_forward_tokens"),
            ("dummy", "max_forward_requests"),
            ("dummy", "max_model_len"),
            ("dev", "max_forward_tokens"),
            ("dev", "max_forward_requests"),
            ("vllm", "max_num_seqs"),
            ("vllm", "max_num_batched_tokens"),
            ("sglang", "max_running_requests"),
            ("sglang", "max_total_tokens"),
        ] {
            let mut options = String::new();
            if ty == "dummy" {
                options.push_str("vocab_size = 151936\narch_name = \"qwen3\"\n");
            }
            options.push_str(&format!("{key} = 1\n"));
            let text = format!(
                "[[model]]\nname = \"m\"\nhf_repo = \"x\"\n[model.driver]\n\
                 type = \"{ty}\"\ndevice = [\"cpu\"]\n[model.driver.options]\n{options}"
            );
            let cfg: Config = toml::from_str(&text).unwrap();
            let err = match cfg.validate() {
                Ok(()) => panic!("type={ty} key={key} unexpectedly accepted"),
                Err(err) => err.to_string(),
            };
            assert!(err.contains(key), "type={ty} key={key} got: {err}");
        }
    }

    #[test]
    fn parses_ipc_profiles_and_spin_override() {
        let latency = r#"
[[model]]
name = "m"
hf_repo = "x"
[model.driver]
type = "portable"
device = "cpu"
ipc_profile = "latency"
"#;
        let cfg: Config = toml::from_str(latency).unwrap();
        assert_eq!(cfg.models[0].driver.ipc_profile, Some(IpcProfile::Latency));
        assert!(cfg.models[0].driver.use_inproc_polling_channel());
        assert_eq!(cfg.models[0].driver.effective_spin_budget_us(), u64::MAX);

        let power = r#"
[[model]]
name = "m"
hf_repo = "x"
[model.driver]
type = "portable"
device = "cpu"
ipc_profile = "power"
"#;
        let cfg: Config = toml::from_str(power).unwrap();
        assert_eq!(cfg.models[0].driver.ipc_profile, Some(IpcProfile::Power));
        assert_eq!(cfg.models[0].driver.effective_spin_budget_us(), 0);

        let override_spin = r#"
[[model]]
name = "m"
hf_repo = "x"
[model.driver]
type = "portable"
device = "cpu"
ipc_profile = "latency"
spin_budget_us = 25
"#;
        let cfg: Config = toml::from_str(override_spin).unwrap();
        assert_eq!(cfg.models[0].driver.effective_spin_budget_us(), 25);
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
    fn accepts_subprocess_drivers() {
        // dev/vllm/sglang are hosted out-of-process by
        // `crate::subprocess_driver::SubprocessDriver`.
        for ty in ["dev", "vllm", "sglang"] {
            let toml_text = format!(
                "[[model]]\nname = \"m\"\nhf_repo = \"x\"\n[model.driver]\n\
                 type = \"{ty}\"\ndevice = [\"cuda:0\"]\n"
            );
            let cfg: Config = toml::from_str(&toml_text).unwrap();
            cfg.validate().unwrap_or_else(|e| panic!("type={ty}: {e}"));
        }
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

    #[test]
    fn parses_cuda_native_config() {
        let cuda = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
gpu_mem_utilization = 0.90
memory_profile = "balanced"
runtime_quant = "fp8"
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.models[0].driver.kind, DriverKind::CudaNative);
        let opts: CudaNativeDriverOptions =
            cfg.models[0].driver.options.clone().try_into().unwrap();
        assert_eq!(opts.gpu_mem_utilization, 0.90);
        assert_eq!(opts.memory_profile, CudaMemoryProfile::Balanced);
        assert_eq!(opts.runtime_quant, "fp8");
        assert_eq!(opts.weight_dtype, "bfloat16"); // default
    }

    #[test]
    fn cuda_native_options_default_when_omitted() {
        let cuda = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().unwrap();
        let opts: CudaNativeDriverOptions =
            cfg.models[0].driver.options.clone().try_into().unwrap();
        assert_eq!(opts.swap_pool_size, 0);
        assert_eq!(opts.gpu_mem_utilization, 0.90);
        assert_eq!(opts.memory_profile, CudaMemoryProfile::Auto);
        assert_eq!(opts.ready_timeout_s, 600.0);
    }

    #[test]
    fn rejects_invalid_cuda_memory_profile() {
        let cuda = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
memory_profile = "aggressive"
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("memory_profile"), "got: {err}");
        assert!(err.contains("aggressive"), "got: {err}");
    }

    #[test]
    fn rejects_unknown_cuda_option() {
        let cuda = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
manual_capacity = 1
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("manual_capacity"), "got: {err}");
    }

    #[test]
    fn rejects_options_for_wrong_embedded_driver_type() {
        let stale = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
gpu_mem_utilization = 0.50
vocab_size = 151936
arch_name = "qwen3"
"#;
        let cfg: Config = toml::from_str(stale).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("gpu_mem_utilization"), "got: {err}");
        assert!(err.contains("Dummy"), "got: {err}");
    }

    #[test]
    fn rejects_non_string_subprocess_python_options() {
        let bad = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "vllm"
device = ["cuda:0"]

[model.driver.options]
venv = 123
"#;
        let cfg: Config = toml::from_str(bad).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("venv"), "got: {err}");
        assert!(err.contains("string"), "got: {err}");
    }
}
