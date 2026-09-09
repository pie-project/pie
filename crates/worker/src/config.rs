use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use controller_api::Role;
pub use engine::runahead::Runahead;
use serde::{Deserialize, Serialize};

pub mod backend;
pub mod layout;
pub mod schema;
pub mod units;

pub use backend::{
    CudaNativeEngineOptions, MetalEngineOptions, VulkanEngineOptions, WgpuEngineOptions,
};
pub use units::{ByteSize, Duration};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub sandbox: SandboxConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub executor: ExecutorConfig,
    #[serde(default)]
    pub offload: OffloadConfig,
    pub model: ModelConfig,
}

impl Config {
    pub fn parse(s: &str) -> Result<Self> {
        let file: toml::Table = toml::from_str(s).map_err(|e| {
            if s.contains("[[model]]") {
                anyhow::anyhow!(
                    "parse config: {e}\n\
                     hint: pie serves exactly one model — use a single `[model]` table, \
                     not a `[[model]]` list."
                )
            } else {
                anyhow::anyhow!("parse config: {e}")
            }
        })?;
        let reshaped = crate::config::layout::reshape(file)?;
        let s = &toml::to_string(&reshaped).map_err(|e| anyhow::anyhow!("reshape config: {e}"))?;
        let mut cfg: Config = toml::from_str(s).map_err(|e| {
            if s.contains("[[model]]") {
                anyhow::anyhow!(
                    "parse config: {e}\n\
                     hint: pie serves exactly one model — use a single `[model]` table, \
                     not a `[[model]]` list."
                )
            } else {
                anyhow::anyhow!("parse config: {e}")
            }
        })?;
        cfg.model.resolve_drafter()?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn state_diagnostics(&mut self, words: &str) -> Result<()> {
        match self.model.engine.kind {
            EngineKind::CudaNative | EngineKind::Metal => {
                self.model.engine.options.insert(
                    "diagnostics".to_string(),
                    toml::Value::String(words.to_string()),
                );
                Ok(())
            }
            other => anyhow::bail!(
                "`--diag` names engine diagnostics and the {} engine has no \
                 diagnostics record; the cuda and metal shells do",
                other.as_str()
            ),
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.server.validate()?;
        self.runtime.validate()?;
        self.sandbox.validate()?;
        self.cluster.validate()?;
        self.executor.validate()?;
        self.offload.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorConfig {
    #[serde(default = "default_executor_max_clients")]
    pub max_clients: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_clients: default_executor_max_clients(),
        }
    }
}

impl ExecutorConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.max_clients > 0,
            "cluster.max_clients must be greater than zero"
        );
        Ok(())
    }
}

fn default_executor_max_clients() -> usize {
    4
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OffloadConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub prefill_min_suffix_tokens: usize,
    #[serde(default = "default_offload_max_outstanding")]
    pub max_outstanding_per_partner: u32,
    #[serde(default)]
    pub transfer: OffloadTransfer,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            prefill_min_suffix_tokens: 0,
            max_outstanding_per_partner: default_offload_max_outstanding(),
            transfer: OffloadTransfer::Auto,
        }
    }
}

impl OffloadConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.max_outstanding_per_partner > 0,
            "cluster.max_outstanding_per_partner must be greater than zero"
        );
        Ok(())
    }
}

fn default_offload_max_outstanding() -> u32 {
    4
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OffloadTransfer {
    Inline,
    Nixl,
    #[default]
    Auto,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterConfig {
    #[serde(default)]
    pub controller: Option<String>,
    #[serde(default)]
    pub role: Option<Role>,
    #[serde(default)]
    pub gateways: Vec<String>,
}

impl ClusterConfig {
    fn validate(&self) -> Result<()> {
        if self.controller.is_some() {
            ensure!(
                self.role.is_some(),
                "[cluster] role is required when controller is set"
            );
        }
        if matches!(self.role, Some(Role::Prefill | Role::Encode)) {
            ensure!(
                self.controller.is_some(),
                "[cluster] prefill and encode executors require a controller"
            );
        }
        Ok(())
    }
}

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
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    #[serde(default = "default_max_upload")]
    pub max_upload: ByteSize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            verbose: false,
            registry: default_registry(),
            worker_threads: default_worker_threads(),
            max_upload: default_max_upload(),
        }
    }
}

impl ServerConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.worker_threads > 0, "server.worker_threads must be > 0");
        ensure!(
            self.max_upload.as_bytes() > 0,
            "server.max_upload must be > 0"
        );
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
fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(64))
        .unwrap_or(4)
}
fn default_max_upload() -> ByteSize {
    ByteSize::from_mib(256)
}

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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxConfig {
    #[serde(default)]
    pub allow_fs: bool,
    #[serde(default = "default_fs_scratch_dir")]
    pub fs_scratch_dir: PathBuf,
    #[serde(default = "default_true")]
    pub allow_network: bool,
    #[serde(default = "default_network_allowed_hosts")]
    pub network_allowed_hosts: Vec<String>,
    #[serde(default = "default_max_instances")]
    pub max_instances: u32,
    #[serde(default = "default_max_memory")]
    pub max_memory: ByteSize,
    #[serde(default)]
    pub warm_memory: ByteSize,
    #[serde(default = "default_warm_slots")]
    pub warm_slots: u32,
    #[serde(default = "default_true")]
    pub python_snapshot: bool,
    #[serde(default = "default_true")]
    pub python_runtime: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            allow_fs: false,
            fs_scratch_dir: default_fs_scratch_dir(),
            allow_network: true,
            network_allowed_hosts: default_network_allowed_hosts(),
            max_instances: default_max_instances(),
            max_memory: default_max_memory(),
            warm_memory: ByteSize::from_mib(0),
            warm_slots: default_warm_slots(),
            python_snapshot: true,
            python_runtime: true,
        }
    }
}

impl SandboxConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.max_instances > 0, "sandbox.max_instances must be > 0");
        ensure!(
            self.max_memory.as_bytes() > 0,
            "sandbox.max_memory must be > 0"
        );
        Ok(())
    }
}

fn default_max_instances() -> u32 {
    1000
}
fn default_max_memory() -> ByteSize {
    ByteSize::from_mib(4096)
}
fn default_warm_slots() -> u32 {
    100
}
fn default_fs_scratch_dir() -> PathBuf {
    std::env::temp_dir().join("pie")
}
fn default_network_allowed_hosts() -> Vec<String> {
    vec!["*".to_string()]
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    pub model: String,
    #[serde(default)]
    pub sku: Option<String>,
    #[serde(default)]
    pub drafter: Option<String>,
    pub engine: EngineConfig,
    #[serde(default)]
    pub weight_cache_dir: String,
    #[serde(default)]
    pub adapter_dir: String,
    #[serde(default = "default_weight_dtype")]
    pub weight_dtype: String,
    #[serde(default)]
    pub device_weight_budget: Option<ByteSize>,
    #[serde(default)]
    pub host_weight_budget: Option<ByteSize>,
    #[serde(default = "default_deferred_tier")]
    pub deferred_tier: bool,
    #[serde(default)]
    pub max_patches: Option<u32>,
    #[serde(default)]
    pub max_images: Option<u32>,
    #[serde(default)]
    pub max_voxels: Option<u32>,
    #[serde(default)]
    pub max_clips: Option<u32>,
    #[serde(default)]
    pub adapters: AdapterConfig,
}

fn default_weight_dtype() -> String {
    "bfloat16".to_string()
}

fn default_deferred_tier() -> bool {
    true
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterConfig {
    #[serde(default)]
    pub seats: Option<u32>,
    #[serde(default)]
    pub registered: Vec<RegisteredAdapter>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegisteredAdapter {
    pub id: u32,
    #[serde(default)]
    pub planes: std::collections::BTreeMap<String, String>,
}

impl AdapterConfig {
    #[must_use]
    pub fn seats(&self) -> u32 {
        self.seats.unwrap_or_else(|| {
            self.registered
                .iter()
                .map(|adapter| adapter.id.saturating_add(1))
                .max()
                .unwrap_or(0)
        })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seats() == 0 && self.registered.is_empty()
    }

    fn validate(&self) -> Result<()> {
        let seats = self.seats();
        for adapter in &self.registered {
            ensure!(
                adapter.id < seats,
                "model.adapters: adapter id {} is past the {seats} seat(s) this \
                 deployment asks for; raise `[model.adapters] seats` or renumber it",
                adapter.id
            );
            for (bank, path) in &adapter.planes {
                ensure!(
                    Path::new(path).is_absolute(),
                    "model.adapters: the plane for bank {bank:?} of adapter {} must be \
                     an absolute path (got {path:?})",
                    adapter.id
                );
            }
        }
        Ok(())
    }
}

impl ModelConfig {
    pub fn adapter_mount(&self) -> Option<std::path::PathBuf> {
        (!self.adapter_dir.is_empty()).then(|| std::path::PathBuf::from(&self.adapter_dir))
    }

    #[must_use]
    pub fn residency(&self) -> engine::Residency {
        engine::Residency {
            device_weight_budget: self.device_weight_budget.map(|b| b.as_bytes()),
            host_weight_budget: self.host_weight_budget.map(|b| b.as_bytes()),
            deferred_tier: self.deferred_tier,
        }
    }

    #[must_use]
    pub fn patch_ceilings(&self) -> (Option<u32>, Option<u32>) {
        (self.max_patches, self.max_images)
    }

    #[must_use]
    pub fn voxel_ceilings(&self) -> (Option<u32>, Option<u32>) {
        (self.max_voxels, self.max_clips)
    }

    pub fn resolve_drafter(&mut self) -> Result<()> {
        let Some(drafter) = self.drafter.as_deref() else {
            return Ok(());
        };
        let target = self.model.trim();
        ensure!(
            !target.contains('/') || !target.ends_with(".zt"),
            "model.drafter = {drafter:?} needs model.model to name the target repository \
             (as `pie model list` prints it), not an artifact path {target:?}; name the row \
             with model.sku instead"
        );
        let Some(published) = models::published::lookup(target, drafter) else {
            let known: Vec<&str> = models::published::for_target(target).map(|p| p.drafter).collect();
            bail!(
                "model.drafter = {drafter:?}: no published head of that name for {target:?} in this \
                 build{}",
                if known.is_empty() {
                    "; it knows none for that target — name the row with model.sku".to_string()
                } else {
                    format!("; it knows {known:?}")
                }
            );
        };
        match &self.sku {
            Some(sku) if sku != published.sku => bail!(
                "model.sku = {sku:?} and model.drafter = {drafter:?} name different rows (the \
                 drafter's is {:?}); state one of them",
                published.sku
            ),
            _ => self.sku = Some(published.sku.to_string()),
        }
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.is_empty(),
            "model.name must be a non-empty string"
        );
        ensure!(
            !self.model.trim().is_empty(),
            "model.model must name a stored artifact or a path to one \
             (`pie model list` shows what is available)"
        );
        self.engine.validate()?;
        ensure!(
            self.weight_cache_dir.is_empty() || Path::new(&self.weight_cache_dir).is_absolute(),
            "model.weight_cache_dir must be an absolute path (got {:?}); \
             leave it empty for $PIE_HOME/cache/weights",
            self.weight_cache_dir
        );
        ensure!(
            self.adapter_dir.is_empty() || Path::new(&self.adapter_dir).is_absolute(),
            "model.adapter_dir must be an absolute path (got {:?}); \
             leave it empty to mount no shared adapters at all",
            self.adapter_dir
        );
        for (key, rows) in [
            ("max_patches", self.max_patches),
            ("max_images", self.max_images),
        ] {
            ensure!(
                rows != Some(0),
                "model.{key} = 0 admits no image at all; omit the key to let the engine \
                 derive a ladder from the model text, or state a positive count"
            );
        }
        for (key, budget) in [
            ("device_weight_budget", self.device_weight_budget),
            ("host_weight_budget", self.host_weight_budget),
        ] {
            if let Some(budget) = budget {
                ensure!(
                    budget.as_bytes() > 0,
                    "model.{key} is zero, and no load can hold zero weight bytes; \
                     state a real ceiling or omit the key for uncapped"
                );
            }
        }
        self.adapters.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default = "default_request_timeout")]
    pub request_timeout: Duration,
    #[serde(default = "default_submit_deadline")]
    pub submit_deadline: Duration,
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout: Duration,
    #[serde(default = "default_frame_size")]
    pub frame_size: u32,
    #[serde(default = "default_frame_dispatch_depth")]
    pub frame_dispatch_depth: u32,
    #[serde(default)]
    pub max_concurrent_processes: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            request_timeout: default_request_timeout(),
            submit_deadline: default_submit_deadline(),
            silence_timeout: default_silence_timeout(),
            frame_size: default_frame_size(),
            frame_dispatch_depth: default_frame_dispatch_depth(),
            max_concurrent_processes: None,
        }
    }
}

impl RuntimeConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.request_timeout.as_micros() > 0,
            "runtime.request_timeout must be > 0"
        );
        ensure!(
            self.submit_deadline.as_micros() > 0,
            "runtime.submit_deadline must be > 0"
        );
        ensure!(
            self.silence_timeout.as_micros() > 0,
            "runtime.silence_timeout must be > 0"
        );
        ensure!(
            self.silence_timeout >= self.submit_deadline,
            "runtime.silence_timeout must not be shorter than submit_deadline: \
             a kill that lands before the leash would fail guests the leash exists to spare"
        );
        ensure!(self.frame_size >= 1, "runtime.frame_size must be >= 1");
        ensure!(
            self.frame_dispatch_depth >= 1,
            "runtime.frame_dispatch_depth must be >= 1"
        );
        ensure!(
            self.frame_size <= u32::from(Runahead::STEPS_MAX),
            "runtime.frame_size must be at most {} (got {}): it is `k` in the engine's \
             staging formula `frames_in_flight * k + 1`, and the frame scheduler was \
             built and measured around that bound \
             (`engine::runahead::Runahead::STEPS_MAX`)",
            Runahead::STEPS_MAX,
            self.frame_size
        );
        ensure!(
            self.frame_dispatch_depth <= u32::from(Runahead::MAX_FRAMES),
            "runtime.frame_dispatch_depth must be at most {} (got {}): the engine \
             publishes its staging ring's free set as one 64-bit word, and \
             `frames_in_flight * {} + 1` must fit in it \
             (`engine::runahead::Runahead::MAX_FRAMES`)",
            Runahead::MAX_FRAMES,
            self.frame_dispatch_depth,
            Runahead::STEPS_MAX
        );
        if let Some(n) = self.max_concurrent_processes {
            ensure!(n > 0, "runtime.max_concurrent_processes must be > 0 if set");
        }
        Ok(())
    }
}

fn default_request_timeout() -> Duration {
    Duration::from_secs(120)
}

fn default_submit_deadline() -> Duration {
    Duration::from_millis(50)
}

fn default_silence_timeout() -> Duration {
    Duration::from_secs(30)
}

fn default_frame_size() -> u32 {
    2
}

fn default_frame_dispatch_depth() -> u32 {
    2
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EngineConfig {
    #[serde(rename = "type")]
    pub kind: EngineKind,
    #[serde(deserialize_with = "deserialize_string_or_list")]
    pub device: Vec<String>,
    #[serde(default = "default_tp_size")]
    pub tensor_parallel_size: u32,
    #[serde(default = "default_activation_dtype")]
    pub activation_dtype: String,
    #[serde(default)]
    pub options: toml::Table,
}

impl EngineConfig {
    fn validate(&self) -> Result<()> {
        ensure!(!self.device.is_empty(), "engine.device must be non-empty");
        match self.kind {
            EngineKind::CudaNative => {
                let opts: CudaNativeEngineOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [engine] options for engine type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                opts.validate()?;
            }
            EngineKind::Metal => {
                if let Some(fraction) = self
                    .options
                    .get("gpu_mem_utilization")
                    .and_then(toml::Value::as_float)
                {
                    ensure!(
                        fraction.is_finite() && fraction > 0.0 && fraction <= 1.0,
                        "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
                    );
                }
            }
            EngineKind::Vulkan => {
                let opts: VulkanEngineOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [engine] options for engine type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                opts.validate()?;
            }
            EngineKind::Wgpu => {
                let opts: WgpuEngineOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                    anyhow::anyhow!(
                        "invalid [engine] options for engine type {:?}: {e}",
                        self.kind,
                    )
                })?;
                opts.validate()?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    CudaNative,
    Metal,
    Vulkan,
    Wgpu,
}

impl EngineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::CudaNative => "cuda_native",
            EngineKind::Metal => "metal",
            EngineKind::Vulkan => "vulkan",
            EngineKind::Wgpu => "wgpu",
        }
    }
}

fn default_tp_size() -> u32 {
    1
}
fn default_activation_dtype() -> String {
    "bfloat16".to_string()
}
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

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_METAL: &str = r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "metal"
device = ["cpu"]
"#;

    #[test]
    fn config_every_case() {
        rejects_the_legacy_unit_suffixed_names();
        a_silence_timeout_under_the_submit_deadline_is_refused();
        the_diag_flag_states_the_engines_diagnostics_key();
        a_stated_diagnostics_key_survives_the_reshape();
        parses_minimal_metal_config();
        every_engine_kind_round_trips_through_its_config_string();
        adapters_are_absent_by_default_and_that_is_zero_seats();
        the_seat_count_is_stated_once_or_derived_from_the_roster();
        an_adapter_past_its_seats_or_on_a_relative_path_is_refused_by_name();
        rejects_a_cache_section();
    }

    fn rejects_the_legacy_unit_suffixed_names() {
        for (section, legacy) in [
            ("sandbox", "wasm_max_memory_mb = 4096"),
            ("sandbox", "wasm_warm_memory_mb = 0"),
            ("server", "max_upload_mb = 256"),
            ("runtime", "request_timeout_secs = 120"),
            ("runtime", "submit_deadline_us = 50000"),
            ("runtime", "silence_timeout_secs = 30"),
        ] {
            let toml = format!("{MINIMAL_METAL}\n[{section}]\n{legacy}\n");
            assert!(
                toml::from_str::<Config>(&toml).is_err(),
                "{legacy} should no longer parse"
            );
        }
    }

    fn a_silence_timeout_under_the_submit_deadline_is_refused() {
        let toml = format!(
            "{MINIMAL_METAL}\n[runtime]\n\
             submit_deadline = \"5s\"\nsilence_timeout = \"1s\"\n"
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("must not be shorter than submit_deadline"),
            "got: {err}"
        );
    }

    fn the_diag_flag_states_the_engines_diagnostics_key() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.state_diagnostics("tier-trace,kernel-profile=2")
            .expect("the metal shell has a diagnostics record");
        assert_eq!(
            cfg.model
                .engine
                .options
                .get("diagnostics")
                .and_then(toml::Value::as_str),
            Some("tier-trace,kernel-profile=2"),
            "the words reach the engine's own options table"
        );

        let vulkan = MINIMAL_METAL.replace("type = \"metal\"", "type = \"vulkan\"");
        let mut cfg: Config = toml::from_str(&vulkan).unwrap();
        let why = cfg
            .state_diagnostics("tier-trace")
            .expect_err("the vulkan shell has none");
        assert!(
            why.to_string().contains("vulkan"),
            "and the refusal names the flavor: {why}"
        );
    }

    fn a_stated_diagnostics_key_survives_the_reshape() {
        let toml = format!("{MINIMAL_METAL}\n[engine]\ndiagnostics = \"cut-trace\"\n");
        let cfg = Config::parse(&toml).expect("an [engine] table beside [model.engine]");
        assert_eq!(
            cfg.model
                .engine
                .options
                .get("diagnostics")
                .and_then(toml::Value::as_str),
            Some("cut-trace"),
            "`[engine] diagnostics` lands in the engine-specific options bag"
        );
    }

    fn parses_minimal_metal_config() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.engine.kind, EngineKind::Metal);
        assert_eq!(cfg.model.engine.device, vec!["cpu".to_string()]);
        assert_eq!(cfg.server.port, 8080);
    }

    fn every_engine_kind_round_trips_through_its_config_string() {
        const KINDS: &[(EngineKind, &str)] = &[
            (EngineKind::CudaNative, "cuda_native"),
            (EngineKind::Metal, "metal"),
            (EngineKind::Vulkan, "vulkan"),
            (EngineKind::Wgpu, "wgpu"),
        ];
        for (kind, spelled) in KINDS {
            assert_eq!(kind.as_str(), *spelled, "{kind:?} names itself");
            let toml = format!(
                "[model]\nname = \"default\"\nmodel = \"Qwen/Qwen3-0.6B\"\n\n\
                 [model.engine]\ntype = \"{spelled}\"\ndevice = [\"cpu\"]\n"
            );
            let cfg: Config = toml::from_str(&toml)
                .unwrap_or_else(|e| panic!("`type = \"{spelled}\"` does not parse: {e}"));
            assert_eq!(cfg.model.engine.kind, *kind);
            cfg.validate()
                .unwrap_or_else(|e| panic!("a minimal `{spelled}` config does not validate: {e}"));
            let round = toml::Value::try_from(*kind).expect("a kind serializes");
            assert_eq!(round.as_str(), Some(*spelled));
        }
        assert_eq!(
            KINDS.len(),
            4,
            "an engine kind was added without a line here, so nothing checks its \
             config spelling"
        );
    }

    fn adapters_are_absent_by_default_and_that_is_zero_seats() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.model.adapters.is_empty());
        assert_eq!(cfg.model.adapters.seats(), 0);
        assert!(cfg.model.adapters.registered.is_empty());
    }

    fn the_seat_count_is_stated_once_or_derived_from_the_roster() {
        let toml = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\n\
             [[model.adapters.registered]]\n\
             id = 2\n\
             planes = { \"layer.0.lora_a\" = \"/adapters/0/a.bin\" }\n",
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.adapters.seats(), 3);
        assert_eq!(cfg.model.adapters.registered.len(), 1);
        assert_eq!(
            cfg.model.adapters.registered[0].planes["layer.0.lora_a"],
            "/adapters/0/a.bin"
        );

        let stated = toml.replace("[model.adapters]", "[model.adapters]\nseats = 8");
        let cfg: Config = toml::from_str(&stated).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.adapters.seats(), 8);
    }

    fn an_adapter_past_its_seats_or_on_a_relative_path_is_refused_by_name() {
        let past = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseats = 2\n\
             [[model.adapters.registered]]\n\
             id = 5\n",
        );
        let cfg: Config = toml::from_str(&past).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("adapter id 5 is past the 2 seat"),
            "got: {err}"
        );

        let relative = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\n\
             [[model.adapters.registered]]\n\
             id = 0\n\
             planes = { \"layer.0.lora_a\" = \"a.bin\" }\n",
        );
        let cfg: Config = toml::from_str(&relative).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("must be \n             an absolute path")
                || err.contains("absolute path"),
            "got: {err}"
        );

        let typo = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseat = 2\n",
        );
        toml::from_str::<Config>(&typo)
            .expect_err("a near-miss of `seats` must be refused by name");
    }

    fn rejects_a_cache_section() {
        let toml = format!("{MINIMAL_METAL}\n[cache]\nptir_dir = \"/tmp/x\"\n");
        assert!(toml::from_str::<Config>(&toml).is_err());
    }

}
