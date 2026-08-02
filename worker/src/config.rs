//! Pie standalone server config — TOML schema mirror of `pie.config`.
//!
//! Same TOML the legacy Python server consumed. Embedded drivers
//! ([`DriverKind::CudaNative`] / [`DriverKind::Metal`] / [`DriverKind::Dummy`])
//! are dispatched in [`crate::serve::start_engine`] via
//! [`crate::serve::topology::resolve_flavor`].
//!
//! The Rust [`Config`] type below is the user-facing TOML schema; the
//! conversion to `pie_engine::bootstrap::Config` (the runtime's own config)
//! happens in [`crate::translate`].

use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use pie_controller_rpc::Role;
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// Top-level
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Client-facing listener, plus what pie fetches inferlets from.
    #[serde(default)]
    pub server: ServerConfig,
    /// OpenTelemetry export. Off by default.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// Tokio + wasmtime tuning and the sandbox an inferlet runs in.
    #[serde(default)]
    pub runtime: RuntimeConfig,
    /// Distributed-cluster topology (controller + role + gateways). Absent, or
    /// `controller` unset ⇒ single-node (gateway-free local inference).
    #[serde(default)]
    pub cluster: ClusterConfig,
    /// Limits on remote clients leasing this worker's KV space.
    #[serde(default)]
    pub executor: ExecutorConfig,
    /// Disaggregated serving: moving prefill and KV to partner workers.
    #[serde(default)]
    pub offload: OffloadConfig,
    /// The single `[model]` table. Pie serves exactly one model.
    pub model: ModelConfig,
}

impl Config {
    /// Parse a TOML config string into a validated [`Config`]. **Pure**: no file
    /// IO, no env, no clap — sourcing the string (file locate/read + env merge)
    /// is the bin layer's job (Seam 2). The role lib owns only the
    /// domain parse + validation.
    /// Parse the operator's file.
    ///
    /// The file's sections are not this struct's fields -- see
    /// [`crate::config_layout`] for why and for the mapping. Reshaping first
    /// means everything below still sees the shape it was written against, and
    /// the driver options still land in a `deny_unknown_fields` struct.
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
        let reshaped = crate::config_layout::reshape(file)?;
        let s = &toml::to_string(&reshaped).map_err(|e| anyhow::anyhow!("reshape config: {e}"))?;
        let cfg: Config = toml::from_str(s).map_err(|e| {
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
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.server.validate()?;
        self.runtime.validate()?;
        self.cluster.validate()?;
        self.executor.validate()?;
        self.offload.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutorConfig {
    /// Remote clients that may hold a scratch lease at once.
    ///
    /// Not just a connection count: the KV pool is divided evenly between the
    /// slots, so each client gets `total_pages / max_clients` pages and raising
    /// this shrinks every client's share. Refused at validation if it exceeds
    /// the pool's page count, since a slot with no pages cannot serve anyone.
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
    /// Serve prefill and KV from partner workers rather than only locally.
    ///
    /// Off by default. Turning it on also makes the worker publish an artifact
    /// digest, because partners must agree on the weight layout they trade
    /// pages in.
    #[serde(default)]
    pub enabled: bool,
    /// Shortest suffix worth offloading a prefill for. `0` derives one from
    /// the transport -- 512 tokens over NIXL, 2048 inline -- since the
    /// worthwhile length is a property of how fast the pages move.
    #[serde(default)]
    pub prefill_min_suffix_tokens: usize,
    /// Transfers in flight to any one partner before the next waits.
    #[serde(default = "default_offload_max_outstanding")]
    pub max_outstanding_per_partner: u32,
    /// How KV pages cross between workers: `inline` in the message,
    /// `nixl` via RDMA, or `auto` to use NIXL where it is available.
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

// -----------------------------------------------------------------------------
// [cluster]
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Units
// -----------------------------------------------------------------------------
//
// A duration or a size is written with its unit -- `"50ms"`, `"4GiB"` -- rather
// than carried in the field name. The old spelling put the unit in the name and
// the number in the value, which meant the two could disagree only silently and
// that the file could not be read without knowing the schema: `submit_deadline_us
// = 50000` and `request_timeout_secs = 120` sat in the same table.
//
// It also drifted. Before this, one file held `request_timeout_secs` and
// `silence_timeout_secs` next to `ready_timeout_s` and `shutdown_timeout_s` --
// two spellings of "seconds" -- and `_us`, `_mb`, `_gb` besides.

/// A duration written with its unit: `"50ms"`, `"120s"`, `"2m"`.
///
/// Accepted units: `ns`, `us`, `ms`, `s`, `m`, `h`. A bare number is refused
/// rather than assumed to be seconds -- assuming is how `_us` and `_secs` came
/// to live in one table.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration(std::time::Duration);

impl Duration {
    pub const fn from_millis(ms: u64) -> Self {
        Self(std::time::Duration::from_millis(ms))
    }
    pub const fn from_secs(s: u64) -> Self {
        Self(std::time::Duration::from_secs(s))
    }
    pub fn as_micros(&self) -> u64 {
        self.0.as_micros().min(u64::MAX as u128) as u64
    }
    pub fn as_secs(&self) -> u64 {
        self.0.as_secs()
    }
    pub fn as_secs_f64(&self) -> f64 {
        self.0.as_secs_f64()
    }
}

fn parse_duration(text: &str) -> std::result::Result<Duration, String> {
    let t = text.trim();
    let split = t
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .ok_or_else(|| {
            format!("duration {t:?} has no unit; write one of 120s, 50ms, 2m")
        })?;
    let (value, unit) = t.split_at(split);
    let value: f64 = value
        .parse()
        .map_err(|_| format!("duration {t:?} has an unparseable number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("duration {t:?} must be finite and non-negative"));
    }
    let nanos = match unit.trim() {
        "ns" => value,
        "us" | "\u{b5}s" => value * 1e3,
        "ms" => value * 1e6,
        "s" => value * 1e9,
        "m" => value * 6e10,
        "h" => value * 3.6e12,
        other => {
            return Err(format!(
                "duration {t:?} has unknown unit {other:?}; use ns, us, ms, s, m, h"
            ));
        }
    };
    Ok(Duration(std::time::Duration::from_nanos(nanos as u64)))
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        parse_duration(&raw).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Duration {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let us = self.as_micros();
        if us % 1_000_000 == 0 {
            s.serialize_str(&format!("{}s", us / 1_000_000))
        } else if us % 1_000 == 0 {
            s.serialize_str(&format!("{}ms", us / 1_000))
        } else {
            s.serialize_str(&format!("{us}us"))
        }
    }
}

/// A byte size written with its unit: `"256MiB"`, `"4GiB"`.
///
/// Binary units only (`B`, `KiB`, `MiB`, `GiB`, `TiB`), because that is what
/// the `_mb`/`_gb` fields this replaces always meant -- each multiplied by
/// 1024*1024, never 1000*1000.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteSize(u64);

impl ByteSize {
    pub const fn from_mib(mib: u64) -> Self {
        Self(mib * 1024 * 1024)
    }
    pub const fn as_bytes(&self) -> u64 {
        self.0
    }
    pub const fn as_mib(&self) -> u64 {
        self.0 / (1024 * 1024)
    }
    pub fn as_gib_f64(&self) -> f64 {
        self.0 as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

fn parse_byte_size(text: &str) -> std::result::Result<ByteSize, String> {
    let t = text.trim();
    let split = t
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .ok_or_else(|| {
            format!("size {t:?} has no unit; write one of 512B, 256MiB, 4GiB")
        })?;
    let (value, unit) = t.split_at(split);
    let value: f64 = value
        .parse()
        .map_err(|_| format!("size {t:?} has an unparseable number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("size {t:?} must be finite and non-negative"));
    }
    let scale: f64 = match unit.trim() {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        "TiB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => {
            return Err(format!(
                "size {t:?} has unknown unit {other:?}; use B, KiB, MiB, GiB, TiB \
                 (binary units only)"
            ));
        }
    };
    Ok(ByteSize((value * scale) as u64))
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        parse_byte_size(&raw).map_err(serde::de::Error::custom)
    }
}

impl Serialize for ByteSize {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;
        const KIB: u64 = 1024;
        let b = self.0;
        let text = if b % GIB == 0 && b != 0 {
            format!("{}GiB", b / GIB)
        } else if b % MIB == 0 && b != 0 {
            format!("{}MiB", b / MIB)
        } else if b % KIB == 0 && b != 0 {
            format!("{}KiB", b / KIB)
        } else {
            format!("{b}B")
        };
        s.serialize_str(&text)
    }
}

/// Distributed-cluster topology. Absent, or `controller` unset ⇒ single-node
/// (the worker terminates clients directly; no controller/gateway).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterConfig {
    /// Controller endpoint (`tcp://host:port`, a bare `host:port`, or
    /// `unix:/path`); set ⇒ this worker joins a distributed cluster.
    #[serde(default)]
    pub controller: Option<String>,
    /// This worker's role (required when `controller` is set).
    #[serde(default)]
    pub role: Option<Role>,
    /// Gateway endpoint(s) this worker dials INTO (M3 inversion; distributed).
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

// -----------------------------------------------------------------------------
// [server]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address the client edge binds. Loopback by default -- a reachable port
    /// should be something an operator asks for.
    #[serde(default = "default_host")]
    pub host: String,
    /// Port the client edge binds.
    #[serde(default = "default_port")]
    pub port: u16,
    /// Verbose engine logging, and passed down to the embedded driver.
    /// Independent of `--log-level`, which sets the tracing filter.
    #[serde(default)]
    pub verbose: bool,
    /// Where `pie inferlet` downloads from, and where the engine fetches a
    /// program it is asked to run but does not have.
    #[serde(default = "default_registry")]
    pub registry: String,
    /// Hard cap on inferlets admitted at once. **Omit to derive it** from the
    /// driver's `max_forward_requests`, which is what fills a batch.
    ///
    /// A physical safety limit, not a scheduling knob: past the point where
    /// every batch is full, more concurrent processes add queueing and not
    /// throughput.
    #[serde(default)]
    pub max_concurrent_processes: Option<usize>,
    /// Apply the host-side snapshot optimization to Python components.
    ///
    /// On by default. It only affects startup cost, so turning it off is a
    /// debugging step -- it changes which wasmtime linker variant is built.
    #[serde(default = "default_true")]
    pub python_snapshot: bool,
    /// Ask this boot to measure the memory planner instead of scoring it.
    ///
    /// Set by `pie config tune` on the config it derives in memory, never
    /// read from a file — `#[serde(skip)]`, so it cannot be written down and
    /// cannot outlive the boot that asked for it. See
    /// [`CudaNativeDriverOptions::calibrate_planner`] for why a measurement
    /// must not be a persisted setting.
    ///
    /// It rides here rather than in `[model.driver.options]` because that table
    /// is the file's, and this never comes from the file. Same route as
    /// `verbose`: a typed field the engine applies to whatever driver options it
    /// builds.
    #[serde(skip)]
    pub calibrate_planner: bool,
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
            calibrate_planner: false,
        }
    }
}

impl ServerConfig {
    fn validate(&self) -> Result<()> {
        if let Some(n) = self.max_concurrent_processes {
            ensure!(n > 0, "runtime.max_concurrent_processes must be > 0 if set");
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
// [telemetry]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// Export traces over OTLP.
    #[serde(default)]
    pub enabled: bool,
    /// OTLP collector to export to.
    #[serde(default = "default_otlp_endpoint")]
    pub endpoint: String,
    /// `service.name` on exported spans.
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
    /// Tokio worker threads. Derived from the visible CPUs, capped at 64.
    ///
    /// The cap is measured, not arbitrary: past ~64 the runtime's own
    /// scheduling overhead adds variance without adding parallelism (on a
    /// 256-thread EPYC 7773X, +0.5% mean tok/s and about triple the stdev
    /// without it). Raise it only for heavy non-inference work in-process.
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    /// Instances the wasmtime pooling allocator may hold. A ceiling on
    /// concurrent inferlets, reserved up front.
    #[serde(default = "default_wasm_max_instances")]
    pub wasm_max_instances: u32,
    /// Linear memory one instance may address.
    #[serde(default = "default_wasm_max_memory")]
    pub wasm_max_memory: ByteSize,
    /// Linear memory kept resident when an instance is returned to the pool,
    /// rather than decommitted. Trades RSS for a cheaper next start; `0B`
    /// keeps none.
    #[serde(default)]
    pub wasm_warm_memory: ByteSize,
    /// Unused pool slots kept warm rather than torn down.
    #[serde(default = "default_wasm_warm_slots")]
    pub wasm_warm_slots: u32,
    /// Give each inferlet a private `/scratch` with read-write access.
    /// Off by default: nothing else in the sandbox reaches a filesystem.
    #[serde(default)]
    pub allow_fs: bool,
    /// Where the per-process `/scratch` directories are made. Ignored unless
    /// `allow_fs` is set.
    #[serde(default = "default_fs_scratch_dir")]
    pub fs_scratch_dir: PathBuf,
    /// Allow outbound network from inferlets at all. `false` is the tight
    /// setting -- it is the only one that also stops `wasi:http`.
    #[serde(default = "default_true")]
    pub allow_network: bool,
    /// Hosts an inferlet may reach, `["*"]` for any.
    ///
    /// Filters `wasi:sockets` only. `wasi:http` bypasses the per-socket hook
    /// because its host stack resolves names itself, so this list does not
    /// constrain it -- use `allow_network = false` when that matters.
    #[serde(default = "default_network_allowed_hosts")]
    pub network_allowed_hosts: Vec<String>,
    /// Largest blob a client may upload in one request.
    #[serde(default = "default_max_upload")]
    pub max_upload: ByteSize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: default_worker_threads(),
            wasm_max_instances: default_wasm_max_instances(),
            wasm_max_memory: default_wasm_max_memory(),
            wasm_warm_memory: ByteSize::from_mib(0),
            wasm_warm_slots: default_wasm_warm_slots(),
            allow_fs: false,
            fs_scratch_dir: default_fs_scratch_dir(),
            allow_network: true,
            network_allowed_hosts: default_network_allowed_hosts(),
            max_upload: default_max_upload(),
        }
    }
}

impl RuntimeConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.worker_threads > 0,
            "server.worker_threads must be > 0"
        );
        ensure!(
            self.wasm_max_instances > 0,
            "sandbox.max_instances must be > 0"
        );
        ensure!(
            self.wasm_max_memory.as_bytes() > 0,
            "sandbox.max_memory must be > 0"
        );
        ensure!(
            self.max_upload.as_bytes() > 0,
            "server.max_upload must be > 0"
        );
        Ok(())
    }
}

fn default_worker_threads() -> usize {
    // Cap at 64 — pie's scheduler produces enough concurrent host work
    // at high request concurrency. Beyond ~64 workers the runtime's
    // scheduling overhead (queue management, wake propagation) starts
    // adding variance without adding parallelism. Measured on AMD EPYC
    // 7773X (256 threads visible): tok/s mean +0.5%, stdev cut to ~1/3
    // by capping. Users with heavier non-inference work in the same
    // process can override via `[runtime] worker_threads = ...`.
    std::thread::available_parallelism()
        .map(|n| n.get().min(64))
        .unwrap_or(4)
}
fn default_wasm_max_instances() -> u32 {
    1000
}
fn default_wasm_max_memory() -> ByteSize {
    ByteSize::from_mib(4096)
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
fn default_max_upload() -> ByteSize {
    ByteSize::from_mib(256)
}

// -----------------------------------------------------------------------------
// [model]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// What clients ask for this model by. Required, and free-form: it names
    /// the deployment, not the checkpoint.
    pub name: String,
    /// What to serve: a store name (`Qwen--Qwen3-0.6B`, as `pie model list`
    /// prints it) or a path to a `.zt` artifact. See `weights::resolve`.
    ///
    /// Named `hf_repo` until the store existed, and still accepted under that
    /// name — the struct is `deny_unknown_fields`, so a bare rename would greet
    /// an existing config with *unknown field `hf_repo`* and *missing field
    /// `model`* at startup rather than with a working boot.
    #[serde(alias = "hf_repo")]
    pub model: String,
    /// Which backend runs the model, on what devices.
    pub driver: DriverConfig,
    /// Batching and timeout policy. Every field has a measured default; the
    /// frame knobs are part of the guest contract.
    #[serde(default)]
    pub scheduler: SchedulerConfig,
    /// Where this model's materialized-weight artifacts are kept between runs.
    ///
    /// Per-model rather than process-global because the artifact IS the model:
    /// it is that checkpoint's weights, already laid out for this driver, and
    /// none of it is shared with another model. Empty derives
    /// `$PIE_HOME/cache/weights` -- a cache, beside the driver's other ones.
    /// NOT `$PIE_HOME/models`, which is the `.zt` artifact store: an artifact
    /// is portable and costs a re-download to replace, while these are device
    /// bytes for one driver, one TP layout and one ABI version, rebuilt by a
    /// single cold load.
    ///
    /// The artifacts are the size of the weights -- tens to hundreds of GB --
    /// which is why this is a path pie cannot always pick for you. The driver
    /// declines a write it has no room for rather than filling the disk.
    #[serde(default)]
    pub weight_cache_dir: String,
}

impl ModelConfig {
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
        self.driver.validate()?;
        self.scheduler.validate()?;
        // Relative resolves against the driver's working directory, which the
        // operator did not choose and which differs between the worker and a
        // driver launched as its own process.
        ensure!(
            self.weight_cache_dir.is_empty()
                || Path::new(&self.weight_cache_dir).is_absolute(),
            "model.weight_cache_dir must be an absolute path (got {:?}); \
             leave it empty for $PIE_HOME/cache/weights",
            self.weight_cache_dir
        );
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// [model.scheduler]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerConfig {
    /// How long a client request may run before the engine gives up on it.
    ///
    /// The outermost of the three clocks here and the only one about the
    /// client rather than the pipeline: `submit_deadline` leashes a straggler
    /// and `silence_timeout` terminates an abandoned process, but this one
    /// bounds the answer a caller is waiting for.
    #[serde(default = "default_request_timeout")]
    pub request_timeout: Duration,
    /// How long a pipeline that is HARD-BLOCKING a frame's seal may go
    /// without submitting before the engine stops waiting for it, in
    /// microseconds.
    ///
    /// This does not fail the pipeline. At the deadline the lane is dropped
    /// from the wait-set — an involuntary `forward.park()` — so the boundary
    /// seals at once; frames it already submitted still dispatch, and its
    /// next fire rejoins it. What the number buys is therefore epoch density,
    /// not safety: how long the fleet waits for a straggler before going on
    /// without it. Setting it too low costs a little density and never a
    /// request. Termination is a separate and far longer verdict, in
    /// `silence_timeout_secs`.
    ///
    /// Small (50ms) because it measures a much narrower interval than its size
    /// suggests. The clock runs only while the lane is an awaited member with
    /// nothing submitted — it is stopped by run-ahead (a lane with a queued
    /// frame is not blocking), by an unretired dispatch (the engine owes it a
    /// result, so the whole GPU wave is free), by a bind in flight, and by
    /// `forward.park()`. The host resubmit round trip already has its own
    /// headroom in [`SchedulerConfig::frame_submit_depth`].
    ///
    /// Measured: on the contention suite a breach happens roughly once in
    /// several thousand requests, and when it does the lane is 0.1-3ms over
    /// the line — an ordinary task-wakeup tail, not a broken guest. Killing
    /// for that was the wrong response, which is why this only leashes now.
    ///
    /// Exposed to guests verbatim as `model.submit-deadline-us()`.
    #[serde(default = "default_submit_deadline")]
    pub submit_deadline: Duration,
    /// How long a lane may stay silent in total — through the leash above and
    /// on past it — before the engine terminates its process, in seconds.
    ///
    /// This one IS a verdict, so it is generous. A pipeline that means to go
    /// quiet calls `forward.park()`, which ends the silence and is never
    /// killed however long it stays away; that is the contract this enforces.
    /// The leash already keeps a straggler from holding the fleet, so nothing
    /// but a genuinely abandoned process ever reaches this.
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout: Duration,
    /// Waves per frame (*k*): how many token steps the wait-all quorum admits
    /// before it runs. A deployment constant, fixed at engine start exactly
    /// like the KV page size — never renegotiated per frame, never adapted
    /// from runtime timing. Guests read it as `model.frame-size()` and size
    /// their frames and channels to it, so it is part of the guest contract.
    ///
    /// Two. At k=1 the quorum runs once per token, and above ~64 concurrent
    /// processes the fleet stops overlapping batches entirely: measured duty
    /// collapses from 1.7 to 1.0 and goes bimodal, costing 29% throughput and
    /// 28% latency at concurrency 256. k=2 halves the number of quorum
    /// boundaries and holds duty at 1.6 with no regression at any lower
    /// concurrency. k=3 and k=4 measure the same as k=2 while costing more
    /// driver staging depth (CONTENTION_FOLLOWUP §20.8).
    ///
    /// Bounded above by the CUDA driver, not by taste — see [`Self::validate`].
    #[serde(default = "default_frame_size")]
    pub frame_size: u32,
    /// Frames a guest keeps submitted into the engine: one running, plus the
    /// rest queued behind it so the device always has work while the guest is
    /// back on the host deciding what to submit next. Too few and the pipeline collapses
    /// to lockstep — submit, wait, submit, wait — because the host resubmit
    /// round trip lands squarely in the critical path.
    ///
    /// Three, sized for the default `frame_size = 2`, where it covers the
    /// round trip with one frame to spare. **A frame is k waves of device
    /// time, so this and `frame_size` are not independent: a deployment that
    /// moves k must re-measure this.** Three frames at k=1 cover half the real
    /// time they cover at k=2, and undersizing this window is what collapsed
    /// k=1 throughput (CONTENTION_FOLLOWUP §20.11).
    ///
    /// Guests never read this directly — they get `model.channel-capacity()`,
    /// which is derived from it — so it can move without touching the guest
    /// contract. Not to be confused with `frame_dispatch_depth`, which is the
    /// engine running ahead of the DRIVER rather than the guest running ahead
    /// of the engine.
    #[serde(default = "default_frame_submit_depth")]
    pub frame_submit_depth: u32,
    /// Frames the engine keeps POSTED to the driver but not yet retired: the
    /// dispatch loop's enqueue horizon. Where `frame_submit_depth` is the
    /// guest running ahead of the engine, this is the engine running ahead of
    /// the device, and it is what keeps the GPU from idling between frames —
    /// at 2 one frame executes while the next is already uploaded.
    ///
    /// **This is a real two-sided trade-off, not a value with one right
    /// answer.** Dispatching is the allocation-credit gate, so each frame of
    /// depth commits physical KV pages early; and a frame's contents freeze
    /// when it is sealed, so deeper dispatch executes batches that were packed
    /// against a staler queue. On hardware with headroom the idle it removes
    /// wins and 2 is right — that is where the default was tuned. On a fully
    /// batched, zero-headroom fleet there is no idle left to remove and the
    /// two costs dominate: measured 610ms at depth 1 against 1034ms at depth 2
    /// on a pure-attention model, 8 lanes, ~29 of 32 rows per launch.
    ///
    /// Bounded jointly with `frame_size` by the driver — see [`Self::validate`].
    #[serde(default = "default_frame_dispatch_depth")]
    pub frame_dispatch_depth: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            request_timeout: default_request_timeout(),
            submit_deadline: default_submit_deadline(),
            silence_timeout: default_silence_timeout(),
            frame_size: default_frame_size(),
            frame_submit_depth: default_frame_submit_depth(),
            frame_dispatch_depth: default_frame_dispatch_depth(),
        }
    }
}

/// Pinned upload staging slots the CUDA driver allocates, mirroring
/// `kUploadStagingDepth` in `driver/cuda/src/runahead.hpp` (there:
/// `kSchedulerMaxInFlight * kMaxPipelinedFrameSize + 1` = 3 * 4 + 1).
///
/// The driver stages per STEP, not per frame, and one frame carries up to
/// `frame_size` steps — so the engine's steps in flight are
/// `frame_dispatch_depth * frame_size` and the pool must strictly EXCEED
/// them. A pool sized exactly to peak occupancy has every slot pending when
/// the next submit arrives, so the acquire blocks in `cudaEventSynchronize`
/// for a full GPU step (~1.6ms measured on the 4090 c64 decode workload).
///
/// Exceeding it is never a memory error — the pools are fixed arrays, so a
/// submit simply waits for a slot and every submit re-serializes with no
/// diagnostic at all. That silence is why this is rejected here rather than
/// clamped: the config layer is the only place it can be caught.
/// Public because `pie config tune` enumerates knob candidates against
/// this bound rather than generating combinations the engine will refuse --
/// and a second copy of the number is exactly the disagreement this
/// constant's own doc warns about.
pub const UPLOAD_STAGING_DEPTH: u32 = 13;

impl SchedulerConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.request_timeout.as_micros() > 0,
            "scheduler.request_timeout must be > 0"
        );
        ensure!(
            self.submit_deadline.as_micros() > 0,
            "scheduler.submit_deadline must be > 0"
        );
        ensure!(
            self.silence_timeout.as_micros() > 0,
            "scheduler.silence_timeout must be > 0"
        );
        ensure!(
            self.silence_timeout >= self.submit_deadline,
            "scheduler.silence_timeout must not be shorter than submit_deadline: \
             a kill that lands before the leash would fail guests the leash exists to spare"
        );
        ensure!(self.frame_size >= 1, "scheduler.frame_size must be >= 1");
        ensure!(
            self.frame_dispatch_depth >= 1,
            "scheduler.frame_dispatch_depth must be >= 1"
        );
        ensure!(
            self.frame_submit_depth >= 2,
            "scheduler.frame_submit_depth must be at least 2: one frame runs while the rest \
             stay queued, so a value of 1 leaves nothing queued and puts the host resubmit \
             round trip back in the critical path — the lockstep this window exists to prevent"
        );
        // The driver coupling. Checked as the product rather than as a cap on
        // each factor, which is both correct and less restrictive: what the
        // staging pool sees is steps, and it cannot tell 2 frames of 3 steps
        // from 3 frames of 2.
        let steps_in_flight = self.frame_dispatch_depth * self.frame_size;
        ensure!(
            steps_in_flight < UPLOAD_STAGING_DEPTH,
            "scheduler.frame_dispatch_depth * frame_size must be < {UPLOAD_STAGING_DEPTH} \
             (got {} * {} = {steps_in_flight}): the CUDA driver stages one pinned upload slot \
             per STEP and allocates {UPLOAD_STAGING_DEPTH} of them, so at or above that every \
             submit blocks waiting for a slot and re-serializes with no error reported \
             (kUploadStagingDepth in driver/cuda/src/runahead.hpp)",
            self.frame_dispatch_depth,
            self.frame_size
        );
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

fn default_frame_submit_depth() -> u32 {
    3
}

fn default_frame_dispatch_depth() -> u32 {
    2
}

// -----------------------------------------------------------------------------
// [model.driver]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // forwarded to the embedded driver via TOML; not all
// fields are read on the Rust side yet.
pub struct DriverConfig {
    /// Driver discriminator. Embedded drivers (`cuda_native`,
    /// `metal`, `dummy`) run in-process.
    #[serde(rename = "type")]
    pub kind: DriverKind,
    /// Single string or list of strings — both accepted on input.
    #[serde(deserialize_with = "deserialize_string_or_list")]
    pub device: Vec<String>,
    /// Ranks the model is sharded across. Must divide the device list.
    #[serde(default = "default_tp_size")]
    pub tensor_parallel_size: u32,
    /// Compute dtype for activations, e.g. `"bfloat16"`. Separate from
    /// `weight_dtype` and `kv_cache_dtype`: a deployment can store weights
    /// narrower than it computes.
    #[serde(default = "default_activation_dtype")]
    pub activation_dtype: String,
    /// Seed for sampling. Also mixed into the artifact digest, so two workers
    /// with different seeds do not trade cached weight layouts.
    #[serde(default = "default_random_seed")]
    pub random_seed: u64,
    /// Driver-specific knobs. Embedded drivers parse this into typed
    /// option structs.
    #[serde(default)]
    pub options: toml::Table,
}

impl DriverConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.device.is_empty(),
            "driver.device must be non-empty"
        );
        match self.kind {
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
                validate_kv_cache_dtype(&opts.kv_cache_dtype)?;
            }
            DriverKind::Metal => {
                let opts: MetalDriverOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [model.driver.options] for driver type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                validate_kv_cache_dtype(&opts.kv_cache_dtype)?;
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
        }
        Ok(())
    }
}

fn validate_kv_cache_dtype(value: &str) -> Result<()> {
    const VALID: &[&str] = &[
        "auto",
        "bf16",
        "bfloat16",
        "fp8_e4m3",
        "fp8_e5m2",
        "int8_per_token_head",
        "fp8_per_token_head",
        "fp4_e2m1",
        "nvfp4",
    ];
    ensure!(
        VALID.contains(&value),
        "invalid kv_cache_dtype {:?}; expected one of: {}",
        value,
        VALID.join(", ")
    );
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DriverKind {
    /// Native CUDA driver — embedded as a static lib in `pie-worker`
    /// (requires `--features driver-cuda`).
    CudaNative,
    /// Rust dummy driver — random tokens, no model load. Always
    /// embedded in `pie-worker`.
    Dummy,
    /// Native MLX + Metal driver for Apple Silicon — embedded as a static
    /// lib in `pie-worker` (requires `--features driver-metal`, macOS only).
    Metal,
}

impl DriverKind {
    pub fn as_str(self) -> &'static str {
        match self {
            DriverKind::CudaNative => "cuda_native",
            DriverKind::Dummy => "dummy",
            DriverKind::Metal => "metal",
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

/// `[model.driver.options]` for `type = "metal"` (Apple Silicon MLX/Metal
/// driver) — page geometry, forward limits, and timeouts; the metal driver
/// speaks the embedded in-process ABI. `device` is the `metal:N` selector
/// filled from `model.driver.device`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetalDriverOptions {
    /// KV page size in tokens. Used as given -- unlike the CUDA driver, the
    /// Metal driver has no planner to derive one.
    pub kv_page_size: u32,
    /// KV pages to allocate. Used directly, and 1024 is a real default.
    ///
    /// The CUDA driver's nearest equivalent is `max_total_pages`, which is a
    /// different quantity with a name that now says so: a ceiling over a
    /// derived number, usually absent.
    pub total_pages: u32,
    /// Tokens one forward pass may carry, across all requests in the batch.
    pub max_forward_tokens: u32,
    /// Requests one forward pass may carry. Also what `max_concurrent_processes`
    /// derives from when the operator leaves it unset.
    pub max_forward_requests: u32,
    /// Host-memory KV pages to swap into. `0` disables swapping.
    pub cpu_pages: u32,
    /// Tokens the KV ring holds across the whole resident fleet. Absent -- the
    /// default -- keeps the driver's own constant, which is what a `pie serve`
    /// fleet wants and what every run got before this existed.
    ///
    /// The one knob that shrinks the KV, and it only shrinks: the driver
    /// clamps to its own ceiling, so this cannot ask for a ring it will not
    /// build. `total_pages` is NOT that knob and never was -- the simple
    /// families derive their pool from this context and discard it.
    pub max_model_len: Option<u32>,
    /// Dtype KV pages are stored in. `"auto"` follows the activation dtype;
    /// a narrower one buys pages at some accuracy.
    pub kv_cache_dtype: String,
    /// Page routed MoE experts in from a mapping of the checkpoint instead of
    /// keeping every expert resident in the heap.
    ///
    /// The same knob, spelled the same way, as the CUDA driver's -- because it
    /// is the same decision: a residency trade the operator makes about a
    /// model. What the two backends *do* with it differs (CUDA copies through
    /// a bounded slab, Metal binds over a file-backed mapping and lets the
    /// kernel evict), which is a backend's business and not the operator's.
    ///
    /// Off by default: it trades resident memory for page faults, which only
    /// pays when the weights do not comfortably fit.
    pub stream_routed_experts: bool,
    /// How many bytes the routed experts may occupy on the device, or `None`
    /// to keep the whole bank resident.
    ///
    /// A stronger statement than `stream_routed_experts` and a different
    /// mechanism, not a dial on the same one. Streaming binds the bank over a
    /// mapping, and on Apple Silicon every mapped page is WIRED -- so it moves
    /// bytes out of the heap but bounds nothing. A budget turns the mapping
    /// off and pages experts through a slab of exactly this size, which is the
    /// only setting under which a checkpoint larger than the machine can be
    /// admitted at all. It costs a submit-and-wait per mixture layer, so it is
    /// for when the alternative is not running.
    ///
    /// `None` and not 0 for "unset", the way the CUDA driver spells
    /// `expert_cache`: the C++ side already reads an absent key as "keep the
    /// bank resident", so a sentinel would be a second spelling of one thing.
    ///
    /// The C++ has read `[model].expert_slab_bytes` since the slab landed;
    /// what was missing was any way for an operator to say it, which made the
    /// one feature that admits an oversized model reachable only from a test
    /// binary's environment variable.
    pub expert_slab_bytes: Option<u64>,
    /// Metal device string, e.g. `"metal:0"`. Populated from
    /// `model.driver.device` rather than written here.
    #[serde(skip)]
    pub device: String,
    /// Driver-side verbose logging. Populated from `server.verbose` rather
    /// than written here.
    #[serde(skip)]
    pub verbose: bool,
    /// How long to wait for the driver's caps handshake before giving up.
    /// Generous because it covers loading the weights.
    pub ready_timeout: Duration,
    /// How long to wait for the driver to drain before abandoning it.
    pub shutdown_timeout: Duration,
}

impl Default for MetalDriverOptions {
    fn default() -> Self {
        Self {
            kv_page_size: 32,
            total_pages: 1024,
            max_forward_tokens: 10240,
            max_forward_requests: 512,
            cpu_pages: 0,
            max_model_len: None,
            kv_cache_dtype: "auto".to_string(),
            stream_routed_experts: false,
            expert_slab_bytes: None,
            device: "metal:0".to_string(),
            verbose: false,
            ready_timeout: Duration::from_secs(120),
            shutdown_timeout: Duration::from_secs(5),
        }
    }
}

/// `[model.driver.options]` for `type = "dummy"`. The dummy driver
/// fabricates everything a real driver would otherwise read from
/// model weights — `vocab_size` and `arch_name` are required because no
/// safe default exists. Page geometry and timeouts have generic defaults;
/// the driver derives its synthetic KV page pool from these limits.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DummyDriverOptions {
    /// Vocabulary size advertised in the caps handshake. Should match
    /// the tokenizer the artifact carries. When `None` the
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

    /// How long to wait for the caps handshake. Short, because the dummy
    /// driver fabricates its answer rather than loading anything.
    #[serde(default = "default_dummy_ready_timeout")]
    pub ready_timeout: Duration,
}

fn default_dummy_ready_timeout() -> Duration {
    Duration::from_secs(5)
}

/// `[model.driver.options]` for `type = "cuda_native"`.
/// Mirrors `pie/src/pie_driver_cuda_native/config.py::CudaNativeDriverConfig`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaNativeDriverOptions {
    /// Fraction of each GPU's memory pie may use, weights included.
    ///
    /// What is left after the weights becomes the KV pool, so this is the
    /// knob that sizes it -- `max_total_pages` only caps the result.
    pub gpu_mem_utilization: f64,
    /// Which serving shape the memory planner optimizes its layout for:
    /// `auto` to infer it, `latency` for few concurrent requests, or
    /// `throughput` for many.
    pub memory_profile: CudaMemoryProfile,
    /// KV page size in tokens. **Omit to let the driver's memory planner
    /// derive one** by scoring candidates against the serving profile, which
    /// is what every deployment has been getting: this field reached the
    /// driver but the planner never read it, so only the (now deleted)
    /// `PIE_CUDA_KV_PAGE_SIZE` could pin it. Setting it pins it, and the
    /// planner searches a single-candidate lattice.
    pub kv_page_size: Option<u32>,
    /// Dtype KV pages are stored in. `"auto"` follows the activation dtype;
    /// a narrower one buys pages at some accuracy.
    pub kv_cache_dtype: String,
    /// Host-memory KV pages to swap into, reaching the driver as its
    /// `cpu_pages`. `0` disables swapping.
    pub swap_pool_size: u32,
    /// HARD cap on the runtime KV page count. **Omit to derive it from
    /// `gpu_mem_utilization`.** Setting it forces a tiny deterministic pool
    /// for contention/preempt tests + CI, independent of the forward-layout
    /// floor.
    ///
    /// Named for what it is rather than for what Metal calls its own field.
    /// Both were `total_pages` and they are not the same quantity: there the
    /// value IS the pool and 1024 is a real default; here it is a ceiling over
    /// a number derived from `gpu_mem_utilization`, and its absence is the
    /// normal case. One name for two meanings is a question a reader cannot
    /// answer from the file.
    pub max_total_pages: Option<u32>,
    /// HARD pin on the prefill token budget the forward step is built for.
    ///
    /// **Omit to let the memory planner choose.** Setting it collapses that
    /// axis of the planner's lattice to a single candidate, the same way
    /// `kv_page_size` does — which is the point: a value measured on this
    /// machine beats one scored by a model of it.
    ///
    /// `pie config tune` and `calibrate_planner` are how you get a number
    /// worth pinning. A guess here is worse than absence, because absence still
    /// gets the planner's judgement.
    ///
    /// Named for the same quantity Metal calls `max_forward_tokens`, because
    /// here they ARE the same quantity — unlike `total_pages`, where one name
    /// covered two different things and this driver's field had to be renamed
    /// `max_total_pages` to break the collision.
    pub max_forward_tokens: Option<u32>,
    /// HARD pin on the decode width — how many requests one forward step may
    /// carry. **Omit to let the memory planner choose.**
    ///
    /// Pinning this moves `[server] max_concurrent_processes` underneath you
    /// when that key is absent: admission derives from the driver's request
    /// cap, so the two are one decision. Set both, or neither.
    pub max_forward_requests: Option<u32>,
    /// Measure the forward step on this boot instead of scoring it.
    ///
    /// The memory planner picks `max_forward_tokens` by scoring a candidate
    /// lattice with an analytic model, and where that model disagreed with
    /// reality the driver grew per-(model, GPU) special cases carrying
    /// hand-measured constants. Setting this times the real forward body across
    /// the budget ladder on THIS device and caches the winner, so the next boot
    /// selects by evidence instead.
    ///
    /// **Not settable, and not a setting.** `pie config tune` turns it on
    /// for the one boot it runs and never writes it anywhere; see below for why
    /// it cannot be a key.
    ///
    /// It arrived as `PIE_CUDA_PLANNER_CALIBRATE`, and when the flag-deletion
    /// rule removed that, it came back here — the choice was framed as "config
    /// or environment variable", and both are wrong in the same way. This is not
    /// a description of a deployment; it is one run of a measurement. Written
    /// down, it needs the operator to perform a three-step ritual ("turn it on,
    /// boot once, turn it off"), and forgetting the third step is not a
    /// hypothetical failure: the planner abandons its score on a calibration
    /// boot and builds the LARGEST forward shape it can fit, accepting the
    /// starved KV pool that leaves, on the stated reasoning that such a boot
    /// serves nothing. Nothing made that true. So the ritual is gone and the
    /// third step cannot be forgotten, because there is no step to forget.
    ///
    /// `#[serde(skip)]` for the same reason `device` and `verbose` below are:
    /// pie populates it, a config file does not. It also means the struct's
    /// `deny_unknown_fields` refuses a hand-written `calibrate_planner = true`
    /// rather than honouring it.
    ///
    /// Refused for `tensor_parallel_size > 1` and for recurrent-state models,
    /// with a reason on stderr — see `batch/planner_calibration.hpp`.
    #[serde(skip)]
    pub calibrate_planner: bool,
    /// Dtype weights are materialized in. Separate from `activation_dtype`:
    /// narrower weights and wider compute is a normal combination.
    pub weight_dtype: String,
    /// CUDA device string, e.g. `"cuda:0"`. Populated by the caller
    /// from `model.driver.device`; set on the C++ side via
    /// `cudaSetDevice` (see `driver/cuda/src/engine.cpp`).
    #[serde(skip)]
    pub device: String,
    /// Driver-side verbose logging. Populated from `server.verbose` rather
    /// than written here.
    #[serde(skip)]
    pub verbose: bool,
    /// Runtime quantization mode applied during CUDA layout-plan
    /// materialization. Empty = none; `"fp8"` and `"int8"` enable
    /// per-channel symmetric quantization for supported projection weights.
    pub runtime_quant: String,
    /// GPT-OSS MXFP4 MoE policy. `"auto"` selects native packed MXFP4 GEMM
    /// on supported Blackwell-class GPUs/builds and routed dequant on legacy
    /// GPUs; `"routed_dequant"`/`"packed"` force the packed-weight
    /// BF16-scratch fallback; `"bf16"`/`"dequant"` eagerly materialize BF16
    /// experts; `"native"` requires true MXFP4 GEMM kernels.
    pub mxfp4_moe: String,
    /// Gemma-4 native MTP assistant checkpoint used by
    /// `.system_speculation()` on cuda_native. **Omit to let the CUDA driver
    /// auto-discover** the paired `-assistant` checkpoint from the Hugging
    /// Face cache when available.
    pub mtp_assistant_snapshot_dir: Option<String>,
    /// Maximum number of MTP draft tokens returned per system-spec step.
    pub mtp_num_drafts: u32,
    /// Page routed MoE experts through a bounded VRAM slab instead of keeping
    /// every expert resident. Bounds the resident set by the slab rather than
    /// by the model, which is what lets a large MoE run on a GPU that cannot
    /// hold it. Off by default: for a model that fits, this is strictly
    /// slower, and it disables CUDA graph capture besides.
    pub stream_routed_experts: bool,
    /// The expert slab, in GiB. **Omit to derive one** at startup from what
    /// is left after the resident weights and the KV pool. Ignored unless
    /// `stream_routed_experts` is set.
    pub expert_cache: Option<ByteSize>,
    /// A pinned host DRAM tier behind the slab, in GiB. **Omit for none.**
    ///
    /// The slab bounds what the GPU holds; this bounds what host memory holds
    /// behind it, in the same slot-shaped form. A miss the tier can serve is
    /// one host-to-device copy of bytes already in the form the kernels read,
    /// instead of a checkpoint read, a plan and a transform -- so it is worth
    /// setting exactly when the experts do not fit in VRAM but do fit in RAM.
    /// Ignored unless `stream_routed_experts` is set.
    pub expert_host_cache: Option<ByteSize>,
    /// Operator opt-in for system speculation (MTP). Default false: the runtime
    /// drives the auto-drafter only when this is true. Speculation is a
    /// latency-regime win (helps at low batch, costs at compute saturation), so
    /// it's off unless explicitly enabled — matching vLLM/SGLang convention.
    pub enable_system_speculation: bool,

    /// How long to wait for the driver's caps handshake before giving up.
    /// Generous because it covers loading and laying out the weights.
    pub ready_timeout: Duration,
    /// How long to wait for the driver to drain before abandoning it.
    pub shutdown_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
/// Which serving objective the driver's memory planner optimises for.
///
/// The planner searches a `(kv_page_size x decode_target x prefill_target)`
/// lattice and scores each candidate. This picks the objective, and it exists
/// because the planner **cannot infer it**: every input to that score is a
/// static fact -- SM count, model shape, KV headroom, arena pressure -- and
/// none of them say whether the traffic is interactive or batch.
///
/// `Balanced` and `Capacity` used to be spellable here and are not any more:
///
/// - `Balanced` is what `Auto` picks when the middle wins. Naming it added a
///   way to pin a choice the search already makes.
/// - `Capacity` was never a policy family. It resolved to *exactly* `Latency`'s
///   decode and prefill targets and differed only in KV page size (32 rather
///   than 16). That is a page-size choice wearing a policy's name, and since
///   `kv_page_size` is now honoured it is sayable directly:
///   `profile = "latency"` with `kv_page_size = 32`.
///
/// Both remain *internal* policy families that `Auto` still evaluates, so the
/// default deployment's search is unchanged.
pub enum CudaMemoryProfile {
    #[default]
    Auto,
    Latency,
    Throughput,
}

impl Default for CudaNativeDriverOptions {
    fn default() -> Self {
        Self {
            gpu_mem_utilization: 0.90,
            memory_profile: CudaMemoryProfile::Auto,
            kv_page_size: None,
            kv_cache_dtype: "auto".to_string(),
            swap_pool_size: 0,
            max_total_pages: None,
            max_forward_tokens: None,
            max_forward_requests: None,
            calibrate_planner: false,
            weight_dtype: "bfloat16".to_string(),
            device: String::new(),
            verbose: false,
            runtime_quant: String::new(),
            mxfp4_moe: "auto".to_string(),
            mtp_assistant_snapshot_dir: None,
            mtp_num_drafts: 3,
            stream_routed_experts: false,
            expert_cache: None,
            expert_host_cache: None,
            enable_system_speculation: false,
            ready_timeout: Duration::from_secs(600),
            shutdown_timeout: Duration::from_secs(5),
        }
    }
}

impl CudaNativeDriverOptions {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "driver.gpu_mem_utilization must be finite and in (0.0, 1.0]"
        );
        const MXFP4: &[&str] = &[
            "auto",
            "routed_dequant",
            "packed",
            "bf16",
            "dequant",
            "eager_bf16",
            "native",
        ];
        ensure!(
            self.mxfp4_moe.is_empty() || MXFP4.contains(&self.mxfp4_moe.as_str()),
            "driver.mxfp4_moe must be one of {:?}",
            MXFP4
        );
        ensure!(
            self.mtp_num_drafts <= 32,
            "driver.mtp_num_drafts must be in 0..=32"
        );
        // Present means the operator chose a size, so a present zero is a
        // contradiction rather than a way to say "derive" -- that is what
        // omitting the key is for.
        if let Some(size) = self.expert_cache {
            ensure!(
                size.as_bytes() > 0,
                "driver.expert_cache must be > 0; \
                 omit it to derive one at startup"
            );
        }
        if let Some(size) = self.expert_host_cache {
            ensure!(
                size.as_bytes() > 0,
                "driver.expert_host_cache must be > 0; \
                 omit it for no host tier"
            );
        }
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "driver.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        if let Some(size) = self.kv_page_size {
            ensure!(
                size > 0,
                "driver.kv_page_size must be > 0; \
                 omit it to let the memory planner derive one"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL_METAL: &str = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "metal"
device = ["cpu"]
"#;

    #[test]
    fn units_round_trip_through_their_string_form() {
        assert_eq!(parse_duration("50ms").unwrap(), Duration::from_millis(50));
        assert_eq!(parse_duration("120s").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_duration("2m").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_byte_size("4GiB").unwrap().as_bytes(), 4 << 30);
        assert_eq!(parse_byte_size("256MiB").unwrap().as_mib(), 256);
    }

    #[test]
    fn a_bare_number_is_refused_rather_than_assumed() {
        // Assuming a unit is how `_us` and `_secs` came to live in one table.
        let err = parse_duration("120").unwrap_err();
        assert!(err.contains("has no unit"), "got: {err}");
        let err = parse_byte_size("256").unwrap_err();
        assert!(err.contains("has no unit"), "got: {err}");
    }

    #[test]
    fn decimal_units_are_refused_for_sizes() {
        // The `_mb`/`_gb` fields this replaces always meant MiB/GiB -- each
        // multiplied by 1024*1024, never 1000*1000. Accepting "MB" would let a
        // config mean 5% less than it says.
        let err = parse_byte_size("256MB").unwrap_err();
        assert!(err.contains("binary units only"), "got: {err}");
    }

    #[test]
    fn rejects_the_legacy_unit_suffixed_names() {
        // Renamed, not aliased: deny_unknown_fields turns an old config into a
        // clear error naming the key, which is how this repo has handled every
        // other config rename (see the rejects_legacy_* tests below).
        for legacy in [
            "wasm_max_memory_mb = 4096",
            "max_upload_mb = 256",
            "wasm_warm_memory_mb = 0",
        ] {
            let toml = format!("{MINIMAL_METAL}\n[runtime]\n{legacy}\n");
            assert!(
                toml::from_str::<Config>(&toml).is_err(),
                "{legacy} should no longer parse"
            );
        }
        for legacy in [
            "request_timeout_secs = 120",
            "submit_deadline_us = 50000",
            "silence_timeout_secs = 30",
        ] {
            let toml = format!("{MINIMAL_METAL}\n[model.scheduler]\n{legacy}\n");
            assert!(
                toml::from_str::<Config>(&toml).is_err(),
                "{legacy} should no longer parse"
            );
        }
    }

    #[test]
    fn scheduler_durations_parse_from_their_units() {
        let toml = format!(
            "{MINIMAL_METAL}\n[model.scheduler]\n\
             request_timeout = \"90s\"\nsubmit_deadline = \"25ms\"\n\
             silence_timeout = \"1m\"\n"
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.scheduler.request_timeout.as_secs(), 90);
        assert_eq!(cfg.model.scheduler.submit_deadline.as_micros(), 25_000);
        assert_eq!(cfg.model.scheduler.silence_timeout.as_secs(), 60);
    }

    #[test]
    fn a_silence_timeout_under_the_submit_deadline_is_refused() {
        // The comparison is now between two Durations rather than between a
        // seconds count and a microseconds count.
        let toml = format!(
            "{MINIMAL_METAL}\n[model.scheduler]\n\
             submit_deadline = \"5s\"\nsilence_timeout = \"1s\"\n"
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("must not be shorter than submit_deadline"), "got: {err}");
    }

    #[test]
    fn parses_minimal_metal_config() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.driver.kind, DriverKind::Metal);
        assert_eq!(cfg.model.driver.device, vec!["cpu".to_string()]);
        assert_eq!(cfg.server.port, 8080);
    }

    /// `model` is the name now, `hf_repo` still parses.
    ///
    /// Not politeness: the struct is `deny_unknown_fields` and the field is
    /// required, so without the alias every config `pie config init` ever wrote
    /// would fail at startup with two errors at once — *unknown field
    /// `hf_repo`* and *missing field `model`* — instead of booting.
    #[test]
    fn the_old_spelling_of_the_model_key_still_parses() {
        let with_new = MINIMAL_METAL.replace("hf_repo =", "model =");
        assert_ne!(
            with_new, MINIMAL_METAL,
            "the fixture should use the old key"
        );

        let old: Config = toml::from_str(MINIMAL_METAL).unwrap();
        let new: Config = toml::from_str(&with_new).unwrap();
        old.validate().unwrap();
        new.validate().unwrap();
        assert_eq!(old.model.model, new.model.model);
        assert!(!new.model.model.is_empty());
    }

    /// An empty model is caught at parse rather than at driver boot, where it
    /// would surface as a path error.
    #[test]
    fn an_empty_model_is_refused() {
        let blank = MINIMAL_METAL.replace("hf_repo = \"Qwen/Qwen3-0.6B\"", "hf_repo = \"\"");
        let cfg: Config = toml::from_str(&blank).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("model.model"), "{err}");
    }

    #[test]
    fn weight_cache_dir_defaults_to_derived() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        // Empty means "derive $PIE_HOME/cache/weights" at the worker layer,
        // not "off":
        // the driver only sees off when it is handed nothing at all.
        assert_eq!(cfg.model.weight_cache_dir, "");
    }

    #[test]
    fn weight_cache_dir_parses_when_set() {
        let toml = MINIMAL_METAL.replace(
            "hf_repo = \"Qwen/Qwen3-0.6B\"",
            "hf_repo = \"Qwen/Qwen3-0.6B\"\nweight_cache_dir = \"/mnt/big/pie-models\"",
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.weight_cache_dir, "/mnt/big/pie-models");
    }

    #[test]
    fn rejects_a_relative_weight_cache_dir() {
        // Relative resolves against the driver's cwd, which the operator did
        // not choose and which differs between the worker and a spawned driver.
        let toml = MINIMAL_METAL.replace(
            "hf_repo = \"Qwen/Qwen3-0.6B\"",
            "hf_repo = \"Qwen/Qwen3-0.6B\"\nweight_cache_dir = \"models\"",
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("model.weight_cache_dir must be an absolute path"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_a_cache_section() {
        // [cache] existed briefly and was withdrawn: four of its five fields
        // were conventions or non-choices. deny_unknown_fields makes a config
        // written against it fail loudly rather than be silently ignored.
        let toml = format!("{MINIMAL_METAL}\n[cache]\nptir_dir = \"/tmp/x\"\n");
        assert!(toml::from_str::<Config>(&toml).is_err());
    }

    #[test]
    fn scheduler_frame_defaults_are_the_shipped_deployment() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.scheduler.frame_size, 2);
        assert_eq!(cfg.model.scheduler.frame_submit_depth, 3);
        assert_eq!(cfg.model.scheduler.frame_dispatch_depth, 2);
    }

    #[test]
    fn rejects_a_run_ahead_window_that_leaves_nothing_queued() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        // One frame submitted is the running frame alone — no run-ahead at
        // all, so the host round trip returns to the critical path.
        cfg.model.scheduler.frame_submit_depth = 1;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("scheduler.frame_submit_depth must be at least 2")
        );

        cfg.model.scheduler.frame_submit_depth = 2;
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_more_steps_in_flight_than_the_driver_stages() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        let steps = |c: &Config| {
            c.model.scheduler.frame_dispatch_depth * c.model.scheduler.frame_size
        };

        // The pool must strictly exceed steps in flight; equality is the case
        // that blocks in `cudaEventSynchronize` on every submit.
        cfg.model.scheduler.frame_dispatch_depth = 13;
        cfg.model.scheduler.frame_size = 1;
        assert_eq!(steps(&cfg), UPLOAD_STAGING_DEPTH);
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("must be < 13")
        );

        cfg.model.scheduler.frame_dispatch_depth = 12;
        assert_eq!(steps(&cfg), UPLOAD_STAGING_DEPTH - 1);
        cfg.validate().unwrap();

        // The product is what matters, not either factor: 3 x 4 and 4 x 3 are
        // the same 12 steps to the staging pool, and both are legal even
        // though each exceeds the other field's historical clamp.
        cfg.model.scheduler.frame_dispatch_depth = 3;
        cfg.model.scheduler.frame_size = 4;
        cfg.validate().unwrap();
        cfg.model.scheduler.frame_dispatch_depth = 4;
        cfg.model.scheduler.frame_size = 3;
        cfg.validate().unwrap();

        // ...and 2 x 7 = 14 is refused even though each factor looks modest.
        cfg.model.scheduler.frame_dispatch_depth = 2;
        cfg.model.scheduler.frame_size = 7;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_frame_geometry() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.model.scheduler.frame_size = 0;
        assert!(cfg.validate().is_err());

        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.model.scheduler.frame_dispatch_depth = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn executor_roles_require_controller() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.cluster.role = Some(Role::Prefill);
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("require a controller")
        );

        cfg.cluster.controller = Some("127.0.0.1:7000".to_string());
        cfg.validate().unwrap();
    }

    #[test]
    fn rejects_legacy_metal_kv_page_knob() {
        let stale = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "metal"
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
[model]
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
    fn rejects_public_driver_capacity_knobs() {
        for (ty, key) in [
            ("dummy", "max_forward_tokens"),
            ("dummy", "max_forward_requests"),
            ("dummy", "max_model_len"),
        ] {
            let mut options = String::new();
            if ty == "dummy" {
                options.push_str("vocab_size = 151936\narch_name = \"qwen3\"\n");
            }
            options.push_str(&format!("{key} = 1\n"));
            let text = format!(
                "[model]\nname = \"m\"\nhf_repo = \"x\"\n[model.driver]\n\
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
    fn device_string_or_list() {
        let one = r#"
[model]
name = "m"
hf_repo = "x"
[model.driver]
type = "metal"
device = "cuda:0"
"#;
        let cfg: Config = toml::from_str(one).unwrap();
        assert_eq!(cfg.model.driver.device, vec!["cuda:0".to_string()]);
    }

    #[test]
    fn every_config_field_carries_a_doc_comment() {
        // `pie config list` prints each key's first doc line, so a field with
        // no doc comment is a key the CLI cannot explain -- and the reason
        // `configuration.mdx` drifted is that its descriptions lived away from
        // the fields they described. Reading the source is crude but it reads
        // the definition itself, which is the only copy that cannot be stale.
        let source = include_str!("config.rs");
        let lines: Vec<&str> = source.lines().collect();
        let mut current = "";
        let mut undocumented = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            if let Some(rest) = line.strip_prefix("pub struct ") {
                current = rest
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or("");
            }
            let Some(rest) = line.strip_prefix("    pub ") else {
                continue;
            };
            // A field is an identifier followed immediately by `:`. Without
            // the second half this also matches `pub const fn from_secs(s:`
            // in the newtypes' impl blocks.
            let field: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if field.is_empty() || !rest[field.len()..].starts_with(':') {
                continue;
            }
            // Walk back over any attributes to reach what precedes the field.
            let mut j = i;
            while j > 0 && lines[j - 1].trim_start().starts_with("#[") {
                j -= 1;
            }
            if j == 0 || !lines[j - 1].trim_start().starts_with("///") {
                undocumented.push(format!("{current}.{field}"));
            }
        }
        assert!(
            undocumented.is_empty(),
            "config fields with no doc comment: {undocumented:?}"
        );
    }

    #[test]
    fn rejects_the_cuda_total_pages_spelling() {
        // Metal's `total_pages` IS the pool; CUDA's was a ceiling over a
        // derived number. Two meanings behind one name is a question a reader
        // cannot answer from the file, so the CUDA one became
        // `max_total_pages` -- and an existing config carrying the old
        // spelling must say so rather than silently losing its cap.
        let legacy = r#"
[model]
name = "a"
hf_repo = "x"
[driver]
type = "cuda_native"
device = ["cuda:0"]
total_pages = 512
"#;
        let err = Config::parse(legacy).unwrap_err().to_string();
        assert!(err.contains("total_pages"), "got: {err}");
        // The rejection lists what IS accepted, which is how the new name is
        // discoverable without reading this test.
        assert!(err.contains("max_total_pages"), "got: {err}");
    }

    #[test]
    fn rejects_legacy_binary_path() {
        // Both driver option structs carried it "for compatibility with the
        // Python wrapper", and nothing anywhere read it -- the drivers are
        // linked in, so there has never been an executable to point at.
        let legacy = r#"
[model]
name = "a"
hf_repo = "x"
[driver]
type = "metal"
device = ["metal:0"]
binary_path = "/opt/pie/driver"
"#;
        let err = Config::parse(legacy).unwrap_err().to_string();
        assert!(err.contains("binary_path"), "got: {err}");
    }

    #[test]
    fn rejects_legacy_auth_section() {
        // `[auth]` is gone because nothing behind it authenticated anything:
        // no code ever read `authorized_users.toml`, the gateway states that
        // the edge has already authed and it does not re-verify, and the
        // engine answered AuthProve with "Already authenticated" without
        // looking at the signature. `enabled = true` -- the old default --
        // therefore reported a protection that did not exist.
        //
        // An existing config carrying the section must fail loudly rather than
        // be ignored, because someone who wrote `enabled = true` deserves to
        // find out it never meant anything.
        let legacy = r#"
[auth]
enabled = true

[model]
name = "a"
hf_repo = "x"
[driver]
type = "metal"
device = ["metal:0"]
"#;
        let err = Config::parse(legacy).unwrap_err().to_string();
        assert!(err.contains("auth"), "got: {err}");
    }

    #[test]
    fn rejects_legacy_multi_model_list() {
        // The old `[[model]]` array form is gone: pie serves exactly one
        // model. Parsing must fail with a comprehensible hint rather than a
        // raw serde type error.
        let legacy = r#"
[[model]]
name = "a"
hf_repo = "x"
[model.driver]
type = "metal"
device = ["cuda:0"]

[[model]]
name = "b"
hf_repo = "y"
[model.driver]
type = "metal"
device = ["cuda:1"]
"#;
        let err = Config::parse(legacy).unwrap_err().to_string();
        assert!(
            err.contains("exactly one") && err.contains("[[model]]"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_unknown_top_level_keys() {
        let bad = r#"
nonsense = true

[model]
name = "m"
hf_repo = "x"
[model.driver]
type = "metal"
device = ["cpu"]
"#;
        assert!(toml::from_str::<Config>(bad).is_err());
    }

    #[test]
    fn parses_cuda_native_config() {
        let cuda = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
gpu_mem_utilization = 0.90
memory_profile = "throughput"
runtime_quant = "fp8"
mxfp4_moe = "routed_dequant"
mtp_assistant_snapshot_dir = "/models/gemma4-mtp"
mtp_num_drafts = 6
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.driver.kind, DriverKind::CudaNative);
        let opts: CudaNativeDriverOptions = cfg.model.driver.options.clone().try_into().unwrap();
        assert_eq!(opts.gpu_mem_utilization, 0.90);
        assert_eq!(opts.memory_profile, CudaMemoryProfile::Throughput);
        assert_eq!(opts.runtime_quant, "fp8");
        assert_eq!(opts.mxfp4_moe, "routed_dequant");
        assert_eq!(opts.mtp_assistant_snapshot_dir.as_deref(), Some("/models/gemma4-mtp"));
        assert_eq!(opts.mtp_num_drafts, 6);
        assert_eq!(opts.weight_dtype, "bfloat16"); // default
        // Absent = derive: the planner scores candidates unless pinned.
        assert_eq!(opts.kv_page_size, None);
        assert_eq!(opts.kv_cache_dtype, "auto"); // default
    }

    #[test]
    fn kv_page_size_pins_the_planner_when_set() {
        // Before this landed the field reached the driver and the planner
        // ignored it: only PIE_CUDA_KV_PAGE_SIZE could pin a page size. A
        // non-zero value now means the operator has made the choice the
        // planner's candidate search exists to make.
        let toml = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
kv_page_size = 16
"#;
        let cfg: Config = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        let opts: CudaNativeDriverOptions = cfg.model.driver.options.clone().try_into().unwrap();
        assert_eq!(opts.kv_page_size, Some(16));
    }

    #[test]
    fn cuda_native_options_default_when_omitted() {
        let cuda = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().unwrap();
        let opts: CudaNativeDriverOptions = cfg.model.driver.options.clone().try_into().unwrap();
        assert_eq!(opts.swap_pool_size, 0);
        assert_eq!(opts.gpu_mem_utilization, 0.90);
        assert_eq!(opts.memory_profile, CudaMemoryProfile::Auto);
        assert_eq!(opts.mxfp4_moe, "auto");
        assert!(opts.mtp_assistant_snapshot_dir.is_none());
        assert_eq!(opts.mtp_num_drafts, 3);
        assert_eq!(opts.ready_timeout, Duration::from_secs(600));
        assert_eq!(opts.kv_cache_dtype, "auto");
    }

    #[test]
    fn rejects_invalid_embedded_kv_cache_dtype() {
        let bad = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
kv_cache_dtype = "turboquant"
"#;
        let cfg: Config = toml::from_str(bad).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("kv_cache_dtype"), "got: {err}");
        assert!(err.contains("fp8_e4m3"), "got: {err}");
        assert!(err.contains("nvfp4"), "got: {err}");
    }

    #[test]
    fn rejects_a_retired_memory_profile() {
        // "balanced" and "capacity" were spellable and are not any more. They
        // must fail loudly rather than be accepted and silently mean something
        // else -- a config written against the old set names a real intent.
        for retired in ["balanced", "capacity"] {
            let cuda = format!(
                r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
memory_profile = "{retired}"
"#
            );
            // Note where this fails: `[model.driver.options]` is a free-form
            // `toml::Table` in the schema, so the enum is only checked when
            // `validate()` converts it per driver kind -- not at parse.
            let cfg: Config = toml::from_str(&cuda).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(
                err.contains(retired) || err.contains("memory_profile"),
                "{retired} should be rejected; got: {err}"
            );
        }
    }

    #[test]
    fn rejects_invalid_cuda_memory_profile() {
        let cuda = r#"
[model]
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
[model]
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
    fn rejects_invalid_cuda_mxfp4_policy() {
        let cuda = r#"
[model]
name = "default"
hf_repo = "openai/gpt-oss-20b"

[model.driver]
type = "cuda_native"
device = ["cuda:0"]

[model.driver.options]
mxfp4_moe = "mystery"
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("mxfp4_moe"), "got: {err}");
    }

    #[test]
    fn rejects_options_for_wrong_embedded_driver_type() {
        let stale = r#"
[model]
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
}
