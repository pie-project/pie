//! Pie standalone server config — TOML schema mirror of `pie.config`.
//!
//! Same TOML the legacy Python server consumed. Embedded engines
//! ([`EngineKind::CudaNative`] / [`EngineKind::Metal`] / [`EngineKind::Wgpu`])
//! are dispatched in [`crate::runtime::start_runtime`] via
//! [`crate::preflight::resolve_flavor`].
//!
//! The Rust [`Config`] type below is the user-facing TOML schema; the
//! conversion to `::runtime::bootstrap::Config` (the runtime's own config)
//! happens in [`crate::translate`].

use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use controller_api::Role;
// **THE RUN-AHEAD DEPTHS COME FROM THE ENGINE CONTRACT'S OWN MODULE** (alto
// design §9, article 8). This crate used to hold `UPLOAD_STAGING_DEPTH = 13`,
// a literal transcribed out of a deleted C++ header, and bound the
// deployment's depths against it. The formula lives in one place now and this
// file reads it.
pub use engine::runahead::Runahead;
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
    /// Batching and timeout policy. Every field has a measured default; the
    /// frame knobs are part of the guest contract.
    #[serde(default)]
    pub runtime: RuntimeConfig,
    /// The box an inferlet runs in: what it may reach, and how big it may get.
    #[serde(default)]
    pub sandbox: SandboxConfig,
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
    /// the engine options still land in a `deny_unknown_fields` struct.
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
    /// `nixl` via RDMA, or `auto` to use NIXL where it is available. No
    /// shipped build hosts NIXL, so `nixl` refuses the boot and `auto` is
    /// `inline` -- see [`OffloadTransfer`].
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

/// How KV pages cross between workers.
///
/// `Nixl` IS SPELLABLE AND NO SHIPPED BUILD HOSTS IT. The RDMA path lives
/// behind `worker`'s `nixl` feature, which `pie` — the only member with a
/// binary — deliberately does not forward (`scripts/ci-gate-audit.py` names
/// the exclusion and the reason: `transport`'s NIXL engine is a stub). So in
/// every build a user can run, `offload.transfer = "nixl"` parses and then
/// fails the boot at `link::partner::PartnerLinkManager::new` with
/// *"offload.transfer=nixl requires feature \"nixl\""*. That refusal is the
/// current truth about this variant, not a missing feature line.
///
/// `Auto` takes NIXL where it is available, which today is nowhere, and so
/// costs nothing.
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
        .ok_or_else(|| format!("duration {t:?} has no unit; write one of 120s, 50ms, 2m"))?;
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
        if us.is_multiple_of(1_000_000) {
            s.serialize_str(&format!("{}s", us / 1_000_000))
        } else if us.is_multiple_of(1_000) {
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
        .ok_or_else(|| format!("size {t:?} has no unit; write one of 512B, 256MiB, 4GiB"))?;
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
        let text = if b.is_multiple_of(GIB) && b != 0 {
            format!("{}GiB", b / GIB)
        } else if b.is_multiple_of(MIB) && b != 0 {
            format!("{}MiB", b / MIB)
        } else if b.is_multiple_of(KIB) && b != 0 {
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
    /// Verbose server logging, and passed down to the embedded engine.
    /// Independent of `--log-level`, which sets the tracing filter.
    #[serde(default)]
    pub verbose: bool,
    /// Where `pie inferlet` downloads from, and where the engine fetches a
    /// program it is asked to run but does not have.
    #[serde(default = "default_registry")]
    pub registry: String,
    /// Tokio worker threads. Derived from the visible CPUs, capped at 64.
    ///
    /// The cap is measured, not arbitrary: past ~64 the runtime's own
    /// scheduling overhead adds variance without adding parallelism (on a
    /// 256-thread EPYC 7773X, +0.5% mean tok/s and about triple the stdev
    /// without it). Raise it only for heavy non-inference work in-process.
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    /// Largest blob a client may upload in one request.
    #[serde(default = "default_max_upload")]
    pub max_upload: ByteSize,
    /// Ask this boot to measure the memory planner instead of scoring it.
    ///
    /// Set by `pie config tune` on the config it derives in memory, never
    /// read from a file — `#[serde(skip)]`, so it cannot be written down and
    /// cannot outlive the boot that asked for it. See
    /// [`CudaNativeEngineOptions::calibrate_planner`] for why a measurement
    /// must not be a persisted setting.
    ///
    /// It rides here rather than in `[model.engine.options]` because that table
    /// is the file's, and this never comes from the file. Same route as
    /// `verbose`: a typed field the server applies to whatever engine options it
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
            worker_threads: default_worker_threads(),
            max_upload: default_max_upload(),
            calibrate_planner: false,
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
    // Cap at 64 — pie's scheduler produces enough concurrent host work
    // at high request concurrency. Beyond ~64 workers the runtime's
    // scheduling overhead (queue management, wake propagation) starts
    // adding variance without adding parallelism. Measured on AMD EPYC
    // 7773X (256 threads visible): tok/s mean +0.5%, stdev cut to ~1/3
    // by capping. Users with heavier non-inference work in the same
    // process can override via `[server] worker_threads = ...`.
    std::thread::available_parallelism()
        .map(|n| n.get().min(64))
        .unwrap_or(4)
}
fn default_max_upload() -> ByteSize {
    ByteSize::from_mib(256)
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
// [sandbox]
// -----------------------------------------------------------------------------

/// The box an inferlet runs in: its walls and its size.
///
/// The `wasm_` prefix the size knobs carried is gone. The section already says
/// which box these size, and a prefix on every key of a section is the same
/// dead weight `worker.` was on every key of the file.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SandboxConfig {
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
    /// Instances the wasmtime pooling allocator may hold. A ceiling on
    /// concurrent inferlets, reserved up front.
    #[serde(default = "default_max_instances")]
    pub max_instances: u32,
    /// Linear memory one instance may address.
    #[serde(default = "default_max_memory")]
    pub max_memory: ByteSize,
    /// Linear memory kept resident when an instance is returned to the pool,
    /// rather than decommitted. Trades RSS for a cheaper next start; `0B`
    /// keeps none.
    #[serde(default)]
    pub warm_memory: ByteSize,
    /// Unused pool slots kept warm rather than torn down.
    #[serde(default = "default_warm_slots")]
    pub warm_slots: u32,
    /// Apply the host-side snapshot optimization to Python components.
    ///
    /// On by default. It only affects bootstrap cost, so turning it off is a
    /// debugging step -- it changes which wasmtime linker variant is built.
    #[serde(default = "default_true")]
    pub python_snapshot: bool,
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
    pub model: String,
    /// **WHICH SKU OF THAT CHECKPOINT TO SERVE**, or omit to let the load
    /// identify one.
    ///
    /// Identification asks every import in the catalog and takes the FIRST
    /// whose contract builds and whose plan fits the checkpoint's shapes. That
    /// is the right answer whenever exactly one row fits — and there is one
    /// family where several do. A vision checkpoint holds a text trunk AND a
    /// tower, so it fits the text-only row and the vision row both, and the
    /// text row is deliberately first: a two-unit load stands the fold down
    /// and cost 14.9% of throughput when it was tried the other way round
    /// (`model::qwen_3`'s own measurement). So the default is the cheap row
    /// and an operator who wants the tower NAMES it.
    ///
    /// It is a `[model]` key rather than an engine option because it is a fact
    /// about the load, and it is a SKU string rather than a boolean because
    /// the id space is the catalog's own — `pie model list` prints it and
    /// `model::catalog()` is the only table that defines it.
    #[serde(default)]
    pub sku: Option<String>,
    /// Which backend runs the model, on what devices.
    pub engine: EngineConfig,
    /// Where this model's materialized-weight artifacts are kept between runs.
    ///
    /// Per-model rather than process-global because the artifact IS the model:
    /// it is that checkpoint's weights, already laid out for this engine, and
    /// none of it is shared with another model. Empty derives
    /// `$PIE_HOME/cache/weights` -- a cache, beside the engine's other ones.
    /// NOT `$PIE_HOME/models`, which is the `.zt` artifact store: an artifact
    /// is portable and costs a re-download to replace, while these are device
    /// bytes for one engine, one TP layout and one ABI version, rebuilt by a
    /// single cold load.
    ///
    /// The artifacts are the size of the weights -- tens to hundreds of GB --
    /// which is why this is a path pie cannot always pick for you. The engine
    /// declines a write it has no room for rather than filling the disk.
    #[serde(default)]
    pub weight_cache_dir: String,
    /// Dtype weights are materialized in. Separate from `activation_dtype`:
    /// narrower weights and wider compute is a normal combination.
    ///
    /// A model fact rather than an engine one -- it is what the checkpoint
    /// holds -- which is why it sits here and `activation_dtype` and
    /// `kv_cache_dtype`, the dtypes the engine computes and stores in, sit in
    /// `[engine]`.
    #[serde(default = "default_weight_dtype")]
    pub weight_dtype: String,
    /// **How many weight bytes this load may keep on the DEVICE**, written
    /// with its unit (`"18GiB"`). Omit for uncapped, which lands the whole
    /// weight table on the device and is what this deployment did before the
    /// key existed.
    ///
    /// **NOT `[engine] gpu_mem_utilization`, which is the OTHER budget.**
    /// That one is the fraction of the card pie's elastic physical pool -- KV
    /// pages and scratch -- may commit; this one is the ceiling on the
    /// model's own weight table (tier T0). One says how much of the model
    /// stays on the card, the other how much of the card is pie's. Alto
    /// design §7's unified accounting is where the two meet; until it lands
    /// they are stated separately and neither derives the other.
    ///
    /// A `[model]` key rather than an engine option because it is portable:
    /// the budget is a fact about the load, and every shell that streams
    /// weights reads the same two numbers. WHICH planes a shell can hold
    /// fewer of is the shell's own property, so a budget it cannot meet by
    /// holding less refuses the load by name with both numbers rather than
    /// silently holding more (`engine::Residency`).
    #[serde(default)]
    pub device_weight_budget: Option<ByteSize>,
    /// **How many weight bytes this load may keep in the PINNED HOST cache**
    /// (tier T1), written with its unit (`"64GiB"`). Omit for uncapped.
    ///
    /// T1 is the tier a device miss reads over UVA instead of stalling on the
    /// checkpoint. An engine that holds everything on the device keeps zero
    /// host-resident weight bytes and every host budget admits it; one that
    /// streams routed experts pins EVERY expert of every streamed bank -- the
    /// pinned copy is authoritative and the device slab is a cache over it --
    /// so its host demand is those banks whole, and a budget under that
    /// refuses like any other.
    #[serde(default)]
    pub host_weight_budget: Option<ByteSize>,
    /// **THE SECOND ROW AXIS'S CEILING** (alto multimodal §5.5): the most
    /// patch rows one fire may carry, over every image of every lane in it.
    ///
    /// **OMIT IT.** A vision SKU serves with zero configuration: the shell
    /// reads the loaded TEXT, and a plan that states a patch axis gets rungs
    /// at whole images from the patch lattice's floor up to a ceiling argued
    /// from the token rectangle. A plan that states none gets no ladder at
    /// all, which is what every text-only SKU has always had.
    ///
    /// **A PLAIN INTEGER AND NOT A [`ByteSize`]**, unlike the two weight
    /// budgets above it: those are byte counts an operator writes with a unit,
    /// this is a ROW COUNT like `max_forward_tokens` beside it, and giving it
    /// a unit would invite `"4KiB"` for a number that is not bytes.
    ///
    /// A `[model]` key rather than `[model.engine.options]` because it is
    /// portable — every shell that serves a tower states its ceilings the same
    /// way, which is debt 6's rule and the one `max_adapters` already follows.
    #[serde(default)]
    pub max_patches: Option<u32>,
    /// **THE PATCH AXIS'S LANE CEILING** (alto multimodal §5.5): the most
    /// IMAGES one fire may carry.
    ///
    /// Omit it, as above. Not the same number as `max_patches` and not derived
    /// from it by the engine's own doctrine — a lane may submit three images
    /// or none — but the DEFAULT is argued from it: as many images as the
    /// patch ceiling holds if every one of them is the smallest whole image.
    #[serde(default)]
    pub max_images: Option<u32>,
    /// **The correction class's two operator decisions** (alto design §8):
    /// how many adapter seats this deployment intends to use, and which
    /// adapters to write into them at boot.
    ///
    /// A `[model]` key rather than an engine option because it is portable:
    /// every shell that seats banks seats them the same way, and debt 6's
    /// complaint about `[model.engine.options]` was precisely that no
    /// configuration crossed backends. Absent means what it has always
    /// meant — zero seats, no adapters, and the correction op never launches.
    #[serde(default)]
    pub adapters: AdapterConfig,
}

fn default_weight_dtype() -> String {
    "bfloat16".to_string()
}

// -----------------------------------------------------------------------------
// [model.adapters]
// -----------------------------------------------------------------------------

/// What an operator states about LoRA adapters: a capacity and a roster.
///
/// **THE CAPACITY IS AN INTENT, NOT A POOL SIZE** (design §8, decision 17).
/// Every bank a plan carries is reserved at load whatever this number is —
/// the capacity is a SHAPE the model text declares — so `seats` states how
/// many the DEPLOYMENT intends to register, and `model_compiler::compile`
/// refuses a load whose intent is bigger than what the text seats. That
/// refusal is the reason to state it at all: an operator who asks for
/// sixteen seats from a model that declares eight finds out at boot, by
/// name, rather than at the ninth registration.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterConfig {
    /// How many adapter rows this deployment intends to use.
    ///
    /// Omit to derive it from [`registered`](AdapterConfig::registered): a
    /// roster of three adapters needs three seats and nobody should have to
    /// say so twice. State it to reserve room for adapters that arrive later.
    #[serde(default)]
    pub seats: Option<u32>,
    /// The adapters to write into those seats at boot, in the order given.
    ///
    /// Registration is a control-plane residency verb — once per adapter,
    /// never on the fire path (`engine::adapter`) — so boot is exactly
    /// where it belongs.
    #[serde(default)]
    pub registered: Vec<RegisteredAdapter>,
}

/// One adapter, as an operator names it.
///
/// **THE PLANES ARE RAW BYTES AND THE PADDING IS THE CALLER'S**, which is the
/// contract's own rule (`engine::adapter`): an adapter trained at rank 4
/// in a bank declared at rank 16 is submitted zero-padded, because `A`'s
/// unused ranks are trailing ROWS and `B`'s are a stride inside every row, and
/// a shell that padded a short plane's prefix would be right for one and wrong
/// for the other. So a file here is one bank's slot, exactly, in the bank's
/// declared dtype and layout — the same bytes
/// [`AdapterPlane::bytes`](engine::AdapterPlane) carries. A file of the
/// wrong length is refused by the engine, by name, with both numbers.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegisteredAdapter {
    /// Which row of every named bank this fills — the id a lane routes to.
    pub id: u32,
    /// Bank name (the plan's own `Param` spelling) -> the file holding one
    /// slot of it. A bank this map omits keeps what it held, which is what
    /// makes registering one site at a time expressible.
    #[serde(default)]
    pub planes: std::collections::BTreeMap<String, String>,
}

impl AdapterConfig {
    /// The capacity to bake against: what the operator stated, else what the
    /// roster needs.
    ///
    /// One number with one owner (article 8). A roster whose highest id is
    /// `n` needs `n + 1` seats, because ids are rows and rows are counted
    /// from zero.
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

    /// Nothing to seat and nothing to register — the shape every deployment
    /// had before this section existed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seats() == 0 && self.registered.is_empty()
    }

    /// The two things this layer can answer without a device.
    ///
    /// Everything else about an adapter — does the bank exist, is the slot
    /// this many bytes — is the engine's, and it refuses by name. What is
    /// answerable here is whether the roster fits the capacity the operator
    /// asked for, and whether a plane path resolves the same way from the
    /// worker and from an engine launched with a working directory the
    /// operator did not choose.
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
    /// **The two weight budgets, in the form the engine's load contract
    /// states them** (alto design §7) -- what `[model] device_weight_budget`
    /// and `[model] host_weight_budget` are for.
    ///
    /// Both absent is [`engine::Residency::uncapped`], which is exactly the
    /// behaviour this worker had before the keys existed: the whole weight
    /// table lands on the device at load and never moves. There is no mode
    /// enum on either side of this seam -- residency is two numbers, and two
    /// numbers name every tier mix including the ones nobody has built yet.
    #[must_use]
    pub fn residency(&self) -> engine::Residency {
        engine::Residency {
            device_weight_budget: self.device_weight_budget.map(|b| b.as_bytes()),
            host_weight_budget: self.host_weight_budget.map(|b| b.as_bytes()),
        }
    }

    /// **The second row axis's two ceilings, in the form the engine's load
    /// contract states them** (alto multimodal §5.5) — what `[model]
    /// max_patches` and `[model] max_images` are for.
    ///
    /// Both absent is the honest default and the common case: the shell reads
    /// the loaded text and derives a ladder when the plan states a patch axis,
    /// so a vision SKU serves with zero configuration and a text-only one
    /// keeps the literal `None` it has always had. There is no mode enum on
    /// either side of this seam — a ladder is two numbers and a floor, and the
    /// floor is the compiler's statute.
    #[must_use]
    pub fn patch_ceilings(&self) -> (Option<u32>, Option<u32>) {
        (self.max_patches, self.max_images)
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
        // Relative resolves against the engine's working directory, which the
        // operator did not choose and which differs between the worker and an
        // engine launched as its own process.
        ensure!(
            self.weight_cache_dir.is_empty() || Path::new(&self.weight_cache_dir).is_absolute(),
            "model.weight_cache_dir must be an absolute path (got {:?}); \
             leave it empty for $PIE_HOME/cache/weights",
            self.weight_cache_dir
        );
        // A ceiling of zero bytes admits no plan at all, so it is a typo
        // rather than a policy -- and it is refused HERE, by the key's own
        // name, instead of reaching the engine as an `Impossible` naming
        // plane byte counts the operator never wrote down.
        // **A CEILING OF ZERO ROWS ADMITS NO IMAGE**, so it is a typo rather
        // than a policy — and it is refused here, by the key's own name,
        // rather than reaching the compiler as a `Budget` refusal about a
        // rectangle the operator never wrote down. Omitting the key is how a
        // deployment says "derive one"; writing `0` is how it says nothing at
        // all, and the two must not be the same sentence.
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

// -----------------------------------------------------------------------------
// [runtime]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    /// How long a client request may run before the runtime gives up on it.
    ///
    /// The outermost of the three clocks here and the only one about the
    /// client rather than the pipeline: `submit_deadline` leashes a straggler
    /// and `silence_timeout` terminates an abandoned process, but this one
    /// bounds the answer a caller is waiting for.
    #[serde(default = "default_request_timeout")]
    pub request_timeout: Duration,
    /// How long a pipeline that is HARD-BLOCKING a frame's seal may go
    /// without submitting before the runtime stops waiting for it, in
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
    /// frame is not blocking), by an unretired dispatch (the runtime owes it a
    /// result, so the whole GPU wave is free), by a bind in flight, and by
    /// `forward.park()`. The host resubmit round trip already has its own
    /// headroom in the run-ahead window
    /// ([`engine::runahead::Runahead::submit_depth`], derived from
    /// [`RuntimeConfig::frame_dispatch_depth`]).
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
    /// on past it — before the runtime terminates its process, in seconds.
    ///
    /// This one IS a verdict, so it is generous. A pipeline that means to go
    /// quiet calls `forward.park()`, which ends the silence and is never
    /// killed however long it stays away; that is the contract this enforces.
    /// The leash already keeps a straggler from holding the fleet, so nothing
    /// but a genuinely abandoned process ever reaches this.
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout: Duration,
    /// Waves per frame (*k*): how many token steps the wait-all quorum admits
    /// before it runs. A deployment constant, fixed at runtime start exactly
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
    /// engine staging depth (CONTENTION_FOLLOWUP §20.8).
    ///
    /// Bounded above by the CUDA engine, not by taste — see [`Self::validate`].
    #[serde(default = "default_frame_size")]
    pub frame_size: u32,
    // `frame_submit_depth` STOOD HERE AND IS DERIVED NOW (alto E, survey
    // debt 8). It was "frames a guest keeps submitted into the runtime: one
    // running, plus the rest queued behind it", defaulting to 3 beside a
    // `frame_dispatch_depth` of 2 — and its own documentation had to explain
    // that the two were not independent. They are one number:
    // `engine::runahead::Runahead::submit_depth` is `frames_in_flight + 1`,
    // which answers 3 at the shipped default, and
    // `Runahead::channel_capacity(k)` is what a guest reads as
    // `model.channel-capacity()`. A deployment that wants a longer window
    // moves `frame_dispatch_depth`, which is the number the engine also
    // carves its staging ring from — one number, one owner (article 8).
    /// Frames the runtime keeps POSTED to the engine but not yet retired: the
    /// dispatch loop's enqueue horizon — the runtime running ahead of the
    /// device, and what keeps the GPU from idling between frames: at 2 one
    /// frame executes while the next is already uploaded. The guest's own
    /// window is this plus one
    /// ([`engine::runahead::Runahead::submit_depth`]), so this is the single
    /// knob behind both.
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
    /// Bounded jointly with `frame_size` by the engine — see [`Self::validate`].
    #[serde(default = "default_frame_dispatch_depth")]
    pub frame_dispatch_depth: u32,
    /// Hard cap on inferlets admitted at once. **Omit to derive it** from the
    /// engine's `max_forward_requests`, which is what fills a batch.
    ///
    /// A physical safety limit, not a scheduling knob: past the point where
    /// every batch is full, more concurrent processes add queueing and not
    /// throughput.
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
        // ── THE ENGINE COUPLING, DERIVED RATHER THAN TRANSCRIBED (alto F2b).
        //
        // What stood here was a product checked against `UPLOAD_STAGING_DEPTH
        // = 13` — a literal copied out of a C++ header this tree deleted,
        // whose derivation (`3 × 4 + 1`) existed nowhere in the source. The
        // formula is `engine::runahead::Runahead`'s now, the engine's staging
        // ring is CARVED from these two numbers rather than fixed at thirteen,
        // and so the check is no longer "does the product fit a constant" but
        // "are these two numbers ones the engine can size a ring from".
        //
        // Two bounds, one per factor, because they bound different things:
        // `frame_size` is `k` in the staging formula and `frame_dispatch_depth`
        // is the multiplier. The multiplier's ceiling is a fact about the
        // engine's free-slot word (`Runahead::MAX_FRAMES`, see its doc), and
        // `k`'s is the frame scheduler's own `STEPS_MAX`.
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

// -----------------------------------------------------------------------------
// [model.engine]
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)] // forwarded to the embedded engine via TOML; not all
// fields are read on the Rust side yet.
pub struct EngineConfig {
    /// Which engine hosts this model. `cuda_native` is the one this build
    /// runs; `metal`, `vulkan` and `wgpu` are accepted spellings that boot
    /// refuses by name until their engines return.
    #[serde(rename = "type")]
    pub kind: EngineKind,
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
    /// How many KV pages the engine opens its pool with. `None` is the
    /// backend's own default.
    ///
    /// # Why it is declared here and read somewhere else
    ///
    /// It is not used by this struct at all. `engine-vulkan`'s and
    /// `engine-wgpu`'s backends both parse it out of the RAW config bytes --
    /// neither has this type in scope -- and both document it in those words:
    /// "a number the boot config can override".
    ///
    /// It could not be overridden. The struct that carried the key is
    /// `deny_unknown_fields`, so the key those two backends read was rejected
    /// by the deserializer before either of them saw the bytes.
    ///
    /// So this field exists to be ACCEPTED. That is a real job under
    /// `deny_unknown_fields`, and the alternative -- dropping the denial --
    /// would silently swallow every typo in the table. Declaring the key that
    /// somebody else reads is the narrow version of the same permission.
    #[serde(default)]
    pub kv_pages: Option<u32>,
    /// How long to wait for the engine's caps handshake before giving up.
    /// Generous because it covers loading and laying out the weights.
    #[serde(default = "default_ready_timeout")]
    pub ready_timeout: Duration,
    /// How long to wait for the engine to drain before abandoning it.
    #[serde(default = "default_shutdown_timeout")]
    pub shutdown_timeout: Duration,
    /// Engine-specific knobs. Embedded engines parse this into typed
    /// option structs.
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
                        // `[engine]`, not `model.engine.options`: the operator
                        // wrote a file, and the file spells these as plain
                        // `[engine]` keys. The internal path is what
                        // `config_layout::reshape` produced from it, and
                        // naming it here sends the reader looking for a table
                        // their config does not contain.
                        anyhow::anyhow!(
                            "invalid [engine] options for engine type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                opts.validate()?;
                validate_kv_cache_dtype(&opts.kv_cache_dtype)?;
            }
            // NOTHING TO CHECK, because there is no schema left to check
            // against: the three option tables went with the engines that
            // read them. The kind itself is still refused, by name and
            // once, at `engine_ffi::Flavor::from_kind`.
            EngineKind::Metal | EngineKind::Vulkan | EngineKind::Wgpu => {}
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

/// Which engine a `[model.engine] type` names.
///
/// THE LAST THREE ARE NAMED, NOT OFFERED, and no build flag changes that.
/// Their engines left the workspace at R3, so there is no `--features
/// engine-metal|engine-vulkan|engine-wgpu` to rebuild with — the crates such
/// a feature would name are not members. The names stay so that a deployment
/// asking for one is told what happened, by `engine_ffi::retired_msg`, rather
/// than being told its config is malformed. They come back at P5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    /// Native CUDA engine — embedded as a static lib in `worker`
    /// (requires `--features engine-cuda-13`).
    CudaNative,
    /// Native MLX + Metal engine for Apple Silicon.
    Metal,
    /// Pure-Rust Vulkan engine: portable rather than vendor-specific, on
    /// whatever Vulkan 1.3 device the machine exposes.
    Vulkan,
    /// The WebGPU shell — one binary over Vulkan, Metal, D3D12 or WebGPU,
    /// whichever the machine has.
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
fn default_random_seed() -> u64 {
    42
}
fn default_ready_timeout() -> Duration {
    Duration::from_secs(600)
}
fn default_shutdown_timeout() -> Duration {
    Duration::from_secs(5)
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
// Engine-specific options (typed views over `EngineConfig::options`)
// -----------------------------------------------------------------------------

/// `[model.engine.options]` for `type = "cuda_native"`.
/// Mirrors `pie/src/pie_driver_cuda_native/config.py::CudaNativeDriverConfig`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct CudaNativeEngineOptions {
    /// `[model] id`: the operator's answer to "which model is this".
    ///
    /// Absent — the ordinary case — the engine identifies the checkpoint
    /// from its TENSORS against the catalog every engine links. Present,
    /// it names a row directly, for a checkpoint that is genuinely a
    /// known model under an unknown name: a fine-tune, a re-upload, a
    /// mirror that renamed the directory.
    ///
    /// It is an OVERRIDE and not a bypass. The named row's manifest is
    /// still matched, so this cannot be used to load a checkpoint as
    /// something it is not — which is the failure the whole arrangement
    /// exists to prevent.
    pub model_id: Option<String>,
    /// Fraction of each GPU's memory pie may use, weights included.
    ///
    /// What is left after the weights becomes the KV pool, so this is the
    /// knob that sizes it -- `max_total_pages` only caps the result.
    ///
    /// **AND IT REACHES AN ENGINE NOW** (alto streaming §3 item 5,
    /// `next.md` B1). This key was written into the boot document's
    /// `[batching]` table, which the C++ engine read and the Rust shell never
    /// did, so for four waves it was declared, defaulted, validated and
    /// schema'd here and read by nothing: the elastic pool took ~100% of
    /// whatever the card had free. It is an `[engine]` key on the wire now and
    /// `engine_cuda::device::elastic::budget_bytes` is the one arithmetic that
    /// reads it -- `total x utilization - (total - free) - floor`, which at
    /// `1.0` is byte for byte the pool this deployment had before.
    pub gpu_mem_utilization: f64,
    /// Which serving shape the memory planner optimizes its layout for:
    /// `auto` to infer it, `latency` for few concurrent requests, or
    /// `throughput` for many.
    pub memory_profile: CudaMemoryProfile,
    /// KV page size in tokens. **Omit to let the engine's memory planner
    /// derive one** by scoring candidates against the serving profile, which
    /// is what every deployment has been getting: this field reached the
    /// engine but the planner never read it, so only the (now deleted)
    /// `PIE_CUDA_KV_PAGE_SIZE` could pin it. Setting it pins it, and the
    /// planner searches a single-candidate lattice.
    pub kv_page_size: Option<u32>,
    /// Dtype KV pages are stored in. `"auto"` follows the activation dtype;
    /// a narrower one buys pages at some accuracy.
    pub kv_cache_dtype: String,
    /// Host-memory KV pages to swap into, reaching the engine as its
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
    /// covered two different things and this engine's field had to be renamed
    /// `max_total_pages` to break the collision.
    pub max_forward_tokens: Option<u32>,
    /// HARD pin on the decode width — how many requests one forward step may
    /// carry. **Omit to let the memory planner choose.**
    ///
    /// Pinning this moves `[runtime] max_concurrent_processes` underneath you
    /// when that key is absent: admission derives from the engine's request
    /// cap, so the two are one decision. Set both, or neither.
    pub max_forward_requests: Option<u32>,
    /// Measure the forward step on this boot instead of scoring it.
    ///
    /// The memory planner picks `max_forward_tokens` by scoring a candidate
    /// lattice with an analytic model, and where that model disagreed with
    /// reality the engine grew per-(model, GPU) special cases carrying
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
    /// CUDA device string, e.g. `"cuda:0"`. Populated by the caller
    /// from `model.engine.device`; set on the C++ side via
    /// `cudaSetDevice` (see `crates/engine-cuda/csrc/src/engine.cpp`).
    #[serde(skip)]
    pub device: String,
    /// Engine-side verbose logging. Populated from `server.verbose` rather
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
    /// `.system_speculation()` on cuda_native. **Omit to let the CUDA engine
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
    /// The expert slab, in GiB. **Omit to derive one** at bootstrap from what
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
    /// **HOW MUCH OF A FIRE THE SHELL RECORDS**: `"off"` (the golden eager
    /// path), `"shaped"` (eager, with graph-shaped padded schedules — the
    /// attribution arm) or `"on"` (bodies: captured at load, replayed after).
    /// **Omit for the shell's own default**, which is `"on"` — the other two
    /// are diagnostic arms and the shell prints a line when it is asked to
    /// serve one, because an uncaptured decode pays ~470 kernel launches of
    /// host time per token-step.
    ///
    /// Written into the boot document as `[engine] graphs`, which the runtime
    /// has read since the palo rewrite and which nothing wrote until now.
    pub graphs: Option<String>,
    /// **D4's PAD.** Before each walk the CUDA shell stamps the fire's rows and
    /// its bucket onto every stream context, and the entries that hand a shape
    /// to cuBLASLt round their `M` up to the bucket, so the library's
    /// unpublished arm table stops being a function of the batch. **Omit for
    /// on**, which is what every deployment has been getting.
    ///
    /// `false` is the A/B arm the tail-waste measurement needs. The tokens are
    /// byte-identical across it: everything the padding computes lands in rows
    /// no reader has.
    pub pad: Option<bool>,
    /// **BODIES**: under `graphs = "on"`, one exec per `(bucket, present
    /// set)`, captured at LOAD for every point of the lattice the deployment
    /// can realize and replayed across row counts off the staged-geometry
    /// seat — no host rebinding at all, and no capture on the serving path.
    /// **Omit for on**, which is the shipping graph path: since the tier-2
    /// campaign a body is the only recorded path there is, and the regions a
    /// graph cannot hold are re-issued eagerly between its execs rather than
    /// refusing the composition. `false` is the DIAGNOSTIC arm — graphs on,
    /// schedules graph-shaped, every fire walking — and it is what the bodies
    /// gate diffs against for token identity.
    ///
    /// Three keys stood beside this one and named the FOLD (`fold`,
    /// `pipeline`, `fold_disable`): one exec per BUCKET, rebound on the host
    /// per fire. The tier-2 campaign deleted the fold along with the keyed
    /// capture path — a body pays nothing per fire where the fold paid a
    /// restatement per present node — so the three are gone from this struct.
    /// They were `Option`s defaulting to absent, so a deployment that never
    /// stated them is unaffected; one that did is refused by name here
    /// (`deny_unknown_fields`) rather than silently ignored.
    pub bodies: Option<bool>,
    /// **`Fallback::Copy` WHERE THE COMPILER'S TABLE ASKS FOR ONE.** Below the
    /// copy/split crossover — every bucket a decode fire lands in — a copy
    /// measured 1.07x the ideal against a split's 1.82x. **Omit for on.**
    /// `false` is the A/B arm and the free oracle: a copy computes the same
    /// bytes over the same rows, and only a byte-for-byte diff against a split
    /// can settle that.
    pub fallback_copy: Option<bool>,
    /// **THE GROUPED ARM**: the ops whose kernels walk a segment list are named
    /// to the compiler, so a withdrawn consumer is served as ONE launch over
    /// that list instead of one per rectangle. **Omit for on.** `false` names
    /// none of them, which is the off arm of a Grouped-versus-Split
    /// measurement — the same row order, a different answer on it.
    pub grouped: Option<bool>,
    /// **HOW MANY SIDE STREAMS THE COMPILER MAY HAND OUT.** `0` bakes an
    /// artifact with no fork group, no event point and stream 0 on every
    /// region — the artifact this compiler produced before concurrency
    /// existed, which is the only honest off arm for a streams measurement.
    /// **Omit for the device profile's own figure.**
    pub side_streams: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
/// Which serving objective the engine's memory planner optimises for.
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

impl Default for CudaNativeEngineOptions {
    fn default() -> Self {
        Self {
            model_id: None,
            gpu_mem_utilization: 0.90,
            memory_profile: CudaMemoryProfile::Auto,
            kv_page_size: None,
            kv_cache_dtype: "auto".to_string(),
            swap_pool_size: 0,
            max_total_pages: None,
            max_forward_tokens: None,
            max_forward_requests: None,
            calibrate_planner: false,
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
            graphs: None,
            pad: None,
            bodies: None,
            fallback_copy: None,
            grouped: None,
            side_streams: None,
        }
    }
}

/// `[model.engine.options]` for `type = "metal"` (Apple Silicon MLX/Metal
/// engine) — page geometry and forward limits; the metal engine speaks the
/// embedded in-process ABI. `device` is the `metal:N` selector filled from
/// `model.engine.device`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MetalEngineOptions {
    /// `[model] id`: the operator's answer to "which model is this".
    ///
    /// Absent — the ordinary case — the engine identifies the checkpoint
    /// from its TENSORS against the catalog every engine links. Present,
    /// it names a row directly, for a checkpoint that is genuinely a
    /// known model under an unknown name: a fine-tune, a re-upload, a
    /// mirror that renamed the directory.
    ///
    /// It is an OVERRIDE and not a bypass. The named row's manifest is
    /// still matched, so this cannot be used to load a checkpoint as
    /// something it is not — which is the failure the whole arrangement
    /// exists to prevent.
    pub model_id: Option<String>,
    /// KV page size in tokens. Used as given -- unlike the CUDA engine, the
    /// Metal engine has no planner to derive one.
    pub kv_page_size: u32,
    /// KV pages to allocate. Used directly, and 1024 is a real default.
    ///
    /// The CUDA engine's nearest equivalent is `max_total_pages`, which is a
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
    /// default -- keeps the engine's own constant, which is what a `pie serve`
    /// fleet wants and what every run got before this existed.
    ///
    /// The one knob that shrinks the KV, and it only shrinks: the engine
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
    /// The same knob, spelled the same way, as the CUDA engine's -- because it
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
    /// `None` and not 0 for "unset", the way the CUDA engine spells
    /// `expert_cache`: the C++ side already reads an absent key as "keep the
    /// bank resident", so a sentinel would be a second spelling of one thing.
    ///
    /// The C++ has read `[model].expert_slab_bytes` since the slab landed;
    /// what was missing was any way for an operator to say it, which made the
    /// one feature that admits an oversized model reachable only from a test
    /// binary's environment variable.
    pub expert_slab_bytes: Option<u64>,
    /// Metal device string, e.g. `"metal:0"`. Populated from
    /// `model.engine.device` rather than written here.
    #[serde(skip)]
    pub device: String,
    /// Engine-side verbose logging. Populated from `server.verbose` rather
    /// than written here.
    #[serde(skip)]
    pub verbose: bool,
}

impl Default for MetalEngineOptions {
    fn default() -> Self {
        Self {
            model_id: None,
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
        }
    }
}

impl CudaNativeEngineOptions {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.gpu_mem_utilization.is_finite()
                && self.gpu_mem_utilization > 0.0
                && self.gpu_mem_utilization <= 1.0,
            "engine.gpu_mem_utilization must be finite and in (0.0, 1.0]"
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
            "engine.mxfp4_moe must be one of {:?}",
            MXFP4
        );
        ensure!(
            self.mtp_num_drafts <= 32,
            "engine.mtp_num_drafts must be in 0..=32"
        );
        // Present means the operator chose a size, so a present zero is a
        // contradiction rather than a way to say "derive" -- that is what
        // omitting the key is for.
        if let Some(size) = self.expert_cache {
            ensure!(
                size.as_bytes() > 0,
                "engine.expert_cache must be > 0; \
                 omit it to derive one at bootstrap"
            );
        }
        if let Some(size) = self.expert_host_cache {
            ensure!(
                size.as_bytes() > 0,
                "engine.expert_host_cache must be > 0; \
                 omit it for no host tier"
            );
        }
        if let Some(pages) = self.max_total_pages {
            ensure!(
                pages > 0,
                "engine.max_total_pages must be > 0; \
                 omit it to derive from gpu_mem_utilization"
            );
        }
        if let Some(size) = self.kv_page_size {
            ensure!(
                size > 0,
                "engine.kv_page_size must be > 0; \
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
model = "Qwen/Qwen3-0.6B"

[model.engine]
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

    #[test]
    fn scheduler_durations_parse_from_their_units() {
        let toml = format!(
            "{MINIMAL_METAL}\n[runtime]\n\
             request_timeout = \"90s\"\nsubmit_deadline = \"25ms\"\n\
             silence_timeout = \"1m\"\n"
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.runtime.request_timeout.as_secs(), 90);
        assert_eq!(cfg.runtime.submit_deadline.as_micros(), 25_000);
        assert_eq!(cfg.runtime.silence_timeout.as_secs(), 60);
    }

    #[test]
    fn a_silence_timeout_under_the_submit_deadline_is_refused() {
        // The comparison is now between two Durations rather than between a
        // seconds count and a microseconds count.
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

    #[test]
    fn parses_minimal_metal_config() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.engine.kind, EngineKind::Metal);
        assert_eq!(cfg.model.engine.device, vec!["cpu".to_string()]);
        assert_eq!(cfg.server.port, 8080);
    }

    /// Every engine kind's `as_str` is the word a config file spells it with.
    ///
    /// # What this is guarding
    ///
    /// The two spellings come from different places and nothing ties them
    /// together: `as_str` is a hand-written match, and what a file parses is
    /// serde's `rename_all = "snake_case"` over the variant name. They agree
    /// today for every kind and there is no compiler error if one stops --
    /// `config_schema::default_values` builds a config by INTERPOLATING
    /// `as_str` into `type = "..."` and parsing it back, so a kind whose two
    /// spellings differed would make the schema listing silently fall back to
    /// an empty table, and `pie config set` would offer no engine keys at all.
    ///
    /// Written as a walk over an explicit list rather than one assertion per
    /// kind, so that a kind added without a line here is a kind that fails
    /// this test's own count.
    #[test]
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
            // And back out through serde, which is the direction
            // `config_schema::default_values` relies on.
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

    // `the_wgpu_options_refuse_a_key_this_engine_does_not_read` STOOD HERE,
    // and `rejects_legacy_metal_kv_page_knob` and
    // `rejects_options_for_wrong_embedded_engine_type` below it. All three
    // asserted `deny_unknown_fields` on an option table that no longer
    // exists: only `cuda_native` names a struct now, and
    // `rejects_unknown_cuda_option` makes the same claim about the one that
    // does.

    // `the_old_spelling_of_the_model_key_still_parses` STOOD HERE. It proved
    // that the model key's retired spelling parsed to the same field as
    // `model`, by parsing the fixture both ways. The alias is gone, so there
    // is one spelling and no second parse to compare it against.

    /// An empty model is caught at parse rather than at engine boot, where it
    /// would surface as a path error.
    #[test]
    fn an_empty_model_is_refused() {
        let blank = MINIMAL_METAL.replace("model = \"Qwen/Qwen3-0.6B\"", "model = \"\"");
        let cfg: Config = toml::from_str(&blank).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("model.model"), "{err}");
    }

    /// The key two engines document as an override actually parses.
    ///
    /// `engine-vulkan` and `engine-wgpu` both read `kv_pages` out of the raw
    /// config bytes and both call it "a number the boot config can override".
    /// It was not one: the struct that carried it is `deny_unknown_fields`, so
    /// the deserializer refused the key before either backend saw the bytes,
    /// and the only way to set it was to not use this type. Nothing noticed
    /// because nothing here reads it -- the field is declared to be
    /// ACCEPTED, and a test is the only thing that can say so.
    ///
    /// It sits in `[engine]` now, with the rest of the KV geometry.
    #[test]
    fn the_kv_pages_override_two_engines_read_is_one_this_config_accepts() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(
            cfg.model.engine.kv_pages, None,
            "absent must stay absent, or every boot overrides a default it \
             never meant to"
        );

        let with = MINIMAL_METAL.replace("device = [\"cpu\"]", "device = [\"cpu\"]\nkv_pages = 8");
        assert_ne!(with, MINIMAL_METAL, "the fixture should carry the key");
        let cfg: Config = toml::from_str(&with)
            .expect("`[engine] kv_pages` is the key engine-vulkan and engine-wgpu read");
        cfg.validate().unwrap();
        assert_eq!(cfg.model.engine.kv_pages, Some(8));

        // ...and the denial is still doing its job for everything else.
        let typo = MINIMAL_METAL.replace("device = [\"cpu\"]", "device = [\"cpu\"]\nkv_page = 8");
        toml::from_str::<Config>(&typo)
            .expect_err("a near-miss of the key must still be refused by name");
    }

    #[test]
    fn weight_cache_dir_defaults_to_derived() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        // Empty means "derive $PIE_HOME/cache/weights" at the worker layer,
        // not "off":
        // the engine only sees off when it is handed nothing at all.
        assert_eq!(cfg.model.weight_cache_dir, "");
    }

    #[test]
    fn weight_cache_dir_parses_when_set() {
        let toml = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\nweight_cache_dir = \"/mnt/big/pie-models\"",
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.weight_cache_dir, "/mnt/big/pie-models");
    }

    /// **`[model.adapters]` IS ABSENT IN EVERY DEPLOYMENT THAT HAS ONE
    /// TODAY**, and absent means the shape that was hard-coded before it
    /// existed: zero seats, no roster, and the correction op never launches.
    #[test]
    fn adapters_are_absent_by_default_and_that_is_zero_seats() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.model.adapters.is_empty());
        assert_eq!(cfg.model.adapters.seats(), 0);
        assert!(cfg.model.adapters.registered.is_empty());
    }

    /// The capacity is stated once: either the operator says it, or the
    /// roster does (article 8 — one number, one owner).
    #[test]
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
        // Highest id 2 means three rows, because ids are rows counted from
        // zero.
        assert_eq!(cfg.model.adapters.seats(), 3);
        assert_eq!(cfg.model.adapters.registered.len(), 1);
        assert_eq!(
            cfg.model.adapters.registered[0].planes["layer.0.lora_a"],
            "/adapters/0/a.bin"
        );

        // Stated wins, and states room the roster does not need yet.
        let stated = toml.replace("[model.adapters]", "[model.adapters]\nseats = 8");
        let cfg: Config = toml::from_str(&stated).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.adapters.seats(), 8);
    }

    /// The two refusals this layer can make without a device.
    #[test]
    fn an_adapter_past_its_seats_or_on_a_relative_path_is_refused_by_name() {
        let past = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseats = 2\n\
             [[model.adapters.registered]]\n\
             id = 5\n",
        );
        let cfg: Config = toml::from_str(&past).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("adapter id 5 is past the 2 seat"), "got: {err}");

        let relative = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\n\
             [[model.adapters.registered]]\n\
             id = 0\n\
             planes = { \"layer.0.lora_a\" = \"a.bin\" }\n",
        );
        let cfg: Config = toml::from_str(&relative).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("must be \n             an absolute path") || err.contains("absolute path"), "got: {err}");

        // And a near-miss of a key is still refused, as everywhere else.
        let typo = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseat = 2\n",
        );
        toml::from_str::<Config>(&typo)
            .expect_err("a near-miss of `seats` must be refused by name");
    }

    #[test]
    fn rejects_a_relative_weight_cache_dir() {
        // Relative resolves against the engine's cwd, which the operator did
        // not choose and which differs between the worker and a spawned engine.
        let toml = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\nweight_cache_dir = \"models\"",
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
        assert_eq!(cfg.runtime.frame_size, 2);
        assert_eq!(cfg.runtime.frame_dispatch_depth, 2);
        // AND THE GUEST'S WINDOW IS THAT NUMBER'S ARITHMETIC (alto E). It was
        // `frame_submit_depth = 3`, configured beside this one; the shipped
        // deployment's answer is unchanged, and there is no second knob that
        // can be set to disagree with it.
        assert_eq!(
            Runahead::of(cfg.runtime.frame_dispatch_depth as u8).submit_depth(),
            3
        );
    }

    /// A run-ahead window that leaves the guest nothing queued is now
    /// UNREPRESENTABLE rather than refused.
    ///
    /// `runtime.frame_submit_depth = 1` used to be a config a `validate`
    /// clause had to catch: one frame submitted is the running frame alone,
    /// so the host round trip returns to the critical path. The window is
    /// `frame_dispatch_depth + 1` now and `frame_dispatch_depth >= 1` is
    /// already enforced above, so the smallest window a config can express is
    /// two — one running, one being built.
    #[test]
    fn the_smallest_expressible_window_still_leaves_one_frame_queued() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.runtime.frame_dispatch_depth = 1;
        cfg.validate().unwrap();
        assert_eq!(Runahead::of(1).submit_depth(), 2);

        cfg.runtime.frame_dispatch_depth = 0;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("runtime.frame_dispatch_depth must be >= 1")
        );
    }

    /// **THE DEPTHS ARE BOUNDED BY THE FORMULA, NOT BY A LITERAL** (alto
    /// F2b).
    ///
    /// What this test used to assert was `frame_dispatch_depth * frame_size <
    /// 13` — the product against a constant transcribed out of a deleted C++
    /// header. The engine CARVES its staging ring from these two numbers now
    /// (`frames_in_flight * STEPS_MAX + 1` pinned slots, allocated at load),
    /// so there is no fixed pool for a product to overflow; what is left is
    /// whether each factor is one the formula can be evaluated at.
    #[test]
    fn the_depths_are_bounded_by_the_runahead_formula() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();

        // `frame_size` is `k`, and `STEPS_MAX` is what the frame scheduler
        // was built and measured around.
        cfg.runtime.frame_size = u32::from(Runahead::STEPS_MAX);
        cfg.runtime.frame_dispatch_depth = 1;
        cfg.validate().unwrap();
        cfg.runtime.frame_size = u32::from(Runahead::STEPS_MAX) + 1;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("runtime.frame_size must be at most")
        );

        // The multiplier's ceiling is a fact about the engine's free-slot
        // WORD: `frames * 4 + 1` has to fit in 64 bits of bitmask.
        cfg.runtime.frame_size = 2;
        cfg.runtime.frame_dispatch_depth = u32::from(Runahead::MAX_FRAMES);
        cfg.validate().unwrap();
        assert!(
            Runahead::of(Runahead::MAX_FRAMES).staging_depth() <= 64,
            "the ceiling is the word's, so the deepest admissible depth must fit it"
        );
        cfg.runtime.frame_dispatch_depth = u32::from(Runahead::MAX_FRAMES) + 1;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("runtime.frame_dispatch_depth must be at most")
        );

        // And the combination the old check refused for being a product of
        // twelve is simply a deeper ring now: 4 frames of 4 steps is
        // `4 * 4 + 1 = 17` slots, carved at load, and nothing blocks.
        cfg.runtime.frame_dispatch_depth = 4;
        cfg.runtime.frame_size = 4;
        cfg.validate().unwrap();
        assert_eq!(Runahead::of(4).staging_depth(), 17);
    }

    /// The default deployment is article 1's floor: two frames in flight, and
    /// the ring the engine carves for it is nine slots.
    #[test]
    fn the_default_deployment_keeps_two_frames_in_flight() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        assert_eq!(cfg.runtime.frame_dispatch_depth, 2);
        let depth = u8::try_from(cfg.runtime.frame_dispatch_depth).unwrap();
        assert_eq!(Runahead::of(depth).frames_in_flight, 2);
        assert_eq!(Runahead::of(depth).staging_depth(), 9);
    }

    #[test]
    fn rejects_zero_frame_geometry() {
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.runtime.frame_size = 0;
        assert!(cfg.validate().is_err());

        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.runtime.frame_dispatch_depth = 0;
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

    // `rejects_public_engine_capacity_knobs` was here, and it went with
    // the engine it was about. It asserted that `max_forward_tokens`,
    // `max_forward_requests` and `max_model_len` are the ENGINE's to
    // derive rather than the operator's to set -- a rule that read as
    // general and was not: it held only for the dummy engine, whose
    // options struct listed none of the three. Both surviving kinds
    // accept all three as declared fields, so retargeting the test onto
    // one of them asserts a rule this schema does not have.

    #[test]
    fn device_string_or_list() {
        let one = r#"
[model]
name = "m"
model = "x"
[model.engine]
type = "metal"
device = "cuda:0"
"#;
        let cfg: Config = toml::from_str(one).unwrap();
        assert_eq!(cfg.model.engine.device, vec!["cuda:0".to_string()]);
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
model = "x"
[engine]
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
        // Every engine option struct carried it "for compatibility with the
        // Python wrapper", and nothing anywhere read it -- the engines are
        // linked in, so there has never been an executable to point at. An
        // unknown `[engine]` key is reshaped into the options table, so what
        // refuses it is the kind's own `deny_unknown_fields`.
        let legacy = r#"
[model]
name = "a"
model = "x"
[engine]
type = "cuda_native"
device = ["cuda:0"]
binary_path = "/opt/pie/engine"
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
model = "x"
[engine]
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
model = "x"

[[model]]
name = "b"
model = "y"
"#;
        let err = Config::parse(legacy).unwrap_err().to_string();
        assert!(
            err.contains("exactly one model") && err.contains("[[model]]"),
            "got: {err}"
        );
    }

    #[test]
    fn rejects_unknown_top_level_keys() {
        let bad = r#"
nonsense = true

[model]
name = "m"
model = "x"
[model.engine]
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
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
gpu_mem_utilization = 0.90
memory_profile = "throughput"
runtime_quant = "fp8"
mxfp4_moe = "routed_dequant"
mtp_assistant_snapshot_dir = "/models/gemma4-mtp"
mtp_num_drafts = 6
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.engine.kind, EngineKind::CudaNative);
        let opts: CudaNativeEngineOptions = cfg.model.engine.options.clone().try_into().unwrap();
        assert_eq!(opts.gpu_mem_utilization, 0.90);
        assert_eq!(opts.memory_profile, CudaMemoryProfile::Throughput);
        assert_eq!(opts.runtime_quant, "fp8");
        assert_eq!(opts.mxfp4_moe, "routed_dequant");
        assert_eq!(
            opts.mtp_assistant_snapshot_dir.as_deref(),
            Some("/models/gemma4-mtp")
        );
        assert_eq!(opts.mtp_num_drafts, 6);
        assert_eq!(cfg.model.weight_dtype, "bfloat16"); // default
        // Absent = derive: the planner scores candidates unless pinned.
        assert_eq!(opts.kv_page_size, None);
        assert_eq!(opts.kv_cache_dtype, "auto"); // default
    }

    #[test]
    fn an_out_of_range_gpu_mem_utilization_refuses_by_the_keys_name() {
        // **THE KEY THAT REACHED NO SHELL NOW SIZES THE ELASTIC POOL** (alto
        // streaming §3 item 5, `next.md` B1), so its range is load-bearing
        // rather than decorative: `0.0` is a deployment with no pool at all
        // and anything over `1.0` is a fraction of a card that does not exist.
        // Both refuse HERE, by the key's name, and again in
        // `engine_cuda::boot` for a boot document nobody wrote through this
        // config.
        for fraction in ["0.0", "1.5", "-0.5", "nan"] {
            let bad = format!(
                r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
gpu_mem_utilization = {fraction}
"#
            );
            let cfg: Config = toml::from_str(&bad).unwrap();
            let err = cfg
                .validate()
                .expect_err("a fraction outside (0.0, 1.0] is not a deployment")
                .to_string();
            assert!(
                err.contains("engine.gpu_mem_utilization"),
                "the refusal names the key; got: {err}"
            );
        }
        // And the whole card is a legal statement — it is what the pool took
        // before the fraction reached a shell, so an operator has to be able
        // to ask for it back.
        let whole = r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
gpu_mem_utilization = 1.0
"#;
        let cfg: Config = toml::from_str(whole).unwrap();
        cfg.validate().unwrap();
        let opts: CudaNativeEngineOptions = cfg.model.engine.options.clone().try_into().unwrap();
        assert_eq!(opts.gpu_mem_utilization, 1.0);
    }

    #[test]
    fn kv_page_size_pins_the_planner_when_set() {
        // Before this landed the field reached the engine and the planner
        // ignored it: only PIE_CUDA_KV_PAGE_SIZE could pin a page size. A
        // non-zero value now means the operator has made the choice the
        // planner's candidate search exists to make.
        let toml = r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
kv_page_size = 16
"#;
        let cfg: Config = toml::from_str(toml).unwrap();
        cfg.validate().unwrap();
        let opts: CudaNativeEngineOptions = cfg.model.engine.options.clone().try_into().unwrap();
        assert_eq!(opts.kv_page_size, Some(16));
    }

    #[test]
    fn cuda_native_options_default_when_omitted() {
        let cuda = r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().unwrap();
        let opts: CudaNativeEngineOptions = cfg.model.engine.options.clone().try_into().unwrap();
        assert_eq!(opts.swap_pool_size, 0);
        assert_eq!(opts.gpu_mem_utilization, 0.90);
        assert_eq!(opts.memory_profile, CudaMemoryProfile::Auto);
        assert_eq!(opts.mxfp4_moe, "auto");
        assert!(opts.mtp_assistant_snapshot_dir.is_none());
        assert_eq!(opts.mtp_num_drafts, 3);
        assert_eq!(cfg.model.engine.ready_timeout, Duration::from_secs(600));
        assert_eq!(opts.kv_cache_dtype, "auto");
    }

    #[test]
    fn rejects_invalid_embedded_kv_cache_dtype() {
        let bad = r#"
[model]
name = "default"
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
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
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
memory_profile = "{retired}"
"#
            );
            // Note where this fails: `[model.engine.options]` is a free-form
            // `toml::Table` in the schema, so the enum is only checked when
            // `validate()` converts it per engine kind -- not at parse.
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
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
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
model = "Qwen/Qwen3-0.6B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
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
model = "openai/gpt-oss-20b"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]

[model.engine.options]
mxfp4_moe = "mystery"
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("mxfp4_moe"), "got: {err}");
    }

    // -------------------------------------------------------------------------
    // [model] max_patches / max_images  (alto multimodal §5.5, the second axis)
    // -------------------------------------------------------------------------

    /// **THE COMMON CASE IS SAYING NOTHING**, and it has to parse to "derive".
    /// A vision SKU serves with zero configuration; the engine reads the
    /// loaded text and a plan that states no patch axis gets no ladder at all.
    #[test]
    fn a_deployment_that_states_no_patch_ceilings_asks_the_engine_to_derive_them() {
        let cuda = r#"
[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"

[model.engine]
type = "cuda_native"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().expect("a config that states nothing is valid");
        assert_eq!(cfg.model.patch_ceilings(), (None, None));
    }

    /// And an operator who has measured their traffic states two plain
    /// integers — row counts, not byte sizes, so no unit and no `ByteSize`.
    #[test]
    fn a_stated_patch_ceiling_is_two_plain_integers() {
        let cuda = r#"
[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"
max_patches = 2304
max_images = 4

[model.engine]
type = "cuda_native"
device = ["cuda:0"]
"#;
        let cfg: Config = toml::from_str(cuda).unwrap();
        cfg.validate().expect("stated ceilings are valid");
        assert_eq!(cfg.model.patch_ceilings(), (Some(2304), Some(4)));
    }

    /// **ZERO IS A TYPO, NOT A POLICY**, and it is refused by the key's own
    /// name rather than reaching the compiler as a refusal about a rectangle
    /// the operator never wrote down. Omitting the key is how a deployment
    /// says "derive one"; the two sentences must not be the same.
    #[test]
    fn a_patch_ceiling_of_zero_is_refused_by_its_own_name() {
        for key in ["max_patches", "max_images"] {
            let cuda = format!(
                r#"
[model]
name = "default"
model = "Qwen/Qwen3.5-0.8B"
{key} = 0

[model.engine]
type = "cuda_native"
device = ["cuda:0"]
"#
            );
            let cfg: Config = toml::from_str(&cuda).unwrap();
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains(key), "got: {err}");
            assert!(err.contains("omit the key"), "got: {err}");
        }
    }

    // -------------------------------------------------------------------------
    // [model] device_weight_budget / host_weight_budget  (alto streaming, W-3)
    // -------------------------------------------------------------------------

    /// `MINIMAL_METAL` with extra `[model]` keys, written INSIDE the one
    /// `[model]` table -- pie serves exactly one model, so a second `[model]`
    /// header is a duplicate-key refusal before any of this is reached.
    fn metal_with_model_keys(extra: &str) -> String {
        format!(
            "[model]\nname = \"default\"\nmodel = \"Qwen/Qwen3-0.6B\"\n{extra}\n\
             [model.engine]\ntype = \"metal\"\ndevice = [\"cpu\"]\n"
        )
    }

    #[test]
    fn both_weight_budgets_reach_residency_as_bytes() {
        // The keys carry a unit like every other size in this file, and the
        // engine's load contract takes plain bytes -- so the conversion
        // happens once, here, and `Residency` never sees a string.
        let toml = metal_with_model_keys(
            "device_weight_budget = \"18GiB\"\nhost_weight_budget = \"64GiB\"",
        );
        let cfg = Config::parse(&toml).expect("both budgets parse");
        let residency = cfg.model.residency();
        assert_eq!(residency.device_weight_budget, Some(18 << 30));
        assert_eq!(residency.host_weight_budget, Some(64 << 30));
    }

    #[test]
    fn absent_weight_budgets_are_uncapped() {
        // The whole point of the pair being optional: a config written before
        // these keys existed reaches the engine as the engine it had.
        let cfg = Config::parse(MINIMAL_METAL).expect("the minimal config parses");
        assert_eq!(cfg.model.device_weight_budget, None);
        assert_eq!(cfg.model.host_weight_budget, None);
        assert_eq!(cfg.model.residency(), engine::Residency::uncapped());
    }

    #[test]
    fn one_weight_budget_may_be_stated_without_the_other() {
        // Two budgets, not a mode: capping the device tier alone is a
        // deployment shape, not a half-written config.
        let device_only = metal_with_model_keys("device_weight_budget = \"512MiB\"");
        let cfg = Config::parse(&device_only).expect("device budget alone parses");
        assert_eq!(
            cfg.model.residency(),
            engine::Residency {
                device_weight_budget: Some(512 << 20),
                host_weight_budget: None,
            }
        );

        let host_only = metal_with_model_keys("host_weight_budget = \"2GiB\"");
        let cfg = Config::parse(&host_only).expect("host budget alone parses");
        assert_eq!(
            cfg.model.residency(),
            engine::Residency {
                device_weight_budget: None,
                host_weight_budget: Some(2 << 30),
            }
        );
    }

    #[test]
    fn a_weight_budget_refuses_at_boot_by_the_keys_name() {
        // A nonsense budget is a boot-time refusal naming the key the
        // operator wrote, not a panic somewhere downstream of the load.
        for (key, nonsense, needle) in [
            ("device_weight_budget", "\"35\"", "has no unit"),
            ("host_weight_budget", "\"35GB\"", "binary units only"),
            ("device_weight_budget", "18", "invalid type"),
        ] {
            let toml = metal_with_model_keys(&format!("{key} = {nonsense}"));
            let err = Config::parse(&toml)
                .expect_err("a nonsense budget must refuse")
                .to_string();
            assert!(err.contains(key), "refusal does not name {key}: {err}");
            assert!(err.contains(needle), "got: {err}");
        }
    }

    #[test]
    fn a_zero_weight_budget_refuses_by_the_keys_name() {
        // Parseable and still nonsense: no load holds zero weight bytes. The
        // engine would refuse it too, but naming plane byte counts the
        // operator never wrote; this one names the key.
        for key in ["device_weight_budget", "host_weight_budget"] {
            let toml = metal_with_model_keys(&format!("{key} = \"0B\""));
            let err = Config::parse(&toml)
                .expect_err("a zero budget must refuse")
                .to_string();
            assert!(err.contains(&format!("model.{key}")), "got: {err}");
            assert!(err.contains("zero weight bytes"), "got: {err}");
        }
    }
}
