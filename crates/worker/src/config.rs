//! The operator's TOML schema — every key `pie serve` reads.
//!
//! Every [`EngineKind`] now has a build that hosts it — the two portable
//! shells landed — so a config asking for one is refused by a missing feature
//! flag rather than by the name being unhostable.
//!
//! [`Config`] is the user-facing TOML schema; conversion to the runtime's own
//! config happens in [`crate::translate`].

use std::path::{Path, PathBuf};

use anyhow::{Result, bail, ensure};
use controller_api::Role;
// Run-ahead depths come from the engine contract's own module.
pub use engine::runahead::Runahead;
use serde::{Deserialize, Serialize};

/// Backend-specific option structs (typed views over `EngineConfig::options`).
pub mod backend;
/// Where a key LIVES in the operator's file (section list, moved-key map).
pub mod layout;
/// The dotted-path schema `pie config set`/`get` walk.
pub mod schema;
/// The unit-carrying value types (`"50ms"`, `"4GiB"`): [`Duration`], [`ByteSize`].
pub mod units;

pub use backend::{
    CudaNativeEngineOptions, MetalEngineOptions, VulkanEngineOptions, WgpuEngineOptions,
};
pub use units::{ByteSize, Duration};

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
    /// Parse the operator's file into a validated [`Config`].
    ///
    /// Pure: no file IO, no env, no clap. The file's sections are reshaped
    /// first — see [`crate::config::layout`] — so the rest of this parse
    /// still sees the shape the file was written against.
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
    /// Remote clients that may hold a scratch lease at once; the KV pool is
    /// divided evenly across this many slots.
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
    /// Off by default; enabling it also publishes an artifact digest.
    #[serde(default)]
    pub enabled: bool,
    /// Shortest suffix worth offloading a prefill for; `0` derives one from
    /// the transport (512 tokens over NIXL, 2048 inline).
    #[serde(default)]
    pub prefill_min_suffix_tokens: usize,
    /// Transfers in flight to any one partner before the next waits.
    #[serde(default = "default_offload_max_outstanding")]
    pub max_outstanding_per_partner: u32,
    /// How KV pages cross between workers — see [`OffloadTransfer`]. No
    /// shipped build hosts NIXL, so `nixl` refuses the boot.
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
/// No shipped build hosts `Nixl`: it parses but fails the boot at
/// `link::partner::PartnerLinkManager::new` with a missing-feature error.
/// `Auto` takes NIXL where available, which today is nowhere.
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
    /// Gateway endpoint(s) this worker dials into (distributed).
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
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    /// Largest blob a client may upload in one request.
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
    // Cap at 64: beyond that the scheduling overhead adds variance without
    // adding parallelism. Override via `[server] worker_threads = ...`.
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
    /// Filters `wasi:sockets` only; `wasi:http` bypasses it, so use
    /// `allow_network = false` instead when that matters.
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
    /// Fetch the Python WASM runtime at boot when it is missing. Python
    /// inferlets need it; Rust inferlets do not.
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
    /// Which SKU of that checkpoint to serve, or omit to let the load identify
    /// one — the cheapest row whose contract and plan fit the checkpoint.
    #[serde(default)]
    pub sku: Option<String>,
    /// Which published draft head to serve `model` with, by its short name
    /// (`dflash`, `dflash2`): `sku` looked up in the catalog's table of
    /// published heads (`models::drafter::PUBLISHED`) for the target `model`
    /// names, so a deployment says which drafter it wants rather than which
    /// row spells it. The artifact must already carry the head — `pie model
    /// import <target> --drafter <name>` is what puts it there. Refused when
    /// `model` is a path (the table keys on repository ids) or names a target
    /// the table lacks; `sku` beside it must agree.
    #[serde(default)]
    pub drafter: Option<String>,
    /// Which backend runs the model, on what devices.
    pub engine: EngineConfig,
    /// Where this model's materialized-weight artifacts are kept between runs.
    /// Empty derives `$PIE_HOME/cache/weights` (distinct from the `.zt`
    /// artifact store at `$PIE_HOME/models`).
    #[serde(default)]
    pub weight_cache_dir: String,
    /// Where this deployment's shared adapters live, or empty to mount none.
    /// A read-only directory with one subdirectory per adapter, each holding
    /// an `adapter.toml` and the plane files it names.
    #[serde(default)]
    pub adapter_dir: String,
    /// Dtype weights are materialized in. Separate from `activation_dtype`
    /// (in `[engine]`, the dtype compute happens in) — narrower weights and
    /// wider compute is a normal combination.
    #[serde(default = "default_weight_dtype")]
    pub weight_dtype: String,
    /// How many weight bytes this load may keep on the device (tier T0),
    /// written with its unit (`"18GiB"`). Omit for uncapped. Distinct from
    /// `[engine] gpu_mem_utilization`, which budgets KV pages and scratch,
    /// not the weight table.
    #[serde(default)]
    pub device_weight_budget: Option<ByteSize>,
    /// How many weight bytes this load may keep in the pinned host cache
    /// (tier T1), written with its unit (`"64GiB"`). Omit for uncapped.
    #[serde(default)]
    pub host_weight_budget: Option<ByteSize>,
    /// The most patch rows one fire may carry, over every image of every lane
    /// in it. Omit it: a vision SKU derives a ceiling from the checkpoint's
    /// own shapes, and a text-only SKU wants no ladder at all.
    #[serde(default)]
    pub max_patches: Option<u32>,
    /// The patch axis's lane ceiling: the most images one fire may carry.
    /// Omit it, as above; the default is derived from `max_patches`.
    #[serde(default)]
    pub max_images: Option<u32>,
    /// How many adapter seats this deployment intends to use, and which
    /// adapters to write into them at boot. Absent means zero seats.
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
/// The capacity is an intent, not a pool size: `seats` states how many the
/// deployment intends to register, and a load whose intent exceeds what the
/// model text seats is refused at compile.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdapterConfig {
    /// How many adapter rows this deployment intends to use. Omit to derive
    /// it from [`registered`](AdapterConfig::registered); state it to reserve
    /// room for adapters that arrive later.
    #[serde(default)]
    pub seats: Option<u32>,
    /// The adapters to write into those seats at boot, in the order given.
    #[serde(default)]
    pub registered: Vec<RegisteredAdapter>,
}

/// One adapter, as an operator names it.
///
/// The planes are raw bytes and the padding is the caller's: a file here is
/// one bank's slot, exactly, in the bank's declared dtype and layout. A file
/// of the wrong length is refused by the engine, by name, with both numbers.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RegisteredAdapter {
    /// Which row of every named bank this fills — the id a lane routes to.
    pub id: u32,
    /// Bank name (the plan's own `Param` spelling) to the file holding one
    /// slot of it. A bank this map omits keeps what it held.
    #[serde(default)]
    pub planes: std::collections::BTreeMap<String, String>,
}

impl AdapterConfig {
    /// The capacity to bake against: what the operator stated, else what the
    /// roster needs (a roster whose highest id is `n` needs `n + 1` seats).
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

    /// Nothing to seat and nothing to register.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seats() == 0 && self.registered.is_empty()
    }

    /// Checks what this layer can answer without a device: whether the
    /// roster fits the stated capacity, and every plane path is absolute.
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
    /// The shared-adapter mount as the boot seam spells it: `Some(path)` when
    /// the operator stated one, `None` (the feature off) for the empty
    /// default.
    pub fn adapter_mount(&self) -> Option<std::path::PathBuf> {
        (!self.adapter_dir.is_empty()).then(|| std::path::PathBuf::from(&self.adapter_dir))
    }

    /// The two weight budgets, in the form the engine's load contract states
    /// them. Both absent is [`engine::Residency::uncapped`].
    #[must_use]
    pub fn residency(&self) -> engine::Residency {
        engine::Residency {
            device_weight_budget: self.device_weight_budget.map(|b| b.as_bytes()),
            host_weight_budget: self.host_weight_budget.map(|b| b.as_bytes()),
        }
    }

    /// The second row axis's two ceilings, in the form the engine's load
    /// contract states them. Both absent derives a ladder from the loaded
    /// text when the plan states a patch axis.
    #[must_use]
    pub fn patch_ceilings(&self) -> (Option<u32>, Option<u32>) {
        (self.max_patches, self.max_images)
    }

    /// Resolve `[model] drafter` into `[model] sku` through the published
    /// heads table. Called once at load, before validation.
    ///
    /// # Errors
    ///
    /// A `model` the table cannot key (a path), a name it does not know for
    /// this target, or a `sku` that names another row.
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
        let Some(published) = models::drafter::published(target, drafter) else {
            let known: Vec<&str> = models::drafter::published_for(target).map(|p| p.drafter).collect();
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
        // Relative would resolve against a working directory the operator
        // did not choose and that differs between worker and engine process.
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
        // Zero is a typo, not a policy: omitting the key derives a ceiling,
        // writing `0` admits no image at all.
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
    /// The outermost of the three clocks here — bounds the answer a caller
    /// is waiting for, distinct from `submit_deadline` and `silence_timeout`.
    #[serde(default = "default_request_timeout")]
    pub request_timeout: Duration,
    /// How long a pipeline hard-blocking a frame's seal may go without
    /// submitting before the runtime stops waiting for it, in microseconds.
    /// Does not fail the pipeline — the lane is dropped from the wait-set
    /// (an involuntary `forward.park()`) so the boundary seals at once.
    /// Exposed to guests verbatim as `model.submit-deadline-us()`.
    #[serde(default = "default_submit_deadline")]
    pub submit_deadline: Duration,
    /// How long a lane may stay silent in total — through the leash above and
    /// on past it — before the runtime terminates its process, in seconds.
    /// A verdict, so it is generous: a lane that calls `forward.park()` is
    /// never killed however long it stays away.
    #[serde(default = "default_silence_timeout")]
    pub silence_timeout: Duration,
    /// Waves per frame (*k*): how many token steps the wait-all quorum admits
    /// before it runs. A deployment constant, fixed at runtime start like the
    /// KV page size. Bounded above by the CUDA engine — see [`Self::validate`].
    #[serde(default = "default_frame_size")]
    pub frame_size: u32,
    /// Frames the runtime keeps posted to the engine but not yet retired: the
    /// dispatch loop's enqueue horizon, keeping the GPU from idling between
    /// frames. Bounded jointly with `frame_size` — see [`Self::validate`].
    #[serde(default = "default_frame_dispatch_depth")]
    pub frame_dispatch_depth: u32,
    /// Hard cap on inferlets admitted at once. Omit to derive it from the
    /// engine's `max_forward_requests`, which is what fills a batch.
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
        // Engine coupling: `frame_size` is `k` and `frame_dispatch_depth` is the
        // multiplier in `engine::runahead::Runahead`'s staging formula.
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
pub struct EngineConfig {
    /// Which engine hosts this model. Every spelling parses; a flavor this
    /// build does not host is refused by name at boot.
    #[serde(rename = "type")]
    pub kind: EngineKind,
    /// Single string or list of strings — both accepted on input.
    #[serde(deserialize_with = "deserialize_string_or_list")]
    pub device: Vec<String>,
    /// Ranks the model is sharded across. Must divide the device list.
    #[serde(default = "default_tp_size")]
    pub tensor_parallel_size: u32,
    /// Compute dtype for activations, e.g. `"bfloat16"`. Separate from
    /// `weight_dtype`: a deployment can store weights narrower than it
    /// computes.
    #[serde(default = "default_activation_dtype")]
    pub activation_dtype: String,
    // `random_seed`, `kv_pages`, `ready_timeout` and `shutdown_timeout` are
    // not keys of this table: an engine is a function call in this process,
    // not a subprocess to seed, wait on or abandon.
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
                        // `[engine]`, not `model.engine.options`: that's what
                        // the operator's file actually spells these keys as.
                        anyhow::anyhow!(
                            "invalid [engine] options for engine type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                opts.validate()?;
            }
            // Read straight off the options table so an otherwise-valid
            // `[model.engine.options]` isn't refused for a key this arm
            // doesn't police. Absent means the engine's 0.90 default.
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
            // Typed, like the CUDA arm and unlike Metal's: this table is
            // parsed with `deny_unknown_fields`, so a key the shell never
            // reads is refused here rather than ignored at boot.
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
            // Typed too, and for the same reason: `deny_unknown_fields`
            // makes a key the shell never reads a refusal here rather than a
            // line quietly ignored at boot.
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

/// Which engine a `[model.engine] type` names.
///
/// Every name here is now offered: `Vulkan` and `Wgpu` were named-not-hosted
/// until their shells landed, and each is one `--features` flag away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    /// Native CUDA engine — embedded as a static lib in `worker`
    /// (requires `--features cuda`).
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
    fn rejects_the_legacy_unit_suffixed_names() {
        // Renamed, not aliased: deny_unknown_fields turns an old config into
        // a clear error naming the key.
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

    #[test]
    fn parses_minimal_metal_config() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.engine.kind, EngineKind::Metal);
        assert_eq!(cfg.model.engine.device, vec!["cpu".to_string()]);
        assert_eq!(cfg.server.port, 8080);
    }

    /// Every engine kind's `as_str` is the word a config file spells it with;
    /// `as_str` is a hand-written match and nothing ties it to serde's
    /// `rename_all` if they drift.
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
            // Back out through serde, the direction `schema::default_values` relies on.
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

    /// `[model.adapters]` absent is the shape hard-coded before it existed:
    /// zero seats, no roster, and the correction op never launches.
    #[test]
    fn adapters_are_absent_by_default_and_that_is_zero_seats() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert!(cfg.model.adapters.is_empty());
        assert_eq!(cfg.model.adapters.seats(), 0);
        assert!(cfg.model.adapters.registered.is_empty());
    }

    /// The capacity is stated once: either the operator says it, or the
    /// roster does — one number, one owner.
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

        // And a near-miss of a key is still refused, as everywhere else.
        let typo = MINIMAL_METAL.replace(
            "model = \"Qwen/Qwen3-0.6B\"",
            "model = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseat = 2\n",
        );
        toml::from_str::<Config>(&typo)
            .expect_err("a near-miss of `seats` must be refused by name");
    }

    #[test]
    fn rejects_a_cache_section() {
        // [cache] existed briefly and was withdrawn.
        let toml = format!("{MINIMAL_METAL}\n[cache]\nptir_dir = \"/tmp/x\"\n");
        assert!(toml::from_str::<Config>(&toml).is_err());
    }

    // -------------------------------------------------------------------------
    // [model] max_patches / max_images  (the second row axis)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // [model] device_weight_budget / host_weight_budget
    // -------------------------------------------------------------------------
}
