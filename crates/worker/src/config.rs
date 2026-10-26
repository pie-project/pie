use std::path::{Path, PathBuf};

use anyhow::{Result, ensure};
use controller_api::Role;
pub use engine::runahead::Runahead;
use serde::{Deserialize, Serialize};

/// Backend-specific option structs (typed views over `EngineConfig::options`).
pub mod backend;
/// The dotted-path schema `pie config set`/`get` walk.
pub mod schema;

pub use backend::{
    CerebrasEngineOptions, CudaEngineOptions, MetalEngineOptions, VulkanEngineOptions,
    WgpuEngineOptions, XlaEngineOptions,
};
pub use units::{ByteSize, Duration};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Client-facing listener.
    #[serde(default)]
    pub server: ServerConfig,
    /// OpenTelemetry export. Off unless an endpoint is named.
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    /// The single `[model]` table. Pie serves exactly one model.
    pub model: ModelConfig,
    /// Which backend runs the model, on what devices.
    pub engine: EngineConfig,
    /// Batching and timeout policy. Every field has a measured default; the
    /// frame knobs are part of the guest contract.
    #[serde(default)]
    pub runtime: RuntimeConfig,
    /// The box an inferlet runs in: what it may reach, and how big it may get.
    #[serde(default)]
    pub sandbox: SandboxConfig,
    /// The cluster plane: its policies, and the topology this worker joins
    /// (controller + role + gateways). `controller` unset ⇒ single-node.
    #[serde(default)]
    pub cluster: ClusterConfig,
    /// Other workers' part in this one's passes, and its in theirs.
    #[serde(default)]
    pub offload: OffloadConfig,
}

impl Config {
    /// Parse the operator's file into a validated [`Config`]. The file's
    /// shape is this struct's: a key is where it is written.
    ///
    /// Pure: no file IO, no env, no clap.
    pub fn parse(s: &str) -> Result<Self> {
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

    /// State a diagnostics word list into `[engine] diagnostics`
    /// — what `pie serve --diag …` / `pie run --diag …` does to the config it
    /// just read, so a person debugging never edits a file to turn a trace on.
    ///
    /// The words themselves are the shell's vocabulary and are parsed there
    /// (`engine_cuda::Diagnostics`, `engine_metal::Diagnostics`), at boot,
    /// where an unknown one refuses by name.
    ///
    /// # Errors
    ///
    /// An engine flavor with no diagnostics record. A flag that quietly did
    /// nothing would be the same silence the typed record exists to end.
    pub fn state_diagnostics(&mut self, words: &str) -> Result<()> {
        match self.engine.kind {
            EngineKind::Cuda | EngineKind::Metal => {
                self.engine.options.insert(
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
        self.engine.validate()?;
        self.server.validate()?;
        self.runtime.validate()?;
        self.sandbox.validate()?;
        self.cluster.validate()?;
        self.offload.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OffloadConfig {
    /// Shortest uncached suffix worth a peer's prefill; `0`: 2048 tokens.
    #[serde(default)]
    pub prefill_min_suffix_tokens: usize,
    /// How KV pages cross between workers — see [`OffloadTransfer`].
    #[serde(default)]
    pub transfer: OffloadTransfer,
    /// Where this worker serves other workers' passes (`host:port`); omit to
    /// serve none.
    #[serde(default)]
    pub listen: Option<String>,
    /// Workers to prefill this worker's passes on, by their `listen`
    /// address. The first that serves this model in this class is taken.
    #[serde(default)]
    pub peers: Vec<String>,
    /// Pool workers (`pie-pool`) to send every run this worker keeps to, by
    /// their bulk address.
    #[serde(default)]
    pub pools: Vec<String>,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            prefill_min_suffix_tokens: 0,
            transfer: OffloadTransfer::Auto,
            listen: None,
            peers: Vec::new(),
            pools: Vec::new(),
        }
    }
}

impl OffloadConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            cfg!(feature = "nixl") || self.transfer != OffloadTransfer::Nixl,
            "[offload] transfer = \"nixl\" needs a build with the `nixl` feature; \"auto\" \
             takes the bulk link"
        );
        Ok(())
    }
}

/// How KV pages cross between workers.
///
/// `Nixl` reads a peer's kv in place, device to device, where both workers
/// export kv of one layout; a build without the `nixl` feature (built
/// against `NIXL_PREFIX`) refuses it. Otherwise the pages' bytes ride the
/// bulk link, a peer's prefill streamed back a layer at a time as it is
/// computed. `Auto` takes the bulk link: the stream overlaps the transfer
/// with the prefill, a read in place starts only once it is done, and only
/// the operator knows whether NIXL reaches the peer's device memory directly
/// (RDMA, NVLink) or copies it through host memory over TCP.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OffloadTransfer {
    Inline,
    Nixl,
    #[default]
    Auto,
}

/// Distributed-cluster topology. Absent, or `controller` unset ⇒ single-node
/// (the worker terminates clients directly; no controller/gateway).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ClusterConfig {
    /// The cluster-plane policies, run by the controller: each a built-in
    /// policy's name or a component path, no two granted the same write;
    /// omit to run none. Single-node only: a distributed worker's controller
    /// states its own. The worker plane's is `[runtime] worker_policy`.
    #[serde(default)]
    pub cluster_policies: Vec<String>,
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
            ensure!(
                self.cluster_policies.is_empty(),
                "[cluster] cluster_policies is the controller's to run: a worker that \
                 joins one (`controller` set) runs none, so state them in the \
                 controller's file"
            );
        }
        ensure!(
            self.role != Some(Role::Encode),
            "[cluster] role \"encode\" is not served: no worker computes a media encode for \
             another"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Address the client edge binds (`host:port`). Loopback by default -- a
    /// reachable port should be something an operator asks for; port 0 asks
    /// the OS for a free one.
    #[serde(default = "default_listen")]
    pub listen: std::net::SocketAddr,
    /// Verbose server logging, and passed down to the embedded engine.
    /// Independent of `--log-level`, which sets the tracing filter.
    #[serde(default)]
    pub verbose: bool,
    /// Tokio worker threads. Derived from the visible CPUs: half of them,
    /// capped at 64.
    #[serde(default = "default_worker_threads")]
    pub worker_threads: usize,
    /// Largest blob a client may upload in one request.
    #[serde(default = "default_max_upload")]
    pub max_upload: ByteSize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            verbose: false,
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

fn default_listen() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([127, 0, 0, 1], 8080))
}
fn default_true() -> bool {
    true
}
// Half the CPUs leave the rest to the engine's own threads (its serve loop,
// the tokenizer and lane pools); a box whose quota the workers alone fill
// (24 of an H100 node's) settles a decode fleet into lockstep, its guests
// trailing every frame: 7.5k tok/s against 18.7k at 12 on DeepSeek-V4's
// miniature, 64 lanes.
fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 64))
        .unwrap_or(4)
}
fn default_max_upload() -> ByteSize {
    ByteSize::from_mib(256)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryConfig {
    /// The OTLP collector to export traces to (`http://localhost:4317`);
    /// omit to export none.
    #[serde(default)]
    pub endpoint: Option<String>,
    /// `service.name` on exported spans.
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            endpoint: None,
            service_name: default_service_name(),
        }
    }
}

fn default_service_name() -> String {
    "pie".to_string()
}

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
    #[serde(default = "default_scratch_dir")]
    pub scratch_dir: PathBuf,
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
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            allow_fs: false,
            scratch_dir: default_scratch_dir(),
            allow_network: true,
            network_allowed_hosts: default_network_allowed_hosts(),
            max_instances: default_max_instances(),
            max_memory: default_max_memory(),
            warm_memory: ByteSize::from_mib(0),
            warm_slots: default_warm_slots(),
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
fn default_scratch_dir() -> PathBuf {
    // wasm32 has no temp dir to ask for (`temp_dir` panics); the browser's
    // sandbox reads no files anyway.
    if cfg!(target_arch = "wasm32") {
        PathBuf::from("/tmp/pie")
    } else {
        std::env::temp_dir().join("pie")
    }
}
fn default_network_allowed_hosts() -> Vec<String> {
    vec!["*".to_string()]
}

/// `[model] rope_scaling`: the YaRN a deployment stretches its model's
/// native context by.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RopeScalingConfig {
    #[serde(alias = "type")]
    pub rope_type: String,
    pub factor: f64,
    #[serde(default)]
    pub original_max_position_embeddings: Option<u32>,
}

impl RopeScalingConfig {
    /// The whole YaRN factor this states.
    ///
    /// # Errors
    ///
    /// A type other than YaRN, or a factor not a whole number past one.
    pub fn yarn(&self) -> Result<u32> {
        ensure!(
            self.rope_type == "yarn",
            "model.rope_scaling: pie stretches a served context by YaRN, not {:?}; other \
             scalings are what a model's own package states",
            self.rope_type
        );
        ensure!(
            self.factor > 1.0 && self.factor.fract() == 0.0 && self.factor < 1024.0,
            "model.rope_scaling.factor is a whole number past 1, not {}",
            self.factor
        );
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "checked whole and in range above"
        )]
        Ok(self.factor as u32)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    /// What clients ask for this model by. Required, and free-form: it names
    /// the deployment, not the checkpoint.
    pub name: String,
    /// What to serve: a store name (`Qwen--Qwen3-0.6B`, as `pie model list`
    /// prints it) or a path to a `.zt` artifact. See `weights::resolve`.
    pub source: String,
    /// The weight precision to serve (`bf16`, `u4g64`, or a mixed recipe's
    /// banks joined with `+`, as `u4g64+u2g64`). The checkpoint stores its own
    /// and is served at it; stating one picks between artifacts of one model
    /// and refuses a checkpoint stored at another. Omit to serve what is
    /// stored.
    #[serde(default)]
    pub weight_precision: Option<String>,
    /// The dtype the kv cache is kept at. Omit for the model's default.
    #[serde(default)]
    pub kv_dtype: Option<String>,
    /// Parts of the checkpoint to leave unloaded (`vision`, `selfcond`).
    #[serde(default)]
    pub skip_parts: Vec<String>,
    /// The drafter to serve with (`mtp`, `dflash`, `dflash2`, `dspark`,
    /// `eagle`), or `none` to serve without the checkpoint's. The artifact
    /// must carry it — `pie model import <target> --drafter <name>` is what
    /// adds a published head. Omit to serve with whatever the checkpoint
    /// carries.
    #[serde(default)]
    pub drafter: Option<String>,
    /// A rope scaling to serve the model past its native context, as
    /// config.json (or vLLM's `--hf-overrides`) spells one: `{ rope_type =
    /// "yarn", factor = 4 }`, the factor whole, its
    /// `original_max_position_embeddings` the package's native context.
    /// Unstated, a YaRN the checkpoint's config.json states is served where
    /// the model's package does not turn one on its own.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Where this model's materialized-weight artifacts are kept between runs.
    /// Empty derives `$PIE_HOME/cache/weights` (distinct from the `.zt`
    /// artifact store at `$PIE_HOME/artifacts`).
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
    /// written with its unit (`"18GiB"`). Omit for uncapped.
    #[serde(default)]
    pub device_weight_budget: Option<ByteSize>,
    /// How many weight bytes this load may keep in the pinned host cache
    /// (tier T1), written with its unit (`"64GiB"`). Omit for uncapped.
    #[serde(default)]
    pub host_weight_budget: Option<ByteSize>,
    /// How many device bytes the kv pages and recurrent-state slots take
    /// (`"20GiB"`), per device. Omit to fill what `device_memory_share`
    /// leaves once the weights and working set are down. A budget the device
    /// does not hold refuses the load; it is never cut to fit.
    #[serde(default)]
    pub device_kv_budget: Option<ByteSize>,
    /// The share of the device's memory this load may take: weights, working
    /// set and kv pool. What other processes hold when the load begins comes
    /// off it, as on a card another server shares.
    #[serde(default = "default_device_memory_share")]
    pub device_memory_share: f64,
    /// How many pinned host bytes a preempted process's kv pages and
    /// recurrent-state slots may be suspended into (`"32GiB"`), shared by
    /// the tensor-parallel ranks: one rs row per device slot, then kv pages.
    /// Omit for every seat at full context, at least one device pool and at
    /// most four (cuda), up to half the host memory the load leaves free (so
    /// workers sharing a host each find room); a stated budget is capped by
    /// all of it. Pinned memory is slow to give back: a worker that dies
    /// holds its host's other GPU processes ~0.2 s a GiB of it. The
    /// container's cgroup limit (v1 or v2) counts as the host's. pie cannot
    /// compute kv again, so this is where memory contention goes: it cannot
    /// be turned off.
    #[serde(default)]
    pub host_kv_budget: Option<ByteSize>,
    /// How many disk bytes suspended kv may spill into once host swap is
    /// full (`"200GiB"`), shared by the ranks, each in its own slot file
    /// under `disk_kv_dir` (default `$PIE_HOME/cache/kv/<checkpoint
    /// digest>`), unlinked once open so it never outlives the server unless
    /// `disk_kv_persistent`. Metal
    /// and Vulkan keep no host tier, so there it is the only one, and
    /// suspended rs slots go to it too. Omit for four host tiers' worth on
    /// cuda, or every seat at full context elsewhere, within a quarter of
    /// the free space there (skipped when the disk cannot take it); `"0B"`
    /// turns the disk tier off.
    #[serde(default)]
    pub disk_kv_budget: Option<ByteSize>,
    /// Where the disk tier's slot files go. Omit for
    /// `$PIE_HOME/cache/kv/<checkpoint digest>`.
    #[serde(default)]
    pub disk_kv_dir: Option<std::path::PathBuf>,
    /// Keeps the disk tier for the next run: its slot files stay under
    /// `disk_kv_dir` with a journal of the sealed pages they hold, and a
    /// restart takes those pages back as cache. Swap and cache share
    /// `disk_kv_budget`, suspended kv first. Off by default; cuda only, and
    /// not for a model with recurrent state or a window.
    #[serde(default)]
    pub disk_kv_persistent: bool,
    /// Whether a warm boot builds the pinned host copy of the weights in the
    /// background. On (the default) T1's planes are verified where they lie in the artifact and served from
    /// there while a background thread builds the page-locked copy, so the
    /// load answers sooner and the first fires take page faults until the
    /// fill lands. `false` is the eager arm: the page-locked image is built
    /// before the load answers.
    ///
    /// Needs a device that reports `pageableMemoryAccess` (CUDA 12.2+ HMM);
    /// where it does not, the eager path is what happens whatever this says.
    #[serde(default = "default_true")]
    pub host_weight_deferred: bool,
    /// The most patch rows one fire may carry, over every image of every lane
    /// in it. Omit it: a vision deployment derives a ceiling from the checkpoint's
    /// own shapes, and a text-only deployment wants no ladder at all.
    #[serde(default)]
    pub max_patches: Option<u32>,
    /// The patch axis's lane ceiling: the most images one fire may carry.
    /// Omit it, as above; the default is derived from `max_patches`.
    #[serde(default)]
    pub max_images: Option<u32>,
    /// The most rows the widest voxel value of one fire may hold: a
    /// `vae.decode` pass is admitted at this over its growth in latents, a
    /// `vae.encode` pass at this in pixels. Omit it and the engine fits it to
    /// the working set its memory share leaves.
    #[serde(default)]
    pub max_voxels: Option<u32>,
    /// The most clips one fire may carry, over every lane. Omit it and one
    /// is derived from the lane ceiling.
    #[serde(default)]
    pub max_clips: Option<u32>,
    /// How many adapter seats this deployment intends to use, and which
    /// adapters to write into them at boot. Absent means zero seats.
    #[serde(default)]
    pub adapters: AdapterConfig,
}

fn default_device_memory_share() -> f64 {
    engine::load::DEFAULT_DEVICE_MEMORY_SHARE
}

fn default_weight_dtype() -> String {
    "bfloat16".to_string()
}

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
    /// The adapters to write into those seats at boot, in the order given.
    #[serde(default)]
    pub seats: Option<u32>,
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
    /// Nothing to seat and nothing to register.
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

    /// The two weight budgets and the tier arm, in the form the engine's load
    /// contract states them. Both budgets absent and the arm on is
    /// [`engine::Residency::uncapped`].
    /// The second row axis's two ceilings, in the form the engine's load
    /// contract states them. Both absent derives a ladder from the loaded
    /// text when the plan states a patch axis.
    /// The THIRD row axis's two ceilings, in the same form. Both absent
    /// derives a ladder from the loaded text when the plan states a voxel
    /// axis.
    #[must_use]
    pub fn residency(&self) -> engine::Residency {
        engine::Residency {
            device_weight_budget: self.device_weight_budget.map(|b| b.as_bytes()),
            host_weight_budget: self.host_weight_budget.map(|b| b.as_bytes()),
            deferred_tier: self.host_weight_deferred,
            device_kv_budget: self.device_kv_budget.map(|b| b.as_bytes()),
            device_memory_share: self.device_memory_share,
            host_kv_budget: self.host_kv_budget.map(|b| b.as_bytes()),
            kv_seats: None,
            disk_kv_budget: self.disk_kv_budget.map(|b| b.as_bytes()),
            disk_kv_dir: self.disk_kv_dir.clone(),
            disk_kv_persistent: self.disk_kv_persistent,
            kv_export: false,
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

    /// What this config changes about the deployment its checkpoint states.
    ///
    /// # Errors
    ///
    /// A precision or kv dtype word this build does not know; a part or
    /// drafter word is checked against the model when the deployment is
    /// composed.
    pub fn overrides(&self) -> Result<runtime::engine::load::Overrides> {
        let dtype = |word: &str, field: &str| {
            poem::star::dtype_of(word.trim())
                .ok_or_else(|| anyhow::anyhow!("model.{field} = {word:?} names no dtype"))
        };
        let precision = self
            .weight_precision
            .as_deref()
            .map(|words| {
                words
                    .split('+')
                    .map(|w| {
                        poem::star::precision_of(w.trim()).ok_or_else(|| {
                            anyhow::anyhow!("model.weight_precision = {w:?} names no dtype")
                        })
                    })
                    .collect()
            })
            .transpose()?;
        let kv = self
            .kv_dtype
            .as_deref()
            .map(|w| dtype(w, "kv_dtype"))
            .transpose()?;
        let off = self
            .skip_parts
            .iter()
            .map(|word| word.trim().to_string())
            .collect();
        let drafter = match self.drafter.as_deref().map(str::trim) {
            None => None,
            Some("none") => Some(None),
            Some(word) => Some(Some(word.to_string())),
        };
        Ok(runtime::engine::load::Overrides {
            precision,
            kv,
            off,
            drafter,
            tp: None,
            ep: None,
            rank: None,
            expert_replicas: None,
            yarn: self
                .rope_scaling
                .as_ref()
                .map(RopeScalingConfig::yarn)
                .transpose()?,
            yarn_original: self
                .rope_scaling
                .as_ref()
                .and_then(|r| r.original_max_position_embeddings),
        })
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            !self.name.is_empty(),
            "model.name must be a non-empty string"
        );
        ensure!(
            !self.source.trim().is_empty(),
            "model.source must name a stored artifact or a path to one \
             (`pie model list` shows what is available)"
        );
        self.overrides()?;
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
        ensure!(
            self.device_memory_share.is_finite()
                && self.device_memory_share > 0.0
                && self.device_memory_share <= 1.0,
            "model.device_memory_share must be finite and in (0.0, 1.0]"
        );
        ensure!(
            self.device_kv_budget.is_none_or(|b| b.as_bytes() > 0),
            "model.device_kv_budget is zero, and no load serves without a kv pool; \
             state a real budget or omit the key to fill what device_memory_share leaves"
        );
        ensure!(
            self.host_kv_budget.is_none_or(|b| b.as_bytes() > 0),
            "model.host_kv_budget is zero, but pie cannot compute kv again: under \
             memory contention suspended kv has to go to host memory. State a real \
             budget or omit the key for every seat at full context"
        );
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
    /// The worker-plane PLEX policy: a built-in policy's name (`noop`,
    /// `priority`) or a component path; omit to run none. The cluster
    /// plane's are `[cluster] cluster_policies`.
    #[serde(default)]
    pub worker_policy: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            submit_deadline: default_submit_deadline(),
            silence_timeout: default_silence_timeout(),
            frame_size: default_frame_size(),
            frame_dispatch_depth: default_frame_dispatch_depth(),
            max_concurrent_processes: None,
            worker_policy: None,
        }
    }
}

impl RuntimeConfig {
    fn validate(&self) -> Result<()> {
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

/// `[engine]`: the keys every engine reads, and beside them the named
/// engine's own, each checked against that engine's options struct.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineConfig {
    /// Which engine hosts this model. Every kind parses; one this build does
    /// not host is refused by name at boot.
    #[serde(rename = "type")]
    pub kind: EngineKind,
    /// The devices it runs on, by name (`cuda:0`, `vulkan:1`, `tpu:0`): one
    /// string or a list.
    #[serde(deserialize_with = "deserialize_string_or_list")]
    pub device: Vec<String>,
    /// Ranks the model is sharded across. Must divide the device list.
    #[serde(default = "default_tp_size")]
    pub tensor_parallel_size: u32,
    /// Ranks a MoE model's routed experts spread across, each rank one
    /// device running its own requests (attention data-parallel, experts
    /// expert-parallel). Must equal the device list; tensor parallelism
    /// beside it is not served yet.
    #[serde(default = "default_ep_size")]
    pub expert_parallel_size: u32,
    /// Slots each expert-parallel rank holds past its share of a MoE layer's
    /// experts, which the busiest experts' replicas fill.
    #[serde(default)]
    pub expert_replicas: u32,
    /// Compute dtype for activations, e.g. `"bfloat16"`. Separate from
    /// `weight_dtype`: a deployment can store weights narrower than it
    /// computes.
    #[serde(default = "default_activation_dtype")]
    pub activation_dtype: String,
    /// Engine-specific knobs, written beside the common keys. Embedded
    /// engines parse this into typed option structs.
    #[serde(flatten)]
    pub options: toml::Table,
}

impl EngineConfig {
    fn validate(&self) -> Result<()> {
        ensure!(!self.device.is_empty(), "engine.device must be non-empty");
        match self.kind {
            EngineKind::Cuda => {
                let opts: CudaEngineOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                    anyhow::anyhow!(
                        "invalid [engine] options for engine type {:?}: {e}",
                        self.kind,
                    )
                })?;
                opts.validate()?;
            }
            EngineKind::Metal => {}
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
            EngineKind::Xla => {
                let opts: XlaEngineOptions = toml::Value::Table(self.options.clone())
                    .try_into()
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "invalid [engine] options for engine type {:?}: {e}",
                            self.kind,
                        )
                    })?;
                opts.validate()?;
            }
            EngineKind::Cerebras => {
                let opts: CerebrasEngineOptions = toml::Value::Table(self.options.clone())
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

/// Which engine `[engine] type` names.
///
/// Every name here is now offered: `Vulkan` and `Wgpu` were named-not-hosted
/// until their shells landed, and each is one `--features` flag away.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    /// Native CUDA engine — embedded as a static lib in `worker`
    /// (requires `--features cuda`).
    Cuda,
    /// Native MLX + Metal engine for Apple Silicon.
    Metal,
    /// Pure-Rust Vulkan engine: portable rather than vendor-specific, on
    /// whatever Vulkan 1.3 device the machine exposes.
    Vulkan,
    /// The WebGPU shell — one binary over Vulkan, Metal, D3D12 or WebGPU,
    /// whichever the machine has.
    Wgpu,
    /// The XLA shell: a PJRT plugin, libtpu on a TPU host.
    Xla,
    /// The Cerebras shell: the fabric simulator or a CS system reached by cmaddr.
    Cerebras,
}

impl EngineKind {
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::Cuda => "cuda",
            EngineKind::Metal => "metal",
            EngineKind::Vulkan => "vulkan",
            EngineKind::Wgpu => "wgpu",
            EngineKind::Xla => "xla",
            EngineKind::Cerebras => "cerebras",
        }
    }
}

fn default_tp_size() -> u32 {
    1
}
fn default_ep_size() -> u32 {
    1
}
fn default_activation_dtype() -> String {
    "bfloat16".to_string()
}
/// Accept either a single string or a list of strings.
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
source = "Qwen/Qwen3-0.6B"

[engine]
type = "metal"
device = ["metal:0"]
"#;

    /// A key no section declares is refused by name, rather than read as
    /// nothing.
    /// `--diag` writes `[engine] diagnostics`, and a flavor with no
    /// diagnostics record says so instead of swallowing the flag.
    /// The word list is the shell's, so this side keeps it whole rather than
    /// parsing it — a config that states one is the string it stated.
    /// Every engine kind's `as_str` is the word a config file spells it with;
    /// `as_str` is a hand-written match and nothing ties it to serde's
    /// `rename_all` if they drift.
    /// `[model.adapters]` absent is the shape hard-coded before it existed:
    /// zero seats, no roster, and the correction op never launches.
    /// The capacity is stated once: either the operator says it, or the
    /// roster does — one number, one owner.
    /// The two refusals this layer can make without a device.
    #[test]
    fn config_every_case() {
        an_unknown_key_is_refused();
        a_distributed_worker_states_no_cluster_policies();
        a_silence_timeout_under_the_submit_deadline_is_refused();
        the_diag_flag_states_the_engines_diagnostics_key();
        an_engine_s_own_key_is_written_beside_the_common_ones();
        parses_minimal_metal_config();
        every_engine_kind_round_trips_through_its_config_string();
        adapters_are_absent_by_default_and_that_is_zero_seats();
        the_seat_count_is_stated_once_or_derived_from_the_roster();
        an_adapter_past_its_seats_or_on_a_relative_path_is_refused_by_name();
    }

    fn an_unknown_key_is_refused() {
        for (section, key) in [
            ("server", "host = \"127.0.0.1\""),
            ("model", "model = \"x\""),
            ("runtime", "policy = \"noop\""),
            ("cache", "dir = \"/tmp/x\""),
        ] {
            let toml = format!("{MINIMAL_METAL}\n[{section}]\n{key}\n");
            assert!(
                Config::parse(&toml).is_err(),
                "[{section}] {key} should not parse"
            );
        }
    }

    fn a_distributed_worker_states_no_cluster_policies() {
        let toml = format!(
            "{MINIMAL_METAL}\n[cluster]\ncontroller = \"10.0.0.1:7000\"\nrole = \"decode\"\n\
             cluster_policies = [\"persist-prefix\"]\n"
        );
        let err = Config::parse(&toml).unwrap_err().to_string();
        assert!(err.contains("the controller's to run"), "got: {err}");
        let alone =
            format!("{MINIMAL_METAL}\n[cluster]\ncluster_policies = [\"persist-prefix\"]\n");
        Config::parse(&alone).expect("single-node runs them in its own controller");
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
            cfg.engine
                .options
                .get("diagnostics")
                .and_then(toml::Value::as_str),
            Some("tier-trace,kernel-profile=2"),
            "the words reach the engine's own options table"
        );

        let vulkan = MINIMAL_METAL
            .replace("type = \"metal\"", "type = \"vulkan\"")
            .replace("metal:0", "vulkan:0");
        let mut cfg: Config = toml::from_str(&vulkan).unwrap();
        let why = cfg
            .state_diagnostics("tier-trace")
            .expect_err("the vulkan shell has none");
        assert!(
            why.to_string().contains("vulkan"),
            "and the refusal names the flavor: {why}"
        );
    }

    fn an_engine_s_own_key_is_written_beside_the_common_ones() {
        let toml = MINIMAL_METAL.replace(
            "device = [\"metal:0\"]",
            "device = [\"metal:0\"]\ndiagnostics = \"cut-trace\"",
        );
        let cfg = Config::parse(&toml).expect("an engine key beside the common ones");
        assert_eq!(
            cfg.engine
                .options
                .get("diagnostics")
                .and_then(toml::Value::as_str),
            Some("cut-trace"),
            "it lands in the engine-specific options"
        );
        assert!(!cfg.engine.options.contains_key("type"));
    }

    fn parses_minimal_metal_config() {
        let cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.engine.kind, EngineKind::Metal);
        assert_eq!(cfg.engine.device, vec!["metal:0".to_string()]);
        assert_eq!(cfg.server.listen.port(), 8080);
    }

    fn every_engine_kind_round_trips_through_its_config_string() {
        const KINDS: &[(EngineKind, &str)] = &[
            (EngineKind::Cuda, "cuda"),
            (EngineKind::Metal, "metal"),
            (EngineKind::Vulkan, "vulkan"),
            (EngineKind::Wgpu, "wgpu"),
            (EngineKind::Xla, "xla"),
            (EngineKind::Cerebras, "cerebras"),
        ];
        for (kind, spelled) in KINDS {
            assert_eq!(kind.as_str(), *spelled, "{kind:?} names itself");
            let toml = format!(
                "[model]\nname = \"default\"\nsource = \"Qwen/Qwen3-0.6B\"\n\n\
                 [engine]\ntype = \"{spelled}\"\ndevice = [\"{spelled}:0\"]\n"
            );
            let cfg: Config = toml::from_str(&toml)
                .unwrap_or_else(|e| panic!("`type = \"{spelled}\"` does not parse: {e}"));
            assert_eq!(cfg.engine.kind, *kind);
            cfg.validate()
                .unwrap_or_else(|e| panic!("a minimal `{spelled}` config does not validate: {e}"));
            let round = toml::Value::try_from(*kind).expect("a kind serializes");
            assert_eq!(round.as_str(), Some(*spelled));
        }
        assert_eq!(
            KINDS.len(),
            6,
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
        let plane = if cfg!(windows) {
            "C:/adapters/0/a.bin"
        } else {
            "/adapters/0/a.bin"
        };
        let toml = MINIMAL_METAL.replace(
            "source = \"Qwen/Qwen3-0.6B\"",
            &format!(
                "source = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\n\
                 [[model.adapters.registered]]\n\
                 id = 2\n\
                 planes = {{ \"layer.0.lora_a\" = \"{plane}\" }}\n"
            ),
        );
        let cfg: Config = toml::from_str(&toml).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.adapters.seats(), 3);
        assert_eq!(cfg.model.adapters.registered.len(), 1);
        assert_eq!(
            cfg.model.adapters.registered[0].planes["layer.0.lora_a"],
            plane
        );

        let stated = toml.replace("[model.adapters]", "[model.adapters]\nseats = 8");
        let cfg: Config = toml::from_str(&stated).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.model.adapters.seats(), 8);
    }

    fn an_adapter_past_its_seats_or_on_a_relative_path_is_refused_by_name() {
        let past = MINIMAL_METAL.replace(
            "source = \"Qwen/Qwen3-0.6B\"",
            "source = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseats = 2\n\
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
            "source = \"Qwen/Qwen3-0.6B\"",
            "source = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\n\
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
            "source = \"Qwen/Qwen3-0.6B\"",
            "source = \"Qwen/Qwen3-0.6B\"\n\n[model.adapters]\nseat = 2\n",
        );
        toml::from_str::<Config>(&typo)
            .expect_err("a near-miss of `seats` must be refused by name");
    }
}
