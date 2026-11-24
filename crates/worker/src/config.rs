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
        Self::parse_file(s, None)
    }

    /// [`Config::parse`] of the file at `path`: a key no section declares
    /// is refused by the file, line and dotted key, to be deleted.
    pub fn parse_file(s: &str, path: Option<&Path>) -> Result<Self> {
        let at = |line: Option<usize>| {
            let file = path.map_or_else(|| "config".to_string(), |p| p.display().to_string());
            line.map_or(file.clone(), |line| format!("{file}:{line}"))
        };
        let cfg: Config = toml::from_str(s).map_err(|e| match unknown_field(e.message()) {
            Some(field) => {
                let start = e.span().map(|span| span.start);
                let table = start.map(|start| table_at(s, start)).unwrap_or_default();
                let line = start.map(|start| s[..start].matches('\n').count() + 1);
                anyhow::anyhow!("{}: {}", at(line), unknown_key(&table, field))
            }
            None => anyhow::anyhow!("{}: {e}", at(None)),
        })?;
        cfg.validate()
            .map_err(|e| match e.downcast_ref::<UnknownKey>() {
                Some(UnknownKey { table, field }) => anyhow::anyhow!(
                    "{}: {}",
                    at(line_of(s, table, field)),
                    unknown_key(table, field)
                ),
                None => e,
            })?;
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
    /// `pie serve --no-measure`: the load times nothing (`[engine] measure
    /// = false`), refused by name on an engine that does not skip yet.
    pub fn skip_measuring(&mut self) -> Result<()> {
        self.engine.measure = false;
        self.engine.validate()
    }

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

    /// The most `[runtime] frame_dispatch_depth` an engine kind takes: a
    /// staging ring's bound ([`Runahead::MAX_FRAMES`]), or for the cerebras
    /// engine, which posts frames to a host queue, [`Runahead::UNSTAGED_MAX_FRAMES`].
    fn max_frames(kind: EngineKind) -> u8 {
        match kind {
            EngineKind::Cerebras => Runahead::UNSTAGED_MAX_FRAMES,
            _ => Runahead::MAX_FRAMES,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.engine.validate()?;
        self.server.validate()?;
        self.runtime.validate(Self::max_frames(self.engine.kind))?;
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
    /// Whether clients may install their own programs, each tenant under
    /// its own names. Off, a client runs only the system's: the built-ins
    /// and what `pie inferlet install` put here.
    #[serde(default = "default_true")]
    pub allow_upload: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            verbose: false,
            worker_threads: default_worker_threads(),
            max_upload: default_max_upload(),
            allow_upload: true,
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
// (24 of an H100 node's, or 48 of a dual-socket B200 NUMA node's) settles a
// decode fleet into lockstep, its guests trailing every frame: 7.5k tok/s
// against 18.7k at 12 on DeepSeek-V4's miniature, 64 lanes.
fn default_worker_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(1, 16))
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
    #[serde(default = "default_warm_memory")]
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
            warm_memory: default_warm_memory(),
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
    2048
}
fn default_max_memory() -> ByteSize {
    ByteSize::from_mib(4096)
}
fn default_warm_memory() -> ByteSize {
    ByteSize::from_mib(16)
}
fn default_warm_slots() -> u32 {
    1024
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

/// A key no section of the config declares: in `table` (dotted, empty at
/// the top), named `field`.
#[derive(Debug)]
struct UnknownKey {
    table: String,
    field: String,
}

impl std::fmt::Display for UnknownKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&unknown_key(&self.table, &self.field))
    }
}

impl std::error::Error for UnknownKey {}

/// The refusal of an unknown key, whatever key it is.
fn unknown_key(table: &str, field: &str) -> String {
    let key = if table.is_empty() {
        field.to_string()
    } else {
        format!("{table}.{field}")
    };
    format!("unknown key `{key}`; delete it (it isn't a setting)")
}

/// The field a deserializer's "unknown field" error names.
fn unknown_field(message: &str) -> Option<&str> {
    message.strip_prefix("unknown field `")?.split('`').next()
}

/// The table (dotted) a TOML document's byte `at` lies in: the last header
/// before it.
fn table_at(text: &str, at: usize) -> String {
    let mut table = String::new();
    for line in text[..at.min(text.len())].lines() {
        if let Some(header) = header(line) {
            table = header;
        }
    }
    table
}

/// The table a header line opens, `[a.b]` or `[[a.b]]`.
fn header(line: &str) -> Option<String> {
    let line = line.trim();
    let inner = line
        .strip_prefix("[[")
        .and_then(|l| l.strip_suffix("]]"))
        .or_else(|| line.strip_prefix('[').and_then(|l| l.strip_suffix(']')))?;
    Some(
        inner
            .split('.')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join("."),
    )
}

/// The line (from 1) where `field` is stated in `table`.
fn line_of(text: &str, table: &str, field: &str) -> Option<usize> {
    let mut current = String::new();
    for (at, line) in text.lines().enumerate() {
        if let Some(header) = header(line) {
            current = header;
            continue;
        }
        let stated = line.split('=').next().map(str::trim);
        if current == table && stated == Some(field) {
            return Some(at + 1);
        }
    }
    None
}

/// `[model] rope_scaling`: the YaRN a deployment stretches its model's
/// native context by.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RopeScalingConfig {
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
    /// Dtype weights are materialized in.
    #[serde(default = "default_weight_dtype")]
    pub weight_dtype: String,
    /// How many adapter seats this deployment intends to use, and which
    /// adapters to write into them at boot. Absent means zero seats.
    #[serde(default)]
    pub adapters: AdapterConfig,
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
        self.adapters.validate()?;
        Ok(())
    }
}

/// `[runtime]`. `runtime.submit_deadline` and `runtime.silence_timeout` were
/// removed (no lane is dropped from a frame, and no process terminated, on
/// a timer any more): a config that still carries either is refused at load
/// as an unknown field. To migrate, delete the line.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
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
    /// engine's fire lanes, which is what fills a batch.
    #[serde(default)]
    pub max_concurrent_processes: Option<usize>,
    /// The worker-plane PLEX 1.1 policy: a built-in policy's name (`noop`)
    /// or a component path; omit to run none. The cluster
    /// plane's are `[cluster] cluster_policies`.
    #[serde(default)]
    pub worker_policy: Option<String>,
    /// What handle snapshots may take on disk; past it a save is refused.
    #[serde(default = "default_snapshot_budget")]
    pub snapshot_budget: ByteSize,
}

fn default_snapshot_budget() -> ByteSize {
    ByteSize::from_mib(64 * 1024)
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            frame_size: default_frame_size(),
            frame_dispatch_depth: default_frame_dispatch_depth(),
            max_concurrent_processes: None,
            worker_policy: None,
            snapshot_budget: default_snapshot_budget(),
        }
    }
}

impl RuntimeConfig {
    fn validate(&self, max_frames: u8) -> Result<()> {
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
            self.frame_dispatch_depth <= u32::from(max_frames),
            "runtime.frame_dispatch_depth must be at most {} (got {}) on this engine: \
             an engine with a staging ring publishes its free set as one 64-bit \
             word, and `frames_in_flight * {} + 1` must fit in it \
             (`engine::runahead::Runahead::MAX_FRAMES`; the cerebras engine, \
             which has none, takes up to `Runahead::UNSTAGED_MAX_FRAMES`)",
            max_frames,
            self.frame_dispatch_depth,
            Runahead::STEPS_MAX
        );
        if let Some(n) = self.max_concurrent_processes {
            ensure!(n > 0, "runtime.max_concurrent_processes must be > 0 if set");
        }
        Ok(())
    }
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
    /// The longest sequence (prompt + output) one lane holds, in tokens: a
    /// product limit, not a memory one. Omit for the model's own context,
    /// shortened to what the device holds where one sequence at it would
    /// not fit.
    #[serde(default)]
    pub max_context: Option<u32>,
    /// Whether a load times the device measurements no stored or committed
    /// profile states under a matching key (its rates, each dense kernel's
    /// curve at the model's planes, the dense settle), and keeps them for
    /// the next boot. `false` times nothing: each is served by the engine's
    /// estimate, named in the load log, and nothing is stored (`pie serve
    /// --no-measure`). Vulkan, wgpu, and a one-device cuda engine (CUDA or
    /// ROCm); xla, whose load times nothing on the device, states it to the
    /// runtime, which estimates the host cover rather than timing it.
    #[serde(default = "default_measure")]
    pub measure: bool,
    /// `[engine.memory]`: the optional caps for sharing a machine.
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Engine-specific knobs, written beside the common keys. Embedded
    /// engines parse this into typed option structs.
    #[serde(flatten)]
    pub options: toml::Table,
}

/// `[engine.memory]`: what a load may take of the machine. Memory sizing
/// reads none of these by default: the engine measures the device and
/// uses all of it, and how it is split between weights, state and fires
/// follows demand at run time. An empty table is the best configuration
/// for a dedicated machine; each key is a cap for sharing one.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    /// The most device memory this load takes (`"20GiB"`), per device, for
    /// a device shared with another process. Omit to use the whole device.
    pub budget_device: Option<ByteSize>,
    /// The tokens of kv the device holds (`3248 * page size`), for a run
    /// that must compare at a fixed cache: the device pool holds exactly
    /// the pages they fill, beside the weights and the shapes it always
    /// holds, and the load is refused where `budget_device` (or the whole
    /// device) cannot hold that. Omit to size the cache by what the device
    /// has. cuda only.
    pub kv_device_tokens: Option<u64>,
    /// The most pinned host memory suspended kv pages and state slots may
    /// take (`"32GiB"`), shared by the tensor-parallel ranks. Omit for every
    /// seat at full context, within half of what the host has free (so
    /// workers sharing a host each find room). pie cannot compute kv again,
    /// so this is where memory contention goes: it cannot be turned off.
    pub budget_host: Option<ByteSize>,
    /// The disk tier: the most disk suspended kv may spill into once the
    /// host tier is full (`"200GiB"`), shared by the ranks, in slot files
    /// under `disk_path`. Omitted or 0: no disk tier (the tiers are the
    /// device and, where there is one, the host). Nothing sizes it unstated.
    pub budget_disk: Option<ByteSize>,
    /// Where the disk tier's slot files go. Omit for
    /// `$PIE_HOME/cache/kv/<checkpoint digest>`.
    pub disk_path: Option<PathBuf>,
    /// Keeps the disk tier for the next run: its slot files stay under
    /// `disk_path` with a journal of the sealed pages they hold, and a
    /// restart takes those pages back as cache. Off by default; cuda only,
    /// and not for a model with recurrent state or a window.
    pub disk_persistent: bool,
    /// How much of the device budget this load gives back to the system's
    /// memory pressure, in `[0, 1]`: the device target is
    /// `floor + (budget - floor) * (1 - yield * pressure)`, with pressure
    /// the OS's own reading in `0..1` (Linux and Android PSI, macOS's
    /// available memory). `0` (a dedicated machine) holds the whole budget
    /// and reads nothing; above 0, a load on a system whose pressure cannot
    /// be read is refused by name. The floor is what the load promised: one
    /// lane at the full `max_context`, the reserved chunks, the activation
    /// span and the dense weights, so a long `max_context` raises the floor
    /// and with it lowers how far yield can press. Once pressure goes, the
    /// pool grows back only on demand, up to the target: nothing is
    /// allocated for nobody. At `1` the pool trims under moderate pressure
    /// too, by design; `0` never trims. Omitted, it is `1` where the engine
    /// states unified memory (Metal, phones, DGX Spark: the pool and the
    /// host share one RAM) and `0` on separate device memory.
    ///
    /// Idle cache is not timed out by the runtime: with no worker policy
    /// installed it stays until a demand needs its room, and is then
    /// reclaimed in retention-key, then least-recently-used, order. How
    /// long it is kept past that is the policy's to state, with its
    /// retention keys.
    #[serde(rename = "yield")]
    pub yield_share: Option<f64>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            budget_device: None,
            kv_device_tokens: None,
            budget_host: None,
            budget_disk: None,
            disk_path: None,
            disk_persistent: false,
            yield_share: None,
        }
    }
}

impl MemoryConfig {
    /// What a load may take of the machine, in the form the engine's load
    /// contract states it. The weights start held (every streamable atom
    /// resident where the pool seats them all); live demand may take them
    /// off the device later.
    #[must_use]
    pub fn residency(&self) -> engine::Residency {
        engine::Residency {
            device: engine::Allowance {
                budget: self.budget_device.map(|b| b.as_bytes()),
            },
            device_kv_budget: None,
            device_kv_tokens: self.kv_device_tokens,
            host_kv_budget: self.budget_host.map(|b| b.as_bytes()),
            kv_seats: None,
            disk_kv_budget: self.budget_disk.map(|b| b.as_bytes()).filter(|&b| b > 0),
            disk_kv_dir: self.disk_path.clone(),
            disk_kv_persistent: self.disk_persistent,
            kv_export: false,
            host_programs: None,
            weights_held: true,
        }
    }

    /// How much of the device budget this load gives back to the system's
    /// memory pressure; `None` where the deployment states none, for the
    /// load to resolve by the engines' memory.
    ///
    /// # Errors
    ///
    /// `yield` is outside `[0, 1]`, refused by name.
    pub fn yields(&self) -> Result<Option<runtime::governor::Yield>> {
        Ok(self
            .yield_share
            .map(runtime::governor::Yield::new)
            .transpose()?)
    }

    fn validate(&self) -> Result<()> {
        self.yields()?;
        ensure!(
            self.kv_device_tokens != Some(0),
            "engine.memory.kv_device_tokens = 0 holds no kv; state the tokens the device \
             holds or omit the key"
        );
        let disk = self.budget_disk.is_some_and(|b| b.as_bytes() > 0);
        ensure!(
            disk || (self.disk_path.is_none() && !self.disk_persistent),
            "engine.memory.disk_path or disk_persistent names a disk tier, but budget_disk \
             states none (omitted or 0 is no disk tier); state budget_disk or remove them"
        );
        for (key, budget) in [
            ("budget_device", self.budget_device),
            ("budget_host", self.budget_host),
        ] {
            ensure!(
                budget.is_none_or(|b| b.as_bytes() > 0),
                "engine.memory.{key} is zero, and a load needs some of every tier it \
                 has; state a real budget or omit the key"
            );
        }
        Ok(())
    }
}

impl EngineConfig {
    /// The named engine's own keys as its options struct.
    ///
    /// # Errors
    ///
    /// A key the engine does not declare, or a value of the wrong type.
    pub fn options_as<T: serde::de::DeserializeOwned>(&self) -> Result<T> {
        toml::Value::Table(self.options.clone())
            .try_into()
            .map_err(|e: toml::de::Error| match unknown_field(e.message()) {
                Some(field) => anyhow::Error::new(UnknownKey {
                    table: "engine".to_string(),
                    field: field.to_string(),
                }),
                None => anyhow::anyhow!(
                    "invalid [engine] options for engine type {:?}: {e}",
                    self.kind,
                ),
            })
    }

    fn validate(&self) -> Result<()> {
        ensure!(!self.device.is_empty(), "engine.device must be non-empty");
        ensure!(
            self.max_context != Some(0),
            "engine.max_context = 0 holds no sequence; omit it for the model's own context"
        );
        self.memory.validate()?;
        ensure!(
            self.measure
                || matches!(
                    self.kind,
                    EngineKind::Vulkan | EngineKind::Wgpu | EngineKind::Metal | EngineKind::Xla
                )
                || (self.kind == EngineKind::Cuda && self.device.len() == 1),
            "engine.measure = false skips the measurements a load takes on the vulkan, wgpu, \
             metal and xla engines and a one-device cuda engine; the {} engine{} still measures outside \
             that path (new-memory/FOLLOWUPS.md), so omit the key",
            self.kind.as_str(),
            if self.kind == EngineKind::Cuda {
                " across devices"
            } else {
                ""
            }
        );
        ensure!(
            self.memory.kv_device_tokens.is_none() || self.kind == EngineKind::Cuda,
            "engine.memory.kv_device_tokens sizes the cuda engine's device pool; the {} engine \
             does not size its pool by tokens, so omit the key",
            self.kind.as_str()
        );
        match self.kind {
            EngineKind::Cuda => {
                self.options_as::<CudaEngineOptions>()?;
            }
            EngineKind::Metal => {
                self.options_as::<MetalEngineOptions>()?;
            }
            EngineKind::Vulkan => {
                self.options_as::<VulkanEngineOptions>()?;
            }
            EngineKind::Wgpu => self.options_as::<WgpuEngineOptions>()?.validate()?,
            EngineKind::Xla => {
                self.options_as::<XlaEngineOptions>()?;
            }
            EngineKind::Cerebras => self.options_as::<CerebrasEngineOptions>()?.validate()?,
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
fn default_measure() -> bool {
    true
}
fn default_ep_size() -> u32 {
    1
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
        a_submit_deadline_is_an_unknown_key();
        the_diag_flag_states_the_engines_diagnostics_key();
        measuring_is_skipped_where_the_engine_skips_and_refused_elsewhere();
        an_engine_s_own_key_is_written_beside_the_common_ones();
        parses_minimal_metal_config();
        every_engine_kind_round_trips_through_its_config_string();
        adapters_are_absent_by_default_and_that_is_zero_seats();
        the_seat_count_is_stated_once_or_derived_from_the_roster();
        an_adapter_past_its_seats_or_on_a_relative_path_is_refused_by_name();
        an_empty_memory_table_is_the_dedicated_machine();
        the_memory_keys_are_seven_and_prefixed();
        the_dispatch_depth_is_bounded_by_the_engine_kind();
        activation_dtype_is_an_unknown_key();
    }

    fn an_empty_memory_table_is_the_dedicated_machine() {
        let cfg = Config::parse(MINIMAL_METAL).unwrap();
        assert_eq!(cfg.engine.memory, MemoryConfig::default());
        assert_eq!(
            cfg.engine.memory.yields().unwrap(),
            None,
            "unset: the load resolves it by the engines' memory"
        );
        let zero = format!("{MINIMAL_METAL}\n[engine.memory]\nyield = 0\n");
        assert_eq!(
            Config::parse(&zero).unwrap().engine.memory.yields().unwrap(),
            Some(runtime::governor::Yield::NONE),
            "a configured 0 is not unset"
        );
        let residency = cfg.engine.memory.residency();
        assert_eq!(residency.device, engine::Allowance::WHOLE);
        assert!(residency.weights_held);

        let shared = format!(
            "{MINIMAL_METAL}\n[engine.memory]\nbudget_device = \"20GiB\"\n\
             budget_host = \"8GiB\"\nyield = 0.5\n"
        );
        let cfg = Config::parse(&shared).unwrap();
        assert_eq!(
            cfg.engine.memory.residency().device.budget,
            Some(20 << 30),
            "budget_device caps the device"
        );
        assert_eq!(cfg.engine.memory.residency().host_kv_budget, Some(8 << 30));
        assert_eq!(cfg.engine.memory.yields().unwrap().map(|y| y.share()), Some(0.5));

        for bad in ["yield = -0.1", "yield = 1.5", "yield = nan"] {
            let toml = format!("{MINIMAL_METAL}\n[engine.memory]\n{bad}\n");
            let refused = Config::parse(&toml).expect_err(bad);
            assert!(
                format!("{refused:#}").contains("engine.memory.yield"),
                "{bad}: {refused:#}"
            );
        }
        let zero = format!("{MINIMAL_METAL}\n[engine.memory]\nbudget_device = 0\n");
        assert!(Config::parse(&zero).is_err(), "a zero budget is refused");
        // The knobs `yield` replaced are no key at all.
        for old in ["governor_backoff = 0.8", "governor_release_idle = true"] {
            let toml = format!("{MINIMAL_METAL}\n[engine.memory]\n{old}\n");
            let refused = Config::parse(&toml).expect_err(old);
            assert!(
                format!("{refused:#}").contains("unknown key"),
                "{old}: {refused:#}"
            );
        }

        // The kv tokens reach the cuda engine's load; another engine, which
        // does not size its pool by them, refuses the key by name.
        let cuda = MINIMAL_METAL
            .replace("\"metal\"", "\"cuda\"")
            .replace("metal:0", "cuda:0");
        let tokens = format!("{cuda}\n[engine.memory]\nkv_device_tokens = 51968\n");
        let cfg = Config::parse(&tokens).unwrap();
        assert_eq!(cfg.engine.memory.residency().device_kv_tokens, Some(51968));
        let metal = format!("{MINIMAL_METAL}\n[engine.memory]\nkv_device_tokens = 51968\n");
        let refused = Config::parse(&metal).expect_err("metal refuses the key");
        assert!(
            format!("{refused:#}").contains("kv_device_tokens"),
            "{refused:#}"
        );
        let zero = format!("{cuda}\n[engine.memory]\nkv_device_tokens = 0\n");
        assert!(Config::parse(&zero).is_err(), "zero tokens should be refused");
    }

    /// The disk tier is what `budget_disk` states: omitted or 0 is no disk
    /// tier (nothing sizes one unstated), a positive budget is its size, and
    /// a disk path or persistence with no tier is refused by name.
    #[test]
    fn the_disk_tier_is_stated_config_only() {
        let unset = Config::parse(MINIMAL_METAL).unwrap();
        assert_eq!(unset.engine.memory.residency().disk_kv_budget, None, "no disk tier");
        let zero = format!("{MINIMAL_METAL}\n[engine.memory]\nbudget_disk = \"0B\"\n");
        let cfg = Config::parse(&zero).expect("0 is no disk tier, not refused");
        assert_eq!(cfg.engine.memory.residency().disk_kv_budget, None);
        let sized = format!("{MINIMAL_METAL}\n[engine.memory]\nbudget_disk = \"2GiB\"\n");
        let cfg = Config::parse(&sized).unwrap();
        assert_eq!(cfg.engine.memory.residency().disk_kv_budget, Some(2 << 30));
        for stray in ["disk_path = \"/nvme/pie\"", "disk_persistent = true"] {
            for budget in ["", "budget_disk = \"0B\"\n"] {
                let toml = format!("{MINIMAL_METAL}\n[engine.memory]\n{budget}{stray}\n");
                let refused = Config::parse(&toml).expect_err(stray);
                assert!(
                    format!("{refused:#}").contains("budget_disk states none"),
                    "{stray}: {refused:#}"
                );
            }
        }
    }

    fn the_memory_keys_are_seven_and_prefixed() {
        let keys: Vec<String> = schema::fields(EngineKind::Cuda)
            .into_iter()
            .map(|field| field.key)
            .filter(|key| key.starts_with("engine.memory."))
            .collect();
        assert_eq!(
            keys,
            [
                "engine.memory.budget_device",
                "engine.memory.kv_device_tokens",
                "engine.memory.budget_host",
                "engine.memory.budget_disk",
                "engine.memory.disk_path",
                "engine.memory.disk_persistent",
                "engine.memory.yield",
            ]
        );
    }

    /// A staged engine keeps the 64-bit ring's 15 frames; only the cerebras
    /// shell, which posts frames to a host queue, takes more.
    /// Nothing reads a compute dtype from the config: a file that states
    /// one is refused as it would be for any key no struct declares.
    fn activation_dtype_is_an_unknown_key() {
        let file = std::path::Path::new("/etc/pie/config.toml");
        // An engine's own key, read through its options struct: lines 6-9
        // are `[engine]`, `type`, `device`, the dropped key.
        let err = Config::parse_file(
            &MINIMAL_METAL.replace(
                "device = [\"metal:0\"]",
                "device = [\"metal:0\"]\nactivation_dtype = \"bfloat16\"",
            ),
            Some(file),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "/etc/pie/config.toml:9: unknown key `engine.activation_dtype`; delete it (it \
             isn't a setting)"
        );
        // Any other section's: the deserializer's own span.
        let err = Config::parse_file(
            &MINIMAL_METAL.replace(
                "source = \"Qwen/Qwen3-0.6B\"",
                "source = \"Qwen/Qwen3-0.6B\"\nmax_contxt = 8",
            ),
            Some(file),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "/etc/pie/config.toml:5: unknown key `model.max_contxt`; delete it (it isn't a \
             setting)"
        );
        let err = Config::parse(&MINIMAL_METAL.replace(
            "[engine]",
            "[engine.memory]\nbudget_dev = \"1GiB\"\n[engine]",
        ))
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("config:7: unknown key `engine.memory.budget_dev`"),
            "{err}"
        );
        Config::parse(MINIMAL_METAL).expect("without it, the file loads");
    }

    fn the_dispatch_depth_is_bounded_by_the_engine_kind() {
        for (depth, ok) in [(15, true), (16, false)] {
            let toml = format!("{MINIMAL_METAL}\n[runtime]\nframe_dispatch_depth = {depth}\n");
            let cfg: Config = toml::from_str(&toml).unwrap();
            assert_eq!(cfg.validate().is_ok(), ok, "metal at depth {depth}");
        }
        assert_eq!(
            Config::max_frames(EngineKind::Cerebras),
            Runahead::UNSTAGED_MAX_FRAMES
        );
        for kind in [
            EngineKind::Cuda,
            EngineKind::Metal,
            EngineKind::Vulkan,
            EngineKind::Wgpu,
            EngineKind::Xla,
        ] {
            assert_eq!(Config::max_frames(kind), Runahead::MAX_FRAMES, "{kind:?}");
        }
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

    /// The leash is gone, and its key with it: a config that sets it is
    /// refused by name.
    fn a_submit_deadline_is_an_unknown_key() {
        let toml = format!("{MINIMAL_METAL}\n[runtime]\nsubmit_deadline = \"5s\"\n");
        let err = toml::from_str::<Config>(&toml).unwrap_err().to_string();
        assert!(
            err.contains("unknown field `submit_deadline`"),
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

    fn measuring_is_skipped_where_the_engine_skips_and_refused_elsewhere() {
        let vulkan = MINIMAL_METAL
            .replace("type = \"metal\"", "type = \"vulkan\"")
            .replace("metal:0", "vulkan:0");
        let cfg: Config = toml::from_str(&vulkan).unwrap();
        assert!(cfg.engine.measure, "a load measures unless told not to");
        let skipping = vulkan.replace(
            "device = [\"vulkan:0\"]",
            "device = [\"vulkan:0\"]\nmeasure = false",
        );
        let cfg: Config = toml::from_str(&skipping).unwrap();
        assert!(!cfg.engine.measure);
        cfg.engine.validate().expect("vulkan skips");
        assert!(
            cfg.engine.options.get("measure").is_none(),
            "a common key, not the engine's own"
        );
        // `pie serve --no-measure`.
        let mut cfg: Config = toml::from_str(&vulkan).unwrap();
        cfg.skip_measuring().expect("vulkan skips");
        assert!(!cfg.engine.measure);
        let mut cfg: Config = toml::from_str(MINIMAL_METAL).unwrap();
        cfg.skip_measuring().expect("metal skips");
        assert!(!cfg.engine.measure);
        let xla = MINIMAL_METAL
            .replace("type = \"metal\"", "type = \"xla\"")
            .replace("metal:0", "tpu:0");
        let mut cfg: Config = toml::from_str(&xla).unwrap();
        cfg.skip_measuring().expect("xla skips");
        assert!(!cfg.engine.measure);
        let cuda = MINIMAL_METAL
            .replace("type = \"metal\"", "type = \"cuda\"")
            .replace("metal:0", "cuda:0");
        let mut cfg: Config = toml::from_str(&cuda).unwrap();
        cfg.skip_measuring().expect("a one-device cuda engine skips");
        assert!(!cfg.engine.measure);
        // Still refused by name: a cuda engine over more than one device (a
        // tensor-parallel group or a lockstep, whose ranks still measure
        // outside the keyed path).
        let group = cuda.replace("[\"cuda:0\"]", "[\"cuda:0\", \"cuda:1\"]");
        let mut cfg: Config = toml::from_str(&group).unwrap();
        let why = cfg
            .skip_measuring()
            .expect_err("a cuda group measures outside the skip path yet");
        assert!(
            why.to_string().contains("the cuda engine across devices still measures"),
            "{why}"
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
