//! The runtime inside a host process (the browser tab, an app): the host
//! opens its engine, [`load`] lands an artifact on it, [`start`] boots the
//! runtime, and clients trade `pie serve`'s MessagePack frames through
//! in-process sessions.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::bootstrap::BootstrapHandle;
use crate::engine::EngineBox;
use crate::model::ModelMetadata;
use crate::server::ClientId;

#[cfg(not(target_arch = "wasm32"))]
mod server;
#[cfg(not(target_arch = "wasm32"))]
pub use server::Server;

/// What an embedding host boots with; `gpu_mem_utilization` goes into the
/// host's engine boot document.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BootConfig {
    pub sku: Option<String>,
    pub engine: bool,
    pub max_forward_tokens: u32,
    pub max_forward_requests: u32,
    pub max_total_pages: u32,
    pub max_state_slots: u32,
    pub max_model_len: u32,
    pub gpu_mem_utilization: f32,
    pub sandbox_memory_mb: usize,
    pub max_concurrent_processes: Option<usize>,
    pub frame_size: u32,
    pub frame_dispatch_depth: u32,
    pub verbose: bool,
}

impl Default for BootConfig {
    fn default() -> Self {
        BootConfig {
            sku: None,
            engine: true,
            max_forward_tokens: 512,
            max_forward_requests: 8,
            max_total_pages: 512,
            max_state_slots: 64,
            max_model_len: 4096,
            gpu_mem_utilization: 0.9,
            sandbox_memory_mb: 512,
            max_concurrent_processes: None,
            frame_size: 8,
            frame_dispatch_depth: 2,
            verbose: false,
        }
    }
}

impl BootConfig {
    /// TOML text, or the same document as JSON (what a page hands over
    /// from an object).
    pub fn parse(text: &str) -> Result<Self> {
        if text.trim_start().starts_with('{') {
            return serde_json::from_str(text).context("parse the boot config");
        }
        toml::from_str(text).context("parse the boot config")
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BootSummary {
    pub model: String,
    pub sku: String,
    pub trace: String,
    pub weight_bytes: u64,
    pub kv_pages: u32,
    pub kv_page_size: u32,
    pub max_lanes: u32,
    pub max_tokens: u32,
}

/// Where a host keeps the runtime's files, and what it calls itself.
#[derive(Debug, Clone)]
pub struct Host {
    /// `host` in the runtime config and the telemetry service name.
    pub name: String,
    /// Holds `inferlets/`, `scratch/` and `languages/`.
    pub home: PathBuf,
    pub worker_threads: usize,
}

fn lift_metadata(artifact: &Path) -> Result<ModelMetadata> {
    use checkpoint::file::read::{parse_metadata, read_meta};

    const CONFIG_OBJECT: &str = "model/config";
    let checkpoint = parse_metadata(artifact)
        .map_err(|err| anyhow!("cannot read {}: {err}", artifact.display()))?;
    let config = read_meta(&checkpoint, CONFIG_OBJECT)?.ok_or_else(|| {
        anyhow!(
            "artifact {} carries no {CONFIG_OBJECT}; re-import it with `pie model import`",
            artifact.display()
        )
    })?;
    let mut objects = Vec::with_capacity(tokenizer::canonical::OBJECTS.len() + 1);
    for name in tokenizer::canonical::OBJECTS {
        let Some(bytes) = read_meta(&checkpoint, name)? else {
            objects.clear();
            break;
        };
        objects.push((name.to_string(), bytes));
    }
    if !objects.is_empty() {
        for name in tokenizer::canonical::OPTIONAL_OBJECTS {
            if let Some(bytes) = read_meta(&checkpoint, name)? {
                objects.push((name.to_string(), bytes));
            }
        }
    }
    Ok(ModelMetadata {
        tokenizer: (!objects.is_empty()).then_some(objects),
        config,
    })
}

/// An artifact landed on its engine, ready for [`start`].
pub struct Loaded {
    metadata: ModelMetadata,
    engines: Vec<crate::bootstrap::EngineConfig>,
    summary: BootSummary,
    model_id: String,
    kv_page_size: usize,
}

/// Lands `artifact` on `engine` (blocking). With no engine the runtime boots
/// without one, enough for programs that never run a forward pass.
pub fn load(
    config: &BootConfig,
    artifact: &Path,
    model_name: &str,
    engine: Option<(EngineBox, poem_ir::Platform)>,
) -> Result<Loaded> {
    let metadata = lift_metadata(artifact)?;
    let (engines, summary, model_id, kv_page_size) = match engine {
        Some((backend, platform)) => {
            let (engine, summary) = load_engine(config, artifact, model_name, backend, platform)?;
            let model_id = summary.trace.clone();
            let page = summary.kv_page_size as usize;
            (vec![engine], summary, model_id, page)
        }
        None => {
            let sku = checkpoint::file::serve::stamp_of(artifact)?
                .map(|stamp| stamp.sku)
                .ok_or_else(|| anyhow!("{} carries no serving stamp", artifact.display()))?;
            (
                Vec::new(),
                BootSummary {
                    model: model_name.to_string(),
                    sku: sku.clone(),
                    trace: sku.clone(),
                    weight_bytes: 0,
                    kv_pages: 0,
                    kv_page_size: 16,
                    max_lanes: 0,
                    max_tokens: 0,
                },
                sku,
                16,
            )
        }
    };
    Ok(Loaded {
        metadata,
        engines,
        summary,
        model_id,
        kv_page_size,
    })
}

fn load_engine(
    config: &BootConfig,
    artifact: &Path,
    model_name: &str,
    mut backend: EngineBox,
    platform: poem_ir::Platform,
) -> Result<(crate::bootstrap::EngineConfig, BootSummary)> {
    let budgets = ::engine::load::Budgets {
        max_lanes: config.max_forward_requests.max(1),
        max_tokens: config.max_forward_tokens.max(1),
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
        max_context: config.max_model_len.max(16),
        slots: config.max_state_slots.max(1),
        pages: config.max_total_pages.max(1),
        max_patches: None,
        max_images: None,
        max_voxels: None,
        max_clips: None,
    };
    let residency = ::engine::load::Residency::default();

    let frames_in_flight = u8::try_from(config.frame_dispatch_depth.max(1)).unwrap_or(u8::MAX);
    let request = crate::engine::load::request_of(
        config.sku.as_deref(),
        artifact,
        platform,
        budgets,
        residency,
        -1,
        frames_in_flight,
    )?;
    let sku = request.trace.name.clone();
    tracing::info!(sku, "loading weights");
    let loaded = backend.load(request).map_err(anyhow::Error::from)?;
    tracing::info!(
        weight_bytes = loaded.facts.weight_bytes,
        kv_pages = loaded.caps.pools.kv_pages,
        "weights resident"
    );

    let caps = loaded.caps;
    let summary = BootSummary {
        model: model_name.to_string(),
        sku: sku.clone(),
        trace: loaded.facts.trace_name.clone(),
        weight_bytes: loaded.facts.weight_bytes,
        kv_pages: caps.pools.kv_pages,
        kv_page_size: caps.pools.kv_page_size,
        max_lanes: caps.limits.max_lanes,
        max_tokens: caps.limits.max_tokens,
    };

    let engine = crate::bootstrap::EngineConfig {
        total_pages: caps.pools.kv_pages as usize,
        window_pages: caps.pools.window_pages,
        window_tokens: caps.pools.window_tokens,
        cpu_pages: 0,
        disk_pages: 0,
        cpu_window_pages: 0,
        kv_copy: caps.kv_copy,
        backend_kind: backend.kind().to_string(),
        rs_cache_required: caps.pools.state_slots != 0,
        rs_cache_slots: caps.pools.state_slots as usize,
        rs_cache_slot_bytes: caps.pools.state_slot_bytes,
        rs_host_slots: caps.pools.host_state_slots as usize,
        has_mtp_logits: caps.profile.has_mtp_logits,
        mtp_depth: caps.profile.mtp_depth,
        draft_block: caps.profile.draft_block,
        draft_mask_token: caps.profile.draft_mask_token,
        draft_bidirectional: caps.profile.draft_bidirectional,
        draft_proposals_from: caps.profile.draft_proposals_from,
        has_value_head: caps.profile.has_value_head,
        has_kv_envelopes: false,
        has_attn_page_mask: caps.profile.has_attn_page_mask,
        has_attn_score: caps.profile.has_attn_score,
        has_lora: caps.profile.has_lora,
        device_geometry_port_mask: caps.ports,
        limits: crate::engine::SchedulerLimits {
            max_forward_requests: caps.limits.max_lanes as usize,
            max_forward_tokens: caps.limits.max_tokens as usize,
            max_page_refs: caps.limits.max_page_refs as usize,
            max_context: caps.limits.max_context as usize,
        },
        engine_backend: backend,
    };
    Ok((engine, summary))
}

/// A booted runtime; dropping it leaves the runtime running. Bootstrap is
/// single-use per process.
pub struct Embedded {
    handle: BootstrapHandle,
    pub summary: BootSummary,
}

impl Embedded {
    pub async fn shutdown(self) -> Result<()> {
        self.handle.shutdown().await
    }
}

/// Boots the runtime over what [`load`] landed, registering `builtin_programs`.
pub async fn start(
    config: &BootConfig,
    host: &Host,
    artifact: PathBuf,
    loaded: Loaded,
    builtin_programs: Vec<crate::bootstrap::BuiltinProgram>,
) -> Result<Embedded> {
    let Loaded {
        metadata,
        engines,
        summary,
        model_id,
        kv_page_size,
    } = loaded;
    let boot = runtime_config(
        config,
        host,
        artifact,
        model_id,
        kv_page_size,
        metadata,
        engines,
        builtin_programs,
    );
    let handle = crate::bootstrap::bootstrap(boot).await?;
    Ok(Embedded { handle, summary })
}

#[allow(clippy::too_many_arguments)]
fn runtime_config(
    config: &BootConfig,
    host: &Host,
    artifact: PathBuf,
    model_id: String,
    kv_page_size: usize,
    metadata: ModelMetadata,
    engines: Vec<crate::bootstrap::EngineConfig>,
    builtin_programs: Vec<crate::bootstrap::BuiltinProgram>,
) -> crate::bootstrap::Config {
    crate::bootstrap::Config {
        host: host.name.clone(),
        port: 0,
        cache_dir: host.home.join("inferlets"),
        builtin_programs,
        verbose: config.verbose,
        log_dir: None,
        telemetry: crate::bootstrap::TelemetryConfig {
            enabled: false,
            endpoint: String::new(),
            service_name: host.name.clone(),
        },
        runtime: crate::bootstrap::RuntimeConfig {
            worker_threads: host.worker_threads,
            wasm_max_instances: 16,
            wasm_max_memory_mb: config.sandbox_memory_mb,
            wasm_warm_memory_mb: 0,
            wasm_warm_slots: 0,
            allow_fs: false,
            fs_scratch_dir: host.home.join("scratch"),
            allow_network: false,
            network_allowed_hosts: Vec::new(),
            max_upload_mb: 256,
            languages_dir: host.home.join("languages"),
            compile_cache_dir: None,
        },
        model: crate::bootstrap::ModelConfig {
            name: "default".into(),
            model_id,
            kv_page_size,
            tokenizer_path: artifact,
            metadata,
            engines,
            scheduler: crate::bootstrap::SchedulerConfig {
                submit_deadline_us: 50_000,
                silence_timeout_secs: 30,
                frame_size: config.frame_size,
                frame_dispatch_depth: config.frame_dispatch_depth,
            },
        },
        skip_tracing: true,
        max_concurrent_processes: config.max_concurrent_processes,
    }
}

/// Hands `session` one MessagePack `ClientMessage`.
pub fn send_frame(session: ClientId, frame: &[u8]) -> Result<()> {
    let message: client_api::ClientMessage =
        rmp_serde::from_slice(frame).map_err(|e| anyhow!("client frame: {e}"))?;
    crate::server::send_client_message(session, message)
}

/// Up to `max` server frames, waiting at most `max_wait_ms` for the first.
pub async fn recv_frames(session: ClientId, max_wait_ms: u64, max: usize) -> Result<Vec<Vec<u8>>> {
    let messages = crate::server::recv_messages(session, max_wait_ms, max).await?;
    messages
        .iter()
        .map(|message| rmp_serde::to_vec_named(message).map_err(anyhow::Error::from))
        .collect()
}
