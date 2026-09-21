use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

use serde::{Deserialize, Serialize};

use runtime::model::ModelMetadata;

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
    pub device_memory_mb: Option<u64>,
    pub power_preference: String,
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
            device_memory_mb: None,
            power_preference: "high-performance".into(),
            sandbox_memory_mb: 512,
            max_concurrent_processes: None,
            frame_size: 8,
            frame_dispatch_depth: 2,
            verbose: false,
        }
    }
}

impl BootConfig {
    pub fn parse(toml_text: &str) -> Result<Self> {
        toml::from_str(toml_text).context("parse the boot config")
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

pub const MODELS_DIR: &str = "/models";

pub struct Mount {
    pub len: u64,
    pub fetch: Box<dyn ztensor::memfs::Fetch>,
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

pub async fn boot(
    config: BootConfig,
    model_name: &str,
    mount: Mount,
    device: Option<crate::engine::Device>,
) -> Result<BootSummary> {
    let artifact = PathBuf::from(MODELS_DIR).join(model_name);
    let Mount { len, fetch } = mount;
    ztensor::memfs::mount_lazy(&artifact, len, fetch);
    tracing::info!(
        path = %artifact.display(),
        len,
        window = ztensor::memfs::LAZY_WINDOW,
        "artifact mounted lazily"
    );

    let loaded = {
        let config = config.clone();
        let artifact = artifact.clone();
        let model_name = model_name.to_string();
        on_green_thread("load", move || {
            load(&config, &artifact, &model_name, device)
        })
        .await?
    };
    let Loaded {
        metadata,
        engines,
        summary,
        model_id,
        kv_page_size,
    } = loaded;
    if let Some(stats) = ztensor::memfs::lazy_stats(&artifact) {
        tracing::info!(
            requests = stats.requests,
            mib = stats.bytes >> 20,
            "artifact ranges fetched for the boot"
        );
    }

    // Freed before the runtime spawns its fibers: wasm-bindgen's glue writes
    // import results through signed i32 offsets, so a fiber stack above 2 GiB
    // breaks every string-returning import.
    if let Some(chunks) = ztensor::memfs::unmount(&artifact) {
        let held: Vec<usize> = chunks
            .parts()
            .iter()
            .map(|part| std::sync::Arc::strong_count(part) - 1)
            .collect();
        tracing::info!(
            ?held,
            lazy = chunks.is_lazy(),
            "artifact unmounted (references left per chunk)"
        );
    }
    let probe = vec![0u8; 1 << 20];
    tracing::info!(
        probe_mib = (probe.as_ptr() as usize) >> 20,
        "a fresh allocation lands here after the unmount"
    );
    drop(probe);

    let boot = runtime_config(
        &config,
        artifact.clone(),
        model_id,
        kv_page_size,
        metadata,
        engines,
    );
    let handle = runtime::bootstrap::bootstrap(boot).await?;
    std::mem::forget(handle);
    Ok(summary)
}

struct Loaded {
    metadata: ModelMetadata,
    engines: Vec<runtime::bootstrap::EngineConfig>,
    summary: BootSummary,
    model_id: String,
    kv_page_size: usize,
}

fn load(
    config: &BootConfig,
    artifact: &Path,
    model_name: &str,
    device: Option<crate::engine::Device>,
) -> Result<Loaded> {
    let metadata = lift_metadata(artifact)?;
    let (engines, summary, model_id, kv_page_size) = match device {
        Some(device) => {
            let (engine, summary) = load_engine(config, artifact, model_name, device)?;
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

#[cfg(target_arch = "wasm32")]
const LOAD_STACK: usize = 8 << 20;

#[cfg(target_arch = "wasm32")]
async fn on_green_thread<T: 'static>(
    name: &str,
    f: impl FnOnce() -> Result<T> + 'static,
) -> Result<T> {
    web_std::thread::Builder::new()
        .name(name.to_string())
        .stack_size(LOAD_STACK)
        .spawn(f)
        .with_context(|| format!("spawn the {name} thread"))?
        .await
}

#[cfg(not(target_arch = "wasm32"))]
async fn on_green_thread<T: 'static>(
    _name: &str,
    f: impl FnOnce() -> Result<T> + 'static,
) -> Result<T> {
    f()
}

fn load_engine(
    config: &BootConfig,
    artifact: &Path,
    model_name: &str,
    device: crate::engine::Device,
) -> Result<(runtime::bootstrap::EngineConfig, BootSummary)> {
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

    let mut backend = crate::engine::open(config, device)?;
    let frames_in_flight = u8::try_from(config.frame_dispatch_depth.max(1)).unwrap_or(u8::MAX);
    let request = runtime::engine::load::request_of(
        config.sku.as_deref(),
        artifact,
        model_ir::Platform::Wgpu,
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

    let engine = runtime::bootstrap::EngineConfig {
        total_pages: caps.pools.kv_pages as usize,
        cpu_pages: 0,
        kv_copy: caps.kv_copy,
        backend_kind: backend.kind().to_string(),
        rs_cache_required: caps.pools.state_slots != 0,
        rs_cache_slots: caps.pools.state_slots as usize,
        rs_cache_slot_bytes: caps.pools.state_slot_bytes,
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
        limits: runtime::engine::SchedulerLimits {
            max_forward_requests: caps.limits.max_lanes as usize,
            max_forward_tokens: caps.limits.max_tokens as usize,
            max_page_refs: caps.limits.max_page_refs as usize,
            max_context: caps.limits.max_context as usize,
        },
        engine_backend: backend,
    };
    Ok((engine, summary))
}

fn runtime_config(
    config: &BootConfig,
    artifact: PathBuf,
    model_id: String,
    kv_page_size: usize,
    metadata: ModelMetadata,
    engines: Vec<runtime::bootstrap::EngineConfig>,
) -> runtime::bootstrap::Config {
    runtime::bootstrap::Config {
        host: "browser".into(),
        port: 0,
        cache_dir: PathBuf::from("/programs"),
        verbose: config.verbose,
        log_dir: None,
        registry_url: "https://registry.pie-project.org/".into(),
        telemetry: runtime::bootstrap::TelemetryConfig {
            enabled: false,
            endpoint: String::new(),
            service_name: "pie-web".into(),
        },
        runtime: runtime::bootstrap::RuntimeConfig {
            worker_threads: 1,
            wasm_max_instances: 16,
            wasm_max_memory_mb: config.sandbox_memory_mb,
            wasm_warm_memory_mb: 0,
            wasm_warm_slots: 0,
            allow_fs: false,
            fs_scratch_dir: PathBuf::from("/scratch"),
            allow_network: false,
            network_allowed_hosts: Vec::new(),
            max_upload_mb: 256,
            py_runtime_dir: PathBuf::from("/py-runtime"),
        },
        model: runtime::bootstrap::ModelConfig {
            name: "default".into(),
            model_id,
            kv_page_size,
            tokenizer_path: artifact,
            metadata,
            engines,
            scheduler: runtime::bootstrap::SchedulerConfig {
                submit_deadline_us: 50_000,
                silence_timeout_secs: 30,
                frame_size: config.frame_size,
                frame_dispatch_depth: config.frame_dispatch_depth,
            },
        },
        skip_tracing: true,
        max_concurrent_processes: config.max_concurrent_processes,
        python_snapshot: false,
    }
}
