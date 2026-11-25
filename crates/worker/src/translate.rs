use std::path::Path;

use anyhow::Result;

use crate::backend::ModelEngines;
use crate::config;

/// The page size the runtime keys on when no engine reports one.
pub const ENGINELESS_PAGE_SIZE: u32 = 16;

pub fn build(
    user: &config::Config,
    home: &Path,
    builtins: Vec<runtime::bootstrap::BuiltinProgram>,
    engines: ModelEngines,
    metadata: runtime::model::ModelMetadata,
) -> Result<runtime::bootstrap::Config> {
    if engines.groups.is_empty() {
        anyhow::bail!(
            "internal: model {:?} has zero native engines; \
             expected at least one engine per model",
            user.model.name,
        );
    }
    let model = build_model(user, engines, metadata)?;
    runtime_config(user, home, builtins, model)
}

/// A runtime with no engine over `artifact`, keyed by its stamped `deployment`.
pub fn build_without_engine(
    user: &config::Config,
    home: &Path,
    builtins: Vec<runtime::bootstrap::BuiltinProgram>,
    artifact: &Path,
    deployment: &str,
    metadata: runtime::model::ModelMetadata,
) -> Result<runtime::bootstrap::Config> {
    let model = runtime::bootstrap::ModelConfig {
        memory_yield: user.engine.memory.yields()?,
        name: user.model.name.clone(),
        model_id: deployment.to_string(),
        kv_page_size: ENGINELESS_PAGE_SIZE as usize,
        tokenizer_path: artifact.to_path_buf(),
        metadata,
        engines: Vec::new(),
        scheduler: scheduler(&user.runtime, 0),
        media_device: media_device(&user.engine),
    };
    runtime_config(user, home, builtins, model)
}

/// Every built-in inferlet, as `pie serve` registers them. The browser
/// registers none, so a wasm32 build carries none.
#[cfg(not(target_arch = "wasm32"))]
pub fn builtins() -> Vec<runtime::bootstrap::BuiltinProgram> {
    builtins::all()
        .iter()
        .map(|b| runtime::bootstrap::BuiltinProgram {
            name: b.name,
            component: b.component,
            source: b.source,
        })
        .collect()
}

fn runtime_config(
    user: &config::Config,
    home: &Path,
    builtin_programs: Vec<runtime::bootstrap::BuiltinProgram>,
    model: runtime::bootstrap::ModelConfig,
) -> Result<runtime::bootstrap::Config> {
    #[cfg(not(target_arch = "wasm32"))]
    let policy = {
        let spec = &user.runtime.worker_policy;
        Some((spec.clone(), builtins::policy_component(spec)?))
    };
    // The browser runs no policy.
    #[cfg(target_arch = "wasm32")]
    let policy = None;
    Ok(runtime::bootstrap::Config {
        host: user.server.listen.ip().to_string(),
        port: user.server.listen.port(),
        cache_dir: home.join("inferlets"),
        snapshots: Some((
            home.join("snapshots"),
            user.runtime.snapshot_budget.as_bytes(),
        )),
        builtin_programs,
        verbose: user.server.verbose,
        log_dir: Some(home.join("logs")),
        telemetry: runtime::bootstrap::TelemetryConfig {
            enabled: user.telemetry.endpoint.is_some(),
            endpoint: user.telemetry.endpoint.clone().unwrap_or_default(),
            service_name: user.telemetry.service_name.clone(),
        },
        runtime: runtime::bootstrap::RuntimeConfig {
            worker_threads: user.server.worker_threads,
            wasm_max_instances: user.sandbox.max_instances,
            wasm_max_memory_mb: user.sandbox.max_memory.as_mib() as usize,
            wasm_warm_memory_mb: user.sandbox.warm_memory.as_mib() as usize,
            wasm_warm_slots: user.sandbox.warm_slots,
            allow_fs: user.sandbox.allow_fs,
            fs_scratch_dir: user.sandbox.scratch_dir.clone(),
            allow_network: user.sandbox.allow_network,
            network_allowed_hosts: user.sandbox.network_allowed_hosts.clone(),
            max_upload_mb: user.server.max_upload.as_mib() as usize,
            allow_upload: user.server.allow_upload,
            allow_weight_reload: user.server.allow_weight_reload,
            languages_dir: home.join("languages"),
            compile_cache_dir: Some(home.join("cache").join("wasmtime")),
        },
        model,
        skip_tracing: true,
        max_concurrent_processes: user.runtime.max_concurrent_processes,
        policy,
    })
}

/// The runtime's scheduler knobs; `engine_depth`: the engine's preferred dispatch depth (0: none).
fn scheduler(
    runtime: &config::RuntimeConfig,
    engine_depth: u32,
) -> runtime::bootstrap::SchedulerConfig {
    let c = runtime::bootstrap::SchedulerConfig {
        frame_size: runtime.frame_size,
        frame_dispatch_depth: runtime.dispatch_depth(engine_depth),
    };
    tracing::info!(
        frame_size = c.frame_size,
        frame_dispatch_depth = c.frame_dispatch_depth,
        "runtime frames"
    );
    c
}

fn build_model(
    user: &config::Config,
    engines: ModelEngines,
    metadata: runtime::model::ModelMetadata,
) -> Result<runtime::bootstrap::ModelConfig> {
    let (m, runtime) = (&user.model, &user.runtime);
    let group0_caps = engines.groups[0].caps.clone();
    let snapshot_dir = engines.groups[0].snapshot_dir.clone();
    let engines_facts = engines.groups[0].facts.trace_name.clone();
    let tokenizer_path = if metadata.tokenizer.is_some() {
        snapshot_dir.clone()
    } else {
        let tokenizer_json = snapshot_dir.join("tokenizer.json");
        if tokenizer_json.exists() {
            tokenizer_json
        } else {
            snapshot_dir.join("tiktoken.model")
        }
    };

    let engines = engines
        .groups
        .into_iter()
        .enumerate()
        .map(|(group, g)| {
            let disk_kept = kept_disk(&user.engine.memory, group, &g);
            runtime::bootstrap::EngineConfig {
                chunks: g.caps.chunks.clone(),
                windowed_state: g.caps.windowed_state,
                disk_kept,
                draft_block: g.caps.draft_block,
                draft_mask_token: g.caps.draft_mask_token,
                draft_proposals_from: g.caps.draft_proposals_from,
                has_attn_page_mask: g.caps.has_attn_page_mask,
                has_lora: g.caps.has_lora,
                adapter_attachments: g.caps.adapter_attachments,
                rs_snapshots: g.caps.rs_snapshots,
                guest_cost: g.caps.guest_cost,
                frames_in_flight: g.caps.frames_in_flight,
                device_geometry_port_mask: g.caps.ports,
                host_ops: g.caps.host_ops,
                facts: g.caps.facts.clone(),
                inputs: g.caps.inputs.clone(),
                classes: g.caps.classes.clone(),
                limits: runtime::engine::SchedulerLimits {
                    max_forward_requests: g.caps.limits.max_lanes as usize,
                    max_forward_tokens: g.caps.limits.max_tokens as usize,
                    max_page_refs: g.caps.limits.max_page_refs as usize,
                    max_context: g.caps.limits.max_context as usize,
                    prefill_rows: g.caps.limits.prefill_rows as usize,
                    prefill_lanes: g.caps.limits.prefill_lanes as usize,
                    max_readouts: g.caps.limits.max_readouts as usize,
                    max_replayed_rows: g.caps.limits.replayed_rows as usize,
                    shapes: None,
                    readout_store: 0,
                },
                steps: g.caps.steps.clone(),
                mixed_lanes: g.caps.mixed_lanes.clone(),
                engine_backend: g.backend,
            }
        })
        .collect();

    Ok(runtime::bootstrap::ModelConfig {
        memory_yield: user.engine.memory.yields()?,
        name: m.name.clone(),
        model_id: engines_facts,
        kv_page_size: group0_caps.chunks.page_tokens() as usize,
        tokenizer_path,
        metadata,
        engines,
        scheduler: scheduler(runtime, group0_caps.limits.dispatch_depth),
        media_device: media_device(&user.engine),
    })
}

/// The CUDA device a media encode runs on: the cuda engine's first device
/// (`cuda:<n>`), else device 0.
fn media_device(engine: &config::EngineConfig) -> u32 {
    if engine.kind != config::EngineKind::Cuda {
        return 0;
    }
    engine
        .device
        .first()
        .and_then(|device| device.strip_prefix("cuda:"))
        .and_then(|index| index.parse().ok())
        .unwrap_or(0)
}

/// The disk tier `g` keeps for the next run, its journal beside its slot
/// files, when `[engine.memory] disk_persistent` asked for one and the
/// engine kept it.
fn kept_disk(
    memory: &config::MemoryConfig,
    group: usize,
    g: &crate::backend::GroupEngine,
) -> Option<runtime::bootstrap::KeptDisk> {
    let chunks = &g.caps.chunks;
    match (&g.disk_kv_dir, chunks.disk_persistent) {
        (Some(dir), true) => Some(runtime::bootstrap::KeptDisk {
            journal: dir.join(format!("g{group}.journal")),
            kept: chunks.disk_kept,
        }),
        _ => {
            if memory.disk_persistent {
                tracing::warn!(
                    engine = %g.backend.kind(),
                    disk_pages = chunks.kv_disk_pages(),
                    "disk_persistent: this engine keeps no disk tier across runs; it is scratch"
                );
            }
            None
        }
    }
}
