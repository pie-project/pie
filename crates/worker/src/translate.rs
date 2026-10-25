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
    let model = build_model(&user.model, &user.runtime, engines, metadata)?;
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
        name: user.model.name.clone(),
        model_id: deployment.to_string(),
        kv_page_size: ENGINELESS_PAGE_SIZE as usize,
        tokenizer_path: artifact.to_path_buf(),
        metadata,
        engines: Vec::new(),
        scheduler: scheduler(&user.runtime),
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
            version: b.version,
            component: b.component,
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
    let policy = match user.runtime.worker_policy.as_deref() {
        Some(spec) => Some((spec.to_string(), builtins::policy_component(spec)?)),
        None => None,
    };
    // The browser runs no policy.
    #[cfg(target_arch = "wasm32")]
    let policy = None;
    Ok(runtime::bootstrap::Config {
        host: user.server.listen.ip().to_string(),
        port: user.server.listen.port(),
        cache_dir: home.join("inferlets"),
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
            languages_dir: home.join("languages"),
            compile_cache_dir: Some(home.join("cache").join("wasmtime")),
        },
        model,
        skip_tracing: true,
        max_concurrent_processes: user.runtime.max_concurrent_processes,
        policy,
    })
}

fn scheduler(runtime: &config::RuntimeConfig) -> runtime::bootstrap::SchedulerConfig {
    runtime::bootstrap::SchedulerConfig {
        submit_deadline_us: runtime.submit_deadline.as_micros(),
        silence_timeout_secs: runtime.silence_timeout.as_secs(),
        frame_size: runtime.frame_size,
        frame_dispatch_depth: runtime.frame_dispatch_depth,
    }
}

fn build_model(
    m: &config::ModelConfig,
    runtime: &config::RuntimeConfig,
    engines: ModelEngines,
    metadata: runtime::model::ModelMetadata,
) -> Result<runtime::bootstrap::ModelConfig> {
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
            let backend_kind = g.backend.kind().to_string();
            let disk_kept = kept_disk(m, group, &g);
            runtime::bootstrap::EngineConfig {
                total_pages: g.caps.pools.kv_pages as usize,
                window_pages: g.caps.pools.window_pages,
                window_tokens: g.caps.pools.window_tokens,
                cpu_pages: g.caps.pools.host_kv_pages as usize,
                cpu_window_pages: g.caps.pools.host_window_pages,
                disk_pages: g.caps.pools.disk_kv_pages,
                disk_kept,
                kv_copy: g.caps.kv_copy,
                backend_kind,
                rs_cache_required: g.caps.pools.state_slots != 0,
                rs_cache_slots: g.caps.pools.state_slots as usize,
                one_frame_when_short: g.caps.pools.one_frame_when_short,
                seated_first: g.caps.pools.seated_first,
                rs_cache_slot_bytes: g.caps.pools.state_slot_bytes,
                rs_host_slots: g.caps.pools.host_state_slots as usize,
                rs_buffer_pages: g.caps.pools.buffer_pages,
                rs_host_buffer_pages: g.caps.pools.host_buffer_pages,
                has_mtp_logits: g.caps.has_mtp_logits,
                mtp_depth: g.caps.mtp_depth,
                draft_block: g.caps.draft_block,
                draft_mask_token: g.caps.draft_mask_token,
                draft_proposals_from: g.caps.draft_proposals_from,
                has_attn_page_mask: g.caps.has_attn_page_mask,
                has_attn_score: g.caps.has_attn_score,
                has_lora: g.caps.has_lora,
                adapter_attachments: g.caps.adapter_attachments,
                rs_snapshots: g.caps.rs_snapshots,
                device_geometry_port_mask: g.caps.ports,
                facts: g.caps.facts.clone(),
                inputs: g.caps.inputs.clone(),
                classes: g.caps.classes.clone(),
                limits: runtime::engine::SchedulerLimits {
                    max_forward_requests: g.caps.limits.max_lanes as usize,
                    max_forward_tokens: g.caps.limits.max_tokens as usize,
                    max_page_refs: g.caps.limits.max_page_refs as usize,
                    max_context: g.caps.limits.max_context as usize,
                    prefill_rows: g.caps.limits.prefill_rows as usize,
                    max_readouts: g.caps.limits.max_readouts as usize,
                    max_scoring_lanes: g.caps.limits.max_scoring_lanes as usize,
                },
                engine_backend: g.backend,
            }
        })
        .collect();

    Ok(runtime::bootstrap::ModelConfig {
        name: m.name.clone(),
        model_id: engines_facts,
        kv_page_size: group0_caps.pools.kv_page_size as usize,
        tokenizer_path,
        metadata,
        engines,
        scheduler: scheduler(runtime),
    })
}

/// The disk tier `g` keeps for the next run, its journal beside its slot
/// files, when `disk_kv_persistent` asked for one and the engine kept it.
fn kept_disk(
    m: &config::ModelConfig,
    group: usize,
    g: &crate::backend::GroupEngine,
) -> Option<runtime::bootstrap::KeptDisk> {
    let pools = &g.caps.pools;
    match (&g.disk_kv_dir, pools.disk_kv_persistent) {
        (Some(dir), true) => Some(runtime::bootstrap::KeptDisk {
            journal: dir.join(format!("g{group}.journal")),
            kept: pools.disk_kv_kept,
        }),
        _ => {
            if m.disk_kv_persistent {
                tracing::warn!(
                    engine = %g.backend.kind(),
                    disk_pages = pools.disk_kv_pages,
                    "disk_kv_persistent: this engine keeps no disk tier across runs; it is scratch"
                );
            }
            None
        }
    }
}
