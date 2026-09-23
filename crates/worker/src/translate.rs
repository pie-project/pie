use anyhow::Result;

use crate::backend::ModelEngines;
use crate::config;

pub fn build(
    user: &config::Config,
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

    let pie_home = bootstrap::paths::pie_home();
    let cache_dir = bootstrap::paths::inferlets_dir();
    let log_dir = Some(pie_home.join("logs"));

    let model = build_model(&user.model, &user.runtime, engines, metadata)?;

    Ok(runtime::bootstrap::Config {
        host: user.server.host.clone(),
        port: user.server.port,
        cache_dir,
        builtin_programs: builtins::all()
            .iter()
            .map(|b| runtime::bootstrap::BuiltinProgram {
                name: b.name,
                version: b.version,
                component: b.component,
            })
            .collect(),
        verbose: user.server.verbose,
        log_dir,
        telemetry: runtime::bootstrap::TelemetryConfig {
            enabled: user.telemetry.enabled,
            endpoint: user.telemetry.endpoint.clone(),
            service_name: user.telemetry.service_name.clone(),
        },
        runtime: runtime::bootstrap::RuntimeConfig {
            worker_threads: user.server.worker_threads,
            wasm_max_instances: user.sandbox.max_instances,
            wasm_max_memory_mb: user.sandbox.max_memory.as_mib() as usize,
            wasm_warm_memory_mb: user.sandbox.warm_memory.as_mib() as usize,
            wasm_warm_slots: user.sandbox.warm_slots,
            allow_fs: user.sandbox.allow_fs,
            fs_scratch_dir: user.sandbox.fs_scratch_dir.clone(),
            allow_network: user.sandbox.allow_network,
            network_allowed_hosts: user.sandbox.network_allowed_hosts.clone(),
            max_upload_mb: user.server.max_upload.as_mib() as usize,
            languages_dir: bootstrap::paths::languages_dir(),
            compile_cache_dir: Some(bootstrap::paths::compile_cache_dir()),
        },
        model,
        skip_tracing: true,
        max_concurrent_processes: user.runtime.max_concurrent_processes,
    })
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
        .map(|g| {
            let backend_kind = g.backend.kind().to_string();
            runtime::bootstrap::EngineConfig {
                total_pages: g.caps.pools.kv_pages as usize,
                cpu_pages: 0,
                kv_copy: g.caps.kv_copy,
                backend_kind,
                rs_cache_required: g.caps.pools.state_slots != 0,
                rs_cache_slots: g.caps.pools.state_slots as usize,
                rs_cache_slot_bytes: g.caps.pools.state_slot_bytes,
                has_mtp_logits: g.caps.profile.has_mtp_logits,
                mtp_depth: g.caps.profile.mtp_depth,
                draft_block: g.caps.profile.draft_block,
                draft_mask_token: g.caps.profile.draft_mask_token,
                draft_bidirectional: g.caps.profile.draft_bidirectional,
                draft_proposals_from: g.caps.profile.draft_proposals_from,
                has_value_head: g.caps.profile.has_value_head,
                has_kv_envelopes: false,
                has_attn_page_mask: g.caps.profile.has_attn_page_mask,
                has_attn_score: g.caps.profile.has_attn_score,
                has_lora: g.caps.profile.has_lora,
                device_geometry_port_mask: g.caps.ports,
                limits: runtime::engine::SchedulerLimits {
                    max_forward_requests: g.caps.limits.max_lanes as usize,
                    max_forward_tokens: g.caps.limits.max_tokens as usize,
                    max_page_refs: g.caps.limits.max_page_refs as usize,
                    max_context: g.caps.limits.max_context as usize,
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
        scheduler: runtime::bootstrap::SchedulerConfig {
            submit_deadline_us: runtime.submit_deadline.as_micros(),
            silence_timeout_secs: runtime.silence_timeout.as_secs(),
            frame_size: runtime.frame_size,
            frame_dispatch_depth: runtime.frame_dispatch_depth,
        },
    })
}
