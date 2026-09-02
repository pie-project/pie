//! Translate the standalone's user-facing TOML config (`crate::config`) into
//! the runtime's internal `runtime::bootstrap::Config`: scalars from the user
//! TOML, dirs from [`bootstrap::paths::pie_home`], capability/backend bundles
//! collected before bootstrap.

use anyhow::Result;

use crate::backend::ModelEngines;
use crate::config;

/// The one place config units become the runtime's plain numbers: `Duration`
/// and `ByteSize` carry their unit through the config layer, but the
/// bootstrap structs still take `_secs`/`_us`/`_mb` scalars.
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
    let cache_dir = pie_home.join("programs");
    let log_dir = Some(pie_home.join("logs"));

    let model = build_model(&user.model, &user.runtime, engines, metadata)?;

    Ok(runtime::bootstrap::Config {
        host: user.server.host.clone(),
        port: user.server.port,
        cache_dir,
        verbose: user.server.verbose,
        log_dir,
        registry_url: user.server.registry.clone(),
        telemetry: runtime::bootstrap::TelemetryConfig {
            enabled: user.telemetry.enabled,
            endpoint: user.telemetry.endpoint.clone(),
            service_name: user.telemetry.service_name.clone(),
        },
        // `bootstrap::RuntimeConfig` still carries the tokio pool and sandbox
        // in one bag; `crate::config` splits them into `[server]`/`[sandbox]`,
        // which is why field names differ from the paths they read.
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
            py_runtime_dir: pie_home.join("py-runtime"),
        },
        model,
        // The `bootstrap` lib installs the global tracing subscriber;
        // the runtime must NOT re-init it (double global-init panics on boot).
        skip_tracing: true,
        max_concurrent_processes: user.runtime.max_concurrent_processes,
        python_snapshot: user.sandbox.python_snapshot,
    })
}

fn build_model(
    m: &config::ModelConfig,
    runtime: &config::RuntimeConfig,
    engines: ModelEngines,
    metadata: runtime::model::ModelMetadata,
) -> Result<runtime::bootstrap::ModelConfig> {
    // Arch + kv_page_size + tokenizer come from group 0; all groups serve
    // the same model so they agree.
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
            // Reads `Capabilities`' three records: pools for capacities,
            // limits for ceilings, profile for what a guest may name.
            runtime::bootstrap::EngineConfig {
                total_pages: g.caps.pools.kv_pages as usize,
                // The host-swap pool is a deployment's, not a load's; no
                // engine reserves one on the caller's behalf.
                cpu_pages: 0,
                kv_copy: g.caps.kv_copy,
                backend_kind,
                rs_cache_required: g.caps.pools.state_slots != 0,
                rs_cache_slots: g.caps.pools.state_slots as usize,
                rs_cache_slot_bytes: g.caps.pools.state_slot_bytes,
                has_mtp_logits: g.caps.profile.has_mtp_logits,
                mtp_depth: g.caps.profile.mtp_depth,
                has_value_head: g.caps.profile.has_value_head,
                // `has_kv_envelopes` has no successor: it advertised a
                // model-gated ETA intrinsic the profile does not carry, and
                // the runtime's `EtaCaps` is the only reader.
                has_kv_envelopes: false,
                has_attn_page_mask: g.caps.profile.has_attn_page_mask,
                has_attn_score: g.caps.profile.has_attn_score,
                has_lora: g.caps.profile.has_lora,
                device_geometry_port_mask: g.caps.ports,
                limits: runtime::engine::SchedulerLimits {
                    max_forward_requests: g.caps.limits.max_lanes as usize,
                    max_forward_tokens: g.caps.limits.max_tokens as usize,
                    max_page_refs: g.caps.limits.max_page_refs as usize,
                },
                engine_backend: g.backend,
            }
        })
        .collect();

    Ok(runtime::bootstrap::ModelConfig {
        name: m.name.clone(),
        // The plan's own name, as the model text declared it; comes off
        // `LoadFacts` rather than an engine echoing back the operator's string.
        model_id: engines_facts,
        kv_page_size: group0_caps.pools.kv_page_size as usize,
        tokenizer_path,
        metadata,
        engines,
        // Batching is a deployment's, not a model's; arrives as its own argument.
        scheduler: runtime::bootstrap::SchedulerConfig {
            request_timeout_secs: runtime.request_timeout.as_secs(),
            submit_deadline_us: runtime.submit_deadline.as_micros(),
            silence_timeout_secs: runtime.silence_timeout.as_secs(),
            frame_size: runtime.frame_size,
            frame_dispatch_depth: runtime.frame_dispatch_depth,
        },
    })
}
