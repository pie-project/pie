//! Translate the standalone's user-facing TOML config (`crate::config`)
//! into the runtime's internal `::runtime::bootstrap::Config`.
//!
//! The runtime's `bootstrap::Config` mirrors what `pie/server.py`
//! constructs through the pyo3 `pie._runtime.Config` builder. We do
//! the same construction here in pure Rust, sourcing:
//!   * scalars from the user TOML
//!   * dirs (cache/log/runtime) from [`crate::paths::pie_home`] (`~/.pie/...`)
//!   * capability/backend bundles collected before bootstrap.

use std::path::PathBuf;

use anyhow::Result;

use crate::config;
use crate::embedded_engine::EngineCapabilities;

/// Per-engine bundle created before bootstrap.
pub struct GroupEngine {
    /// What the load can do — the device, the pools, the ceilings, the
    /// guest-visible profile.
    pub caps: EngineCapabilities,
    /// What the load came out as: its plan's name and the bytes it landed.
    pub facts: engine_api::LoadFacts,
    /// Where this worker resolved the checkpoint.
    ///
    /// **THE CALLER'S ANSWER, NOT THE ENGINE'S.** It was
    /// `EngineCapabilities::snapshot_dir`, echoed back by an engine that had
    /// just been handed it; the contract dropped it for exactly that reason
    /// (`engine-api::caps`: "`snapshot_dir`/`model_id`/`arch_name` say where
    /// the caller's own checkpoint came from").
    pub snapshot_dir: PathBuf,
    /// The device behind it.
    pub backend: ::runtime::engine::EngineBox,
}

/// Per-model bundle of concrete engine backends. One model with DP=N produces
/// `N` entries here; one entry per bootstrap engine config.
pub struct ModelEngines {
    pub groups: Vec<GroupEngine>,
}

/// The one place config units become the runtime's plain numbers. `Duration`
/// and `ByteSize` carry their unit through the config layer; `runtime`'s
/// bootstrap structs still take `_secs`/`_us`/`_mb` scalars, so the conversion
/// happens here and only here.
pub fn build(
    user: &config::Config,
    engines: ModelEngines,
    metadata: ::runtime::model::ModelMetadata,
) -> Result<::runtime::bootstrap::Config> {
    if engines.groups.is_empty() {
        anyhow::bail!(
            "internal: model {:?} has zero native engines; \
             expected at least one engine per model",
            user.model.name,
        );
    }

    let pie_home = crate::paths::pie_home();
    let cache_dir = pie_home.join("programs");
    let log_dir = Some(pie_home.join("logs"));

    let model = build_model(&user.model, &user.runtime, engines, metadata)?;

    Ok(::runtime::bootstrap::Config {
        host: user.server.host.clone(),
        port: user.server.port,
        cache_dir,
        verbose: user.server.verbose,
        log_dir,
        registry_url: user.server.registry.clone(),
        telemetry: ::runtime::bootstrap::TelemetryConfig {
            enabled: user.telemetry.enabled,
            endpoint: user.telemetry.endpoint.clone(),
            service_name: user.telemetry.service_name.clone(),
        },
        // ONE STRUCT ON THIS SIDE, TWO ON OURS. `bootstrap::RuntimeConfig`
        // still carries the tokio pool and the sandbox in one bag, and still
        // spells the guest knobs with the `wasm_` prefix; `crate::config` has
        // split them into `[server]` (how the process is sized) and
        // `[sandbox]` (the box a guest runs in). This is the seam that joins
        // them back up, which is why the field names differ from the paths
        // they read.
        runtime: ::runtime::bootstrap::RuntimeConfig {
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
        // The `bootstrap` lib (Seam 2) installs the global tracing subscriber;
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
    metadata: ::runtime::model::ModelMetadata,
) -> Result<::runtime::bootstrap::ModelConfig> {
    // Arch + kv_page_size + tokenizer come from group 0; all groups
    // serve the same model so they agree. Per-group caps can differ in
    // memory-derived capacities — those flow through the per-engine entries.
    let group0_caps = engines.groups[0].caps.clone();
    let snapshot_dir = engines.groups[0].snapshot_dir.clone();
    let engines_facts = engines.groups[0].facts.trace_name.clone();
    // The metadata was lifted once when the model was resolved; this only
    // decides which of the two shapes the runtime is being handed. Only the
    // tokenizer half varies -- the config is there either way.
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
            // THREE RECORDS, NOT ONE FLAT STRUCT. Every line below reads
            // the half of `Capabilities` its own subject lives in — `pools`
            // for capacities, `limits` for ceilings, `profile` for what a
            // guest program may name — which is the split
            // `engine-api::caps`'s header argues for.
            ::runtime::bootstrap::EngineConfig {
                total_pages: g.caps.pools.kv_pages as usize,
                // The host-swap pool is a DEPLOYMENT's, not a load's: the
                // contract has no field for it because no engine reserves
                // one on the caller's behalf.
                cpu_pages: 0,
                kv_copy: g.caps.kv_copy,
                backend_kind,
                rs_cache_required: g.caps.pools.state_slots != 0,
                rs_cache_slots: g.caps.pools.state_slots as usize,
                rs_cache_slot_bytes: g.caps.pools.state_slot_bytes,
                elastic_page_bytes: g.caps.pools.elastic_page_bytes,
                elastic_budget_pages: g.caps.pools.elastic_budget_pages,
                has_mtp_logits: g.caps.profile.has_mtp_logits,
                has_mtp_drafts: g.caps.profile.has_mtp_drafts,
                has_value_head: g.caps.profile.has_value_head,
                // `has_kv_envelopes` has no successor: it advertised a
                // model-gated PTIR intrinsic the profile does not carry, and
                // the runtime's `PtirCaps` is the only reader.
                has_kv_envelopes: false,
                has_attn_page_mask: g.caps.profile.has_attn_page_mask,
                has_attn_score: g.caps.profile.has_attn_score,
                has_lora: g.caps.profile.has_lora,
                device_geometry_port_mask: g.caps.ports,
                // palo B-pipelined-geometry: this said whether an engine
                // resolves a STEP's descriptor ports when the step runs
                // rather than for the whole frame. The contract has no field
                // for it, and no shell in this workspace interleaves the two
                // halves — the CUDA shell stages every geometry vector from
                // the host before the walk. False is what both of those say.
                resolves_geometry_per_step: false,
                limits: ::runtime::engine::SchedulerLimits {
                    max_forward_requests: g.caps.limits.max_lanes as usize,
                    max_forward_tokens: g.caps.limits.max_tokens as usize,
                    max_page_refs: g.caps.limits.max_page_refs as usize,
                },
                engine_backend: g.backend,
            }
        })
        .collect();

    Ok(::runtime::bootstrap::ModelConfig {
        name: m.name.clone(),
        // The operator's answer, or the model's own name. It was
        // `EngineCapabilities::model_id`, echoed back by an engine that was
        // handed it.
        // The plan's own name, as the model text declared it — which is what
        // a `model_id` ever was: the row this checkpoint loaded as. It comes
        // off `LoadFacts` now instead of off an engine echoing back the string
        // the operator handed it.
        model_id: engines_facts,
        kv_page_size: group0_caps.pools.kv_page_size as usize,
        tokenizer_path,
        metadata,
        engines,
        // Batching is a DEPLOYMENT's, not a model's: it hung off
        // `ModelConfig` while the config could serve several, and one config
        // serves one model now, so it is `Config::runtime` and arrives here
        // as its own argument.
        scheduler: ::runtime::bootstrap::SchedulerConfig {
            request_timeout_secs: runtime.request_timeout.as_secs(),
            submit_deadline_us: runtime.submit_deadline.as_micros(),
            silence_timeout_secs: runtime.silence_timeout.as_secs(),
            frame_size: runtime.frame_size,
            frame_submit_depth: runtime.frame_submit_depth,
            frame_dispatch_depth: runtime.frame_dispatch_depth,
        },
    })
}
