//! Translate the standalone's user-facing TOML config (`crate::config`)
//! into the runtime's internal `pie_engine::bootstrap::Config`.
//!
//! The runtime's `bootstrap::Config` mirrors what `pie/server.py`
//! constructs through the pyo3 `pie._runtime.Config` builder. We do
//! the same construction here in pure Rust, sourcing:
//!   * scalars from the user TOML
//!   * dirs (cache/log/auth) from `pie_engine::util` (`~/.pie/...`)
//!   * caps/native-driver bundles collected before bootstrap.

use std::path::PathBuf;

use anyhow::Result;

use crate::config;
use crate::embedded_driver::DriverCapabilities;

/// Per-driver bundle created before bootstrap.
pub struct GroupDriver {
    pub caps: DriverCapabilities,
    pub native: pie_engine::driver::NativeDriver,
}

/// Per-model bundle of concrete native drivers. One model with DP=N produces
/// `N` entries here; one entry per bootstrap driver config.
pub struct ModelDrivers {
    pub groups: Vec<GroupDriver>,
}

pub fn build(
    user: &config::Config,
    drivers: ModelDrivers,
) -> Result<pie_engine::bootstrap::Config> {
    if drivers.groups.is_empty() {
        anyhow::bail!(
            "internal: model {:?} has zero native drivers; \
             expected at least one driver per model",
            user.model.name,
        );
    }

    let pie_home = pie_engine::util::get_pie_home();
    let cache_dir = pie_home.join("programs");
    let log_dir = Some(pie_home.join("logs"));
    let auth_dir = pie_home.join("auth");

    let model = build_model(&user.model, drivers)?;

    Ok(pie_engine::bootstrap::Config {
        host: user.server.host.clone(),
        port: user.server.port,
        auth: pie_engine::bootstrap::AuthConfig {
            enabled: user.auth.enabled,
            authorized_users_dir: auth_dir,
        },
        cache_dir,
        verbose: user.server.verbose,
        log_dir,
        registry_url: user.server.registry.clone(),
        telemetry: pie_engine::bootstrap::TelemetryConfig {
            enabled: user.telemetry.enabled,
            endpoint: user.telemetry.endpoint.clone(),
            service_name: user.telemetry.service_name.clone(),
        },
        runtime: pie_engine::bootstrap::RuntimeConfig {
            worker_threads: user.runtime.worker_threads,
            wasm_max_instances: user.runtime.wasm_max_instances,
            wasm_max_memory_mb: user.runtime.wasm_max_memory_mb,
            wasm_warm_memory_mb: user.runtime.wasm_warm_memory_mb,
            wasm_warm_slots: user.runtime.wasm_warm_slots,
            allow_fs: user.runtime.allow_fs,
            fs_scratch_dir: user.runtime.fs_scratch_dir.clone(),
            allow_network: user.runtime.allow_network,
            network_allowed_hosts: user.runtime.network_allowed_hosts.clone(),
            max_upload_mb: user.runtime.max_upload_mb,
        },
        model,
        // The `bootstrap` lib (Seam 2) installs the global tracing subscriber;
        // the runtime must NOT re-init it (double global-init panics on boot).
        skip_tracing: true,
        max_concurrent_processes: user.server.max_concurrent_processes,
        python_snapshot: user.server.python_snapshot,
    })
}

fn build_model(
    m: &config::ModelConfig,
    drivers: ModelDrivers,
) -> Result<pie_engine::bootstrap::ModelConfig> {
    // Arch + kv_page_size + tokenizer come from group 0; all groups
    // serve the same model so they agree. Per-group caps can differ in
    // memory-derived capacities — those flow through the per-driver entries.
    let group0_caps = drivers.groups[0].caps.clone();
    let snapshot_dir = PathBuf::from(&group0_caps.snapshot_dir);
    let tokenizer_json = snapshot_dir.join("tokenizer.json");
    let tokenizer_path = if tokenizer_json.exists() {
        tokenizer_json
    } else {
        snapshot_dir.join("tiktoken.model")
    };

    let drivers = drivers
        .groups
        .into_iter()
        .map(|g| pie_engine::bootstrap::DriverConfig {
            total_pages: g.caps.total_pages as usize,
            cpu_pages: g.caps.swap_pool_size as usize,
            rs_cache_required: g.caps.rs_cache_required,
            rs_cache_slots: g.caps.rs_cache_slots as usize,
            rs_cache_slot_bytes: g.caps.rs_cache_slot_bytes,
            limits: pie_engine::driver::SchedulerLimits {
                max_forward_requests: g.caps.max_forward_requests as usize,
                max_forward_tokens: g.caps.max_forward_tokens as usize,
                max_page_refs: g.caps.max_page_refs as usize,
            },
            native_driver: g.native,
        })
        .collect();

    Ok(pie_engine::bootstrap::ModelConfig {
        name: m.name.clone(),
        arch_name: group0_caps.arch_name,
        kv_page_size: group0_caps.kv_page_size as usize,
        tokenizer_path,
        drivers,
        scheduler: pie_engine::bootstrap::SchedulerConfig {
            request_timeout_secs: m.scheduler.request_timeout_secs,
            restore_pause_at_utilization: m.scheduler.restore_pause_at_utilization,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_caps() -> DriverCapabilities {
        DriverCapabilities {
            abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
            total_pages: 1024,
            kv_page_size: 32,
            swap_pool_size: 0,
            max_forward_tokens: 4096,
            max_forward_requests: 512,
            max_page_refs: 262144,
            arch_name: "qwen3".into(),
            vocab_size: 151936,
            max_model_len: 4096,
            activation_dtype: "bfloat16".into(),
            snapshot_dir: "/tmp/snapshot".into(),
            storage_backend: String::new(),
            max_tile_bytes: 0,
            preferred_alignment: 0,
            mxfp4_moe_policy: String::new(),
            native_mxfp4_moe: false,
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
        }
    }

    fn fixture_group(caps: DriverCapabilities) -> GroupDriver {
        let dummy = pie_driver_dummy_lib::DummyDriverOptions {
            total_pages: caps.total_pages,
            kv_page_size: caps.kv_page_size,
            swap_pool_size: caps.swap_pool_size,
            vocab_size: caps.vocab_size,
            max_model_len: caps.max_model_len,
            arch_name: caps.arch_name.clone(),
            activation_dtype: caps.activation_dtype.clone(),
            snapshot_dir: caps.snapshot_dir.clone(),
            max_forward_tokens: caps.max_forward_tokens,
            max_forward_requests: caps.max_forward_requests,
            max_page_refs: caps.max_page_refs,
            callback_delay_ms: 0,
            operation_log: None,
        };
        let (native, _) = pie_engine::driver::NativeDriver::dummy(dummy).unwrap();
        GroupDriver { caps, native }
    }

    #[test]
    fn translates_minimal_config() {
        let toml_text = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dummy"
device = ["cpu"]

[model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
"#;
        let user: config::Config = toml::from_str(toml_text).unwrap();
        user.validate().unwrap();

        // `build` resolves the tokenizer by probing the snapshot dir
        // (tokenizer.json, else tiktoken.model — the Kimi fallback added in
        // 150e0087). Point the fixture at a real temp dir containing a
        // tokenizer.json so the primary path resolves deterministically.
        let snap = tempfile::tempdir().unwrap();
        std::fs::write(snap.path().join("tokenizer.json"), b"{}").unwrap();
        let mut caps = fixture_caps();
        caps.snapshot_dir = snap.path().to_string_lossy().into_owned();

        let cfg = build(
            &user,
            ModelDrivers {
                groups: vec![fixture_group(caps)],
            },
        )
        .unwrap();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8080);
        let m = &cfg.model;
        assert_eq!(m.name, "default");
        assert_eq!(m.arch_name, "qwen3");
        assert_eq!(m.kv_page_size, 32);
        assert_eq!(m.tokenizer_path, snap.path().join("tokenizer.json"));
        assert_eq!(m.drivers.len(), 1);
        assert_eq!(m.drivers[0].total_pages, 1024);
        assert_eq!(m.drivers[0].limits.max_page_refs, 262144);
    }

    #[test]
    fn translates_dp_two_groups() {
        let toml_text = r#"
[model]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "dummy"
device = ["cuda:0", "cuda:1"]

[model.driver.options]
vocab_size = 151936
arch_name = "qwen3"
"#;
        let user: config::Config = toml::from_str(toml_text).unwrap();
        user.validate().unwrap();

        // DP=2 → two groups, each with its own driver channel + caps.
        let mut g1 = fixture_caps();
        g1.total_pages = 2048;

        let cfg = build(
            &user,
            ModelDrivers {
                groups: vec![fixture_group(fixture_caps()), fixture_group(g1)],
            },
        )
        .unwrap();
        let m = &cfg.model;
        assert_eq!(m.drivers.len(), 2);
        assert_eq!(m.drivers[0].total_pages, 1024);
        assert_eq!(m.drivers[1].total_pages, 2048);
    }

    #[test]
    fn requires_at_least_one_driver() {
        let user: config::Config = toml::from_str(
            r#"
[model]
name = "a"
hf_repo = "x"
[model.driver]
type = "dummy"
device = ["cpu"]
"#,
        )
        .unwrap();
        let err = build(&user, ModelDrivers { groups: vec![] })
            .err()
            .unwrap()
            .to_string();
        assert!(
            err.contains("zero native drivers"),
            "got: {err}"
        );
    }
}
