//! Translate the standalone's user-facing TOML config (`crate::config`)
//! into the runtime's internal `pie::bootstrap::Config`.
//!
//! The runtime's `bootstrap::Config` mirrors what `pie/server.py`
//! constructs through the pyo3 `pie._runtime.Config` builder. We do
//! the same construction here in pure Rust, sourcing:
//!   * scalars from the user TOML
//!   * dirs (cache/log/auth) from `pie::path` (`~/.pie/...`)
//!   * caps from the per-model
//!     [`ModelHandshake`] inputs collected at boot.

use std::path::PathBuf;

use anyhow::Result;

use crate::config;
use crate::embedded_driver::DriverCapabilities;

/// Per-DP-group handshake snapshot taken right after a driver thread
/// emits caps.
pub struct GroupHandshake {
    /// Caps the driver returned over the `ready_cb` callback.
    pub caps: DriverCapabilities,
}

/// Per-model bundle of group handshakes. One model with DP=N produces
/// `N` entries here; one entry per `DriverConfig` in the resulting
/// bootstrap config. Group ordering must match the runtime's flat
/// driver-index assignment (`driver::register_driver` returns indices
/// in call order).
pub struct ModelHandshake {
    pub groups: Vec<GroupHandshake>,
}

pub fn build(
    user: &config::Config,
    handshakes: &[ModelHandshake],
) -> Result<pie::bootstrap::Config> {
    if handshakes.len() != 1 {
        anyhow::bail!(
            "internal: expected exactly one handshake bundle for the single \
             model, got {}",
            handshakes.len()
        );
    }
    let handshake = &handshakes[0];
    if handshake.groups.is_empty() {
        anyhow::bail!(
            "internal: model {:?} has zero group handshakes; \
             expected at least one driver per model",
            user.model.name,
        );
    }

    let pie_home = pie::path::get_pie_home();
    let cache_dir = pie_home.join("programs");
    let log_dir = Some(pie_home.join("logs"));
    let auth_dir = pie_home.join("auth");

    let model = build_model(&user.model, handshake);

    Ok(pie::bootstrap::Config {
        host: user.server.host.clone(),
        port: user.server.port,
        auth: pie::bootstrap::AuthConfig {
            enabled: user.auth.enabled,
            authorized_users_dir: auth_dir,
        },
        cache_dir,
        verbose: user.server.verbose,
        log_dir,
        registry_url: user.server.registry.clone(),
        telemetry: pie::bootstrap::TelemetryConfig {
            enabled: user.telemetry.enabled,
            endpoint: user.telemetry.endpoint.clone(),
            service_name: user.telemetry.service_name.clone(),
        },
        runtime: pie::bootstrap::RuntimeConfig {
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

fn build_model(m: &config::ModelConfig, hs: &ModelHandshake) -> pie::bootstrap::ModelConfig {
    // Arch + kv_page_size + tokenizer come from group 0; all groups
    // serve the same model so they agree. Per-group caps can differ in
    // memory-derived capacities — those flow through the per-driver entries.
    let group0_caps = &hs.groups[0].caps;
    let snapshot_dir = PathBuf::from(&group0_caps.snapshot_dir);
    let tokenizer_json = snapshot_dir.join("tokenizer.json");
    let tokenizer_path = if tokenizer_json.exists() {
        tokenizer_json
    } else {
        snapshot_dir.join("tiktoken.model")
    };

    let drivers = hs
        .groups
        .iter()
        .map(|g| pie::bootstrap::DriverConfig {
            total_pages: g.caps.total_pages as usize,
            cpu_pages: g.caps.swap_pool_size as usize,
            rs_cache_required: g.caps.rs_cache_required,
            rs_cache_slots: g.caps.rs_cache_slots as usize,
            rs_cache_slot_bytes: g.caps.rs_cache_slot_bytes,
            rs_cache_spec_rollback: g.caps.rs_cache_spec_rollback,
            limits: pie::driver::SchedulerLimits {
                max_forward_requests: g.caps.max_forward_requests as usize,
                max_forward_tokens: g.caps.max_forward_tokens as usize,
                max_page_refs: g.caps.max_page_refs as usize,
                max_logit_rows: g.caps.max_logit_rows as usize,
                max_prob_rows: g.caps.max_prob_rows as usize,
                max_sampler_rows: g.caps.max_sampler_rows as usize,
                max_custom_mask_bytes: g.caps.max_custom_mask_bytes as usize,
                max_logprob_labels: g.caps.max_logprob_labels as usize,
            },
        })
        .collect();

    pie::bootstrap::ModelConfig {
        name: m.name.clone(),
        arch_name: group0_caps.arch_name.clone(),
        kv_page_size: group0_caps.kv_page_size as usize,
        tokenizer_path,
        system_speculation_supported: hs
            .groups
            .iter()
            .all(|g| g.caps.system_speculation_supported),
        enable_system_speculation: hs.groups.iter().all(|g| g.caps.enable_system_speculation),
        drivers,
        scheduler: pie::bootstrap::SchedulerConfig {
            request_timeout_secs: m.scheduler.request_timeout_secs,
            restore_pause_at_utilization: m.scheduler.restore_pause_at_utilization,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_caps() -> DriverCapabilities {
        DriverCapabilities {
            total_pages: 1024,
            kv_page_size: 32,
            swap_pool_size: 0,
            max_forward_tokens: 4096,
            max_forward_requests: 512,
            max_page_refs: 262144,
            max_logit_rows: 4096,
            max_prob_rows: 4096,
            max_custom_mask_bytes: 8 * 1024 * 1024,
            max_sampler_rows: 4096,
            max_logprob_labels: 4096,
            arch_name: "qwen3".into(),
            vocab_size: 151936,
            max_model_len: 4096,
            activation_dtype: "bfloat16".into(),
            snapshot_dir: "/tmp/snapshot".into(),
            shmem_name: Some("/pie_shmem_g0".into()),
            rs_cache_required: false,
            rs_cache_slots: 0,
            rs_cache_slot_bytes: 0,
            rs_cache_spec_rollback: false,
            system_speculation_supported: false,
            enable_system_speculation: false,
        }
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

        let handshakes = vec![ModelHandshake {
            groups: vec![GroupHandshake {
                caps: fixture_caps(),
            }],
        }];

        let cfg = build(&user, &handshakes).unwrap();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8080);
        let m = &cfg.model;
        assert_eq!(m.name, "default");
        assert_eq!(m.arch_name, "qwen3");
        assert_eq!(m.kv_page_size, 32);
        assert_eq!(
            m.tokenizer_path,
            PathBuf::from("/tmp/snapshot/tokenizer.json")
        );
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
        g1.shmem_name = Some("/pie_shmem_g1".into());
        g1.total_pages = 2048;

        let handshakes = vec![ModelHandshake {
            groups: vec![
                GroupHandshake {
                    caps: fixture_caps(),
                },
                GroupHandshake { caps: g1 },
            ],
        }];

        let cfg = build(&user, &handshakes).unwrap();
        let m = &cfg.model;
        assert_eq!(m.drivers.len(), 2);
        assert_eq!(m.drivers[0].total_pages, 1024);
        assert_eq!(m.drivers[1].total_pages, 2048);
    }

    #[test]
    fn handshake_count_must_match() {
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
        let err = build(&user, &[]).unwrap_err().to_string();
        assert!(
            err.contains("expected exactly one handshake bundle"),
            "got: {err}"
        );
    }

    #[test]
    fn empty_groups_rejected() {
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
        let err = build(&user, &[ModelHandshake { groups: vec![] }])
            .unwrap_err()
            .to_string();
        assert!(err.contains("zero group handshakes"), "got: {err}");
    }
}
