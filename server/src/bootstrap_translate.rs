//! Translate the standalone's user-facing TOML config (`crate::config`)
//! into the runtime's internal `pie::bootstrap::Config`.
//!
//! The runtime's `bootstrap::Config` mirrors what `pie/server.py`
//! constructs through the pyo3 `pie._runtime.Config` builder. We do
//! the same construction here in pure Rust, sourcing:
//!   * scalars from the user TOML
//!   * dirs (cache/log/auth) from `pie::path` (`~/.pie/...`)
//!   * caps + cold-path hostname from the per-model
//!     [`ModelHandshake`] inputs collected at boot.

use std::path::PathBuf;

use anyhow::Result;

use crate::config;
use crate::embedded_driver::DriverCapabilities;

/// Per-DP-group handshake snapshot taken right after a driver thread
/// emits caps and its cold-path `RpcServer` is up.
pub struct GroupHandshake {
    /// `RpcServer::server_name()` — the cold-path channel the runtime
    /// connects to via `device::spawn(hostname, ...)`.
    pub rpc_server_name: String,
    /// Caps the driver returned over the `ready_cb` callback.
    pub caps: DriverCapabilities,
}

/// Per-model bundle of group handshakes. One model with DP=N produces
/// `N` entries here; one entry per `DeviceConfig` in the resulting
/// bootstrap config. Group ordering must match the runtime's flat
/// device-index assignment (`device::spawn` returns indices in call
/// order).
pub struct ModelHandshake {
    pub groups: Vec<GroupHandshake>,
}

pub fn build(
    user: &config::Config,
    handshakes: &[ModelHandshake],
) -> Result<pie::bootstrap::Config> {
    if handshakes.len() != user.models.len() {
        anyhow::bail!(
            "internal: {} models in TOML but {} handshake bundles",
            user.models.len(),
            handshakes.len()
        );
    }
    for (m, hs) in user.models.iter().zip(handshakes.iter()) {
        if hs.groups.is_empty() {
            anyhow::bail!(
                "internal: model {:?} has zero group handshakes; \
                 expected at least one driver per model",
                m.name,
            );
        }
    }

    let pie_home = pie::path::get_pie_home();
    let cache_dir = pie_home.join("programs");
    let log_dir = Some(pie_home.join("logs"));
    let auth_dir = pie_home.join("auth");

    let models = user
        .models
        .iter()
        .zip(handshakes.iter())
        .map(|(m, hs)| build_model(m, hs))
        .collect();

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
        models,
        skip_tracing: false,
        max_concurrent_processes: user.server.max_concurrent_processes,
        python_snapshot: user.server.python_snapshot,
    })
}

fn build_model(
    m: &config::ModelConfig,
    hs: &ModelHandshake,
) -> pie::bootstrap::ModelConfig {
    // Arch + kv_page_size + tokenizer come from group 0; all groups
    // serve the same model so they agree. Per-group caps differ only
    // in `total_pages` / `swap_pool_size` (potentially) — those flow
    // through the per-device entries.
    let group0_caps = &hs.groups[0].caps;
    let tokenizer_path =
        PathBuf::from(&group0_caps.snapshot_dir).join("tokenizer.json");

    let devices = hs
        .groups
        .iter()
        .map(|g| pie::bootstrap::DeviceConfig {
            hostname: g.rpc_server_name.clone(),
            total_pages: g.caps.total_pages as usize,
            cpu_pages: g.caps.swap_pool_size as usize,
            max_batch_tokens: g.caps.max_batch_tokens as usize,
            max_batch_size: g.caps.max_batch_size as usize,
        })
        .collect();

    pie::bootstrap::ModelConfig {
        name: m.name.clone(),
        arch_name: group0_caps.arch_name.clone(),
        kv_page_size: group0_caps.kv_page_size as usize,
        tokenizer_path,
        devices,
        scheduler: pie::bootstrap::SchedulerConfig {
            batch_policy: m.scheduler.batch_policy.clone(),
            request_timeout_secs: m.scheduler.request_timeout_secs,
            default_token_limit: m.scheduler.default_token_limit,
            default_endowment_pages: m.scheduler.default_endowment_pages,
            admission_oversubscription_factor: m.scheduler.admission_oversubscription_factor,
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
            max_batch_tokens: 10240,
            max_batch_size: 512,
            arch_name: "qwen3".into(),
            vocab_size: 151936,
            max_model_len: 4096,
            activation_dtype: "bfloat16".into(),
            snapshot_dir: "/tmp/snapshot".into(),
            shmem_name: "/pie_shmem_g0".into(),
        }
    }

    #[test]
    fn translates_minimal_config() {
        let toml_text = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "portable"
device = ["cpu"]
"#;
        let user: config::Config = toml::from_str(toml_text).unwrap();
        user.validate().unwrap();

        let handshakes = vec![ModelHandshake {
            groups: vec![GroupHandshake {
                rpc_server_name: "/tmp/test/socket".into(),
                caps: fixture_caps(),
            }],
        }];

        let cfg = build(&user, &handshakes).unwrap();
        assert_eq!(cfg.host, "127.0.0.1");
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.models.len(), 1);
        let m = &cfg.models[0];
        assert_eq!(m.name, "default");
        assert_eq!(m.arch_name, "qwen3");
        assert_eq!(m.kv_page_size, 32);
        assert_eq!(m.tokenizer_path, PathBuf::from("/tmp/snapshot/tokenizer.json"));
        assert_eq!(m.devices.len(), 1);
        assert_eq!(m.devices[0].hostname, "/tmp/test/socket");
        assert_eq!(m.devices[0].total_pages, 1024);
        assert_eq!(m.scheduler.batch_policy, "adaptive");
    }

    #[test]
    fn translates_dp_two_model() {
        let toml_text = r#"
[[model]]
name = "default"
hf_repo = "Qwen/Qwen3-0.6B"

[model.driver]
type = "portable"
device = ["cuda:0", "cuda:1"]
"#;
        let user: config::Config = toml::from_str(toml_text).unwrap();
        user.validate().unwrap();

        // DP=2 → two groups, each with its own RpcServer + caps.
        let mut g1 = fixture_caps();
        g1.shmem_name = "/pie_shmem_g1".into();
        g1.total_pages = 2048;

        let handshakes = vec![ModelHandshake {
            groups: vec![
                GroupHandshake {
                    rpc_server_name: "/tmp/test/socket-0".into(),
                    caps: fixture_caps(),
                },
                GroupHandshake {
                    rpc_server_name: "/tmp/test/socket-1".into(),
                    caps: g1,
                },
            ],
        }];

        let cfg = build(&user, &handshakes).unwrap();
        let m = &cfg.models[0];
        assert_eq!(m.devices.len(), 2);
        assert_eq!(m.devices[0].hostname, "/tmp/test/socket-0");
        assert_eq!(m.devices[0].total_pages, 1024);
        assert_eq!(m.devices[1].hostname, "/tmp/test/socket-1");
        assert_eq!(m.devices[1].total_pages, 2048);
    }

    #[test]
    fn handshake_count_must_match() {
        let user: config::Config = toml::from_str(
            r#"
[[model]]
name = "a"
hf_repo = "x"
[model.driver]
type = "portable"
device = ["cpu"]
"#,
        )
        .unwrap();
        let err = build(&user, &[]).unwrap_err().to_string();
        assert!(err.contains("1 models in TOML but 0 handshake bundles"));
    }

    #[test]
    fn empty_groups_rejected() {
        let user: config::Config = toml::from_str(
            r#"
[[model]]
name = "a"
hf_repo = "x"
[model.driver]
type = "portable"
device = ["cpu"]
"#,
        )
        .unwrap();
        let err = build(
            &user,
            &[ModelHandshake { groups: vec![] }],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("zero group handshakes"), "got: {err}");
    }
}
