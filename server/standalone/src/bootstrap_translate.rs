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

/// Per-model handshake snapshot taken right after the driver thread
/// emits caps and the cold-path `RpcServer` is up.
pub struct ModelHandshake {
    /// `RpcServer::server_name()` — the cold-path channel the runtime
    /// connects to via `device::spawn(hostname, ...)`.
    pub rpc_server_name: String,
    /// Caps the driver returned over the `ready_cb` callback.
    pub caps: DriverCapabilities,
}

pub fn build(
    user: &config::Config,
    handshakes: &[ModelHandshake],
) -> Result<pie::bootstrap::Config> {
    if handshakes.len() != user.models.len() {
        anyhow::bail!(
            "internal: {} models in TOML but {} handshakes",
            user.models.len(),
            handshakes.len()
        );
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
    let caps = &hs.caps;
    let tokenizer_path = PathBuf::from(&caps.snapshot_dir).join("tokenizer.json");

    // v0: each model maps to exactly one device. Multi-replica DP per
    // model lands post-v1 alongside multi-driver spawn.
    let device = pie::bootstrap::DeviceConfig {
        hostname: hs.rpc_server_name.clone(),
        total_pages: caps.total_pages as usize,
        cpu_pages: caps.swap_pool_size as usize,
        max_batch_tokens: caps.max_batch_tokens as usize,
        max_batch_size: caps.max_batch_size as usize,
    };

    pie::bootstrap::ModelConfig {
        name: m.name.clone(),
        arch_name: caps.arch_name.clone(),
        kv_page_size: caps.kv_page_size as usize,
        tokenizer_path,
        devices: vec![device],
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
            rpc_server_name: "/tmp/test/socket".into(),
            caps: fixture_caps(),
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
        assert!(err.contains("1 models in TOML but 0 handshakes"));
    }
}
