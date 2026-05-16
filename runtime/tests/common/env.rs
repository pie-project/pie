//! Mock test environment for integration tests.
//!
//! Provides `MockEnv` which bundles a mock driver backend with a
//! complete `Config`, ready to pass to `bootstrap::bootstrap()`.

use std::path::PathBuf;
use std::sync::Arc;

use tempfile::TempDir;

use pie::bootstrap::{
    AuthConfig, Config, DriverConfig, ModelConfig, RuntimeConfig, SchedulerConfig, TelemetryConfig,
};

use super::mock_device::{Behavior, MockBackend};

// =============================================================================
// MockEnv
// =============================================================================

/// A self-contained test environment with mock device backends.
///
/// Owns the mock backend (keeping IPC servers alive) and temporary
/// directories. Everything is cleaned up on drop.
pub struct MockEnv {
    /// The mock device backend (RPC servers + poll threads).
    pub backend: MockBackend,
    /// The generated Config, ready for `bootstrap::bootstrap()`.
    config: Config,
    /// Temporary directories (cache_dir, auth dir) — cleaned up on drop.
    _temp_dirs: Vec<TempDir>,
}

impl MockEnv {
    /// Returns a clone of the config to pass to `bootstrap::bootstrap()`.
    pub fn config(&self) -> Config {
        self.config.clone()
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Create a mock test environment with the given parameters.
///
/// This:
/// 1. Creates temporary directories for cache and auth
/// 2. Spawns mock RPC servers (one per device)
/// 3. Builds a `Config` with `skip_tracing: true` and `auth.enabled: false`
/// 4. Sets `tokenizer_path` to the bundled test fixture
pub fn create_mock_env(
    model_name: &str,
    num_devices: usize,
    num_pages: usize,
    behavior: Arc<dyn Behavior>,
) -> MockEnv {
    let backend = MockBackend::new(num_devices, behavior);

    let temp_cache = TempDir::new().expect("Failed to create temp cache dir");
    let temp_auth = TempDir::new().expect("Failed to create temp auth dir");

    // Path to bundled test tokenizer fixture
    let tokenizer_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/common/fixtures/test_tokenizer.json");

    // The pre-registered mock channels live at `backend.driver_ids()`.
    // `bootstrap::bootstrap` allocates its own DriverIds starting from
    // the global counter — these may not match the mock IDs in the same
    // process. Tests that depend on full end-to-end flow are expected
    // to consume the IDs from MockBackend directly rather than re-route
    // through bootstrap's allocator.
    let drivers: Vec<DriverConfig> = (0..num_devices)
        .map(|_| DriverConfig {
            total_pages: num_pages,
            cpu_pages: 0,
            limits: pie::driver::SchedulerLimits {
                max_forward_requests: 32,
                max_forward_tokens: 4096,
                max_page_refs: num_pages,
                max_sampler_rows: usize::MAX,
                max_custom_mask_bytes: usize::MAX,
                max_logprob_labels: usize::MAX,
            },
        })
        .collect();

    let config = Config {
        host: "127.0.0.1".into(),
        port: 0,
        auth: AuthConfig {
            enabled: false,
            authorized_users_dir: temp_auth.path().to_path_buf(),
        },
        cache_dir: temp_cache.path().to_path_buf(),
        verbose: false,
        log_dir: None,
        registry_url: String::new(),
        telemetry: TelemetryConfig {
            enabled: false,
            endpoint: String::new(),
            service_name: String::new(),
        },
        models: vec![ModelConfig {
            name: model_name.to_string(),
            arch_name: String::new(),
            kv_page_size: 16,
            tokenizer_path,
            drivers,
            scheduler: SchedulerConfig {
                batch_policy: "adaptive".into(),
                request_timeout_secs: 30,
                default_token_limit: None,  // unlimited by default
                default_endowment_pages: 4, // small endowment for mock GPUs
                // Permissive for tests: allow up to 32× overbook so fixtures
                // don't trip the admission gate on small-capacity mock devices.
                admission_oversubscription_factor: 32.0,
                restore_pause_at_utilization: 0.85,
                speculation_depth: 1,
            },
        }],
        runtime: RuntimeConfig {
            worker_threads: 4,
            wasm_max_instances: 1000,
            wasm_max_memory_mb: 4096,
            wasm_warm_memory_mb: 0,
            wasm_warm_slots: 100,
            allow_fs: false,
            fs_scratch_dir: temp_cache.path().to_path_buf(),
            allow_network: false,
            network_allowed_hosts: vec![],
            max_upload_mb: 256,
        },
        skip_tracing: true,
        max_concurrent_processes: None,
        python_snapshot: false,
    };

    MockEnv {
        backend,
        config,
        _temp_dirs: vec![temp_cache, temp_auth],
    }
}
