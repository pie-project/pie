//! Mock test environment for integration tests.
//!
//! Provides `MockEnv` which bundles a mock device backend with a
//! complete `Config`, ready to pass to `bootstrap::bootstrap()`.

use std::path::PathBuf;
use std::sync::Arc;

use tempfile::TempDir;

use pie::bootstrap::{
    AuthConfig, Config, DeviceConfig, ModelConfig, SchedulerConfig, TelemetryConfig,
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
    let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/common/fixtures/test_tokenizer.json");

    let devices: Vec<DeviceConfig> = backend
        .server_names()
        .iter()
        .map(|name| DeviceConfig {
            hostname: name.clone(),
            total_pages: num_pages,
            cpu_pages: 0,
            max_batch_size: 32,
            max_batch_tokens: 4096,
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
            devices,
            scheduler: SchedulerConfig {
                max_in_flight_batches: 4,
                request_timeout_secs: 30,
                max_wait_ms: 10,
                min_batch_for_optimization: 1,
            },
        }],
        skip_tracing: true,
        allow_filesystem: false,
    };

    MockEnv {
        backend,
        config,
        _temp_dirs: vec![temp_cache, temp_auth],
    }
}
