//! Mock test environment for integration tests.

use std::path::PathBuf;
use std::sync::Arc;

use tempfile::TempDir;

use pie_engine::bootstrap::{
    AuthConfig, Config, DriverConfig, ModelConfig, RuntimeConfig, SchedulerConfig, TelemetryConfig,
};
use pie_engine::driver::{NativeDriver, SchedulerLimits};

use super::mock_device::{Behavior, MockBackend};

fn native_dummy_driver(num_pages: usize) -> NativeDriver {
    let (native, _) = NativeDriver::dummy(pie_driver_dummy_lib::DummyDriverOptions {
        total_pages: num_pages as u32,
        kv_page_size: 16,
        swap_pool_size: 0,
        vocab_size: 32,
        max_model_len: 8192,
        arch_name: "test-dummy".into(),
        activation_dtype: "f32".into(),
        snapshot_dir: String::new(),
        max_forward_tokens: 4096,
        max_forward_requests: 32,
        max_page_refs: num_pages.max(1) as u32,
        callback_delay_ms: 0,
        operation_log: None,
    })
    .expect("create dummy native driver");
    native
}

pub struct MockEnv {
    pub backend: MockBackend,
    model_name: String,
    num_devices: usize,
    num_pages: usize,
    temp_cache: TempDir,
    temp_auth: TempDir,
}

impl MockEnv {
    pub fn config(&self) -> Config {
        let tokenizer_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/common/fixtures/test_tokenizer.json");

        let drivers: Vec<DriverConfig> = (0..self.num_devices)
            .map(|_| DriverConfig {
                total_pages: self.num_pages,
                cpu_pages: self.num_pages * 4,
                rs_cache_required: false,
                rs_cache_slots: 0,
                rs_cache_slot_bytes: 0,
                limits: SchedulerLimits {
                    max_forward_requests: 32,
                    max_forward_tokens: 4096,
                    max_page_refs: self.num_pages,
                },
                native_driver: native_dummy_driver(self.num_pages),
            })
            .collect();

        Config {
            host: "127.0.0.1".into(),
            port: 0,
            auth: AuthConfig {
                enabled: false,
                authorized_users_dir: self.temp_auth.path().to_path_buf(),
            },
            cache_dir: self.temp_cache.path().to_path_buf(),
            verbose: false,
            log_dir: None,
            registry_url: String::new(),
            telemetry: TelemetryConfig {
                enabled: false,
                endpoint: String::new(),
                service_name: String::new(),
            },
            model: ModelConfig {
                name: self.model_name.clone(),
                arch_name: String::new(),
                kv_page_size: 16,
                tokenizer_path,
                drivers,
                scheduler: SchedulerConfig {
                    request_timeout_secs: 30,
                    restore_pause_at_utilization: 0.85,
                },
            },
            runtime: RuntimeConfig {
                worker_threads: 4,
                wasm_max_instances: 1000,
                wasm_max_memory_mb: 4096,
                wasm_warm_memory_mb: 0,
                wasm_warm_slots: 100,
                allow_fs: false,
                fs_scratch_dir: self.temp_cache.path().to_path_buf(),
                allow_network: false,
                network_allowed_hosts: vec![],
                max_upload_mb: 256,
            },
            skip_tracing: true,
            max_concurrent_processes: None,
            python_snapshot: false,
        }
    }
}

pub fn create_mock_env(
    model_name: &str,
    num_devices: usize,
    num_pages: usize,
    behavior: Arc<dyn Behavior>,
) -> MockEnv {
    MockEnv {
        backend: MockBackend::new(num_devices, behavior),
        model_name: model_name.to_string(),
        num_devices,
        num_pages,
        temp_cache: TempDir::new().expect("Failed to create temp cache dir"),
        temp_auth: TempDir::new().expect("Failed to create temp auth dir"),
    }
}
