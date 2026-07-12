//! Mock test environment for integration tests.

use std::path::PathBuf;
use std::sync::Arc;

use tempfile::TempDir;

use pie_engine::bootstrap::{
    Config, DriverConfig, ModelConfig, RuntimeConfig, SchedulerConfig, TelemetryConfig,
};
use pie_engine::driver::{DriverBackend, SchedulerLimits};

use super::mock_device::{Behavior, MockBackend, launch_observer};

/// The mock model's logits/output vocab. MUST match what the engine model
/// reports (`Model::vocab_size()`, which reads `vocab_size` from the
/// fixture `config.json` beside the tokenizer, falling back to the
/// tokenizer vocab): a guest declares its `logits` intrinsic as
/// `[rows, output-vocab-size]` and the dummy driver validates that decl
/// against ITS capability vocab — a mismatch rejects every logits-using
/// PTIR program at bind.
fn fixture_vocab_size() -> u32 {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/common/fixtures");
    let cfg =
        std::fs::read_to_string(fixtures.join("config.json")).expect("read fixture config.json");
    let cfg: serde_json::Value = serde_json::from_str(&cfg).expect("parse fixture config.json");
    match cfg.get("vocab_size").and_then(|v| v.as_u64()) {
        Some(v) => v as u32,
        None => {
            let tokenizer =
                pie_tokenizer::Tokenizer::from_file(&fixtures.join("test_tokenizer.json"))
                    .expect("load fixture tokenizer");
            tokenizer.vocab_size() as u32
        }
    }
}

fn dummy_driver_backend(
    num_pages: usize,
    behavior: Arc<dyn Behavior>,
    operation_log: Arc<std::sync::Mutex<Vec<String>>>,
) -> DriverBackend {
    let (backend, _) = DriverBackend::dummy(pie_driver_dummy_lib::DummyDriverOptions {
        total_pages: num_pages as u32,
        kv_page_size: 16,
        swap_pool_size: 0,
        vocab_size: fixture_vocab_size(),
        max_model_len: 8192,
        arch_name: "test-dummy".into(),
        activation_dtype: "f32".into(),
        snapshot_dir: String::new(),
        max_forward_tokens: 4096,
        max_forward_requests: 32,
        max_page_refs: num_pages.max(1) as u32,
        callback_delay_ms: 0,
        reject_launches: false,
        reject_launches_remaining: 0,
        fail_launches_after_accept: false,
        retry_launches_remaining: 0,
        operation_log: Some(operation_log),
        launch_observer: Some(launch_observer(behavior)),
    })
    .expect("create dummy driver backend");
    backend
}

pub struct MockEnv {
    pub backend: MockBackend,
    model_name: String,
    num_devices: usize,
    num_pages: usize,
    behavior: Arc<dyn Behavior>,
    temp_cache: TempDir,
    /// Dummy-driver operation log (shared across every device driver): op
    /// names plus `launch-shape tokens=N programs=P` entries for geometry
    /// assertions.
    operation_log: Arc<std::sync::Mutex<Vec<String>>>,
}

impl MockEnv {
    /// Snapshot of the dummy-driver operation log.
    #[allow(dead_code)]
    pub fn operations(&self) -> Vec<String> {
        self.operation_log.lock().unwrap().clone()
    }

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
                driver_backend: dummy_driver_backend(
                    self.num_pages,
                    self.behavior.clone(),
                    Arc::clone(&self.operation_log),
                ),
            })
            .collect();

        Config {
            host: "127.0.0.1".into(),
            port: 0,
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
                py_runtime_dir: self.temp_cache.path().join("py-runtime"),
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
    let operation_log = Arc::new(std::sync::Mutex::new(Vec::new()));
    MockEnv {
        backend: MockBackend::new(
            num_devices,
            behavior.clone(),
            fixture_vocab_size(),
            Arc::clone(&operation_log),
        ),
        model_name: model_name.to_string(),
        num_devices,
        num_pages,
        behavior,
        temp_cache: TempDir::new().expect("Failed to create temp cache dir"),
        operation_log,
    }
}
