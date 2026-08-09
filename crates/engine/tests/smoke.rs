//! Smoke tests — verify the system boots and services are reachable.
//!
//! All tests share a single bootstrap (global statics are per-process).
//! The tokio runtime is kept alive for the duration of the test process.

use std::sync::{Arc, OnceLock};
mod common;
use common::{MockEnv, create_mock_env, mock_device::EchoBehavior};

/// Shared state: MockEnv + tokio runtime (must outlive the process).
struct TestState {
    /// Held, never read: `MockEnv` owns the `TempDir` backing the model cache,
    /// so dropping it would delete the fixture directory out from under the
    /// still-running engine.
    #[allow(dead_code, reason = "liveness guard for the MockEnv-owned TempDir")]
    env: MockEnv,
    #[allow(dead_code)]
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("test-model", 4, 64, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            engine::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

#[test]
fn bootstrap_succeeds() {
    let _ = state();
}

#[test]
fn model_registered() {
    let _ = state();
    assert_eq!(engine::model::model().name(), "test-model");
}

#[test]
fn all_devices_reachable() {
    let _ = state();
    for i in 0..4 {
        let spec = engine::driver::get_spec(i).unwrap();
        assert_eq!(spec.num_kv_pages, 64);
        assert_eq!(spec.limits.max_forward_requests, 32);
    }
}

#[test]
fn tokenizer_round_trip() {
    let _ = state();
    let model = engine::model::model();
    assert_eq!(model.name(), "test-model");

    let tokens = model.tokenize("hello");
    assert!(!tokens.is_empty(), "tokenize should produce tokens");

    let text = model.detokenize(&tokens);
    assert!(!text.is_empty(), "detokenize should produce text");
}
