//! Smoke tests — verify the system boots and services are reachable.
//!
//! All tests share a single bootstrap (global statics are per-process).
//! The tokio runtime is kept alive for the duration of the test process.

use std::sync::{Arc, OnceLock};
mod common;
use common::{MockEnv, create_mock_env, mock_device::EchoBehavior};

/// Shared state: MockEnv + tokio runtime (must outlive the process).
struct TestState {
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("test-model", 4, 64, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
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
    assert_eq!(pie_model::model().name(), "test-model");
}

#[test]
fn all_devices_reachable() {
    let s = state();
    s.rt.block_on(async {
        for i in 0..4 {
            let spec = pie_engine::driver::get_spec(i).await.unwrap();
            assert_eq!(spec.num_kv_pages, 64);
            assert_eq!(spec.limits.max_forward_requests, 32);
        }
    });
}

#[test]
fn tokenizer_round_trip() {
    let _ = state();
    let model = pie_model::model();
    assert_eq!(model.name(), "test-model");

    let tokens = model.tokenize("hello");
    assert!(!tokens.is_empty(), "tokenize should produce tokens");

    let text = model.detokenize(&tokens);
    assert!(!text.is_empty(), "detokenize should produce text");
}
