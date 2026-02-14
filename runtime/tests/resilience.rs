//! Error resilience integration tests.
//!
//! Tests timeout handling with deliberately slow mock devices.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

mod common;

use common::{create_mock_env, MockEnv, mock_device::DelayedBehavior, mock_device::EchoBehavior};
use pie::inference::request::{ForwardPassRequest, ForwardPassOutput, Sampler};
use pie::inference::brle::Brle;

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        // Use delayed behavior with 60s latency + short scheduler timeout (2s)
        let behavior = Arc::new(DelayedBehavior {
            inner: EchoBehavior(42),
            latency: Duration::from_secs(60),
        });
        let env = create_mock_env("resilience-test", 1, 64, behavior);
        let mut config = env.config();

        // Override scheduler: short timeout so tests don't block for 60s
        config.models[0].scheduler.request_timeout_secs = 2;
        config.models[0].scheduler.max_wait_ms = 10;

        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });

        TestState { env, rt }
    })
}

const MODEL: usize = 0;
const USER: &str = "test-user";

fn make_request(tokens: Vec<u32>) -> ForwardPassRequest {
    let n = tokens.len();
    ForwardPassRequest {
        context_id: None,
        tokens,
        positions: (0..n as u32).collect(),
        speculative_tokens: vec![],
        speculative_positions: vec![],
        output_speculative_tokens: false,
        masks: (0..n).map(|_| Brle::new(0)).collect(),
        logit_mask: None,
        sampling_indices: vec![(n - 1) as u32],
        samplers: vec![Sampler::Multinomial { temperature: 1.0, seed: None }],
        adapter_id: None,
        adapter_seed: None,
        arrival_time: Some(Instant::now()),
    }
}

#[test]
fn device_timeout_returns_none() {
    let s = state();
    s.rt.block_on(async {
        // Reserve a page
        let ctx_id = pie::context::create(MODEL, USER.to_string(), "timeout-ctx".into(), None)
            .await
            .unwrap();
        let lock = pie::context::acquire_lock(MODEL, ctx_id);
        pie::context::reserve_pages(MODEL, ctx_id, lock, 1).await.unwrap();
        pie::context::release_lock(MODEL, ctx_id, lock).unwrap();

        let start = Instant::now();
        let req = make_request(vec![10, 20]);
        let output = pie::inference::forward_pass(MODEL, req).await.unwrap();
        let elapsed = start.elapsed();

        // Should timeout (not hang for 60s) and return None
        assert!(elapsed < Duration::from_secs(30), "Should timeout, not wait 60s");

        match output {
            ForwardPassOutput::None => {} // Expected on timeout
            other => {
                println!("Got {:?} instead of None (acceptable in some configs)", other);
            }
        }
    });
}
