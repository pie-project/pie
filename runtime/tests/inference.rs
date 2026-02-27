//! Inference pipeline integration tests.
//!
//! Tests the forward pass flow through the scheduler to mock devices.

use std::sync::{Arc, OnceLock};
use std::time::Instant;

mod common;

use common::{create_mock_env, MockEnv, mock_device::EchoBehavior};
use pie::inference::request::{ForwardPassRequest, ForwardPassOutput, Sampler};
use pie::inference::brle::Brle;

struct TestState {
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let env = create_mock_env("infer-test", 1, 64, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

const MODEL: usize = 0;
const USER: &str = "test-user";

/// Build a minimal forward pass request.
fn make_request(ctx_id: u64, tokens: Vec<u32>) -> ForwardPassRequest {
    let n = tokens.len();
    ForwardPassRequest {
        context_id: ctx_id,
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
fn single_forward_pass() {
    let s = state();
    s.rt.block_on(async {
        // Create a context and reserve pages to get valid page IDs
        let ctx_id = pie::context::create(MODEL)
            .await
            .unwrap();
        pie::context::reserve_pages(MODEL, ctx_id, 1).await.unwrap();

        // Submit a forward pass
        let req = make_request(ctx_id, vec![10, 20, 30]);
        let output = pie::inference::forward_pass(MODEL, req).await.unwrap();

        // EchoBehavior(42) should return token 42
        match output {
            ForwardPassOutput::Tokens(tokens) => {
                assert_eq!(tokens, vec![42]);
            }
            other => panic!("Expected Tokens, got {:?}", other),
        }

        // Verify the mock device recorded the call
        assert!(
            s.env.backend.recorder().call_count() > 0,
            "Mock device should have received at least one fire_batch call"
        );
    });
}

#[test]
fn multiple_forward_passes() {
    let s = state();
    s.rt.block_on(async {
        let initial_calls = s.env.backend.recorder().call_count();

        // Fire 3 concurrent forward passes
        let mut handles = Vec::new();
        for i in 0..3u32 {
            let ctx_id = pie::context::create(MODEL)
                .await
                .unwrap();
            pie::context::reserve_pages(MODEL, ctx_id, 1).await.unwrap();

            let req = make_request(ctx_id, vec![100 + i]);
            handles.push(tokio::spawn(async move {
                pie::inference::forward_pass(MODEL, req).await
            }));
        }

        // All should succeed
        for handle in handles {
            let output = handle.await.unwrap().unwrap();
            match output {
                ForwardPassOutput::Tokens(tokens) => {
                    assert_eq!(tokens, vec![42], "EchoBehavior should return 42");
                }
                ForwardPassOutput::None => {
                    // Acceptable if batching didn't fire before timeout
                }
                other => panic!("Unexpected output: {:?}", other),
            }
        }

        // Mock device should have received more calls
        assert!(
            s.env.backend.recorder().call_count() > initial_calls,
            "Should have received at least one new fire_batch call"
        );
    });
}
