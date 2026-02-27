//! E2E KV cache contention tests.
//!
//! Exercises multi-GPU contention resolution using real WASM inferlets
//! (the `generate` test inferlet) through the full workflow stack.
//!
//! Setup: 2 GPUs, 8 pages each, page_size=16 (128 tokens/device).
//! The `generate` inferlet runs fill → flush → generate(5 steps),
//! allocating KV pages. Multiple concurrent processes naturally
//! exhaust pages and trigger eviction, wait queue, and restore paths.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::{create_mock_env, MockEnv, mock_device::EchoBehavior, inferlets};

use pie::workflow;

const GENERATE: &str = "generate@0.1.0";

/// Timeout for a workflow to complete. Generous to allow contention resolution.
const WORKFLOW_TIMEOUT: Duration = Duration::from_secs(30);

/// Shared state: MockEnv + tokio runtime.
struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        inferlets::build_inferlets();

        let rt = tokio::runtime::Runtime::new().unwrap();
        // 2 GPUs, 8 pages each — tight budget to force contention
        let env = create_mock_env("contention-model", 2, 8, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
            // Pre-install the generate inferlet so tests don't race on build
            inferlets::add_and_install("generate").await;
        });
        TestState { env, rt }
    })
}

/// Submit a workflow and await its result, panicking with a descriptive message on failure.
async fn submit_and_await(label: &str, username: &str, json: &str) -> String {
    let (id, rx) = workflow::submit(username, json, None).await
        .unwrap_or_else(|e| panic!("[{label}] submit failed: {e}"));

    match tokio::time::timeout(WORKFLOW_TIMEOUT, rx).await {
        Ok(Ok(Ok(output))) => {
            eprintln!("[{label}] workflow {id} completed: {output}");
            output
        }
        Ok(Ok(Err(e))) => panic!("[{label}] workflow {id} failed: {e}"),
        Ok(Err(_)) => panic!("[{label}] workflow {id} channel dropped"),
        Err(_) => panic!("[{label}] workflow {id} timed out — contention may be deadlocked"),
    }
}

/// Build a Fork JSON expression with `n` concurrent generate branches.
fn fork_json(n: usize) -> String {
    let branches: Vec<String> = (0..n)
        .map(|_| format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#))
        .collect();
    format!(r#"{{"type":"fork","branches":[{}]}}"#, branches.join(","))
}

/// Build a Pipe JSON expression: generate → generate (serial chain of `n` stages).
fn pipe_json(n: usize) -> String {
    let stages: Vec<String> = (0..n)
        .map(|_| format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#))
        .collect();
    format!(r#"{{"type":"pipe","stages":[{}]}}"#, stages.join(","))
}

// =============================================================================
// Test 1: Concurrent independent workflows resolve contention
// =============================================================================

/// Two independent workflows, each running `generate`, submitted concurrently.
/// Both should complete despite only 8 pages per GPU — contention is resolved
/// via the invested-importance eviction policy and per-device wait queues.
#[test]
fn concurrent_workflows_resolve_contention() {
    let s = state();
    s.rt.block_on(async {
        let json = format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#);

        let (f1, f2) = tokio::join!(
            submit_and_await("wf1", "user-a", &json),
            submit_and_await("wf2", "user-b", &json),
        );
        assert!(!f1.is_empty());
        assert!(!f2.is_empty());
    });
}

// =============================================================================
// Test 2: Fork workflow with 3 concurrent generate branches
// =============================================================================

/// A single workflow using Fork to spawn 3 concurrent `generate` branches.
/// All branches share the same 8-page GPU budget, forcing the arbiter to
/// serialize access via eviction and wait queue management.
/// All 3 branches must complete — the result should be a 3-element array.
#[test]
fn fork_workflow_contention() {
    let s = state();
    s.rt.block_on(async {
        let json = fork_json(3);
        let output = submit_and_await("fork3", "fork-user", &json).await;

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_array(), "Fork result should be an array, got: {parsed}");
        assert_eq!(parsed.as_array().unwrap().len(), 3, "expected 3 results");
    });
}

// =============================================================================
// Test 3: Sequential workflows after cleanup
// =============================================================================

/// Submit a workflow, wait for completion, then submit a second.
/// The second workflow should reuse pages freed by the first —
/// no contention, no stale arbiter/wait-queue state.
#[test]
fn sequential_workflows_after_eviction() {
    let s = state();
    s.rt.block_on(async {
        let json = format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#);
        submit_and_await("seq1", "seq-user", &json).await;
        submit_and_await("seq2", "seq-user", &json).await;
    });
}

// =============================================================================
// Test 4: High fan-out fork (8 concurrent branches on 8-page GPUs)
// =============================================================================

/// 8 concurrent generate processes sharing 2 GPUs × 8 pages.
/// This is extreme contention: each generate needs pages and all 8 compete.
/// The arbiter must evict and rotate efficiently. All 8 must complete.
#[test]
fn high_fanout_fork() {
    let s = state();
    s.rt.block_on(async {
        let json = fork_json(8);
        let output = submit_and_await("fork8", "fanout-user", &json).await;

        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed.as_array().unwrap().len(), 8, "expected 8 results");
    });
}

// =============================================================================
// Test 5: Five independent concurrent workflows
// =============================================================================

/// 5 independent workflows submitted simultaneously, each running generate.
/// Cross-workflow contention: different workflow_ids → different arbiter weights.
/// All 5 must complete, exercising the wait queue priority ordering.
#[test]
fn five_concurrent_workflows() {
    let s = state();
    s.rt.block_on(async {
        let json = format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#);

        let params: Vec<_> = (0..5)
            .map(|i| (format!("multi{i}"), format!("user-{i}")))
            .collect();
        let futures: Vec<_> = params.iter()
            .map(|(label, user)| submit_and_await(label, user, &json))
            .collect();

        let results = futures::future::join_all(futures).await;
        for (i, r) in results.iter().enumerate() {
            assert!(!r.is_empty(), "workflow {i} produced empty output");
        }
    });
}

// =============================================================================
// Test 6: Wave stress — 3 waves of 4 concurrent workflows
// =============================================================================

/// Submits 3 successive waves of 4 concurrent workflows.
/// Each wave starts after the previous wave completes.
/// Tests that cleanup between waves is correct — no leaked arbiter nodes,
/// no stale wait queue entries, no page pool corruption.
#[test]
fn wave_stress() {
    let s = state();
    s.rt.block_on(async {
        let json = format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#);

        for wave in 0..3 {
            let params: Vec<_> = (0..4)
                .map(|i| (format!("wave{wave}_{i}"), format!("wave-{wave}-user-{i}")))
                .collect();
            let futures: Vec<_> = params.iter()
                .map(|(label, user)| submit_and_await(label, user, &json))
                .collect();

            let results = futures::future::join_all(futures).await;
            for (i, r) in results.iter().enumerate() {
                assert!(!r.is_empty(), "wave {wave} workflow {i} produced empty output");
            }
            eprintln!("[wave_stress] wave {wave} complete ({} workflows)", results.len());
        }
    });
}

// =============================================================================
// Test 7: Two concurrent fork workflows (6 simultaneous processes)
// =============================================================================

/// Two workflows, each forking 3 generate branches, submitted concurrently.
/// 6 processes total compete on 2 GPUs × 8 pages, exercising cross-workflow
/// contention with intra-workflow Fork concurrency. Both workflows must
/// complete with all 3 branches producing results.
#[test]
fn concurrent_fork_workflows() {
    let s = state();
    s.rt.block_on(async {
        let json = fork_json(3);

        let (out1, out2) = tokio::join!(
            submit_and_await("cfork1", "user-x", &json),
            submit_and_await("cfork2", "user-y", &json),
        );

        let p1: serde_json::Value = serde_json::from_str(&out1).unwrap();
        let p2: serde_json::Value = serde_json::from_str(&out2).unwrap();
        assert_eq!(p1.as_array().unwrap().len(), 3, "fork1 should have 3 results");
        assert_eq!(p2.as_array().unwrap().len(), 3, "fork2 should have 3 results");
    });
}

// =============================================================================
// Test 8: Mixed pipe + fork — sequential generation feeding into fan-out
// =============================================================================

/// A workflow that pipes into a fork: pipe(generate, fork(generate, generate)).
/// The first generate runs alone, then its output feeds two concurrent generates.
/// Exercises serial-then-parallel contention patterns.
#[test]
fn pipe_then_fork() {
    let s = state();
    s.rt.block_on(async {
        let json = format!(r#"{{
            "type": "pipe",
            "stages": [
                {{"type": "process", "program_name": "{GENERATE}"}},
                {{"type": "fork", "branches": [
                    {{"type": "process", "program_name": "{GENERATE}"}},
                    {{"type": "process", "program_name": "{GENERATE}"}},
                    {{"type": "process", "program_name": "{GENERATE}"}}
                ]}}
            ]
        }}"#);

        let output = submit_and_await("pipe_fork", "pf-user", &json).await;
        let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_array(), "pipe→fork result should be an array");
        assert_eq!(parsed.as_array().unwrap().len(), 3, "expected 3 fork results");
    });
}

// =============================================================================
// Test 9: Rapid-fire burst — 10 independent workflows as fast as possible
// =============================================================================

/// Submits 10 workflows in rapid succession without waiting between submits.
/// All 10 must complete. Exercises the arbiter and wait queue under maximum
/// queueing pressure from many independent workflows.
#[test]
fn rapid_fire_burst() {
    let s = state();
    s.rt.block_on(async {
        let json = format!(r#"{{"type":"process","program_name":"{GENERATE}"}}"#);

        // Submit all 10 as fast as possible
        let mut receivers = Vec::with_capacity(10);
        for i in 0..10 {
            let (id, rx) = workflow::submit(&format!("burst-{i}"), &json, None).await.unwrap();
            eprintln!("[burst] submitted workflow {i}: {id}");
            receivers.push((i, id, rx));
        }

        // Now await all
        for (i, id, rx) in receivers {
            match tokio::time::timeout(WORKFLOW_TIMEOUT, rx).await {
                Ok(Ok(Ok(output))) => {
                    eprintln!("[burst] workflow {i} ({id}) completed: {output}");
                }
                Ok(Ok(Err(e))) => panic!("[burst] workflow {i} ({id}) failed: {e}"),
                Ok(Err(_)) => panic!("[burst] workflow {i} ({id}) channel dropped"),
                Err(_) => panic!("[burst] workflow {i} ({id}) timed out"),
            }
        }
    });
}

