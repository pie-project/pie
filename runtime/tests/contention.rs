//! E2E KV-cache contention tests.
//!
//! Exercises KV page contention using real WASM inferlets (the `generate`
//! test inferlet) driven through direct, concurrent `process::spawn`.
//!
//! The workflow DSL (Fork/Pipe) is gone, so the old fork/pipe harness is
//! re-expressed as direct process spawns: a Fork of N branches becomes N
//! processes spawned concurrently; a Pipe becomes a serial spawn→await chain.
//! Ordering is governed by the single-model FCFS policy — each process is
//! tagged with a monotonic launch sequence; eviction prefers the
//! newest-launched victim and the restore queue serves the oldest first.
//!
//! Setup: 2 GPUs, 8 pages each, page_size=16. The `generate` inferlet runs
//! fill → flush → generate(5 steps), allocating KV pages. Many concurrent
//! processes share the tight page budget, exercising eviction, the wait
//! queue, and restore. Every process must complete (no deadlock) within the
//! timeout — that completion is the contention-resolution assertion.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use tokio::sync::oneshot;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::EchoBehavior};

use pie::process;
use pie::program::ProgramName;

/// Name of the test inferlet (installed at version 0.1.0).
const GENERATE: &str = "generate";

/// Timeout for a single process to complete. Generous to allow contention
/// resolution (eviction → wait → restore) under a tight page budget.
const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);

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
        // 2 GPUs, 8 pages each — tight budget to force contention.
        let env = create_mock_env("contention-model", 2, 8, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
            // Pre-install the generate inferlet so tests don't race on build.
            inferlets::add_and_install(GENERATE).await;
        });
        TestState { env, rt }
    })
}

fn program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

/// Spawn a `generate` process and return a receiver for its final result.
/// The process begins running immediately (FCFS launch-seq is assigned at
/// spawn); callers collect receivers to keep processes concurrent.
fn spawn_generate(username: &str, input: &str) -> oneshot::Receiver<Result<String, String>> {
    let (tx, rx) = oneshot::channel();
    process::spawn(
        username.to_string(),
        program_name(GENERATE),
        input.to_string(),
        None,  // client_id
        false, // capture_outputs (result still delivered via result_tx)
        Some(tx),
    )
    .unwrap_or_else(|e| panic!("[{username}] spawn failed: {e}"));
    rx
}

/// Await a single process result, panicking descriptively on error or timeout.
async fn await_result(label: &str, rx: oneshot::Receiver<Result<String, String>>) -> String {
    match tokio::time::timeout(PROCESS_TIMEOUT, rx).await {
        Ok(Ok(Ok(output))) => {
            eprintln!("[{label}] completed: {output}");
            output
        }
        Ok(Ok(Err(e))) => panic!("[{label}] process failed: {e}"),
        Ok(Err(_)) => panic!("[{label}] result channel dropped"),
        Err(_) => panic!("[{label}] timed out — contention may be deadlocked"),
    }
}

/// Spawn `n` `generate` processes concurrently (distinct users), await all,
/// and assert every one completes with non-empty output. Returns the outputs.
///
/// This is the direct-spawn replacement for a Fork of `n` branches.
async fn fan_out(label: &str, n: usize) -> Vec<String> {
    // Spawn all first so they run concurrently and contend for pages.
    let receivers: Vec<_> = (0..n)
        .map(|i| {
            (
                format!("{label}-{i}"),
                spawn_generate(&format!("{label}-user-{i}"), &format!(r#"{{"branch":"{i}"}}"#)),
            )
        })
        .collect();

    let mut outputs = Vec::with_capacity(n);
    for (lbl, rx) in receivers {
        let out = await_result(&lbl, rx).await;
        assert!(!out.is_empty(), "[{lbl}] produced empty output");
        outputs.push(out);
    }
    outputs
}

// =============================================================================
// Test 1: Two concurrent independent processes resolve contention
// =============================================================================

/// Two independent `generate` processes (different users) spawned concurrently.
/// Both must complete despite only 8 pages per GPU — contention is resolved
/// via FCFS eviction and per-device wait/restore queues.
#[test]
fn concurrent_processes_resolve_contention() {
    let s = state();
    s.rt.block_on(async {
        let r1 = spawn_generate("user-a", r#"{"n":1}"#);
        let r2 = spawn_generate("user-b", r#"{"n":2}"#);
        let o1 = await_result("p1", r1).await;
        let o2 = await_result("p2", r2).await;
        assert!(!o1.is_empty());
        assert!(!o2.is_empty());
    });
}

// =============================================================================
// Test 2: Fan-out of 3 concurrent processes
// =============================================================================

/// Three `generate` processes spawned concurrently (replaces a Fork of 3).
/// All share the 8-page GPU budget, forcing serialization via eviction and
/// the wait queue. All 3 must complete.
#[test]
fn fan_out_contention() {
    let s = state();
    s.rt.block_on(async {
        let outs = fan_out("fan3", 3).await;
        assert_eq!(outs.len(), 3, "expected 3 completed processes");
    });
}

// =============================================================================
// Test 3: Sequential processes after cleanup
// =============================================================================

/// Run one process to completion, then a second. The second should reuse
/// pages freed by the first — no stale wait-queue / page-pool state.
#[test]
fn sequential_processes_after_eviction() {
    let s = state();
    s.rt.block_on(async {
        let o1 = await_result("seq1", spawn_generate("seq-user", r#"{"seq":1}"#)).await;
        assert!(!o1.is_empty());
        let o2 = await_result("seq2", spawn_generate("seq-user", r#"{"seq":2}"#)).await;
        assert!(!o2.is_empty());
    });
}

// =============================================================================
// Test 4: High fan-out (8 concurrent processes on 8-page GPUs)
// =============================================================================

/// Eight concurrent `generate` processes sharing 2 GPUs × 8 pages. Extreme
/// contention: the FCFS arbiter must evict (newest-first) and restore
/// (oldest-first) to drain all 8. All must complete.
#[test]
fn high_fanout() {
    let s = state();
    s.rt.block_on(async {
        let outs = fan_out("fan8", 8).await;
        assert_eq!(outs.len(), 8, "expected 8 completed processes");
    });
}

// =============================================================================
// Test 5: Five independent concurrent processes
// =============================================================================

/// Five independent `generate` processes spawned simultaneously. Exercises
/// the wait-queue ordering across distinct launch sequences.
#[test]
fn five_concurrent_processes() {
    let s = state();
    s.rt.block_on(async {
        let outs = fan_out("multi", 5).await;
        assert_eq!(outs.len(), 5);
    });
}

// =============================================================================
// Test 6: Wave stress — 3 waves of 4 concurrent processes
// =============================================================================

/// Three successive waves of 4 concurrent processes; each wave starts after
/// the previous completes. Tests that cleanup between waves is correct — no
/// leaked wait-queue entries or page-pool corruption.
#[test]
fn wave_stress() {
    let s = state();
    s.rt.block_on(async {
        for wave in 0..3 {
            let outs = fan_out(&format!("wave{wave}"), 4).await;
            assert_eq!(outs.len(), 4, "wave {wave} should complete 4 processes");
            eprintln!("[wave_stress] wave {wave} complete (4 processes)");
        }
    });
}

// =============================================================================
// Test 7: Two concurrent fan-outs (6 simultaneous processes)
// =============================================================================

/// Six processes (two logical batches of 3) all spawned concurrently,
/// competing on 2 GPUs × 8 pages. All must complete.
#[test]
fn two_concurrent_fans() {
    let s = state();
    s.rt.block_on(async {
        let mut receivers = Vec::new();
        for batch in 0..2 {
            for i in 0..3 {
                receivers.push((
                    format!("fan{batch}-{i}"),
                    spawn_generate(&format!("user-{batch}-{i}"), r#"{}"#),
                ));
            }
        }
        let mut completed = 0;
        for (label, rx) in receivers {
            let out = await_result(&label, rx).await;
            assert!(!out.is_empty());
            completed += 1;
        }
        assert_eq!(completed, 6, "all 6 concurrent processes should complete");
    });
}

// =============================================================================
// Test 8: Serial-then-fan — sequential process feeding into a fan-out
// =============================================================================

/// Re-expresses pipe(generate, fork(generate × 3)): one process runs alone,
/// then three run concurrently. Exercises serial-then-parallel contention.
#[test]
fn serial_then_fan() {
    let s = state();
    s.rt.block_on(async {
        let first = await_result("serial", spawn_generate("sf-user", r#"{}"#)).await;
        assert!(!first.is_empty());

        let outs = fan_out("sf-fan", 3).await;
        assert_eq!(outs.len(), 3, "expected 3 fan-out processes");
    });
}

// =============================================================================
// Test 9: Rapid-fire burst — 10 processes spawned as fast as possible
// =============================================================================

/// Spawn 10 processes in rapid succession without waiting between spawns.
/// All 10 must complete — maximum queueing pressure on the wait/restore path.
#[test]
fn rapid_fire_burst() {
    let s = state();
    s.rt.block_on(async {
        let receivers: Vec<_> = (0..10)
            .map(|i| (i, spawn_generate(&format!("burst-{i}"), r#"{}"#)))
            .collect();

        for (i, rx) in receivers {
            let out = await_result(&format!("burst-{i}"), rx).await;
            assert!(!out.is_empty(), "burst {i} produced empty output");
        }
    });
}
