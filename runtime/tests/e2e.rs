//! End-to-end inferlet execution tests.
//!
//! These tests actually **run** real WASM inferlets through the full stack:
//! program add → install → process::spawn() → linker instantiation → WASM
//! execution → process completion.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::EchoBehavior};

use pie::process;
use pie::program::ProgramName;

/// Timeout for a single process to complete.
const PROCESS_TIMEOUT: Duration = Duration::from_secs(10);

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
        let env = create_mock_env("test-model", 1, 16, Arc::new(EchoBehavior(42)));
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

fn program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

/// Spawn a process within the tokio runtime and wait for it to complete.
/// Returns true if the process exited within the timeout.
fn spawn_and_wait(s: &TestState, name: &str, input: String) -> bool {
    let pid = s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        process::spawn(
            "test-user".into(),
            program_name(name),
            input,
            None,
            false,
            None,
        )
        .expect("spawn")
    });

    // Wait in a tokio context so the WASM task can make progress
    s.rt.block_on(async {
        tokio::time::timeout(PROCESS_TIMEOUT, async {
            loop {
                if !process::list().contains(&pid) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok()
    })
}

/// Spawn a process and capture its actual return value (the inferlet's
/// `Result<String, String>`), so a test can assert the inferlet ran to a
/// meaningful result rather than merely exiting. `Err` if the process did not
/// terminate within the timeout (e.g. a hang on an unresolved host RPC).
fn spawn_and_capture(s: &TestState, name: &str, input: String) -> Result<String, String> {
    let rx = s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        let (tx, rx) = tokio::sync::oneshot::channel();
        process::spawn(
            "test-user".into(),
            program_name(name),
            input,
            None,
            false,
            Some(tx),
        )
        .expect("spawn");
        rx
    });

    s.rt.block_on(async {
        match tokio::time::timeout(PROCESS_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err("process result channel dropped before completion".to_string()),
            Err(_) => Err("process did not complete within timeout (hang)".to_string()),
        }
    })
}

// =============================================================================
// Basic E2E Tests
// =============================================================================

#[test]
fn echo_runs_to_completion() {
    let s = state();
    assert!(
        spawn_and_wait(s, "echo", r#"{"message":"hello world"}"#.into()),
        "echo inferlet should complete within timeout"
    );
}

#[test]
fn error_inferlet_exits() {
    let s = state();
    assert!(
        spawn_and_wait(s, "error", "{}".into()),
        "error inferlet should complete even on error"
    );
}

#[test]
fn context_inferlet_exercises_host_apis() {
    let s = state();
    assert!(
        spawn_and_wait(s, "context", "{}".into()),
        "context inferlet should complete (exercises model, tokenizer, context host APIs)"
    );
}

// The forward-pass / generate pipeline runs e2e on the mock driver: the
// `generate` inferlet decodes via DIRECT WIT bindings (In Gim's directive) —
// raw `inference::ForwardPass` (kv-working-set → input-tokens → sampler →
// execute → output) + `working_set::KvWorkingSet`, no `Context`/`Generator`/
// `collect_tokens` sugar — over the mock `EchoBehavior(42)` (every step returns
// token 42, so 5 steps yield five 42s). We assert the exact returned string so
// the test cannot silently pass on an early exit/error.
#[test]
fn generate_inferlet_exercises_forward_pass() {
    let s = state();
    let result = spawn_and_capture(s, "generate", "{}".into());
    assert_eq!(
        result.as_deref(),
        Ok("generated 5 tokens: [42, 42, 42, 42, 42]"),
        "generate inferlet should run the forward-pass loop end-to-end and \
         return the mock's five echoed tokens (got {result:?})"
    );
}

/// The **linear-model fold-commit surface** runs e2e (echo — In Gim's
/// linear-model WIT revisit). The `linearfold` inferlet drives a MODEL-AGNOSTIC
/// spec-commit loop: the new `model::is_linear()` capability gates a
/// `Forward::rs_working_set` + `Forward::fold(n_acc)` (the linear-model COMMIT —
/// fold only the accepted prefix into the recurrent state). On the attention
/// mock (`is_linear() == false`, all-0 RS caps, fold-granularity 1 per
/// bootstrap.rs) the fold branch is skipped and the pass is an ordinary
/// prefill. Asserts the new `model.is-linear()` + RS shaping caps surface
/// correctly through the full stack (WIT → host binding → SDK), and that the
/// model-agnostic loop executes cleanly. (The linear branch's fold lowering —
/// `rs_fold_lens` + `RS_FLAG_FOLD`, api/inference.rs — rides the existing host
/// path; its full linear-model e2e is the deferred Phase-6/7 RS harness.)
#[test]
fn linearfold_inferlet_exercises_is_linear_surface() {
    let s = state();
    let result = spawn_and_capture(s, "linearfold", "1".into());
    assert_eq!(
        result.as_deref(),
        Ok(
            "linearfold: is_linear=false rs_state_size=0 rs_fold_granularity=1 \
rs_buffer_page_size=0 n_acc=1"
        ),
        "linearfold should surface the new model.is-linear() capability + RS caps \
         e2e; the attention mock reports non-linear with all-0 RS caps (got {result:?})"
    );
}

/// The **sampler-lowering keep-core primitive** runs e2e (echo). The
/// `samplerprobe` inferlet gets a PARAMETRIC top-p sampler on the RAW WIT
/// surface via `sampler::sampler_program` — the standard-sampler spec lowered to
/// an attachable `tensor::Program` + its per-fire `InputBinding`s (logits row +
/// the temperature/top-p submit param tensors), with `geometry::*` for the KV
/// page split and a hand-written decode loop over `ForwardPass`. No hand-built
/// Sampling-IR `Graph`, no `Context`/`Generator` facade. This is the sampler
/// analog of the geometry/carrier keep-core exercises: it proves the full
/// lowering + parametric binding-resolution path (incl. `tensor::from_data`
/// submit tensors) runs end-to-end. On the mock `EchoBehavior(42)` every fire
/// echoes token 42, so 3 steps yield `[42, 42, 42]`.
#[test]
fn samplerprobe_inferlet_exercises_sampler_lowering() {
    let s = state();
    let result = spawn_and_capture(s, "samplerprobe", "{}".into());
    assert_eq!(
        result.as_deref(),
        Ok("sampled 3 tokens: [42, 42, 42]"),
        "samplerprobe should lower a top-p sampler via the keep-core \
         sampler::sampler_program primitive and decode three mock-echoed \
         tokens (got {result:?})"
    );
}

/// The **unified carrier × parametric-sampler keep-core path** runs e2e (echo).
/// The `carrierprobe` inferlet drives a PARAMETRIC top-p sampler through the
/// RUN-AHEAD carrier via `carrier::submit_pass` taking a `LoweredSampler` — the
/// capability that did not exist until the unified signature (the old carrier
/// hardwired `[Logits]`, dropping a parametric sampler's T/p/k submit tensors →
/// `CustomJIT` + wrong sampling, `ptir-carrier-bind-seam-spec §9`). Now greedy
/// (`Argmax`) and parametric share ONE carrier path, and the run-ahead pipeline
/// (eager consumer submit + device carrier inject) composes with the full
/// sampler binding list. On the mock `EchoBehavior(42)` a 3-step pipelined
/// decode yields `[42, 42, 42]`.
#[test]
fn spawn_after_termination() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        // Spawn and immediately terminate
        let pid1 = process::spawn(
            "term-user".into(),
            program_name("echo"),
            r#"{"msg":"will-be-terminated"}"#.into(),
            None,
            false,
            None,
        )
        .expect("spawn for termination");

        process::terminate(pid1, Err("test termination".into()));

        let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
            loop {
                if !process::list().contains(&pid1) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok();
        assert!(completed, "terminated process should disappear");

        // Spawn another — should work fine, no stale state
        let pid2 = process::spawn(
            "term-user".into(),
            program_name("echo"),
            r#"{"msg":"after-termination"}"#.into(),
            None,
            false,
            None,
        )
        .expect("spawn after termination");

        let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
            loop {
                if !process::list().contains(&pid2) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .is_ok();
        assert!(
            completed,
            "process spawned after termination should complete normally"
        );
    });
}
#[test]
fn isolatedtopp_pipelined_runs() {
    let s = state();
    let result = spawn_and_capture(s, "isolatedtopp-pipelined", "{}".into());
    assert_eq!(
        result.as_deref(),
        Ok("{\"tokens\": [42, 42, 42, 42]}"),
        "isolatedtopp-pipelined should run the run-ahead TopP decode (got {result:?})"
    );
}

#[test]
fn multisamp_pipelined_runs() {
    let s = state();
    let result = spawn_and_capture(s, "multisamp-pipelined", "{}".into());
    // 4 kinds (topk/topp/minp/joint) × 4 tokens = 16 echoed 42s.
    assert_eq!(
        result.as_deref(),
        Ok("{\"tokens\": [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]}"),
        "multisamp-pipelined should drive all 4 parametric kinds through the carrier (got {result:?})"
    );
}

#[test]
fn tempgen_pipelined_runs() {
    let s = state();
    let result = spawn_and_capture(s, "tempgen-pipelined", "{}".into());
    assert_eq!(
        result.as_deref(),
        Ok("{\"tokens\": [42, 42, 42, 42, 42, 42, 42, 42]}"),
        "tempgen-pipelined should run the run-ahead Multinomial decode (got {result:?})"
    );
}

// =============================================================================
// Stress & Concurrency
// =============================================================================

/// **Host-side concurrent-DECODE repro** (bravo, for alpha/charlie's arena bug).
/// Spawns N concurrent decodes that each drive a real forward-pass loop (KV
/// working set → arena txns). Uses the DIRECT-WIT-binding `generate` inferlet
/// (In Gim's directive) so it exercises the raw WIT path the M3 inferlets use.
/// On `EchoBehavior(42)` every pipeline MUST return the identical
/// `[42, 42, 42, 42, 42]`; cross-request contamination / arena `unknown object`
/// shows as a divergent or errored return. Fast host mirror of `cuda_bubble.rs`.
#[test]
fn concurrent_decode_fleet() {
    let s = state();
    const FLEET: usize = 8;
    let results: Vec<Result<String, String>> = s.rt.block_on(async {
        inferlets::add_and_install("generate").await;
        // Launch the whole fleet BEFORE awaiting any → all decode concurrently,
        // co-batching in the scheduler (the condition that triggers the bug).
        let rxs: Vec<_> = (0..FLEET)
            .map(|i| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "fleet-user".into(),
                    program_name("generate"),
                    format!(r#"{{"lane":{i}}}"#),
                    None,
                    false,
                    Some(tx),
                )
                .unwrap_or_else(|e| panic!("spawn {i}: {e}"));
                rx
            })
            .collect();
        let mut out = Vec::with_capacity(FLEET);
        for rx in rxs {
            out.push(match tokio::time::timeout(PROCESS_TIMEOUT, rx).await {
                Ok(Ok(r)) => r,
                Ok(Err(_)) => Err("result channel dropped".into()),
                Err(_) => Err("timeout".into()),
            });
        }
        out
    });

    const EXPECT: &str = "generated 5 tokens: [42, 42, 42, 42, 42]";
    let mut ok = 0usize;
    for (i, r) in results.iter().enumerate() {
        match r {
            Ok(s) if s == EXPECT => ok += 1,
            other => eprintln!("[fleet] pipeline {i} diverged: {other:?}"),
        }
    }
    eprintln!("[fleet] {ok}/{FLEET} pipelines produced the identical greedy stream");
    assert_eq!(
        ok, FLEET,
        "concurrent decode must be deterministic across pipelines (arena/batching \
         contamination if not) — {ok}/{FLEET} correct"
    );
}

#[test]
fn concurrent_spawns() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        let count = 10;
        let pids: Vec<_> = (0..count)
            .map(|i| {
                let pid = process::spawn(
                    "stress-user".into(),
                    program_name("echo"),
                    format!(r#"{{"batch":"{i}"}}"#),
                    None,
                    false,
                    None,
                )
                .unwrap_or_else(|e| panic!("spawn {i} failed: {e}"));
                pid
            })
            .collect();

        for (i, pid) in pids.iter().enumerate() {
            let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
                loop {
                    if !process::list().contains(pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .is_ok();
            assert!(
                completed,
                "concurrent echo {i} (pid {pid}) did not complete"
            );
        }
    });
}

#[test]
fn rapid_sequential_spawns() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        for i in 0..50 {
            let pid = process::spawn(
                "seq-user".into(),
                program_name("echo"),
                format!(r#"{{"seq":"{i}"}}"#),
                None,
                false,
                None,
            )
            .unwrap_or_else(|e| panic!("sequential spawn {i} failed: {e}"));

            let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
                loop {
                    if !process::list().contains(&pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .is_ok();
            assert!(
                completed,
                "sequential echo {i} (pid {pid}) did not complete"
            );
        }

        // After all processes finish, the process list should not grow unboundedly
        let remaining = process::list().len();
        assert!(
            remaining < 5,
            "expected few residual processes, got {remaining}"
        );
    });
}

#[test]
fn mixed_success_and_error() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;
        inferlets::add_and_install("error").await;

        let mut pids = Vec::new();
        for i in 0..10 {
            let (name, input) = if i % 2 == 0 {
                ("echo", format!(r#"{{"msg":"ok-{i}"}}"#))
            } else {
                ("error", "{}".to_string())
            };
            let pid = process::spawn(
                "mixed-user".into(),
                program_name(name),
                input,
                None,
                false,
                None,
            )
            .unwrap_or_else(|e| panic!("mixed spawn {i} ({name}) failed: {e}"));
            pids.push((i, name, pid));
        }

        for (i, name, pid) in &pids {
            let completed = tokio::time::timeout(PROCESS_TIMEOUT, async {
                loop {
                    if !process::list().contains(pid) {
                        return;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .is_ok();
            assert!(completed, "mixed {i} ({name}, pid {pid}) did not complete");
        }
    });
}

