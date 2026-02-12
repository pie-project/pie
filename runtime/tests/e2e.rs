//! End-to-end inferlet execution tests.
//!
//! These tests actually **run** real WASM inferlets through the full stack:
//! program add → install → process::spawn() → linker instantiation → WASM
//! execution → process completion.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::{create_mock_env, MockEnv, mock_device::EchoBehavior, inferlets};

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
fn spawn_and_wait(
    s: &TestState,
    name: &str,
    args: Vec<String>,
) -> bool {
    let pid = s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        process::spawn(
            "test-user".into(),
            program_name(name),
            args,
            None,
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

// =============================================================================
// Basic E2E Tests
// =============================================================================

#[test]
fn echo_runs_to_completion() {
    let s = state();
    assert!(
        spawn_and_wait(s, "echo", vec!["hello".into(), "world".into()]),
        "echo inferlet should complete within timeout"
    );
}

#[test]
fn error_inferlet_exits() {
    let s = state();
    assert!(
        spawn_and_wait(s, "error", vec![]),
        "error inferlet should complete even on error"
    );
}

#[test]
fn context_inferlet_exercises_host_apis() {
    let s = state();
    assert!(
        spawn_and_wait(s, "context", vec![]),
        "context inferlet should complete (exercises model, tokenizer, context host APIs)"
    );
}

// =============================================================================
// Stress & Concurrency
// =============================================================================

#[test]
fn concurrent_spawns() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        let count = 10;
        let pids: Vec<_> = (0..count)
            .map(|i| {
                process::spawn(
                    "stress-user".into(),
                    program_name("echo"),
                    vec![format!("batch-{i}")],
                    None,
                    None,
                    false,
                    None,
                )
                .unwrap_or_else(|e| panic!("spawn {i} failed: {e}"))
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
            assert!(completed, "concurrent echo {i} (pid {pid}) did not complete");
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
                vec![format!("seq-{i}")],
                None,
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
            assert!(completed, "sequential echo {i} (pid {pid}) did not complete");
        }

        // After all processes finish, the process list should not grow unboundedly
        let remaining = process::list().len();
        assert!(remaining < 5, "expected few residual processes, got {remaining}");
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
            let (name, args) = if i % 2 == 0 {
                ("echo", vec![format!("ok-{i}")])
            } else {
                ("error", vec![])
            };
            let pid = process::spawn(
                "mixed-user".into(),
                program_name(name),
                args,
                None,
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

#[test]
fn spawn_after_termination() {
    let s = state();
    s.rt.block_on(async {
        inferlets::add_and_install("echo").await;

        // Spawn and immediately terminate
        let pid1 = process::spawn(
            "term-user".into(),
            program_name("echo"),
            vec!["will-be-terminated".into()],
            None,
            None,
            false,
            None,
        )
        .expect("spawn for termination");

        process::terminate(pid1, Some("test termination".into()));

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
            vec!["after-termination".into()],
            None,
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
        assert!(completed, "process spawned after termination should complete normally");
    });
}
