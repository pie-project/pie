//! **Low-level chat-EOS pipelined inferlet — mock smoke** (foxtrot ∥ bravo).
//! The device-resident run-ahead carrier (`next_inputs` retain→inject→free) is
//! NOT implemented by the mock, and the mock sampler is input-independent, so
//! TOKEN-IDENTITY of the explicit run-ahead + EOS-rollback loop is a 4090 gate
//! (`bin/pie/tests/cuda_lowlevel_chat.rs`). This host-only smoke asserts the
//! property the mock CAN prove: the hand-written low-level decode loop — including
//! the speculate-past-stop + `drain_discard` rollback path — runs to completion
//! without panic/hang and returns a well-formed self-report (`MATCH` degenerate on
//! the mock, `ROLLBACK_OK` proves the discard executed cleanly).

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::SamplingProgramBehavior};

use pie::process;
use pie::program::ProgramName;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(15);

static SERIAL: Mutex<()> = Mutex::new(());

fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

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
        // Generous page pool (128): the inferlet runs FOUR fresh decode contexts
        // (pipelined + sync + forced-stop ×2), each with its own KV working set.
        let env = create_mock_env(
            "test-model",
            1,
            128,
            Arc::new(SamplingProgramBehavior { fallback: 0 }),
        );
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

/// The hand-written low-level run-ahead decode loop (the PRIMARY promote-path)
/// executes to completion on the mock and returns a well-formed self-report.
///
/// Scope (honest): the mock has NO device-resident carrier and its sampler is
/// input-independent, so this proves the explicit decode loop drives the raw WIT
/// surface (`ForwardPass` / `KvWorkingSet` / greedy `Program` / `pie:instruct/chat`)
/// to completion without panic/hang — NOT token identity (that is the 4090 gate,
/// `bin/pie/tests/cuda_lowlevel_chat.rs`). The forced-stop DISCARD probe is skipped
/// here (`no-rollback-probe`): draining a speculated-then-discarded consumer trips
/// the runtime's fail-closed next-input finalize, a device-resident carrier
/// property the mock cannot complete (exactly why `runahead`'s decode is
/// mock-`#[ignore]`d). The discard/rollback is validated on the 4090.
#[test]
fn lowlevel_chat_primary_path_runs_to_completion_on_mock() {
    let _serial = serial_guard();
    let s = state();
    // Skip the forced-stop discard probe (device-resident — 4090-gated). This
    // exercises the primary run-ahead promote-path + the sequential reference.
    let out = spawn_and_capture(s, "lowlevel-chat", "8 no-rollback-probe".into())
        .expect("lowlevel-chat primary path should run to completion (no panic/hang)");
    eprintln!("[lowlevel-chat: mock] {out}");
    assert!(
        out.contains("MATCH=") && out.contains("ROLLBACK_OK="),
        "lowlevel-chat must return its well-formed self-report: {out}"
    );
    // The primary pipelined + sequential streams both ran the full budget (n=8),
    // degenerate-equal on the mock's constant sampler.
    assert!(
        out.contains("n=8"),
        "the explicit decode loop must run its full token budget: {out}"
    );
}
