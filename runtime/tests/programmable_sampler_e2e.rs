//! End-to-end tests for the Phase-2 programmable-sampler capability inferlets
//! (WS7): real WASM inferlets driving Sampling-IR programs through the full
//! stack on the mock device's eval-mock executor (`SamplingProgramBehavior`).
//!
//! These run the **sequential late-bind** loop (WS1a): each fire binds a
//! per-step submit-bound host input (grammar additive mask / mirostat μ), the
//! program is `eval`'d over deterministic synthetic logits, and the sampled
//! token flows back to the inferlet. The grammar inferlet self-asserts its
//! constraint invariants, so a successful return string proves the IR path
//! enforced the grammar end-to-end.
//!
//! The real-driver (4090) variants of these are hotel's lane; this is the
//! fast in-CI plumbing gate.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

mod common;
use common::{create_mock_env, inferlets, mock_device::SamplingProgramBehavior, MockEnv};

use pie::process;
use pie::program::ProgramName;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(10);

/// Serializes the capability inferlets. The eval-mock executor applies a single
/// program (`sampling_program_at(0)`) to a whole fire batch (tier-1 scope), so
/// two inferlets whose fires get scheduled into the same batch would cross-
/// apply masks. Running them one at a time keeps each batch single-program.
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
        // The eval-mock executor: runs an attached sampling-program via `eval`
        // over deterministic synthetic logits, returning one token per request.
        let env = create_mock_env(
            "test-model",
            1,
            16,
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

/// Spawn a process and capture its `Result<String, String>` return value.
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
            None,
            None,
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

/// WS3 — constrained / grammar decoding e2e. The grammar inferlet enforces a
/// no-repeat alphabet DFA purely through the IR (`argmax(logits + mask)`); it
/// returns `Err` if any sampled token violates the grammar, so a successful
/// return proves the submit-bound mask path constrained the output end-to-end.
#[test]
fn grammar_inferlet_constrains_output() {
    let _serial = serial_guard();
    let s = state();
    let out = spawn_and_capture(s, "grammar", "{}".into())
        .expect("grammar inferlet should run to completion and satisfy its constraints");
    eprintln!("[grammar e2e] {out}");
    assert!(
        out.contains("\"sampler\":\"grammar\""),
        "unexpected grammar output: {out}"
    );
    assert!(
        out.contains("\"conformant\":true"),
        "grammar output not marked conformant: {out}"
    );
    assert!(
        out.contains("\"count\":12"),
        "expected 12 generated tokens: {out}"
    );
}

/// WS2 — mirostat v2 e2e. The mirostat inferlet runs the sequential μ-update
/// loop; with the multi-output mock marshaling the Scalar S channel, the
/// μ-update actually runs (`s_flowed: true`). The μ-convergence assertion
/// (tail mean surprise → τ) on real logits lives in the real-driver e2e
/// (hotel); on the synthetic-logits mock we assert the loop ran and S flowed.
#[test]
fn mirostat_inferlet_runs_to_completion() {
    let _serial = serial_guard();
    let s = state();
    let out = spawn_and_capture(s, "mirostat", "{}".into())
        .expect("mirostat inferlet should run to completion");
    eprintln!("[mirostat e2e] {out}");
    assert!(
        out.contains("\"sampler\":\"mirostat\""),
        "unexpected mirostat output: {out}"
    );
    assert!(
        out.contains("\"count\":16"),
        "expected 16 generated tokens: {out}"
    );
    // With the multi-output mock marshaling the Scalar S channel, the μ-update
    // runs — S flowed end to end (`s_flowed: true`), not skipped.
    assert!(
        out.contains("\"s_flowed\":true"),
        "mirostat S (surprise) did not flow through — μ-update was skipped: {out}"
    );
}
