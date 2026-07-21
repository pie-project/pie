//! End-to-end tool-use tests.
//!
//! Runs a real WASM inferlet through the full stack — SDK -> generated WIT
//! bindings -> host `pie::instruct::tool_use` -> the model's `Instruct` impl —
//! against a model registered as `gemma4`. Runtime unit tests can only reach
//! the last of those hops, so this is the one place the capability contract
//! and the fused system turn are proven across the boundaries production
//! actually crosses.
//!
//! Lives in its own test binary because `bootstrap` is process-global and this
//! needs a model registered under a specific architecture and tokenizer.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::env::create_mock_env_for_arch;
use common::mock_device::EchoBehavior;
use common::{MockEnv, inferlets};

use pie::process;
use pie::program::ProgramName;
use tokio::sync::oneshot;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);

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
        let env = create_mock_env_for_arch(
            "gemma4-test-model",
            1,
            16,
            Arc::new(EchoBehavior(42)),
            "gemma4",
            Some("gemma4_test_tokenizer.json"),
        );
        let config = env.config();
        rt.block_on(async {
            pie::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

/// Run an inferlet to completion and return whatever it returned.
fn run(name: &str) -> String {
    let s = state();
    let (tx, rx) = oneshot::channel();
    s.rt.block_on(async {
        inferlets::add_and_install(name).await;
        process::spawn(
            "test-user".into(),
            ProgramName::parse(&format!("{name}@0.1.0")).unwrap(),
            String::new(),
            None,
            false,
            Some(tx),
            None,
            None,
        )
        .expect("spawn");

        tokio::time::timeout(PROCESS_TIMEOUT, rx)
            .await
            .expect("inferlet completed within timeout")
            .expect("result channel delivered")
            .expect("inferlet returned Ok")
    })
}

#[test]
fn gemma4_tool_use_crosses_the_sdk_wit_and_host_boundaries() {
    let output = run("tool-use");
    let field = |key: &str| {
        output
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("missing {key} in:\n{output}"))
    };

    // Standalone equip is refused rather than returning zero declarations,
    // which is what stops production from constraining an undeclared toolset.
    let standalone = field("standalone");
    assert!(
        standalone.starts_with("err:"),
        "standalone equip must fail loudly, got {standalone:?}"
    );
    assert!(
        standalone.contains("system-equip"),
        "the error must name the supported call, got {standalone:?}"
    );

    // The fused call declares the tool inside the single system turn.
    let fused = field("fused");
    assert!(
        fused.starts_with("<bos><|turn>system\\nYou are helpful.<|tool>declaration:get_weather{"),
        "fused prompt should open one system turn carrying the declaration, got {fused:?}"
    );
    assert!(fused.ends_with("<tool|><turn|>\\n"), "got {fused:?}");
    assert_eq!(fused.matches("<|turn>system").count(), 1);

    // The ergonomic SDK path agrees with the raw host call token for token.
    assert_eq!(field("ctx_matches"), "true");

    // The ordinary equip -> ctx.tool_decoder path reports a declared call and
    // refuses an undeclared one, with no schemas threaded by hand — proven
    // across the real SDK -> WIT -> host seam, with no grammar matcher in play.
    assert_eq!(field("declared_call"), r#"get_weather|{"city":"Paris"}"#);
    assert_eq!(
        field("undeclared_rejected"),
        "true",
        "ctx.tool_decoder must refuse an undeclared call name"
    );

    // A tool result is framed for the next turn without opening a user turn.
    assert_eq!(
        field("answer"),
        "<|tool_response>response:get_weather{value:<|\"|>18C and clear<|\"|>}<tool_response|>"
    );
}

/// The whole loop: declaration -> call -> observation -> continuation.
///
/// Single-turn coverage cannot catch the failure this defends against. The
/// template's generation prompt is state-dependent — it opens a model turn only
/// when no tool response is outstanding, and emits nothing at all when resuming
/// after one — so a cue that is correct for the first turn can silently be
/// wrong for every turn after it. Asserting the assembled prompt is what pins
/// that the turn opened before the call is still the turn the continuation
/// generates into.
#[test]
fn gemma4_tool_use_multi_turn() {
    let output = run("tool-use");
    let field = |key: &str| {
        output
            .lines()
            .find_map(|line| line.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("missing {key} in:\n{output}"))
    };

    let looped = field("loop");

    // Exactly one model turn across the whole exchange. A second `<|turn>model`
    // would mean the continuation reopened a turn that was never closed — the
    // prompt state the template never produces, and the bug this pins.
    assert_eq!(
        looped.matches("<|turn>model").count(),
        1,
        "continuation must not reopen the model turn, got {looped:?}"
    );
    // The model turn is never closed before the observation: call and response
    // both sit inside it.
    assert!(
        looped.contains(
            "<|turn>model\\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>\
             <|tool_response>response:get_weather{value:<|\"|>18C and clear<|\"|>}<tool_response|>"
        ),
        "call and observation must sit inside one open model turn, got {looped:?}"
    );
    // And the declaration that started it is still the one in the system turn.
    assert!(
        looped.starts_with("<bos><|turn>system\\nYou are helpful.<|tool>declaration:get_weather{")
    );
    assert_eq!(looped.matches("<|turn>system").count(), 1);
}
