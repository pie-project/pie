//! **A rejected frame fails only the request that cannot fit — never its
//! co-batched neighbors.**
//!
//! The live defect this pins down (tts-bench emergency #2017): one request
//! whose KV demand exceeded the driver's budget ceiling turned into
//! `frame commit impossible`, and every request co-batched into the same
//! frame died with `decode_ids take: channel is poisoned` — a single
//! oversized request killed unrelated in-flight decodes.
//!
//! The dummy driver's `prepare_impossible_above_kv_pages` knob is the CPU
//! stand-in for the CUDA driver's elastic-budget ceiling: any frame whose
//! union KV demand exceeds it is rejected `Impossible`, exactly the folded
//! v14 admission shape. Two concurrent `generate` pipelines run under it:
//! one stays within the ceiling, one grows past it. The vLLM-parity
//! semantics under test: pool pressure resolves as a typed PER-REQUEST
//! failure; the neighbor completes.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::mock_device::Behavior;
use common::{MockEnv, create_mock_env_with_admission_ceiling, inferlets};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

/// Plenty of physical pages: the ENGINE pool must never be the limiter here —
/// pressure comes from the driver's admission ceiling alone, as it did live
/// (the engine admitted; the driver's arena commit refused).
const NUM_PAGES: usize = 48;
/// The driver rejects any frame whose union KV demand exceeds 2 pages
/// (16-token pages: 32 tokens).
const IMPOSSIBLE_ABOVE_KV_PAGES: u32 = 2;
/// Stays under the ceiling: prompt (a few tokens) + 8 generated sits in the
/// first page.
const SMALL_BUDGET: usize = 8;
/// Crosses the ceiling mid-decode: prompt + 90 generated > 32 tokens.
const BIG_BUDGET: usize = 90;
/// Keeps fires from both pipelines outstanding together so they co-batch
/// into shared frames (the shape the live poisoning needed).
const FIRE_DELAY_MS: u64 = 10;
const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);

struct NoProbe;
impl Behavior for NoProbe {}

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
        let env = create_mock_env_with_admission_ceiling(
            "test-model",
            1,
            NUM_PAGES,
            Arc::new(NoProbe),
            IMPOSSIBLE_ABOVE_KV_PAGES,
        )
        .with_callback_delay_ms(FIRE_DELAY_MS);
        let config = env.config();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
        });
        TestState { env, rt }
    })
}

fn program_name(name: &str) -> ProgramName {
    ProgramName::parse(&format!("{name}@0.1.0")).unwrap()
}

/// One oversized decode hits the driver ceiling and fails TYPED; the
/// co-batched small decode is untouched and completes.
#[test]
fn oversized_request_fails_alone_neighbor_completes() {
    let s = state();
    let outcomes = s.rt.block_on(async {
        inferlets::add_and_install("generate").await;
        let spawn = |budget: usize| {
            let (tx, rx) = tokio::sync::oneshot::channel();
            process::spawn(
                "containment-user".into(),
                program_name("generate"),
                budget.to_string(),
                None,
                false,
                Some(tx),
            )
            .expect("spawn generate");
            rx
        };
        // Big first so its decode is live while small runs — the two
        // pipelines' fires co-batch into shared frames under the delay.
        let big = spawn(BIG_BUDGET);
        let small = spawn(SMALL_BUDGET);
        let big = tokio::time::timeout(PROCESS_TIMEOUT, big).await;
        let small = tokio::time::timeout(PROCESS_TIMEOUT, small).await;
        (big, small)
    });

    let (big, small) = outcomes;
    let big = big
        .expect("big process timed out")
        .expect("big result channel dropped");
    let small = small
        .expect("small process timed out")
        .expect("small result channel dropped");

    eprintln!("[containment] big: {big:?}");
    eprintln!("[containment] small: {small:?}");

    // The oversized pipeline fails, and fails TYPED: its own launch was
    // rejected by the driver, not swept up in someone else's frame.
    let big_error = big.expect_err("the oversized decode must fail at the ceiling");
    assert!(
        big_error.contains("direct launch rejected"),
        "oversized decode failed with an untyped error: {big_error}"
    );

    // CONTAINMENT: the neighbor that fits must complete — before the fix it
    // died with `channel is poisoned` whenever it shared a frame with the
    // oversized request's rejected fire.
    let small_text =
        small.unwrap_or_else(|error| panic!("a fitting neighbor died with: {error}"));
    assert!(
        small_text.contains(&format!("generated {SMALL_BUDGET} tokens")),
        "unexpected neighbor completion: {small_text}"
    );
}
