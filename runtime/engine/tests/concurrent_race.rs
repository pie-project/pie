//! **Host-side concurrent-decode RACE probe** (bravo, support for alpha's arena
//! bug). The `cuda_concurrent` GPU repro shows concurrent multi-pipeline decode
//! produces garbage + `arena: unknown object` on the real driver, but the plain
//! synchronous mock (`e2e::concurrent_decode_fleet`) is CLEAN — the mock returns
//! instantly, so concurrent forwards never actually overlap in flight and the
//! race never opens. This harness wraps the mock in `DelayedBehavior` (a real
//! `thread::sleep` per fire) so multiple forwards ARE outstanding at once,
//! reproducing the async-latency overlap that the GPU has for free — without a
//! GPU. If the arena/finalization race is host-reproducible, it shows here and
//! alpha can iterate the fix on the mock.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::mock_device::{DelayedBehavior, EchoBehavior};
use common::{MockEnv, create_mock_env, inferlets};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(20);
/// Per-fire simulated device latency — long enough that the whole fleet is in
/// flight together (the overlap the real GPU has), short enough to stay fast.
const FIRE_LATENCY: Duration = Duration::from_millis(15);
const FLEET: usize = 8;

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
        // A constant-token mock WITH per-fire latency: forwards stay outstanding
        // long enough to overlap, so concurrent finalization actually races.
        let behavior = DelayedBehavior {
            inner: EchoBehavior(42),
            latency: FIRE_LATENCY,
        };
        let env = create_mock_env("test-model", 1, 16, Arc::new(behavior));
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

/// N concurrent `generate` (raw-WIT) decodes under simulated device latency. Every pipeline
/// must return the identical `[42,42,42,42,42]`; divergence / arena error = the
/// race reproduced host-side.
#[test]
fn concurrent_decode_under_latency_is_deterministic() {
    let s = state();
    let results: Vec<Result<String, String>> = s.rt.block_on(async {
        inferlets::add_and_install("generate").await;
        let rxs: Vec<_> = (0..FLEET)
            .map(|i| {
                let (tx, rx) = tokio::sync::oneshot::channel();
                process::spawn(
                    "race-user".into(),
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
            other => eprintln!("[race] pipeline {i} diverged: {other:?}"),
        }
    }
    eprintln!(
        "[race] {ok}/{FLEET} pipelines deterministic under {FIRE_LATENCY:?} per-fire latency"
    );
    assert_eq!(
        ok, FLEET,
        "concurrent decode under latency must be deterministic — {ok}/{FLEET} correct \
         (divergence ⇒ the GPU-observed arena/finalization race reproduced host-side)"
    );
}
