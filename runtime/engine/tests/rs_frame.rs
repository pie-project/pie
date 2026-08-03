//! Recurrent-state fires at frame size k > 1, end to end on the mock driver.
//!
//! Until the RS publication seam moved from finalize to prepare, a pass
//! binding recurrent state could not share a frame: `fire` serialized it
//! behind every predecessor fire's SETTLEMENT, but a frame's fires cannot
//! settle until all of them are submitted — so the frame sat one fire short
//! of sealing forever, and `validate_frame` rejected the shape rather than
//! hanging on it. That collapsed k to 1 for every linear/SSM model.
//!
//! RS mappings now publish at prepare, in slot order, under the store lock,
//! so slot i+1 classifies against slot i's decision without waiting for it.
//! This binary pins `PIE_FRAME_SIZE=3` and runs a GDN/hybrid model (the mock
//! reports a recurrent-state pool, so `model.pass-kind()` is not attention) two ways:
//!
//!  * `generate-gdn` — one live RS slot per frame, the shape that used to be
//!    the ONLY legal one. Proves run-ahead across frames still works.
//!  * `generate-gdn-frame` — every slot of the 3-wide frame binds the same
//!    RS working set. This is the shape the engine used to reject.
//!
//! It lives in its own test binary because `PIE_FRAME_SIZE` is read once into
//! a `OnceLock` (`scheduler::configured_frame_size`), so k must be pinned
//! before anything in the process touches the scheduler.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

mod common;
use common::{MockEnv, create_mock_env, inferlets, mock_device::EchoBehavior};

use pie_engine::inferlet::process;
use pie_engine::inferlet::program::ProgramName;

/// The pinned deployment constant under test.
const FRAME_SIZE: usize = 3;

/// Recurrent-state pool the mock model reports. Small on purpose: a slot leak
/// or a double-allocation per fire exhausts it and the run fails loudly
/// instead of passing on a pool that hides the bug.
const RS_SLOTS: usize = 4;

const PROCESS_TIMEOUT: Duration = Duration::from_secs(30);

struct TestState {
    #[allow(dead_code)]
    env: MockEnv,
    rt: tokio::runtime::Runtime,
}

static STATE: OnceLock<TestState> = OnceLock::new();

/// Inferlets this binary runs. Built and installed exactly once, inside the
/// `OnceLock`: `add_and_install` shells out to `cargo build` and mutates the
/// shared program registry, so two tests installing the same program in
/// parallel race ("Component not found").
const PROGRAMS: [&str; 3] = ["generate-gdn", "generate-gdn-frame", "gdn-foldcommit"];

fn state() -> &'static TestState {
    STATE.get_or_init(|| {
        // SAFETY: set before the runtime starts and before any scheduler
        // entry point runs, and this binary is single-threaded until the
        // tokio runtime below is created.
        unsafe {
            std::env::set_var("PIE_FRAME_SIZE", FRAME_SIZE.to_string());
        }
        assert_eq!(
            pie_engine::scheduler::configured_frame_size(),
            FRAME_SIZE,
            "this binary must run at the pinned frame size"
        );

        let rt = tokio::runtime::Runtime::new().unwrap();
        // EchoBehavior's token must be in-vocab for the 20-token fixture
        // tokenizer and neither `<eos>` (0) nor `<bos>` (1).
        let env = create_mock_env("test-model-gdn", 1, 64, Arc::new(EchoBehavior(7)))
            .with_recurrent_state(RS_SLOTS, 4096);
        let config = env.config();
        rt.block_on(async {
            pie_engine::bootstrap::bootstrap(config).await.unwrap();
            for name in PROGRAMS {
                inferlets::add_and_install(name).await;
            }
        });
        TestState { env, rt }
    })
}

fn run(name: &str, input: &str) -> Result<String, String> {
    assert!(
        PROGRAMS.contains(&name),
        "{name} must be installed in state()"
    );
    let s = state();
    let rx = s.rt.block_on(async {
        let (tx, rx) = tokio::sync::oneshot::channel();
        process::spawn(
            "test-user".into(),
            ProgramName::parse(&format!("{name}@0.1.0")).unwrap(),
            input.into(),
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

fn generated_count(output: &str) -> usize {
    // "generated N tokens: [..]"
    output
        .split_whitespace()
        .nth(1)
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("unexpected inferlet output: {output}"))
}

/// The model really is linear here — otherwise every assertion below would
/// pass vacuously against the pure-attention path.
#[test]
fn mock_model_reports_recurrent_state() {
    let _ = state();
    let caps = pie_model::model().rs_caps();
    assert!(
        caps.state_size > 0,
        "the mock must report a recurrent state for this suite to mean anything"
    );
}

/// One live RS slot per frame at k = 3: the legacy shape, still correct.
#[test]
fn single_slot_recurrent_frames_run_at_k3() {
    let out = run("generate-gdn", "6").unwrap_or_else(|e| panic!("generate-gdn failed: {e}"));
    assert_eq!(generated_count(&out), 6, "{out}");
}

/// THE regression test: every slot of a 3-wide frame binds recurrent state.
/// Before the publication seam moved, `validate_frame` rejected this with
/// "binds recurrent state ... cannot share a frame".
#[test]
fn full_width_recurrent_frames_run_at_k3() {
    let out = run("generate-gdn-frame", "9")
        .unwrap_or_else(|e| panic!("full-width RS frame failed at k={FRAME_SIZE}: {e}"));
    assert_eq!(generated_count(&out), 9, "{out}");
}

/// A whole run must not grow the RS pool's footprint: the folded slot is
/// written in place, so a 3-wave frame allocates nothing beyond the first.
/// Re-running with a budget that far exceeds the pool proves it.
#[test]
fn full_width_recurrent_frames_do_not_leak_slots() {
    let budget = RS_SLOTS * 4;
    let out = run("generate-gdn-frame", &budget.to_string())
        .unwrap_or_else(|e| panic!("long full-width RS run failed: {e}"));
    assert_eq!(
        generated_count(&out),
        budget,
        "a {budget}-token run must fit a {RS_SLOTS}-slot RS pool: {out}"
    );
}

/// FOLD-COMMIT end to end: a prefill folds, a chunk is buffered with the
/// fold suppressed, and only the accepted prefix is folded back from the
/// buffer. This is the shape `model.pass-kind()` exists to select, and it
/// exercises the three launch fields the engine never used to populate
/// (`rs_fold_lens`, `rs_buffer_slot_ids`, `rs_buffer_slot_indptr`) —
/// the dummy driver validates their CSR well-formedness on every launch.
#[test]
fn fold_commit_buffers_then_folds_the_accepted_prefix() {
    let out = run("gdn-foldcommit", "2").unwrap_or_else(|e| panic!("fold-commit failed: {e}"));
    assert!(
        out.contains("buffered=4") && out.contains("committed=2") && out.contains("abandoned=2"),
        "{out}"
    );
}

/// Accepting nothing must still be legal: the buffered chunk is abandoned
/// whole and the folded state never moved.
#[test]
fn fold_commit_can_reject_the_whole_speculative_chunk() {
    let out = run("gdn-foldcommit", "0").unwrap_or_else(|e| panic!("full reject failed: {e}"));
    assert!(
        out.contains("committed=0") && out.contains("abandoned=4"),
        "{out}"
    );
}
