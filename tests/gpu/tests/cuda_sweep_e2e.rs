//! **Does a sweep actually happen?**
//!
//! Everything `sweep` and `scheduler::reconfigure` are made of has unit tests,
//! and none of them answer this. The unit tests check arithmetic — a percentile,
//! a candidate filter, a tokens-per-second division — and `reconfigure`'s own
//! test moves three statics in an empty process with no engine under them.
//!
//! What has never been observed is the thing the whole plan rests on: a model
//! that stays resident while the knobs change beneath it, round after round,
//! with load arriving in between. Three ways that could fail and nothing so far
//! would notice:
//!
//!   1. `reconfigure` refuses, because lanes do not actually retire when the
//!      fleet future resolves — the quiesce gate would then reject every round
//!      after the first, and a sweep would measure one candidate N times.
//!   2. It succeeds and does nothing, because something downstream cached the
//!      old value at boot the way guests cache `frame-size`.
//!   3. The rounds run but the engine degrades across them, so later candidates
//!      are penalised for their position rather than judged on their merits.
//!
//! This test is the one that would catch all three, so it asserts on all three
//! rather than on a winner. Which candidate is fastest is NOT asserted: that is
//! a property of the machine, and a test that pins it would fail on the next
//! one.
//!
//! `#[ignore]` (needs cuda + a resident qwen3). Run:
//!   PIE_COMPILER_LAUNCHER=env \
//!     cargo test -p pie-gpu-tests --features driver-cuda \
//!     --test cuda_sweep_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_bin::sweep::{self, Knobs};
use pie_client::client::Client;

const GENERATE: &str = "generate@0.1.0";

/// Lanes per round. Small enough that a round is seconds, large enough that
/// lanes co-batch — a single lane never exercises the frame knobs at all,
/// because there is nothing to overlap it with.
const FLEET: usize = 8;

/// Fleets per candidate. Three is the smallest count that gives a median an
/// outlier cannot move and a spread that means anything.
const REPEATS: usize = 3;

/// The candidates this test drives. Deliberately few and deliberately spread:
/// the point is that the knobs MOVE and keep working, not that the space is
/// covered. `k=1` and `k=4` are the extremes of the guest contract, and the
/// staging bound (`k * dispatch < 13`) admits both here.
fn probe_candidates() -> Vec<Knobs> {
    vec![
        Knobs { frame_size: 2, submit_depth: 3, dispatch_depth: 2 },
        Knobs { frame_size: 1, submit_depth: 3, dispatch_depth: 2 },
        Knobs { frame_size: 4, submit_depth: 5, dispatch_depth: 3 },
        // Back to the first one, last. This is the drift check: the same knobs
        // measured at the start and at the end of a sweep have to agree, or
        // every ranking the sweep produces is confounded by round order.
        Knobs { frame_size: 2, submit_depth: 3, dispatch_depth: 2 },
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore]
async fn a_sweep_measures_many_candidates_against_one_resident_model() -> Result<()> {
    common::init_trace();

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    anyhow::ensure!(
        Command::new("cargo")
            .args(["build", "--target", "wasm32-wasip2", "-p", "generate"])
            .current_dir(&ws)
            .status()?
            .success(),
        "generate wasm build failed"
    );
    let wasm = ws.join("target/wasm32-wasip2/debug/generate.wasm");
    let manifest = ws.join("generate/Pie.toml");

    // ONE boot. Everything after this is the claim under test: the expensive
    // thing happens once and the cheap thing repeats.
    let pie = common::boot_4090().await?;
    let addr = pie.listen_addr.to_string();

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "sweep-e2e").await?;
    setup.authenticate("sweep-e2e", &None).await?;
    setup
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;
    drop(setup);

    // A bare integer budget makes the generate inferlet decode that many
    // tokens, so a lane lives long enough to co-batch with its neighbours.
    let inputs: Vec<String> = (0..FLEET).map(|_| "48".to_string()).collect();

    // Discarded. Without it the first round measured 844 tok/s and the same
    // configuration measured 1228 at the end of the sweep — see `sweep::warmup`.
    sweep::warmup(&addr, GENERATE, &inputs)
        .await
        .context("warmup round")?;

    let mut rounds = Vec::new();
    for (index, knobs) in probe_candidates().into_iter().enumerate() {
        let round = sweep::measure(&addr, GENERATE, &inputs, knobs, REPEATS)
            .await
            .with_context(|| format!("round {index} ({knobs})"))?;
        eprintln!(
            "[sweep-e2e] round {index}: {knobs} -> {:.1} tok/s ±{:.1}%  p95 {:.1} ms  failed {}",
            round.throughput_tok_s,
            round.throughput_rel_sigma * 100.0,
            round.lane_p95_us as f64 / 1_000.0,
            round.failed_lanes,
        );
        rounds.push(round);
    }

    // (1) Every round was allowed to apply its knobs. A refusal surfaces as the
    //     `?` above, so reaching here already proves the quiesce gate opened
    //     four times — i.e. lanes really do retire when the fleet resolves.
    anyhow::ensure!(rounds.len() == 4, "expected four rounds");

    // (2) Every round is a measurement rather than a failure with a duration.
    for (index, round) in rounds.iter().enumerate() {
        anyhow::ensure!(
            round.is_measurement(),
            "round {index} ({}) lost {} of {FLEET} lanes",
            round.knobs,
            round.failed_lanes
        );
        anyhow::ensure!(
            round.throughput_tok_s > 0.0,
            "round {index} ({}) produced no tokens",
            round.knobs
        );
    }

    // (3) The knobs reached the engine. `frame_size` is the one that is
    //     observable from outside the scheduler, because guests are told it —
    //     so read it back through the same accessor the host function serves.
    anyhow::ensure!(
        pie_engine::scheduler::configured_frame_size() == 2,
        "the last round set k=2; the engine reports {}",
        pie_engine::scheduler::configured_frame_size()
    );

    // (4) No drift across the sweep. Rounds 0 and 3 ran identical knobs; if the
    //     engine degrades as rounds accumulate, the sweep ranks position rather
    //     than configuration and every conclusion from it is worthless.
    //
    //     The bound is loose on purpose: this catches a systematic slide, not a
    //     small one. Observed on an L40S at three repeats per candidate: 3.4%,
    //     against 45% before `sweep::warmup` existed.
    //
    //     Note what the gap between this 3.4% and the rounds' own reported
    //     spread (1.2-2.7%) means. Repeats run back to back, so they share
    //     whatever state the machine is in and measure only the within-burst
    //     variation; the between-round variation is larger. `Round::beats` is
    //     therefore somewhat over-confident, and interleaving candidates rather
    //     than batching their repeats is the fix. Not built.
    let first = rounds[0].throughput_tok_s;
    let last = rounds[3].throughput_tok_s;
    let drift = (last - first).abs() / first.max(f64::EPSILON);
    eprintln!(
        "[sweep-e2e] drift check: {first:.1} -> {last:.1} tok/s ({:+.1}%)",
        drift * 100.0 * (last - first).signum()
    );
    anyhow::ensure!(
        drift < 0.25,
        "identical knobs measured {first:.1} tok/s in round 0 and {last:.1} in round 3 \
         ({:.0}% apart): the sweep is ranking round order, not configuration",
        drift * 100.0
    );

    Ok(())
}
