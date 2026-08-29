//! **Does a sweep actually happen?**
//!
//! Everything `sweep` and `scheduler::reconfigure` are made of has unit tests,
//! and none of them answer this. The unit tests check arithmetic — a percentile,
//! a candidate filter, a tokens-per-second division — and `reconfigure`'s own
//! test moves three statics in an empty process with no runtime under them.
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
//!   3. The rounds run but the runtime degrades across them, so later candidates
//!      are penalised for their position rather than judged on their merits.
//!
//! This test is the one that would catch all three, so it asserts on all three
//! rather than on a winner. Which candidate is fastest is NOT asserted: that is
//! a property of the machine, and a test that pins it would fail on the next
//! one.
//!
//! # AND IT IS THE TREE'S ONE COVER FOR CONCURRENT MULTI-LANE SERVING
//!
//! A fourth thing it catches, which is not about sweeping at all and is why
//! the census wave left it RED rather than `#[ignore]`d: eight guests
//! arriving at once co-batch, so this is the only gate in the tree that fires
//! a submission carrying more than one member's lanes through the serving
//! door. It found that such a fire was refused outright —
//!
//! ```text
//! 4 of 8 lanes failed during warmup; the fleet cannot run here at all.
//!   4x text-completion@0.1.0 returned an error: ... direct launch rejected:
//!   invalid submission: slot 0 appears twice in one fire, at lane 1
//! ```
//!
//! — because `Lane::slot`, the sequence's seat in the shell's pools, had no
//! owner: both runtime fire paths stated zero and the caller their comments
//! said would stamp it did not exist. `palo` build log 29 gave the seat to
//! the KV working set (`runtime::store::seat`), which is the runtime's
//! per-sequence identity, and this gate is GREEN on it: four rounds, eight
//! lanes each, `failed 0`. Keep it that way — a regression here is the
//! defect coming back, and the host-side half of the same claim is
//! `runtime::scheduler::batch`'s
//! `two_seated_members_batch_into_a_fire_the_contract_accepts`.
//!
//! `#[ignore]` (needs a CUDA device and the Qwen3.5-0.8B snapshot). Run:
//!   PIE_COMPILER_LAUNCHER=env \
//!     cargo test -p pie-gpu-tests --features engine-cuda-13 --release \
//!     --test cuda_sweep_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;
use pie::sweep::{self, Knobs};

/// The program every lane runs. It was `generate@0.1.0`, a package deleted
/// with the guest workspace's move from `crates/runtime/tests/inferlets` to
/// `tests/inferlets`; `text-completion` replaces it. Nothing here reads the
/// guest's answer — the assertions are all `sweep::Round` — so the repoint is
/// a package name and the shape of the budget it is handed.
const GENERATE: &str = "text-completion@0.1.0";

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
/// staging bound (`k * dispatch < 13`) admits both here. The guest's own
/// window is no longer a candidate axis — it is `dispatch_depth + 1`
/// (`engine::runahead::Runahead::submit_depth`), so moving `dispatch_depth`
/// moves it (alto E).
fn probe_candidates() -> Vec<Knobs> {
    vec![
        Knobs {
            frame_size: 2,
            dispatch_depth: 2,
        },
        Knobs {
            frame_size: 1,
            dispatch_depth: 2,
        },
        Knobs {
            frame_size: 4,
            dispatch_depth: 3,
        },
        // Back to the first one, last. This is the drift check: the same knobs
        // measured at the start and at the end of a sweep have to agree, or
        // every ranking the sweep produces is confounded by round order.
        Knobs {
            frame_size: 2,
            dispatch_depth: 2,
        },
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "the sweep gate: needs a CUDA device, the Qwen3.5-0.8B snapshot and a \
            wasm guest built on the spot"]
async fn a_sweep_measures_many_candidates_against_one_resident_model() -> Result<()> {
    common::init_trace();

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    anyhow::ensure!(
        Command::new("cargo")
            .args(["build", "--target", "wasm32-wasip2", "-p", "text-completion"])
            .current_dir(&ws)
            .status()?
            .success(),
        "text-completion wasm build failed"
    );
    let wasm = ws.join("target/wasm32-wasip2/debug/text_completion.wasm");
    let manifest = ws.join("text-completion/Pie.toml");

    // ONE boot. Everything after this is the claim under test: the expensive
    // thing happens once and the cheap thing repeats.
    let pie = common::boot_cuda().await?;
    let addr = pie.listen_addr.to_string();

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "sweep-e2e").await?;
    setup.authenticate("sweep-e2e", &None).await?;
    setup
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;
    drop(setup);

    // Forty-eight tokens a lane, so a lane lives long enough to co-batch with
    // its neighbours — a lane that finishes before the fleet forms measures the
    // knobs against nothing. It is a `max_tokens` field rather than the bare
    // integer the deleted `generate` guest parsed, because `text-completion`
    // reads a JSON document.
    let inputs: Vec<String> = (0..FLEET)
        .map(|_| serde_json::json!({ "max_tokens": 48 }).to_string())
        .collect();

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

    // (3) The knobs reached the runtime. `frame_size` is the one that is
    //     observable from outside the scheduler, because guests are told it —
    //     so read it back through the same accessor the host function serves.
    anyhow::ensure!(
        runtime::scheduler::configured_frame_size() == 2,
        "the last round set k=2; the runtime reports {}",
        runtime::scheduler::configured_frame_size()
    );

    // (4) No drift across the sweep. Rounds 0 and 3 ran identical knobs; if the
    //     runtime degrades as rounds accumulate, the sweep ranks position rather
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
