//! **Does batch width follow the fleet?**
//!
//! The one property every other throughput claim in this tree rests on, and
//! the one nothing asserted. A serving runtime earns its throughput by firing
//! many lanes in ONE forward; if the batches stay narrow while the fleet grows,
//! the engine does the same total work in N times as many launches and the
//! deployment plateaus at single-lane speed no matter how many guests arrive.
//!
//! It regressed exactly that way and nothing caught it. Measured on this
//! machine at 16 live processes, before the fix in
//! [`EngineLoop::next_request`](runtime): 428 tok/s, `total batches 1298` for
//! 4096 output tokens, `batch size hist [0, 548, 750, 0, 0, 0, 0, 0]` — every
//! fire two or three lanes wide against an engine whose `max_lanes` is 256.
//! The scheduler was not at fault and the wait-all gate was not at fault:
//! thirteen of the sixteen processes had never reached the frame policy at
//! all, because their BIND controls were starved on the engine lane behind an
//! unbroken launch train, and a gate can only gather the lanes that exist. The
//! same cell after: 1932 tok/s, `total batches 141`, hist `[0,4,3,6,19,109,0,0]`.
//!
//! So this gate asserts the property rather than the mechanism, which is what
//! makes it survive the next redesign of the mechanism:
//!
//!   1. **Every lane ran.** A fleet that lost lanes is not a width measurement.
//!   2. **Some fire gathered much of the fleet** — the widest fire in the
//!      window carried at least [`MIN_PEAK_WIDTH_FRACTION`] of the live lanes.
//!   3. **And typically, not just one.** Mean lanes per fire is at least
//!      [`MIN_MEAN_WIDTH_FRACTION`] of the fleet, so a single wide fire
//!      surrounded by narrow ones cannot satisfy (2) alone.
//!
//! A NOTE ON THE UNIT. `total_requests_processed / total_batches` counts lanes
//! per SEALED FRAME, and a frame is `model.frame-size()` waves deep (k = 2 by
//! default), so a fleet-wide frame reads as up to k x FLEET here while each
//! forward carries FLEET. The fractions below are therefore deliberately
//! stated against FLEET rather than against k x FLEET: the gate's business is
//! that width tracks the fleet, and it should not go red the day k changes.
//!
//! AND A SECOND CEILING, WHICH IS NOT A BUG AND IS WHY THIS GATE PINS ITS
//! FLEET. Batch width is bounded by process ADMISSION, admission by the pools'
//! seat count, and `Budgets::slots` is the KV page pool divided by one
//! FULL-CONTEXT block per seat. Capping the pool therefore caps every batch:
//! measured at 256 concurrent inferlets, `--total-pages 16384` seats 64,
//! admits 32 and fires 32-lane forwards; 65536 seats 256, admits 128, fires
//! 128; 131072 seats 512, admits 256, and every steady-state forward carries
//! the whole 256-way fleet. Nothing in the scheduler is involved. `bootstrap`
//! says so at boot now — the warning names the pool and the per-seat block —
//! but a gate whose FLEET outran the seat count would still fail for that
//! reason while reading as a scheduler regression.
//!
//! Neither fraction is a tuned number, because the two regimes are not close.
//! The same A/B on THIS test, the only difference being
//! `LaneTurn::LAUNCH_RUN_BEFORE_CONTROL` set to `u32::MAX` (which restores the
//! unbounded launch-first preference and nothing else):
//!
//! ```text
//! starved  fleet=64   120 tok/s  batches=7940  mean_width= 1.03  peak= 2  hist=[7688,252,0,...]
//! served   fleet=64  3076 tok/s  batches= 265  mean_width=30.91  peak=43  hist=[7,0,0,2,131,125,0,0]
//! ```
//!
//! The thresholds sit in the empty middle of a thirty-fold gap: the peak floor
//! (0.5x FLEET = 32) is 16x what the collapse reached and 0.74x what a healthy
//! run reaches; the mean floor (0.25x FLEET = 16) is 15x the collapsed mean and
//! 0.52x the healthy one. Ramp-in and drain are inside the mean, which is why
//! it is a fraction and not a floor at FLEET.
//!
//! `#[ignore]` (needs a CUDA device and the Qwen3.5-0.8B snapshot). Run:
//!   PIE_COMPILER_LAUNCHER=env \
//!     cargo test -p pie-gpu-tests --features engine-cuda-13 --release \
//!     --test cuda_batch_width -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;
use pie::sweep::{self, fleet};

/// The program every lane runs — the same guest `cuda_sweep_e2e` drives, for
/// the same reason: it is in the tree, it takes a token budget, and nothing
/// here reads its answer beyond "it produced one".
const GENERATE: &str = "text-completion@0.1.0";

/// Live lanes. Large enough that a collapsed scheduler is unmistakable (both
/// regressions fired at a fixed small width regardless of fleet size, so the
/// gap grows with this number) and small enough that the round is seconds.
///
/// Bounded above by what the boot can SEAT, which is not the same as what the
/// engine can fire: `boot_cuda` leaves `max_total_pages` unset, so the pools
/// come out at the contract's default 256 seats, and a recurrent model spends
/// `frame_dispatch_depth` seats per admitted lane — 128 lanes here. Sixty-four
/// sits clear of that. A gate that asked for more would measure the seat pool
/// rather than the scheduler, which is the trap this file's second note is
/// about.
const FLEET: usize = 64;

/// Tokens per lane. A lane has to outlive the fleet's own bring-up or it
/// contributes only to the ramp, which is the part of the run this gate is
/// least interested in.
const MAX_TOKENS: usize = 128;

/// Widest fire in the window, as a fraction of `FLEET`. Not 1.0: lanes join
/// and drain at slightly different instants and the gate has no reason to
/// demand that all sixty-four were ever ready in the same microsecond.
/// Measured 43/64 healthy against 2/64 collapsed.
const MIN_PEAK_WIDTH_FRACTION: f64 = 0.5;

/// Mean lanes per fire, as a fraction of `FLEET`. Measured 30.91/64 healthy
/// against 1.03/64 collapsed.
const MIN_MEAN_WIDTH_FRACTION: f64 = 0.25;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "the batch-width gate: needs a CUDA device, the Qwen3.5-0.8B snapshot \
            and a wasm guest built on the spot"]
async fn batch_width_follows_the_fleet() -> Result<()> {
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

    let pie = common::boot_cuda().await?;
    let addr = pie.listen_addr.to_string();

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "batch-width").await?;
    setup.authenticate("batch-width", &None).await?;
    setup
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;
    drop(setup);

    let inputs: Vec<String> = (0..FLEET)
        .map(|_| serde_json::json!({ "max_tokens": MAX_TOKENS }).to_string())
        .collect();

    // Discarded, and NOT only for the timing reason `sweep::warmup` exists
    // for: the first fleet a fresh runtime sees pays program registration and
    // the opening cohort's whole bring-up, and those batches are legitimately
    // narrow. Measuring them would make the gate assert on the ramp.
    sweep::warmup(&addr, GENERATE, &inputs)
        .await
        .context("warmup round")?;

    // The window: everything between these two reads is the measured fleet.
    let before = runtime::scheduler::get_stats().await;
    let run = fleet::run(&addr, GENERATE, &inputs).await;
    let after = runtime::scheduler::get_stats().await;

    // (1) Every lane ran.
    anyhow::ensure!(
        run.failed_lanes() == 0,
        "{} of {FLEET} lanes failed: {:?}",
        run.failed_lanes(),
        run.failures,
    );
    anyhow::ensure!(
        run.total_tokens() > 0,
        "the fleet produced no tokens at all"
    );

    let batches = after
        .total_batches
        .saturating_sub(before.total_batches)
        .max(1);
    let lanes_fired = after
        .total_requests_processed
        .saturating_sub(before.total_requests_processed);
    let mean_width = lanes_fired as f64 / batches as f64;
    // A running maximum, so it is read absolutely rather than differenced —
    // the warmup ran the same FLEET, so nothing before this window could have
    // set it higher than this window can.
    let peak_width = after.max_forward_requests_observed;

    eprintln!(
        "[batch-width] fleet={FLEET} tokens={} elapsed={:.3}s throughput={:.1}tok/s \
         batches={batches} lanes_fired={lanes_fired} mean_width={mean_width:.2} \
         peak_width={peak_width} hist={:?}",
        run.total_tokens(),
        run.elapsed.as_secs_f64(),
        run.throughput_tok_s(),
        after
            .batch_size_hist
            .iter()
            .zip(before.batch_size_hist)
            .map(|(a, b)| a.saturating_sub(b))
            .collect::<Vec<_>>(),
    );

    // (2) Some fire gathered much of the fleet.
    let peak_floor = (MIN_PEAK_WIDTH_FRACTION * FLEET as f64).ceil() as u64;
    anyhow::ensure!(
        peak_width >= peak_floor,
        "the widest fire this runtime ever built carried {peak_width} lanes against {FLEET} \
         live ones (floor {peak_floor}): no submission ever gathered the fleet, so the \
         batches are not a batch. Look at what is keeping lanes out of the frame policy's \
         wait-set — a starved bind control is one way and was the last one (see \
         `EngineLoop::next_request`)",
    );

    // (3) And not just one of them.
    let mean_floor = MIN_MEAN_WIDTH_FRACTION * FLEET as f64;
    anyhow::ensure!(
        mean_width >= mean_floor,
        "mean fire width {mean_width:.2} over {batches} fires is below {mean_floor:.2} \
         ({MIN_MEAN_WIDTH_FRACTION} of a {FLEET}-lane fleet): width collapsed even though \
         one fire reached {peak_width}",
    );

    Ok(())
}
