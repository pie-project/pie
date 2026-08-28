//! **The engine states the next fire itself** — step 6 of the palo cuda-abi
//! wave (`.wiki/palo/cuda-abi.md` §6d→§6e), gated end to end.
//!
//! Step 5 proved the prebind at the SHELL: alternating two compositions
//! through one folded bucket, a test-side `Shell::expect` before each fire
//! took the ~261 us critical-path rebind and hid it under the previous
//! fire's execution (4.201 → 3.939 ms/fire). Nothing in the engine called
//! `expect`. This gate runs the alternating shape through the SERVING DOOR —
//! boot, websocket, wasm guests, the frame scheduler — with the hint stated
//! by the engine's own driver lane (`Driver::expect_fire`, issued from
//! `engine::scheduler::worker::fire_frame`) and nothing test-side touching
//! the shell at all.
//!
//! # The workload, and why it is this one
//!
//! One carried completion decodes steadily (`text-completion`,
//! [`common::SERVING_PROMPT`]) while a churn lane launches short fresh
//! completions beside it. Every fire stays under the shell's default
//! lattice floor (8 rows — the engine states no bucket lattice, so
//! `driver-cuda`'s `default_lattice` applies), which puts the decode-only
//! waves, the prefill waves and the co-batched mixed waves in ONE bucket
//! with DIFFERENT class signatures — the alternating-inside-one-bucket
//! shape that made every fire of step 5's shell gate rebind, with more than
//! two signatures rotating, which is the shape where two seats cannot hold
//! them all.
//!
//! # What is asserted, and what is only printed
//!
//! **Asserted**: the SOLO phase — the carried prompt decoded with nothing
//! beside it — answers the pinned greedy continuation
//! ([`common::SERVING_GREEDY_16`]) in every mode. That is the engine-level
//! "folded serving says what eager serving says", single-lane, and it is
//! the identity the shell's own gates pin. And the fold's MOTION: the
//! measured mixed window folds its fires; the `PIE_CUDA_PIPELINE=off` arm
//! pays critical-path rebinds (or the workload is not alternating and the
//! gate measures nothing); the pipelined arm turns the pair instead.
//!
//! **Printed, not asserted**: the mixed window's texts and the engine's own
//! ms/fire (`driver_fire_us` when the `profile-fire` probes are compiled
//! in, the always-on `lane_launch_us` beside it). Co-batched token
//! IDENTITY is deliberately not asserted here: what a lane answers when
//! others ride beside it is a property of the composition quantization
//! (D4's bucket claim, and `masked_axis`'s documented mixed-fire red is
//! the standing counterexample), not of this step's hint wiring — the
//! `PIE_FOLD_E2E_MODE=eager|keyed` control arms exist exactly to attribute
//! any drift to the layer it comes from.
//!
//! # Running the A/B
//!
//! One boot per process, and `PIE_CUDA_PIPELINE` is read once at load — so
//! the arms are two invocations, the repo's standing manual-A/B shape:
//!
//! ```text
//! PIE_CUDA_PIPELINE=off cargo test -p pie-gpu-tests \
//!   --features driver-cuda-13,engine/profile-fire --release \
//!   --test cuda_fold_hint_e2e -- --ignored --nocapture   # step-4 fold
//! cargo test -p pie-gpu-tests \
//!   --features driver-cuda-13,engine/profile-fire --release \
//!   --test cuda_fold_hint_e2e -- --ignored --nocapture   # hint wired
//! ```
//!
//! `PIE_FOLD_E2E_DEPTH` states the run-ahead depth (default: the engine's
//! own 2). §6d hoped the depth-2 post point holds fire N+1 sealed while
//! fire N executes; the lane's lookahead makes that observable — `prebinds`
//! stays zero when the successor was still being sealed at fire time, and
//! moves when the queue actually held it.

mod common;

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use client::client::Client;

/// The carried lane's budget: long enough that the churn lane's whole run
/// happens beside a live decode.
const CARRIED_TOKENS: u32 = 96;

/// Fresh completions launched beside the carried decode. Each one is a
/// prefill fire plus a couple of decode fires — the signature churn.
const CHURN_LAUNCHES: usize = 16;

/// The churn prompt, seven tokens — with the carried decode beside it the
/// mixed wave is eight rows, which is exactly the default lattice's floor
/// bucket. The same sentence step 5's shell gate re-seated every odd step.
const CHURN_PROMPT: &str = "Water freezes at a temperature of";

/// Tokens each churn launch decodes past its prefill.
const CHURN_TOKENS: u32 = 2;

/// Which recording mode this process boots — the attribution axis.
/// `folded` is the subject; `keyed` (graphs on, fold off) and `eager`
/// (no recording) are the control arms that say which layer a mixed-window
/// observation belongs to.
#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Folded,
    Keyed,
    Eager,
}

fn build_fixture() -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "text-completion"])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "text-completion wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/text_completion.wasm");
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());
    Ok((wasm, workspace.join("text-completion/Pie.toml")))
}

async fn complete(client: &Client, prompt: &str, max_tokens: u32) -> Result<(String, u64)> {
    let input = serde_json::json!({ "prompt": prompt, "max_tokens": max_tokens });
    let mut proc = client
        .launch_process("text-completion@0.1.0".to_string(), input.to_string(), true)
        .await
        .context("launch text-completion")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    Ok((
        parsed["text"].as_str().unwrap_or_default().to_string(),
        parsed["count"].as_u64().unwrap_or_default(),
    ))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the step-6 gate: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn the_engines_own_hint_reaches_the_fold_without_a_test_side_expect() -> Result<()> {
    common::init_trace();

    let mode = match std::env::var("PIE_FOLD_E2E_MODE").ok().as_deref() {
        Some("eager") => Mode::Eager,
        Some("keyed") => Mode::Keyed,
        _ => Mode::Folded,
    };
    // The fold and the recorder are load-time opt-ins (`PIE_CUDA_FOLD`
    // defaults off; the serving boot's `[driver] graphs` defaults to eager),
    // and this gate is ABOUT them. Set before the boot, on the test's main
    // thread, before any thread the boot spawns can read the environment.
    //
    // SAFETY: nothing else runs yet — the boot below is this process's
    // first, and `Shell::load` reads the variables once.
    unsafe {
        match mode {
            Mode::Folded => {
                std::env::set_var("PIE_CUDA_GRAPHS", "on");
                std::env::set_var("PIE_CUDA_FOLD", "on");
            }
            Mode::Keyed => {
                std::env::set_var("PIE_CUDA_GRAPHS", "on");
                std::env::set_var("PIE_CUDA_FOLD", "off");
            }
            Mode::Eager => std::env::set_var("PIE_CUDA_GRAPHS", "off"),
        }
    }
    let pipeline = !matches!(
        std::env::var("PIE_CUDA_PIPELINE").ok().as_deref(),
        Some("off" | "0" | "false")
    );
    let arm = match (mode, pipeline) {
        (Mode::Eager, _) => "eager",
        (Mode::Keyed, _) => "keyed",
        (Mode::Folded, false) => "pipeline-off",
        (Mode::Folded, true) => "hint-wired",
    };

    let checkpoint = common::resolve_qwen35_snapshot()?;
    let mut toml = common::serving_standalone_toml(&checkpoint);
    let depth: u32 = std::env::var("PIE_FOLD_E2E_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    if depth > 0 {
        toml.push_str(&format!("\n[runtime]\nframe_dispatch_depth = {depth}\n"));
    }
    let (controller, gateway, worker) = pie::derive::derive_standalone(&toml)?;
    let pie = pie::run_standalone(controller, gateway, worker).await?;
    eprintln!(
        "[fold-e2e] arm={arm} depth={} booted, listen_addr={}",
        if depth > 0 { depth.to_string() } else { "default".into() },
        pie.listen_addr
    );

    let (wasm, manifest) = build_fixture()?;
    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program text-completion")?;

    // ── THE SOLO PHASE, ASSERTED. Nothing rides beside either prompt here,
    //    so the pinned constant applies: folded single-lane serving must say
    //    what the eager serving gates say, token for token, or the fold has
    //    no business being on. This phase is also the warm-up — the JIT, the
    //    tuner, the bucket's arming ladder and the first signatures' binding
    //    fires all happen here, so the measured window below holds revisits.
    let (solo_text, solo_count) = complete(&client, common::SERVING_PROMPT, 20).await?;
    anyhow::ensure!(solo_count == 20, "the solo run was cut short at {solo_count}");
    anyhow::ensure!(
        solo_text.starts_with(common::SERVING_GREEDY_16),
        "the solo continuation drifted under {arm}:\n  got {solo_text:?}\n  \
         want a prefix of {:?}",
        common::SERVING_GREEDY_16
    );
    let (churn_solo, _) = complete(&client, CHURN_PROMPT, CHURN_TOKENS).await?;

    let stats_before = engine::scheduler::get_stats().await;
    let fold_before = engine::driver::fold_observed();
    let began = Instant::now();

    // ── THE MEASURED WINDOW: the carried decode and the churn lane,
    //    concurrent through one client edge. The churn launches run
    //    SEQUENTIALLY on their lane so each prefill lands beside a live
    //    decode instead of beside each other.
    let carried = complete(&client, common::SERVING_PROMPT, CARRIED_TOKENS);
    let churn = async {
        let mut answers: Vec<String> = Vec::with_capacity(CHURN_LAUNCHES);
        for _ in 0..CHURN_LAUNCHES {
            let (text, count) = complete(&client, CHURN_PROMPT, CHURN_TOKENS).await?;
            anyhow::ensure!(
                count == u64::from(CHURN_TOKENS),
                "a churn launch was cut short at {count}"
            );
            answers.push(text);
        }
        Ok::<Vec<String>, anyhow::Error>(answers)
    };
    let (carried, churn) = tokio::join!(carried, churn);
    let elapsed = began.elapsed();
    let (carried_text, carried_count) = carried?;
    let churn_answers = churn?;

    let stats_after = engine::scheduler::get_stats().await;
    let fold_after = engine::driver::fold_observed();

    let launches =
        stats_after.fire.quorum.lane_launch_n - stats_before.fire.quorum.lane_launch_n;
    let fire_us = stats_after.fire.execute.driver_fire_us_sum
        - stats_before.fire.execute.driver_fire_us_sum;
    let lane_us =
        stats_after.fire.quorum.lane_launch_us - stats_before.fire.quorum.lane_launch_us;
    let (folds, rebinds, rebind_us, swaps, prebinds, prebind_us, twins) = (
        fold_after.0 - fold_before.0,
        fold_after.1 - fold_before.1,
        fold_after.2 - fold_before.2,
        fold_after.3 - fold_before.3,
        fold_after.4 - fold_before.4,
        fold_after.5 - fold_before.5,
        fold_after.6 - fold_before.6,
    );
    eprintln!(
        "[fold-e2e] arm={arm} {:.1} ms wall  launches={launches}  \
         driver_fire={:.3} ms/fire (0 = profile-fire not compiled)  \
         lane_launch={:.3} ms/post",
        elapsed.as_secs_f64() * 1e3,
        fire_us as f64 / launches.max(1) as f64 / 1e3,
        lane_us as f64 / launches.max(1) as f64 / 1e3,
    );
    eprintln!(
        "[fold-e2e] arm={arm} [fold-stats] folds={folds} rebinds={rebinds} \
         ({rebind_us} us on the critical path) swaps={swaps} \
         prebinds={prebinds} ({prebind_us} us hidden) twins={twins}"
    );
    // The mixed window's texts, printed for the operator's cross-arm diff —
    // see the header for why they are not asserted against the pinned solo.
    eprintln!(
        "[fold-e2e] arm={arm} carried beside churn ({carried_count} tokens): {carried_text:?}"
    );
    eprintln!(
        "[fold-e2e] arm={arm} churn solo {churn_solo:?}; beside the carried decode: {:?}{}",
        churn_answers[0],
        if churn_answers.iter().all(|answer| answer == &churn_answers[0]) {
            " (all launches agree)".to_string()
        } else {
            format!(" (LAUNCHES DISAGREE: {churn_answers:?})")
        }
    );
    anyhow::ensure!(
        u64::from(CARRIED_TOKENS) == carried_count,
        "the carried lane was cut short at {carried_count}"
    );

    // ── THE MOTION, the fold arms only. Folded fires must dominate the
    //    window, or the gate exercised some other path and says nothing.
    if mode == Mode::Folded {
        anyhow::ensure!(
            folds > launches / 2,
            "the fold served {folds} of {launches} launches; the gate is not \
             measuring the folded path"
        );
        if pipeline {
            // The pair must turn: revisited compositions land as swaps or
            // prebinds, not as critical-path rebinds. The split between the
            // two is the depth question §6e reports — a prebind needs the
            // successor QUEUED at fire time — so neither is asserted alone.
            anyhow::ensure!(
                swaps + prebinds > 0,
                "the pipelined arm never turned the ping-pong pair: \
                 rebinds={rebinds} swaps={swaps} prebinds={prebinds}"
            );
        } else {
            // Step 4's fold exactly: one exec per bucket, no twin, so the
            // signature churn must pay critical-path rebinds — the cost the
            // other arm exists to remove.
            anyhow::ensure!(
                rebinds > 0,
                "the unpipelined arm never rebound, so the workload's \
                 compositions are not alternating inside a bucket"
            );
            anyhow::ensure!(
                prebinds == 0 && twins == 0,
                "pipeline machinery moved with the pipeline off: \
                 prebinds={prebinds} twins={twins}"
            );
        }
    }

    pie.shutdown().await;
    Ok(())
}
