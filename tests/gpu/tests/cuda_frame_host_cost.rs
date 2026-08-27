//! **palo D0's instrument**: three decode fixtures, one boot, one process,
//! with the scheduler's own counters read across each launch.
//!
//! Build log 19 attributed a ten-millisecond-a-token gap to "submit_frame vs
//! submit host cost" from numbers taken in two different processes at two
//! different frame sizes. Two processes cannot separate a host-path cost from
//! a thermal state, and two frame sizes cannot separate the frame path from
//! the fixture that drives it — and neither can separate a per-TOKEN rate
//! from a per-LAUNCH constant divided by a single token budget, which is what
//! that number turned out to be. This file removes all three confounds: ONE
//! boot, ONE frame size, the fixtures back to back with the arm order
//! reversed on the second pass, and TWO token budgets so the SLOPE between
//! them is the steady-state cost with every per-launch constant differenced
//! away.
//!
//! What the three fixtures differ by, stated so the numbers can be read:
//!
//! * `text-completion` carries its token through the HOST — `submit` then
//!   `take_host().await`, one fire in flight, `GeometryClass::Host`.
//! * `token-healing --heal=false` carries it on the DEVICE — `ptir::run_ahead`
//!   keeps a window of frames outstanding and the shell reads the token off
//!   the ring (`GeometryClass::DecodeEnvelope`). It also asks for the whole
//!   vocabulary at startup, which is where its per-launch constant lives.
//! * `naive-baseline` is device-carried and `run_ahead`-driven in exactly the
//!   same way and never asks for the vocabulary — so it separates the frame
//!   path from that constant — and it SAMPLES, which is what put the sampler
//!   epilogue's host fold on the clock in the first place.
//!
//! The first two answer `common::SERVING_GREEDY_16`, so the tokens are a
//! control: a change that moved them is not a perf change. The third samples,
//! so only its clock is read here.
//!
//! `PIE_FRAME_SIZE` selects k (default 2). At k = 1 `submit_frame`
//! short-circuits before `validate_frame` and both fixtures take literally the
//! same host function, which is what makes the k = 1 reading the fixture
//! difference with the frame path subtracted out.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features driver-cuda-13,pie/profile-fire \
//!   --release --test cuda_frame_host_cost -- --ignored --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use client::client::Client;

/// Token budgets, run in order. TWO of them, and that is the point: a gap
/// that is a per-TOKEN cost scales with the budget and a gap that is a
/// per-LAUNCH setup cost does not, so the marginal ms/token —
/// `(t_long − t_short) / (long − short)` — is the only figure that can be
/// compared against a per-token device fire. A single budget cannot tell the
/// two apart, and build log 19's numbers were single-budget.
const BUDGETS: [u32; 2] = [64, 256];

fn frame_size() -> u32 {
    std::env::var("PIE_FRAME_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

fn build_fixture(pkg: &str, wasm_stem: &str) -> Result<(PathBuf, PathBuf)> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", pkg])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "{pkg} wasm build failed");
    let wasm = workspace.join(format!("target/wasm32-wasip2/debug/{wasm_stem}.wasm"));
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());
    Ok((wasm, workspace.join(format!("{pkg}/Pie.toml"))))
}

/// One launch's scheduler-visible cost.
struct Arm {
    label: &'static str,
    budget: u32,
    total_ms: f64,
    text: String,
    count: u64,
    ms_per_token: f64,
    batches: u64,
    batch_lat_avg_ms: f64,
    bubble_p50_us: u64,
    bubble_p99_us: u64,
    envelopes: u64,
    /// Device STARVATION, from the frame policy's own stamp: micros the
    /// device sat idle between a retirement and the next frame post, and how
    /// many such gaps. This is the honest bubble — `bubble_us_hist` counts
    /// only the gaps, so its p50 is a median OVER STARVATIONS and says
    /// nothing about how often they happen.
    device_idle_us: u64,
    device_idle_gaps: u64,
    /// The driver lane's own busy time per posted launch: the floor a frame
    /// can never beat, with queueing excluded (unlike `batch_lat_avg`, which
    /// is post→retire and therefore counts time spent behind another frame).
    lane_launch_us: u64,
    lane_launch_n: u64,
    /// Scheduler-thread serial ingest, and the guest's own turnaround.
    accept_us: u64,
    accept_calls: u64,
    turnaround_sum_us: u64,
    turnaround_n: u64,
    /// `execute` children (profile-fire only).
    batch_build_us: u64,
    driver_fire_us: u64,
    seal_events: u64,
    seal_while_executing: u64,
    /// The guest thread's own submit, phase by phase (palo D0).
    host: engine::scheduler::HostSubmitStats,
}

/// The bucket bounds `engine::scheduler::stats::BUBBLE_HIST_UPPER_US` states.
/// Restated here rather than imported because that module is `pub(crate)` —
/// the same restatement `cuda_contention` makes, for the same reason.
const BUBBLE_HIST_UPPER_US: [u64; 16] = [
    1, 2, 4, 8, 16, 32, 64, 100, 150, 250, 500, 1_000, 2_000, 8_000, 32_000, u64::MAX,
];

fn bubble_percentile(hist: &[u64], q: f64) -> u64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0;
    }
    let target = (total as f64 * q).ceil() as u64;
    let mut cum = 0u64;
    for (i, &count) in hist.iter().enumerate() {
        cum += count;
        if cum >= target {
            return BUBBLE_HIST_UPPER_US[i];
        }
    }
    *BUBBLE_HIST_UPPER_US.last().unwrap()
}

async fn run_arm(
    client: &Client,
    label: &'static str,
    program: &str,
    input: serde_json::Value,
) -> Result<Arm> {
    let before = engine::scheduler::get_stats().await;
    let envelopes_before = engine::driver::envelopes_resolved();
    let started = Instant::now();
    let mut proc = client
        .launch_process(program.to_string(), input.to_string(), true)
        .await
        .with_context(|| format!("launch {program}"))?;
    let out = proc
        .wait_for_return()
        .await
        .with_context(|| format!("wait_for_return {program}"))?;
    let elapsed = started.elapsed();
    let after = engine::scheduler::get_stats().await;
    let envelopes = engine::driver::envelopes_resolved() - envelopes_before;

    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    let text = parsed["text"].as_str().unwrap_or_default().to_string();
    let count = parsed["count"].as_u64().unwrap_or_default();

    let batches = after.total_batches - before.total_batches;
    let latency_us = after.cumulative_batch_latency_us - before.cumulative_batch_latency_us;
    let hist: Vec<u64> = after
        .bubble_us_hist
        .iter()
        .zip(before.bubble_us_hist)
        .map(|(a, b)| a - b)
        .collect();

    Ok(Arm {
        label,
        budget: input["max_tokens"].as_u64().unwrap_or(0) as u32,
        total_ms: elapsed.as_secs_f64() * 1e3,
        text,
        count,
        ms_per_token: elapsed.as_secs_f64() * 1e3 / count.max(1) as f64,
        batches,
        batch_lat_avg_ms: latency_us as f64 / batches.max(1) as f64 / 1e3,
        bubble_p50_us: bubble_percentile(&hist, 0.50),
        bubble_p99_us: bubble_percentile(&hist, 0.99),
        envelopes,
        device_idle_us: after.fire.quorum.device_idle_us - before.fire.quorum.device_idle_us,
        device_idle_gaps: after.fire.quorum.device_idle_gaps - before.fire.quorum.device_idle_gaps,
        lane_launch_us: after.fire.quorum.lane_launch_us - before.fire.quorum.lane_launch_us,
        lane_launch_n: after.fire.quorum.lane_launch_n - before.fire.quorum.lane_launch_n,
        accept_us: after.fire.quorum.accept_us - before.fire.quorum.accept_us,
        accept_calls: after.fire.quorum.accept_calls - before.fire.quorum.accept_calls,
        turnaround_sum_us: after.fire.quorum.turnaround_sum_us - before.fire.quorum.turnaround_sum_us,
        turnaround_n: after.fire.quorum.turnaround_n - before.fire.quorum.turnaround_n,
        batch_build_us: after.fire.execute.batch_build_us_sum - before.fire.execute.batch_build_us_sum,
        driver_fire_us: after.fire.execute.driver_fire_us_sum - before.fire.execute.driver_fire_us_sum,
        seal_events: after.fire.quorum.seal_events - before.fire.quorum.seal_events,
        seal_while_executing: after.fire.quorum.seal_while_executing
            - before.fire.quorum.seal_while_executing,
        host: engine::scheduler::HostSubmitStats {
            submits: after.host_submit.submits - before.host_submit.submits,
            total_us: after.host_submit.total_us - before.host_submit.total_us,
            drain_settled_us: after.host_submit.drain_settled_us
                - before.host_submit.drain_settled_us,
            geometry_us: after.host_submit.geometry_us - before.host_submit.geometry_us,
            kv_prepare_us: after.host_submit.kv_prepare_us - before.host_submit.kv_prepare_us,
            translation_us: after.host_submit.translation_us - before.host_submit.translation_us,
            scheduler_submit_us: after.host_submit.scheduler_submit_us
                - before.host_submit.scheduler_submit_us,
            shadow_advance_us: after.host_submit.shadow_advance_us
                - before.host_submit.shadow_advance_us,
            validate_frame_us: after.host_submit.validate_frame_us
                - before.host_submit.validate_frame_us,
            validate_frame_calls: after.host_submit.validate_frame_calls
                - before.host_submit.validate_frame_calls,
        },
    })
}

fn report(arm: &Arm) {
    eprintln!(
        "[d0] {:<16} n={:<4} {:>7.1} ms total {:>6.2} ms/tok  batches={:<4} \
         batch_lat_avg={:>6.2} ms  bubble p50={:>6} us p99={:>7} us  envelopes={:<4} tokens={}",
        arm.label,
        arm.budget,
        arm.total_ms,
        arm.ms_per_token,
        arm.batches,
        arm.batch_lat_avg_ms,
        arm.bubble_p50_us,
        arm.bubble_p99_us,
        arm.envelopes,
        arm.count,
    );
    let per_tok = |us: u64| us as f64 / arm.count.max(1) as f64 / 1e3;
    eprintln!(
        "[d0]   {:<16}       idle={:>6.2} ms/tok over {:<4} gaps ({:>5.2} ms each)  \
         lane_launch={:>5.2} ms/post ({} posts)  accept={:>5.1} us/call ({})  \
         turnaround={:>5.2} ms  build={:>5.1} us/post  driver_fire={:>5.2} ms/post  \
         seal_exec={}/{}",
        "",
        per_tok(arm.device_idle_us),
        arm.device_idle_gaps,
        arm.device_idle_us as f64 / arm.device_idle_gaps.max(1) as f64 / 1e3,
        arm.lane_launch_us as f64 / arm.lane_launch_n.max(1) as f64 / 1e3,
        arm.lane_launch_n,
        arm.accept_us as f64 / arm.accept_calls.max(1) as f64,
        arm.accept_calls,
        arm.turnaround_sum_us as f64 / arm.turnaround_n.max(1) as f64 / 1e3,
        arm.batch_build_us as f64 / arm.lane_launch_n.max(1) as f64,
        arm.driver_fire_us as f64 / arm.lane_launch_n.max(1) as f64 / 1e3,
        arm.seal_while_executing,
        arm.seal_events,
    );
    let h = &arm.host;
    let n = h.submits.max(1) as f64;
    eprintln!(
        "[d0]   {:<16}       submit={:>6.1} us/fire over {} = drain {:>5.1} + geometry {:>5.1} \
         + kv {:>6.1} + xlat {:>5.1} + sched {:>5.1} + shadow {:>5.1}   \
         validate_frame={:>5.1} us x{}",
        "",
        h.total_us as f64 / n,
        h.submits,
        h.drain_settled_us as f64 / n,
        h.geometry_us as f64 / n,
        h.kv_prepare_us as f64 / n,
        h.translation_us as f64 / n,
        h.scheduler_submit_us as f64 / n,
        h.shadow_advance_us as f64 / n,
        h.validate_frame_us as f64 / h.validate_frame_calls.max(1) as f64,
        h.validate_frame_calls,
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "palo D0's instrument: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn the_frame_path_and_the_plain_path_measured_in_one_process() -> Result<()> {
    common::init_trace();
    let k = frame_size();
    let pie = common::boot_serving_frame(Some(k)).await?;
    eprintln!("[d0] booted at frame_size={k}, listen_addr={}", pie.listen_addr);

    let (host_wasm, host_manifest) = build_fixture("text-completion", "text_completion")?;
    let (dev_wasm, dev_manifest) = build_fixture("token-healing", "token_healing")?;
    // THE THIRD ARM, and it is the control that separates the fixture from
    // the engine: `naive-baseline` is device-carried and `run_ahead`-driven
    // exactly like `token-healing`, and it never calls `model::vocabs()`. If
    // the device path's per-launch constant follows the RUN-AHEAD it shows up
    // here too; if it follows the VOCABULARY it does not. Its sampler is not
    // greedy, so only its clock is read — never its tokens.
    let (naive_wasm, naive_manifest) = build_fixture("naive-baseline", "naive_baseline")?;

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client
        .authenticate("test-user", &None)
        .await
        .context("auth")?;
    client
        .add_program(&host_wasm, &host_manifest, true)
        .await
        .context("add_program text-completion")?;
    client
        .add_program(&dev_wasm, &dev_manifest, true)
        .await
        .context("add_program token-healing")?;
    client
        .add_program(&naive_wasm, &naive_manifest, true)
        .await
        .context("add_program naive-baseline")?;

    let host_of = |n: u32| serde_json::json!({
        "prompt": common::SERVING_PROMPT,
        "max_tokens": n,
    });
    let dev_of = |n: u32| serde_json::json!({
        "prompt": common::SERVING_PROMPT,
        "max_tokens": n,
        "heal": false,
    });

    // A warm pass first — the JIT, the cubin cache and the autotuner all tune
    // on a shape's SECOND sighting (build log 11), so a cold arm is measuring
    // a different tactic ladder than a warm one.
    let _ = run_arm(&client, "warmup/host", "text-completion@0.1.0", host_of(64)).await?;
    let _ = run_arm(&client, "warmup/dev", "token-healing@0.1.0", dev_of(64)).await?;
    let _ = run_arm(&client, "warmup/naive", "naive-baseline@0.1.0", host_of(64)).await?;

    let mut arms = Vec::new();
    for &n in &BUDGETS {
        for pass in 0..2 {
            // Order reversed on the second pass: a difference that follows the
            // ORDER is warm-up, and a difference that follows the FIXTURE is
            // the subject.
            if pass == 0 {
                arms.push(run_arm(&client, "host/submit", "text-completion@0.1.0", host_of(n)).await?);
                arms.push(run_arm(&client, "device/run_ahead", "token-healing@0.1.0", dev_of(n)).await?);
            } else {
                arms.push(run_arm(&client, "device/run_ahead", "token-healing@0.1.0", dev_of(n)).await?);
                arms.push(run_arm(&client, "host/submit", "text-completion@0.1.0", host_of(n)).await?);
            }
            arms.push(run_arm(&client, "device/naive", "naive-baseline@0.1.0", host_of(n)).await?);
        }
    }

    eprintln!("[d0] ── k={k} ─────────────────────────────────────────────");
    for arm in &arms {
        report(arm);
    }

    // The marginal cost: the slope between the two budgets, per label. This is
    // the steady-state decode cost with every per-launch constant — the guest's
    // own setup, the prefill, the process spawn — differenced away.
    for label in ["host/submit", "device/run_ahead", "device/naive"] {
        let mean = |n: u32| -> f64 {
            let xs: Vec<f64> = arms
                .iter()
                .filter(|a| a.label == label && a.budget == n)
                .map(|a| a.total_ms)
                .collect();
            xs.iter().sum::<f64>() / xs.len().max(1) as f64
        };
        let (short, long) = (BUDGETS[0], BUDGETS[1]);
        let marginal = (mean(long) - mean(short)) / f64::from(long - short);
        let fixed = mean(short) - marginal * f64::from(short);
        eprintln!(
            "[d0] {label:<16} marginal={marginal:.3} ms/tok   per-launch fixed={fixed:.1} ms"
        );
    }

    for arm in arms.iter().filter(|a| a.label != "device/naive") {
        assert_eq!(
            arm.count, u64::from(arm.budget),
            "{} returned {} of {} tokens",
            arm.label, arm.count, arm.budget
        );
        assert!(
            arm.text.starts_with(common::SERVING_GREEDY_16),
            "{} answered {:?}; the tokens are the control and they must not move",
            arm.label,
            arm.text
        );
    }

    // THE GATE, and it is a ratio rather than a millisecond, because a
    // millisecond is a claim about this box.
    //
    // A run-ahead lane's whole promise is that its NEXT frame is fully
    // submitted before the current one retires — `seal_while_executing /
    // seal_events`, the frame policy's own chain-engagement counter. At 1.0
    // the guest's host work is behind the device and invisible; at 0.0 every
    // frame boundary starts from a standing still and every microsecond the
    // guest spends is a microsecond the device does not compute.
    //
    // This is what palo D0 found broken and what it fixed. `naive-baseline`
    // read 0/129 before the fold learned to skip its own dead arithmetic —
    // one 5.3 ms `HostShadow::advance` per fire, on the guest's thread,
    // between the take and the resubmit — and 127/129 after. A threshold in
    // milliseconds would have moved with the box; this number could not have
    // passed by accident on any box.
    //
    // Asked at the LONG budget only: a short launch is mostly its own
    // priming, where a standing start is the truth rather than a defect.
    for arm in arms
        .iter()
        .filter(|a| a.label.starts_with("device/") && a.budget == BUDGETS[1])
    {
        let engagement = arm.seal_while_executing as f64 / arm.seal_events.max(1) as f64;
        assert!(
            engagement >= 0.90,
            "{} sealed {} of {} frames while the device was executing ({:.2} \
             chain engagement): a run-ahead lane that assembles its next frame \
             only after the current one retires is not running ahead, and the \
             device idled {:.2} ms/token over {} gaps to prove it",
            arm.label,
            arm.seal_while_executing,
            arm.seal_events,
            engagement,
            arm.device_idle_us as f64 / arm.count.max(1) as f64 / 1e3,
            arm.device_idle_gaps,
        );
    }

    pie.shutdown().await;
    Ok(())
}
