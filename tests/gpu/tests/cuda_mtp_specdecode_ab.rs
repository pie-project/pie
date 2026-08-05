//! **Two independent `forward-hybrid` spec-decoders, run back to back** on the
//! REAL 4090 + Qwen3.5-0.8B:
//!
//!   A = `mtp-specdecode`     B = `mtp-native-verify` (host mode)
//!
//! This began as a device-resident A/B: A was to read its drafts through a
//! `Binding::MtpDrafts` intrinsic plus a `carrier::next_inputs_drafts`
//! retain/inject command, so the `[k]` drafts never round-tripped through the
//! host. Both halves of that capability were REMOVED in the ptir refactor, so A
//! and B are now the SAME algorithm reached through two independently written
//! call sites. The decode times printed below are therefore NOT a
//! device-residency signal — A runs first and absorbs warm-up — and must not be
//! read as a delta. Do not restore the "device-resident is faster" claim without
//! restoring the intrinsic.
//!
//! What the pair is still worth, and why it is kept:
//!
//!   * **A second `pie:inferlet/forward-hybrid` client.** `mtp-specdecode` used
//!     to build through the generic `pie:inferlet/forward`, which the host now
//!     refuses on a model with a folded recurrent state; it was ported with the
//!     interface split. Two independently written call sites exercise the
//!     fold-behind + `discard-buffered` boundary, so a boundary regression has
//!     to break both to go unnoticed.
//!
//! What the pair CANNOT do is compare their trajectories, and the temptation is
//! worth naming: both trace the same graph over the same weights in the same
//! boot, which looks like it should fix the reduction order and so every argmax
//! tie. Measured, it does not — see the gate at the bottom of this file.
//!
//! ⚠️ GPU-only (the `MtpLogits` intrinsic is disabled in the mock profile).
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_MTP_DRAFT_TOKENS=4 cargo test -p pie-gpu-tests \
//!     --features driver-cuda --test cuda_mtp_specdecode_ab -- --ignored --nocapture

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use pie_client::client::Client;

mod common;

/// Both inferlets decode until `generated >= MAX_TOKENS` (16) on top of a
/// prompt they tokenize themselves, so `committed` is always at least the token
/// budget. Deliberately does NOT include the prompt: its length is the
/// tokenizer's business, and this is a "the loop ran to completion" floor, not
/// an exact count — an accepting window commits more than one token, so the
/// bound overshoots by a trajectory-dependent amount.
const MIN_COMMITTED: usize = 16;

/// Draft window k, handed to the driver as `mtp_num_drafts`. `PIE_MTP_DRAFT_TOKENS`
/// selects the arm; below 2 there is nothing to A/B, so it is ignored.
fn draft_k() -> u32 {
    std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&k| k >= 2)
        .unwrap_or(4)
}

/// Build one inferlet crate to wasm32-wasip2.
fn build_wasm(ws: &Path, pkg: &str) -> Result<()> {
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", pkg])
        .current_dir(ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for {pkg}");
    Ok(())
}

/// Launch one inferlet (already added) on a fresh session, timing the decode
/// (launch → return), and return `(result_json, decode_elapsed)`.
async fn run_inferlet(
    listen_addr: &std::net::SocketAddr,
    prog: &str,
    k: u32,
) -> Result<(String, std::time::Duration)> {
    let c = Client::connect_with_identity(&format!("ws://{listen_addr}/v1/ws"), "test-user")
        .await
        .context("connect session")?;
    c.authenticate("test-user", &None)
        .await
        .context("auth session")?;
    let t0 = Instant::now();
    let mut proc = c
        .launch_process(prog.to_string(), k.to_string(), true)
        .await
        .with_context(|| format!("launch {prog}"))?;
    let json = proc
        .wait_for_return()
        .await
        .with_context(|| format!("wait_for_return {prog}"))?;
    let dt = t0.elapsed();
    drop(c);
    Ok((json, dt))
}

/// Parse `mean_accept=<f>` and `committed=<n>` from an inferlet result line.
fn parse_metrics(json: &str) -> (f64, usize) {
    let mean = json
        .split("mean_accept=")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(f64::NAN);
    let committed = json
        .split("committed=")
        .nth(1)
        .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    (mean, committed)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "(b) device-resident MTP spec-decode A/B: needs the 4090 + cuda + Qwen3.5-0.8B (MTP head). \
            Run: PIE_MTP_DRAFT_TOKENS=4 PIE_COMPILER_LAUNCHER=env"]
async fn mtp_specdecode_device_ab() -> Result<()> {
    common::init_trace();
    let k = draft_k();
    eprintln!("[specdecode-ab] k = {k}");

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    build_wasm(&ws, "mtp-specdecode")?;
    build_wasm(&ws, "mtp-native-verify")?;

    let pie = common::boot_4090_mtp(k).await?;
    eprintln!(
        "[specdecode-ab] booted Qwen3.5-0.8B, listen_addr={}",
        pie.listen_addr
    );

    // Register both programs on one setup session.
    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup
        .authenticate("test-user", &None)
        .await
        .context("auth setup")?;
    for (pkg, file) in [
        ("mtp-specdecode", "mtp_specdecode.wasm"),
        ("mtp-native-verify", "mtp_native_verify.wasm"),
    ] {
        let wasm = ws.join(format!("target/wasm32-wasip2/debug/{file}"));
        let man = ws.join(format!("{pkg}/Pie.toml"));
        setup
            .add_program(&wasm, &man, true)
            .await
            .with_context(|| format!("add_program {pkg}"))?;
    }
    drop(setup);

    let (a_json, a_dt) = run_inferlet(&pie.listen_addr, "mtp-specdecode@0.1.0", k).await?;
    eprintln!("[specdecode-ab] A (mtp-specdecode) [{a_dt:?}]: {a_json}");
    let (b_json, b_dt) = run_inferlet(&pie.listen_addr, "mtp-native-verify@0.1.0", k).await?;
    eprintln!("[specdecode-ab] B (mtp-native-verify) [{b_dt:?}]: {b_json}");

    pie.shutdown().await;

    anyhow::ensure!(
        a_json.contains("mtp-specdecode"),
        "A (mtp-specdecode) did not return (fire error / seam?): {a_json}"
    );
    anyhow::ensure!(
        b_json.contains("mtp-native-verify"),
        "B (mtp-native-verify) did not return: {b_json}"
    );

    let (a_mean, a_commit) = parse_metrics(&a_json);
    let (b_mean, b_commit) = parse_metrics(&b_json);
    eprintln!("═══════════════════ MTP spec-decode A/B — 4090 / Qwen3.5-0.8B ═══════════════════");
    eprintln!(
        "  A mtp-specdecode    : mean_accept={a_mean:.2}  committed={a_commit}  decode={a_dt:?}"
    );
    eprintln!(
        "  B mtp-native-verify : mean_accept={b_mean:.2}  committed={b_commit}  decode={b_dt:?}"
    );
    eprintln!("  (the decode times are NOT comparable: A runs first and absorbs warm-up)");

    // What this pair can and cannot assert.
    //
    // It CANNOT assert trajectory equality, and the temptation is worth naming
    // because two consecutive runs will happily suggest otherwise. A and B
    // trace the same graph over the same weights in the same engine boot, so it
    // is easy to argue they must take the same reduction order and therefore
    // break every argmax tie the same way. Measured over repeated runs they do
    // not: the same arm yields mean_accept 0.15, 0.15, 0.00 on consecutive
    // launches, because the drafts feed back into the next window and one tie
    // broken the other way by ordinary bf16 reduction-order noise forks the
    // whole remaining trajectory. `mtp-native-verify`'s own source says the
    // same thing about three identical host-mode launches.
    //
    // So the acceptance rate is not a gate, and neither is the exact token
    // count: a window that accepts its drafts commits more than one token, so
    // the loop's `generated < MAX_TOKENS` bound can overshoot by different
    // amounts on different trajectories.
    //
    // What IS invariant is that both decoders RUN — each completes its decode
    // loop, folds and discards its way to the token budget, and reports a
    // well-formed acceptance rate. That is the coverage claim: two independent
    // `forward-hybrid` call sites exercise the fold/discard boundary, and a
    // boundary regression shows up as a fire error or a stalled loop, not as a
    // shifted acceptance rate.
    anyhow::ensure!(
        a_mean.is_finite() && b_mean.is_finite(),
        "could not parse mean_accept from both runs (A={a_json}, B={b_json})"
    );
    let floor = MIN_COMMITTED;
    anyhow::ensure!(
        a_commit >= floor && b_commit >= floor,
        "a decoder stopped short of its token budget: A committed={a_commit}, \
         B committed={b_commit}, expected at least {floor}"
    );
    anyhow::ensure!(
        (0.0..=f64::from(k)).contains(&a_mean) && (0.0..=f64::from(k)).contains(&b_mean),
        "mean_accept out of range for k={k}: A={a_mean}, B={b_mean}"
    );
    Ok(())
}
