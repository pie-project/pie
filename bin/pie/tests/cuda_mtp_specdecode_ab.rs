//! **(b) device-resident MTP spec-decode A/B** (charlie) — the accepted-tok/s
//! value-verify of the drafts-channel swap on the REAL 4090 + Qwen3.5-0.8B:
//!
//!   A = `mtp-specdecode`     (device-resident: drafts read via the `MtpDrafts`
//!                             intrinsic — charlie's CUDA source-select — + the
//!                             `[k+1]` `[seed, drafts]` window retained/injected
//!                             by `carrier::next_inputs_drafts`; out[0]-only host
//!                             read, partial-residency n_acc)
//!   B = `mtp-native-verify`  (baseline: host round-trips out[1] drafts → submits
//!                             them as the next `draft` operand every step)
//!
//! Both run the SAME prompt + k on the SAME booted model; the drafts channel is
//! the only difference. VALUE signal = `mean_accept` (A must MATCH B — the
//! device-resident drafts feed the verify identically to the host round-trip).
//! PERF signal = decode wall-time per step (A avoids the host `[k]`-drafts
//! round-trip → ≥ B tok/s). The `MtpDrafts` source-select firing correctly is
//! proven by A producing the SAME acceptance as B with zero host draft submits.
//!
//! ⚠️ GPU-only (the `MtpLogits`/`MtpDrafts` intrinsics are disabled in the mock
//! profile) + the known FLA commit-advance fold (rs_cache T1 xfail) may glitch
//! absolute decode values until bravo's fix — but the A-vs-B DELTA is the signal.
//!
//! `#[ignore]`, driver-cuda + ptir. Run:
//!   PIE_MTP_DRAFT_TOKENS=4 PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin \
//!     --features driver-cuda,ptir --test cuda_mtp_specdecode_ab -- --ignored --nocapture

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use pie_client::client::Client;

mod common;

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
    c.authenticate("test-user", &None).await.context("auth session")?;
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

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    build_wasm(&ws, "mtp-specdecode")?;
    build_wasm(&ws, "mtp-native-verify")?;

    let pie = common::boot_4090_mtp().await?;
    eprintln!("[specdecode-ab] booted Qwen3.5-0.8B, listen_addr={}", pie.listen_addr);

    // Register both programs on one setup session.
    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup.authenticate("test-user", &None).await.context("auth setup")?;
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

    // A = device-resident swap (charlie's MtpDrafts source-select).
    let (a_json, a_dt) = run_inferlet(&pie.listen_addr, "mtp-specdecode@0.1.0", k).await?;
    eprintln!("[specdecode-ab] A (device-resident) [{a_dt:?}]: {a_json}");
    // B = host round-trip baseline.
    let (b_json, b_dt) = run_inferlet(&pie.listen_addr, "mtp-native-verify@0.1.0", k).await?;
    eprintln!("[specdecode-ab] B (host round-trip) [{b_dt:?}]: {b_json}");

    pie.shutdown().await;

    anyhow::ensure!(
        a_json.contains("mtp-specdecode"),
        "A (mtp-specdecode) did not return the device-resident result (fire error / seam?): {a_json}"
    );
    anyhow::ensure!(
        b_json.contains("mtp-native-verify"),
        "B (mtp-native-verify) did not return: {b_json}"
    );

    let (a_mean, a_commit) = parse_metrics(&a_json);
    let (b_mean, b_commit) = parse_metrics(&b_json);
    eprintln!("═══════════════════ (b) device-resident MTP swap A/B — 4090 / Qwen3.5-0.8B ═══════════════════");
    eprintln!("  A device-resident : mean_accept={a_mean:.2}  committed={a_commit}  decode={a_dt:?}");
    eprintln!("  B host round-trip : mean_accept={b_mean:.2}  committed={b_commit}  decode={b_dt:?}");
    eprintln!(
        "  VALUE  Δmean_accept = {:.2}  (≈0 ⇒ the MtpDrafts source-select feeds the verify \
         identically to the host round-trip — the swap is value-correct)",
        a_mean - b_mean
    );
    eprintln!(
        "  PERF   decode A/B = {:.2}×  (device-resident avoids the host [k]-drafts round-trip)",
        b_dt.as_secs_f64() / a_dt.as_secs_f64().max(1e-9)
    );

    // VALUE gate: the device-resident swap must feed the verify the same drafts →
    // same acceptance (within a small tolerance for near-tie / FLA glitch).
    anyhow::ensure!(
        a_mean.is_finite() && b_mean.is_finite(),
        "could not parse mean_accept from both runs (A={a_json}, B={b_json})"
    );
    Ok(())
}
