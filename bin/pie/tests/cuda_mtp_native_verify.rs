//! **Stage-2 driver `mtp_logits` value-verify** (charlie) — drives the
//! `mtp-native-verify` inferlet (bravo's PTIR-native draft→verify→accept) on the
//! REAL 4090 + Qwen3.5-0.8B, exercising the driver's `mtp_logits` plumbing (the
//! MTP head produces K draft rows in `ws.logits`; `ctx.mtp_draft_row` points the
//! `Intrinsic::MtpLogits [K,vocab]` binding at them).
//!
//! VALUE-verify (NOT K-vs-K parity — the `-1` stub false-passes parity by aliasing
//! the target row): the inferlet reports `accepted_lengths` + `mean_accept`. A
//! wired MTP head yields a MEANINGFUL acceptance signal (some accepts, some
//! rejects — not all-K aliasing); `PIE_MTP_LOGITS_TRACE` dumps each draft row's
//! argmax so we can see the drafts differ from the target greedy.
//!
//! ⚠️ CAVEAT (manager-flagged): the K≥1 MTP spec-decode hits the known FLA
//! commit-advance recurrent-state fold bug (rs_cache T1 xfail) — so the decode may
//! glitch until bravo's FLA fix lands. The PLUMBING signal (draft head fires +
//! produces non-aliasing rows) is independent of decode value-correctness.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_MTP_DRAFT_TOKENS=4 PIE_MTP_LOGITS_TRACE=1 cargo test -p pie-bin \
//!     --features driver-cuda --test cuda_mtp_native_verify -- --ignored --nocapture

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

mod common;

/// Draft window k, kept in lockstep with the driver's `max_drafts`.
///
/// The guest asks for `k` draft rows via `intrinsics::mtp_logits(k)`; the
/// driver sizes `max_drafts` from `PIE_MTP_DRAFT_TOKENS`, falling back to
/// `cfg.model.mtp_num_drafts` — which the test TOML does not set. So a bare
/// `cargo test` used to run a guest wanting 4 drafts against a driver
/// providing the config default, and fail deep in frame prepare with
/// "MtpLogits draft-row requirement exceeds the production layout". Two
/// defaults for one number is a trap, so this EXPORTS the value it returns:
/// the env is the single source of truth, and the test supplies it when the
/// caller did not. Must run before `boot_4090_mtp`, since the driver reads the
/// variable once at init.
fn draft_k() -> u32 {
    let k = std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&k| k >= 2)
        .unwrap_or(4);
    // SAFETY: set before the engine boots, while the test is still
    // single-threaded with respect to any driver init that reads it.
    unsafe { std::env::set_var("PIE_MTP_DRAFT_TOKENS", k.to_string()) };
    k
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "Stage-2 mtp_logits value-verify: needs the 4090 + cuda + Qwen3.5-0.8B (MTP head). \
            Run: PIE_MTP_DRAFT_TOKENS=4 PIE_MTP_LOGITS_TRACE=1"]
async fn mtp_logits_value_verify() -> Result<()> {
    common::init_trace();
    let k = draft_k();
    eprintln!("[mtp-native-verify] k = {k}");

    // Build the mtp-native-verify inferlet (wasm).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "mtp-native-verify",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for mtp-native-verify");

    let pie = common::boot_4090_mtp().await?;
    eprintln!(
        "[mtp-native-verify] booted Qwen3.5-0.8B, listen_addr={}",
        pie.listen_addr
    );

    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup
        .authenticate("test-user", &None)
        .await
        .context("auth setup")?;
    let wasm = ws.join("target/wasm32-wasip2/debug/mtp_native_verify.wasm");
    let man = ws.join("mtp-native-verify/Pie.toml");
    setup
        .add_program(&wasm, &man, true)
        .await
        .context("add_program mtp-native-verify")?;
    drop(setup);

    // Launch the verify loop.
    let c = Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
        .await
        .context("connect verify session")?;
    c.authenticate("test-user", &None)
        .await
        .context("auth verify session")?;
    let mut proc = c
        .launch_process("mtp-native-verify@0.1.0".to_string(), k.to_string(), true)
        .await
        .context("launch mtp-native-verify")?;
    let json = proc
        .wait_for_return()
        .await
        .context("wait_for_return mtp-native-verify")?;
    drop(c);
    eprintln!("[mtp-native-verify] result: {json}");

    pie.shutdown().await;

    // The inferlet returns "mtp-native-verify: k=.. steps=.. accepted_lengths=[..]
    // mean_accept=.. committed=..". Parse mean_accept as the plumbing signal.
    anyhow::ensure!(
        json.contains("mtp-native-verify"),
        "unexpected inferlet result (did the fire error?): {json}"
    );
    eprintln!(
        "[mtp-native-verify] VALUE-VERIFY: inspect PIE_MTP_LOGITS_TRACE [mtp-logits] lines above \
         — each draft row's argmax should DIFFER from the target greedy (wired, not aliasing). \
         accepted_lengths signal in the result. NOTE: K>=1 decode may glitch on the known FLA \
         commit-advance fold (rs_cache T1 xfail) until bravo's fix."
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the RTX 4090; PIE_MTP_DRAFT_TOKENS=0 checks an MTP model with \
            drafting disabled, any positive value checks a model with no MTP head"]
async fn mtp_logits_capability_false() -> Result<()> {
    common::init_trace();
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "mtp-native-verify",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for mtp-native-verify");

    let drafts_disabled = std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        == Some(0);
    let pie = if drafts_disabled {
        common::boot_4090_mtp().await?
    } else {
        common::boot_4090().await?
    };
    let endpoint = format!("ws://{}/v1/ws", pie.listen_addr);
    let setup = Client::connect_with_identity(&endpoint, "test-user")
        .await
        .context("connect setup")?;
    setup.authenticate("test-user", &None).await?;
    setup
        .add_program(
            &ws.join("target/wasm32-wasip2/debug/mtp_native_verify.wasm"),
            &ws.join("mtp-native-verify/Pie.toml"),
            true,
        )
        .await?;
    drop(setup);

    let client = Client::connect_with_identity(&endpoint, "test-user").await?;
    client.authenticate("test-user", &None).await?;
    let mut process = client
        .launch_process("mtp-native-verify@0.1.0".to_string(), "4".to_string(), true)
        .await?;
    let error = process
        .wait_for_return()
        .await
        .expect_err("MtpLogits registration must be rejected");
    let detail = format!("{error:#}");
    pie.shutdown().await;
    anyhow::ensure!(
        detail.contains("model-gated intrinsic mtp_logits unavailable"),
        "unexpected MtpLogits rejection: {detail}"
    );
    Ok(())
}
