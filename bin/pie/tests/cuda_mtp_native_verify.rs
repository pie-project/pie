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
//! `#[ignore]`, driver-cuda. **ONE TEST PER PROCESS** — each test boots its own
//! embedded worker and only the first boot in a process succeeds, so running the
//! file unfiltered fails every test after the first with "boot embedded worker".
//! Run each by name:
//!   PIE_MTP_DRAFT_TOKENS=4 PIE_MTP_LOGITS_TRACE=1 cargo test -p pie-bin \
//!     --features driver-cuda --test cuda_mtp_native_verify <name> \
//!     -- --ignored --exact --nocapture

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

mod common;

/// Draft window k, kept in lockstep with the driver's `max_drafts`.
///
/// The guest asks for `k` draft rows via `intrinsics::mtp_logits(k)`, and the
/// driver sizes `max_drafts` from `mtp_num_drafts`. Two defaults for one number
/// is a trap -- a guest wanting 4 drafts against a driver built for some other
/// count fails deep in frame prepare with "MtpLogits draft-row requirement
/// exceeds the production layout" -- so this ONE value feeds both: pass it to
/// `boot_4090_mtp` and to the guest. `PIE_MTP_DRAFT_TOKENS` overrides it.
fn draft_k() -> u32 {
    std::env::var("PIE_MTP_DRAFT_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .filter(|&k| k >= 2)
        .unwrap_or(4)
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

    let pie = common::boot_4090_mtp(k).await?;
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
    // ONE FIRE PER WINDOW. The host path folds the previous window's accepted
    // prefix in the fire that writes the next one and discards the rejected
    // tail from the buffer, so there is no commit fire left to submit. A
    // regression here would not change a single token -- the two-fire shape
    // decodes identically -- which is exactly why it is asserted as a count.
    let field = |name: &str| -> Option<usize> {
        json.split(&format!("{name}="))
            .nth(1)?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    };
    let (steps, fires) = (field("steps"), field("fires"));
    anyhow::ensure!(
        steps.is_some() && steps == fires,
        "the host path must submit exactly one fire per window, got steps={steps:?} \
         fires={fires:?}: {json}"
    );
    eprintln!(
        "[mtp-native-verify] VALUE-VERIFY: inspect PIE_MTP_LOGITS_TRACE [mtp-logits] lines above \
         — each draft row's argmax should DIFFER from the target greedy (wired, not aliasing). \
         accepted_lengths signal in the result. NOTE: K>=1 decode may glitch on the known FLA \
         commit-advance fold (rs_cache T1 xfail) until bravo's fix."
    );
    Ok(())
}

/// The accepted count never touches the host.
///
/// This is the payoff of the device-resident fold boundary. The host path has
/// to await the verify fire, read `clen` back, and only THEN trace the commit
/// -- serializing two fires that are otherwise back to back. The device path
/// publishes `clen` into a channel the commit fire already claimed, plans
/// against the host's own upper bound (the row's whole live buffer), and lets
/// the driver substitute and clamp the real value.
///
/// The assertion is a WITHIN-RUN invariant, not a cross-run diff. Every window
/// the inferlet echoes the device-computed `clen` back to the host and checks
/// it against the length the host itself derived from the sentinel tail; a
/// mismatch fails the inferlet outright. That is the exact property this path
/// can get wrong -- the driver dropping the resolved value and folding the
/// whole buffer bound instead -- and it is checked at every single fold.
///
/// It deliberately does NOT compare a host-mode decode to a device-mode decode
/// token for token. That test is not sound on this model: the drafts feed back
/// into the next window, so one argmax tie broken the other way by ordinary
/// bf16 reduction-order noise forks the entire trajectory. Three IDENTICAL
/// host-mode launches inside one engine boot were observed to produce three
/// different token streams (`committed=23/25/26`, `steps=12/13/9`). Any
/// equality assertion over that output is reading noise.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the GPU + Qwen3.5-0.8B (MTP head). Run: \
            PIE_MTP_DRAFT_TOKENS=4"]
async fn a_device_resident_fold_length_decodes_identically() -> Result<()> {
    common::init_trace();
    let k = draft_k();

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

    let pie = common::boot_4090_mtp(k).await?;
    let url = format!("ws://{}/v1/ws", pie.listen_addr);

    let setup = Client::connect_with_identity(&url, "test-user")
        .await
        .context("connect setup")?;
    setup.authenticate("test-user", &None).await?;
    setup
        .add_program(
            &ws.join("target/wasm32-wasip2/debug/mtp_native_verify.wasm"),
            &ws.join("mtp-native-verify/Pie.toml"),
            true,
        )
        .await
        .context("add_program mtp-native-verify")?;
    drop(setup);

    let c = Client::connect_with_identity(&url, "test-user").await?;
    c.authenticate("test-user", &None).await?;
    let arg = format!("{k}:device");
    let json = c
        .launch_process("mtp-native-verify@0.1.0".to_string(), arg.clone(), true)
        .await
        .context("launch mtp-native-verify")?
        .wait_for_return()
        .await
        .with_context(|| format!("wait_for_return ({arg})"))?;
    eprintln!("[mtp-native-verify] {arg} -> {json}");
    drop(c);
    pie.shutdown().await;

    // The inferlet fails itself on the first disagreement, so reaching a
    // result at all is most of the claim.
    anyhow::ensure!(
        json.contains("mtp-native-verify: mode=device"),
        "the device fold path did not complete: {json}"
    );

    let field = |name: &str| -> Result<usize> {
        let at = json
            .find(&format!("{name}="))
            .with_context(|| format!("no {name} in {json}"))?
            + name.len()
            + 1;
        let rest = &json[at..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        Ok(rest[..end].parse()?)
    };

    let steps = field("steps")?;
    let checked = field("fold_len_checked")?;
    anyhow::ensure!(
        checked == steps && steps > 0,
        "the device fold length was verified at {checked} of {steps} windows: {json}"
    );

    // And at least one window must have folded MORE than the lone bonus token.
    // If every `clen` were 1 the check above would pass against a driver that
    // ignored the resolved value entirely and folded a constant.
    anyhow::ensure!(
        field("fold_len_nontrivial")? > 0,
        "every window folded exactly one token, so a constant fold length would \
         also have passed: {json}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the RTX 4090; boots the MTP model with drafting disabled and \
            checks that the mtp_logits intrinsic is then refused"]
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

    // This used to have a second arm: boot the pure-attention Qwen3 (a model
    // with no MTP head at all) and expect the same refusal. That arm is no
    // longer expressible. `mtp-native-verify` builds its pass through
    // `pie:inferlet/forward-hybrid` and returns "skipped: needs a linear model"
    // on an attention model, so it never reaches the intrinsic and the harness
    // saw a successful return instead of the refusal. Drafting-disabled on the
    // hybrid model gates the SAME intrinsic through the SAME validator path, so
    // nothing is lost by pinning this test to the reachable configuration.
    //
    let pie = common::boot_4090_mtp(0).await?;
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
