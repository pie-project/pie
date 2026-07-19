//! #31 greedy-v0 self-spec verify e2e — DEVICE-ALIAS draft source, GPU (delta).
//!
//! Boots the 4090 + real driver and launches the `selfspec` inferlet, which drives
//! `mtp_self_spec_greedy_observable` (foxtrot's A2 keystone) through echo's
//! `SelfSpecDraftInput` resolver — the verify's `[k]` drafts are DEVICE-RESIDENT off
//! `pi.tokens + sample_row + 1` (the input tokens after the anchor), source-selected
//! by the manifest flag, NO host upload. This is the e2e proof of the de-hardwired
//! draft binding + the A2 target-correction splice.
//!
//! Composed branch: foxtrot keystone+correction (`45a8f014`) + echo resolver
//! (`SelfSpecDraftInput → pi.tokens+sample_row+1`) + alpha ballast
//! (`resolve_bindings(.., &[])`). The verify program is byte-identical to
//! `spec_verify_greedy` (source-agnostic); only the draft SOURCE moved host→device.
//!
//! Non-degenerate by construction (the inferlet asserts each):
//!   * ACCEPT-ALL: `D = g` ⇒ `V == g` (device-alias read of the greedy continuation).
//!   * REJECT-MID: `D = g` w/ `D[j]` perturbed (j=k/2) ⇒ `V == [g0..g_{j-1}, g_j, 0..]`
//!     — the perturbed draft forces a reject AT j (a wrong-buffer bind rejects
//!     elsewhere) AND the A2 correction `t_j == g[j]` is spliced at the boundary.
//! The reference `A` is `g` via the Argmax SAMPLER — a different argmax path than the
//! verify's matrix intrinsic, so the cross-check catches a matrix-argmax regression
//! (#35-A class), not just the eq/cumprod/select DAG.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_selfspec31 -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#31 greedy-v0 self-spec verify: needs the 4090 + cuda + qwen-3-0.6b + the SelfSpecDraftInput device-alias resolver"]
async fn self_spec_device_alias_verify_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[self-spec] booted, listen_addr={}", pie.listen_addr);

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "selfspec"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "selfspec wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/selfspec.wasm");
    let manifest = ws.join("selfspec/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client
        .authenticate("test-user", &None)
        .await
        .context("auth")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;
    eprintln!("[self-spec] program installed, launching device-alias self-spec verify…");

    let mut proc = client
        .launch_process("selfspec@0.1.0".to_string(), "{\"k\":4}".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[self-spec] returned: {json}");

    pie.shutdown().await;

    // Precondition for the 0-sentinel detector to be UNAMBIGUOUS (extended for A2):
    // the accepted drafts AND the boundary correction t_j must be ∈ [1,vocab). A
    // token-0 anywhere there collides with the past-boundary 0-sentinels → the gate
    // can't discriminate accept/correction from reject. Fail loud, don't false-pass.
    anyhow::ensure!(
        json.contains("DRAFTS_NONZERO=true"),
        "0-sentinel self-spec detector is AMBIGUOUS — a draft or the boundary \
         correction is token-0 (indistinguishable from the reject→0 sentinel). \
         Re-seed with a prompt whose greedy continuation + correction are non-zero: {json}"
    );
    // ACCEPT-ALL: drafts == g ⇒ the verify device-alias-reads the greedy continuation
    // from `pi.tokens+sample_row+1` and every row matches ⇒ V == g (full [k], no
    // sentinel). A wrong-buffer bind (pi.sampled / wrong offset) reads garbage ⇒ rows
    // mismatch ⇒ fails.
    anyhow::ensure!(
        json.contains("ACCEPT_ALL_OK=true"),
        "self-spec ACCEPT-ALL failed — the device-aliased draft read (D=g) did NOT \
         yield V==g: echo's SelfSpecDraftInput bind read the wrong buffer/offset, or \
         the verify DAG is not value-exact: {json}"
    );
    // THE LOAD-BEARING GATE — REJECT-MID + the A2 correction. Perturbing draft j
    // (≠ g[j], nonzero) must (a) force a reject EXACTLY at j (proving the verify READ
    // the device-aliased perturbed draft — a wrong bind rejects elsewhere), and (b)
    // splice the target correction t_j == g[j] at the boundary (A2). Output must be
    // [g0..g_{j-1}, g_j, 0..] (0-sentinel non-truncating so the correction + suffix
    // are observable). A wrong bind, a collapsed cumprod, or a missing correction
    // each fails this.
    anyhow::ensure!(
        json.contains("BIND_DEVICE_ALIAS_OK=true"),
        "self-spec device-alias bind UNPROVEN — the reject did not land at j on the \
         perturbed draft: echo's resolver read the wrong buffer (pi.sampled/offset) \
         so the verify did not consume the device-aliased draft: {json}"
    );
    anyhow::ensure!(
        json.contains("A2_CORRECTION_OK=true"),
        "A2 correction MISSING/wrong — the reject boundary did not splice the target \
         greedy token t_j == g[j] (emitted the perturbed draft, a 0, or a wrong \
         token): the full-reject stall is NOT killed and v0 is not greedy-exact: {json}"
    );
    anyhow::ensure!(
        json.contains("REJECT_MID_OK=true"),
        "self-spec REJECT-MID failed — the perturbed draft did not yield the A2 \
         accept set [accepted-prefix, correction g_j, 0..]: a collapsed cross-row \
         cumprod, a row-leak past the reject, or a wrong-buffer draft read: {json}"
    );
    // The composite cross-check (V == greedy_verify_a2(D, g) for BOTH arms), the
    // honest end-to-end gate against the independent Argmax-sampler reference g.
    anyhow::ensure!(
        json.contains("CROSSCHECK_OK=true"),
        "self-spec cross-check failed — the verify output diverged from \
         greedy_verify_a2(D, g) (independent Argmax-sampler reference): {json}"
    );
    Ok(())
}
