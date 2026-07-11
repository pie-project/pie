//! **Drafts-channel retain plumbing gate** (charlie) — the FLEET=1-style
//! byte-identity proof for the `[k+1]` `[seed, drafts]` device window through the
//! `pipeline_source_kind=1` retain path, on the real 4090 + Qwen3.5-0.8B.
//!
//! The `drafts-retain-e2e` inferlet fires `mtp_specdecode(vocab, k)` +
//! `carrier::next_inputs_drafts(pass, k)` → the driver retains the `[k+1]` window
//! (out[2]=seed→row0, out[1]=drafts→rows1..k) off bravo's `mtp_drafts` buffer.
//! With `PIE_DRAFTS_VERIFY=1` the driver dumps the RETAIN window (and, when a
//! consumer injects, the INJECT values + `src_rows`). This harness asserts the
//! inferlet fired the retain and prints the EXPECTED window
//! (`DRAFTS_RETAIN_EXPECTED seed=.. drafts=..`) for byte-identity comparison with
//! the `[drafts-verify] RETAIN window[k+1]=[..]` driver line — the retain
//! composition is byte-correct iff they match. The host `src_rows=[0..=k]` wiring
//! is unit-tested in `runtime/src/api/next_input_map.rs`
//! (`drafts_channel_consumer_injects_window_src_rows`) — this is its on-device
//! value-identity companion.
//!
//! `#[ignore]` (needs the 4090 + cuda + Qwen3.5-0.8B MTP head). Run:
//!   PIE_DRAFTS_VERIFY=1 PIE_MTP_DRAFT_TOKENS=4 PIE_COMPILER_LAUNCHER=env \
//!     cargo test -p pie-bin --features driver-cuda,ptir \
//!     --test cuda_drafts_retain -- --ignored --nocapture

use std::path::Path;
use std::process::Command;

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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "drafts-retain plumbing gate: needs the 4090 + cuda + Qwen3.5-0.8B (MTP head). \
            Run: PIE_DRAFTS_VERIFY=1 PIE_MTP_DRAFT_TOKENS=4"]
async fn drafts_retain_window_byte_identity() -> Result<()> {
    common::init_trace();
    let k = draft_k();
    eprintln!("[drafts-retain] k = {k}");

    // Build the drafts-retain-e2e inferlet (wasm).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "drafts-retain-e2e",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for drafts-retain-e2e");

    let pie = common::boot_4090_mtp().await?;
    eprintln!(
        "[drafts-retain] booted Qwen3.5-0.8B, listen_addr={}",
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
    let wasm = ws.join("target/wasm32-wasip2/debug/drafts_retain_e2e.wasm");
    let man = ws.join("drafts-retain-e2e/Pie.toml");
    setup
        .add_program(&wasm, &man, true)
        .await
        .context("add_program drafts-retain-e2e")?;
    drop(setup);

    let c = Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
        .await
        .context("connect session")?;
    c.authenticate("test-user", &None)
        .await
        .context("auth session")?;
    let mut proc = c
        .launch_process("drafts-retain-e2e@0.1.0".to_string(), k.to_string(), true)
        .await
        .context("launch drafts-retain-e2e")?;
    let json = proc
        .wait_for_return()
        .await
        .context("wait_for_return drafts-retain-e2e")?;
    drop(c);
    eprintln!("[drafts-retain] result: {json}");

    pie.shutdown().await;

    // The inferlet returns "DRAFTS_RETAIN_EXPECTED k=.. seed=[..] drafts=[..]
    // window=[..]". Structural gate: the retain fired (3 outputs present, out[2]
    // seed extracted) and returned the composed window.
    anyhow::ensure!(
        json.contains("DRAFTS_RETAIN_EXPECTED"),
        "retain did not fire / 3 outputs missing (did the fire error?): {json}"
    );
    eprintln!(
        "[drafts-retain] BYTE-IDENTITY: the `window=[seed, drafts]` above MUST equal the \
         `[drafts-verify] RETAIN window[k+1]=[..]` driver line (retain composed out[2]=seed→row0, \
         out[1]=drafts→rows1..k). If a consumer injected, `[drafts-verify] INJECT src_rows=[0..={k}] \
         injected=[..]` MUST equal that window too (byte-identity through retain→inject)."
    );
    Ok(())
}
