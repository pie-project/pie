//! 1a run-ahead carryover GPU verify (golf ∥ delta). Boots the 4090 + real
//! driver and launches the `runahead` inferlet, which decodes greedily through
//! [`collect_tokens_pipelined`] — each pass's sampled token is carried into the
//! next pass's input by the device-side carrier (producer `source_link` →
//! consumer `carried_input` + `inject_link` + `free_link`), sequentially (await
//! before submit, so RETAIN precedes INJECT).
//!
//! This drives the executor-hook path (retain → inject → free) in a real fire.
//! delta's carrier instrumentation asserts `consumer.pi.tokens[dest] ==
//! producer's sample` device-side; the guest token stream MUST also match the
//! synchronous `collect_tokens` stream on the same prompt (greedy ⇒ any
//! divergence is a real carryover bug, not sampling noise).
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_runahead -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "1a run-ahead carryover verify: needs the 4090 + cuda + qwen-3-0.6b + delta's carrier"]
async fn runahead_carryover_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[runahead] booted, listen_addr={}", pie.listen_addr);

    // Build the run-ahead carryover inferlet.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "runahead"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "runahead wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/runahead.wasm");
    let manifest = ws.join("runahead/Pie.toml");

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
    eprintln!("[runahead] program installed, launching carryover decode…");

    let mut proc = client
        .launch_process("runahead@0.1.0".to_string(), "8".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[runahead] returned: {json}");

    pie.shutdown().await;

    // The inferlet runs two scenarios and reports `MATCH=<bool>` (Scenario A)
    // and `CLEAR_OK=<bool>` (Scenario B). Greedy ⇒ deterministic, so any
    // divergence is a real bug, not sampling noise.
    //
    // Scenario A — carrier token-exactness (1a milestone regression): the
    // run-ahead carrier stream equals the synchronous greedy stream.
    anyhow::ensure!(
        json.contains("MATCH=true"),
        "run-ahead carryover diverged from the synchronous stream: {json}"
    );
    // Non-degeneracy anchor — the stream must positively equal the verified
    // milestone tokens, not merely self-agree. Closes the blind spot where both
    // the pipelined and sync paths read the SAME broken `output()` (e.g. a
    // fast-path returning zeros) and FALSE-pass `pipelined == sync`.
    anyhow::ensure!(
        json.contains("ANCHOR_OK=true"),
        "run-ahead output is degenerate — not the milestone tokens \
         [198,9707,1879,374,264,4285,2025,429]; likely a broken output()/fast-path \
         returning zeros (pipelined==sync passed but both read the broken path): {json}"
    );
    // Scenario B — #26 dangling-carry clear (fresh-generate host-clear): a 2nd
    // generate on the same context after a stop-terminated pipelined generate
    // must drop the dangling carry (no stale inject, free-link not mishandled).
    anyhow::ensure!(
        json.contains("CLEAR_OK=true"),
        "#26 clear failed: 2nd generate on the same context diverged — a stale \
         carried token was injected or the device free-link was mishandled: {json}"
    );
    Ok(())
}
