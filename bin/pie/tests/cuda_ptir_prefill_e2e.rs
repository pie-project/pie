//! PTIR multi-token PROMPT PREFILL — DEVICE REPRODUCER of the driver gap (4090).
//! The classic `forward-pass` removal hinges on reproducing a variable-length
//! prompt prefill (what classic `carrier::submit_pass` does) on the ptir path.
//!
//! FINDING (device-verified, 2026-07-09): a naive N-wide prefill fire on the ptir
//! device-geometry path does NOT attend the prompt — the continuation is
//! incoherent (garbage/degenerate), because the driver paths are DECODE-SHAPED:
//! the dense-AttnMask pack keys off `lanes = #sequences` (one mask row per lane)
//! and the explicit-KV-write is a single-cell-per-lane write; neither expresses
//! the `[N_query, KV]` mask + N-cell write a prefill needs. The standard
//! (pure-causal, page-derived-write) route also fails (ambiguous KvLen
//! read/write ranges for a fresh multi-query fire). See the inferlet doc-comment
//! for the exact executor.cpp citations. Closing this is a load-bearing driver
//! contract call — FLAGGED, not invented.
//!
//! This test is therefore a REPRODUCER / go-green target, NOT a passing gate:
//! it asserts only that the N-wide fire ran end-to-end without a fire-path crash
//! (the pipeline plumbing is correct). It deliberately does NOT assert output
//! correctness — coherence needs an oracle, and today's output is the
//! characterized garbage. It goes fully green once the driver prefill path lands
//! and the continuation is coherent (human-eval `text=…`).
//!
//!   PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_ptir_prefill_e2e -j6 -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "ptir prompt-prefill device e2e: needs the 4090 + cuda + qwen-3-0.6b + the ptir feature"]
async fn ptir_prefill_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[ptir-prefill-e2e] booted, listen_addr={}", pie.listen_addr);

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "ptir-prefill-e2e",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "ptir-prefill-e2e wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/ptir_prefill_e2e.wasm");
    let manifest = ws.join("ptir-prefill-e2e/Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());

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
    eprintln!("[ptir-prefill-e2e] program installed, launching prefill+decode…");

    let mut proc = client
        .launch_process("ptir-prefill-e2e@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[ptir-prefill-e2e] returned: {out}");

    pie.shutdown().await;

    anyhow::ensure!(
        out.contains("PTIR_PREFILL_E2E"),
        "ptir prefill e2e: unexpected return (expected a `PTIR_PREFILL_E2E …` summary; \
         a fire-path crash / prefill reject returns the error string): {out:?}"
    );
    // Plumbing-only gate: the N-wide prefill fire + shared-pool decode ran e2e
    // without a fire-path crash and harvested tokens. This does NOT assert output
    // correctness — see the module doc: today's continuation is the characterized
    // garbage (the driver prefill-attention gap). Goes fully green (coherent
    // continuation) once the driver prefill path lands.
    let has_tokens = out.contains("tokens=[") && !out.contains("tokens=[]");
    anyhow::ensure!(
        has_tokens,
        "ptir prefill e2e: no tokens harvested (fire-path plumbing broke): {out:?}"
    );
    eprintln!(
        "[ptir-prefill-e2e] ran e2e (plumbing OK). NOTE: output correctness is NOT \
         asserted — this is the driver-prefill-gap reproducer; inspect `text=…` for \
         coherence once the driver prefill path lands: {out}"
    );
    Ok(())
}
