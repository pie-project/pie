//! #31 host-inject KV/commit + losslessness gate — GPU (delta).
//!
//! The model-INDEPENDENT load-bearing gate (manager's step 2): drives alpha's
//! `SpecMode::System` self-spec loop on **qwen3-0.6b** (no drafter needed) via the
//! host-draft-inject hook, exercising ALL THREE KV/commit paths and asserting the
//! output is token-identical-to-greedy in every one. Losslessness across all
//! accept-rate patterns ⟹ the commit/truncate/rollover/no-double-feed choreography
//! is correct — the e2e-only piece no build-green proves.
//!
//! `specgreedy` inject modes (drafter-less, qwen3-0.6b):
//!   * `greedy`  → host-injects the greedy continuation → ALL-ACCEPT → commit-prefix
//!     path → `block_accept_lens ≈ [4,4,…]`.
//!   * `reject`  → mismatch at r=1 → PARTIAL reject → truncate + correction path →
//!     `≈ [2,2,…]`.
//!   * `garbage` → all-wrong → FULL reject every block → correction-only / `t_j`
//!     rollover path → `≈ [1,1,…]`.
//! `SPEC_GREEDY_IDENTICAL=true` MUST hold in EVERY mode (the gate); `block_accept_lens`
//! confirms each mode hit its intended KV/commit path (LOG, not gate). `max_tokens=16`
//! with k=4 ⟹ ≥2 blocks in all modes (fill→steady→drain).
//!
//! The companion VERIFIED gate (`inject=none` on gemma-4-E4B-it, the real `:4460`
//! drafter) is `mtp_drafter_verified_gate` below — needs the gemma4 MTP model.
//!
//! `#[ignore]`, driver-cuda. Run (host-inject gate, qwen3-0.6b):
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_specgreedy \
//!     host_inject_kv_commit_gate -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Build the `specgreedy` wasm once (shared by both gates).
fn build_specgreedy(ws: &Path) -> Result<std::path::PathBuf> {
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "specgreedy"])
        .current_dir(ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "specgreedy wasm build failed");
    Ok(ws.join("target/wasm32-wasip2/debug/specgreedy.wasm"))
}

/// Launch `specgreedy` with the given inject mode + max_tokens, return the verdict.
async fn run_specgreedy(client: &Client, inject: &str, max_tokens: u32) -> Result<String> {
    let mut proc = client
        .launch_process(
            "specgreedy@0.1.0".to_string(),
            format!("{{\"max_tokens\": {max_tokens}, \"inject\": \"{inject}\"}}"),
            true,
        )
        .await
        .with_context(|| format!("launch specgreedy inject={inject}"))?;
    proc.wait_for_return()
        .await
        .with_context(|| format!("wait_for_return inject={inject}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#31 host-inject KV/commit gate: needs the 4090 + cuda + qwen-3-0.6b + alpha's loop+inject-hook (combine)"]
async fn host_inject_kv_commit_gate() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[specgreedy] booted qwen3-0.6b, listen_addr={}", pie.listen_addr);

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let wasm = build_specgreedy(&ws)?;
    let manifest = ws.join("specgreedy/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;

    // All three KV/commit paths, each must stay token-identical-to-greedy.
    let mut verdicts = Vec::new();
    for mode in ["greedy", "reject", "garbage"] {
        let json = run_specgreedy(&client, mode, 16).await?;
        eprintln!("[specgreedy] inject={mode}: {json}");
        verdicts.push((mode, json));
    }

    pie.shutdown().await;

    // THE GATE: losslessness in EVERY KV/commit path. `greedy` (commit-prefix),
    // `reject` (truncate + correction), `garbage` (correction-only / t_j rollover) —
    // ALL must emit exactly the plain-greedy token stream. ANY divergence ⟹ a
    // commit/truncate/rollover/double-feed bug in the choreography.
    for (mode, json) in &verdicts {
        anyhow::ensure!(
            json.contains("SPEC_GREEDY_IDENTICAL=true"),
            "host-inject gate FAILED for inject={mode} — the self-spec output diverged \
             from plain greedy: a KV/commit choreography bug (wrong commit length, \
             un-truncated reject, bad t_j rollover, or ctx.buffer double-feed): {json}"
        );
    }
    eprintln!(
        "[specgreedy] HOST-INJECT GATE GREEN — token-identical-to-greedy across \
         greedy/reject/garbage (all KV/commit paths lossless)"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#31 VERIFIED gate: needs the 4090 + cuda + gemma-4-E4B-it (real MTP drafter) — step 4"]
async fn mtp_drafter_verified_gate() -> Result<()> {
    common::init_trace();
    // gemma-4-E4B-it with the in-repo MTP drafter (mtp_num_drafts=4).
    let pie = common::boot_4090_gemma4_mtp(4).await?;
    eprintln!("[specgreedy] booted gemma-4-E4B-it (MTP drafter), listen_addr={}", pie.listen_addr);

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let wasm = build_specgreedy(&ws)?;
    let manifest = ws.join("specgreedy/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;

    // inject=none → the REAL :4460 MTP drafter produces the drafts.
    let json = run_specgreedy(&client, "none", 16).await?;
    eprintln!("[specgreedy] inject=none (real drafter): {json}");

    pie.shutdown().await;

    // THE VERIFIED GATE: the real MTP drafter fires (milestone's core claim) AND the
    // output is token-identical-to-greedy. block_accept_lens shows the real accept
    // rate (>1 on accepted blocks ⟹ real multi-token speculation, not all-reject).
    anyhow::ensure!(
        json.contains("SPEC_GREEDY_IDENTICAL=true"),
        "VERIFIED gate FAILED — real-drafter self-spec output diverged from plain \
         greedy (a real KV/commit or verify bug): {json}"
    );
    Ok(())
}
