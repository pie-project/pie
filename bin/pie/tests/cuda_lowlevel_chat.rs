//! **Low-level chat-EOS pipelined inferlet — 4090 value-verify** (foxtrot ∥ bravo).
//! Boots the 4090 + real driver and launches the `lowlevel-chat` inferlet, which
//! hand-writes the whole decode loop on the RAW WIT API (no helper SDK) — explicit
//! run-ahead + EOS-rollback — and self-asserts:
//!
//!   - `MATCH=true`        — the explicit pipelined stream == the sequential
//!     reference stream, byte-for-byte (greedy ⇒ deterministic; the device carrier
//!     `next_inputs(&[0])` correctly injects each fire's sample into the next).
//!   - `ROLLBACK_OK=true`  — under a FORCED early stop (a speculated successor MUST
//!     be discarded), the pipelined stream still equals the sequential one and the
//!     `drain_discard` rollback completes cleanly (depth-1 fire-past-EOS discard,
//!     `ptir-pipelined-eos-rollback-spec` §4.3). This is the piece the mock cannot
//!     validate (fail-closed carrier finalize is device-resident).
//!
//! Where `cuda_runahead` proves the raw carrier on a plain prompt WITHOUT rollback,
//! this proves the carrier + the EOS-rollback on a CHAT generation — the pattern In
//! Gim wants every chat inferlet to follow (explicit pipelining on the low-level
//! API, no `collect_*` helper).
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_lowlevel_chat -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "low-level chat run-ahead + EOS-rollback verify: needs the 4090 + cuda + qwen-3-0.6b + delta's carrier"]
async fn lowlevel_chat_runahead_rollback_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[lowlevel-chat] booted, listen_addr={}", pie.listen_addr);

    // Build the low-level chat inferlet (raw-WIT explicit run-ahead + rollback).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "lowlevel-chat"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "lowlevel-chat wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/lowlevel_chat.wasm");
    let manifest = ws.join("lowlevel-chat/Pie.toml");

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
    eprintln!("[lowlevel-chat] program installed, launching explicit run-ahead chat decode…");

    // Default input "8" runs the forced-stop discard probe (no "no-rollback-probe").
    let mut proc = client
        .launch_process("lowlevel-chat@0.1.0".to_string(), "8".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[lowlevel-chat] returned: {json}");

    pie.shutdown().await;

    // Carrier token-identity: the explicit pipelined stream == the sequential one.
    anyhow::ensure!(
        json.contains("MATCH=true"),
        "explicit run-ahead stream diverged from the sequential reference: {json}"
    );
    // DEEP carrier token-identity: the depth-k pre-submission stream (the production
    // lever for co-batch residency + reduce-R) == the sequential one. Run with
    // `PIE_SCHED_MAX_IN_FLIGHT=4` to exercise true 4-in-flight residency (the
    // byte-identity holds at any cap; the residency is exercised at cap≥depth).
    anyhow::ensure!(
        json.contains("DEEP_MATCH=true"),
        "deep (depth-k) pre-submission carrier stream diverged from the sequential \
         reference: {json}"
    );
    // EOS-rollback: the forced-stop discard (drain_discard) completed cleanly and
    // the streams stayed equal — the ≤1 over-shot fire was discarded correctly.
    anyhow::ensure!(
        json.contains("ROLLBACK_OK=true"),
        "EOS-rollback (drain_discard depth-1 fire-past-EOS discard) failed under a \
         forced early stop: {json}"
    );
    // Non-degeneracy anchor — the pipelined stream must positively produce tokens
    // (n>0) and NOT be all-constant (the mock's degenerate failure mode). A real
    // chat generation of a non-trivial prompt yields a varied stream; a broken
    // output()/carrier returning zeros or a constant would be caught here.
    anyhow::ensure!(
        !json.contains("n=0"),
        "low-level chat produced no tokens — the explicit decode loop stalled: {json}"
    );
    anyhow::ensure!(
        !degenerate_constant_stream(&json),
        "low-level chat output is degenerate (all-constant) — likely a broken \
         output()/carrier (MATCH passed but both read a broken path): {json}"
    );
    Ok(())
}

/// True if the reported `pipe=[…]` stream is non-empty and every element is the
/// same value (the mock's degenerate constant-token failure mode). On the real
/// driver a genuine chat decode must vary.
fn degenerate_constant_stream(json: &str) -> bool {
    let Some(start) = json.find("pipe=[") else {
        return false;
    };
    let rest = &json[start + "pipe=[".len()..];
    let Some(end) = rest.find(']') else {
        return false;
    };
    let inner = &rest[..end];
    let vals: Vec<&str> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    vals.len() > 1 && vals.iter().all(|&v| v == vals[0])
}
