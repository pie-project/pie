//! A4 mask-migration DEVICE e2e — real driver (4090). Second exercise of the
//! token-at-a-time, B=1 explicit-write PTIR geometry the A4 mask inferlets
//! migrated onto (superseding the classic `forward-pass` +
//! `attention_mask(list<brle>)` surface). The `attention-sink` inferlet drives a
//! single-sequence masked decode: one token per pass, the input token DEVICE
//! loop-carried (argmax fed back), and ALL geometry — position, KV length,
//! WSlot/WOff write descriptor, and the sink+window `attn_mask` — evolved in-graph
//! in the epilogue, submitted through the `inferlet::ptir` bridge
//! (`forward-pass.new` / `pipeline.submit`):
//!
//!   guest sink-decode program
//!     → runtime device-geometry submit (PageLease grants, run-ahead FIFO)
//!     → driver descriptor resolver (`resolve_descriptors` → `FireGeometry`)
//!     → B2 explicit-KV-write (`launch_write_kv_explicit_bf16` honoring
//!       WSlot/WOff — the single sequence's append cell written in place)
//!     → attention under the packed dense sink+window mask
//!     → harvest → the guest's `out` take.
//!
//! This is the INTEGRATION gate for the sink+window variant of the A4 mask
//! geometry: the mask admits KV position j iff (j <= p) AND (j < sink OR
//! j + window > p) — the initial sink tokens plus the most recent `window`. It
//! reuses the SAME driver path the beam-designb + sliding-window e2es proved,
//! at B=1 with a compound sink+window mask.
//!
//!   cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_attention_sink_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "A4 mask-migration device e2e: needs the 4090 + cuda + qwen-3-0.6b"]
async fn attention_sink_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!(
        "[attention-sink-e2e] booted, listen_addr={}",
        pie.listen_addr
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("attention-sink");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "attention-sink"])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "attention-sink wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/attention_sink.wasm");
    let manifest = dir.join("Pie.toml");
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
    eprintln!("[attention-sink-e2e] program installed, launching sink decode…");

    // Small sink + window + a few tokens: the decode passes quickly exceed
    // sink+window, so the compound sink+window mask is exercised.
    let mut proc = client
        .launch_process(
            "attention-sink@0.1.0".to_string(),
            r#"{"max_tokens": 8, "sink_size": 2, "window_size": 4}"#.to_string(),
            true,
        )
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[attention-sink-e2e] returned: {out}");

    pie.shutdown().await;

    // A successful non-empty decode proves the full masked fire path completed.
    anyhow::ensure!(
        !out.trim().is_empty(),
        "A4 attention-sink mask e2e produced no decoded output: {out:?}"
    );
    eprintln!(
        "[attention-sink-e2e] A4 GREEN — token-at-a-time B=1 masked decode fired e2e \
         (explicit write + in-graph sink+window mask): {out}"
    );
    Ok(())
}
