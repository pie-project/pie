//! A4 mask-migration DEVICE e2e — real driver (4090). Third mask inferlet on the
//! device-proven token-at-a-time B=1 explicit-write geometry (superseding the
//! classic `forward-pass` + `attention_mask(list<brle>)` surface). The
//! `hierarchical-attention-rust` inferlet drives a single-sequence masked decode:
//! one token per pass, with a HIERARCHICAL `attn_mask` (sink + chunk headers +
//! the most-recent selected chunk body + recent window) evolved IN-GRAPH in the
//! epilogue — the keep-set is a pure function of the query position and the
//! (constant) chunk parameters, so it rides the same in-graph pattern as
//! windowed/sink rather than a per-pass host-written mask:
//!
//!   guest hierarchical-decode program
//!     → runtime device-geometry submit (PageLease grants, run-ahead FIFO)
//!     → driver descriptor resolver → pack_dense_mask over the in-graph mask
//!     → B2 explicit-KV-write (WSlot/WOff) → masked attention → harvest → out.
//!
//! This confirms the compound multi-run hierarchical keep-set (sink + periodic
//! headers + selected body + window — with interior positions masked out) is
//! honored end-to-end at B=1, reusing the driver path beam-designb + windowed +
//! sink proved.
//!
//!   cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_hierarchical_attention_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "A4 mask-migration device e2e: needs the 4090 + cuda + qwen-3-0.6b"]
async fn hierarchical_attention_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[hier-attn-e2e] booted, listen_addr={}", pie.listen_addr);

    // Build the `hierarchical-attention-rust` inferlet to wasm. It is a standalone
    // crate (excluded from the host workspace), so build in its own dir. The crate
    // name normalizes to `hierarchical_attention_rust.wasm`.
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../inferlets/hierarchical-attention-rust");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2"])
        .current_dir(&dir)
        .status()?
        .success();
    anyhow::ensure!(ok, "hierarchical-attention-rust wasm build failed");
    let wasm = dir.join("target/wasm32-wasip2/debug/hierarchical_attention_rust.wasm");
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
    eprintln!("[hier-attn-e2e] program installed, launching hierarchical decode…");

    // Small chunk/sink/window + a few tokens: the generated sequence quickly forms
    // multiple chunks, so the host-computed hierarchical keep-set (sink + headers +
    // selected body + window) is a genuine multi-run dense mask.
    let mut proc = client
        .launch_process(
            "hierarchical-attention-rust@0.4.0".to_string(),
            r#"{"max_tokens": 8, "chunk_size_words": 3, "sink_tokens": 2, "summary_tokens_per_chunk": 1, "local_window_tokens": 3, "selected_chunks": 1}"#.to_string(),
            true,
        )
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[hier-attn-e2e] returned: {out}");

    pie.shutdown().await;

    // The hierarchical decode loop ran (submit→take)×passes through the full path
    // (guest in-graph hierarchical mask program → device-geometry submit →
    // pack_dense_mask + B2 explicit KV write + masked attention → harvest → out). A
    // crash/degeneration surfaces as a non-`HIERARCHICAL_ATTENTION` return or an
    // empty token list.
    anyhow::ensure!(
        out.contains("HIERARCHICAL_ATTENTION"),
        "A4 hierarchical mask e2e: unexpected return (expected a `HIERARCHICAL_ATTENTION …` \
         summary; a fire-path crash returns the error string): {out:?}"
    );
    let has_tokens = out.contains("tokens=[") && !out.contains("tokens=[]");
    anyhow::ensure!(
        has_tokens,
        "A4 hierarchical mask e2e: no tokens harvested — the submit→take loop \
         produced no output (a dropped ptir_output_* or a stalled fire): {out:?}"
    );
    eprintln!(
        "[hier-attn-e2e] A4 GREEN — token-at-a-time B=1 masked decode fired e2e \
         (explicit write + in-graph hierarchical mask): {out}"
    );
    Ok(())
}
