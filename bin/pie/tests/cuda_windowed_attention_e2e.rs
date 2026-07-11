//! A4 mask-migration DEVICE e2e — real driver (4090). First end-to-end exercise
//! of the token-at-a-time, B=1 explicit-write PTIR geometry that the A4 mask
//! inferlets migrated onto (superseding the classic `forward-pass` +
//! `attention_mask(list<brle>)` surface). The `windowed-attention` inferlet
//! drives a single-sequence masked decode: one token per pass, the input token
//! host-fed (prompt tokens during prefill, then the sampled token), and ALL
//! geometry — position, KV length, WSlot/WOff write descriptor, and the
//! sliding-window `attn_mask` — evolved in-graph in the epilogue, submitted
//! through the `inferlet::ptir` bridge (`forward-pass.new` / `pipeline.submit`):
//!
//!   guest windowed-decode program
//!     → runtime device-geometry submit (PageLease grants, run-ahead FIFO)
//!     → driver descriptor resolver (`resolve_descriptors` → `FireGeometry`)
//!     → B2 explicit-KV-write (`launch_write_kv_explicit_bf16` honoring
//!       WSlot/WOff — the single sequence's append cell written in place)
//!     → attention under the packed dense sliding-window mask
//!     → harvest → the guest's `out` take.
//!
//! This is the INTEGRATION gate for the A4 token-at-a-time mask geometry: it
//! proves the B=1 masked decode flows guest → device-geometry submit → the
//! driver's explicit write + masked attention → harvest, end to end on real
//! logits, without crashing or degenerating. It reuses the SAME driver path the
//! beam-designb e2e proved (explicit write + dense mask), at B=1 with an
//! in-graph sliding window instead of the beam ancestry mask.
//!
//!   cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_windowed_attention_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "A4 mask-migration device e2e: needs the 4090 + cuda + qwen-3-0.6b"]
async fn windowed_attention_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!(
        "[windowed-attn-e2e] booted, listen_addr={}",
        pie.listen_addr
    );

    // Build the `windowed-attention` inferlet to wasm. It is a standalone crate
    // (excluded from the host workspace), so build in its own dir. The crate name
    // normalizes to `windowed_attention.wasm`.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../inferlets/windowed-attention");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2"])
        .current_dir(&dir)
        .status()?
        .success();
    anyhow::ensure!(ok, "windowed-attention wasm build failed");
    let wasm = dir.join("target/wasm32-wasip2/debug/windowed_attention.wasm");
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
    eprintln!("[windowed-attn-e2e] program installed, launching windowed decode…");

    // Small window + a few tokens: the chat-templated prompt already exceeds
    // window=4, so the sliding-window mask is exercised on the decode passes.
    let mut proc = client
        .launch_process(
            "windowed-attention@0.4.0".to_string(),
            r#"{"max_tokens": 8, "window_size": 4}"#.to_string(),
            true,
        )
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[windowed-attn-e2e] returned: {out}");

    pie.shutdown().await;

    // The windowed decode loop ran (feed→submit→take)×passes through the full
    // path (guest in-graph mask program → device-geometry submit → B2 explicit KV
    // write + masked attention → harvest → out). A crash/degeneration surfaces as
    // a non-`WINDOWED_ATTENTION` return or an empty token list.
    anyhow::ensure!(
        out.contains("WINDOWED_ATTENTION"),
        "A4 windowed mask e2e: unexpected return (expected a `WINDOWED_ATTENTION …` \
         summary; a fire-path crash returns the error string): {out:?}"
    );
    let has_tokens = out.contains("tokens=[") && !out.contains("tokens=[]");
    anyhow::ensure!(
        has_tokens,
        "A4 windowed mask e2e: no tokens harvested — the feed→submit→take loop \
         produced no output (a dropped ptir_output_* or a stalled fire): {out:?}"
    );
    eprintln!(
        "[windowed-attn-e2e] A4 GREEN — token-at-a-time B=1 masked decode fired e2e \
         (explicit write + in-graph sliding-window mask): {out}"
    );
    Ok(())
}
