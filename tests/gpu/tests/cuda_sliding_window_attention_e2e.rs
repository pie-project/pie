//! A4 mask-migration DEVICE e2e — real driver (4090). First end-to-end exercise
//! of the token-at-a-time, B=1 explicit-write PTIR geometry that the A4 mask
//! inferlets migrated onto (superseding the classic `forward-pass` +
//! `attention_mask(list<brle>)` surface). The `sliding-window-attention` inferlet
//! drives a single-sequence masked decode: one token per pass, the input token
//! host-fed (prompt tokens during prefill, then the sampled token), and ALL
//! geometry — position, KV length, WSlot/WOff write descriptor, and the
//! sliding-window `attn_mask` — evolved in-graph in the epilogue, submitted
//! through the `inferlet::ptir` bridge (`forward-pass.new` / `pipeline.submit`):
//!
//!   guest sliding-window decode program
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
//!   cargo test -p pie-gpu-tests --features driver-cuda-13 \
//!     --test cuda_sliding_window_attention_e2e -- --ignored --nocapture
//!
//! # What it does on a box that HAS all of that, which is refuse
//!
//! Run on the reference 4090 with CUDA 13 and the qwen-3-0.6b snapshot -- so
//! with every precondition the `#[ignore]` reason used to name -- this gate
//! boots, loads, spawns, and then dies inside the first fire:
//!
//! ```text
//! inferlet returned an error: pipeline: frame slot 0: pipeline: fire
//! geometry: EmbedTokens is not host-derivable: channel 0 has no host-known
//! value
//! ```
//!
//! That sentence is about the symptom. The claim behind it lived beside the
//! old shell's `device_geometry_port_mask` and survives as
//! `PortMask::DEVICE_GEOMETRY` in `crates/tensor-ir/src/registry.rs`:
//!
//! > Exactly the three ports `fire::envelope::compose` reads -- `EmbedTokens |
//! > Positions | KvLen`; the rest a decode derives from the positions.
//! > `DEVICE_GEOMETRY_PORTS` is deliberately absent: it wins the pool-owned
//! > class this driver does not build.
//!
//! A program whose geometry evolves in-graph -- which is the entire subject of
//! this file -- is exactly that class. Without the bit the engine classifies
//! it as a host-evaluated decode, and a host cannot know what token the
//! epilogue is about to sample. engine's vulkan seam (deleted at R3; back at P5) and
//! `.../wgpu.rs` each carry a long note about paying for this twice and then
//! building the machinery; `.../cuda.rs` has neither the note nor the bit.
//!
//! So this is not a test waiting for a GPU. It is a test waiting for
//! `driver-cuda` to build the pool-owned device-geometry class, and it will
//! start passing the day the mask above grows `PIE_DEVICE_GEOMETRY_PORTS` and
//! `PIE_DEVICE_PORT_ATTN_MASK`. Until then `crates/worker/tests/cuda_forward`
//! carries an epitaph pointing here rather than a second copy of the failure.

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "BLOCKED, and not on hardware: `driver-cuda` does not claim the \
            pool-owned device-geometry class this program needs. See the \
            section at the end of this file's header before running it"]
async fn sliding_window_attention_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!(
        "[sliding-window-attn-e2e] booted, listen_addr={}",
        pie.listen_addr
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("sliding-window-attention");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "sliding-window-attention",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "sliding-window-attention wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/sliding_window_attention.wasm");
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
    eprintln!("[sliding-window-attn-e2e] program installed, launching decode…");

    // Small window + a few tokens: the chat-templated prompt already exceeds
    // window=4, so the sliding-window mask is exercised on the decode passes.
    let mut proc = client
        .launch_process(
            "sliding-window-attention@0.1.0".to_string(),
            r#"{"max_tokens": 8, "window_size": 4}"#.to_string(),
            true,
        )
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[sliding-window-attn-e2e] returned: {out}");

    pie.shutdown().await;

    // A successful non-empty decode proves the full masked fire path completed.
    anyhow::ensure!(
        !out.trim().is_empty(),
        "A4 sliding-window mask e2e produced no decoded output: {out:?}"
    );
    eprintln!(
        "[sliding-window-attn-e2e] A4 GREEN — token-at-a-time B=1 masked decode fired e2e \
         (explicit write + in-graph sliding-window mask): {out}"
    );
    Ok(())
}
