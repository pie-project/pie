//! A4 mask-migration DEVICE e2e — real driver. First end-to-end exercise
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
//! Run on the reference device with CUDA 13 and the shipping snapshot -- so
//! with every precondition the `#[ignore]` reason used to name -- this gate
//! boots, loads, spawns, and then dies inside the first fire:
//!
//! ```text
//! inferlet returned an error: pipeline: frame slot 0: pipeline: fire
//! geometry: EmbedTokens is not host-derivable: channel 0 has no host-known
//! value
//! ```
//!
//! That sentence is about the symptom, and it names the wrong port because
//! `EmbedTokens` is the first one the fire path reaches. The claim behind it
//! is `crates/tensor-ir/src/registry.rs`, where a driver's served set is one
//! of two constants and neither holds the port this file is about:
//!
//! > `DECODE_ENVELOPE`: the three ports a decode envelope resolves -- the
//! > token ids, their positions, and each request's readable KV extent.
//! >
//! > `DEVICE_GEOMETRY`: the decode envelope plus the page table, the row
//! > split, and the adapter routing.
//!
//! `Port::AttnMask` is in NEITHER. `driver-cuda::api` answers
//! `GeometryClass::DecodeEnvelope` (palo build log 18: a decode-envelope
//! lane's page table is the ENGINE's, so `bind_instance` refuses the wider
//! class by name through `Capabilities::admits`), and a mask this guest
//! re-derives in its epilogue and `put`s every fire is therefore a value the
//! host is asked to know and cannot. The engine classifies the program as a
//! host-evaluated decode and refuses at the first fire rather than running on
//! a guess.
//!
//! So this is not a test waiting for a GPU. It is a test waiting for a shell
//! that resolves `Port::AttnMask` on the device, which is a `driver-cuda`
//! `program/ports.rs` wave: build log 18 built that read for `EmbedTokens`
//! alone and said which four of `DEVICE_GEOMETRY` it deliberately did not
//! claim. `cuda_attention_sink_e2e` in this directory is blocked on the same
//! port with a sink+window mask instead of a sliding one.

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "BLOCKED, and not on hardware: `driver-cuda` advertises \
            `GeometryClass::DecodeEnvelope`, whose port set is `EmbedTokens | \
            Positions | KvLen`; `Port::AttnMask` is in no `PortMask` any class \
            denotes, so the window this guest puts on every fire is a port no \
            shell in this tree resolves. See the section at the end of this \
            file's header before running it"]
async fn sliding_window_attention_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_cuda().await?;
    eprintln!(
        "[sliding-window-attn-e2e] booted, listen_addr={}",
        pie.listen_addr
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
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
