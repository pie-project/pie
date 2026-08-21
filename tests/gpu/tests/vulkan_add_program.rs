//! DIAGNOSTIC (echo): cheap repro of the chunked-`add_program` session-bridge
//! deadlock. Boots the **Vulkan** driver through the same `run_standalone` +
//! client-WS edge as the 4090 harness, then uploads the ~12 MB `generate` wasm
//! (≈50 × 256 KiB chunks) and asserts `add_program` returns within a timeout.
//! A hang here reproduces the gateway/worker turn-model bug without touching
//! CUDA — seconds per iteration.
//!
//! # Why Vulkan, for a test that does not care about the driver
//!
//! It does not care, and that is the point: what it exercises is the
//! driver-AGNOSTIC client edge, so it only ever wanted the CHEAPEST boot on a
//! machine with no CUDA. That used to be the dummy driver, which fabricated
//! weights and loaded nothing; the dummy driver is deleted (see the note where
//! `boot_dummy` stood in `common/mod.rs`) and there is no driverless boot left.
//! Vulkan is what "cheapest boot without CUDA" means now.
//!
//! The price is an artifact: this wants `PIE_VULKAN_ARTIFACT` where it used to
//! want nothing, and it loads a real 0.6B model where it used to load none.
//! Still seconds rather than the 4090 harness's minutes, which is the property
//! the diagnostic is for.
//!
//! Run:
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!     --test vulkan_add_program -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result};
use client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "diagnostic: needs PIE_VULKAN_ARTIFACT + a built generate.wasm"]
async fn add_program_completes_over_the_client_edge() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[diag] booted, listen_addr={}", pie.listen_addr);

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "generate"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "generate wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/generate.wasm");
    let manifest = ws.join("generate/Pie.toml");

    let client = tokio::time::timeout(
        Duration::from_secs(20),
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user"),
    )
    .await
    .context("connect TIMED OUT")?
    .context("connect")?;
    eprintln!("[diag] connected ✓");

    tokio::time::timeout(
        Duration::from_secs(20),
        client.authenticate("test-user", &None),
    )
    .await
    .context("authenticate TIMED OUT")?
    .context("auth")?;
    eprintln!(
        "[diag] authed ✓; uploading {} bytes…",
        std::fs::metadata(&wasm)?.len()
    );

    tokio::time::timeout(
        Duration::from_secs(60),
        client.add_program(&wasm, &manifest, true),
    )
    .await
    .context("add_program TIMED OUT (repro: chunked-upload bridge deadlock)")?
    .context("add_program failed")?;
    eprintln!("[diag] program installed ✓");

    pie.shutdown().await;
    Ok(())
}
