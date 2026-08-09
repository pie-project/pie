//! A sentence, generated on a Vulkan device, through everything.
//!
//! The engine's own tests prove the seam stages a real checkpoint and that
//! four greedy steps match a CPU reference. This proves the rest of the
//! stack: the composition root boots with `[driver] type = "vulkan"`, a wasm
//! inferlet is installed over the websocket a deployment is reached by, and
//! the `chat-completion` program -- prompt prefill, a device-carried PTIR
//! decode loop, an in-graph top-p sampler -- returns text.
//!
//! The gate is `Paris`, for the reason the CUDA twin uses it: a stack that
//! prefills without ATTENDING the prompt still returns fluent tokens, so the
//! only assertion worth making is one that a wrong answer fails.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_chat_completion_e2e -- --ignored --nocapture
//! ```
//!
//! # Why the temperature is zero
//!
//! Greedy, so the gate is exact rather than likely -- and because the top-p
//! arm of the shared PTIR interpreter panics on this path
//! (`crates/driver/src/op.rs`'s `PIVOT_THRESHOLD`, `CummassLe`: the payload
//! value arrives with no lanes and the index is taken unchecked). That is a
//! gap in the interpreter every host-side driver shares, not in this
//! backend's forward, so it is recorded here rather than worked around.
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn chat_completion_on_the_vulkan_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-chat] booted, listen_addr={}", pie.listen_addr);

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("chat-completion");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "chat-completion",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "chat-completion wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/chat_completion.wasm");
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
    eprintln!("[vulkan-chat] program installed, launching generation…");

    let input = serde_json::json!({
        "prompt": "The capital of France is",
        "system": "You are a helpful assistant. Answer concisely.",
        "max_tokens": 24,
        "temperature": 0.0,
        "top_p": 0.95,
    })
    .to_string();

    let mut proc = client
        .launch_process("chat-completion@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[vulkan-chat] returned: {out:?}");

    pie.shutdown().await;

    anyhow::ensure!(
        !out.trim().is_empty(),
        "empty continuation, so the prefill/decode plumbing broke: {out:?}"
    );
    anyhow::ensure!(
        out.to_lowercase().contains("paris"),
        "the continuation did not attend the prompt (expected `Paris`): {out:?}"
    );
    eprintln!("[vulkan-chat] GREEN — a Vulkan device answered: {out:?}");
    Ok(())
}
