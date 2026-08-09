//! A completion the model SAMPLES, rather than one it takes the argmax of.
//!
//! `vulkan_chat_completion_e2e` runs at temperature zero, which routes around
//! most of the sampler: greedy needs no distribution, no nucleus and no
//! threshold. Every other temperature runs `PIVOT_THRESHOLD`, and that op used
//! to end the process.
//!
//! # What it caught
//!
//! `pivot_threshold` has two operands and only one of them rides in `args`.
//! The other -- the `k` of a rank cut, the `p` of a nucleus, the floor of a
//! probability cut -- rides in `pred_payload`, because the device path binds it
//! into a fixed slot. The host interpreter's stage index is built by walking
//! each op's `args`, so that operand was never marked, never evaluated, and
//! kept the default cell: zero lanes. `op::eval_op` then read lane zero of it.
//!
//! ```text
//! thread 'pie-driver-0' panicked at crates/driver/src/op.rs:843:35:
//! index out of bounds: the len is 0 but the index is 0
//! ```
//!
//! And the panic was not the worst of it. It killed the driver thread with a
//! frame in flight, so nothing ever completed the launch and the engine sat in
//! its stall loop -- `driver 0 stalled for 7030.132606596s (no progress, work
//! queued or in flight)` -- until the process was killed by hand. A request at
//! any temperature but zero hung the server.
//!
//! The fix is in `crates/driver/src/plan.rs`: the walk marks `pred_payload`
//! too. It is shared code, so it was never this backend's bug -- but it is the
//! reason this backend's own e2e is pinned to greedy, and nothing else in the
//! tree fires a sampled completion at a real device.
//!
//! # What it asserts
//!
//! Little, on purpose. A sampled continuation is a distribution, not a
//! sentence, so this asks only for what sampling guarantees: the call RETURNS,
//! within the test's patience, having produced non-empty text. The failure it
//! is built to catch is not a wrong word -- it is a panic, or a hang, and both
//! of those it catches by finishing at all.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_sampled_completion -- --ignored --nocapture
//! ```
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
async fn a_sampled_completion_returns_instead_of_hanging() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-sampled] booted, listen_addr={}", pie.listen_addr);

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

    // Two temperatures, because they take different arms of the same op: 0.7
    // with a nucleus below one is `CummassLe`, and the higher one with a wider
    // nucleus keeps more of the tail. Both used to panic identically.
    for (temperature, top_p) in [(0.7, 0.95), (1.0, 0.99)] {
        let input = serde_json::json!({
            "prompt": "What is the capital of France? Answer with one word.",
            "system": "You are a helpful assistant.",
            "max_tokens": 24,
            "temperature": temperature,
            "top_p": top_p,
        })
        .to_string();
        let mut proc = client
            .launch_process("chat-completion@0.1.0".to_string(), input, true)
            .await
            .context("launch")?;
        // A hang is the failure this exists for, so it is bounded rather than
        // awaited: before the fix this returned nothing, ever, and the run had
        // to be killed by hand.
        let out = tokio::time::timeout(std::time::Duration::from_secs(180), proc.wait_for_return())
            .await
            .context("a sampled completion did not return within 180s")?
            .context("wait_for_return")?;
        eprintln!("[vulkan-sampled] t={temperature} top_p={top_p} -> {out:?}");
        anyhow::ensure!(
            !out.trim().is_empty(),
            "a sampled completion returned no text at temperature {temperature}"
        );
    }

    pie.shutdown().await;
    eprintln!("[vulkan-sampled] GREEN — sampling returns");
    Ok(())
}
