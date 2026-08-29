//! **THE CHAT TURN, END TO END ON THE DEVICE.**
//!
//! The `chat-completion` inferlet is the default chat-generation path — prompt
//! prefill, then a device-carried decode loop with an in-graph top-p sampler,
//! all of it authored in `inferlet::ptir` — and this boots the standalone over
//! the real CUDA shell and asserts the continuation is coherent.
//!
//! It runs against `DECODE_ENVELOPE` and nothing wider, which is why it is a
//! gate rather than a blocked one: the only port the guest DECIDES is
//! `EmbedTokens` (the sampled token, fed back through the channel the `embed`
//! port reads), and the `w_slot` / `w_off` / `page_indptr` the epilogue also
//! carries are pure arithmetic over the KV length that `pareval` folds
//! host-side on every fire (palo build log 18). Its neighbours in this
//! directory that put an `attn_mask` on the device are refused for exactly the
//! port this one does not need.
//!
//! Booted through `common::boot_cuda`, which resolves the shipping
//! `qwen35-d0.8b-bf16-kv-bf16` snapshot; the `Qwen/Qwen3-0.6B` this file used
//! to name is a checkpoint no `::model::qwen_3::IMPORTS` row can claim.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_chat_completion_e2e -- --ignored --nocapture
//! ```

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the chat-turn gate: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn chat_completion_on_real_engine() -> Result<()> {
    common::init_trace();
    let pie = common::boot_cuda().await?;
    eprintln!(
        "[chat-completion-e2e] booted, listen_addr={}",
        pie.listen_addr
    );

    // chat-completion is part of the curated inferlet test workspace.
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
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
    eprintln!("[chat-completion-e2e] program installed, launching generation…");

    // A raw factual completion with a low temperature so a WORKING prefill+decode
    // has an unambiguous coherent continuation (the prompt must be attended).
    let input = serde_json::json!({
        "prompt": "The capital of France is",
        "system": "You are a helpful assistant. Answer concisely.",
        "max_tokens": 24,
        "temperature": 0.1,
        "top_p": 0.95,
    })
    .to_string();

    let mut proc = client
        .launch_process("chat-completion@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[chat-completion-e2e] returned: {out:?}");

    pie.shutdown().await;

    // Coherence gate: the migrated ptir path must attend the prompt and produce
    // a non-empty continuation. "Paris" is the unambiguous factual answer for a
    // prefill that actually attends "The capital of France is".
    anyhow::ensure!(
        !out.trim().is_empty(),
        "chat-completion e2e: empty continuation (prefill/decode plumbing broke): {out:?}"
    );
    let lower = out.to_lowercase();
    anyhow::ensure!(
        lower.contains("paris"),
        "chat-completion e2e: continuation did not attend the prompt (expected 'Paris'): {out:?}"
    );
    eprintln!("[chat-completion-e2e] GREEN — PTIR chat completion attended the prompt: {out:?}");
    Ok(())
}
