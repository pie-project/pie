//! text-completion PTIR migration — DEVICE e2e on the 4090.
//!
//! Proves the `text-completion` inferlet (migrated off the classic
//! `forward-pass` onto `inferlet::ptir`: prompt prefill + device-carried decode
//! loop + in-graph top-p/temperature sampler) generates a COHERENT continuation
//! on the real cuda driver. This is the go-green for the classic-forward-pass
//! removal endgame (step 2/3): the default chat-generation path now rides PTIR.
//!
//!   PIE_COMPILER_LAUNCHER=env cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_text_completion_e2e -j6 -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "text-completion ptir device e2e: needs the 4090 + cuda + qwen-3-0.6b"]
async fn text_completion_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!(
        "[text-completion-e2e] booted, listen_addr={}",
        pie.listen_addr
    );

    // text-completion is a standalone cdylib crate (excluded from the workspace);
    // build it in place.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../inferlets/text-completion");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2"])
        .current_dir(&dir)
        .status()?
        .success();
    anyhow::ensure!(ok, "text-completion wasm build failed");
    let wasm = dir.join("target/wasm32-wasip2/debug/text_completion.wasm");
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
    eprintln!("[text-completion-e2e] program installed, launching generation…");

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
        .launch_process("text-completion@0.2.15".to_string(), input, true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[text-completion-e2e] returned: {out:?}");

    pie.shutdown().await;

    // Coherence gate: the migrated ptir path must attend the prompt and produce
    // a non-empty continuation. "Paris" is the unambiguous factual answer for a
    // prefill that actually attends "The capital of France is".
    anyhow::ensure!(
        !out.trim().is_empty(),
        "text-completion e2e: empty continuation (prefill/decode plumbing broke): {out:?}"
    );
    let lower = out.to_lowercase();
    anyhow::ensure!(
        lower.contains("paris"),
        "text-completion e2e: continuation did not attend the prompt (expected 'Paris'): {out:?}"
    );
    eprintln!("[text-completion-e2e] GREEN — ptir text-completion attended the prompt: {out:?}");
    Ok(())
}
