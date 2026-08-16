//! A second architecture, on the same driver, on the same device.
//!
//! Every other Vulkan gate here serves Qwen3-0.6B. That is one geometry and
//! one weight naming, so a driver that had hard-coded either would pass all of
//! them. This one serves Qwen2.5-1.5B-Instruct instead, and it differs in the
//! two ways that reach this crate:
//!
//! * **Attention biases.** Qwen2 carries `q/k/v` biases; Qwen3 does not. The
//!   lowering binds them or it does not, per model, and nothing but a Qwen2
//!   forward pass distinguishes a driver that reads the bias from one that
//!   quietly drops it -- a dropped bias still returns fluent tokens.
//! * **A different head geometry.** 12 query heads over 2 KV heads at 128
//!   wide, against Qwen3's 16 over 8. The GQA fan-out is 6 rather than 2, and
//!   the page stride follows the KV head count, so every size this driver
//!   derives from the model rather than assumes is re-derived here.
//!
//! It is also 1.5B rather than 0.6B, which is the only reason the two are not
//! merged into one gate: they are separate boots because the artifacts are
//! separate, and one boot per process.
//!
//! ```text
//! PIE_VULKAN_ARTIFACT=/tmp/q4full.zt:/tmp/q25full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_second_model -- --ignored --nocapture
//! ```
//!
//! The SECOND entry of the list is the one served here; see
//! `common::boot_vulkan_nth`.
//!
//! # Building the artifact this needs
//!
//! From a pre-quantized MLX Qwen2.5-1.5B-Instruct checkpoint (`config.json` +
//! `model.safetensors`), plus the tokenizer the runtime parses at boot
//! unconditionally -- an MLX checkpoint usually carries no tokenizer, so it
//! is fetched and staged beside the weights:
//!
//! ```text
//! mkdir -p /tmp/q25t && ln -s /tmp/q25/model.safetensors /tmp/q25t/ \
//!   && cp /tmp/q25/config.json /tmp/q25t/
//! for f in tokenizer.json tokenizer_config.json; do curl -sSL -o /tmp/q25t/$f \
//!   https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/$f; done
//! pie model build /tmp/q25t --quant int4 --backend vulkan --out /tmp/q25full.zt
//! ```
//!
//! The symlink rather than a copy is deliberate: the weights are the large
//! half and `pie model build` only reads them.
//!
//! # What it asserts, and why that and not more
//!
//! `Paris`, greedy, for the reason the Qwen3 twin uses it: a stack that
//! prefills without ATTENDING the prompt still returns fluent tokens, so the
//! only assertion worth making is one a wrong answer fails. A numeric parity
//! check against a CPU reference would be stronger, and it belongs in the
//! driver's own device tests where a reference exists -- what this gate is
//! for is the whole stack, booted, on a model it was not written against.
//!
//! The prompt is deliberately NOT the Qwen3 gate's prompt verbatim: this
//! model has no `<think>` preamble, so a 24-token budget that the other gate
//! needs for its preamble is more than enough here.
//!
//! # That the bias claim above is true, measured
//!
//! Claiming a gate covers something is cheap, so this one was checked by
//! breaking the thing it claims to cover: `Weights::hold` was made to zero
//! every tensor whose name contains `_bias` before it reached the device,
//! which is what a driver that bound the bias but never read it would do.
//!
//! * This gate zeroed **84** tensors (28 layers x q/k/v) and failed with
//!   `raudiectiecttheses.setPositionoler...` -- not a wrong answer, no answer
//!   at all.
//! * `vulkan_chat_completion_e2e`, under the SAME mutation, zeroed **zero**
//!   tensors and stayed green. Qwen3 has no projection biases, so no gate
//!   that serves it can fail this way however carefully it is written.
//!
//! That second line is the whole argument for this file: it is not a
//! duplicate of the Qwen3 gate with a bigger model, it is the only gate here
//! that fails when the bias path breaks.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a SECOND built artifact"]
async fn a_second_architecture_answers_on_the_vulkan_driver() -> Result<()> {
    common::init_trace();
    // `qwen2` rather than `qwen3` as the `[model] name`. It is a label the
    // engine reports and hashes, not a selector -- the driver reads the
    // architecture out of the artifact -- but a gate whose whole subject is
    // "a different model" should not be the one place that says otherwise.
    let pie = common::boot_vulkan_nth(1, "qwen2", 256).await?;
    eprintln!("[vulkan-2nd] booted, listen_addr={}", pie.listen_addr);

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
    eprintln!("[vulkan-2nd] program installed, launching generation…");

    let input = serde_json::json!({
        "prompt": "What is the capital city of France? Answer with the city name only.",
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
    eprintln!("[vulkan-2nd] returned: {out:?}");

    pie.shutdown().await;

    anyhow::ensure!(
        !out.trim().is_empty(),
        "empty continuation, so the prefill/decode plumbing broke on this model: {out:?}"
    );
    anyhow::ensure!(
        out.to_lowercase().contains("paris"),
        "the continuation did not attend the prompt (expected `Paris`): {out:?}"
    );
    eprintln!("[vulkan-2nd] GREEN — a Qwen2 answered on the Vulkan driver: {out:?}");
    Ok(())
}
