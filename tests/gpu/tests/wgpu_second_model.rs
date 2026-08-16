//! A second architecture, on the same driver, on the same device.
//!
//! Every other wgpu gate here serves Qwen3-0.6B, and so does every test in
//! `crates/driver-wgpu`. That is one geometry and one weight naming, so a
//! driver that had hard-coded either would pass all of them. This one serves
//! Qwen2.5-1.5B-Instruct instead, and it differs in the two ways that reach
//! this crate:
//!
//! * **Attention biases.** Qwen2 carries `q/k/v` biases; Qwen3 does not. The
//!   lowering binds them or it does not, per model, and nothing but a Qwen2
//!   forward pass distinguishes a driver that reads the bias from one that
//!   quietly drops it — a dropped bias still returns fluent tokens.
//! * **A different head geometry.** 12 query heads over 2 KV heads, against
//!   Qwen3's 16 over 8. The GQA fan-out is 6 rather than 2, and the page
//!   stride follows the KV head count, so every size this driver derives from
//!   the model rather than assumes is re-derived here.
//!
//! It is also 1.5B rather than 0.6B, which is the only reason the two are not
//! one gate: one boot serves one model.
//!
//! ```text
//! PIE_HOME=/tmp/piehome PIE_WGPU_MODEL_2=Qwen--Qwen2.5-1.5B-Instruct \
//!   cargo test -p pie-gpu-tests --features driver-wgpu \
//!   --test wgpu_second_model -- --ignored --nocapture
//! ```
//!
//! # The harness said `qwen3` and could not have said otherwise
//!
//! `common::wgpu_standalone_toml` hard-coded `[model] name = "qwen3"`, while
//! the Vulkan twin took the label as a parameter. So no wgpu gate could serve
//! a second architecture at all, and the one file whose entire subject is "a
//! different model" would have been the last place still saying `qwen3`. The
//! builder now takes the name; `boot_wgpu_named` is what this gate calls.
//!
//! The label is not a selector — the driver reads the architecture out of the
//! artifact — which is exactly why leaving it wrong would have been invisible.
//!
//! # The artifact
//!
//! `pie model build --backend wgpu` and `--backend vulkan` author the SAME
//! artifact: `src/ops/model/build.rs` maps `metal | vulkan | wgpu` to one
//! bind policy (in-place projections, MLX naming), and says so. So a `.zt`
//! built for either serves here, and this gate names a cache entry rather
//! than a file for the same reason the other wgpu gates do — this driver
//! quantizes through the load plan and reads what `$PIE_HOME` holds.
//!
//! ```text
//! pie model build <a Qwen2.5-1.5B-Instruct checkpoint> --quant int4 \
//!   --backend wgpu --out $PIE_HOME/models/Qwen--Qwen2.5-1.5B-Instruct.zt
//! ```
//!
//! # What it asserts, and why that and not more
//!
//! `Paris`, greedy, for the reason the Qwen3 gates use it: a stack that
//! prefills without ATTENDING the prompt still returns fluent tokens, so the
//! only assertion worth making is one a wrong answer fails. A numeric parity
//! check against a CPU reference would be stronger and it already exists, in
//! `crates/driver-wgpu/tests/serving.rs`, where a reference can be computed.
//! What this gate is for is the whole stack, booted, on a model it was not
//! written against.
//!
//! # That the bias claim above is true, measured
//!
//! Claiming a gate covers something is cheap, so this one was checked the way
//! `driver-vulkan`'s twin was: by breaking the thing it claims to cover.
//! `Shell::hold` was made to zero every tensor whose name contains `_bias`
//! before it reached the device, which is what a driver that bound the bias
//! and never read it would do.
//!
//! * This gate zeroed **84** tensors (28 layers x q/k/v) and stopped
//!   answering — not a wrong answer, an empty one.
//! * `wgpu_many_conversations`, under the SAME mutation, zeroed **zero**
//!   tensors and stayed green. Qwen3 has no projection biases, so no gate
//!   that serves it can fail this way however carefully it is written.
//!
//! That second line is the whole argument for this file. It is not the Qwen3
//! gate with a bigger model; it is the only gate here that fails when the
//! bias path breaks.

#![cfg(feature = "driver-wgpu")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// The cache entry this gate serves, and how to say otherwise.
///
/// A variable of its own rather than a second entry in `PIE_WGPU_MODEL`: that
/// one names the model every OTHER gate serves, and a gate proving a second
/// architecture cannot say that with the first's variable.
fn second_model() -> String {
    std::env::var("PIE_WGPU_MODEL_2")
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "Qwen--Qwen2.5-1.5B-Instruct".to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a WebGPU adapter and a SECOND model in the cache"]
async fn a_second_architecture_answers_on_the_wgpu_driver() -> Result<()> {
    common::init_trace();
    let model = second_model();
    // `qwen2` rather than `qwen3` as the `[model] name`. It is a label the
    // engine reports and hashes, not a selector — the driver reads the
    // architecture out of the artifact — but a gate whose whole subject is
    // "a different model" should not be the one place that says otherwise.
    let pie = common::boot_wgpu_named(&model, "qwen2", 256).await?;
    eprintln!("[wgpu-2nd] booted on {model}, listen_addr={}", pie.listen_addr);

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
    eprintln!("[wgpu-2nd] program installed, launching generation…");

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
    eprintln!("[wgpu-2nd] returned: {out:?}");

    pie.shutdown().await;

    anyhow::ensure!(
        !out.trim().is_empty(),
        "empty continuation, so the prefill/decode plumbing broke on this model: {out:?}"
    );
    anyhow::ensure!(
        out.to_lowercase().contains("paris"),
        "the continuation did not attend the prompt (expected `Paris`): {out:?}"
    );
    eprintln!("[wgpu-2nd] GREEN — a Qwen2 answered on the WebGPU driver: {out:?}");
    Ok(())
}
