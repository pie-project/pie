//! **B3, THE GATE**: the serving door, end to end.
//!
//! `pie serve`'s own composition root boots the engine over the CUDA shell, a
//! wasm inferlet is installed through the client edge a deployment is reached
//! by, and tokens come back. Everything between is real: `bootstrap::init` →
//! `derive_standalone` → `run_standalone` (embedded controller + gateway +
//! worker over loopback) → `create_driver_backend` → `engine::driver::load`
//! → `driver_cuda::Shell::load` → wasmtime component → the inferlet host glue
//! → `pipeline::fire` → `Driver::fire`.
//!
//! In-process rather than a spawned `pie serve`, because the boot is the same
//! function `src/main.rs` calls and a socket in between proves nothing this does
//! not — and because one boot per process is the runtime's rule (the driver
//! grabs the device, `auth` panics on a second boot), so a spawn would buy a
//! second process to say the same thing.
//!
//! The assertion is a FACT, not fluency: greedy decoding of "The capital of
//! France is" begins " Paris", the same continuation
//! `engine/tests/cuda_boot_smoke` pins one fire below this and
//! `driver-cuda/tests/serve_smoke` pins one below that. Three paths to one
//! device have to agree about it.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features driver-cuda-13 \
//!   --test cuda_serve_round_trip -- --ignored --nocapture
//! ```

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// The prompt, and why it is this one: the answer is a single well-known
/// token, so a continuation that is merely fluent still fails.
const PROMPT: &str = "The capital of France is";

/// What greedy decoding answers with. The first token is the boot smoke's
/// pinned one; the rest is what this model then says.
const EXPECTED_PREFIX: &str = " Paris";

/// Sixteen tokens, not one: the prompt is five tokens and a KV page is
/// sixteen, so the generation crosses a page boundary. A decode loop whose
/// page CSR is frozen at bind answers the first fifteen tokens correctly and
/// falls apart at the sixteenth.
const MAX_TOKENS: u32 = 16;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the serving gate: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn a_text_completion_inferlet_round_trips_through_the_serving_door() -> Result<()> {
    common::init_trace();
    let pie = common::boot_serving().await?;
    eprintln!("[b3] booted, listen_addr={}", pie.listen_addr);

    // The fixture is built here rather than committed as a `.wasm`, so the
    // gate cannot pass against a stale binary of a guest contract that moved.
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "text-completion"])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "text-completion wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/text_completion.wasm");
    let manifest = workspace.join("text-completion/Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());

    // The gateway serves the multi-turn client WebSocket at `/v1/ws`, gated on
    // the `x-pie-identity` trust-edge header. A standalone has no edge proxy,
    // so the header is injected here.
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
    eprintln!("[b3] program installed, launching…");

    let input = serde_json::json!({
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
    })
    .to_string();

    let mut proc = client
        .launch_process("text-completion@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;

    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    let text = parsed["text"].as_str().unwrap_or_default();
    let count = parsed["count"].as_u64().unwrap_or_default();
    eprintln!("[b3] {PROMPT:?} -> {text:?} ({count} tokens)");

    assert_eq!(
        count, MAX_TOKENS as u64,
        "the inferlet returned {count} of {MAX_TOKENS} tokens"
    );
    assert!(!text.is_empty(), "the completion is empty");
    assert!(
        text.starts_with(EXPECTED_PREFIX),
        "greedy continuation of {PROMPT:?} was {text:?}, and the same \
         checkpoint one layer down answers {EXPECTED_PREFIX:?}"
    );

    pie.shutdown().await;
    Ok(())
}
