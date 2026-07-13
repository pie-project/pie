//! Standalone composition-root boot smoke — the Phase-2 regression guard (build/packaging).
//!
//! The `bin/pie` analogue of the gateway M3 smoke: boots the **embedded controller + gateway +
//! worker** in one process over loopback via the composition seam `pie_bin::run_standalone`
//! (= `controller::embed` → `EmbeddedControl` → `gateway::bind(worker_listen :0)` →
//! `worker::run_with(.., EmbeddedControl, [gw.worker_addr])` → `gw.serve()`, binding ephemeral),
//! then proves the planes co-reside + the real client path round-trips a ping, a direct-FFI
//! inferlet, and a text completion through the dummy driver.
//!
//! **One boot per test process.** The runtime owns process-global singletons — `auth` panics
//! "Service already spawned" on a 2nd boot (`runtime/src/auth.rs:31`) and the dummy driver grabs a
//! fixed POSIX shmem (`/pie_shmem_g0`) — so the gate is a *single* boot-once test that runs both
//! the Tier-1 plane/addr checks and the ping-through-ingress check sequentially. (The same
//! constraint applies to the `run` follow-on: one standalone per process.)
//!
//! Fixture: `tests/fixtures/smoke-model-ascii/tokenizer.json` — a real **128-token byte-level-BPE** tokenizer
//! (charlie's pure-stdlib generator replicating `model/tokenizer/bpe.rs build_byte_to_unicode` exactly;
//! `model.type=BPE`, `pre_tokenizer.type=ByteLevel`, empty `merges` → each ASCII byte = 1 token). **Boot-validated**
//! (booted `bin/worker` → exit 0). The runtime parses the tokenizer at boot unconditionally
//! (`model::register` → `Tokenizer::from_file`), so it must be valid — this is. The direct-channel
//! and chat-completion inferlets use the fixture's single-byte token range and verify the actual
//! dummy-driver PTIR path.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::{Context, Result};
use pie_bin::derive::derive_standalone;
use pie_bin::{Mode, run_standalone};
use pie_client::client::Client;

/// The one standalone TOML (`[controller]`/`[gateway]`/`[worker]` sections); `derive_standalone`
/// splits + hands each section to its role lib's `Config::parse`. `Mode::Local` pins the client edge
/// to loopback but keeps the *configured port* (so `pie local` has a predictable address), so the test
/// must itself request an ephemeral one — `[gateway] listen = 127.0.0.1:0` — else both checks collide on
/// the `0.0.0.0:8080` default ("Address already in use"). `worker_listen` is already forced ephemeral by
/// compose. The worker runs the always-linked **dummy** driver against a local snapshot (no GPU, no
/// download — R3); auth off. The dummy driver's `[..options]` are explicit (`vocab_size = 128`
/// constrains synthetic samples to valid single-byte UTF-8 IDs from the 128-token fixture;
/// `arch_name` required — the fixture dir has only `tokenizer.json`, no `config.json` for the
/// standalone to auto-discover them from).
fn standalone_toml(snapshot: &str) -> String {
    format!(
        "[controller]\n\
         \n\
         [gateway]\n\
         listen = \"127.0.0.1:0\"\n\
         \n\
         [worker]\n\
         [worker.auth]\n\
         enabled = false\n\
         \n\
         [worker.model]\n\
         name = \"smoke\"\n\
         hf_repo = \"{snapshot}\"\n\
         \n\
         [worker.model.driver]\n\
         type = \"dummy\"\n\
         device = [\"cpu\"]\n\
         \n\
         [worker.model.driver.options]\n\
         vocab_size = 128\n\
         arch_name = \"qwen3\"\n"
    )
}

/// Absolute path to the committed fixture snapshot dir (contains `tokenizer.json`).
fn fixture_snapshot() -> String {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/smoke-model-ascii");
    p.to_string_lossy().into_owned()
}

async fn boot() -> Result<pie_bin::StandaloneHandle> {
    let (controller, gateway, worker) = derive_standalone(&standalone_toml(&fixture_snapshot()))?;
    run_standalone(controller, gateway, worker, Mode::Local).await
}

fn build_direct_channel_inferlet() -> Result<(PathBuf, PathBuf)> {
    let workspace =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let status = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "direct-channel-e2e",
        ])
        .current_dir(&workspace)
        .status()
        .context("build direct-channel-e2e inferlet")?;
    anyhow::ensure!(status.success(), "direct-channel-e2e build failed");

    let wasm = workspace.join("target/wasm32-wasip2/debug/direct_channel_e2e.wasm");
    let manifest = workspace.join("direct-channel-e2e/Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing inferlet wasm: {}", wasm.display());
    Ok((wasm, manifest))
}

fn build_chat_completion_inferlet() -> Result<(PathBuf, PathBuf)> {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let crate_dir = workspace.join("chat-completion");
    let status = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "chat-completion",
        ])
        .current_dir(&workspace)
        .status()
        .context("build chat-completion inferlet")?;
    anyhow::ensure!(status.success(), "chat-completion build failed");

    let wasm = workspace.join("target/wasm32-wasip2/debug/chat_completion.wasm");
    let manifest = crate_dir.join("Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing inferlet wasm: {}", wasm.display());
    Ok((wasm, manifest))
}

/// THE GATE — boots ONCE (process-global singletons forbid a 2nd boot) and runs all four checks:
/// (1) Tier-1: the composition root assembles all three planes in-proc over loopback, the worker
/// dials in, and `StandaloneHandle` surfaces both resolved ephemeral addrs. (2) Ping: a `Ping`
/// round-trips the real client path (REST → ingress → session → dispatch → worker → push_tokens →
/// SSE). (3) Direct FFI: a client uploads and launches a real WASM inferlet, which executes a PTIR
/// program through the dummy driver and returns the computed value. (4) Text completion: the
/// curated inferlet fixture runs prefill and decode against synthetic logits and returns decoded text.
/// Then shuts down.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn standalone_runs_ping_direct_ffi_and_chat_completion_then_shuts_down() -> Result<()> {
    let pie = boot().await?;

    // (1) Tier-1: three planes co-reside, worker dialed in, both ephemeral loopback addrs resolved.
    assert_ne!(
        pie.listen_addr.port(),
        0,
        "client-facing edge must bind a real ephemeral port"
    );
    assert_ne!(
        pie.worker_addr.port(),
        0,
        "worker dial-in must bind a real ephemeral port (worker co-resides + dialed in)"
    );
    assert!(
        pie.listen_addr.ip().is_loopback() && pie.worker_addr.ip().is_loopback(),
        "standalone is loopback-only"
    );

    // (2) Ping through ingress — exercises the full client path with no tokenization. Serialize the
    // body with serde_json (a bin/pie dep) — reqwest here lacks the `json` feature.
    let payload = serde_json::to_vec(&serde_json::json!({ "type": "ping", "corr_id": 1 }))?;
    let resp = tokio::time::timeout(
        Duration::from_secs(20),
        reqwest::Client::new()
            .post(format!("http://{}/v1/generate", pie.listen_addr))
            .header("x-pie-identity", "smoke/test") // REQUIRED trust-edge gate (else 401)
            .header("content-type", "application/json")
            .body(payload)
            .send(),
    )
    .await
    .context("REST ping timed out")??;
    assert_eq!(resp.status(), 200, "ingress one-shot must accept the turn");

    let body = tokio::time::timeout(Duration::from_secs(20), resp.text())
        .await
        .context("REST ping body timed out")??;
    assert!(
        body.contains("[DONE]"),
        "the turn must stream back and reach the clean [DONE] sentinel; got: {body}"
    );

    // (3) Upload and run a real inferlet through client → gateway → worker → engine → dummy FFI.
    let (wasm, manifest) = build_direct_channel_inferlet()?;
    let client = tokio::time::timeout(
        Duration::from_secs(20),
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "smoke/test"),
    )
    .await
    .context("client connect timed out")?
    .context("connect client")?;
    tokio::time::timeout(
        Duration::from_secs(20),
        client.authenticate("smoke/test", &None),
    )
    .await
    .context("client authentication timed out")?
    .context("authenticate client")?;
    tokio::time::timeout(
        Duration::from_secs(30),
        client.add_program(&wasm, &manifest, true),
    )
    .await
    .context("program upload timed out")?
    .context("upload direct-channel-e2e")?;

    let mut process = tokio::time::timeout(
        Duration::from_secs(20),
        client.launch_process("direct-channel-e2e@0.1.0".into(), "{}".into(), true),
    )
    .await
    .context("inferlet launch timed out")?
    .context("launch direct-channel-e2e")?;
    let result = tokio::time::timeout(Duration::from_secs(20), process.wait_for_return())
        .await
        .context("inferlet completion timed out")?
        .context("wait for direct-channel-e2e")?;
    assert_eq!(result, "value=42");
    drop(process);

    // (4) Run the production chat-completion inferlet. Dummy logits are synthetic, so only
    // execution and non-empty decoding are meaningful; coherence belongs to real-driver tests.
    let (wasm, manifest) = build_chat_completion_inferlet()?;
    tokio::time::timeout(
        Duration::from_secs(30),
        client.add_program(&wasm, &manifest, true),
    )
    .await
    .context("chat-completion upload timed out")?
    .context("upload chat-completion")?;
    let input = serde_json::json!({
        "prompt": "Say hello.",
        "system": "Answer briefly.",
        "max_tokens": 32,
        "temperature": 0.1,
        "top_p": 0.95,
    })
    .to_string();
    let mut process = tokio::time::timeout(
        Duration::from_secs(20),
        client.launch_process("chat-completion@0.1.0".into(), input, true),
    )
    .await
    .context("chat-completion launch timed out")?
    .context("launch chat-completion")?;
    let completion = tokio::time::timeout(Duration::from_secs(30), process.wait_for_return())
        .await
        .context("chat-completion timed out")?
        .context("wait for chat-completion")?;
    anyhow::ensure!(
        !completion.is_empty(),
        "Dummy chat-completion returned no decoded text"
    );
    eprintln!("[dummy-chat-completion] returned: {completion:?}");

    drop(process);
    drop(client);
    pie.shutdown().await;
    Ok(())
}
