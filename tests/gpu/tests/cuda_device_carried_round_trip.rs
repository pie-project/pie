//! **B3's SECOND GATE**: a legacy device-carried decode serves, and it says
//! exactly what the host-carried one says.
//!
//! `cuda_serve_round_trip` proves the serving door with `text-completion`,
//! whose decode brings its token back to the host and sends it down again —
//! one host round trip per token, which was the honest depth for a shell that
//! resolved no descriptor port. This file is the other half: `token-healing`,
//! a legacy fixture whose epilogue writes the sampled token straight into the
//! channel the `embed` port reads and never tells the host. Against the old
//! shell that fire was refused by name (`EmbedTokens is not host-derivable`);
//! against this one the shell reads the token off the ring the epilogue wrote
//! (`driver_cuda::program::ports`).
//!
//! # Why `token-healing`, of all the device-carried fixtures
//!
//! Because it is GREEDY, and the claim is an identity rather than a
//! plausibility. Its unhealed baseline (`heal = false`) encodes the caller's
//! prompt verbatim, argmaxes the prefill row under an all-true mask, and then
//! runs a `reduce_argmax` decode loop whose token goes back into `token_in` —
//! which is `text-completion`'s decode exactly, minus the round trip. Same
//! prompt, same weights, same arithmetic, differing in one line of guest
//! source. So a difference in the continuation is a difference in WHICH CELL
//! the shell read, and nothing else.
//!
//! # Three claims, two launches, one boot
//!
//! 1. **The corpus fixture serves, as committed.** `naive-baseline` is the
//!    shape every algorithm inferlet in `tests/inferlets` is written to — one
//!    chunked prefill, then a `ptir::run_ahead` decode loop whose epilogue
//!    writes the sampled token back into the `embed` channel and its own next
//!    position, extent, write slot and page CSR beside it — and against a
//!    shell answering `PortMask::NONE` every one of them died on the first
//!    fire with `EmbedTokens is not host-derivable`. It SAMPLES, so there is
//!    no greedy reference: what is asserted is that the fire happens and that
//!    the envelope counter says every decode fire read its token off a ring.
//!    The text is printed, not checked — a fluency assertion would pass on a
//!    shell that read the wrong cell, which is what claim 2 exists to catch.
//! 2. **The greedy fixture says the same thing the host-carried one says.**
//!    The first sixteen tokens are `common::SERVING_GREEDY_16`, which is what
//!    `cuda_serve_round_trip` pins for the host-carried fixture and what
//!    `cuda_boot_smoke` and `serve_smoke` pin one and two layers below that.
//! 3. **The token never reached the host.**
//!    `engine::driver::envelopes_resolved()` counts the one thing that
//!    HAPPENS when a round trip does not: an envelope read off a device ring
//!    in front of the walk. One per decode fire, counted per launch.
//!
//! **The sampled arm runs FIRST and the greedy one second**, which is the
//! opposite of the order that reads naturally, and deliberately: the identity
//! claim is then made through a recurrent slot and a page block that another
//! sequence has already used. See below.
//!
//! # It used to be two binaries with one launch each, and that was a bug
//!
//! The runtime's rule is one boot per process (the driver grabs the device,
//! `auth` panics on a second boot), and `cuda_runahead_depth1` is still its
//! own binary for that reason: it boots at `frame_size = 1` and a boot is
//! not a launch parameter.
//!
//! `cuda_naive_baseline_serves` was a third binary at THIS boot — the same
//! `frame_size = 2`, the same checkpoint, the same prompt — and its only
//! reason for being separate was a second rule that turned out not to be a
//! rule. A SECOND launch through one boot answered differently from the
//! first: three identical `text-completion` launches in one process gave
//! `" Paris.\nThe capital of France is"`, then `" the capital of France is
//! the capital of"`, then `" France is France is France is France is"`, at
//! zero envelopes. That was not a property of launching twice. It was
//! `qwen35`'s eighteen GDN layers reading the previous launch's recurrent
//! bank, because nothing on the engine's path cleared a slot a fresh
//! sequence took — fixed in `driver_cuda::serve`, gated by
//! `cuda_launch_isolation`, and the two arms below are the collapse of the
//! routing-around.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features driver-cuda-13 \
//!   --test cuda_device_carried_round_trip -- --ignored --nocapture
//! ```

mod common;

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use client::client::Client;

/// Sixty-four tokens: sixteen for the identity claim (which is what
/// `common::SERVING_GREEDY_16` pins, and what crosses a KV page boundary) and
/// the rest so the ms/token number is not mostly process startup.
const MAX_TOKENS: u32 = 64;

/// The corpus arm's budget. Sixteen is enough to cross a page boundary, which
/// is all a sampled continuation can be held to.
const CORPUS_TOKENS: u32 = 16;

/// Waves per frame. Two, because that is the number this gate is about: at
/// k = 1 `submit_frame` short-circuits before `validate_frame` ever runs, so a
/// chained slot is only expressible from two up. `cuda_runahead_depth1` is the
/// other arm.
const FRAME_SIZE: u32 = 2;

/// Build one guest fixture to wasm and return `(wasm, manifest)`.
///
/// Built here rather than committed as a `.wasm`, so the gate cannot pass
/// against a stale binary of a guest contract that moved.
fn build_fixture(pkg: &str, wasm_stem: &str) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", pkg])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "{pkg} wasm build failed");
    let wasm = workspace.join(format!("target/wasm32-wasip2/debug/{wasm_stem}.wasm"));
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());
    Ok((wasm, workspace.join(format!("{pkg}/Pie.toml"))))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the device-carried gate: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn a_device_carried_decode_says_what_the_host_carried_one_says() -> Result<()> {
    common::init_trace();
    let pie = common::boot_serving_frame(Some(FRAME_SIZE)).await?;
    eprintln!(
        "[b3-dev] booted at frame_size={FRAME_SIZE}, listen_addr={}",
        pie.listen_addr
    );

    let (corpus_wasm, corpus_manifest) = build_fixture("naive-baseline", "naive_baseline")?;
    let (greedy_wasm, greedy_manifest) = build_fixture("token-healing", "token_healing")?;

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client
        .authenticate("test-user", &None)
        .await
        .context("auth")?;
    client
        .add_program(&corpus_wasm, &corpus_manifest, true)
        .await
        .context("add_program naive-baseline")?;
    client
        .add_program(&greedy_wasm, &greedy_manifest, true)
        .await
        .context("add_program token-healing")?;

    // ── ARM 1: the corpus claim, and it runs first on purpose ────────────
    let before = engine::driver::envelopes_resolved();
    let input = serde_json::json!({
        "prompt": common::SERVING_PROMPT,
        "max_tokens": CORPUS_TOKENS,
    })
    .to_string();
    let mut proc = client
        .launch_process("naive-baseline@0.1.0".to_string(), input, true)
        .await
        .context("launch naive-baseline")?;
    let out = proc
        .wait_for_return()
        .await
        .context("wait_for_return naive-baseline")?;
    let envelopes = engine::driver::envelopes_resolved() - before;

    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    let text = parsed["text"].as_str().unwrap_or_default();
    let count = parsed["count"].as_u64().unwrap_or_default();
    eprintln!("[b3-corpus] {text:?} ({count} tokens, {envelopes} envelopes)");

    assert_eq!(
        count, CORPUS_TOKENS as u64,
        "the corpus fixture returned {count} of {CORPUS_TOKENS} tokens"
    );
    assert!(!text.is_empty(), "the completion is empty");
    assert!(
        envelopes >= u64::from(CORPUS_TOKENS) - 1,
        "one fire per generated token after the prefill, every one of them \
         reading its token off a device ring; {envelopes} envelope(s) says \
         some took the host path"
    );

    // ── ARM 2: the identity claim, through a slot arm 1 already used ─────
    let before = engine::driver::envelopes_resolved();
    let started = Instant::now();
    let input = serde_json::json!({
        "prompt": common::SERVING_PROMPT,
        "max_tokens": MAX_TOKENS,
        // The unhealed baseline: the prompt is encoded verbatim and the first
        // mask is all-true, so the prefill's masked argmax IS the greedy
        // argmax and the decode loop is `reduce_argmax` throughout.
        "heal": false,
    })
    .to_string();
    let mut proc = client
        .launch_process("token-healing@0.1.0".to_string(), input, true)
        .await
        .context("launch token-healing")?;
    let out = proc
        .wait_for_return()
        .await
        .context("wait_for_return token-healing")?;
    let elapsed = started.elapsed();
    let envelopes = engine::driver::envelopes_resolved() - before;

    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    let text = parsed["text"].as_str().unwrap_or_default();
    let count = parsed["count"].as_u64().unwrap_or_default();
    eprintln!(
        "[b3-dev] k={FRAME_SIZE} device-carried -> {text:?} ({count} tokens, \
         {envelopes} envelopes, {:.1} ms total, {:.2} ms/token)",
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e3 / count.max(1) as f64
    );

    assert_eq!(
        count, MAX_TOKENS as u64,
        "the inferlet returned {count} of {MAX_TOKENS} tokens"
    );
    assert!(
        text.starts_with(common::SERVING_GREEDY_16),
        "the device-carried continuation was {text:?}; the host-carried \
         fixture answers {:?} for the same prompt, and the two differ in one \
         line of guest source — whether the sampled token goes back through \
         the host",
        common::SERVING_GREEDY_16
    );
    assert!(
        envelopes >= u64::from(MAX_TOKENS) - 1,
        "one fire per generated token after the prefill, and every one of \
         those had to read its token off the ring the previous epilogue \
         wrote; {envelopes} envelope(s) says some of them took the host path"
    );

    pie.shutdown().await;
    Ok(())
}
