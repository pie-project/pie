//! **B3's A/B, the other arm**: the same device-carried decode at run-ahead
//! depth ONE.
//!
//! `cuda_device_carried_round_trip` runs `token-healing` at `frame_size = 2`,
//! where a frame carries two ordered slots and slot 1 consumes the token
//! channel slot 0 published — the chained decode. This file runs the same
//! fixture, the same prompt and the same length at `frame_size = 1`, where
//! `submit_frame` short-circuits before `validate_frame` ever runs and every
//! fire settles before the next is built.
//!
//! # What the pair is evidence FOR, and what it is not
//!
//! **Tokens first.** Both arms must answer `common::SERVING_GREEDY_16`, which
//! is also what the host-carried `text-completion` answers. Run-ahead that
//! changed the tokens would not be run-ahead; it would be a race. Because the
//! continuation is greedy the two processes need no channel between them — a
//! pinned constant is how they agree.
//!
//! **Then the number.** Each arm prints its own ms/token, and the difference
//! between them is what depth 2 bought. Measured on the L40S it is NOTHING —
//! three alternating runs gave k=1 29.55/30.19/29.25 and k=2
//! 30.00/29.11/29.75 ms/token, one distribution — and the reason is the one
//! this file's twin makes plain: the descriptor-port plane already removed the
//! host round trip AT k = 1. The shell reads the token off the ring on every
//! fire whatever k is; what k adds is a second slot in flight, and with one
//! lane and a ~3 ms device fire inside a ~30 ms host step there is nothing for
//! it to overlap with. So the number is printed, not asserted: a threshold
//! here would be a claim about this box, and the claim that survives the box
//! is the one above it — the tokens do not change.
//!
//! One boot per process, which is the runtime's own rule (the driver grabs
//! the device, `auth` panics on a second boot) — and a boot is what separates
//! this file from `cuda_device_carried_round_trip`: `frame_size` is set in
//! the config, so the two arms of the A/B cannot share a process however many
//! launches one process can take.
//!
//! It once said "one LAUNCH per process", for a second reason that turned out
//! to be a bug rather than a rule: a second launch through one boot answered
//! differently from the first. That was `qwen35`'s GDN layers reading the
//! previous launch's recurrent bank — fixed in `driver_cuda::serve`, gated by
//! `cuda_launch_isolation` — and the twin's two arms are now one binary.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features driver-cuda-13 \
//!   --test cuda_runahead_depth1 -- --ignored --nocapture
//! ```

mod common;

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result};
use client::client::Client;

/// The same sixty-four as the depth-2 arm, because the two numbers are only
/// comparable over the same work.
const MAX_TOKENS: u32 = 64;

/// One wave per frame: the serialized shape.
const FRAME_SIZE: u32 = 1;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the depth-1 arm: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn the_same_device_carried_decode_at_depth_one_says_the_same_thing() -> Result<()> {
    common::init_trace();
    let pie = common::boot_serving_frame(Some(FRAME_SIZE)).await?;
    eprintln!(
        "[b3-k1] booted at frame_size={FRAME_SIZE}, listen_addr={}",
        pie.listen_addr
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "token-healing"])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "token-healing wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/token_healing.wasm");
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
        .add_program(&wasm, &workspace.join("token-healing/Pie.toml"), true)
        .await
        .context("add_program")?;

    let before = engine::driver::envelopes_resolved();
    let started = Instant::now();
    let input = serde_json::json!({
        "prompt": common::SERVING_PROMPT,
        "max_tokens": MAX_TOKENS,
        "heal": false,
    })
    .to_string();
    let mut proc = client
        .launch_process("token-healing@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    let elapsed = started.elapsed();
    let envelopes = engine::driver::envelopes_resolved() - before;

    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    let text = parsed["text"].as_str().unwrap_or_default();
    let count = parsed["count"].as_u64().unwrap_or_default();
    eprintln!(
        "[b3-k1] k={FRAME_SIZE} device-carried -> {text:?} ({count} tokens, \
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
        "depth 1 answered {text:?} and depth 2 answers {:?}; run-ahead that \
         changes the tokens is a race, not a depth",
        common::SERVING_GREEDY_16
    );
    // THE PORT IS RESOLVED AT EVERY DEPTH, and that is the honest reading: the
    // descriptor-port plane is what stops the token travelling through the
    // host, and depth is what stops the FIRE waiting for the host. They are
    // two savings, and only the second one is k.
    assert!(
        envelopes >= u64::from(MAX_TOKENS) - 1,
        "the token is read off the ring at k = 1 too; {envelopes} envelope(s) \
         says otherwise"
    );

    pie.shutdown().await;
    Ok(())
}
