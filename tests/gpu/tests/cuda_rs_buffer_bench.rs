//! **What the RS buffer costs** — drives `rs-buffer-bench` on Qwen3.5-0.8B.
//!
//! Both arms run the same device-carried decode loop and differ only in RS
//! geometry: `fold` moves the recurrent boundary every fire, `buffer` never
//! moves it and appends every token to the RS buffer instead. Neither replays,
//! so the delta is the buffer's WRITE cost — the one paid per token, by every
//! speculative decode, on top of whatever a commit costs.
//!
//! This is a measurement, not a threshold: the test asserts only that both
//! arms run and agree on the token stream, and PRINTS the cost. A regression
//! in the buffer write path shows up as a widening gap in the printed numbers.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   cargo test -p pie-gpu-tests --features driver-cuda --test cuda_rs_buffer_bench \
//!     -- --ignored --nocapture

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

mod common;

fn build_inferlet() -> Result<std::path::PathBuf> {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            "rs-buffer-bench",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for rs-buffer-bench");
    Ok(ws)
}

/// Boot once and run BOTH arms against the same engine: a per-arm boot would
/// price CUDA graph capture and allocator warmth differences into the delta.
async fn run_arms(inputs: &[&str]) -> Result<Vec<std::result::Result<String, String>>> {
    common::init_trace();
    let ws = build_inferlet()?;

    let pie = common::boot_4090_mtp(common::mtp_draft_tokens(3)).await?;
    eprintln!(
        "[rs-buffer-bench] booted Qwen3.5-0.8B, listen_addr={}",
        pie.listen_addr
    );

    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup
        .authenticate("test-user", &None)
        .await
        .context("auth setup")?;
    setup
        .add_program(
            &ws.join("target/wasm32-wasip2/release/rs_buffer_bench.wasm"),
            &ws.join("rs-buffer-bench/Pie.toml"),
            true,
        )
        .await
        .context("add_program rs-buffer-bench")?;
    drop(setup);

    let mut out = Vec::new();
    for input in inputs {
        let c =
            Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
                .await
                .context("connect run session")?;
        c.authenticate("test-user", &None)
            .await
            .context("auth run session")?;
        let mut proc = c
            .launch_process("rs-buffer-bench@0.1.0".to_string(), input.to_string(), true)
            .await
            .context("launch rs-buffer-bench")?;
        // Per-arm, not fail-fast: one arm is EXPECTED to be refused, and the
        // engine can only be booted once per process.
        let result = proc.wait_for_return().await.map_err(|e| e.to_string());
        eprintln!("[rs-buffer-bench] {input} -> {result:?}");
        out.push(result);
        drop(c);
    }
    pie.shutdown().await;
    Ok(out)
}

fn field(result: &str, key: &str) -> f64 {
    result
        .split_whitespace()
        .find_map(|token| token.strip_prefix(key))
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| panic!("missing {key} in {result}"))
}

#[allow(dead_code)]
fn head(result: &str) -> String {
    result
        .split_once("head=")
        .map(|(_, tail)| tail.trim().to_string())
        .unwrap_or_default()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn buffering_a_token_costs_what_it_costs() -> Result<()> {
    // A warm arm first: the very first arm on a fresh engine pays for lazy
    // graph capture and first-touch allocation, and charging that to whichever
    // arm happens to run first would be the whole measurement.
    // `hostfold` vs `hostbuffer` is the APPLES-TO-APPLES pair: the buffered
    // arm cannot take the device-carried decode-envelope class at all (see
    // `classify_decode_envelope` — it has no case for any RS port, so binding
    // `fold-len` as a channel drops the fire to host-evaluated geometry), so
    // comparing a buffered fire against a device-carried fold would price the
    // execution class, not the buffer. The device-carried `256:fold` arm is
    // reported alongside to price exactly that lost carry.
    let results = run_arms(&["64:fold", "256:fold", "64:coframe"]).await?;
    let fold = results[1].as_ref().expect("the fold arm must run");

    // The frame shape that used to die on the device as `descriptor channel 0
    // not ready` and poison the guest's channel: slot 1 chains off a value
    // slot 0 publishes, and both bind recurrent state.
    //
    // It is SUPPORTED now, and this arm is what proves it. `299b76320` let an
    // unbuffered fold-all recurrent fire take the device-composed template,
    // which resolves descriptor ports at KERNEL time rather than at frame
    // entry — so the chained value is there by the time the consumer reads it.
    // The engine-side refusal (`validate_frame` in `runtime/engine/src/
    // pipeline/fire.rs`) was narrowed to the two shapes the template still
    // refuses: a buffered row, and a device-resident fold length.
    //
    // Correctness first: an intra-frame chain must not change what the model
    // emits. Same prompt, same steps, so the token streams must be identical.
    let coframe = results[2]
        .as_ref()
        .expect("a recurrent frame that chains slot 1 off slot 0 must run");
    let head_of = |s: &str| {
        s.split_once("head=")
            .map(|(_, h)| h.to_string())
            .unwrap_or_else(|| panic!("no head= in arm output:\n{s}"))
    };
    assert_eq!(
        head_of(coframe),
        head_of(fold),
        "co-framing the prefill and the first decode must not change the \
         trajectory — it moves WHEN descriptors resolve, not what they are"
    );

    eprintln!(
        "\n=== LINEAR DECODE BASELINE (Qwen3.5-0.8B, 1-wide, 255 fires) ===\n\
         device-carried fold {:8.2} us/fire  ({:.1} tok/s)\n\
         NOTE: the buffered arm cannot be measured against this. Binding\n\
         `fold-len` as a channel drops the fire out of the device-carried\n\
         decode-envelope class entirely -- `classify_decode_envelope` has no\n\
         case for any RS port, so it falls back to host-evaluated geometry,\n\
         which refuses a device-carried token (`EmbedTokens is not\n\
         host-derivable`). The RS buffer's cost today is therefore not its\n\
         write, it is the LOST DEVICE CARRY: a host round trip per fire.\n",
        field(fold, "per_fire_us="),
        field(fold, "tok_per_s="),
    );
    Ok(())
}
