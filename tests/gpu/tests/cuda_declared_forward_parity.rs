//! Stage 3 hard gate: the declared forward is token-identical to the
//! hand-written one, on the same GPU, the same fire, the same reduction
//! order.
//!
//! The `PIE_DECLARED_FORWARD` gate is shell-controlled, so the harness runs
//! this twice — gate-OFF (hand-written `llama_like_forward_paged`) then
//! gate-ON (`llama_like_forward_declared` walking the traced form) — and the
//! parity claim is the two token vectors being byte-identical, compared
//! through a cross-invocation record file (the `cuda_mtp_stage1` idiom).
//!
//! Two disciplines make the exact-match bar legitimate here where
//! `cuda_contention` correctly refuses it:
//! * single request — co-batched decode is not batch-invariant (bf16
//!   reduction order flips argmax near ties; `cuda_contention.rs`), so the
//!   comparison never mixes batch compositions;
//! * both invocations run the full fast-path vocabulary, fused decode
//!   postprocess included. The v0 harness pinned
//!   `PIE_CUDA_DECODE_FUSED_POST=0` on both sides because the declared
//!   executor spoke only the hand-written UNFUSED path and the fused
//!   postprocess rounds differently; the executor now carries the fused
//!   decode-QKV peephole (declared_forward.cpp), and this file's DELETION
//!   of that override is the proof the peephole matched — parity now holds
//!   with both sides fused, under the environment's default gates.
//!
//! `#[ignore]`, driver-cuda. Run OFF then ON, the second invocation gates:
//!   cargo test -p pie-gpu-tests --no-default-features --features driver-cuda \
//!     --test cuda_declared_forward_parity -- --ignored --nocapture
//!   PIE_DECLARED_FORWARD=1 cargo test ... --test cuda_declared_forward_parity ...

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

const MAX_TOKENS: usize = 48;

fn record_path(declared: bool) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pie_declared_forward_parity_{}.txt",
        if declared { "on" } else { "off" }
    ))
}

fn write_record(declared: bool, text: &str) -> Result<()> {
    std::fs::write(record_path(declared), text).context("write parity record")
}

fn read_record(declared: bool) -> Option<String> {
    std::fs::read_to_string(record_path(declared)).ok()
}

/// The sampled text out of naive-baseline's JSON result. Text rather than
/// token ids because that is what the inferlet returns; byte-equal text over
/// 48 sampled tokens is the same claim (the tokenizer is deterministic).
fn parse_text(result: &str) -> Option<String> {
    let key = "\"text\":";
    let at = result.find(key)? + key.len();
    let rest = result[at..].trim_start();
    let mut chars = rest.char_indices();
    let (_, '"') = chars.next()? else { return None };
    let mut out = String::new();
    let mut escaped = false;
    for (_, c) in chars {
        if escaped {
            out.push(c);
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            return Some(out);
        } else {
            out.push(c);
        }
    }
    None
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a CUDA GPU + qwen-3-0.6b; run gate-OFF then gate-ON"]
async fn declared_forward_token_parity() -> Result<()> {
    common::init_trace();
    let declared = std::env::var("PIE_DECLARED_FORWARD")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    let pie = common::boot_4090().await?;
    eprintln!(
        "[declared-parity] booted listen={} declared={declared}",
        pie.listen_addr
    );

    // naive-baseline over the engine-test inferlets: it is the golden model
    // pie-application-plan.md names — a seeded Gumbel-max sampler whose
    // stream depends on every logit the forward produced. (`generate` from
    // the engine-test workspace echoes a channel constant; it exercises the
    // pipeline, not the forward.)
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-wasip2",
            "-p",
            "naive-baseline",
        ])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "naive-baseline wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/release/naive_baseline.wasm");
    let manifest = ws.join("naive-baseline/Pie.toml");

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

    // Single request, fixed seed: the stream depends on every logit and on
    // nothing else that varies across the gate.
    let input = format!(
        "{{\"prompt\": \"The old clockmaker examined the strange timepiece \
         carefully\", \"max_tokens\": {MAX_TOKENS}, \"seed\": 7}}"
    );
    let mut proc = client
        .launch_process("naive-baseline@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    let text =
        parse_text(&json).with_context(|| format!("no text in result: {json}"))?;
    eprintln!(
        "[declared-parity] declared={declared} text={:?}",
        &text[..text.len().min(80)]
    );
    write_record(declared, &text)?;

    if let Some(counterpart) = read_record(!declared) {
        anyhow::ensure!(
            text == counterpart,
            "declared-vs-handwritten token streams diverge.\n  this run \
             (declared={declared}): {text:?}\n  counterpart: \
             {counterpart:?}\nA declared executor that calls the same \
             kernels in the same order must match bit-for-bit — find the \
             kernel it substituted."
        );
        eprintln!("[declared-parity] PASS: text byte-identical across the gate");
    } else {
        eprintln!(
            "[declared-parity] recorded; run the counterpart invocation \
             (PIE_DECLARED_FORWARD={}) to compare",
            if declared { "0" } else { "1" }
        );
    }

    pie.shutdown().await;
    Ok(())
}
