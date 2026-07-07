//! §6.2 PTIR stage-program e2e — real driver (4090). The north-star other half:
//! a guest registers a greedy-argmax PTIR container, instantiates it (seed BOS),
//! submits, and takes the output — flowing guest → `ptir_host::submit` →
//! `submit_async` → scheduler → charlie's executor hook (`ptir_program_*` decode
//! → `PtirInstance` fire on the model's logits → harvest → `ptir_output_*`) →
//! `extract_per_request` → `marshal_response` → `out.take()`.
//!
//! PATH B (delta's SDK): the guest builds the container parametrically for the
//! model's vocab (Qwen3-0.6B, V=151936) — echo's `bind` requires `Logits` dim ==
//! model vocab, so a vocab-8 baked container can't be reused against Qwen.
//!
//! Bring-up: run with `PIE_PTIR_TRACE=1` — the driver prints `[ptir-hook] FIRED`
//! iff the empty-KV-projection req reaches the executor hook (case a) vs is gated
//! by `forward_prepare`/plan upstream (case b). The correctness assert (taken
//! token == the model's greedy argmax of BOS) tightens once the path is green.
//!
//!   PIE_PTIR_TRACE=1 cargo test -p pie-bin --features driver-cuda,ptir \
//!     --test cuda_ptir_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "§6.2 PTIR stage-program e2e: needs the 4090 + cuda + qwen-3-0.6b + the ptir feature"]
async fn ptir_greedy_stage_program_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[ptir-e2e] booted, listen_addr={}", pie.listen_addr);

    // Build the PATH-B guest inferlet (builds the greedy container for the live
    // model's vocab via echo's ptir container types, then register→instantiate→
    // submit→take). It's a member of the runtime test-inferlets workspace.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "ptir-greedy-e2e"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "ptir-greedy-e2e wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/ptir_greedy_e2e.wasm");
    let manifest = ws.join("ptir-greedy-e2e/Pie.toml");

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
    eprintln!("[ptir-e2e] program installed, launching greedy stage-program…");

    let mut proc = client
        .launch_process("ptir-greedy-e2e@0.1.0".to_string(), "{\"bos\": 1}".to_string(), true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[ptir-e2e] returned: {out}");

    pie.shutdown().await;

    // The register→instantiate→submit→take loop returned a token through the full
    // driver stage-runner path (guest → hook fires on real logits → argmax →
    // harvest → marshal → take). Parse `token=<N>` and assert it's a valid,
    // non-degenerate token id: `N > 0` and `N < vocab`. This rules out the exact
    // failure modes the wire had during bring-up — a 0/empty read from a dropped
    // `ptir_output_*` (the `no cell available` root) reads back as token 0, and a
    // garbage/OOB read exceeds the vocab. The full greedy-equality assert
    // (`N == greedy_argmax(model.forward(BOS=1))`) lands once the inferlet also
    // returns a plain sampling-IR greedy decode of the same BOS to diff against.
    let token: u32 = out
        .trim()
        .strip_prefix("token=")
        .and_then(|s| s.trim().parse().ok())
        .with_context(|| format!("§6.2 e2e: could not parse `token=<N>` from: {out:?}"))?;
    // Qwen3-0.6B vocab (the inferlet builds Logits[1, V] for the live model).
    const QWEN3_0_6B_VOCAB: u32 = 151_936;
    // Ground truth: HF Qwen3-0.6B argmax of the next token after context=[BOS=1],
    // verified against transformers 5.9 (bf16 logit 8.75, fp32 8.84; 19148 scores
    // only ~4.0). A HARD value gate — the PTIR stage-runner must reproduce the
    // model's real greedy argmax, not just any plausible token.
    const QWEN3_0_6B_BOS1_ARGMAX: u32 = 14582;
    anyhow::ensure!(
        token > 0 && token < QWEN3_0_6B_VOCAB,
        "§6.2 e2e: implausible token {token} (expected 0 < N < {QWEN3_0_6B_VOCAB}) — a dropped \
         `ptir_output_*` reads back 0, an OOB read exceeds vocab: {out}"
    );
    anyhow::ensure!(
        token == QWEN3_0_6B_BOS1_ARGMAX,
        "§6.2 e2e CORRECTNESS FAILED — PTIR stage-program token {token} != HF Qwen3-0.6B \
         ground-truth greedy argmax {QWEN3_0_6B_BOS1_ARGMAX} for BOS=[1]. (Was 19148 when the \
         tier-0 stage-runner misread the bf16 `ws.logits` as f32.): {out}"
    );
    eprintln!("[ptir-e2e] §6.2 GREEN — greedy stage-program token {token} == HF ground truth {QWEN3_0_6B_BOS1_ARGMAX}");
    Ok(())
}
