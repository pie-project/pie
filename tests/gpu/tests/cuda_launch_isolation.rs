//! **THE LAUNCH-ISOLATION GATE**: one boot, three launches, one answer.
//!
//! Every other serving gate in this directory launches ONCE per process, and
//! that is why none of them saw this. A sequence is supposed to be private to
//! its launch: the inferlet asks for pages, writes its state into the pool's
//! slot, and its teardown gives both back. Launch N+1 starts from zero and
//! must not be able to tell that launch N ever ran.
//!
//! It could. Three identical `text-completion` launches through one booted
//! worker answered
//!
//! ```text
//! 1  " Paris.\nThe capital of France is"          <- correct
//! 2  " the capital of France is the capital of"   <- echo-shaped
//! 3  " France is France is France is France is"   <- worse
//! ```
//!
//! at zero envelopes, so nothing in the descriptor-port plane was on that path
//! (`palo` build log 18, "found en route, out of scope, unfixed").
//!
//! # What it was, since echo-shaped garbage has more than one cause
//!
//! Not the KV pages. A recycled page carries its last tenant's bytes and
//! always has, and that is sound because `kv_len` says nothing lives past the
//! append — the rows are overwritten before they are read. The probe agreed:
//! every launch arrived at `held == 0` with the same page block and its own
//! correct extents.
//!
//! It was the RECURRENT bank. `qwen35-d0.8b` is a GDN/attention hybrid —
//! eighteen of its twenty-four layers are gated-delta scans over a per-slot
//! conv+delta state, and a scan reads its WHOLE state on its first step, so
//! there is no `kv_len` to make a stale one harmless. The shell zeroes those
//! banks in `Pools::clear`, called from `Shell::open`, which the contract has
//! no verb for and an engine that keeps its own page table therefore never
//! reached. Launch 1 was right because the slab was `cudaMemset` at load;
//! launches 2 and 3 continued launch 1's sequence through eighteen layers
//! while the six attention layers read a correct five-token prompt, which is
//! exactly what "fluent, built out of the prompt's own words, worse each
//! time" looks like. `driver_cuda::serve` now clears the slot on the fire
//! that states `held == 0`.
//!
//! The assertion is an identity, not a fluency: launch 2 and launch 3 must
//! say what launch 1 said, token for token. Greedy decoding is what makes
//! that claim available at all.
//!
//! # Both token paths, because the leak was under both
//!
//! `text-completion` brings its token back to the host; `token-healing
//! --heal=false` carries it on the device through the descriptor port. They
//! differ in one line of guest source and agree token for token
//! (`cuda_device_carried_round_trip`), so running the three-launch pattern
//! through each is what says the lifecycle is fixed rather than one token
//! path.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features driver-cuda-13 \
//!   --test cuda_launch_isolation -- --ignored --nocapture --test-threads=1
//! ```

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// Sixteen: the prompt is five tokens and a KV page is sixteen, so every
/// launch crosses a page boundary and touches a second pool page. That is not
/// what this gate is about, but it costs nothing and it means a page-table
/// regression cannot hide behind a passing isolation claim.
const MAX_TOKENS: u32 = 16;

/// Three, not two: a leak that carries state forward compounds, and the third
/// launch is where a two-launch gate's "close enough" stops being arguable.
const LAUNCHES: usize = 3;

/// Build one guest fixture to wasm and return `(wasm, manifest)`. Built here
/// rather than committed, so the gate cannot pass against a stale binary of a
/// guest contract that moved.
fn build_fixture(pkg: &str, wasm_stem: &str) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", pkg])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "{pkg} wasm build failed");
    let wasm = workspace.join(format!("target/wasm32-wasip2/debug/{wasm_stem}.wasm"));
    let manifest = workspace.join(format!("{pkg}/Pie.toml"));
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());
    anyhow::ensure!(
        manifest.exists(),
        "missing manifest: {}",
        manifest.display()
    );
    Ok((wasm, manifest))
}

/// Launch `program` `LAUNCHES` times over ONE connection to one booted
/// worker, and return what each launch answered.
async fn three_launches(
    listen_addr: &std::net::SocketAddr,
    pkg: &str,
    wasm_stem: &str,
    program: &str,
    input: serde_json::Value,
) -> Result<Vec<String>> {
    let (wasm, manifest) = build_fixture(pkg, wasm_stem)?;

    let client = Client::connect_with_identity(&format!("ws://{listen_addr}/v1/ws"), "test-user")
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

    let mut answers = Vec::with_capacity(LAUNCHES);
    for i in 0..LAUNCHES {
        let mut proc = client
            .launch_process(program.to_string(), input.to_string(), true)
            .await
            .with_context(|| format!("launch {i}"))?;
        let out = proc
            .wait_for_return()
            .await
            .with_context(|| format!("wait_for_return {i}"))?;
        let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
        let text = parsed["text"].as_str().unwrap_or_default().to_string();
        eprintln!("[isolation] {pkg} launch {i} -> {text:?}");
        answers.push(text);
    }
    Ok(answers)
}

/// Every launch says the same thing, and the first one says the pinned thing.
fn assert_identical(pkg: &str, answers: &[String]) {
    assert!(
        answers[0].starts_with(common::SERVING_GREEDY_16),
        "{pkg} launch 0 answered {:?}, not the pinned greedy continuation {:?}",
        answers[0],
        common::SERVING_GREEDY_16
    );
    for (i, a) in answers.iter().enumerate().skip(1) {
        assert_eq!(
            a, &answers[0],
            "{pkg} launch {i} answered {a:?} but launch 0 answered {:?} — \
             a sequence is not private to its launch",
            answers[0]
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the launch-isolation gate: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn three_launches_through_one_boot_answer_identically() -> Result<()> {
    common::init_trace();
    let pie = common::boot_serving().await?;
    eprintln!("[isolation] booted, listen_addr={}", pie.listen_addr);

    // The host-carried token path.
    let host = three_launches(
        &pie.listen_addr,
        "text-completion",
        "text_completion",
        "text-completion@0.1.0",
        serde_json::json!({
            "prompt": common::SERVING_PROMPT,
            "max_tokens": MAX_TOKENS,
        }),
    )
    .await?;

    // The device-carried token path, greedy so the same constant pins it.
    let device = three_launches(
        &pie.listen_addr,
        "token-healing",
        "token_healing",
        "token-healing@0.1.0",
        serde_json::json!({
            "prompt": common::SERVING_PROMPT,
            "max_tokens": MAX_TOKENS,
            "heal": false,
        }),
    )
    .await?;

    assert_identical("text-completion", &host);
    assert_identical("token-healing", &device);

    pie.shutdown().await;
    Ok(())
}
