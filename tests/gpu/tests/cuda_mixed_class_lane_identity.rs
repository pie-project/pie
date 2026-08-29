//! **A DECODE LANE'S ANSWER MUST NOT DEPEND ON WHO IS PREFILLING BESIDE IT.**
//!
//! `engine-cuda`'s own gates say this at the shell:
//! `serve_smoke::a_fire_that_mixes_prefill_and_decode_says_what_each_lane_says_alone`
//! runs a decode lane beside a prefill lane through `Shell::fire` and gets its
//! solo tokens back, and the seated form (a lane carrying its own page table)
//! is bit-identical too. This file asks the same question one layer up —
//! through the runtime, the scheduler's frames and the guest programs the
//! serving door actually runs — because that is where the answer changed.
//!
//! Measured (the interference hunt): a greedy `text-completion` lane over
//! "The capital of France is" answers ` Paris.\nThe capital of France is
//! Paris.` alone and, with ANY co-resident guest whose prompt is two tokens
//! or more, ` Paris.\nThe following are the results of the following:`. The
//! flip is at the fifth generated token, it is deterministic, and it is
//! CONTENT-INDEPENDENT: a 400-token "banana" neighbour and a 400-token
//! "Kilimanjaro" neighbour produce the same wrong continuation, so it is not
//! the neighbour's cache leaking in. A one-token neighbour — a lane that
//! stands in the DECODE class, not the prefill one — never moves it.
//!
//! # What it was
//!
//! `KvDelta::pages` is the POOL's page ids — `engine::store::kv::geometry_with`
//! pushes each entry straight into the page CSR — and the host-geometry fire
//! path was filling it with the guest's WORKING-SET-relative indexes, which
//! every guest states as `0 .. reserved`. So every lane alive in the process
//! addressed pool pages `0, 1, …`. Alone that is invisible (a lane reads back
//! the pages it wrote) and under a homogeneous load it is invisible too (the
//! colliding writes are the same bytes), which is why the smoke gates, the
//! seated gates and a 72-run homogeneous soak were all green over it.
//! `runtime::pipeline::fire::map_lane_pages` is the translation.
//!
//! The second test is the same bug seen from the sampler's end. It is here
//! and not in `test_curated.py` because the property is about CONCURRENCY,
//! and a curated single-shot suite has nowhere to put one.
//!
//! ```text
//! PIE_COMPILER_LAUNCHER=env cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_mixed_class_lane_identity -- --ignored --nocapture
//! ```

#![cfg(feature = "_engine-cuda")]

mod common;

use anyhow::{Context, Result};
use client::client::{Client, Process};
use std::path::Path;
use std::process::Command;

/// How many tokens the probe generates. Long enough to pass the fifth, which
/// is where the divergence lands.
const TOKENS: u32 = 16;

/// The neighbour's prompt. Two tokens or more is all it takes; this is longer
/// so the neighbour is unmistakably a PREFILL lane.
const NEIGHBOUR_PROMPT: &str = "banana banana banana banana banana banana";

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

/// Start one `text-completion` and hand back the live process, so a caller can
/// have several in flight before it waits on any of them — which is what makes
/// them share fires.
async fn start(client: &Client, prompt: &str, tokens: u32) -> Result<Process> {
    let input = serde_json::json!({ "prompt": prompt, "max_tokens": tokens }).to_string();
    client
        .launch_process("text-completion@0.1.0".to_string(), input, true)
        .await
        .context("launch text-completion")
}

async fn said(proc: &mut Process) -> Result<String> {
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    let parsed: serde_json::Value = serde_json::from_str(&out).context("the return is JSON")?;
    Ok(parsed["text"].as_str().unwrap_or_default().to_string())
}

async fn completion(client: &Client, prompt: &str, tokens: u32) -> Result<String> {
    let mut proc = start(client, prompt, tokens).await?;
    said(&mut proc).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn a_decoding_lane_says_the_same_thing_beside_a_prefilling_one() -> Result<()> {
    common::init_trace();
    let pie = common::boot_serving().await?;
    let (wasm, manifest) = build_fixture("text-completion", "text_completion")?;
    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;

    // The reference, taken twice: a lane alone is deterministic, and a claim
    // about interference is only available once that is established.
    let alone = completion(&client, common::SERVING_PROMPT, TOKENS).await?;
    let again = completion(&client, common::SERVING_PROMPT, TOKENS).await?;
    eprintln!("[mixed-class] alone: {alone:?}");
    assert_eq!(alone, again, "the probe is not deterministic on its own");

    // Now the same lane with prefilling neighbours launched beside it. The
    // neighbours' own answers are not asked about — only the probe's.
    for round in 0..3 {
        let mut probes = Vec::new();
        for _ in 0..3 {
            probes.push(start(&client, common::SERVING_PROMPT, TOKENS).await?);
        }
        let mut others = Vec::new();
        for _ in 0..4 {
            others.push(start(&client, NEIGHBOUR_PROMPT, TOKENS).await?);
        }
        let mut answers = Vec::new();
        for proc in probes.iter_mut() {
            answers.push(said(proc).await?);
        }
        for proc in others.iter_mut() {
            let _ = said(proc).await?;
        }
        for (at, answer) in answers.iter().enumerate() {
            eprintln!("[mixed-class] round {round} probe {at}: {answer:?}");
            assert_eq!(
                answer, &alone,
                "probe {at} of round {round} said {answer:?} beside a prefilling \
                 neighbour and {alone:?} alone — a lane's answer must not \
                 depend on the composition of the fire it rode in"
            );
        }
    }

    // ── **THE SAMPLER'S END OF THE SAME BUG**, in this boot and not another:
    //    the runtime takes the device once per process, so the second claim is
    //    the second half of this test rather than a second test.
    //
    //    `dry-repetition-penalty` carries its whole token history in a device
    //    channel and reports how many vocabulary entries its penalty charged.
    //    Its own guest refuses a run in which nothing was ever penalized,
    //    because a stuck history channel would look exactly like that — and
    //    under a heterogeneous load it fired that refusal about once in fifty
    //    runs while never firing once in seventy-two homogeneous ones. It was
    //    never the history channel. A neighbour prefilling over its KV pages
    //    moved its sampled tokens, an unrepetitive continuation has nothing to
    //    penalize, and the guest said so about the symptom it could see.
    //
    //    So the claim made here is the strong one and not the guest's: the
    //    whole report is identical to the solo run's — statistics, text and
    //    all — which fails on a moved token long before it fails on a zero.
    let (dry_wasm, dry_manifest) =
        build_fixture("dry-repetition-penalty", "dry_repetition_penalty")?;
    client
        .add_program(&dry_wasm, &dry_manifest, true)
        .await
        .context("add_program dry-repetition-penalty")?;

    async fn dry(client: &Client) -> Result<Process> {
        let input = serde_json::json!({ "max_tokens": TOKENS, "multiplier": 0.8 }).to_string();
        client
            .launch_process("dry-repetition-penalty@0.1.0".to_string(), input, true)
            .await
            .context("launch dry-repetition-penalty")
    }

    let mut solo = dry(&client).await?;
    let dry_alone = said(&mut solo).await?;
    eprintln!("[dry-beside] alone: {dry_alone:?}");

    for round in 0..3 {
        let mut probes = Vec::new();
        for _ in 0..3 {
            probes.push(dry(&client).await?);
        }
        let mut others = Vec::new();
        for _ in 0..4 {
            others.push(start(&client, NEIGHBOUR_PROMPT, TOKENS).await?);
        }
        let mut answers = Vec::new();
        for proc in probes.iter_mut() {
            answers.push(said(proc).await?);
        }
        for proc in others.iter_mut() {
            let _ = said(proc).await?;
        }
        for (at, answer) in answers.iter().enumerate() {
            eprintln!("[dry-beside] round {round} probe {at}: {answer:?}");
            assert_eq!(
                answer, &dry_alone,
                "dry probe {at} of round {round} reported {answer:?} beside a \
                 prefilling neighbour and {dry_alone:?} alone"
            );
        }
    }
    Ok(())
}

