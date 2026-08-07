//! The declared drive is token-identical to the hand-written pass, for
//! the families whose gate is still opt-in: gemma-4 and gpt-oss.
//!
//! `cuda_declared_forward_parity` is the same claim for llama_like, and
//! this file differs from it in one way that matters. That harness
//! compares ONE sampled stream across the two invocations, which is
//! legitimate for qwen-3-0.6b because a single-request fire there is
//! reproducible. It is NOT legitimate here:
//!
//! * gpt-oss's first run after boot answered one seed differently from
//!   its next five, on BOTH sides of the gate — a warm-up artifact of
//!   the hand-written pass that the declared drive faithfully
//!   reproduces. A one-sample harness comparing run 1 against run 2
//!   would have failed a correct drive;
//! * gemma-4 through a chat inferlet flips a tail token between runs of
//!   one process, so a single sample is not even self-consistent.
//!
//! So each side is sampled `SAMPLES` times and the claim is that the two
//! SEQUENCES match — the distribution including its quirks, which is a
//! strictly stronger statement than one sample and the only one these
//! deployments can support. A drive that agreed on the steady state but
//! not on the warm-up would fail here, and should.
//!
//! Both observations above come from ad-hoc harnesses, not from this
//! one: on the runs that landed this file every sample was stable on
//! both sides for both families. The sampling is insurance against a
//! flake that has been seen rather than a reproduction of one, and it
//! costs three extra 48-token generations against a model that is
//! already booted.
//!
//! `#[ignore]`, driver-cuda, and each family needs its checkpoint in the
//! HF cache. Run both polarities; the second invocation gates:
//!
//! ```text
//!   cargo test -p pie-gpu-tests --release --no-default-features \
//!     --features driver-cuda --test cuda_declared_family_parity \
//!     -- --ignored --nocapture
//!   PIE_DECLARED_FORWARD_GEMMA4=1 PIE_DECLARED_FORWARD_GPT_OSS=1 \
//!     cargo test ... --test cuda_declared_family_parity -- --ignored --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use pie_bin::derive::derive_standalone;
use pie_bin::run_standalone;
use pie_client::client::Client;

/// How many times each side is sampled. Four is enough to separate a
/// warm-up artifact (run 1 differs, runs 2..N agree) from a real
/// disagreement, which is the shape both families actually exhibit.
const SAMPLES: usize = 4;
const MAX_TOKENS: usize = 48;

/// One family under test: its checkpoint, and the env var that arms its
/// drive.
struct Family {
    name: &'static str,
    hub_dir: &'static str,
    gate: &'static str,
}

const GEMMA4: Family = Family {
    name: "gemma4",
    hub_dir: "models--google--gemma-4-E4B-it",
    gate: "PIE_DECLARED_FORWARD_GEMMA4",
};

/// gemma-4's SECOND geometry, and the reason it earns a row: 35 layers
/// (odd), `kv_heads = 1` (MQA, where E4B has 2), 20 of 35 layers KV-shared,
/// hidden 1536. The facts derivation's interval reduction, its trailing
/// KV-shared run and the arms' per-layer head widths all get a different
/// set of numbers than E4B can give them.
const GEMMA4_E2B: Family = Family {
    name: "gemma4_e2b",
    hub_dir: "models--google--gemma-4-E2B-it",
    gate: "PIE_DECLARED_FORWARD_GEMMA4",
};

const GPT_OSS: Family = Family {
    name: "gpt_oss",
    hub_dir: "models--openai--gpt-oss-20b",
    gate: "PIE_DECLARED_FORWARD_GPT_OSS",
};

/// A checkpoint snapshot in the HF cache. Unlike the qwen resolver this
/// does not require `model.safetensors` by name: both of these ship
/// sharded weights.
fn resolve_snapshot(hub_dir: &str) -> Result<String> {
    let hub = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/huggingface")
        })
        .join("hub")
        .join(hub_dir)
        .join("snapshots");
    let snap = std::fs::read_dir(&hub)
        .with_context(|| format!("{hub_dir} not in the HF cache at {}", hub.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").exists())
        .with_context(|| format!("no snapshot with a config.json under {}", hub.display()))?;
    Ok(snap.to_string_lossy().into_owned())
}

fn record_path(family: &str, declared: bool) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pie_declared_family_parity_{family}_{}.txt",
        if declared { "on" } else { "off" }
    ))
}

/// The sampled text out of naive-baseline's JSON result.
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

async fn run_family(family: &Family) -> Result<()> {
    common::init_trace();
    // MIRROR the driver's own polarity: these two gates are opt-in, so
    // an UNSET env means the hand-written pass. Reading it the other way
    // would file both invocations under one slot and compare a run
    // against itself — a gate that passes because it stopped asking.
    let declared = std::env::var(family.gate)
        .map(|v| !v.is_empty() && !v.starts_with('0'))
        .unwrap_or(false);

    let snapshot = resolve_snapshot(family.hub_dir)?;
    let toml = common::cuda_standalone_toml(&snapshot);
    let (controller, gateway, worker) = derive_standalone(&toml)?;
    let pie = run_standalone(controller, gateway, worker).await?;
    eprintln!(
        "[family-parity/{}] booted listen={} declared={declared}",
        family.name, pie.listen_addr
    );

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
    client.authenticate("test-user", &None).await.context("auth")?;
    client
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;

    // Single request, fixed seed, sampled SAMPLES times. Same request
    // every time: what varies between runs is the deployment's own
    // warm-up, and reproducing THAT is the claim.
    let input = format!(
        "{{\"prompt\": \"The old clockmaker examined the strange timepiece \
         carefully\", \"max_tokens\": {MAX_TOKENS}, \"seed\": 7}}"
    );
    let mut texts = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        let mut proc = client
            .launch_process("naive-baseline@0.1.0".to_string(), input.clone(), true)
            .await
            .context("launch")?;
        let json = proc.wait_for_return().await.context("wait_for_return")?;
        let text = parse_text(&json).with_context(|| format!("no text in result: {json}"))?;
        eprintln!(
            "[family-parity/{}] sample {i}: {:?}",
            family.name,
            &text[..text.len().min(60)]
        );
        texts.push(text);
    }
    // A record is the whole SEQUENCE; `\u{1}` cannot occur in sampled
    // text, so it is a safe joiner.
    let record = texts.join("\u{1}");
    std::fs::write(record_path(family.name, declared), &record).context("write record")?;

    if let Ok(counterpart) = std::fs::read_to_string(record_path(family.name, !declared)) {
        let theirs: Vec<&str> = counterpart.split('\u{1}').collect();
        let ours: Vec<&str> = record.split('\u{1}').collect();
        anyhow::ensure!(
            ours == theirs,
            "{}: declared-vs-handwritten sampled sequences diverge.\n  \
             this run (declared={declared}): {ours:#?}\n  counterpart: \
             {theirs:#?}\nA declared executor that calls the same kernels \
             in the same order must match sample for sample — including \
             any warm-up difference, which both sides should have.",
            family.name
        );
        eprintln!(
            "[family-parity/{}] PASS: {SAMPLES} samples byte-identical across the gate",
            family.name
        );
    } else {
        eprintln!(
            "[family-parity/{}] recorded; run the counterpart invocation ({}={}) to compare",
            family.name,
            family.gate,
            if declared { "0" } else { "1" }
        );
    }

    pie.shutdown().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a CUDA GPU + gemma-4-E4B; run gate-OFF then gate-ON"]
async fn gemma4_declared_forward_parity() -> Result<()> {
    run_family(&GEMMA4).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a CUDA GPU + gpt-oss-20b; run gate-OFF then gate-ON"]
async fn gpt_oss_declared_forward_parity() -> Result<()> {
    run_family(&GPT_OSS).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a CUDA GPU + gemma-4-E2B; run gate-OFF then gate-ON"]
async fn gemma4_e2b_declared_forward_parity() -> Result<()> {
    run_family(&GEMMA4_E2B).await
}
