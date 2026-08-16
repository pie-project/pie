//! The sampler's OP SET, computed on the device and checked against itself.
//!
//! `vulkan_programmable_sampler` proves the channel plane carries a value
//! back and a value forward: one scalar each way, eight times. This proves
//! something the round trip does not, which is that the ops at the far end
//! compute what their names say.
//!
//! `sampling-primitives` publishes six channels from ONE PTIR epilogue -- the
//! greedy token, the raw logits, a softmax, a log-softmax, an entropy, and a
//! top-p keep-mask -- and three of them are a full 151936-wide vocabulary
//! read back to the host. Then it checks them against each other, which is
//! the part that makes it a gate rather than a demo:
//!
//! * the token, the logits' argmax and the probabilities' argmax agree;
//! * `exp(log_probabilities[token])` equals `probabilities[token]` -- two
//!   independently computed device outputs meeting within 1e-4;
//! * the entropy is finite and inside `[0, ln(vocab)]`;
//! * the keep-mask is boolean, non-empty, and a DESCENDING PREFIX -- no
//!   dropped probability exceeds any kept one;
//! * the kept mass reaches `top_p` and is minimal in doing so, which is the
//!   exact `cummass_le` contract rather than a rounded restatement of it.
//!
//! Every one of those is the guest's own assertion, and every one of them
//! only runs if this driver delivered all six channels intact. A plane that
//! filled the wide ones short, or off by an offset, or in the wrong order,
//! fails one of them by name. That is why this gate's own body is thin: the
//! sharp claims belong beside the data, and re-stating them here would be a
//! second copy to keep in step.
//!
//! # What this file adds
//!
//! Non-vacuity. Each of the guest's checks passes trivially on a degenerate
//! distribution -- a one-hot has zero entropy, a nucleus of the whole
//! vocabulary is a descending prefix, and `exp(ln p) == p` holds for any `p`
//! at all. So the summary line is parsed and held to what a REAL forward pass
//! over real weights produces: a spread distribution, and a cut that cut.
//!
//! ```text
//! PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_sampling_primitives -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// The `top_p` the inferlet states. Not a parameter: it is a `const` in its
/// source, and this gate's mass claim is against that number.
const TOP_P: f32 = 0.9;

/// The vocabulary the served checkpoint has. Used only for an upper bound, so
/// a different checkpoint makes this looser rather than wrong.
const VOCAB: f32 = 151_936.0;

/// One `k=v` field of the inferlet's summary line, by name.
fn field(line: &str, key: &str) -> Result<f32> {
    let raw = line
        .split_whitespace()
        .find_map(|kv| kv.strip_prefix(&format!("{key}=")))
        .with_context(|| format!("no `{key}` in the inferlet's answer: {line}"))?;
    raw.parse::<f32>()
        .with_context(|| format!("`{key}` is not a number: {raw}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn the_sampler_primitives_agree_with_each_other() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!(
        "[vulkan-primitives] booted, listen_addr={}",
        pie.listen_addr
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("sampling-primitives");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "sampling-primitives",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "sampling-primitives wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/sampling_primitives.wasm");
    let manifest = dir.join("Pie.toml");
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
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;

    let mut proc = client
        .launch_process(
            "sampling-primitives@0.1.0".to_string(),
            "{}".to_string(),
            true,
        )
        .await
        .context("launch")?;
    // Bounded: three of the six channels are a vocabulary wide, and a plane
    // that fills them short is a wait rather than an error.
    let out = tokio::time::timeout(std::time::Duration::from_secs(300), proc.wait_for_return())
        .await
        .context("the primitives sweep did not return within 300s")?
        // Every one of the guest's cross-checks surfaces HERE, by its own
        // sentence, because a violated one returns an error rather than a
        // summary.
        .context("wait_for_return")?;
    eprintln!("[vulkan-primitives] -> {out}");

    let probability = field(&out, "probability")?;
    let log_probability = field(&out, "log_probability")?;
    let entropy = field(&out, "entropy")?;
    let kept = field(&out, "nucleus_kept")?;
    let mass = field(&out, "nucleus_mass")?;

    anyhow::ensure!(
        probability > 0.0 && probability <= 1.0,
        "the chosen token's probability is {probability}, which is not a probability"
    );
    // The same meeting the guest checks, computed the other way round: it
    // takes `exp` of the log, this takes `ln` of the probability. A device
    // that wrote one of the two channels from the other would satisfy both,
    // but a device that computed one of them wrongly satisfies neither.
    anyhow::ensure!(
        (probability.ln() - log_probability).abs() < 1e-3,
        "ln({probability}) is {}, and the device's log-probability is \
         {log_probability}",
        probability.ln()
    );
    anyhow::ensure!(
        entropy > 0.1 && entropy < VOCAB.ln(),
        "an entropy of {entropy} nats is not a real distribution: a one-hot \
         has zero, and every check the guest makes passes on one"
    );
    anyhow::ensure!(
        kept >= 1.0 && kept < VOCAB,
        "the nucleus kept {kept} of {VOCAB} tokens, so the cut did not cut"
    );
    anyhow::ensure!(
        (TOP_P..=1.0 + 1e-3).contains(&mass),
        "the nucleus holds {mass} of the mass against a top_p of {TOP_P}"
    );

    pie.shutdown().await;
    eprintln!(
        "[vulkan-primitives] GREEN — entropy {entropy} nats, nucleus {kept} tokens \
         holding {mass}"
    );
    Ok(())
}
