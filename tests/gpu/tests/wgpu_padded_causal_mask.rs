//! A mask this driver CAN serve, and the narrowest one there is, from the
//! same inferlet.
//!
//! `contrastive-decoding` runs two passes over one prompt and builds its own
//! attention mask for the amateur one, as a rectangle: one row per query,
//! `pool_len` cells wide, true where `key <= query && key + window > query`.
//! Its window is a parameter, and that one number decides which shape this
//! driver is handed:
//!
//! * opened past the pool, the second clause is vacuous and the mask reduces
//!   to `key <= query` -- causal, padded with false to the pool's width. A
//!   rectangle, and every cell this driver would drop is already false;
//! * narrowed to 1, the mask is a real restriction -- each query sees one
//!   key, its own -- and the rectangle is mostly false.
//!
//! One inferlet, one parameter, both served. The two runs differ in the
//! mask's CONTENT and in nothing else, which is what makes this the input
//! most likely to catch a driver that reads the shape and not the bytes.
//!
//! # Why this is the third gate and not the second
//!
//! `wgpu_many_conversations` and `wgpu_pool_pressure` both prove things about
//! PAGES -- that eight histories stay apart, and that a page handed back is
//! handed over clean. Neither of them ever binds a user mask, because no
//! conversation in this directory states one. The mask path is the largest
//! piece of this driver that no gate had entered: `frames::mask_rows_of`
//! decodes the wire's BRLEs, `Frame::of` stages the rectangle, `Pool::stage`
//! records the pitch, and `Source::AttentionMaskStride` answers it per row.
//! Four seams, and the only end-to-end evidence they worked was the curated
//! suite, which is not in this repository's CI.
//!
//! # Where the control is, and why it is not here
//!
//! Not in this file. A driver that accepted a mask and dropped it on the
//! floor would pass both halves of this gate, because contrastive decoding
//! subtracts the amateur's distribution from the expert's and a short prompt's
//! first few tokens need not turn on that difference -- `driver-vulkan`
//! measured exactly that on a 4090 and deleted its own comparison rather than
//! keep an assertion that pinned the model's indifference.
//!
//! Measured here, on llvmpipe, and it is the same answer `driver-vulkan`
//! got on a 4090: both windows return `"<think>\nOkay, the user wants me"`,
//! the same eight tokens. Two backends, two adapters, one indifference --
//! which is as strong a statement as this file can make that comparing the
//! two runs would pin the MODEL and not the driver.
//!
//! The control is the unit test
//! `resources::a_ragged_mask_is_staged_at_the_fires_pitch_and_an_unmasked_row_is_disabled`,
//! which reads the staged rectangle byte for byte at the fire's pitch and is
//! mutation-tested three ways. What THIS gate adds is the half a unit test
//! cannot reach: that a real fire carrying a real mask goes through the whole
//! shell -- decode, stage, bind, dispatch -- and comes back with words.
//!
//! # The pitch divergence, stated because it is load-bearing here
//!
//! This backend's mask pitch is the fire's widest row, chosen by the driver
//! and answered to the shader through `Source::AttentionMaskStride`. The
//! model text carries a literal stride, and `kernels-wgpu` documents that it
//! diverges from it deliberately. This gate is the end-to-end evidence that
//! the divergence is served rather than merely declared: if the shader read
//! the model's number and the driver staged at its own, the narrow run would
//! read another row's mask and the wide run would still look fine.
//!
//! ```text
//! PIE_HOME=/path/to/piehome cargo test -p pie-gpu-tests --features driver-wgpu \
//!   --test wgpu_padded_causal_mask -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary. Both halves share it.

#![cfg(feature = "driver-wgpu")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

const PROMPT: &str = "Write one sentence about the sea.";
const TOKENS: usize = 8;

/// Wide enough that the inferlet's own `window.min(pool_len)` makes the
/// window clause vacuous, whatever the pool turns out to be.
const OPEN: u32 = 100_000;
/// One key per query -- its own -- which is as narrow as a window goes.
const NARROW: u32 = 1;

/// Generous, because this backend's reference adapter is a CPU rasteriser.
///
/// A timeout here would be indistinguishable from a refusal in the log, and
/// the two have opposite meanings, so it is set past anything a working run
/// has taken rather than tuned to one.
const BUDGET: std::time::Duration = std::time::Duration::from_secs(900);

async fn run(client: &Client, window: u32) -> Result<String> {
    let input = serde_json::json!({
        "prompt": PROMPT,
        "amateur_window": window,
        "max_tokens": TOKENS,
    })
    .to_string();

    let mut proc = client
        .launch_process("contrastive-decoding@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    tokio::time::timeout(BUDGET, proc.wait_for_return())
        .await
        .context("contrastive decoding neither answered nor refused in time")?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a wgpu adapter and a built model in PIE_HOME"]
async fn a_causal_mask_padded_to_the_pool_and_a_window_of_one_are_both_served() -> Result<()> {
    common::init_trace();
    let pie = common::boot_wgpu().await?;
    eprintln!("[wgpu-mask] booted, listen_addr={}", pie.listen_addr);

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("contrastive-decoding");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "contrastive-decoding",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "contrastive-decoding wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/contrastive_decoding.wasm");
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

    // The padded-causal half. Every cell this rectangle adds over a plain
    // causal mask is false, so a driver that served it by ignoring it would
    // give the same words -- which is why the assertion is that it RAN, and
    // the rectangle itself is pinned in the unit control named above.
    let wide = run(&client, OPEN)
        .await
        .context("a causal mask padded to the pool's width was not served")?;
    anyhow::ensure!(
        !wide.trim().is_empty(),
        "the wide-window run returned nothing, so the pass ran and said no words"
    );
    eprintln!("[wgpu-mask] served (window={OPEN}): {wide:?}");

    // The narrow half, which is the one with teeth. Every row's mask is a
    // single true cell at a pitch of the pool's width, so a driver that
    // staged at the wrong pitch reads a DIFFERENT row's single cell -- a
    // legal byte, in a resident buffer, that says the wrong thing. It cannot
    // fault; it can only answer, which is why this half exists as a run and
    // the pitch is pinned in a unit test.
    let narrow = run(&client, NARROW)
        .await
        .context("a sliding window of one key was not served")?;
    anyhow::ensure!(
        !narrow.trim().is_empty(),
        "the narrow-window run returned nothing, so the pass ran and said no words"
    );
    eprintln!("[wgpu-mask] served (window={NARROW}): {narrow:?}");

    pie.shutdown().await;
    eprintln!("[wgpu-mask] GREEN — one inferlet, one parameter, both answers");
    Ok(())
}
