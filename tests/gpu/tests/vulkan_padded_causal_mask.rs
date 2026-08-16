//! A mask this driver CAN serve, and one it cannot, from the same inferlet.
//!
//! `contrastive-decoding` runs two models against one prompt and builds its
//! own attention mask for the amateur pass, as a rectangle: one row per query,
//! `pool_len` cells wide, true where `key <= query && key + window > query`.
//! Its window is a parameter, and that one number decides which of the two
//! answers this driver must give:
//!
//! * opened past the pool, the second clause is vacuous and the mask reduces
//!   to `key <= query` -- causal, padded with false to the pool's width. This
//!   driver must SERVE it, because dropping cells that are already false
//!   drops nothing;
//! * narrowed to 1, the mask is a real restriction -- each query sees one key,
//!   its own -- and this driver must serve that too, and run.
//!
//! One inferlet, one parameter, both served. The two cases differ in the
//! mask's CONTENT and in nothing else, which is exactly the distinction the
//! driver was not making when it refused the second.
//!
//! # The second half used to be a refusal
//!
//! It was, for as long as a real window was outside what this driver could
//! express: `Pool::stage` shipped a rectangle of zeros and `sdpa_paged_decode`
//! read the model text's literal stride, so a window could only be refused by
//! name. Both halves are built now -- the pool stages the guest's own
//! allow-bytes, and `Source::AttentionMaskStride` answers the fire's pitch --
//! and this gate moved with them rather than being deleted, because what it
//! guards against is unchanged: a mask that reaches the shader as nothing.
//!
//! # What it was doing instead
//!
//! Refusing both. `frames::unserved_in` tested `plan.has_user_mask` as the
//! first half of its condition, so a guest that named a mask never had its
//! rows read; and `causal_row` compared the row's WIDTH against the query's
//! position, so a rectangle was refused for being a rectangle even when the
//! flag was not set. Row 0 of the wide-window run arrives as
//! `runs=[0, 1, 47]` over `total_size=48` -- one true cell, forty-seven false
//! ones -- which was measured off the wire under an `eprintln!` before either
//! was changed.
//!
//! Three inferlets were refused by that. This one, and the prefills of
//! `sliding-window-attention` and `attention-sink`, whose masks are `key <=
//! query` and nothing else. Those two now pass this driver and stop at an
//! engine wall further on, which is not this driver's to lift; this one runs
//! and answers.
//!
//! # Where the control went
//!
//! A driver that accepted masks and dropped them would pass both halves of
//! this gate, so neither half is the control. The control is the unit test
//! `resources::a_windowed_row_is_staged_as_that_window_and_padded_to_the_fires_pitch`,
//! which pins the rectangle this run stages -- byte for byte, at the fire's
//! pitch, with the enable byte set -- and is mutation-tested against both.
//!
//! That split is a measurement, not a preference. This gate first asked the
//! two runs to answer DIFFERENTLY, and they do not: with the window at 1 the
//! wire carries one allowed key per row, this driver decodes and stages
//! exactly that (both printed at every seam on a 4090), and the inferlet
//! still returns the same eight tokens. Contrastive decoding subtracts the
//! amateur from the expert, and this prompt's first eight tokens do not turn
//! on the difference. An assertion on the text would have pinned the model's
//! indifference and called it the driver's mask.
//!
//! ```text
//! PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_padded_causal_mask -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary. Both halves share it.

#![cfg(feature = "driver-vulkan")]

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
    tokio::time::timeout(std::time::Duration::from_secs(300), proc.wait_for_return())
        .await
        .context("contrastive decoding neither answered nor refused within 300s")?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn a_causal_mask_padded_to_the_pool_is_served_and_a_window_is_not() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-mask] booted, listen_addr={}", pie.listen_addr);

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

    // The half this driver must serve.
    let text = run(&client, OPEN).await.context(
        "a causal mask padded to the pool's width was not served, which is the \
         defect this gate exists for",
    )?;
    anyhow::ensure!(
        !text.trim().is_empty(),
        "the wide-window run returned nothing, so the pass ran and said no words"
    );
    eprintln!("[vulkan-mask] served: {text:?}");

    // The half that used to be refused, and is now served.
    //
    // This gate was written when a real window was outside what this driver
    // could express: `Pool::stage` shipped a rectangle of zeros and
    // `sdpa_paged_decode` read the model text's literal stride, so the only
    // honest answer was a refusal by name. Both halves are built now -- the
    // pool stages the guest's own allow-bytes and answers the fire's pitch
    // through `Source::AttentionMaskStride` -- so the refusal this half pinned
    // is gone, and running is the claim in its place.
    //
    // # Why this half no longer compares the two answers
    //
    // Because they are the same, and the mask is not why. Measured on a 4090
    // with the runs and the rectangle printed at every seam: at a window of 1
    // the wire carries `runs=[26, 1, 21]` per row -- one key allowed, its own
    // -- `frames::mask_rows_of` decodes exactly that, and `Frame::mask_from`
    // stages it as a single set byte at a pitch of 48. The mask arrives. The
    // inferlet still answers "<think>\nOkay, the user wants me" for a window
    // of 1 and for one of 100000, because contrastive decoding subtracts the
    // amateur's distribution from the expert's and this prompt's first eight
    // tokens do not turn on that difference.
    //
    // So an end-to-end comparison here would pin the model's indifference, not
    // the driver's staging. The staging is pinned where it can be READ, by
    // `resources::a_windowed_row_is_staged_as_that_window_and_padded_to_the_fires_pitch`,
    // which is the same rectangle this run produces and is mutation-tested
    // against both the pitch and the enable byte.
    let narrow = run(&client, NARROW)
        .await
        .context("a sliding window of 1 was refused, and this driver now serves one")?;
    anyhow::ensure!(
        !narrow.trim().is_empty(),
        "the narrow-window run returned nothing, so the pass ran and said no words"
    );
    eprintln!("[vulkan-mask] windowed: {narrow:?}");

    pie.shutdown().await;
    eprintln!("[vulkan-mask] GREEN — one inferlet, one parameter, both answers");
    Ok(())
}
