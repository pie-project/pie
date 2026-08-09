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
//! * at its default of 8, the mask is a real restriction that no kernel this
//!   lowering names would apply, so this driver must REFUSE it, by name, at
//!   the first launch.
//!
//! One inferlet, one parameter, both answers. That is why this gate is this
//! inferlet and not a fixture: the two cases differ in the mask's CONTENT and
//! in nothing else, which is exactly the distinction the driver was not
//! making.
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
//! # The refusal half is the control
//!
//! A fix that made the driver accept masks would pass the first half of this
//! test and be a silent correctness disaster -- attention outside its window,
//! fluently, with nothing said. So the second half asks for the same run with
//! the default window and requires it to FAIL, carrying this driver's own
//! sentence. Neither half is meaningful without the other.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
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
/// The inferlet's default, and a real restriction.
const NARROW: u32 = 8;

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

    // The half it must refuse, in this driver's own words. Without this, a fix
    // that simply stopped checking masks would pass the half above.
    let refused = run(&client, NARROW).await;
    let why = match refused {
        Ok(text) => anyhow::bail!(
            "a sliding window of {NARROW} was SERVED, and answered {text:?} -- \
             attention outside its window, fluently, with nothing said"
        ),
        Err(why) => format!("{why:#}"),
    };
    anyhow::ensure!(
        why.contains("does not serve a user mask"),
        "the window was refused, but for {why:?} rather than by the name of \
         the capability that is missing"
    );
    eprintln!("[vulkan-mask] refused, by name");

    pie.shutdown().await;
    eprintln!("[vulkan-mask] GREEN — one inferlet, one parameter, both answers");
    Ok(())
}
