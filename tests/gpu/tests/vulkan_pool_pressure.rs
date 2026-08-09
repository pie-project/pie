//! A pool too small for all of them, and every conversation still answered.
//!
//! Every other gate here runs with 256 pages -- four thousand tokens -- which
//! no conversation in this directory comes close to. Nothing is ever reused:
//! each working set is handed fresh pages and keeps them to the end. This
//! boots with twenty-four, and four conversations of about nine pages each,
//! so the pool cannot hold them all and its pages must be handed back and
//! given to somebody else while the run is still going.
//!
//! A stale page is the failure that looks like nothing. The keys of a
//! conversation that finished are still sitting in the pages it returned, so a
//! driver that hands a page over without the new working set owning it reads
//! real text rather than zeros, and answers fluently about somebody else's
//! prompt.
//!
//! # What was measured, and what is therefore not claimed
//!
//! The driver's admission has three answers and this reaches ONE of them.
//! `Shell::admit` grows the pool rather than refusing when a frame needs more
//! than it holds, so `Exhausted` is not on this path at all, and `Impossible`
//! means past the ceiling of the card. Instrumenting both, and the growth
//! itself, showed none of the three: the engine sizes its own demand to the
//! pool it was told about and never posts a frame that does not fit.
//!
//! So what this proves is the recycling, which is what a small pool actually
//! produces -- thirty-six pages of demand served by twenty-four pages of
//! memory. It does not prove the re-post path, which the engine does not take,
//! and it does not assert HOW the four were serialised, which is the
//! scheduler's business.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_pool_pressure -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// Pages in the pool.
///
/// Each conversation below is around a hundred tokens of prompt plus forty
/// generated, which is nine pages of sixteen. So one fits with room to spare
/// and four do not, which is the condition this file exists to create. It must
/// stay well above ONE conversation's demand: a frame past the pool's ceiling
/// is `Impossible` and the request fails rather than waiting.
const KV_PAGES: u32 = 24;

/// The codes, distinct in their digits so no match is half-right.
const CODES: [&str; 4] = ["4271", "8163", "3948", "5602"];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn a_pool_too_small_for_all_of_them_still_answers_each_one() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan_with_pages(KV_PAGES).await?;
    eprintln!(
        "[vulkan-pool] booted with {KV_PAGES} pages, {}",
        pie.listen_addr
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("chat-completion");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "chat-completion",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "chat-completion wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/chat_completion.wasm");
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

    let mut running = Vec::new();
    for code in CODES {
        let prompt = format!(
            "I am reading out an access code and I need you to confirm it back to me. \
             Before the code, here is some context you can ignore: the room is on the \
             third floor, the door is blue, and the meeting is in the afternoon. The \
             access code is {code}. Please repeat the access code exactly."
        );
        let input = serde_json::json!({
            "prompt": prompt,
            "system": "You are a helpful assistant. Repeat the access code.",
            "max_tokens": 40,
            "temperature": 0.0,
            "top_p": 0.95,
        })
        .to_string();
        running.push(
            client
                .launch_process("chat-completion@0.1.0".to_string(), input, true)
                .await
                .context("launch")?,
        );
    }
    eprintln!("[vulkan-pool] {} conversations launched", CODES.len());

    let mut answers = Vec::new();
    for proc in &mut running {
        answers.push(proc.wait_for_return().await.context("wait_for_return")?);
    }
    pie.shutdown().await;

    let mut wrong = Vec::new();
    for (i, (code, out)) in CODES.iter().zip(&answers).enumerate() {
        eprintln!("[vulkan-pool] {code} -> {out:?}");
        if !out.contains(code) {
            wrong.push(format!(
                "{i} was given {code} and did not repeat it: {out:?}"
            ));
        }
        for (j, other) in CODES.iter().enumerate() {
            if j != i && out.contains(other) {
                wrong.push(format!(
                    "{i} repeated conversation {j}'s code {other}: {out:?}"
                ));
            }
        }
    }
    anyhow::ensure!(
        wrong.is_empty(),
        "{} of {} conversations answered wrongly under pool pressure:\n  {}",
        wrong.len(),
        CODES.len(),
        wrong.join("\n  ")
    );
    eprintln!(
        "[vulkan-pool] GREEN — {} conversations, {KV_PAGES} pages",
        CODES.len()
    );
    Ok(())
}
