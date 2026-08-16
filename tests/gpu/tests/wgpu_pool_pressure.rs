//! A pool too small for all of them, and every conversation still answered.
//!
//! `wgpu_many_conversations` runs with five hundred pages -- eight thousand
//! tokens -- which no conversation in this directory comes close to. Nothing
//! is ever reused there: each working set is handed fresh pages and keeps
//! them to the end, so it proves that eight histories stay apart and says
//! nothing about what happens when the pool has to hand a page back.
//!
//! This boots with twenty-four pages and four conversations of about nine
//! pages each. The pool cannot hold them all, so its pages must be returned
//! and given to somebody else while the run is still going.
//!
//! A stale page is the failure that looks like nothing. The keys of a
//! conversation that finished are still sitting in the pages it returned, so
//! a driver that hands a page over without the new working set owning it
//! reads real text rather than zeros, and answers fluently about somebody
//! else's prompt. The codes are what make that visible: an answer holding a
//! NEIGHBOUR's code names the conversation whose keys it read.
//!
//! # What this reaches, and what it therefore does not claim
//!
//! `Shell::admit` has three answers and a small pool reaches one of them.
//! It GROWS the pool rather than refusing when a frame needs more than the
//! book holds, so `Launched::Exhausted` is not on this path, and
//! `Launched::Impossible` means past what the adapter could ever bind. The
//! engine sizes its own demand to the pool it was told about and does not
//! post a frame that will not fit, which is why the sibling gate measured all
//! three and saw none.
//!
//! So what this proves is the RECYCLING -- thirty-six pages of demand served
//! by twenty-four pages of memory -- and not the re-post path, and it does not
//! assert how the four were serialised, which is the scheduler's business.
//!
//! ```text
//! PIE_HOME=/path/to/piehome cargo test -p pie-gpu-tests --features driver-wgpu \
//!   --test wgpu_pool_pressure -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-wgpu")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// Pages in the pool.
///
/// Each conversation below is around a hundred tokens of prompt plus forty
/// generated, which is nine pages of sixteen. So one fits with room to spare
/// and four do not, which is the condition this file exists to create. It has
/// to stay well above ONE conversation's demand: a frame past the pool's
/// ceiling is `Impossible` and the request fails rather than waiting.
const KV_PAGES: u32 = 24;

/// The codes, distinct in their digits so no match is half-right.
const CODES: [&str; 4] = ["4271", "8163", "3948", "5602"];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a WebGPU adapter and a model in $PIE_HOME's cache"]
async fn a_pool_too_small_for_all_of_them_still_answers_each_one() -> Result<()> {
    common::init_trace();
    let pie = common::boot_wgpu_with_pages(KV_PAGES).await?;
    eprintln!(
        "[wgpu-pool] booted with {KV_PAGES} pages, {}",
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
    eprintln!("[wgpu-pool] {} conversations launched", CODES.len());

    let mut answers = Vec::new();
    for proc in &mut running {
        answers.push(proc.wait_for_return().await.context("wait_for_return")?);
    }
    pie.shutdown().await;

    let mut wrong = Vec::new();
    for (i, (code, out)) in CODES.iter().zip(&answers).enumerate() {
        eprintln!("[wgpu-pool] {code} -> {out:?}");
        if !out.contains(code) {
            wrong.push(format!(
                "{i} was given {code} and did not repeat it: {out:?}"
            ));
        }
        // Naming WHICH other one it read: under recycling that is the
        // conversation whose returned pages this one was handed.
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
        "{} of {} conversations answered from recycled pages that were not theirs:\n  {}",
        wrong.len(),
        CODES.len(),
        wrong.join("\n  ")
    );
    eprintln!(
        "[wgpu-pool] GREEN — {} conversations over {KV_PAGES} pages",
        CODES.len()
    );
    Ok(())
}
