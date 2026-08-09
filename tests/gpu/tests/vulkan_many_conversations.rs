//! Eight conversations, one pool, and every code comes back to its own.
//!
//! `vulkan_two_conversations` proves the page translation with the smallest
//! case that can see it. This is the case the pool itself is under: eight
//! working sets alive at once, each several pages, allocated and freed and
//! re-allocated while the others are running. What it adds over two:
//!
//! * frames with more than a couple of members, which is where a per-request
//!   table that is read with the wrong stride stops being invisible;
//! * pages RECYCLED -- a conversation that finishes returns its pages to the
//!   pool and a later one is handed them, so a stale read is a read of
//!   somebody's real keys rather than of zeros;
//! * `Launched::Exhausted`, if the pool is smaller than the demand, which the
//!   engine answers by re-posting the frame. That path is either taken here
//!   or it is not, and the test does not assert which -- what it asserts is
//!   that the answers are right either way.
//!
//! # Why codes and not questions
//!
//! Each conversation is given a made-up code and asked to repeat it. The
//! model cannot know one from world knowledge and cannot guess another
//! conversation's, so an answer holding the right code is an answer that
//! attended the right pages -- and one holding a NEIGHBOUR's code names the
//! conversation it read. A question with a real answer would gate this on the
//! checkpoint instead.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_many_conversations -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// How many at once. Enough that the scheduler batches them and the pool
/// recycles, small enough that the whole thing is one minute of GPU.
const CONVERSATIONS: usize = 8;

/// The codes, one per conversation.
///
/// Distinct in their DIGITS rather than in a prefix, so a match cannot be
/// half-right: `contains` on the digits alone is what the assertions use, and
/// no code is a substring of another.
const CODES: [&str; CONVERSATIONS] = [
    "4271", "8163", "3948", "5602", "7315", "2984", "6127", "9436",
];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn eight_conversations_keep_their_own_pages() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-many] booted, listen_addr={}", pie.listen_addr);

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
        // Padded past a single page on purpose: a conversation that fits in
        // one page can only ever collide in page 0, and the translation is a
        // per-page map.
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
    eprintln!("[vulkan-many] {CONVERSATIONS} conversations launched");

    let mut answers = Vec::new();
    for proc in &mut running {
        answers.push(proc.wait_for_return().await.context("wait_for_return")?);
    }
    pie.shutdown().await;

    let mut wrong = Vec::new();
    for (i, (code, out)) in CODES.iter().zip(&answers).enumerate() {
        eprintln!("[vulkan-many] {code} -> {out:?}");
        if !out.contains(code) {
            wrong.push(format!(
                "{i} was given {code} and did not repeat it: {out:?}"
            ));
        }
        // Naming WHICH other one it read, because that is the difference
        // between a driver that translates wrongly and one that does not
        // translate at all.
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
        "{} of {CONVERSATIONS} conversations answered from the wrong pages:\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
    eprintln!("[vulkan-many] GREEN — {CONVERSATIONS} conversations, {CONVERSATIONS} histories");
    Ok(())
}
