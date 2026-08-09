//! Two conversations that begin with the same words.
//!
//! `vulkan_two_conversations` proves two histories stay apart when they have
//! nothing in common. This asks the harder version: the same driver, over
//! prompts whose first several PAGES are identical token for token, which is
//! the input a prefix cache exists to collapse.
//!
//! # What was measured, and what is therefore not yet asserted
//!
//! The engine can share pages structurally -- `KvStore::adopt_cached_prefix`
//! grafts a cached prefix into an empty working set, and writes CoW off it --
//! but the probe that would CALL it, `pipeline::fire::kv::match_prefix`, is
//! not wired into the live path. Its own doc says so: "until then this is
//! exercised by the store tests only". So on this build the second
//! conversation recomputes the preamble into pages of its own, and no fire
//! this test produces names one page twice.
//!
//! That is worth writing down rather than assuming, because of what happens
//! when the wiring lands. `resources::Frame::of` refuses a frame in which two
//! requests name the same page -- `Unstageable::SharedPage` -- and for
//! independent conversations that refusal is exactly right: two of them
//! writing one page is the corruption the other gate is about. A grafted
//! prefix is the case where it is wrong. Those pages are READ by both and
//! written by neither, and "named twice" stops implying "written twice" the
//! day the probe is switched on. This driver will refuse a correct plan, in a
//! fault rather than in silence, and this file is where to start.
//!
//! # What it does assert today
//!
//! That a long identical preamble does not make two conversations answer each
//! other. Each is asked twice -- once alone, so the second launch has a warm
//! prefix to match if matching ever begins, and once with both in flight --
//! and each has a one-word answer the other cannot produce. It is the
//! two-conversation gate's claim over the input most likely to break it, and
//! it is also the gate that will turn red rather than quiet when sharing
//! arrives.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_shared_prefix -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// Several hundred tokens of text both conversations open with, word for word.
///
/// Length is the point. Prefix sharing is page-granular, so a preamble shorter
/// than a page can be shared in nothing at all and the test would assert
/// against a plan the engine never made.
const SHARED: &str = "The following is a reference sheet that I keep at my desk and reread \
     before every session, because the details are easy to mix up and getting them wrong is \
     expensive. It begins with the general rules. Measurements are always written in metric \
     units, dates are always written with the year first, and names are always written with \
     the family name last. Quantities smaller than one are written with a leading zero, and \
     quantities larger than a thousand are written with a separator every three digits. \
     Anything uncertain is marked with a question mark in the margin rather than guessed at, \
     and anything corrected is struck through rather than erased, so that the older reading \
     stays legible. The sheet is copied out by hand once a month, which is slow, but the \
     copying is how the rules are remembered. After the general rules come the specific \
     notes, and the specific notes are what I actually look up. They are grouped by subject, \
     each subject on its own line, and each line is short enough to read at a glance without \
     losing my place in whatever else I was doing at the time. Here are the notes for today.";

/// A continuation, and the word that says whose history was attended.
struct Ask {
    tail: &'static str,
    want: &'static str,
    reject: &'static str,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn a_shared_prefix_still_gives_each_conversation_its_own_answer() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-prefix] booted, listen_addr={}", pie.listen_addr);

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

    let asks = [
        Ask {
            tail: " The subject is geography: the country is France, the language is \
                   French, the currency is the euro. Question: what is the capital city \
                   of France? Answer with one word.",
            want: "paris",
            reject: "jupiter",
        },
        Ask {
            tail: " The subject is astronomy: the planet is a gas giant, it has a great \
                   red spot, and it has many moons. Question: what is the largest planet \
                   in the solar system? Answer with one word.",
            want: "jupiter",
            reject: "paris",
        },
    ];

    // The first is launched and AWAITED before the second is launched, which is
    // what gives the second one something to share: a prefix nobody has
    // computed yet cannot be reused. Then both are asked again, together, so a
    // frame that mixes them is possible too.
    let mut answers = Vec::new();
    for ask in &asks {
        let out = ask_once(&client, ask).await?;
        eprintln!("[vulkan-prefix] warm {:?} -> {out:?}", ask.want);
        answers.push(out);
    }

    let mut running = Vec::new();
    for ask in &asks {
        running.push(
            client
                .launch_process("chat-completion@0.1.0".to_string(), body(ask), true)
                .await
                .context("launch")?,
        );
    }
    eprintln!("[vulkan-prefix] both relaunched over the warm prefix");
    for proc in &mut running {
        answers.push(proc.wait_for_return().await.context("wait_for_return")?);
    }
    pie.shutdown().await;

    for (ask, out) in asks.iter().chain(asks.iter()).zip(&answers) {
        let text = out.to_lowercase();
        anyhow::ensure!(
            text.contains(ask.want),
            "a conversation did not attend its own tail (expected `{}`): {out:?}",
            ask.want
        );
        anyhow::ensure!(
            !text.contains(ask.reject),
            "a conversation answered the other one's question (`{}`) over a prefix \
             they share: {out:?}",
            ask.reject
        );
    }
    eprintln!("[vulkan-prefix] GREEN — one preamble, two answers, twice over");
    Ok(())
}

/// The launch body. Two hundred tokens, which is four times what the short
/// prompts need: this one hands the model a page of rules to think ABOUT, and
/// a budget that ends inside the `<think>` preamble fails the gate for a
/// reason that has nothing to do with pages.
fn body(ask: &Ask) -> String {
    serde_json::json!({
        "prompt": format!("{SHARED}{}", ask.tail),
        "system": "You are a helpful assistant. Answer with a single word.",
        "max_tokens": 200,
        "temperature": 0.0,
        "top_p": 0.95,
    })
    .to_string()
}

async fn ask_once(client: &Client, ask: &Ask) -> Result<String> {
    let mut proc = client
        .launch_process("chat-completion@0.1.0".to_string(), body(ask), true)
        .await
        .context("launch")?;
    proc.wait_for_return().await.context("wait_for_return")
}
