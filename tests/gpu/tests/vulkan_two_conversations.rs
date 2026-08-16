//! Two conversations at once, and neither one reads the other's keys.
//!
//! `vulkan_chat_completion_e2e` proves a sentence comes back. It cannot prove
//! this: one conversation's page table is an identity map onto itself, so a
//! driver that ignores where the pool actually PUT its pages is consistent
//! with itself and answers perfectly.
//!
//! The pages a fire names are its own working set's -- page 0 is the first
//! page of THIS conversation -- and `FrameSubmission::kv_translation` says
//! which pool page each one landed in. A second conversation's page 0 is not
//! the first one's, and a driver that skips the translation gives them both
//! pool page 0: the second writes its keys over the first's and reads them
//! back as its own history. Nothing errors. The answer stays fluent, because
//! attention over the wrong keys is still attention, and it is wrong about a
//! prompt the other conversation asked.
//!
//! So this asks two questions at the same time, and each has a one-word right
//! answer the other cannot produce.
//!
//! # What it is not
//!
//! It is not a scheduling test. Whether the two land in one frame or in two is
//! the scheduler's business and this does not assert either; what it needs is
//! only that both are alive at once, which two launches and two awaits give.
//! With the translation dropped it fails, and it fails in the exact shape the
//! paragraph above predicts: the conversation that asked about France answers
//!
//! ```text
//! <think> Okay, let's tackle this question. The user is asking about the
//! largest planet in the solar system.
//! ```
//!
//! -- fluent, confident, and about the other conversation's prompt.
//!
//! ```text
//! PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_two_conversations -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// A prompt, and the words that say WHICH history was attended.
///
/// Both are needed and neither is the answer to a quiz. `want` is something
/// only this conversation's own pages could produce, and `reject` is
/// something only the other's could -- a generated continuation that satisfies
/// both is one that read the right keys. Asking for a correct answer instead
/// would gate this on the model rather than on the driver.
struct Ask {
    prompt: &'static str,
    want: &'static str,
    /// A word from the OTHER conversation's subject, which is what reading its
    /// pages sounds like.
    reject: &'static str,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn two_conversations_do_not_read_each_others_pages() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-two] booted, listen_addr={}", pie.listen_addr);

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

    // Long enough that each conversation owns SEVERAL pages: a collision in
    // page 0 alone could be missed if the answer only ever attends the last
    // few tokens, and a prompt that spans pages puts the words that matter
    // where the other conversation is writing.
    let asks = [
        Ask {
            prompt: "I am planning a trip and I keep forgetting the details, so let me \
                     write them down. The country I am visiting is France, the language \
                     spoken there is French, and the currency is the euro. The capital \
                     city of France is",
            want: "paris",
            reject: "planet",
        },
        Ask {
            prompt: "I am studying for an astronomy exam and I keep forgetting the \
                     details, so let me write them down. The planet I am studying is a \
                     gas giant, it has a great red spot, and it has many moons. The \
                     largest planet in the solar system is",
            want: "gas giant",
            reject: "france",
        },
    ];

    // Forty-eight, because this model opens with a `<think>` preamble and a
    // shorter budget ends inside it -- the answer is never reached and the
    // gate fails for a reason that has nothing to do with pages.
    //
    // Both launched before either is awaited, which is the whole point: two
    // conversations that never overlap cannot collide.
    let mut running = Vec::new();
    for ask in &asks {
        let input = serde_json::json!({
            "prompt": ask.prompt,
            "system": "You are a helpful assistant. Answer with a single word.",
            "max_tokens": 48,
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
    eprintln!("[vulkan-two] both conversations launched");

    let mut answers = Vec::new();
    for proc in &mut running {
        answers.push(proc.wait_for_return().await.context("wait_for_return")?);
    }
    pie.shutdown().await;

    for (ask, out) in asks.iter().zip(&answers) {
        eprintln!("[vulkan-two] {:?} -> {out:?}", ask.want);
        let text = out.to_lowercase();
        anyhow::ensure!(
            text.contains(ask.want),
            "a conversation did not attend its own prompt (expected `{}`): {out:?}",
            ask.want
        );
        anyhow::ensure!(
            !text.contains(ask.reject),
            "a conversation answered the OTHER one's question (`{}`), which is what \
             reading its pages looks like: {out:?}",
            ask.reject
        );
    }
    eprintln!("[vulkan-two] GREEN — two conversations, two histories");
    Ok(())
}
