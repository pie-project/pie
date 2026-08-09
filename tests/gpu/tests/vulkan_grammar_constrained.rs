//! A grammar the HOST holds, steering a Vulkan decode token by token.
//!
//! The two channel-plane gates next door prove two shapes of guest control.
//! `vulkan_programmable_sampler` reads a distribution back and decides the
//! token host-side; `vulkan_sampling_primitives` runs a whole op set inside
//! one epilogue and reads six channels out. This is the third shape, and the
//! only one where the host writes INTO the epilogue every step:
//! `json-schema-constrained-decoding` keeps a grammar matcher on the host,
//! and before each fire it puts that matcher's allowed-token mask into a
//! channel the epilogue reads. The read-out is `masked_argmax` -- the mask is
//! an OPERAND of the choice, not a filter applied to the answer afterwards.
//!
//! # Why it is worth its own gate
//!
//! Everything else that reaches this driver decides the token from the logits
//! alone. Here a per-step, request-sized, guest-authored buffer has to arrive
//! at the right dispatch, in the right layout, before the argmax reads it --
//! and it has to arrive AGAIN, with different contents, on the very next fire
//! of the SAME instance. That is the one thing a `Channel::put` per step can
//! get wrong in a way nothing else here would notice: a mask bound once and
//! then reused is still a legal decode, just a decode against a stale
//! grammar.
//!
//! It does not go unnoticed here, because the guest checks the driver's work
//! for us. `constraint.accept_tokens(&[token])` is called on every token that
//! comes back, and it FAILS if the token was not one the matcher allowed. So
//! a run that returns at all is a run in which every single token the driver
//! chose was legal under the mask the host had just written. There is no way
//! to pass this test with a stale mask, a zeroed mask, or an ignored one.
//!
//! # The control is inside the test
//!
//! The second half asks for an object whose required keys are `zqx` and
//! `wbn`. No prompt in this test mentions them, and no unconstrained
//! continuation of "generate a profile" produces them -- they are not words.
//! If the mask were dropped anywhere between the host's `put` and the
//! kernel's read, the model's own preference would win and the matcher would
//! reject the first token off the plan. Getting `{"zqx": N, "wbn": N}` back
//! is only possible if a buffer the host wrote is what the argmax ranged
//! over. That is the assertion this file exists to make, and it needs no
//! mutation of the driver to be meaningful.
//!
//! # What it does NOT assert
//!
//! That any schema terminates. The inferlet decodes greedily, and a JSON
//! grammar permits unbounded whitespace before a closing brace and unbounded
//! items in an unbounded array -- so greedy decoding has attractors that are
//! nothing to do with a driver. Both were measured here: the inferlet's own
//! default schema (`skills` with `minItems` and no `maxItems`) enumerates
//! skills past 256 tokens, and a single-property object reaches
//! `{\n  "age": 24` and then emits `\n  \t\t\t` forever, every repetition of
//! it grammar-legal. Both are guest facts. The schemas used here are the ones
//! measured to close, twice each.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_grammar_constrained -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary. Both runs share it.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;
use serde_json::Value;

/// The schema that asks for what the prompt asks for. Bounded on purpose:
/// two scalar properties and `additionalProperties: false`, which is the
/// shape measured to close rather than wander.
const NATURAL: &str = r#"{
    "type": "object",
    "properties": {
        "name": { "type": "string", "minLength": 1 },
        "age": { "type": "integer", "minimum": 0, "maximum": 150 }
    },
    "required": ["name", "age"],
    "additionalProperties": false
}"#;

/// The control. Same prompt, keys the model has no reason to choose.
const UNNATURAL: &str = r#"{
    "type": "object",
    "properties": {
        "zqx": { "type": "integer", "minimum": 0, "maximum": 9 },
        "wbn": { "type": "integer", "minimum": 0, "maximum": 9 }
    },
    "required": ["zqx", "wbn"],
    "additionalProperties": false
}"#;

const PROMPT: &str = "Generate a profile for a fictional software engineer named Alice.";
const MAX_TOKENS: usize = 96;

async fn generate(client: &Client, schema: &str) -> Result<Value> {
    let input = serde_json::json!({
        "prompt": PROMPT,
        "schema": schema,
        "max_tokens": MAX_TOKENS,
    })
    .to_string();

    let mut proc = client
        .launch_process(
            "json-schema-constrained-decoding@0.1.0".to_string(),
            input,
            true,
        )
        .await
        .context("launch")?;
    // Bounded rather than awaited: the depth-1 host round trip per token is
    // exactly the shape that hangs if a channel stops filling. Measured at
    // about 12 seconds for a closing object.
    let out = tokio::time::timeout(std::time::Duration::from_secs(300), proc.wait_for_return())
        .await
        .context("a grammar-constrained decode did not return within 300s")?
        .context("wait_for_return")?;
    eprintln!("[vulkan-grammar] -> {out}");

    // The inferlet parses this itself before returning, so a parse failure
    // here would mean the answer is not the answer it says it sends.
    serde_json::from_str(&out).with_context(|| format!("the answer is not JSON: {out}"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn a_host_held_grammar_steers_every_token() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-grammar] booted, listen_addr={}", pie.listen_addr);

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("json-schema-constrained-decoding");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "json-schema-constrained-decoding",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "json-schema-constrained-decoding wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/json_schema_constrained_decoding.wasm");
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

    // The schema the prompt agrees with. Reaching a terminated matcher at all
    // means every token was legal under the mask standing at the time.
    let natural = generate(&client, NATURAL).await?;
    let obj = natural
        .as_object()
        .with_context(|| format!("the schema asked for an object, got {natural}"))?;
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    anyhow::ensure!(
        keys == ["age", "name"],
        "the schema forbids additional properties, so these keys are ones no \
         mask allowed: {keys:?}"
    );
    let name = obj["name"]
        .as_str()
        .with_context(|| format!("`name` is not a string: {}", obj["name"]))?;
    anyhow::ensure!(
        !name.is_empty(),
        "`minLength: 1` was in the grammar, so an empty name is a mask that \
         was not consulted"
    );
    let age = obj["age"]
        .as_i64()
        .with_context(|| format!("`age` is not an integer: {}", obj["age"]))?;
    anyhow::ensure!(
        (0..=150).contains(&age),
        "`age` is {age}, outside the 0..=150 the grammar allows"
    );
    eprintln!("[vulkan-grammar] natural: name={name:?} age={age}");

    // The control, in the same boot and against the same prompt. These keys
    // exist nowhere but in the mask.
    let unnatural = generate(&client, UNNATURAL).await?;
    let obj = unnatural
        .as_object()
        .with_context(|| format!("the control schema asked for an object, got {unnatural}"))?;
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    anyhow::ensure!(
        keys == ["wbn", "zqx"],
        "the model was asked for `zqx` and `wbn` and answered with {keys:?}, \
         which is what a decode ranging over its own preference rather than \
         over the host's mask looks like"
    );
    for k in ["zqx", "wbn"] {
        let v = obj[k]
            .as_i64()
            .with_context(|| format!("`{k}` is not an integer: {}", obj[k]))?;
        anyhow::ensure!((0..=9).contains(&v), "`{k}` is {v}, outside 0..=9");
    }
    eprintln!("[vulkan-grammar] control: {unnatural}");

    pie.shutdown().await;
    eprintln!(
        "[vulkan-grammar] GREEN — a natural object and one only a host-written \
         mask can produce"
    );
    Ok(())
}
