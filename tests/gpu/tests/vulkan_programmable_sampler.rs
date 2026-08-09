//! A sampler the GUEST wrote, driving a decode on Vulkan.
//!
//! Every other Vulkan gate here samples the way the engine samples: the
//! inferlet names a temperature and a nucleus, and the distribution never
//! leaves the driver. This one is a different plane. `mirostat-v2-sampling`
//! builds its own `ForwardPass` -- embed, attention with an explicit KV
//! geometry, the read-out -- reads the logits back through a CHANNEL, decides
//! the token host-side, and feeds a control value forward into the next step
//! through two more channels. The driver's part is the channel plane:
//! `register_program`, `register_channel`, `bind_instance`, and then one
//! instance fired per step over that request's own distribution.
//!
//! # Why it is worth its own gate
//!
//! Until this ran, none of that had ever been exercised end to end on this
//! backend. The seam's registry had unit coverage -- a program, a channel and
//! an instance registered and closed, now without a device at all -- and that
//! is coverage of the REGISTRY, not of the plane. A registry that hands back
//! well-formed ids for a program nobody fires is indistinguishable from one
//! that works, and the failure it hides is not a wrong word: it is a channel
//! that never fills, which is a decode loop that waits forever.
//!
//! The sibling proof on CUDA (`programmable_sampler_4090`) cannot stand in for
//! it. It is a different driver, a different lowering, and its own inferlets
//! (`crates/engine/tests/inferlets`) no longer exist in this tree, so it does
//! not build today.
//!
//! # What it asserts
//!
//! That the loop RAN and that the feedback FLOWED, and nothing about
//! convergence. Mirostat's own CUDA gate documents why: on the corrected
//! 151936 vocabulary the natural surprise ceiling is about 1.79 nats, so a
//! target above it is unreachable and a target below it can fall into a
//! repetition attractor. Neither is a driver fact. What IS a driver fact is
//! that every step's logits came back, every step's control value went
//! forward, and the requested number of tokens was produced:
//!
//! * `count` equals the tokens asked for -- a channel that stalls gives fewer;
//! * `final_mu` differs from the `mu0` this test states -- the ONLY way it can
//!   move is a surprise read off a real distribution and fed back, so this is
//!   the round trip's signature;
//! * every reported number is finite -- a channel read off the wrong offset
//!   gives a NaN here rather than an error;
//! * the text is non-empty.
//!
//! The `mu` claim is the one worth controlling, and it can be controlled
//! without touching the driver: run the same inferlet with a learning rate of
//! `1e-9` and the feedback is still computed but can no longer move `mu`,
//! which is what a plane that returned nothing would look like from here. It
//! fails, naming the number that did not move. (A rate of exactly zero does
//! not work as a control -- the inferlet rejects it before it decodes.)
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_programmable_sampler -- --ignored --nocapture
//! ```
//!
//! One boot per process, so this is its own test binary.

#![cfg(feature = "driver-vulkan")]

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use client::client::Client;

/// What the inferlet answers with. Its own schema, read here by the names it
/// states rather than by position.
#[derive(Debug, serde::Deserialize)]
struct Ran {
    sampler: String,
    text: String,
    count: usize,
    final_mu: f32,
    mean_surprise: f32,
    tail_mean_surprise: f32,
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn a_guest_written_sampler_drives_a_decode() -> Result<()> {
    common::init_trace();
    let pie = common::boot_vulkan().await?;
    eprintln!("[vulkan-sampler] booted, listen_addr={}", pie.listen_addr);

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/inferlets");
    let dir = workspace.join("mirostat-v2-sampling");
    let ok = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-wasip2",
            "-p",
            "mirostat-v2-sampling",
        ])
        .current_dir(&workspace)
        .status()?
        .success();
    anyhow::ensure!(ok, "mirostat-v2-sampling wasm build failed");
    let wasm = workspace.join("target/wasm32-wasip2/debug/mirostat_v2_sampling.wasm");
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

    // `mu0` is STATED rather than defaulted. The inferlet's default is
    // `ln(vocab) + 1`, which is a number this test would have to re-derive
    // from the checkpoint to compare against -- and a comparison against a
    // number derived the same way the subject derives it is not a comparison.
    const MU0: f32 = 6.0;
    const TOKENS: usize = 8;
    let input = serde_json::json!({
        "prompt": "Write one sentence about the sea.",
        "tau": 3.0,
        "mu0": MU0,
        "max_tokens": TOKENS,
    })
    .to_string();

    let mut proc = client
        .launch_process("mirostat-v2-sampling@0.1.0".to_string(), input, true)
        .await
        .context("launch")?;
    // Bounded rather than awaited: a channel that never fills is a wait with
    // no end, and that is the failure this gate exists to catch. Measured at
    // about 20 seconds for eight tokens, including the wasm build.
    let out = tokio::time::timeout(std::time::Duration::from_secs(300), proc.wait_for_return())
        .await
        .context("a guest-written sampler did not return within 300s")?
        .context("wait_for_return")?;
    eprintln!("[vulkan-sampler] -> {out}");

    let ran: Ran = serde_json::from_str(&out)
        .with_context(|| format!("the inferlet's answer is not its own schema: {out}"))?;
    anyhow::ensure!(
        ran.sampler == "mirostat-v2",
        "another sampler answered: {}",
        ran.sampler
    );
    anyhow::ensure!(
        ran.count == TOKENS,
        "the loop produced {} of {TOKENS} tokens, so a channel stopped filling",
        ran.count
    );
    anyhow::ensure!(
        !ran.text.trim().is_empty(),
        "the sampler chose {} tokens that decoded to nothing",
        ran.count
    );
    anyhow::ensure!(
        ran.final_mu.is_finite()
            && ran.mean_surprise.is_finite()
            && ran.tail_mean_surprise.is_finite(),
        "a non-finite number came back ({ran:?}), which is what a channel read \
         at the wrong offset looks like from here"
    );
    // The signature of the round trip. `mu` moves only by `lr * (s - tau)`,
    // and `s` is a surprise computed from logits that came back over a
    // channel -- so a driver that filled nothing, or filled the same cell
    // every step, leaves this exactly where the input put it.
    anyhow::ensure!(
        (ran.final_mu - MU0).abs() > 1e-3,
        "mu never moved from the {MU0} it started at, so no surprise was fed \
         back: the channel plane returned nothing the guest could read"
    );
    anyhow::ensure!(
        ran.mean_surprise > 0.0,
        "every step reported zero surprise, so the logits behind them were a \
         constant rather than a distribution"
    );

    pie.shutdown().await;
    eprintln!(
        "[vulkan-sampler] GREEN — {} tokens, mu {MU0} -> {}, mean surprise {}",
        ran.count, ran.final_mu, ran.mean_surprise
    );
    Ok(())
}
