//! The Vulkan standalone, booted and asked for tokens.
//!
//! Everything below `pie serve` has been measured on this card for several
//! milestones -- the kernels, the driver, the engine seam, and a real
//! checkpoint staged through it -- and none of that says the composition root
//! can reach any of it. This does: `run_standalone` boots the embedded
//! controller, gateway and worker with `[driver] type = "vulkan"`, and the
//! turn goes in through the same HTTP edge a deployment is reached by.
//!
//! # What it needs, and why it is `#[ignore]`
//!
//! A Vulkan device, the compiled SPIR-V, and an artifact `pie model build
//! --backend vulkan` authored from a pre-quantized MLX checkpoint -- an
//! artifact rather than a snapshot because this driver reads its declared
//! quantization out of the embedded `model/config`, and one carrying its
//! tokenizer because the runtime parses one at boot unconditionally.
//!
//! ```text
//! PIE_KERNELS_VULKAN_SPV_DIR=<abs>/out/spv PIE_VULKAN_ARTIFACT=/tmp/q4full.zt \
//!   cargo test -p pie-gpu-tests --features driver-vulkan \
//!   --test vulkan_boot_smoke -- --ignored --nocapture
//! ```
//!
//! # What it does not cover
//!
//! The inferlet this turn runs is an echo: it answers from the guest without
//! firing the model. So this gate says the composition root stands up and the
//! edge round-trips, and says nothing about a forward. The forward is gated
//! by `vulkan_chat_completion_e2e`.
//!
//! One boot per process, as every test here: the runtime owns process-global
//! singletons.

#![cfg(feature = "driver-vulkan")]

mod common;

use anyhow::Result;

/// A turn through the real edge, answered by a standalone the Vulkan driver
/// is embedded in.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Vulkan device, the compiled SPIR-V, and a built artifact"]
async fn the_vulkan_standalone_boots_and_answers_a_turn() -> Result<()> {
    let pie = common::boot_vulkan().await?;

    assert_ne!(pie.listen_addr.port(), 0, "the client edge binds a port");
    assert_ne!(pie.worker_addr.port(), 0, "the worker dial-in binds a port");
    assert!(
        pie.listen_addr.ip().is_loopback() && pie.worker_addr.ip().is_loopback(),
        "a standalone is loopback-only"
    );

    let payload = serde_json::to_vec(&serde_json::json!({ "type": "ping", "corr_id": 1 }))?;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/generate", pie.listen_addr))
        .header("x-pie-identity", "vulkan-smoke/test")
        .header("content-type", "application/json")
        .body(payload)
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "ingress accepts the turn");
    let body = resp.text().await?;
    assert!(
        body.contains("[DONE]"),
        "the turn streams to [DONE]: {body}"
    );

    pie.shutdown().await;
    Ok(())
}
