//! **THE STANDALONE BOOT SMOKE**: all three planes co-reside and the HTTP
//! ingress streams a turn.
//!
//! Boots the embedded controller + gateway + worker in one process over
//! loopback with the real CUDA engine (`type = "cuda_native"`) against the
//! shipping dense SKU, then drives one turn through `POST /v1/generate` — the
//! axum ingress, which is the ONE client surface no other gate in this tree
//! touches. `cuda_serve_round_trip` boots the same standalone and goes further
//! through the websocket at `/v1/ws`; `gateway/tests/gateway_smoke` drives the
//! session pipe with no ingress under it. The HTTP edge is the seam between
//! them and this is what holds it.
//!
//! # The SKU, because this file is where the census started
//!
//! It booted `Qwen/Qwen3-0.6B` and named it in three doc lines and an
//! `#[ignore]` reason. This build ships no SKU that checkpoint can be: every
//! row of `models::qwen_3::IMPORTS` claims an artifact by NAME at a qwen3.5
//! geometry, and `runtime::model::ROWS` has no id for it, so
//! `runtime::engine::load` refuses before a fire and the gate would have failed
//! for a reason that is not about booting. It comes up on
//! `qwen35-d0.8b-bf16-kv-bf16` now, through `common::boot_cuda`, which is the
//! checkpoint every other gate in this directory already pins against.
//!
//! Run:
//! ```text
//! cargo test -p pie-gpu-tests --features engine-cuda-13 \
//!   --test cuda_boot_smoke -- --ignored --nocapture
//! ```

mod common;

use anyhow::Result;

/// Boots once with the real CUDA engine and proves all three planes co-reside
/// and that a turn posted at the HTTP ingress streams back to `[DONE]`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "the boot gate: needs a CUDA device and the Qwen3.5-0.8B snapshot"]
async fn the_standalone_boots_and_the_ingress_streams_a_turn() -> Result<()> {
    let pie = common::boot_cuda().await?;

    assert_ne!(
        pie.listen_addr.port(),
        0,
        "client edge must bind a real ephemeral port"
    );
    assert_ne!(
        pie.worker_addr.port(),
        0,
        "worker dial-in must bind a real ephemeral port"
    );
    assert!(
        pie.listen_addr.ip().is_loopback() && pie.worker_addr.ip().is_loopback(),
        "standalone is loopback-only"
    );

    // Ping through ingress — full client path, no tokenization.
    let payload = serde_json::to_vec(&serde_json::json!({ "type": "ping", "corr_id": 1 }))?;
    let resp = reqwest::Client::new()
        .post(format!("http://{}/v1/generate", pie.listen_addr))
        .header("x-pie-identity", "cuda-smoke/test")
        .header("content-type", "application/json")
        .body(payload)
        .send()
        .await?;
    assert_eq!(resp.status(), 200, "ingress must accept the turn");
    let body = resp.text().await?;
    assert!(
        body.contains("[DONE]"),
        "the turn must stream back to [DONE]; got: {body}"
    );

    pie.shutdown().await;
    Ok(())
}
