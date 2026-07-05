//! cuda_native standalone boot smoke (lane L4 / echo, Phase-2 WS7).
//!
//! The cuda_native analogue of `boot_smoke.rs`: boots the embedded controller +
//! gateway + worker in one process over loopback with the **real CUDA driver**
//! (`type = "cuda_native"`) against **qwen-3-0.6b** on the 4090, then pings the
//! full client path. This is the engine-boot half of the programmable-sampler
//! 4090 real-driver pass (echo's piece); the capability tests in
//! `programmable_sampler_4090.rs` reuse the same `common::boot_4090()`.
//!
//! `#[ignore]` (needs the 4090 + a HF-cached qwen-3-0.6b + a `driver-cuda`
//! build). Run with:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_boot_smoke -- --ignored --nocapture

mod common;

use anyhow::Result;

/// Boots once on the 4090 with the real CUDA driver + qwen-3-0.6b and proves
/// all three planes co-reside + the client path round-trips a ping — the
/// engine-boot gate for the programmable-sampler 4090 e2e.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + HF-cached qwen-3-0.6b + a driver-cuda build"]
async fn cuda_native_boots_qwen3_and_pings() -> Result<()> {
    let pie = common::boot_4090().await?;

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
