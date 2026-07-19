//! Grammar mask-apply OP verify (cut #2(a), `MASK_OP_OK`) — GPU (golf ∥ delta).
//! Boots the 4090 + real driver and launches the `grammar` inferlet, which
//! drives the de-hardwired grammar-masking OP (Sampling-IR `0x65 mask-apply`)
//! through the SUBMIT-mask path: a host matcher packs a per-step allowed-token
//! bitmask, binds it submit-bound, and fires `argmax(mask_apply(logits, mask))`.
//!
//! Two non-degenerate asserts (the cut #1 discipline; an all-allowed no-op mask
//! cannot pass):
//!   1. CONFORM — each device token == `apply_mask_argmax(raw_logits, mask)`
//!      recomputed host-side with the byte-identical CPU reference (no
//!      host<->device drift, the class that bit cut #1).
//!   2. FORCED-OUT — the natural (unconstrained) step-0 argmax is disallowed in
//!      the mask AND absent from the output (the `-inf` actually fired).
//!
//! SCOPE: this verifies the `0x65` mask-apply OP via the SUBMIT path only. It
//! does NOT exercise the Late-channel supply (`tensor.write` -> device-alias
//! carrier -> `HostLate`); that production path needs its OWN GPU verify on the
//! merged run-ahead path when the Late wiring lands. `MASK_OP_OK` MUST NOT stand
//! in for the Late-supply verify.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_grammar_op -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "cut #2(a) grammar mask-apply OP verify: needs the 4090 + cuda + qwen-3-0.6b + echo's 0x65 kernel + charlie's reader"]
async fn grammar_mask_op_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[grammar-op] booted, listen_addr={}", pie.listen_addr);

    // Build the grammar mask-apply OP verify inferlet.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "grammar"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "grammar wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/grammar.wasm");
    let manifest = ws.join("grammar/Pie.toml");

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
    eprintln!("[grammar-op] program installed, launching constrained decode…");

    let mut proc = client
        .launch_process("grammar@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[grammar-op] returned: {json}");

    pie.shutdown().await;

    // CONFORM (device==CPU-ref) ∧ FORCED-OUT (natural argmax disallowed +
    // forced out). Non-degenerate by construction: an all-allowed mask fails
    // FORCED-OUT, a wrong device mask fails CONFORM.
    let report: serde_json::Value = serde_json::from_str(&json).context("parse grammar report")?;
    anyhow::ensure!(
        report.get("conform").and_then(|v| v.as_bool()) == Some(true)
            && report.get("forced_out").and_then(|v| v.as_bool()) == Some(true),
        "grammar mask-apply OP verify failed — device mask-apply diverged from \
         the host CPU reference (conform) or the disallowed natural argmax was \
         NOT forced out (-inf didn't fire): {json}"
    );
    Ok(())
}
