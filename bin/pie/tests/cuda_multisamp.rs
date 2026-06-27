//! 4090 #7 per-kind cutover verify (echo): the recognizer-driven dispatch.
//!
//! Runs the `multisamp` inferlet (top-k · top-p · min-p · top-k-top-p in
//! sequence). The `PIE_RECOGNIZER_DISPATCH` gate is shell-controlled (the test
//! sets only the audit), so the harness runs it twice — gate-ON (recognizer
//! drives the dispatch flag-set) vs gate-OFF (legacy flags) — and the per-kind
//! `gate-on≡gate-off` is the two token streams being identical. `PIE_RECOGNIZER_
//! AUDIT=1` additionally asserts (log-only) the recognizer ≡ legacy flag-set per
//! fire across every kind.
//!
//! `#[ignore]`, driver-cuda. Run gate-OFF then gate-ON, diff the `tokens`:
//!   cargo test … --test cuda_multisamp -- --ignored --nocapture          (off)
//!   PIE_RECOGNIZER_DISPATCH=1 cargo test … --test cuda_multisamp …        (on)

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + cuda + qwen-3-0.6b"]
async fn multisamp_on_real_driver() -> Result<()> {
    common::init_trace();
    // SAFETY: set before any worker thread spawns. Audit always on (asserts the
    // recognizer≡legacy flag-set per fire); the DRIVE gate is shell-controlled.
    unsafe {
        std::env::set_var("PIE_RECOGNIZER_AUDIT", "1");
    }

    let pie = common::boot_4090().await?;
    let dispatch = std::env::var("PIE_RECOGNIZER_DISPATCH").unwrap_or_default();
    eprintln!(
        "[multisamp] booted listen={} PIE_RECOGNIZER_DISPATCH={:?}",
        pie.listen_addr, dispatch
    );

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "multisamp"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "multisamp wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/multisamp.wasm");
    let manifest = ws.join("multisamp/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;

    let mut proc = client
        .launch_process("multisamp@0.1.0".to_string(), "{}".to_string(), true, None)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[multisamp] DISPATCH={dispatch:?} returned: {json}");

    // 4 kinds × 4 tokens = 16 valid tokens (no crash / fallback / late-bind miss).
    let lb = json.find('[').context("no [")?;
    let rb = json[lb..].find(']').map(|i| lb + i).context("no ]")?;
    let inner = json[lb + 1..rb].trim();
    let n = if inner.is_empty() { 0 } else { inner.split(',').count() };
    eprintln!("[multisamp] DISPATCH={dispatch:?} n_tokens={n}");
    anyhow::ensure!(n == 16, "expected 16 tokens (4 kinds × 4), got {n}");

    pie.shutdown().await;
    Ok(())
}
