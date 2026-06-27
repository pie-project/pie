//! Phase-1 #12 isolated top-p HW gate (bravo).
//!
//! Fires the `isolatedtopp` inferlet — a SINGLE TopP fire on a fresh context,
//! so the tokens depend ONLY on the prompt (no shared-context top-k pollution
//! that confounds the multisamp-sequence `[…]×4` parity). With the (B) repoint,
//! `generate(Sampler::TopP)` emits the canonical `standard_program` → recognize
//! → extract(T=0.8, p=0.9) → FlashInfer. Run with a fixed seed for a clean
//! token-identity vs the slot-surface baseline captured off `70e8082d`:
//!
//!   PIE_FIXED_SAMPLING_SEED=12345 PIE_SAMPLING_IR_TRACE=1 \
//!     cargo test -p pie-bin --features driver-cuda --test cuda_isolatedtopp \
//!     -- --ignored --nocapture --test-threads=1
//!
//! Asserts 4 tokens (no crash / fallback). Token-identity vs the baseline is a
//! manual/scripted diff of the `[ISOLATED_TOPP] tokens:` line across the two
//! surfaces (same fixed seed).

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs the 4090 + cuda + qwen-3-0.6b"]
async fn isolatedtopp_on_real_driver() -> Result<()> {
    common::init_trace();

    let pie = common::boot_4090().await?;
    let seed = std::env::var("PIE_FIXED_SAMPLING_SEED").unwrap_or_default();
    eprintln!(
        "[isolatedtopp] booted listen={} PIE_FIXED_SAMPLING_SEED={:?}",
        pie.listen_addr, seed
    );

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "isolatedtopp"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "isolatedtopp wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/isolatedtopp.wasm");
    let manifest = ws.join("isolatedtopp/Pie.toml");

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;

    let mut proc = client
        .launch_process("isolatedtopp@0.1.0".to_string(), "{}".to_string(), true, None)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[isolatedtopp] seed={seed:?} returned: {json}");

    let lb = json.find('[').context("no [")?;
    let rb = json[lb..].find(']').map(|i| lb + i).context("no ]")?;
    let inner = json[lb + 1..rb].trim();
    let n = if inner.is_empty() { 0 } else { inner.split(',').count() };
    eprintln!("[isolatedtopp] seed={seed:?} n_tokens={n} tokens=[{inner}]");
    anyhow::ensure!(n == 4, "expected 4 tokens, got {n}");

    // Phase-1 #12 token-identity regression guard (echo HW-confirmed on the
    // slot surface `70e8082d`): under the fixed seed, the program→recognize→
    // extract→FlashInfer path must reproduce the pre-migration FlashInfer
    // tokens. A drift (wrong vocab, wrong-key params, wrong dispatch) trips this.
    if seed == "12345" {
        let toks: Vec<u32> = inner.split(',').filter_map(|s| s.trim().parse().ok()).collect();
        anyhow::ensure!(
            toks == [2025, 304, 272, 481],
            "isolated top-p token-identity regression: seed=12345 expected \
             [2025, 304, 272, 481] (pre-migration FlashInfer), got {toks:?}"
        );
        eprintln!("[isolatedtopp] token-identity guard PASS (== pre-migration [2025,304,272,481])");
    }

    pie.shutdown().await;
    Ok(())
}
