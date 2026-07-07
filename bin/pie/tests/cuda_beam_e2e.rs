//! §6.2 beam [B,P] e2e — real driver (4090). The G2 north-star: the `beam`
//! inferlet (register→instantiate→submit-loop) drives a fork/freeze/heir beam
//! search whose per-step geometry is host-replayed in `ptir_host` (Design X:
//! `out_par`→[B,P] via the existing batch assembly → kvm→BRLE `masks` → a
//! prebuilt [B,P] ForwardRequest) → `submit_prebuilt_async` → charlie's SEAM-1
//! routing gate (a decode-shaped fire carrying a per-cell mask routes to
//! `prefill_custom` + `write_kv_to_pages`; `klen−1` lands each beam's write on
//! its (last live page, last_page_len−1) = WSlot/WOff; the heir appends the
//! shared tail in-place, the frozen fork references it read-only with its kvm
//! masking the heir's cell).
//!
//! This is the INTEGRATION gate for G2: it proves the beam fire flows guest →
//! ptir_host replay → prebuilt submit → the driver's beam attention (mask
//! honored) → harvest → the guest's `out/out_par/out_scr` take, end to end on
//! real logits, without crashing or degenerating.
//!
//!   PIE_PTIR_TRACE=1 cargo test -p pie-bin --features driver-cuda,ptir \
//!     --test cuda_beam_e2e -- --ignored --nocapture
//!
//! Bring-up flags (delta's 2 edges): fire-0 prompt seeding + fresh-slot
//! lifecycle — surface as an early `submit`/`take` error, not a golden diff.

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "§6.2 beam [B,P] e2e: needs the 4090 + cuda + qwen-3-0.6b + the ptir feature"]
async fn beam_bp_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[beam-e2e] booted, listen_addr={}", pie.listen_addr);

    // Build the `beam` inferlet to wasm (member of the runtime test-inferlets ws).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "beam"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "beam wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/beam.wasm");
    let manifest = ws.join("beam/Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;
    eprintln!("[beam-e2e] program installed, launching beam search…");

    let mut proc = client
        .launch_process("beam@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[beam-e2e] returned: {out}");

    pie.shutdown().await;

    // The beam decode loop ran register→instantiate→(submit→take)×steps through
    // the full path (guest → ptir_host [B,P] replay → prebuilt submit → charlie's
    // gate fires beam attention on real logits → harvest → out/out_par/out_scr).
    // A crash/degeneration surfaces as a non-`BEAM` return or an empty token list.
    anyhow::ensure!(
        out.contains("BEAM"),
        "§6.2 beam e2e: unexpected return (expected a `BEAM …` summary; a fire-path \
         crash returns the error string): {out:?}"
    );
    // Non-degeneracy: the beam produced tokens (the submit→take loop completed at
    // least one step feeding real geometry through the gate).
    let has_tokens = out.contains("tokens=[") && !out.contains("tokens=[]");
    anyhow::ensure!(
        has_tokens,
        "§6.2 beam e2e: no beam tokens harvested — the submit→take loop produced an \
         empty hypothesis (a dropped ptir_output_* or a stalled fire): {out:?}"
    );
    eprintln!("[beam-e2e] §6.2 GREEN — beam [B,P] fired e2e through the gate: {out}");
    Ok(())
}
