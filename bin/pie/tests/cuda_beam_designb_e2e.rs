//! §6.2 beam DESIGN B DEVICE e2e — real driver (4090). First real end-to-end
//! exercise of the Design B steady-state (no-compaction) path: logical mask-out
//! + flat tail-append, NO heir election / freeze arithmetic / fresh-page
//! handshake / page reorder. The `beam-designb` inferlet drives a beam search
//! whose per-beam ancestry is encoded entirely in the dense `AttnMask` over a
//! fixed physical page pool; each survivor appends its new token at the next
//! free flat pool position (`WSlot`/`WOff`), and the KV write goes through B2's
//! explicit-KV-write path (`launch_write_kv_explicit_bf16`). Pages/PageIndptr
//! are constant (pool fixed between compactions). Submitted through the
//! `inferlet::ptir` bridge (`forward-pass.new` / `pipeline.submit`) end to end:
//!
//!   guest mask-out beam program
//!     → runtime device-geometry submit (PageLease grants, run-ahead FIFO)
//!     → driver descriptor resolver (`resolve_descriptors` → `FireGeometry`)
//!     → B2 explicit-KV-write (`launch_write_kv_explicit_bf16` honoring
//!       WSlot/WOff so each survivor's flat pool cell is written in place)
//!     → beam attention with the packed dense per-beam mask
//!     → harvest → the guest's `out`/`out_par`/`out_scr` take.
//!
//! This is the INTEGRATION gate for Design B's steady-state path: it proves the
//! mask-out beam fire flows guest → device-geometry submit → the driver's
//! explicit write + masked attention → harvest, end to end on real logits,
//! without crashing or degenerating. Compaction (the generic KV cell-move
//! primitive) is out of scope here — this exercises only the fixed-pool
//! steady state.
//!
//!   PIE_PTIR_TRACE=1 cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_beam_designb_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "§6.2 beam Design B device e2e: needs the 4090 + cuda + qwen-3-0.6b + the ptir feature"]
async fn beam_designb_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[beam-designb-e2e] booted, listen_addr={}", pie.listen_addr);

    // Build the `beam-designb` inferlet to wasm (member of the runtime
    // test-inferlets ws). The crate name normalizes to `beam_designb.wasm`.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "beam-designb"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "beam-designb wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/beam_designb.wasm");
    let manifest = ws.join("beam-designb/Pie.toml");
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
    eprintln!("[beam-designb-e2e] program installed, launching mask-out beam search…");

    let mut proc = client
        .launch_process("beam-designb@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[beam-designb-e2e] returned: {out}");

    pie.shutdown().await;

    // The mask-out beam decode loop ran register→instantiate→(submit→take)×steps
    // through the full path (guest mask-out program → device-geometry submit →
    // B2 explicit KV write + masked beam attention → harvest → out/out_par/out_scr).
    // A crash/degeneration surfaces as a non-`BEAM_DESIGNB` return or an empty
    // token list.
    anyhow::ensure!(
        out.contains("BEAM_DESIGNB"),
        "§6.2 Design B beam e2e: unexpected return (expected a `BEAM_DESIGNB …` \
         summary; a fire-path crash returns the error string): {out:?}"
    );
    // Non-degeneracy: the beam produced tokens (the submit→take loop completed at
    // least one step feeding the mask-out geometry through the resolver).
    let has_tokens = out.contains("tokens=[") && !out.contains("tokens=[]");
    anyhow::ensure!(
        has_tokens,
        "§6.2 Design B beam e2e: no beam tokens harvested — the submit→take loop \
         produced an empty hypothesis (a dropped ptir_output_* or a stalled fire): {out:?}"
    );
    eprintln!(
        "[beam-designb-e2e] §6.2 GREEN — Design B mask-out beam fired e2e \
         (fixed-pool tail-append + B2 explicit write + dense per-beam mask): {out}"
    );
    Ok(())
}
