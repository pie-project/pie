//! §6.2 beam [B,P] DEVICE-GEOMETRY e2e — real driver (4090). The B1/B2/B3
//! north-star: the `beam-devgeo` inferlet drives a fork/freeze/heir beam search
//! whose per-step geometry is traced IN-GRAPH to wire form (CSR
//! `page_indptr = CumSum(np)` + densely-packed live pages bound to the
//! `PageIndptr`/`Pages` ports, plus the explicit `WSlot`/`WOff` write
//! descriptors and the dense `AttnMask`) — NO host replay of the freeze /
//! designated-child / page-turn arithmetic. Submitted through the
//! `inferlet::ptir` bridge (`forward-pass.new` / `pipeline.submit`) end to end:
//!
//!   guest device-geometry program
//!     → runtime B3 unified device-geometry submit (`fire_device_geometry`,
//!       PageLease grants, run-ahead FIFO)
//!     → driver descriptor resolver (`resolve_descriptors` → `FireGeometry`)
//!     → B2 explicit-KV-write (`launch_write_kv_explicit_bf16` honoring
//!       WSlot/WOff so a frozen fork's cell is not overwritten)
//!     → beam attention with the packed dense mask
//!     → harvest → the guest's `out`/`out_par`/`out_scr` take.
//!
//! This is the INTEGRATION gate for Track B's device-geometry path: it proves
//! the beam fire flows guest → device-geometry submit → the driver's explicit
//! write + masked attention → harvest, end to end on real logits, without
//! crashing or degenerating — the replacement for the host-replay `ptir_beam.rs`
//! path (which B5 deletes once this + the B4 shadow-verify are green).
//!
//!   PIE_PTIR_TRACE=1 cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_beam_devgeo_e2e -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "§6.2 beam device-geometry e2e: needs the 4090 + cuda + qwen-3-0.6b + the ptir feature"]
async fn beam_devgeo_bp_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[beam-devgeo-e2e] booted, listen_addr={}", pie.listen_addr);

    // Build the `beam-devgeo` inferlet to wasm (member of the runtime
    // test-inferlets ws). The crate name normalizes to `beam_devgeo.wasm`.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "beam-devgeo"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "beam-devgeo wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/beam_devgeo.wasm");
    let manifest = ws.join("beam-devgeo/Pie.toml");
    anyhow::ensure!(wasm.exists(), "missing wasm: {}", wasm.display());

    let client =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect")?;
    client.authenticate("test-user", &None).await.context("auth")?;
    client.add_program(&wasm, &manifest, true).await.context("add_program")?;
    eprintln!("[beam-devgeo-e2e] program installed, launching device-geometry beam search…");

    let mut proc = client
        .launch_process("beam-devgeo@0.1.0".to_string(), "{}".to_string(), true)
        .await
        .context("launch")?;
    let out = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[beam-devgeo-e2e] returned: {out}");

    pie.shutdown().await;

    // The device-geometry beam decode loop ran register→instantiate→
    // (submit→take)×steps through the full path (guest device-geometry program →
    // B3 device-geometry submit → B2 explicit KV write + masked beam attention →
    // harvest → out/out_par/out_scr). A crash/degeneration surfaces as a
    // non-`BEAM_DEVGEO` return or an empty token list.
    anyhow::ensure!(
        out.contains("BEAM_DEVGEO"),
        "§6.2 device-geometry beam e2e: unexpected return (expected a `BEAM_DEVGEO …` \
         summary; a fire-path crash returns the error string): {out:?}"
    );
    // Non-degeneracy: the beam produced tokens (the submit→take loop completed at
    // least one step feeding device-produced geometry through the resolver).
    let has_tokens = out.contains("tokens=[") && !out.contains("tokens=[]");
    anyhow::ensure!(
        has_tokens,
        "§6.2 device-geometry beam e2e: no beam tokens harvested — the submit→take loop \
         produced an empty hypothesis (a dropped ptir_output_* or a stalled fire): {out:?}"
    );
    eprintln!(
        "[beam-devgeo-e2e] §6.2 GREEN — device-geometry beam [B,P] fired e2e \
         (in-graph wire geometry + B2 explicit write): {out}"
    );
    Ok(())
}
