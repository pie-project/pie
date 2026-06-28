//! Late-channel grammar masking verify (cut #2 production supply, `LATE_MASK_OK`)
//! — GPU (bravo ∥ delta). Boots the 4090 + real driver and launches the
//! `grammar-late` inferlet, which drives the de-hardwired grammar-masking OP
//! (Sampling-IR `0x65 mask-apply`) through the **production Late-channel supply**:
//! a `Readiness::Late` packed mask that rides bravo's device-alias carrier
//! (`sampling_late_device_*`) → `pie_tensor_write_async` (direct WASM-slice→device
//! memcpy, no IPC staging) → echo's `HostLate` resolve.
//!
//! This is the Late-supply counterpart to `cuda_grammar_op` (`MASK_OP_OK`, the
//! SUBMIT op-verify), which explicitly does NOT exercise the device-alias channel.
//! It closes the cut-#1 host↔device supply-drift class on the **real** production
//! carrier.
//!
//! TWO inferlets are launched CONCURRENTLY so their forward passes batch in the
//! FCFS scheduler — the late carriers ride the **batch-merged** request
//! (`extend_sampling_programs_from` concat, the exact cut #1 drop site, now with
//! bravo's durable carrier-preservation guard). The grammar mask is sequential
//! (`mask[N]` needs `token[N-1]`, computed host-side) so a single inferlet cannot
//! self-run-ahead; concurrent procs are how the masked path rides the merge.
//!
//! Non-degeneracy (the cut #1 discipline; delta drives + confirms the trace):
//!   - MERGED path: the batch-merge fires (not a single-req path).
//!   - DEVICE-ALIAS branch: `sampling_late_device_ptrs[k] != 0` (not the staged
//!     blob fallback) — a staged mask would test the wrong channel.
//!   - HONEST gate (in the inferlet): conform vs byte-identical CPU ref ∧
//!     constrained ≠ unconstrained natural argmax ∧ forced-out. A dropped carrier
//!     ⇒ `SkippedLateBindMiss` ⇒ constrained == natural ⇒ assert #2 fails loud; a
//!     misaligned carrier ⇒ wrong mask ⇒ conform fails. No false-pass escape.
//!
//! Run (GPU):
//!   cargo test -p pie-bin --features driver-cuda --test cuda_grammar_late -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// Concurrent grammar-late inferlets — ≥2 so their passes batch-merge the late
/// carriers. Bump to widen the merge if the scheduler doesn't co-batch at 2.
const N_CONCURRENT: usize = 2;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "cut #2 Late-supply masking verify: needs the 4090 + cuda + qwen-3-0.6b + bravo's device-alias carrier + echo's HostLate + 0x65 kernel"]
async fn grammar_late_supply_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[grammar-late] booted, listen_addr={}", pie.listen_addr);

    // Build the Late-supply grammar masking verify inferlet.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "grammar-late"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "grammar-late wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/grammar_late.wasm");
    let manifest = ws.join("grammar-late/Pie.toml");

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
    eprintln!(
        "[grammar-late] program installed, launching {N_CONCURRENT} concurrent constrained decodes (merged late carrier)…"
    );

    // Launch all procs FIRST (no await between) so they run concurrently and the
    // scheduler co-batches their forward passes → the late carrier rides the
    // merged request.
    let mut procs = Vec::with_capacity(N_CONCURRENT);
    for _ in 0..N_CONCURRENT {
        procs.push(
            client
                .launch_process("grammar-late@0.1.0".to_string(), "{}".to_string(), true)
                .await
                .context("launch")?,
        );
    }

    let mut results = Vec::with_capacity(N_CONCURRENT);
    for (i, mut proc) in procs.into_iter().enumerate() {
        let json = proc
            .wait_for_return()
            .await
            .with_context(|| format!("wait_for_return proc {i}"))?;
        eprintln!("[grammar-late] proc {i} returned: {json}");
        results.push(json);
    }

    pie.shutdown().await;

    // Each inferlet reports `LATE_MASK_OK=<bool>` = CONFORM (device==CPU-ref) ∧
    // FORCED-OUT (natural argmax disallowed + forced out via the device-alias
    // mask). Non-degenerate by construction: an all-allowed/dropped mask fails
    // FORCED-OUT, a wrong/misaligned device-alias mask fails CONFORM.
    for (i, json) in results.iter().enumerate() {
        anyhow::ensure!(
            json.contains("LATE_MASK_OK=true"),
            "Late-supply grammar masking verify FAILED (proc {i}) — the device-alias \
             mask diverged from the host CPU reference (conform) or the disallowed \
             natural argmax was NOT forced out (the Late carrier dropped/misaligned \
             ⇒ SkippedLateBindMiss / wrong mask): {json}"
        );
    }
    Ok(())
}
