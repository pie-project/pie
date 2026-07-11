//! §6.1 north-star e2e — composed MTP spec-verify ⟂ PER-POSITION grammar mask on
//! the REAL 4090 + Qwen3-0.6B (bravo). The north-star assembly nucleus, on real HW.
//!
//! Boots the 4090 + real CUDA driver and launches the `mtpverify` inferlet — the
//! §6.1 composition: draft `k` tokens, read the target's `[k, vocab]` logits, apply
//! a grammar mask PER speculative position (a `[k, vocab]` bool matrix via `select`,
//! NOT the packed single-mask `mask_apply` — echo's design constraint that per-row
//! masks cannot be one broadcast), per-row `argmax`, verify the `[k]` draft
//! (`eq → cumprod` prefix-AND → sentinel `[k]`-Token). This is the FULL composed
//! program (mask-`select` + spec-verify DAG) lowering through the real Sampling-IR
//! CUDA backend on genuine model logits — what the mock (`north_star_e2e.rs`)
//! proved host-side, now certified on the device.
//!
//! Distinct from the two half-tests already on GPU:
//!   * `cuda_specverify_k` — the spec-verify DAG (`argmax→eq→cumprod→select`) WITHOUT
//!     a grammar mask.
//!   * `cuda_grammar_op`  — the packed `mask_apply` (0x65) single-mask broadcast,
//!     WITHOUT spec-verify.
//! This test is the first to run BOTH fused in one program, with the per-position
//! `select`-mask that the packed op cannot express.
//!
//! mtpverify is model-agnostic (host drafts verified against Qwen's REAL `[k,vocab]`
//! argmax; the grammar-forced arm is mask-deterministic, independent of the model's
//! natural argmax), so it needs no MTP head. Three verdicts (the inferlet asserts
//! each; all non-degenerate):
//!   * GRAMMAR_FORCES_ACCEPT — an allow-only-`T` mask (every position) forces each
//!     row's masked argmax to `T`, so the draft `[T; k]` accepts in FULL — the
//!     grammar constraint DRIVES the spec-verify outcome (mask ⟂ speculation).
//!   * COMPOSITION_FIRES — an all-allow mask yields a DIFFERENT accept-prefix for the
//!     same draft (the mask changed the verify result — a real composition, not a
//!     passthrough).
//!   * LOOP_CTRL_OK — the multi-step accept-prefix decode loop advances by exactly
//!     each step's accepted length (control-flow correctness of the §6.1 loop).
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_mtpverify -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "§6.1 composed MTP+grammar e2e: needs the 4090 + cuda + qwen-3-0.6b + the matrix select-mask ∘ spec-verify DAG lowering"]
async fn mtp_grammar_composition_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[mtp-verify] booted, listen_addr={}", pie.listen_addr);

    // Build the §6.1 composed spec-verify ⟂ per-position grammar inferlet.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "mtpverify"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "mtpverify wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/mtpverify.wasm");
    let manifest = ws.join("mtpverify/Pie.toml");

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
    eprintln!("[mtp-verify] program installed, launching §6.1 composed MTP+grammar verify (k=4)…");

    // mtpverify parses a plain draft-window size (default 4); "4" ⇒ a [4, vocab]
    // matrix verify window.
    let mut proc = client
        .launch_process("mtpverify@0.1.0".to_string(), "4".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[mtp-verify] returned: {json}");

    pie.shutdown().await;

    // ── The §6.1 land gates (real HW) ────────────────────────────────────────
    // GRAMMAR_FORCES_ACCEPT — the load-bearing composition claim: an allow-only-`T`
    // mask applied per-position (`select`) forces every row's masked argmax to `T`,
    // so the draft `[T; k]` accepts in FULL. This is mask-DETERMINISTIC (independent
    // of Qwen's natural argmax) — it proves the per-position `select`-mask lowers +
    // runs on the real matrix path AND drives the spec-verify DAG's accept result.
    // A broken matrix-select (mask dropped, or broadcast-collapsed to row 0) gives a
    // wrong/short accept-prefix and fails here.
    anyhow::ensure!(
        json.contains("GRAMMAR_FORCES_ACCEPT=true"),
        "§6.1 composition BROKEN on real HW — the per-position allow-only-T select-mask \
         did NOT force a full accept: the matrix `select`-mask failed to lower (dropped, \
         or collapsed to a single broadcast row), or the spec-verify DAG mis-consumed \
         the masked argmax: {json}"
    );
    // COMPOSITION_FIRES — non-degeneracy: an all-allow mask yields a DIFFERENT
    // accept-prefix (Qwen's REAL argmax over the prompt ≠ the forced draft `[T;k]`),
    // so the mask genuinely changed the verify outcome — a real constraint⟂speculation
    // composition, not a passthrough that would pass GRAMMAR_FORCES_ACCEPT trivially.
    anyhow::ensure!(
        json.contains("COMPOSITION_FIRES=true"),
        "§6.1 composition is a PASSTHROUGH on real HW — the all-allow mask gave the SAME \
         accept-prefix as the forced mask, so the grammar constraint did not affect the \
         verify result (mask ignored / no-op select): {json}"
    );
    // LOOP_CTRL_OK — the §6.1 multi-step accept-prefix decode loop advances by exactly
    // each step's accepted length over a known accept/reject-at-1/accept schedule
    // (control-flow correctness: prefix advance, re-draft, termination).
    anyhow::ensure!(
        json.contains("LOOP_CTRL_OK=true"),
        "§6.1 spec-decode loop control flow diverged on real HW — an accept-prefix \
         advance did not equal its accepted length (wrong prefix advance / re-draft \
         wiring): {json}"
    );
    Ok(())
}
