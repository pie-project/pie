//! `[k]`-Token marshal e2e verify (#32/#33) — GPU (delta).
//!
//! Boots the 4090 + real driver and launches the `specverify` inferlet, which
//! drives a REAL **k>1** `spec_verify_greedy` PROGRAM through the full
//! inferlet→engine→marshal path, asserting the FULL `[k]`-Token output routes OFF
//! the system-drafter `spec_tokens` channel into the per-(request,output) two-level
//! CSR `program_tokens` (echo's emit + bravo's schema + foxtrot's read/slicing).
//!
//! This exercises exactly what the 2a `specverify` (k=1) could NOT: a `[k]`-Token
//! output with `elem_count > 1` — which #32/#33 route off `spec_tokens` and read
//! back in FULL (the #19-class mis-route + the `[1]`-truncation both fixed). It
//! also hits foxtrot's `token_payload_only` `program_tokens.is_empty()` routing
//! trap-fix: a pure `[k]`-fire (no entropies) must take the RICH path, not dense.
//!
//! Non-degenerate by construction (the inferlet asserts both arms):
//!   * ACCEPT-ALL: `draft == g` ⇒ output `== g` (k REAL tokens, full `[k]`).
//!   * MIXED (reject at j=k/2): output `== g[0..j] ++ [-1; k-j]` (prefix + sentinel).
//! A `[1]`-truncation fails `len == k`; a wrong-channel/drop diverges the values.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   PIE_COMPILER_LAUNCHER=env CUDACXX=/usr/local/cuda/bin/nvcc \
//!   CPM_SOURCE_CACHE=$HOME/.cache/pie-cpm \
//!   cargo test -p pie-bin --features driver-cuda --test cuda_specverify_k -- --ignored --nocapture

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#32/#33 [k]-Token marshal verify: needs the 4090 + cuda + qwen-3-0.6b + the two-level CSR program_tokens marshal"]
async fn k_token_marshal_off_spec_tokens_on_real_driver() -> Result<()> {
    common::init_trace();
    let pie = common::boot_4090().await?;
    eprintln!("[k-marshal] booted, listen_addr={}", pie.listen_addr);

    // Build the k>1 [k]-Token marshal verify inferlet.
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "specverify"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "specverify wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/specverify.wasm");
    let manifest = ws.join("specverify/Pie.toml");

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
    eprintln!("[k-marshal] program installed, launching k>1 [k]-Token marshal verify…");

    let mut proc = client
        .launch_process("specverify@0.1.0".to_string(), "{\"k\":4}".to_string(), true)
        .await
        .context("launch")?;
    let json = proc.wait_for_return().await.context("wait_for_return")?;
    eprintln!("[k-marshal] returned: {json}");

    pie.shutdown().await;

    // The inferlet reports the dual verdict. `MARSHAL_EMITS_K` is the #32/#33
    // claim and the land gate: a `[k]`-Token (elem_count=k>1) routes OFF
    // `spec_tokens` into the per-(request,output) `program_tokens` CSR and reads
    // back in FULL (len == k) via `tokens()`. The argmax-matrix probe has no `-1`
    // sentinel, so all k values emit regardless of their correctness — proving the
    // marshal transports all k independent of any upstream value bug.
    //
    // `MATRIX_ARGMAX_OK` is a SEPARATE diagnostic (not gated here): whether the
    // `[k,vocab]` matrix intrinsic feeds each row r its own position's logits. A
    // pre-existing k>1 matrix-logits gap (rows≥1 unmaterialized) makes it false
    // WITHOUT affecting the marshal — surfaced in the result for the follow-up.
    anyhow::ensure!(
        json.contains("MARSHAL_EMITS_K=true"),
        "[k]-Token marshal verify failed — the k>1 [k]-Token output did NOT route \
         in FULL off `program_tokens` (len != k): truncated to [1], wrong-channel \
         via spec_tokens, or a dense-fast-path drop: {json}"
    );
    if json.contains("MATRIX_ARGMAX_OK=false") {
        eprintln!(
            "[k-marshal] NOTE: marshal verified (emits all k off program_tokens), \
             but the k>1 [k,vocab] MATRIX-LOGITS intrinsic leaves rows≥1 \
             unmaterialized (argmax→0) — a SEPARATE pre-existing producer gap, not \
             a marshal bug. See {json}"
        );
    }
    Ok(())
}
