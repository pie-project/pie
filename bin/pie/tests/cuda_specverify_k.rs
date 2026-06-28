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
    // #35-A (matrix-logits value-exact): the [k,vocab] matrix intrinsic feeds EACH
    // row r its own position's logits → the argmax-matrix probe == the greedy
    // continuation g[0..k]. NOTE: this argmax-ONLY arm passes even when the full
    // spec-verify DAG is broken (the standalone argmax launches grid=k); it is
    // NECESSARY-not-sufficient. The load-bearing gate is REJECT_MID_OK below.
    anyhow::ensure!(
        json.contains("MATRIX_ARGMAX_OK=true"),
        "[k,vocab] matrix-logits NOT value-exact (#35-A) — the per-row argmax probe \
         diverged from the greedy continuation: the matrix grid fired < k rows: {json}"
    );
    // The full spec-verify DAG (argmax→eq→cumprod→select) must be value-exact. The
    // accept-ALL arm (no reject) passes even with a per-block-collapsed cumprod, so
    // it too is necessary-not-sufficient.
    anyhow::ensure!(
        json.contains("SPEC_ACCEPT_ALL_OK=true"),
        "spec_verify_greedy(k) accept-ALL NOT value-exact (#35-A): {json}"
    );
    // THE NON-DEGENERATE GATE (#35-A land bar): a reject-MID draft `[1,1,0,1]` must
    // give the cross-row accept-prefix `[g0,g1,0,0]` (0-sentinel so the marshal
    // emits all k — `spec_verify_greedy`'s `-1` would TRUNCATE at the first reject,
    // MASKING the bug → a false pass). A batched per-block / grid-collapsed cumprod
    // gives `[g0,0,0,0]` (DAG collapsed to row 0) or `[g0,g1,0,g3]` (row-3 leak) —
    // either fails this. This is the ONLY arm that exercises the cross-row scan on
    // the real `batched=true` executor; closes charlie's land-blocking cumprod bug.
    // Precondition for the 0-sentinel detector to be UNAMBIGUOUS (foxtrot's
    // invariant): all drafts ∈ [1, vocab). A token-0 draft is indistinguishable
    // from the reject→0 sentinel → re-introduces the mask. If the greedy g happens
    // to contain a 0, the gate can't discriminate — fail loud, don't false-pass.
    anyhow::ensure!(
        json.contains("DRAFTS_NONZERO=true"),
        "0-sentinel reject-MID detector is AMBIGUOUS — a draft is token-0 \
         (indistinguishable from the reject→0 sentinel); the gate can't discriminate \
         accept-0 from reject. Re-seed with non-zero drafts: {json}"
    );
    anyhow::ensure!(
        json.contains("REJECT_MID_OK=true"),
        "spec_verify_greedy(k) cross-row cumprod COLLAPSED (#35-A land-block) — a \
         reject-MID draft did not yield the correct accept-prefix [g0,g1,0,0]: the \
         batched lowering folded the matrix DAG into one per-block/grid-1 kernel \
         (matrix rows ≠ batch axis). Needs charlie's batched-rejects-matrix → M=1 \
         2-kernel fix: {json}"
    );
    Ok(())
}
