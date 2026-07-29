//! **fold-commit on the REAL linear model** — drives the `gdn-foldcommit`
//! inferlet on Qwen3.5-0.8B (GDN backbone) so the buffer/fold machinery is
//! exercised against real weights rather than the mock model in
//! `runtime/engine/tests/rs_frame.rs`.
//!
//! `rs_frame.rs` proves the HOST bookkeeping (slot accounting, CSR shapes,
//! refusals). It cannot prove the driver actually replays buffered
//! activations, because its model computes nothing. Everything about the
//! recurrent read path lives on the far side of that boundary, so it needs a
//! GPU test.
//!
//! Two cases, and the split is the point:
//!
//!  - `one_chunk_folds_from_the_buffer` — buffer one chunk, fold a prefix of
//!    it. This is the position the driver supports today.
//!  - `two_chunks_need_the_buffer_read_path` — buffer a chunk, fold PART of
//!    it, then buffer a SECOND chunk onto the surviving tail. The second
//!    append starts from a non-empty buffer, so its recurrence would have to
//!    begin at `folded ⊕ replay(buffer)`. The runtime refuses it today
//!    (`pipeline::fire::rs_plan_for`) and this test PINS that refusal: it is
//!    the acceptance test for the read path, and it flips from "refused for
//!    the right reason" to "runs and is correct" when the read path lands.
//!
//! `#[ignore]`, driver-cuda. Run:
//!   cargo test -p pie-bin --features driver-cuda --test cuda_gdn_foldcommit \
//!     -- --ignored --nocapture

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

mod common;

/// Build `gdn-foldcommit` to wasm and return the workspace dir.
fn build_inferlet() -> Result<std::path::PathBuf> {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/engine/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "gdn-foldcommit"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "wasm build failed for gdn-foldcommit");
    Ok(ws)
}

/// Boot the GDN model, install `gdn-foldcommit`, run it with `input`, and
/// return whatever the inferlet produced — including an error string, since
/// the refusal IS the observation in the two-chunk case.
async fn run_foldcommit(input: &str) -> Result<std::result::Result<String, String>> {
    common::init_trace();
    let ws = build_inferlet()?;

    let pie = common::boot_4090_mtp().await?;
    eprintln!(
        "[gdn-foldcommit] booted Qwen3.5-0.8B, listen_addr={}",
        pie.listen_addr
    );

    let setup =
        Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
            .await
            .context("connect setup")?;
    setup
        .authenticate("test-user", &None)
        .await
        .context("auth setup")?;
    setup
        .add_program(
            &ws.join("target/wasm32-wasip2/debug/gdn_foldcommit.wasm"),
            &ws.join("gdn-foldcommit/Pie.toml"),
            true,
        )
        .await
        .context("add_program gdn-foldcommit")?;
    drop(setup);

    let c = Client::connect_with_identity(&format!("ws://{}/v1/ws", pie.listen_addr), "test-user")
        .await
        .context("connect run session")?;
    c.authenticate("test-user", &None)
        .await
        .context("auth run session")?;
    let mut proc = c
        .launch_process("gdn-foldcommit@0.1.0".to_string(), input.to_string(), true)
        .await
        .context("launch gdn-foldcommit")?;
    let outcome = proc.wait_for_return().await.map_err(|e| e.to_string());
    drop(c);
    pie.shutdown().await;

    match &outcome {
        Ok(json) => eprintln!("[gdn-foldcommit] returned: {json}"),
        Err(error) => eprintln!("[gdn-foldcommit] failed: {error}"),
    }
    Ok(outcome)
}

/// Buffer one chunk, fold a PREFIX of it, abandon the rest. The supported
/// position, and the one `mtp-native-verify` also drives — here without the
/// MTP head in the way, so a failure is unambiguously about fold-commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn one_chunk_folds_from_the_buffer() -> Result<()> {
    // "2" = accept 2 of the 4 speculative tokens, so the fold lands strictly
    // inside the buffer and the abandoned tail is non-empty. A whole-buffer
    // fold would not distinguish "folded the prefix" from "folded everything".
    let result = run_foldcommit("2")
        .await?
        .map_err(|e| anyhow::anyhow!("gdn-foldcommit failed: {e}"))?;

    anyhow::ensure!(
        result.contains("committed=2"),
        "expected the accepted prefix to be committed, got: {result}"
    );
    anyhow::ensure!(
        result.contains("abandoned=2"),
        "expected the rejected tail to be abandoned, got: {result}"
    );
    Ok(())
}

/// A SECOND buffered chunk on top of a partially folded buffer.
///
/// This is the read path in its smallest honest form. After folding 2 of 4
/// buffered tokens the buffer still holds the unfolded tail, so appending a
/// new chunk means the new tokens' recurrence must start from
/// `folded ⊕ replay(buffer)` — and every recurrence today initializes from
/// `recurrent_state[slot]`, the state at the folded boundary.
///
/// Until the read path lands the runtime must REFUSE this, and refuse it for
/// the buffer reason rather than dying somewhere incidental — a fire that ran
/// and quietly ignored the buffered tail would produce a plausible wrong
/// answer with no symptom, which is the failure mode the whole fold-commit
/// design exists to prevent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn two_chunks_need_the_buffer_read_path() -> Result<()> {
    let outcome = run_foldcommit("chain").await?;

    match outcome {
        Err(error) => {
            let lowered = error.to_lowercase();
            anyhow::ensure!(
                lowered.contains("non-empty buffer") || lowered.contains("read path"),
                "the second chunk must be refused BECAUSE of the buffer, not \
                 incidentally. Got: {error}"
            );
            eprintln!(
                "[gdn-foldcommit] second chunk correctly refused (read path absent): {error}"
            );
        }
        Ok(result) => {
            // The read path has landed. Then the chain must be CORRECT, not
            // merely accepted: both chunks folded, nothing silently dropped.
            anyhow::ensure!(
                result.contains("chained=ok"),
                "the second chunk ran, so the read path exists — but the result \
                 does not confirm both chunks folded: {result}"
            );
            eprintln!("[gdn-foldcommit] read path is live and the chain folded: {result}");
        }
    }
    Ok(())
}
