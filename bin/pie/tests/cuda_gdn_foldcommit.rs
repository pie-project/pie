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

    let pie = common::boot_4090_mtp(common::mtp_draft_tokens(3)).await?;
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

/// A fold running THROUGH a non-empty buffer, in the same fire that fills it —
/// a position the planner used to refuse outright ("fold and buffer in
/// separate fires").
///
/// One fire appends two tokens onto a buffer already holding two and folds all
/// four. It is a write and a fold at once: the driver replays the buffered
/// pair ahead of the new pair over the extended `[buffered | new]` layout,
/// scatters the new pair into the buffer, and — because the boundary is the
/// last extended token — lets the ordinary end-of-sequence writeback land the
/// folded state on it.
///
/// The inferlet checks it by equivalence against the same four tokens folded
/// two fires at a time, and draws the comparison AFTER the fold by continuing
/// each arm one token further. So a pass means the folded STATE matches, not
/// merely that the fire ran.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn a_fire_can_append_to_a_buffer_and_fold_through_it() -> Result<()> {
    let result = run_foldcommit("inside")
        .await?
        .map_err(|error| anyhow::anyhow!("fold-through failed: {error}"))?;
    anyhow::ensure!(
        result.contains("agree=yes"),
        "appending onto a buffer and folding through it must agree with the \
         same tokens folded two fires at a time: {result}"
    );
    eprintln!("[gdn-foldcommit] {result}");
    Ok(())
}

/// TWO requests in ONE fire landing their folded boundaries in DIFFERENT
/// places — the shape the planner used to refuse ("a fire folds uniformly
/// today") and the one real serving wants: a request committing while another
/// speculates.
///
/// The two rows run the identical dispatch over the identical layout; they
/// differ only in whether the recurrence PERSISTS, which now travels as a
/// per-row device mask rather than a per-pass flag. The inferlet submits both
/// fires before awaiting either so the batcher composes them, then checks each
/// row against the same shape run SOLO — one token PAST the mixed fire, so a
/// pass pins the resulting states and not the logits along the way.
///
/// It also asserts up front that the two references DISAGREE with each other.
/// Without that the equivalence would hold vacuously and a mask ignored in
/// either direction would go unnoticed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn one_fire_can_fold_one_request_while_another_only_buffers() -> Result<()> {
    let result = run_foldcommit("mixed")
        .await?
        .map_err(|error| anyhow::anyhow!("mixed-position fire failed: {error}"))?;
    anyhow::ensure!(
        result.contains("agree=yes"),
        "a fire that folds one row and buffers another must match the same two \
         shapes run solo: {result}"
    );
    eprintln!("[gdn-foldcommit] {result}");
    Ok(())
}

/// A fold whose LENGTH the host never learns — `t15`.
///
/// A speculative decode's accepted count is produced by the verify fire, on
/// device. Reading it back to choose the commit's `fold-len` puts a host
/// round-trip between two fires that could otherwise have been enqueued
/// together. So the count stays in a channel: the host plans against its own
/// UPPER BOUND (the row's whole live buffer) and the driver resolves the real
/// value and CLAMPS it to that bound.
///
/// Arm A commits from a channel it never awaits, computed as `1 + (argmax %
/// W)` over the append fire's own logits. Arm B awaits that argmax and commits
/// the identical count as a plain constant — the path that already worked. The
/// two must agree one token PAST the commit, which pins the folded STATE
/// rather than the logits along the way.
///
/// Arm C is the negative control: it folds a DIFFERENT count and must diverge,
/// or the equivalence holds vacuously and a dropped device value would pass
/// unnoticed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn the_fold_length_can_live_on_device() -> Result<()> {
    let result = run_foldcommit("device")
        .await?
        .map_err(|error| anyhow::anyhow!("device fold length failed: {error}"))?;
    anyhow::ensure!(
        result.contains("agree=yes"),
        "a commit whose fold length the host never saw must land the folded \
         boundary exactly where the same count as a constant lands it: {result}"
    );
    eprintln!("[gdn-foldcommit] {result}");
    Ok(())
}

/// The fold boundary lands STRICTLY INSIDE the fire's own new tokens.
///
/// `b < n < b + t` — the last RS position the planner refused, and the blocker
/// for a single-fire speculative verify+commit. `commit_len` cannot express
/// it: every kernel implements it as `if (c < Nr) Nr = c`, which TRUNCATES the
/// sequence, so the tokens past the boundary get no outputs at all. The driver
/// cuts the row into two segments and runs the recurrence twice on one stream
/// instead — the head persisting its end-of-sequence state onto the boundary,
/// the tail continuing from that state to produce the rest.
///
/// Both halves of that are pinned, because either can fail alone: the tail's
/// OUTPUTS must match a reference that computed them in a separate fire, and
/// the STATE the boundary left behind must match too. Arm C folds all four
/// tokens instead of two and must disagree, or an implementation that ignored
/// the boundary entirely would pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a GPU + Qwen3.5-0.8B (GDN backbone) in the HF cache"]
async fn the_fold_boundary_can_land_inside_a_fires_own_tokens() -> Result<()> {
    let result = run_foldcommit("interior")
        .await?
        .map_err(|error| anyhow::anyhow!("interior fold boundary failed: {error}"))?;
    anyhow::ensure!(
        result.contains("agree=yes"),
        "a fold landing inside a fire's own tokens must still produce every \
         token's outputs AND leave the boundary where it was asked: {result}"
    );
    eprintln!("[gdn-foldcommit] {result}");
    Ok(())
}

/// A commit that carries no tokens of its own.
///
/// "I have nothing to compute, only move the boundary" said DIRECTLY. The
/// planner currently infers it from `fold-len <= buffered`, which is an
/// incidental fact rather than a statement of intent — and it is the reason a
/// fire cannot fold BEHIND its own new tokens while writing them, since that
/// shape satisfies the same condition while meaning the opposite. A row with
/// zero tokens is unambiguous, and it frees `n <= b` to mean what it says.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA + a local Qwen3.5 checkout"]
async fn a_commit_may_carry_no_tokens_of_its_own() -> Result<()> {
    let result = run_foldcommit("empty")
        .await?
        .map_err(|error| anyhow::anyhow!("empty commit failed: {error}"))?;
    anyhow::ensure!(
        result.contains("agree=yes"),
        "a commit with an empty row must land the same folded state as one \
         padded with a placeholder token: {result}"
    );
    eprintln!("[gdn-foldcommit] {result}");
    Ok(())
}

/// A fire folds BEHIND its own new tokens while writing them.
///
/// The shape the fused speculative loop needs: window `k`'s accepted prefix
/// is only known after window `k` ran, so it is folded by the fire that
/// writes window `k + 1`. One fire per window in steady state instead of two.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires CUDA + a local Qwen3.5 checkout"]
async fn a_fire_may_fold_behind_the_tokens_it_is_writing() -> Result<()> {
    let result = run_foldcommit("behind")
        .await?
        .map_err(|error| anyhow::anyhow!("fold-behind failed: {error}"))?;
    anyhow::ensure!(
        result.contains("agree=yes"),
        "folding a prefix in the same fire that appends past it must match \
         doing the two in sequence: {result}"
    );
    eprintln!("[gdn-foldcommit] {result}");
    Ok(())
}
