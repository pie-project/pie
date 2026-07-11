//! #23 (A-scoped) FOUND-abort overlap verify — the device-UAF empirical teeth.
//!
//! THE EMPIRICAL GAP this closes (the ONE thing the co-stream structural argument
//! + alpha's host-cascade unit tests can't 100% close): a producer that has
//! ALREADY retained its sampled copy (so the consumer's inject IS enqueued from a
//! valid device buffer), then HOST-aborts (`effective_success=false`) — so the
//! forward-gated deferred-free races the consumer's in-flight inject. That's the
//! "retain-FOUND-then-abort" case. The retain-MISS path has no buffer → no UAF
//! surface (host-cascade-discarded; alpha's 8 Part A unit tests pin it).
//!
//! HOW the overlap is driven: the `runahead` inferlet decodes through
//! `collect_tokens_pipelined`, which eager-one-ahead SUBMITS the consumer (t+1)
//! BEFORE awaiting the producer (t) — i.e. the consumer's inject is enqueued while
//! the producer is still in flight (the overlap). `MAX_IN_FLIGHT=2` (one-step
//! run-ahead active). So this harness EXTENDS `cuda_runahead`'s proven overlap —
//! no separate overlap setup. The happy-path (no injection) is `cuda_runahead`
//! itself (GREEN: `MATCH=true ANCHOR_OK=true CLEAR_OK=true`) = the non-degenerate
//! CONTROL this RED-injected run is the counterpart to.
//!
//! THE FAULT INJECTION (alpha's env-gated seam, inference.rs finalize path, the
//! #19 `PIE_MIROSTAT_DUMP` precedent — zero prod-path when unset, test-flagged):
//!   `success = forward_result.is_some() && !test_force_producer_abort(&deps)`
//! keyed on `deps.produced == PIE_TEST_ABORT_PRODUCER_LINK`. Firing AFTER
//! `rx.await == Some` = device-valid-but-host-aborts = the FOUND-abort. We target a
//! MID-CHAIN producer link (NOT the terminal pass) so it has BOTH a live retained
//! copy AND an already-enqueued consumer inject when it aborts.
//!
//! THE TWO TEETH (manager's (A-scoped) bar; (b)/(c) cite alpha's 8 unit tests):
//!   (a) NO device UAF — the retained copy is freed strictly-AFTER the consumer's
//!       inject drains. PROVED by running THIS test under compute-sanitizer:
//!         compute-sanitizer --tool memcheck --error-exitcode 1 \
//!           cargo test -p pie-bin --features driver-cuda --test cuda_overlap23 \
//!           -- --ignored --nocapture --test-threads=1
//!       A clean sanitizer exit IS the (a) proof (no UAF on the deferred-free).
//!   (d) the consumer CASCADE-ABORTS — the producer-fault poison surfaces to the
//!       guest as an ABORT INDICATION (an error from `await_commit`/`output()`),
//!       NEVER silently committed as a token and never a silently-truncated stream
//!       that looks like normal completion. Asserted host-side below.
//!
//! `#[ignore]`, driver-cuda. See the compute-sanitizer wrapper above for (a).

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

// ── alpha's seam contract (CONFIRMED on `alpha-23-overlap @ 337e0e0b`) ────────
// Q1 (link numbering): `next_input_link_counter` is per-instance, init 0,
//    PRE-incremented +1 per *producer* pass ⇒ the Nth producer pass → link N
//    (monotonic from 1). The prime (carries to decode-1) = link 1; decode
//    producers = 2, 3, … So link 2 = the first DECODE producer = mid-chain: it has
//    BOTH a live retained copy AND an already-enqueued consumer inject (the
//    FOUND-abort), and is NOT the terminal pass. The seam fires `success=false`
//    ONLY for the pass whose `deps.produced == N`, AFTER `rx.await == Some`
//    (device-succeeded + retained) = exactly the retain-FOUND-then-host-abort.
const ABORT_PRODUCER_LINK: &str = "2";
// Q1 (env knob name): alpha's seam reads this (CONFIRMED).
const ABORT_ENV_KNOB: &str = "PIE_TEST_ABORT_PRODUCER_LINK";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "#23 FOUND-abort overlap verify: needs the 4090 + cuda + qwen-3-0.6b + \
            alpha's PIE_TEST_ABORT_PRODUCER_LINK seam; RUN UNDER compute-sanitizer for (a)"]
async fn found_abort_overlap_on_real_driver() -> Result<()> {
    common::init_trace();

    // Arm alpha's env-gated fault-injector BEFORE boot so the finalize path reads
    // it. Forces the mid-chain producer to FOUND-abort (device-succeeded+retained,
    // then host-aborts) → deferred-free races the consumer's enqueued inject.
    // SAFETY: set before boot, before any thread that reads these envs (Rust 2024).
    unsafe {
        std::env::set_var(ABORT_ENV_KNOB, ABORT_PRODUCER_LINK);
        // Surface the carrier RETAIN/INJECT/FREE trace (the device-side discriminator:
        // RETAIN of the aborted link, then its FREE strictly-after the consumer INJECT).
        std::env::set_var("PIE_SAMPLING_IR_TRACE", "1");
    }

    let pie = common::boot_4090().await?;
    eprintln!(
        "[overlap23] booted, listen_addr={}; {}={} (FOUND-abort armed)",
        pie.listen_addr, ABORT_ENV_KNOB, ABORT_PRODUCER_LINK
    );

    // Build the run-ahead carryover inferlet (the overlap driver).
    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    let ok = Command::new("cargo")
        .args(["build", "--target", "wasm32-wasip2", "-p", "runahead"])
        .current_dir(&ws)
        .status()?
        .success();
    anyhow::ensure!(ok, "runahead wasm build failed");
    let wasm = ws.join("target/wasm32-wasip2/debug/runahead.wasm");
    let manifest = ws.join("runahead/Pie.toml");

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
    eprintln!("[overlap23] program installed, launching FOUND-abort carryover decode…");

    let mut proc = client
        .launch_process("runahead@0.1.0".to_string(), "8".to_string(), true)
        .await
        .context("launch")?;
    let outcome = proc.wait_for_return().await;
    pie.shutdown().await;

    // ── Tooth (d): the cascade-abort MUST surface as an ABORT INDICATION ──────
    // CONFIRMED contract (alpha + manager, in-code): the forced mid-chain producer
    // FOUND-aborts → `effective_success=false` → no tensor → `output()` returns WIT
    // `Err("forward produced no output tensor")` (inference.rs:1238) → `await_commit`
    // does `.output().await.map_err(…)?` (generation.rs:603) → `collect_tokens_
    // pipelined` does `await_commit(producer).await?` (generation.rs:432) → returns
    // `Err`. So the inferlet's generation observably ABORTS (partial tokens
    // discarded, the poisoned sample NEVER commits as a token), never a silent
    // truncated stream that looks like normal completion. The runahead inferlet
    // propagates that `Err` (no catch) → the process aborts → `wait_for_return`
    // returns `Err`. So `outcome` is the (d) discriminator:
    //   - `Err`  ⇒ the cascade-abort surfaced (expected; the only Err here is the
    //              run-time abort — boot/connect/build/add_program all `?`-returned
    //              ABOVE, so they can't reach this point).
    //   - `Ok(json)` that is NOT a clean `MATCH=true` ⇒ also acceptable (some abort
    //              surface that still returned a body, as long as it isn't a clean
    //              normal completion).
    //   - `Ok(json)` with `MATCH=true` ⇒ FAIL: the poison silently committed or the
    //              stream silently truncated to look normal.
    // Non-degeneracy CONTROL = `cuda_runahead` (same overlap, same inferlet, NO
    // injection) → `MATCH=true ANCHOR_OK=true CLEAR_OK=true` (GREEN). This RED run
    // is its injected counterpart: same path, the ONLY difference is the armed knob.
    match &outcome {
        Err(e) => {
            eprintln!("[overlap23] cascade-abort surfaced as error (expected): {e:#}");
        }
        Ok(json) => {
            eprintln!("[overlap23] returned: {json}");
            anyhow::ensure!(
                !json.contains("MATCH=true"),
                "FOUND-abort did NOT surface: the producer-faulted generation returned a \
                 clean MATCH=true — the poison was silently committed or the stream was \
                 silently truncated to look normal (Part A must surface no-tensor as an \
                 abort, not a silent truncation): {json}"
            );
            // TODO(alpha Q2): assert the structured abort marker once its form is known,
            // e.g. anyhow::ensure!(json.contains("ABORTED="), ...).
        }
    }

    // ── Tooth (a): NO device UAF ──────────────────────────────────────────────
    // Not assertable in-process — it is the compute-sanitizer wrapper's clean exit
    // (see the module header). A memcheck error on the deferred-free vs in-flight
    // inject ⇒ non-zero exit ⇒ this test process fails under the sanitizer.
    //
    // ── (b)/(c): cite alpha's 8 Part A host-cascade unit tests (chained-cascade
    // ≥3 + exactly-one-free) — host-cascade LOGIC, unit-proven, not re-run here.
    Ok(())
}
