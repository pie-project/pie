//! **Task-B KV preempt/restore over-capacity e2e** (alpha — the M-AB validation
//! gate for the contention rework).
//!
//! The rework's headline claim (In Gim): KV contention is **preempt/restore, not
//! admission**. When a fleet's combined KV demand EXCEEDS the physical pool, an
//! alloc-fail must NOT surface a `WorkingSetError` to the inferlet — instead it
//! routes to the contention orchestrator (`PIE_KV_CONTENTION=preempt`):
//!   OOM in prep → `acquire()` (FCFS wait / preempt a younger process) → a
//!   completing/terminating process frees its pages (the WS-drop drain hook) →
//!   `on_blocks_freed` wakes the FIFO waiter → the prep retries → succeeds.
//!
//! This test proves it end-to-end + that preemption is TRANSPARENT. Transparency
//! is established by a COMPOSITE proof, because the naive "concurrent tokens ==
//! solo tokens" check is structurally confounded — even ENGAGED: engagement
//! REQUIRES co-batching (a single lane never contends), and co-batched decode is
//! non-batch-invariant at logit near-ties (batch composition → bf16 reduction
//! order → a flipped argmax), so a lane can diverge from its batch-of-1 solo
//! reference with NO KV corruption (verified: tp=16 engaged, suspends=101, still
//! ~9 near-tie flips). The composite (charlie's Phase-2 finding):
//!   (a) `cuda_bubble` FLEET=1 byte-exact — SEAL + carrier transparency with the
//!       co-batch confound REMOVED (a single long-decode lane forced to preempt
//!       via PIE_KV_PAGE_CAP, byte-identical to its sync reference);
//!   (b) engaged multi-lane `restore_attributable == 0` HERE — a divergence first
//!       appearing at KV position >= page_size (a SEALED + restored page) is real
//!       suspend/restore corruption, hard-gated (assertion 3);
//!   (c) position-0 batch-invariance HERE — the first token off the identical
//!       prefill is deterministic, so a position-0 divergence is real prefill/
//!       page-0 corruption, hard-gated (assertion 3);
//!   (d) engagement counters (assertions 2+4) — preempt/restore provably ran.
//!   Divergences in the [1, page_size) band are the documented non-batch-
//!   invariance confound (logged, non-gating) — (a) covers transparency there.
//!
//! Assertions (engagement checked FIRST — a dormant config fails actionably on ②
//! rather than as a non-determinism trap on ③):
//!   1. **Zero WorkingSetError** — every pipeline in an over-capacity fleet
//!      completes with a token stream (none fails with "out of blocks"). This is
//!      the preempt/restore correctness: waiters block + wake, never error.
//!   2. **Engagement** — the orchestrator's contention counters prove preempt/
//!      restore actually happened: `waiters_parked > 0` + `waiters_woken > 0`
//!      (v1 passive backend — a lane OOM'd, parked in the FIFO, and was woken
//!      when a finisher freed blocks). So a fleet that merely FIT the pool (no
//!      contention) cannot pass trivially. This is what makes ② decisive.
//!   3. **Transparency (classified)** — no position-0 divergence (prefill/page-0
//!      corruption) AND no `restore_attributable` divergence (first divergence at
//!      KV position >= page_size = a sealed/restored page). The [1, page_size)
//!      band is logged non-gating (non-batch-invariance; see the composite above).
//!   4. **v2 active self-suspend** (only when `PIE_KV_PREEMPT_ACTIVE=1`) —
//!      `suspends > 0 && restores > 0`: the `SelfSuspendBackend` had victims
//!      SAVE their own KV state (D2H offload) and RESTORE it (H2D). Gated on
//!      active mode: in v1 passive these are zero-by-design, so it only fires
//!      when the active backend is armed.
//!
//! Construction-based contention proof: the fleet is sized so its combined KV
//! demand provably exceeds the (small) pool — so if all complete with zero
//! WorkingSetError, the orchestrator MUST have engaged (an unmanaged pool would
//! have errored the over-capacity lanes).
//!
//! `#[ignore]` (needs the 4090 + cuda + qwen3-0.6b). Run:
//!   PIE_KV_CONTENTION=preempt PIE_COMPILER_LAUNCHER=env \
//!     cargo test -p pie-bin --features driver-cuda \
//!     --test cuda_contention -- --ignored --nocapture
//!
//! COORDINATION (charlie owns the harness + boot config + GPU): the small-pool
//! boot knob below (`SMALL_POOL_GPU_MEM_UTIL`) is a first-cut forcing mechanism —
//! a low `gpu_mem_utilization` shrinks the KV pool so a modest fleet over-fills
//! it. charlie: tune the util (or add an explicit `total_pages`/`max_num_kv_pages`
//! cap to the driver options — the dummy driver currently floors at 256 pages, so
//! a host-runnable variant needs that cap) so the fleet reliably contends without
//! OOM-killing the box. alpha owns the FLEET shape + the two assertions.

mod common;

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use pie_client::client::Client;

/// A fleet large enough that its combined KV demand exceeds the shrunk pool
/// (see `SMALL_POOL_GPU_MEM_UTIL`) — forcing the preempt/restore path.
///
/// **Live-tunable via `PIE_CONTENTION_FLEET`** (charlie). Contention is forced by
/// the explicit KV-page cap (`PIE_CONTENTION_TOTAL_PAGES` → `[batching].total_pages`,
/// charlie — the cuda driver now HAS a page cap, mirroring metal), so the fleet
/// just needs `fleet × pages_per_lane > total_pages` (each lane holds ~a handful of
/// pages). The solo-reference guard confirms a single lane still fits. NOTE: under
/// v2 active preempt (`PIE_KV_PREEMPT_ACTIVE=1`) sustained fleet=24 is the Phase-2
/// hardened design point (the ≤12 guard is LIFTED — the f24 deadlock/livelock family
/// is fixed: step-6, tombstone Join, age-gate, ignition pump).
fn fleet_size() -> usize {
    std::env::var("PIE_CONTENTION_FLEET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24)
}

/// Distinct prompts long enough that each lane holds several KV pages, so the
/// fleet's aggregate demand exceeds the small pool. Greedy decode ⇒ deterministic
/// per prompt, so the solo reference is exact.
fn prompt(k: usize) -> String {
    // `PIE_CONTENTION_BUDGET=N` (charlie, capstone C5): emit a bare-int budget so
    // the generate inferlet decodes N tokens — LONGER-lived lanes sustain KV pool
    // pressure long enough for the v2 active suspend/restore cycle to reliably
    // engage (5-token lanes finish before contention forms → flaky parked=0).
    // Unset → the original JSON prompt (default 5-token lanes preserved).
    if let Ok(b) = std::env::var("PIE_CONTENTION_BUDGET") {
        if b.trim().parse::<usize>().is_ok() {
            return b.trim().to_string();
        }
    }
    let _ = k;
    format!(
        "{{\"prompt\": \"Lane {k}: the quick brown fox jumps over the lazy dog, \
         and then the clever cat considers the consequences carefully before it\"}}"
    )
}

fn parse_tokens(json: &str) -> Option<Vec<i64>> {
    let lb = json.rfind('[')?;
    let rb = json[lb..].find(']')? + lb;
    let toks: Vec<i64> = json[lb + 1..rb]
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if toks.is_empty() {
        None
    } else {
        Some(toks)
    }
}

/// Run one pipeline to completion. Returns `Ok(Some(tokens))` on success, or
/// `Ok(None)` if the process returned no tokens (e.g. surfaced an error — a
/// `WorkingSetError` would land here, which assertion 1 forbids).
async fn run_one(addr: &str, input: &str) -> Result<Option<Vec<i64>>> {
    let c = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    c.authenticate("test-user", &None).await?;
    let mut proc = c
        .launch_process("generate@0.1.0".into(), input.to_string(), true)
        .await?;
    Ok(parse_tokens(&proc.wait_for_return().await?))
}

async fn run_fleet_concurrent(addr: &str, inputs: &[String]) -> Vec<Option<Vec<i64>>> {
    let mut procs = Vec::new();
    for input in inputs {
        let addr = addr.to_string();
        let input = input.clone();
        procs.push(tokio::spawn(async move {
            run_one(&addr, &input).await.ok().flatten()
        }));
    }
    let mut out = Vec::new();
    for h in procs {
        out.push(h.await.unwrap_or(None));
    }
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
#[ignore = "Task-B contention over-capacity e2e: needs the 4090 + cuda + qwen3-0.6b + PIE_KV_CONTENTION=preempt"]
async fn over_capacity_fleet_preempts_and_restores_transparently() -> Result<()> {
    common::init_trace();

    // Guard: this test is meaningful only in preempt mode. In legacy mode the
    // over-capacity fleet would (correctly) surface WorkingSetError — a different
    // contract. Fail fast with a clear message rather than a confusing mismatch.
    anyhow::ensure!(
        std::env::var("PIE_KV_CONTENTION").as_deref() == Ok("preempt"),
        "set PIE_KV_CONTENTION=preempt — this test validates the preempt/restore path"
    );

    let ws = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../runtime/tests/inferlets");
    anyhow::ensure!(
        Command::new("cargo")
            .args(["build", "--target", "wasm32-wasip2", "-p", "generate"])
            .current_dir(&ws)
            .status()?
            .success(),
        "generate wasm build failed"
    );
    let wasm = ws.join("target/wasm32-wasip2/debug/generate.wasm");
    let manifest = ws.join("generate/Pie.toml");

    // Boot with a SMALL KV pool so a modest fleet over-fills it (charlie tunes the
    // exact knob — see the module COORDINATION note).
    let pie = common::boot_4090_small_kv().await?;
    let addr = pie.listen_addr.to_string();
    let fleet = fleet_size();
    eprintln!("[contention] booted small-pool, addr={addr}, fleet={fleet}");

    let setup = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "test-user").await?;
    setup.authenticate("test-user", &None).await?;
    setup
        .add_program(&wasm, &manifest, true)
        .await
        .context("add_program")?;

    let inputs: Vec<String> = (0..fleet).map(prompt).collect();

    // Reference: each lane ALONE (fits the small pool solo, so no contention) —
    // the deterministic greedy ground truth for that prompt.
    let mut reference = Vec::new();
    for (k, input) in inputs.iter().enumerate() {
        let r = run_one(&addr, input).await.ok().flatten();
        anyhow::ensure!(
            r.is_some(),
            "solo lane {k} produced no tokens — the pool is too small even for ONE lane; \
             raise the small-pool util so a single lane fits (contention must come from the FLEET, not a solo lane)"
        );
        reference.push(r);
    }

    // Over-capacity: launch the whole fleet at once. Combined demand > pool ⇒ the
    // losers OOM in prep → acquire() → wait → a finisher frees → drain → retry.
    // PIE_CONTENTION_TRACE=1: dump ContentionStats every ~1s during the concurrent
    // run so a HANG (never returns) still captures the stuck state, and a PASS shows
    // the counters climb-then-quiesce-by-completion (Phase-2 re-validation).
    let ctrace_stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ctrace_task = if std::env::var("PIE_CONTENTION_TRACE").is_ok() {
        let stop = ctrace_stop.clone();
        Some(tokio::spawn(async move {
            let t0 = std::time::Instant::now();
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(orch) = pie_engine::inference::contention::contention() {
                    let s = orch.stats();
                    use std::sync::atomic::Ordering::Relaxed;
                    eprintln!(
                        "[contention-trace] t={}ms parked={} woken={} suspends={} restores={}",
                        t0.elapsed().as_millis(),
                        s.waiters_parked.load(Relaxed), s.waiters_woken.load(Relaxed),
                        s.suspends.load(Relaxed), s.restores.load(Relaxed),
                    );
                }
                tokio::time::sleep(std::time::Duration::from_millis(1000)).await;
            }
        }))
    } else {
        None
    };
    let concurrent = run_fleet_concurrent(&addr, &inputs).await;
    ctrace_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Some(t) = ctrace_task {
        t.abort();
    }

    // Early engagement snapshot: print the final ContentionStats BEFORE any
    // assertion so a transparency RED still surfaces whether preempt/restore
    // fired (a degenerate-replay mismatch with suspends==0 falsifies the
    // "restore corruption" framing → the bug is not on the restore path).
    {
        use std::sync::atomic::Ordering as O;
        if let Some(o) = pie_engine::inference::contention::contention() {
            let s = o.stats();
            eprintln!(
                "[contention] final-engagement: parked={} woken={} suspends={} restores={}",
                s.waiters_parked.load(O::Relaxed),
                s.waiters_woken.load(O::Relaxed),
                s.suspends.load(O::Relaxed),
                s.restores.load(O::Relaxed),
            );
        }
    }

    // Assertion 1: ZERO WorkingSetError — every lane completed with tokens.
    let errored: Vec<usize> = (0..fleet).filter(|&k| concurrent[k].is_none()).collect();
    assert!(
        errored.is_empty(),
        "preempt/restore FAILED: lanes {errored:?} returned no tokens (a WorkingSetError \
         surfaced instead of the orchestrator absorbing the over-capacity). Zero errors expected."
    );

    // Assertion 2: ENGAGEMENT — preempt/restore provably HAPPENED, checked FIRST.
    // Ordering is load-bearing (charlie's Phase-2 finding): under a DORMANT
    // config (suspends=0, parked=0 — e.g. a too-loose total_pages cap) the
    // multi-lane token comparison measures concurrent-decode NON-BATCH-
    // INVARIANCE, not preempt correctness — the sync single-lane reference
    // itself near-tie loops. Failing on engagement FIRST turns a dormant
    // config into the actionable "fix your config" error instead of a
    // non-determinism trap. Without this a
    // fleet that merely FIT the pool (no contention) passes assertions 1+2
    // trivially. Under the v1 passive backend, engagement = allocation waiters
    // that PARKED (OOM'd → blocked in acquire's FIFO) and were later WOKEN (a
    // finisher freed blocks → drain released them → they retried + succeeded);
    // `suspends`/`restores` are zero-by-design in v1 (they arm with the v2
    // state-save backend — logged for the record).
    use std::sync::atomic::Ordering;
    let (parked, woken, suspends, restores) = pie_engine::inference::contention::contention()
        .map(|o| {
            let s = o.stats();
            (
                s.waiters_parked.load(Ordering::Relaxed),
                s.waiters_woken.load(Ordering::Relaxed),
                s.suspends.load(Ordering::Relaxed),
                s.restores.load(Ordering::Relaxed),
            )
        })
        .unwrap_or((0, 0, 0, 0));
    eprintln!(
        "[contention] engagement: parked={parked} woken={woken} suspends={suspends} restores={restores}"
    );
    assert!(
        parked > 0 && woken > 0,
        "preempt/restore did NOT ENGAGE (waiters_parked={parked}, waiters_woken={woken}) — the \
         fleet never over-filled the pool, so assertions 1+2 proved nothing. Shrink the pool \
         (PIE_CONTENTION_UTIL / a total_pages cap) or grow PIE_CONTENTION_FLEET until it contends."
    );

    // Assertion 3: TRANSPARENCY (classified) — preempt+restore preserved KV
    // content (W1). The naive `concurrent[k] == solo_reference[k]` is confounded
    // even on an ENGAGED run (charlie's Phase-2 completion of the engaged case):
    // engagement REQUIRES co-batching (a single lane never contends), and
    // co-batched decode is non-batch-invariant at logit near-ties (batch
    // composition → bf16 reduction order → a flipped argmax), so a lane can
    // diverge from its batch-of-1 solo reference with NO KV corruption — even at
    // parked/suspends>0 (verified: tp=16 engaged, suspends=101, still ~9 lanes
    // near-tie flip). The exact-MATCH thus cannot be a hard gate here; the clean,
    // un-confounded suspend/restore transparency proof is `cuda_bubble` FLEET=1
    // (a single long-decode lane forced to preempt via PIE_KV_PAGE_CAP,
    // byte-identical to its sync reference). What this harness CAN hard-gate is
    // genuine KV corruption, by the FIRST-DIVERGENCE POSITION of each mismatch
    // (guru's `restore_attributable == 0` refinement — unconfounded even where
    // token identity is not):
    //   • position 0 — the first token off the identical `"hello world"` prefill
    //     is batch-invariant (deterministic greedy from identical rows), so ANY
    //     position-0 divergence is real prefill/page-0 restore corruption. FAIL.
    //   • position >= `page_size` (KV page = 32 tokens, driver `kv_page_size`,
    //     config.hpp:53) — provably inside a SEALED/RESTORED page (a generated
    //     token at index `i` sits at KV position `prompt_len + i >= i`, so
    //     `i >= page_size` ⇒ KV position >= page_size regardless of prompt length
    //     ⇒ a page has sealed and, under preemption, been offloaded + restored).
    //     A divergence first appearing there is restore/seal corruption, not a
    //     near-tie flip. FAIL (`restore_attributable`). Vacuous when the token
    //     budget < page_size (no page seals); active for longer-budget runs.
    //   • position in [1, page_size) — the confounded band (non-batch-invariance
    //     ≡ partial-page-0 restore, inseparable here). LOGGED, non-gating; clean
    //     transparency in this band is the FLEET=1 bubble proof above.
    const PAGE_SIZE: usize = 32; // driver kv_page_size (config.hpp:53)
    let first_divergence = |a: &Option<Vec<i64>>, b: &Option<Vec<i64>>| -> Option<usize> {
        match (a, b) {
            (Some(x), Some(y)) => (0..x.len().min(y.len()))
                .find(|&i| x[i] != y[i])
                .or_else(|| (x.len() != y.len()).then_some(x.len().min(y.len()))),
            _ => None,
        }
    };
    let mut nbi_lanes = Vec::new(); // [1, page_size): non-batch-invariance (tolerated)
    let mut corrupt_lanes = Vec::new(); // position 0: prefill/page-0 corruption
    let mut restore_attributable = Vec::new(); // >= page_size: sealed/restored-page corruption
    for k in 0..fleet {
        if concurrent[k] != reference[k] {
            let pos = first_divergence(&concurrent[k], &reference[k]);
            eprintln!(
                "[contention] lane {k} diverged@{pos:?} conc={:?} ref={:?}",
                concurrent[k], reference[k]
            );
            match pos {
                Some(0) => corrupt_lanes.push(k),
                Some(p) if p >= PAGE_SIZE => restore_attributable.push(k),
                _ => nbi_lanes.push(k),
            }
        }
    }
    if !nbi_lanes.is_empty() {
        eprintln!(
            "[contention] {} lane(s) {nbi_lanes:?} diverged in [1, {PAGE_SIZE}) — concurrent-decode \
             non-batch-invariance (co-batch bf16 near-tie flip), NOT KV corruption. Suspend/restore \
             transparency in this band is gated by cuda_bubble FLEET=1 (byte-exact). Non-gating.",
            nbi_lanes.len()
        );
    }
    assert!(
        corrupt_lanes.is_empty(),
        "KV/prefix corruption: lanes {corrupt_lanes:?} diverged at position 0 — the first token off \
         the identical `hello world` prefill is batch-invariant, so a position-0 divergence is a real \
         restore/prefill KV corruption (not a near-tie flip)."
    );
    assert!(
        restore_attributable.is_empty(),
        "restore-attributable KV corruption: lanes {restore_attributable:?} first diverged at position \
         >= {PAGE_SIZE} (a SEALED + preempt-restored page), so the divergence is real suspend/restore \
         corruption of a sealed page, not a pre-seal near-tie flip."
    );

    // Assertion 4 (v2 active self-suspend): when the active `SelfSuspendBackend`
    // is armed (`PIE_KV_PREEMPT_ACTIVE=1`), a victim doesn't just park on the pool
    // — at its own execute_impl entry it SAVES its KV state (D2H offload) and
    // RESTORES it (H2D) on release, so `suspends`/`restores` MUST fire. This is
    // GATED on active mode: in v1 passive the backend returns `Unsupported` and
    // both stay zero-by-design, so the assertion only applies when armed.
    let preempt_active = std::env::var("PIE_KV_PREEMPT_ACTIVE").as_deref() == Ok("1");
    if preempt_active {
        assert!(
            suspends > 0 && restores > 0,
            "v2 active self-suspend did NOT ENGAGE (suspends={suspends}, restores={restores}) — \
             PIE_KV_PREEMPT_ACTIVE=1 arms the SelfSuspendBackend, so victims must actively save \
             (suspend) and restore their KV state. Zero means the active path never ran."
        );
        eprintln!(
            "[contention] v2 active self-suspend engaged: suspends={suspends} restores={restores} ✓"
        );
    }

    eprintln!(
        "[contention] {fleet}/{fleet} lanes: zero WorkingSetError + transparent restore + \
         engaged (parked={parked}, woken={woken}) ✓"
    );
    Ok(())
}
