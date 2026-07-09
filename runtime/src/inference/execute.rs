//! Forward-pass execution pipeline — moved out of `api/inference.rs`
//! (mechanical relocation; logic unchanged). The engine machinery behind
//! the WIT inference host: `execute_impl`, the forward-transaction
//! lifecycle, contention/preempt, and response marshaling. `api/inference.rs`
//! keeps the thin WIT resource types + Host trait impls.
#![allow(unused_imports)]

use crate::api::pie;
use crate::inference::ForwardOutput;
use crate::inference::paging;
use crate::grammar::compiled_grammar::CompiledGrammar;
use crate::grammar::grammar::Grammar as InternalGrammar;
use crate::grammar::json_schema::{
    JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar,
};
use crate::grammar::matcher::GrammarMatcher;
use crate::grammar::regex::regex_to_grammar;
use crate::instance::InstanceState;
use crate::inference;
use anyhow::Result;
use pie_driver_abi::Brle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use wasmtime::component::Resource;
use wasmtime::component::{Accessor, HasSelf};
use wasmtime_wasi::WasiView;
use crate::api::inference::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecuteProfileSnapshot {
    pub calls: u64,
    pub hits: u64,
    pub misses: u64,
    pub total_us: u64,
    pub prepare_us: u64,
    pub hit_wait_us: u64,
    pub cold_prepare_us: u64,
    pub pin_us: u64,
    pub submit_wait_us: u64,
    pub postprocess_us: u64,
}

pub(crate) struct ExecuteProfileStats {
    calls: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    total_us: AtomicU64,
    prepare_us: AtomicU64,
    hit_wait_us: AtomicU64,
    cold_prepare_us: AtomicU64,
    pin_us: AtomicU64,
    submit_wait_us: AtomicU64,
    postprocess_us: AtomicU64,
}

static EXECUTE_PROFILE: ExecuteProfileStats = ExecuteProfileStats {
    calls: AtomicU64::new(0),
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    total_us: AtomicU64::new(0),
    prepare_us: AtomicU64::new(0),
    hit_wait_us: AtomicU64::new(0),
    cold_prepare_us: AtomicU64::new(0),
    pin_us: AtomicU64::new(0),
    submit_wait_us: AtomicU64::new(0),
    postprocess_us: AtomicU64::new(0),
};

pub(crate) fn execute_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_PROFILE_EXECUTE").is_some())
}

pub(crate) fn elapsed_us(duration: Duration) -> u64 {
    duration.as_micros() as u64
}

pub fn execute_profile_snapshot() -> Option<ExecuteProfileSnapshot> {
    if !execute_profile_enabled() {
        return None;
    }
    Some(ExecuteProfileSnapshot {
        calls: EXECUTE_PROFILE.calls.load(Ordering::Relaxed),
        hits: EXECUTE_PROFILE.hits.load(Ordering::Relaxed),
        misses: EXECUTE_PROFILE.misses.load(Ordering::Relaxed),
        total_us: EXECUTE_PROFILE.total_us.load(Ordering::Relaxed),
        prepare_us: EXECUTE_PROFILE.prepare_us.load(Ordering::Relaxed),
        hit_wait_us: EXECUTE_PROFILE.hit_wait_us.load(Ordering::Relaxed),
        cold_prepare_us: EXECUTE_PROFILE.cold_prepare_us.load(Ordering::Relaxed),
        pin_us: EXECUTE_PROFILE.pin_us.load(Ordering::Relaxed),
        submit_wait_us: EXECUTE_PROFILE.submit_wait_us.load(Ordering::Relaxed),
        postprocess_us: EXECUTE_PROFILE.postprocess_us.load(Ordering::Relaxed),
    })
}

#[derive(Default)]
pub(crate) struct ExecuteProfileSample {
    hit: bool,
    prepare_us: u64,
    hit_wait_us: u64,
    cold_prepare_us: u64,
    pin_us: u64,
    submit_wait_us: u64,
    postprocess_us: u64,
}

pub(crate) fn record_execute_profile(sample: ExecuteProfileSample, total_us: u64) {
    if !execute_profile_enabled() {
        return;
    }
    EXECUTE_PROFILE.calls.fetch_add(1, Ordering::Relaxed);
    if sample.hit {
        EXECUTE_PROFILE.hits.fetch_add(1, Ordering::Relaxed);
    } else {
        EXECUTE_PROFILE.misses.fetch_add(1, Ordering::Relaxed);
    }
    EXECUTE_PROFILE
        .total_us
        .fetch_add(total_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .prepare_us
        .fetch_add(sample.prepare_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .hit_wait_us
        .fetch_add(sample.hit_wait_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .cold_prepare_us
        .fetch_add(sample.cold_prepare_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .pin_us
        .fetch_add(sample.pin_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .submit_wait_us
        .fetch_add(sample.submit_wait_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .postprocess_us
        .fetch_add(sample.postprocess_us, Ordering::Relaxed);
}

/// WASM-facing forward-pass accumulator. The WIT methods append/set
/// into `req: pie_driver_abi::ForwardRequest` directly — at `execute()`
/// we just finalize the per-request indptrs and submit. `model_id`
/// is WASM-side routing info (not on the wire) and `adapter_seed`
/// is stored separately because it doesn't have its own WIT setter
/// but is used at execute-time to populate the adapter binding.
pub(crate) fn empty_forward_request() -> pie_driver_abi::ForwardRequest {
    pie_driver_abi::ForwardRequest {
        adapter_bindings: vec![pie_driver_abi::AdapterBinding {
            adapter_id: -1,
            seed: -1,
        }],
        output_spec_flags: vec![false],
        // Image side-channel CSR roots: the per-image pixel/mrope indptrs and
        // the per-request image_indptr all begin with a leading 0 so the
        // batch-merge in `inference::request` can offset and append cleanly.
        image_indptr: vec![0],
        image_pixel_indptr: vec![0],
        image_mrope_indptr: vec![0],
        // Audio side-channel CSR roots (leading 0, like the image roots).
        audio_feature_indptr: vec![0],
        audio_indptr: vec![0],
        ..Default::default()
    }
}

/// Number of bytes in one element of a `tensor` dtype.
pub(crate) struct ForwardTxnGuard {
    txn: Option<crate::arena::ArenaTxn>,
    model_id: usize,
    driver_idx: usize,
}

impl ForwardTxnGuard {
    fn new(txn: crate::arena::ArenaTxn, model_id: usize, driver_idx: usize) -> Self {
        Self {
            txn: Some(txn),
            model_id,
            driver_idx,
        }
    }

    /// Hand the txn to the normal `finalize_forward_txn` commit/abort path; the
    /// guard is left empty so its `Drop` is a no-op.
    fn take(&mut self) -> Option<crate::arena::ArenaTxn> {
        self.txn.take()
    }
}

impl Drop for ForwardTxnGuard {
    fn drop(&mut self) {
        if let Some(txn) = self.txn.take() {
            // Un-finalized drop (error-return / proc-terminate before finalize):
            // abort the txn to release the kv/rs working-set pins. The normal
            // path already `take()`-d it, so this never double-handles (a
            // committed/aborted txn is consumed by value). `crate::arena::get` is
            // self-contained (no store/accessor needed), so it is safe in `Drop`;
            // the lock is only taken here when finalize did NOT run, so it cannot
            // deadlock against `finalize_forward_txn`.
            let arena_arc = crate::arena::get(self.model_id, self.driver_idx);
            if let Ok(mut arena) = arena_arc.lock() {
                arena.txn_abort(txn);
            }
        }
    }
}

/// In-flight forward state carried from the eager submit across the async
/// driver round-trip to the await→finalize. Phase-2
/// ([`finalize_forward_output`]) consumes it. The owned `txn` keeps the pins /
/// CoW copies alive until commit/abort; `rx` is the driver completion handle.
///
/// Today `execute` builds this then finalizes inline (single WIT method). The
/// 1c run-ahead surface (Option A — the forward-pass IS the in-flight handle)
/// stores it on the forward-pass so the async `output()`/`outputs()` awaits it;
/// the eager `execute` releases its `&mut ctx` borrow at that point so the loop
/// can hold two passes in flight (the `submit_async`→scheduler boundary).
pub(crate) struct PendingForward {
    /// The driver round-trip. `Option` so the two consumers can split: the
    /// async `output()`/`outputs()` path takes it to await; the contention
    /// drain (`drain_retired_fires`) polls it in place (`try_recv`) and
    /// finalizes early when the response already arrived.
    rx: Option<tokio::sync::oneshot::Receiver<Result<ForwardOutput>>>,
    txn: ForwardTxnGuard,
    kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>>,
    /// S4: this forward's KV write-transaction id, so commit/abort at finalize
    /// touches only this forward's repointed slots (`None` ⇒ no KV writes).
    kv_write_txn: Option<crate::working_set::kv::WriteTxnId>,
    seal_hashes: Vec<(u32, u64)>,
    model_id: usize,
    driver_idx: usize,
    rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>>,
    fold_buffered_tokens: Option<u32>,
    /// W11 in-forward RS write: the folded slot staged on the shared txn +
    /// its plan, ADOPTED as the working set's current folded state only in
    /// finalize's COMMIT branch (an abort discards it — prior state stays).
    rs_write: Option<(
        Resource<crate::working_set::rs::RsWorkingSet>,
        crate::working_set::rs::RsWritePlan,
    )>,
    profile_start: Option<Instant>,
    /// #23 overlap abort-isolation: the producer link this pass produced and the
    /// prior producer link it consumed (injected from). At finalize the write-log
    /// uses these to cascade-abort the consumer if its producer aborted, and to
    /// publish this pass's own outcome for its consumer.
    next_input_deps: crate::inference::runahead::NextInputDeps,
}

/// #23 verify (TEST-ONLY, env-gated): force a designated *producer* pass to report
/// failure. Evaluated at the finalize success-determination — AFTER `rx.await`
/// resolved `Some` (the producer's forward device-succeeded and **retained** its
/// sampled token, and in the run-ahead overlap the consumer's inject is already
/// enqueued from that valid retained copy) — so flipping it to failure reproduces
/// the **retain-FOUND-then-host-abort** path: the producer's drain-gated
/// deferred-free races the in-flight inject (compute-sanitizer "free
/// strictly-after-drain"), and the consumer cascade-aborts fail-closed
/// (token-for-token). One mid-chain knob exercises both #23 teeth.
///
/// Keyed on the producer's monotonic link via `PIE_TEST_ABORT_PRODUCER_LINK`
/// (read once). **UNSET ⇒ always `false` ⇒ ZERO production behavior** — the #19
/// `PIE_MIROSTAT_DUMP` env-instrument pattern (test-only; flagged for the land
/// guard). Non-producer passes (no `produced` link) are never targeted.
pub(crate) fn test_force_producer_abort(deps: &crate::inference::runahead::NextInputDeps) -> bool {
    static ABORT_LINK: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let target = *ABORT_LINK.get_or_init(|| {
        std::env::var("PIE_TEST_ABORT_PRODUCER_LINK")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
    });
    abort_target_matches(deps.produced, target)
}

/// Pure targeting predicate for [`test_force_producer_abort`] (env-free, so it is
/// unit-testable): abort iff a target link is configured AND this pass is the
/// producer for it. An unset target (`None`) never matches ⇒ zero production
/// behavior; a non-producer pass (`produced = None`) is never targeted.
pub(crate) fn abort_target_matches(produced: Option<u32>, target: Option<u32>) -> bool {
    target.is_some() && produced == target
}

/// The post-round-trip half of [`await_and_finalize`], callable with direct
/// store access: finalize the forward txn (commit/abort + seal/fold) and
/// reconstruct the declared output tensors. Split out so the contention drain
/// ([`drain_retired_fires`]) can finalize a fire whose response ALREADY
/// arrived from inside the parked/gated task itself — that finalize is what
/// releases the fire's pins and grace refs, which is the whole reclaim story
/// under a deep-running carrier.
pub(crate) fn finalize_received(
    state: &mut InstanceState,
    pending: PendingForward,
    forward_result: Option<ForwardOutput>,
) -> Result<()> {
    let PendingForward {
        rx: _,
        mut txn,
        kv_set,
        kv_write_txn,
        seal_hashes,
        model_id,
        driver_idx,
        rs_fold_set,
        fold_buffered_tokens,
        rs_write,
        profile_start,
        next_input_deps,
    } = pending;
    // #23 verify (A-scoped, env-gated): force a designated producer's forward to
    // report failure AFTER it device-succeeded + retained, reproducing the
    // retain-FOUND-then-abort UAF race for the compute-sanitizer harness. UNSET ⇒
    // no-op (zero production behavior); see `test_force_producer_abort`.
    let success = forward_result.is_some() && !test_force_producer_abort(&next_input_deps);

    // Take the txn out of its guard for the normal commit/abort. The now-empty
    // guard's `Drop` is a no-op; the leak-abort only fires if this finalize is
    // never reached (error-return / proc-terminate).
    let forward_txn = txn
        .take()
        .expect("forward txn consumed exactly once at finalize");
    // #23: finalize resolves the overlap cascade (a consumer whose producer
    // aborted is forced to abort even on driver success). Programmable sampling +
    // its outputs now live entirely in `ptir`, so there is nothing to marshal
    // back here — the forward only commits/aborts its KV/RS txn.
    let _effective_success = state.finalize_forward_txn(
        success,
        forward_txn,
        kv_set,
        kv_write_txn,
        seal_hashes,
        model_id,
        driver_idx,
        rs_fold_set,
        fold_buffered_tokens,
        rs_write,
        next_input_deps,
    )?;
    if let Some(start) = profile_start {
        record_execute_profile(ExecuteProfileSample::default(), elapsed_us(start.elapsed()));
    }
    Ok(())
}

pub(crate) async fn self_suspend_park_restore(
    state: &mut InstanceState,
    pid: crate::process::ProcessId,
    set: &Resource<crate::working_set::kv::KvWorkingSet>,
    model_id: usize,
    driver_idx: usize,
    orch: &crate::inference::contention::ContentionOrchestrator,
) -> u32 {
    // Step 1: STAGE our own pages (allocate CPU dests; GPU blocks stay held +
    // resident so the copy reads valid data), on our own table.
    let arena_arc = crate::arena::get(model_id, driver_idx);
    let cas_arc = crate::working_set::kv_cas::get(model_id, driver_idx);
    let mut plan = {
        let mut arena = arena_arc.lock().unwrap();
        match state.ctx().table.get_mut(set) {
            Ok(ws) => Some(ws.stash_pages_warm(&mut arena)),
            Err(_) => None,
        }
    };
    let mut plan = match plan.take() {
        Some(p) => p,
        None => return 0,
    };
    // Step 2: issue the D2H stash copies WHILE the GPU blocks are still held
    // (the stash-free-before-copy race fix — commit only frees them afterwards).
    for (_slot, _id, mv) in &plan.stash {
        if let Err(e) = crate::driver::copy_d2h(driver_idx, &mv.from, &mv.to) {
            tracing::warn!("preempt D2H stash copy failed: {e:#}");
        }
    }
    // Step 3: NOW free the GPU blocks + repoint to the stash, ref-release shared,
    // set slots Reserved. Nothing to yield (freed_now==0) ⇒ caller decides.
    let freed_now = {
        let mut arena = arena_arc.lock().unwrap();
        let mut cas = cas_arc.lock().unwrap();
        match state.ctx().table.get_mut(set) {
            Ok(ws) => ws.commit_suspend(&mut plan, &mut arena, &mut cas),
            Err(_) => 0,
        }
    };
    if freed_now == 0 {
        return 0;
    }

    // Steps 2b+3, looped: report → park → restore. On a restore-race OutOfBlocks
    // (still fully stashed — restore_pages_warm is all-or-nothing) re-report the
    // SAME freed_now and re-park.
    const MAX_RESTORE_REPARKS: u32 = 64;
    let mut reparks = 0u32;
    loop {
        orch.report_suspended(pid, freed_now);
        orch.park_until_restored(pid).await;
        let restore = {
            let arena_arc = crate::arena::get(model_id, driver_idx);
            let mut arena = arena_arc.lock().unwrap();
            match state.ctx().table.get_mut(set) {
                Ok(ws) => Some(ws.restore_pages_warm(&mut arena, &plan)),
                Err(_) => None,
            }
        };
        match restore {
            Some(Ok(moves)) => {
                for (_slot, mv) in &moves {
                    if let Err(e) = crate::driver::copy_h2d(driver_idx, &mv.to, &mv.from) {
                        tracing::warn!("preempt H2D restore copy failed: {e:#}");
                    }
                }
                break;
            }
            Some(Err(crate::working_set::kv::WorkingSetError::OutOfBlocks { .. }))
                if reparks < MAX_RESTORE_REPARKS =>
            {
                reparks += 1;
                continue;
            }
            // Any other restore failure: RE-PARK AND RETRY — never proceed
            // un-restored. Proceeding leaves the stashed slots `Reserved`:
            // on the generate path the page table silently SKIPS them ⇒ the
            // model decodes with a truncated context ⇒ the degenerate replay
            // (BAR-1's 6/24 corrupt lanes); on the carrier path the same
            // state is the loud "no written page". Silence here is
            // corruption; the retry cap fails LOUD below instead.
            Some(Err(e)) if reparks < MAX_RESTORE_REPARKS => {
                tracing::warn!("preempt restore failed (re-parking to retry): {e}");
                reparks += 1;
                continue;
            }
            Some(Err(e)) => {
                // Retry budget exhausted: give up LOUDLY. The set stays
                // stashed (restore_pages_warm is all-or-nothing), so reads of
                // its written slots fail visibly rather than silently
                // truncating the context.
                tracing::error!(
                    "preempt restore failed {MAX_RESTORE_REPARKS}x — lane proceeds \
                     un-restored and WILL fail loud on its next written-slot read: {e}"
                );
                break;
            }
            None => break, // WS gone (teardown) — nothing to restore.
        }
    }
    freed_now
}

/// Task-B (carrier ⋈ contention): finalize every one of this instance's OWN
/// in-flight fires whose driver response has ALREADY arrived (non-blocking
/// `try_recv`), storing each result as the pass's `harvested` for its later
/// `output()`/`outputs()`. This is the only path by which a gated/parked
/// lane's pins and grace refs can release: finalize is process-local by design
/// (B-refined self-suspend — no cross-process table access), so no other task
/// can run it, and a lane blocked in `execute` can never reach its own
/// `output()` calls. Returns the number of fires still genuinely in flight
/// (response not yet arrived). Stale reps (a pass dropped without `output()`)
/// are pruned lazily.
pub(crate) fn drain_retired_fires(state: &mut InstanceState) -> Result<usize> {
    use tokio::sync::oneshot::error::TryRecvError;
    enum Disp {
        Prune,
        InFlight,
        Ready(Option<ForwardOutput>),
    }
    let reps: Vec<u32> = state.pending_fires.clone();
    let mut in_flight = 0usize;
    for rep in reps {
        let res: Resource<ForwardPass> = Resource::new_borrow(rep);
        let disp = match state.ctx().table.get_mut(&res) {
            Err(_) => Disp::Prune,
            Ok(pass) => match pass.pending.as_mut() {
                None => Disp::Prune,
                Some(p) => match p.rx.as_mut() {
                    None => Disp::Prune,
                    Some(rx) => match rx.try_recv() {
                        Ok(Ok(resp)) => Disp::Ready(Some(resp)),
                        Ok(Err(e)) => {
                            tracing::warn!("drained fire failed: {e:#}");
                            Disp::Ready(None)
                        }
                        Err(TryRecvError::Empty) => Disp::InFlight,
                        Err(TryRecvError::Closed) => Disp::Ready(None),
                    },
                },
            },
        };
        match disp {
            Disp::Prune => state.pending_fires.retain(|&r| r != rep),
            Disp::InFlight => in_flight += 1,
            Disp::Ready(forward_result) => {
                let pending = state
                    .ctx()
                    .table
                    .get_mut(&res)?
                    .pending
                    .take()
                    .expect("pending present: checked above");
                finalize_received(state, pending, forward_result)?;
                state.pending_fires.retain(|&r| r != rep);
            }
        }
    }
    Ok(in_flight)
}

/// Task-B (carrier ⋈ contention) gate: while the pool is contended (a waiter
/// is parked) or this lane is itself park-requested, a deep-running lane must
/// STOP DEEPENING — drain its own retired fires and wait for the rest to
/// retire, so its pins and grace refs release and `classify_for_suspend` can
/// actually yield its pages (a deep carrier otherwise re-pins every context
/// page each fire ⇒ `freed_now→0` ⇒ the C6+carrier starve). Returns when this
/// lane has ZERO un-retired fires or contention has cleared (`force` waits to
/// zero unconditionally — used on the OOM path where the pool is full by
/// definition). Liveness: submitted fires retire autonomously (device
/// progress; scheduler-stashed ones ride waves / deadline dummy-fill), and
/// every retire re-notifies via the scheduler response-dispatch hook. The
/// enable-then-drain-then-await pattern prevents lost wakeups.
pub(crate) async fn drain_own_fires(
    state: &mut InstanceState,
    orch: &crate::inference::contention::ContentionOrchestrator,
    pipeline_id: crate::process::ProcessId,
    force: bool,
) -> Result<()> {
    // Hot-path guard: no drain at all while uncontended — finalize keeps its
    // usual read-order timing (byte-identical to the pre-gate behavior).
    if !force && !(orch.contended() || orch.should_park(pipeline_id)) {
        return Ok(());
    }
    // Gate-phase wave-Leave (the drain-gate analog of the plain-park Leave in
    // `acquire`): a lane blocked HERE awaiting its OWN retires is not submitting,
    // yet it stays in the wave-set — so it accrues straggler misses and the
    // miss-limit would TERMINATE a live, progressing carrier lane (the BAR-2
    // "unresponsive pipeline" false-kill). Emit `Leave{Suspend}` the first time
    // we actually block (so its `dep_stash` parks, resumed on rejoin) and `Join`
    // on exit. This is the PURE replacement for the demote-handler terminate-skip
    // — with the lane out of the wave while gate-blocked, it is never demoted, so
    // the skip is unreachable. The Join is emitted on EVERY exit path (incl. the
    // drain error) so a gate-blocked lane is never left out of the wave-set.
    let mut left_wave = false;
    let result = loop {
        let notify = orch.fire_retired();
        let notified = notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        match drain_retired_fires(state) {
            Err(e) => break Err(e),
            Ok(0) => break Ok(()),
            Ok(_) => {}
        }
        if !force && !(orch.contended() || orch.should_park(pipeline_id)) {
            break Ok(());
        }
        // About to block on a retire → leave the wave (first block only).
        if !left_wave {
            crate::inference::scheduler::notify_pipeline_leave(
                pipeline_id,
                crate::inference::scheduler::LeaveKind::Suspend,
            );
            left_wave = true;
        }
        notified.await;
    };
    if left_wave {
        crate::inference::scheduler::notify_pipeline_join(pipeline_id);
    }
    result
}

/// #21 eager-submit (free fn so the SYNC `HostForwardPass::execute` can call it
/// with `&mut InstanceState` — the sync trait has no `accessor`). Prepares +
/// submits the forward and stores the in-flight [`PendingForward`] on the
/// forward-pass; the async `output()`/`outputs()` await + finalize it. A
/// recoverable prepare/submit failure is stored as the pass's `exec_error`
/// (surfaced by `output()`), NOT returned here — `execute: func()` has no error
/// channel. The outer `Result` is reserved for unrecoverable host traps.
pub(crate) async fn execute_impl(
    state: &mut InstanceState,
    this: Resource<ForwardPass>,
) -> Result<()> {
        let profile_start = execute_profile_enabled().then(Instant::now);
        // M-A1 (wait-for-all): the submitting pipeline's ProcessId (wave membership
        // key). Captured up-front before the accessor borrows `state`.
        let pipeline_id = state.id();
        // Drain the accumulator: the explicit memory descriptors + the staged
        // ForwardRequest. There is no ambient context handle (W5). Every store /
        // resource-table touch in this async func goes through `accessor.with`
        // (P3: the host async fn has no `&mut self`).
        let (
            model_id,
            adapter_seed,
            kv_ws,
            rs_ws,
            fold_buffered_tokens,
            mut req,
            next_input_positions,
            fresh_generate,
        ) = {
            let pass = state.ctx().table.get_mut(&this)?;
            (
                pass.model_id,
                pass.adapter_seed,
                pass.kv_ws.take(),
                pass.rs_ws.take(),
                pass.fold_buffered_tokens.take(),
                std::mem::replace(&mut pass.req, empty_forward_request()),
                std::mem::take(&mut pass.next_input_positions),
                pass.fresh_generate,
            )
        };
        // v1: single-driver. Multi-driver binds the working set's device on
        // first materialization (`bind_driver`), wired at consolidation.
        let driver_idx = 0usize;

        // M-B2 v2 active self-suspend (B-refined). If this pipeline has been
        // FCFS-picked as a preempt victim, it saves its OWN KV state HERE — at its
        // own host-call boundary, with its own resource-table access (so there is
        // NO cross-process Arena→WS lock-order hazard) — frees the pages for the
        // blocked requester, parks until the restore phase releases it, then
        // re-materialises and proceeds with its forward. Gated by
        // `PIE_KV_PREEMPT_ACTIVE` (the active `SelfSuspendBackend`); a no-op on the
        // proven passive v1 path (`should_park` is only ever true once a backend
        // returned `SuspendOutcome::Requested`). Every arena/WS lock is dropped
        // before the `.await` park — guru's invariant: hold NO lock across a park.
        if let Some(orch) = crate::inference::contention::contention() {
            // Gate FIRST (carrier ⋈ contention): under contention (or a park
            // request on us) drain our own retired fires and stop deepening,
            // so the classify below sees UNPINNED pages — a deep lane's pins
            // would otherwise defer its whole working set to grace and the
            // suspend would decline forever (the C6+carrier livelock).
            drain_own_fires(state, orch, pipeline_id, false).await?;
            if orch.should_park(pipeline_id) {
                // Save this forward's working set (the pages in hand), report the
                // freed blocks, park until restored, then re-materialise. If we
                // freed NOTHING (no reclaimable page, or grace-blocked), decline
                // the park so we clear ParkRequested→Running instead of leaking it.
                let freed = if let Some(desc) = kv_ws.as_ref() {
                    self_suspend_park_restore(
                        state,
                        pipeline_id,
                        &desc.set,
                        model_id,
                        driver_idx,
                        orch,
                    )
                    .await
                } else {
                    0
                };
                if freed == 0 {
                    orch.decline_park(pipeline_id);
                }
            }
        }
        // Empty-input guard: a forward must compute at least one query row.
        // Without input rows `qo_indptr` collapses to `[0, 0]` and the pass is a
        // no-op submit; the old context API rejected this. Image/audio spans
        // push placeholder rows into `token_ids`, so this covers all input kinds.
        if let Err(e) = paging::check_input_nonempty(req.token_ids.len()) {
            // Defer to `output()` — `execute: func()` has no error channel.
            state.ctx().table.get_mut(&this)?.exec_error = Some(format!("{e:?}"));
            return Ok(());
        }

        // WIT spec: "if not provided, fallback to causal mask". Then stamp the
        // per-request indptr shape ([0, N]).
        let has_user_mask = !req.masks.is_empty();
        if req.masks.is_empty() && !req.position_ids.is_empty() {
            req.masks = req
                .position_ids
                .iter()
                .map(|&pos| Brle::all_true((pos + 1) as usize))
                .collect();
        }
        req.has_user_mask = has_user_mask;
        req.single_token_mode = !has_user_mask && req.token_ids.len() <= 1;
        req.adapter_bindings[0].seed = adapter_seed.unwrap_or(-1);
        req.qo_indptr = vec![0, req.token_ids.len() as u32];
        req.mask_indptr = vec![0, req.masks.len() as u32];
        req.logit_mask_indptr = vec![0, req.logit_masks.len() as u32];
        req.sampling_indptr = vec![0, req.sampling_indices.len() as u32];
        req.sampler_indptr = vec![0, req.n_samplers() as u32];
        req.spec_indptr = vec![0, req.spec_token_ids.len() as u32];
        req.kv_page_indptr = vec![0];
        // Batch-affinity id: the KV working set's resource handle replaces the
        // old context id (used by the scheduler for request grouping).
        let affinity = kv_ws.as_ref().map(|d| d.set.rep()).unwrap_or(0);
        req.context_ids = vec![affinity as u64];

        let page_size = crate::working_set::page_size::tokens_per_page(model_id);

        // ── prepare: validate descriptors; alloc/CoW + pin write targets; pin
        //    read pages; resolve to driver physical ids — all under one txn.
        //    The whole prepare is synchronous, so it runs inside one
        //    `accessor.with` closure (store + arena both reachable); the owned
        //    `txn` + projection cross back out for the async submit. ──
        type PrepOut = (
            paging::KvProjection,
            crate::arena::ArenaTxn,
            Vec<crate::arena::MovePlan>,
            Option<crate::working_set::kv::WriteTxnId>,
            Option<crate::working_set::rs::RsWritePlan>,
        );
        // Task-B contention (v1): route KV-pool exhaustion to the preempt/restore
        // orchestrator + RETRY the prepare, instead of failing the inferlet. The
        // retry loop wraps the WHOLE `'prepare` — every per-attempt artifact (the
        // arena txn, `move_plans`, the KV write-txn, the rs plan) is created inside
        // the loop body, so a retry starts clean; `req` only mutates on the success
        // path (after the inner `Ok`), so it carries across attempts unchanged.
        // Bounded so a pathological loser fails loud (never spins).
        const MAX_CONTENTION_RETRIES: u32 = 32;
        let mut contention_attempts = 0u32;
        let prepared: std::result::Result<PrepOut, String> = loop {
        // `Some(need)` iff this attempt failed on KV-pool exhaustion — the one
        // arena error routed to the orchestrator (all others fail loud below).
        let mut preempt_need: Option<u32> = None;
        let attempt: std::result::Result<PrepOut, String> = 'prepare: {
            let arena_arc = crate::arena::get(model_id, driver_idx);
            let mut arena = arena_arc.lock().unwrap();
            let mut txn = arena.txn_begin();
            // S4: this forward's KV write-transaction (opened lazily when it first
            // writes a slot). Keys the working set's abort-revert log so this
            // forward's commit/abort is isolated from any concurrently-prepared
            // forward against disjoint slots of the same working set.
            let mut kv_write_txn: Option<crate::working_set::kv::WriteTxnId> = None;
            let mut move_plans: Vec<crate::arena::MovePlan> = Vec::new();

            type InnerOut = (
                paging::KvProjection,
                Vec<u32>,
                Vec<u8>,
                Vec<u32>,
                Vec<u32>,
                Vec<u32>,
                Option<crate::working_set::rs::RsWritePlan>,
            );
            let inner: std::result::Result<InnerOut, String> = 'prep: {
                // KV read context → pinned physical pages. Read only the written
                // valid-token prefix: kv-context `len` may include trailing
                // reserved slots (the WIT permits "trailing reserved slots may be
                // empty") that are not part of attention and may be unwritten;
                // resolving the full `len` would reject them. `valid_pages` is the
                // ceil of valid-tokens. Pure prefill (valid_tokens==0) reads none.
                let (context_pages, valid_tokens) = if let Some(d) = &kv_ws {
                    let valid_pages = d.valid_tokens.div_ceil(page_size);
                    let objs = if valid_pages == 0 {
                        Vec::new()
                    } else {
                        match state.ctx().table.get(&d.set) {
                            Ok(ws) => match ws.resolve_read(d.inp_start, valid_pages) {
                                Ok(o) => o,
                                Err(e) => break 'prep Err(e.to_string()),
                            },
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    };
                    let mut pages = Vec::with_capacity(objs.len());
                    for obj in &objs {
                        if let Err(e) = arena.txn_pin(&mut txn, *obj) {
                            break 'prep Err(e.to_string());
                        }
                        match arena.blocks(*obj) {
                            Ok(b) => pages.push(b[0]),
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    }
                    (pages, d.valid_tokens)
                } else {
                    (Vec::new(), 0)
                };

                // KV write outputs → CoW'd + pinned physical pages. #21 Option-B
                // adapter: the write is the CONTIGUOUS slot range `[output_start,
                // output_start+output_len)`. Per-page valid-len is reconstructed
                // from `offset` (the in-page token offset of the first written
                // row) + `n` (the input-token count): page `i` holds
                // `clamp((offset+n) − i·page_size, 0, page_size)` valid tokens.
                // No indices array, no generation/range guard (dropped under #21
                // — the inferlet owns working-set correctness). `cow_write_slot`
                // CoW-chains from the slot's existing object, preserving any
                // in-flight producer prefix (alpha review check #1).
                let mut writes: Vec<paging::KvWrite> = Vec::new();
                if let Some(d) = &kv_ws {
                    let n = req.token_ids.len() as u32;
                    // Open this forward's write-txn on the target working set (S4).
                    let wtx = {
                        let ws = match state.ctx().table.get_mut(&d.set) {
                            Ok(w) => w,
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        let wtx = ws.begin_write_txn();
                        kv_write_txn = Some(wtx);
                        wtx
                    };
                    for i in 0..d.output_len {
                        let idx = d.output_start + i;
                        let valid_len = (d.offset + n)
                            .saturating_sub(i * page_size)
                            .min(page_size);
                        let cow = {
                            let ws = match state.ctx().table.get_mut(&d.set) {
                                Ok(w) => w,
                                Err(e) => break 'prep Err(e.to_string()),
                            };
                            ws.cow_write_slot(wtx, idx, &mut txn, &mut arena)
                        };
                        let (obj, move_plan) = match cow {
                            Ok(v) => v,
                            // Task-B: KV pool exhausted — flag `need` for the
                            // contention retry (the ONLY arena error so routed;
                            // every other prep failure falls through to fail loud).
                            Err(crate::working_set::kv::WorkingSetError::OutOfBlocks {
                                requested,
                                ..
                            }) => {
                                preempt_need = Some(requested);
                                break 'prep Err("kv pool exhausted (contention retry)".to_string());
                            }
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        if let Some(mp) = move_plan {
                            move_plans.push(mp);
                        }
                        if let Err(e) = arena.txn_pin(&mut txn, obj) {
                            break 'prep Err(e.to_string());
                        }
                        let page = match arena.blocks(obj) {
                            Ok(b) => b[0],
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        writes.push(paging::KvWrite {
                            slot_index: idx,
                            page,
                            valid_len,
                        });
                    }
                }

                // Project onto the contiguous driver page run + last-page length.
                let proj = match paging::project_kv(
                    &context_pages,
                    valid_tokens,
                    &writes,
                    page_size,
                ) {
                    Ok(p) => p,
                    Err(e) => break 'prep Err(format!("{e:?}")),
                };

                // RS v1 — converged in-forward DIRECT write to the folded
                // recurrent-state slot (the GDN MTP forward's `write_state` path:
                // NO buffered slabs). `rs_buffer_slot_ids` stays EMPTY ⇒ the driver
                // writes the new recurrent state straight into the folded slot.
                // `rs_slot_ids[r]` is that folded slot; `RS_FLAG_RESET` zeroes a
                // freshly-allocated slab on the first fire of a sequence.
                let mut rs_slot_ids: Vec<u32> = Vec::new();
                let mut rs_slot_flags: Vec<u8> = Vec::new();
                let rs_buffer_slot_ids: Vec<u32> = Vec::new();
                let mut rs_buffer_slot_indptr: Vec<u32> = vec![0];
                let mut rs_fold_lens: Vec<u32> = Vec::new();
                let mut rs_write_plan: Option<crate::working_set::rs::RsWritePlan> = None;
                // `rs_fold_buffered` is parked (Ph7): the converged in-forward
                // DIRECT-write path keeps NO buffer, so there is nothing to fold.
                // Fail LOUD rather than silently no-op the driver fold while finalize
                // still advances the HOST fold boundary — that host/driver divergence
                // is exactly the silently-wrong class this rewrite removes (echo
                // Finding A). Restores the fail-loud the old marshal had.
                if fold_buffered_tokens.is_some() {
                    break 'prep Err(
                        "rs_fold_buffered is parked (Ph7): the in-forward write path keeps no buffer to fold"
                            .to_string(),
                    );
                }
                if let Some(rs) = &rs_ws {
                    // Stage the folded slot on the SHARED txn (alpha's
                    // `prepare_write_in_txn`): a fresh + reset slab on the first
                    // fire, else a CoW of the prior folded state. Replaces the
                    // `folded_object().unwrap_or(0)` stub that shipped slot 0 + no
                    // RESET for a fresh GDN sequence.
                    let mut plan = {
                        let ws = match state.ctx().table.get_mut(&rs.set) {
                            Ok(w) => w,
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        match ws.prepare_write_in_txn(&mut txn, &mut arena) {
                            Ok(p) => p,
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    };
                    // Shared folded slab (first write after a fork, W11): copy it
                    // device-side before the write, via the KV move-plan channel.
                    if let Some(mp) = plan.cow_move.take() {
                        move_plans.push(mp);
                    }
                    let folded_block = match arena.blocks(plan.folded_slot) {
                        Ok(b) => b[0],
                        Err(e) => break 'prep Err(e.to_string()),
                    };
                    rs_slot_ids.push(folded_block);

                    let mut flag = 0u8;
                    if plan.reset {
                        flag |= pie_driver_abi::RS_FLAG_RESET;
                    }
                    rs_slot_flags.push(flag);
                    // No in-forward fold on the direct-write path (fold parked Ph7).
                    rs_fold_lens.push(0);
                    // Empty buffer for the direct-write path (write_state = true).
                    rs_buffer_slot_indptr.push(rs_buffer_slot_ids.len() as u32);

                    // Stash the plan → `adopt_write(&plan)` in finalize's COMMIT
                    // branch (post-commit-durable), so the folded slot is only
                    // adopted after the forward's write commits (aborts revert it).
                    rs_write_plan = Some(plan);
                }

                Ok((
                    proj,
                    rs_slot_ids,
                    rs_slot_flags,
                    rs_buffer_slot_ids,
                    rs_buffer_slot_indptr,
                    rs_fold_lens,
                    rs_write_plan,
                ))
            };

            // On any prepare failure: abort the txn (discard staged allocs/CoW
            // copies, release pins) and revert any repointed KV slots; the prior
            // mappings stay visible (W13). No driver submission happened.
            let (proj, rs_slot_ids, rs_slot_flags, rs_buffer_slot_ids, rs_buffer_slot_indptr, rs_fold_lens, rs_write_plan) =
                match inner {
                    Ok(v) => v,
                    Err(e) => {
                        arena.txn_abort(txn);
                        drop(arena);
                        if let (Some(d), Some(wtx)) = (&kv_ws, kv_write_txn) {
                            if let Ok(ws) = state.ctx().table.get_mut(&d.set) {
                                ws.abort_writes(wtx);
                            }
                        }
                        break 'prepare Err(e);
                    }
                };

            if !rs_slot_ids.is_empty() {
                req.rs_slot_ids = rs_slot_ids;
                req.rs_slot_flags = rs_slot_flags;
                req.rs_fold_lens = rs_fold_lens;
            }
            if !rs_buffer_slot_ids.is_empty() {
                req.rs_buffer_slot_ids = rs_buffer_slot_ids;
                req.rs_buffer_slot_indptr = rs_buffer_slot_indptr;
            }

            // Release the arena lock BEFORE the async submit; the owned `txn`
            // keeps the pins/CoW copies alive until commit/abort.
            drop(arena);
            Ok((proj, txn, move_plans, kv_write_txn, rs_write_plan))
        }; // end 'prepare

        match attempt {
            Ok(v) => break Ok(v),
            Err(e) => match (preempt_need, crate::inference::contention::contention()) {
                // KV-pool exhaustion + preempt mode wired: await the orchestrator
                // (FCFS-preempt a younger process / wait for a free), then RETRY
                // the whole prepare. The prepare aborted its STAGED txn — but a
                // long lane's working set still holds its committed pages, and
                // its earlier fires' un-finalized txns still hold pins; both are
                // handled below before we can block.
                (Some(need), Some(orch)) => {
                    // Carrier ⋈ contention: we just OOMed — the pool is full by
                    // definition. FULLY drain our own fires first (force=true):
                    // each finalize releases that fire's pins + grace refs, so
                    // the classify below sees the true yieldable set and a
                    // subsequent park never freezes un-finalized own fires.
                    drain_own_fires(state, orch, pipeline_id, true).await?;
                    // Does this requester itself hold pages it could yield? Only
                    // then can the orchestrator route us to SELF-SUSPEND (the
                    // fleet=24/8 deadlock-breaker: a parked page-HOLDER strands its
                    // pages forever) instead of parking as before.
                    let holds_reclaimable = kv_ws.as_ref().is_some_and(|desc| {
                        let arena_arc = crate::arena::get(model_id, driver_idx);
                        let arena = arena_arc.lock().unwrap();
                        state
                            .ctx()
                            .table
                            .get(&desc.set)
                            .map(|ws| ws.has_reclaimable_pages(&arena))
                            .unwrap_or(false)
                    });
                    match orch
                        .acquire_or_self_suspend(pipeline_id, need, holds_reclaimable)
                        .await
                    {
                        Ok(crate::inference::contention::Acquired::Retry)
                            if contention_attempts < MAX_CONTENTION_RETRIES =>
                        {
                            contention_attempts += 1;
                            continue;
                        }
                        // No victim can yield + we hold pages: self-suspend our OWN
                        // set (free the stranded pages for the blocked fleet), park
                        // until restored, re-materialise, then retry.
                        Ok(crate::inference::contention::Acquired::SelfSuspendFirst)
                            if contention_attempts < MAX_CONTENTION_RETRIES =>
                        {
                            let freed = if let Some(desc) = kv_ws.as_ref() {
                                self_suspend_park_restore(
                                    state,
                                    pipeline_id,
                                    &desc.set,
                                    model_id,
                                    driver_idx,
                                    orch,
                                )
                                .await
                            } else {
                                0
                            };
                            if freed == 0 {
                                // The `holds_reclaimable` gate raced into a
                                // pin/grace defer: nothing yielded and NO park
                                // happened. Park as a plain v1 waiter — block on
                                // the next free event rather than hot-spinning
                                // the prepare loop through the retry budget.
                                if orch.acquire(pipeline_id, need).await.is_err() {
                                    break Err(e);
                                }
                            }
                            contention_attempts += 1;
                            continue;
                        }
                        // Retries exhausted, OR the request can never fit
                        // (`ContentionError::Impossible`): fail loud via the legacy
                        // `exec_error` path — never spin.
                        _ => break Err(e),
                    }
                }
                // Not a pool exhaustion, or legacy mode (no orchestrator) → the
                // unchanged legacy error path.
                _ => break Err(e),
            },
        }
        };

        let (proj, txn, move_plans, kv_write_txn, rs_write_plan) = match prepared {
            Ok(v) => v,
            Err(e) => {
                // Recoverable prepare failure — defer it to `output()` (the WIT
                // `execute: func()` has no error channel). The txn / KV slots were
                // already aborted/reverted in the prepare block.
                state.ctx().table.get_mut(&this)?.exec_error = Some(e);
                return Ok(());
            }
        };

        // Issue the device d2d for every CoW'd write target: copy the original
        // page content into the private copy before the driver writes into it.
        //
        // Bug#2 diagnostic (concurrent identical-prompt contamination): these
        // fire-and-forget copies (`copy_d2d` = `ch.notify`) are the only place
        // the host relies on the driver ordering a KV page-copy BEFORE the
        // forward that consumes the copied page. If contamination correlates
        // with a non-empty copy plan, the cross-message copy/forward ordering is
        // the suspect; if decodes contaminate with an EMPTY plan, CoW is not
        // involved and the fault is device-side batched-attention page
        // attribution. The counter lets the GPU repro settle which.
        if !move_plans.is_empty() {
            let pages: usize = move_plans.iter().map(|mp| mp.from.len()).sum();
            tracing::info!(
                target: "ptir::cow",
                driver = driver_idx,
                copies = move_plans.len(),
                pages,
                "forward issued CoW d2d copies (Bug#2 diagnostic)"
            );
        }
        for mp in &move_plans {
            if let Err(e) = crate::driver::copy_d2d(driver_idx, &mp.from, &mp.to) {
                tracing::warn!("forward CoW d2d copy failed: {e:#}");
            }
        }

        // CAS-seal eligibility (W6/W7): host-hash the full pages this forward
        // fills from an EMPTY context (`valid_tokens == 0`), so the forward's
        // tokens start at a page boundary and each page's whole content is known
        // here. Chained from prev_hash 0. Sealing onto a non-empty context needs
        // the context tip's sealed hash (follow-up); until then those pages stay
        // private-dirty (W7) — correct, just no dedup.
        let seal_eligible = !proj.full_page_writes.is_empty()
            && kv_ws.as_ref().map(|d| d.valid_tokens).unwrap_or(0) == 0
            && req.position_ids.len() == req.token_ids.len()
            && req.masks.len() == req.token_ids.len();
        let seal_hashes: Vec<(u32, u64)> = if seal_eligible {
            let page_hashes = crate::working_set::page_hash::compute_page_hashes(
                page_size as usize,
                &req.token_ids,
                &req.position_ids,
                &req.masks,
                0,
                adapter_seed,
            );
            proj
                .full_page_writes
                .iter()
                .filter_map(|&slot| page_hashes.get(slot as usize).map(|&h| (slot, h)))
                .collect()
        } else {
            Vec::new()
        };

        // Carry the KV working-set handle + the txn across the async boundary;
        // finalize (after the driver round-trip) commits (seal full pages) /
        // aborts on them.
        let kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>> = kv_ws
            .as_ref()
            .map(|d| Resource::new_borrow(d.set.rep()));

        // The RS working set whose folded boundary advances on a committed
        // fold-buffered (W9) — v1 rides the rs-working-set.
        let rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>> =
            if fold_buffered_tokens.is_some() {
                rs_ws.as_ref().map(|d| Resource::new_borrow(d.set.rep()))
            } else {
                None
            };

        // W11 in-forward RS write: pair the folded-slot plan staged on the shared
        // txn with a borrow of its working set, carried to finalize. `adopt_write`
        // runs ONLY in the COMMIT branch, so an aborted forward never adopts the
        // staged folded state (the prior state stays current).
        let rs_write: Option<(
            Resource<crate::working_set::rs::RsWorkingSet>,
            crate::working_set::rs::RsWritePlan,
        )> = match (rs_ws.as_ref(), rs_write_plan) {
            (Some(d), Some(plan)) => Some((Resource::new_borrow(d.set.rep()), plan)),
            _ => None,
        };

        // #21 next-inputs carrier: register this pass as a pipeline source (if it
        // declared `next-inputs`) + inject the prior producer's retained sample
        // into this pass. The host owns the global link id; the guest threads
        // none. No-op when neither role applies. Must run AFTER `input-tokens` is
        // staged (it reads `req.token_ids.len()`) and BEFORE submit. The context id
        // (KV working-set rep) scopes the carryover to consecutive same-context
        // passes so a terminal producer's dangling carry can't leak into the next
        // context (0 = no-KV pass ⇒ never a carrier consumer/producer).
        let next_input_context_id = kv_ws.as_ref().map(|d| d.set.rep()).unwrap_or(0);
        // #26 fresh-generate: BEFORE the carrier's consumer-inject, drop any
        // dangling carry left on THIS context by a prior generate's terminal
        // producer (stop-terminal / explicit-restart). Free the stale retained
        // device buffer on this prime's request so it doesn't leak; the prime
        // then does NOT inject the stale token. A different context's pending is
        // left untouched for `apply_next_input_carrier`'s own mismatch branch.
        if fresh_generate {
            // #17 (the FLEET=8 carrier+preempt regression, bisected to the
            // free-all): DRAIN-BEFORE-FREE-ALL. The free contract requires a
            // link be freed only after its LAST consumer's inject drained —
            // the driver is count-agnostic and frees strictly on this signal
            // (executor.cpp "next_input_free_links"). The depth-k free-all
            // below pushes EVERY link on the context, including links whose
            // deep pre-submitted consumers haven't even fired; freed-under ⇒
            // retain-MISS ⇒ #23 cascade-abort ⇒ guest retry churn (the
            // requests=1 crawl). Draining our own fires to zero FIRST makes
            // the precondition hold: every registered consumer has fired and
            // its inject drained (retire-then-free ≡ the intended discard —
            // same finalize, only the timing moves; the terminal dangling
            // carry survives the drain and is reclaimed by the free-all
            // below). Poll-based: no orchestrator dependency, and the
            // generation boundary is latency-insensitive. Caps loud.
            // Gate-phase Leave/Join (same contract as `drain_own_fires`): a lane
            // blocked in this poll loop is not submitting, and with the demote
            // terminate-skip retired, a wave-resident lane would accrue straggler
            // misses here and be miss-limit-TERMINATED mid-drain. Leave on the
            // first actual block; Join on every exit path (incl. the drain error).
            let mut left_wave = false;
            let mut drain_err = None;
            let mut spins = 0u32;
            loop {
                match drain_retired_fires(state) {
                    Err(e) => {
                        drain_err = Some(e);
                        break;
                    }
                    Ok(0) => break,
                    Ok(in_flight) => {
                        spins += 1;
                        if spins > 20_000 {
                            tracing::error!(
                                "fresh-generate drain: {in_flight} own fire(s) still \
                                 un-retired after ~10s — proceeding to free-all; a \
                                 retain-miss may follow"
                            );
                            break;
                        }
                    }
                }
                if !left_wave {
                    crate::inference::scheduler::notify_pipeline_leave(
                        pipeline_id,
                        crate::inference::scheduler::LeaveKind::Suspend,
                    );
                    left_wave = true;
                }
                tokio::time::sleep(std::time::Duration::from_micros(500)).await;
            }
            if left_wave {
                crate::inference::scheduler::notify_pipeline_join(pipeline_id);
            }
            if let Some(e) = drain_err {
                return Err(e.into());
            }
            if let Some(link) = crate::inference::runahead::clear_pending_for_context(
                &mut state.pending_next_input,
                next_input_context_id,
            ) {
                req.push_next_input_free_link(link);
            }
            // #23: a fresh generate starts a clean dependency chain — drop any
            // lingering terminal-producer link from the prior generation.
            state.overlap_links.clear();
            // Depth-k rollback (spec §4): free ALL of the prior generation's carrier
            // producer links on this context, not just the dangling `pending`. A
            // run-ahead STOP over-shoot drops ≤depth−1 speculative fires; the last
            // COMMITTED fire's retained carry is orphaned (its free rode a dropped
            // fire's request). Free-all reclaims it. Idempotent + drain-gated on the
            // driver, so re-freeing already-freed links is a safe no-op.
            for link in crate::inference::runahead::take_produced_links_for_context(
                &mut state.carrier_produced_links,
                next_input_context_id,
            ) {
                req.push_next_input_free_link(link);
            }
        }
        let next_input_deps = crate::inference::runahead::apply_next_input_carrier(
            &mut state.pending_next_input,
            &mut req,
            &next_input_positions,
            next_input_context_id,
        );
        // Depth-k rollback bookkeeping: record every producer link created on this
        // context so a later fresh-`generate()` free-alls them (above).
        if let Some(prod) = next_input_deps.produced {
            crate::inference::runahead::record_produced_link(
                &mut state.carrier_produced_links,
                next_input_context_id,
                prod,
            );
        }
        // #17 orphans-only: this pass's consumer role just freed the link it injected
        // from (host refcount = 1 — `apply_next_input_carrier` emitted its
        // `next_input_free_link`). Drop it from the produced set so the fresh-generate
        // free-all above reclaims ONLY the orphaned (un-consumed) links, never the
        // already-freed chain history (the "too broad" re-free shape).
        if let Some(consumed) = next_input_deps.consumed {
            crate::inference::runahead::remove_produced_link(
                &mut state.carrier_produced_links,
                next_input_context_id,
                consumed,
            );
        }

        // Single-model: the SERVICE routes to the bound model; no model_id arg.
        let submit_result = inference::submit_async(
            req,
            driver_idx,
            proj.physical_page_ids,
            proj.last_page_len,
            Vec::new(),
            Some(pipeline_id),
        );

        let rx = match submit_result {
            Ok(rx) => rx,
            Err(e) => {
                // Submit never reached the driver — abort the txn + revert the
                // repointed KV slots (W13).
                let arena_arc = crate::arena::get(model_id, driver_idx);
                arena_arc.lock().unwrap().txn_abort(txn);
                if let (Some(ko), Some(wtx)) = (&kv_set, kv_write_txn) {
                    if let Ok(ws) = state.ctx().table.get_mut(ko) {
                        ws.abort_writes(wtx);
                    }
                }
                tracing::warn!("inference::submit failed: {e:#}");
                // Defer to `output()` — `execute: func()` has no error channel.
                state.ctx().table.get_mut(&this)?.exec_error = Some(e.to_string());
                return Ok(());
            }
        };

        // Await + finalize (commit) the forward INLINE: publish the KV writes
        // before `execute()` returns. Programmable sampling + its outputs now live
        // entirely in the `ptir` pipeline, so a forward-pass carries no sampled
        // token to defer — it is always standalone (never a pipeline producer).
        let forward_result = match rx.await {
            Ok(Ok(resp)) => Some(resp),
            Ok(Err(e)) => {
                tracing::warn!("forward failed: {e:#}");
                None
            }
            Err(_) => None,
        };
        let success = forward_result.is_some() && !test_force_producer_abort(&next_input_deps);
        state.finalize_forward_txn(
            success,
            txn,
            kv_set,
            kv_write_txn,
            seal_hashes,
            model_id,
            driver_idx,
            rs_fold_set,
            fold_buffered_tokens,
            rs_write,
            next_input_deps,
        )?;
        let _ = profile_start;
        Ok(())
    }

impl InstanceState {
    /// Commit (on driver success) or abort (on failure) the forward transaction
    /// from `execute()`. Commit releases pins, publishes the CoW'd write targets
    /// (`commit_writes`), and CAS-seals eligible full pages; abort discards
    /// staged objects and reverts repointed slots.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn finalize_forward_txn(
        &mut self,
        success: bool,
        txn: crate::arena::ArenaTxn,
        kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>>,
        kv_write_txn: Option<crate::working_set::kv::WriteTxnId>,
        seal_hashes: Vec<(u32, u64)>,
        model_id: usize,
        driver_idx: usize,
        rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>>,
        fold_tokens: Option<u32>,
        rs_write: Option<(
            Resource<crate::working_set::rs::RsWorkingSet>,
            crate::working_set::rs::RsWritePlan,
        )>,
        next_input_deps: crate::inference::runahead::NextInputDeps,
    ) -> Result<bool> {
        // #23 overlap abort-isolation: resolve the cascade. A consumer that
        // injected from a producer link that did NOT explicitly commit (aborted OR
        // unresolved — fail-closed) is forced to abort even on driver success, so a
        // poisoned generation never commits its txn/KV. The write-log also records
        // THIS pass's (effective) outcome under its produced link, chaining the
        // poison downstream. This is device-drain-neutral (host txn/KV only — it
        // never touches the device `retained_next_input` consumer count).
        let success = self.overlap_links.finalize(success, next_input_deps);

        // Carry rollback: a terminal producer that aborted with its carry still
        // pending (no consumer took it in the overlap) leaves a dangling carry —
        // clear it (by its unique global link) so a later same-context pass doesn't
        // inject the aborted sample. (The consumed case is already covered: the
        // consumer emitted the drain-gated free-link at inject; here we only catch
        // the un-consumed terminal.) The device retained buffer is freed by the
        // next pass's `clear_pending_for_context` (no leak).
        if !success {
            if let Some(prod) = next_input_deps.produced {
                self.pending_next_input.retain(|_, p| p.link != prod);
            }
        }

        let arena_arc = crate::arena::get(model_id, driver_idx);
        if success {
            // Commit, publish the repointed slots, then CAS-seal eligible full
            // pages. Lock order is arena → kv_cas, both held sync (no await).
            let mut arena = arena_arc.lock().unwrap();
            arena
                .txn_commit(txn)
                .map_err(|e| anyhow::anyhow!("forward txn_commit failed: {e}"))?;
            if let Some(kv_set) = &kv_set {
                let cas_arc = crate::working_set::kv_cas::get(model_id, driver_idx);
                let mut cas = cas_arc.lock().unwrap();
                if let Ok(ws) = self.ctx().table.get_mut(kv_set) {
                    if let Some(wtx) = kv_write_txn {
                        ws.commit_writes(wtx);
                    }
                    for (slot, hash) in &seal_hashes {
                        if let Err(e) = ws.seal(*slot, *hash, &mut arena, &mut cas) {
                            tracing::warn!("CAS seal of slot {slot} failed: {e}");
                        }
                    }
                }
            }
            // Advance the RS folded boundary on a committed in-forward fold (W9):
            // consume the first `n` buffered tokens into the folded state. Only
            // on success — a fold never advances across an aborted forward.
            if let (Some(n), Some(rs_set)) = (fold_tokens, &rs_fold_set) {
                if let Ok(ws) = self.ctx().table.get_mut(rs_set) {
                    if let Err(e) = ws.advance_fold(n, &mut arena) {
                        tracing::warn!("advance_fold({n}) failed: {e:?}");
                    }
                }
            }

            // W11: adopt the freshly-written folded slot as the RS working set's
            // current folded state — POST-commit, so the durable state only
            // advances after the forward's device write is committed.
            if let Some((rs_set, plan)) = &rs_write {
                match self.ctx().table.get_mut(rs_set) {
                    Ok(ws) => ws.adopt_write(plan),
                    // The guest dropped the RS set mid-flight: the committed folded
                    // slab is orphaned until teardown (bounded). Surface it (echo
                    // Nit C) rather than skip silently.
                    Err(e) => {
                        tracing::warn!("rs adopt_write skipped — rs set gone: {e}");
                    }
                }
            }
        } else {
            {
                let mut arena = arena_arc.lock().unwrap();
                arena.txn_abort(txn);
            }
            if let Some(kv_set) = &kv_set {
                if let Ok(ws) = self.ctx().table.get_mut(kv_set) {
                    if let Some(wtx) = kv_write_txn {
                        ws.abort_writes(wtx);
                    }
                }
            }
        }
        // Task-B contention: finalize may have freed KV pool blocks — a
        // `txn_abort` released this forward's staged allocs (a failed forward,
        // incl. the OOM path), or a CAS-seal hit freed a duplicate page. The
        // arena lock is released above, so it is safe to wake FCFS waiters /
        // restore suspended processes now. No-op unless PIE_KV_CONTENTION=preempt.
        if let Some(o) = crate::inference::contention::contention() {
            o.on_blocks_freed();
        }
        Ok(success)
    }
}
