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




/// Pure targeting predicate for [`test_force_producer_abort`] (env-free, so it is
/// unit-testable): abort iff a target link is configured AND this pass is the
/// producer for it. An unset target (`None`) never matches ⇒ zero production
/// behavior; a non-producer pass (`produced = None`) is never targeted.
pub(crate) fn abort_target_matches(produced: Option<u32>, target: Option<u32>) -> bool {
    target.is_some() && produced == target
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


