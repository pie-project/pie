//! Forward-pass execution pipeline — moved out of `api/inference.rs`
//! (mechanical relocation; logic unchanged). The engine machinery behind
//! the WIT inference host: `execute_impl`, the forward-transaction
//! lifecycle, contention/preempt, and response marshaling. `api/inference.rs`
//! keeps the thin WIT resource types + Host trait impls.
#![allow(unused_imports)]

use crate::api::grammar::*;
use crate::api::pie;
use crate::inference;
use crate::inference::paging;
use crate::inferlet::ProcessCtx;
use anyhow::Result;
use pie_grammar::brle::RunMask;
use pie_grammar::compiled_grammar::CompiledGrammar;
use pie_grammar::grammar::Grammar as InternalGrammar;
use pie_grammar::json_schema::{JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar};
use pie_grammar::matcher::GrammarMatcher;
use pie_grammar::regex::regex_to_grammar;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use wasmtime::component::Resource;
use wasmtime::component::{Accessor, HasSelf};
use wasmtime_wasi::WasiView;

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

// NOTE: the arena-era `self_suspend_park_restore` (warm CPU stash + parked
// restore) was deleted with the typed-store flip: phase 1 has no residency
// tier, so preemption means releasing the victim's working sets and
// recomputing on resume (kv_refact.md, Scheduler contention ladder). The
// exclusive-footprint preempt path lands with the contention rework.
