//! # Inference Module
//!
//! Forward pass management for model execution.
//!
//! Each model gets a dedicated InferenceService that:
//! - Translates logical KV page IDs to physical page IDs
//! - Routes requests to per-driver BatchSchedulers based on page affinity
//!
//! Batch scheduling, RPC execution, and response notification are handled
//! by individual BatchScheduler instances (one per driver).

mod adaptive_policy;
pub mod request;
pub mod scheduler;
pub mod speculator;
pub mod structured;

use tokio::sync::oneshot;

use crate::context::pagestore::PhysicalPageId;
use crate::driver::{DriverId, SchedulerLimits};
use crate::service::{ServiceArray, ServiceHandler};
use anyhow::Result;
use dashmap::DashMap;
use scheduler::BatchScheduler;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;

pub use scheduler::{SchedulerStats, SYSTEM_SPEC_DRAFT_POS_BUCKETS};
pub use speculator::{
    BYPASS_HIT_COUNT, CHAIN_DROP_COUNT, CHAIN_SUBMIT_COUNT, StagedBatch, lookup_for_ctx, try_hit,
};

use speculator::StagedBatchMap;

pub(crate) fn should_use_pass_speculation(driver_idx: usize) -> bool {
    // The chain extender is already gated by scheduler.speculation_depth and
    // request shape. If a staged entry exists, claim it even for a single
    // resident context; otherwise the API path pays pin/actor overhead and
    // the pre-fired chain only gets discovered later inside InferenceService.
    let _ = driver_idx;
    true
}

/// Aggregated inference stats for a single model (across all drivers).
#[derive(Debug, Default, serde::Serialize)]
pub struct InferenceStats {
    pub total_batches: u64,
    pub total_tokens_processed: u64,
    pub total_requests_processed: u64,
    pub max_forward_requests_observed: u64,
    /// Histogram buckets (1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+).
    pub batch_size_hist: [u64; 8],
    pub last_batch_latency_us: u64,
    pub cumulative_batch_latency_us: u64,
    pub avg_batch_latency_us: u64,
    pub avg_permit_wait_us: u64,
    pub avg_fire_prepare_us: u64,
    pub avg_execute_batch_us: u64,
    pub avg_batch_build_us: u64,
    pub avg_driver_fire_us: u64,
    pub avg_response_dispatch_us: u64,
    pub avg_context_tick_submit_us: u64,
    pub avg_stats_update_us: u64,
    pub system_spec_draft_tokens_proposed: u64,
    pub system_spec_draft_tokens_accepted: u64,
    pub system_spec_draft_tokens_proposed_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
    pub system_spec_draft_tokens_accepted_per_pos: [u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS],
}

// =============================================================================
// Public API
// =============================================================================

static SERVICES: std::sync::LazyLock<ServiceArray<Message>> =
    std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference service for a model.
///
/// `speculation_depth` is the per-context depth of pass-level
/// speculative execution (`scheduler.speculation_depth` in toml).
/// `0` disables pass-level speculation entirely.
pub async fn spawn(
    driver_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
    batch_policy: String,
    speculation_depth: u32,
) -> usize {
    // Fetch driver info before entering the sync closure.
    let driver_ids: Vec<DriverId> = driver_indices.to_vec();
    let mut driver_batch_limits = Vec::with_capacity(driver_indices.len());
    for &driver_idx in driver_indices {
        let info = crate::driver::get_spec(driver_idx)
            .await
            .unwrap_or_else(|e| panic!("Failed to get driver info for index {driver_idx}: {e}"));
        driver_batch_limits.push(info.scheduler_limits());
    }

    let model_idx = SERVICES.len();
    SERVICES
        .spawn(move || {
            InferenceService::new(
                model_idx,
                driver_ids,
                driver_batch_limits,
                page_size,
                request_timeout_secs,
                batch_policy,
                speculation_depth as usize,
            )
        })
        .expect("Failed to spawn inference service")
}

/// Submits a pre-resolved forward pass to the appropriate driver scheduler.
///
/// All context operations (ensure_resident, page resolution) must be done
/// by the caller BEFORE calling this. The inference actor just dispatches
/// to the batch scheduler — it never blocks on context operations.
///
/// `extra_pages` lists working pages the ctx has reserved beyond the
/// active prefix in `physical_page_ids`. They are passed to the
/// speculator's chain extender so it can extend the chain across the
/// full reserved range without re-pinning. Pass an empty vec when
/// speculation is disabled or the caller has no extra pages.
pub async fn submit(
    model_idx: usize,
    request: pie_bridge::ForwardRequest,
    driver_idx: usize,
    physical_page_ids: Vec<PhysicalPageId>,
    extra_pages: Vec<PhysicalPageId>,
    last_page_len: u32,
) -> Result<ForwardOutput> {
    let rx = submit_async(
        model_idx,
        request,
        driver_idx,
        physical_page_ids,
        extra_pages,
        last_page_len,
        None,
        true,
    )?;
    rx.await
        .map_err(|_| anyhow::anyhow!("inference submit: scheduler dropped response channel"))?
}

pub fn submit_async(
    model_idx: usize,
    request: pie_bridge::ForwardRequest,
    driver_idx: usize,
    physical_page_ids: Vec<PhysicalPageId>,
    extra_pages: Vec<PhysicalPageId>,
    last_page_len: u32,
    active_page_idx: Option<usize>,
    allow_pass_speculation: bool,
) -> Result<oneshot::Receiver<Result<ForwardOutput>>> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Submit {
            request,
            driver_idx,
            physical_page_ids,
            extra_pages,
            last_page_len,
            active_page_idx,
            allow_pass_speculation,
            response: tx,
        },
    )?;
    Ok(rx)
}

/// Internal forward result shape passed from the scheduler to a waiting
/// inferlet. Normal decode returns a single token per request; carrying that
/// directly avoids allocating a one-request `ForwardResponse` for every token.
#[derive(Debug)]
pub enum ForwardOutput {
    Token(u32),
    Tokens(Vec<u32>),
    Response(pie_bridge::ForwardResponse),
}

impl ForwardOutput {
    pub(crate) fn first_token(&self) -> Option<u32> {
        match self {
            Self::Token(t) => Some(*t),
            Self::Tokens(tokens) => tokens.first().copied(),
            Self::Response(resp) => resp.tokens.first().copied(),
        }
    }

    #[cfg(test)]
    pub(crate) fn from_response(resp: pie_bridge::ForwardResponse) -> Self {
        Self::Response(resp)
    }
}

impl From<pie_bridge::ForwardResponse> for ForwardOutput {
    fn from(resp: pie_bridge::ForwardResponse) -> Self {
        Self::Response(resp)
    }
}

/// Returns aggregated inference stats for a model (lock-free, non-blocking).
pub async fn get_stats(model_idx: usize) -> InferenceStats {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
}

/// Drop any outstanding speculation chain for a ctx that's about to
/// be destroyed. Fire-and-forget: the speculator just clears its
/// per-device deque for this ctx; callers don't need confirmation.
pub fn invalidate_speculation_for_ctx(model_idx: usize, ctx_id: crate::context::ContextId) {
    let _ = SERVICES.send(model_idx, Message::InvalidateSpeculationForCtx { ctx_id });
}

// =============================================================================
// Inference Service
// =============================================================================

/// The inference service handles forward pass operations.
///
/// Routes requests to the appropriate per-driver `BatchScheduler`
/// based on physical page affinity from the context service.
struct InferenceService {
    model_idx: usize,
    num_drivers: usize,
    schedulers: Vec<BatchScheduler>,
    scheduler_stats: Vec<Arc<SchedulerStats>>,
    /// Per-context depth of pass-level speculation. Each ctx can
    /// have up to this many pre-fired stages in its deque. Sourced
    /// from `scheduler.speculation_depth` in the toml. `0` disables
    /// speculation: no staged entries are ever pushed and every
    /// submit goes through the cold path.
    speculation_depth: usize,
    /// Per-device speculation state: a deque of pre-fired stages
    /// per ctx_id. Inferlet `execute()` calls hit-check the front
    /// of the deque; the chain extender pushes new entries as each
    /// fire completes (bounded by `speculation_depth`).
    staged_batch: Vec<StagedBatchMap>,
}

impl std::fmt::Debug for InferenceService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceService").finish()
    }
}

impl InferenceService {
    fn new(
        model_idx: usize,
        driver_ids: Vec<DriverId>,
        driver_batch_limits: Vec<SchedulerLimits>,
        page_size: u32,
        request_timeout_secs: u64,
        batch_policy: String,
        speculation_depth: usize,
    ) -> Self {
        let num_drivers = driver_ids.len();
        let schedulers: Vec<BatchScheduler> = driver_ids
            .iter()
            .enumerate()
            .map(|(driver_idx, &driver_id)| {
                let limits = driver_batch_limits[driver_idx];
                BatchScheduler::new(
                    driver_id,
                    driver_idx,
                    page_size,
                    limits,
                    request_timeout_secs,
                    batch_policy.clone(),
                )
            })
            .collect();

        let scheduler_stats: Vec<_> = schedulers.iter().map(|s| s.stats().clone()).collect();

        let staged_batch: Vec<StagedBatchMap> = (0..num_drivers)
            .map(|_| Arc::new(DashMap::new()))
            .collect();
        speculator::register_model(model_idx, &staged_batch, speculation_depth);

        InferenceService {
            model_idx,
            num_drivers,
            schedulers,
            scheduler_stats,
            speculation_depth,
            staged_batch,
        }
    }

    /// Aggregate stats from all per-driver schedulers.
    fn aggregate_stats(&self) -> InferenceStats {
        let mut total_batches = 0u64;
        let mut total_tokens = 0u64;
        let mut total_requests = 0u64;
        let mut max_forward_requests = 0u64;
        let mut hist = [0u64; 8];
        let mut last_latency = 0u64;
        let mut cumulative_latency = 0u64;
        let mut cumulative_permit_wait = 0u64;
        let mut cumulative_fire_prepare = 0u64;
        let mut cumulative_execute_batch = 0u64;
        let mut cumulative_batch_build = 0u64;
        let mut cumulative_driver_fire = 0u64;
        let mut cumulative_response_dispatch = 0u64;
        let mut cumulative_context_tick_submit = 0u64;
        let mut cumulative_stats_update = 0u64;
        let mut system_spec_draft_tokens_proposed = 0u64;
        let mut system_spec_draft_tokens_accepted = 0u64;
        let mut system_spec_draft_tokens_proposed_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];
        let mut system_spec_draft_tokens_accepted_per_pos =
            [0u64; SYSTEM_SPEC_DRAFT_POS_BUCKETS];

        for s in &self.scheduler_stats {
            total_batches += s.total_batches.load(Relaxed);
            total_tokens += s.total_tokens_processed.load(Relaxed);
            total_requests += s.total_requests_processed.load(Relaxed);
            max_forward_requests =
                max_forward_requests.max(s.max_forward_requests_observed.load(Relaxed));
            for (dst, src) in hist.iter_mut().zip(s.batch_size_hist.iter()) {
                *dst += src.load(Relaxed);
            }
            last_latency = last_latency.max(s.last_batch_latency_us.load(Relaxed));
            cumulative_latency += s.cumulative_latency_us.load(Relaxed);
            cumulative_permit_wait += s.cumulative_permit_wait_us.load(Relaxed);
            cumulative_fire_prepare += s.cumulative_fire_prepare_us.load(Relaxed);
            cumulative_execute_batch += s.cumulative_execute_batch_us.load(Relaxed);
            cumulative_batch_build += s.cumulative_batch_build_us.load(Relaxed);
            cumulative_driver_fire += s.cumulative_driver_fire_us.load(Relaxed);
            cumulative_response_dispatch += s.cumulative_response_dispatch_us.load(Relaxed);
            cumulative_context_tick_submit += s.cumulative_context_tick_submit_us.load(Relaxed);
            cumulative_stats_update += s.cumulative_stats_update_us.load(Relaxed);
            system_spec_draft_tokens_proposed += s.system_spec_draft_tokens_proposed.load(Relaxed);
            system_spec_draft_tokens_accepted += s.system_spec_draft_tokens_accepted.load(Relaxed);
            for (dst, src) in system_spec_draft_tokens_proposed_per_pos
                .iter_mut()
                .zip(s.system_spec_draft_tokens_proposed_per_pos.iter())
            {
                *dst += src.load(Relaxed);
            }
            for (dst, src) in system_spec_draft_tokens_accepted_per_pos
                .iter_mut()
                .zip(s.system_spec_draft_tokens_accepted_per_pos.iter())
            {
                *dst += src.load(Relaxed);
            }
        }

        let avg = |value: u64| {
            if total_batches > 0 {
                value / total_batches
            } else {
                0
            }
        };
        InferenceStats {
            total_batches,
            total_tokens_processed: total_tokens,
            total_requests_processed: total_requests,
            max_forward_requests_observed: max_forward_requests,
            batch_size_hist: hist,
            last_batch_latency_us: last_latency,
            cumulative_batch_latency_us: cumulative_latency,
            avg_batch_latency_us: avg(cumulative_latency),
            avg_permit_wait_us: avg(cumulative_permit_wait),
            avg_fire_prepare_us: avg(cumulative_fire_prepare),
            avg_execute_batch_us: avg(cumulative_execute_batch),
            avg_batch_build_us: avg(cumulative_batch_build),
            avg_driver_fire_us: avg(cumulative_driver_fire),
            avg_response_dispatch_us: avg(cumulative_response_dispatch),
            avg_context_tick_submit_us: avg(cumulative_context_tick_submit),
            avg_stats_update_us: avg(cumulative_stats_update),
            system_spec_draft_tokens_proposed,
            system_spec_draft_tokens_accepted,
            system_spec_draft_tokens_proposed_per_pos,
            system_spec_draft_tokens_accepted_per_pos,
        }
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

/// Messages handled by InferenceService.
#[derive(Debug)]
enum Message {
    /// Submit a pre-resolved forward pass to the scheduler.
    /// All context operations must be done by the caller before sending this.
    Submit {
        request: pie_bridge::ForwardRequest,
        driver_idx: usize,
        physical_page_ids: Vec<PhysicalPageId>,
        /// Pages the ctx has reserved beyond the active prefix.
        /// Passed to the chain extender so it can extend across the
        /// full reserved range without re-allocating.
        extra_pages: Vec<PhysicalPageId>,
        last_page_len: u32,
        active_page_idx: Option<usize>,
        allow_pass_speculation: bool,
        response: oneshot::Sender<Result<ForwardOutput>>,
    },
    GetStats {
        response: oneshot::Sender<InferenceStats>,
    },
    /// Drop the speculation chain for a ctx (called from
    /// `api::context::destroy`). Empties the per-device chain
    /// queue if present.
    InvalidateSpeculationForCtx { ctx_id: crate::context::ContextId },
}

impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Submit {
                request,
                driver_idx,
                physical_page_ids,
                extra_pages,
                last_page_len,
                active_page_idx,
                allow_pass_speculation,
                response,
            } => {
                let idx = driver_idx.min(self.num_drivers.saturating_sub(1));

                // Per-request shape stores the ctx_id in
                // `context_ids[0]`. Default to 0 if missing — only
                // happens for malformed requests, in which case the
                // speculator path is a no-op.
                let ctx_id = request.context_ids.first().copied().unwrap_or(0);

                // Staged self-staging path: check staged_batch for a
                // hit, submit cold otherwise, and chain-extend up to
                // `speculation_depth` pre-fired stages. Inferlet-side
                // hits typically bypass this path via
                // `inference::try_hit` before reaching the actor.
                // Submits reaching this actor are therefore either
                // cold or post-miss (the api layer's try_hit
                // returned None).
                let staged_entry = {
                    let mut deque = self.staged_batch[idx].get_mut(&ctx_id);
                    deque.as_deref_mut().and_then(|deque| {
                        if let Some(front) = deque.front() {
                            if speculator::entry_matches_request(front, &request) {
                                let entry = deque.pop_front();
                                if let Some(entry) = entry.as_ref() {
                                    if !allow_pass_speculation {
                                        entry.allow_extend.store(false, Relaxed);
                                    }
                                }
                                return entry;
                            }
                        }
                        // Fingerprint mismatch — drop the entire chain.
                        // Deeper stages were built on a now-invalid assumption.
                        deque.clear();
                        None
                    })
                };

                let scheduler_handle = self.schedulers[idx].handle();
                let staged_batch_arc = self.staged_batch[idx].clone();
                let request_clone = request.clone();
                let speculation_depth = if allow_pass_speculation {
                    self.speculation_depth
                } else {
                    0
                };

                if let Some(entry) = staged_entry {
                    // HIT: forward the staged rx; the chain
                    // extender that pushed this entry is still
                    // alive and will keep extending. No re-spawn
                    // here (it would duplicate the chain).
                    tokio::spawn(async move {
                        if let Ok(output) = entry.output_rx.await {
                            let _ = response.send(output);
                        }
                    });
                    return;
                }

                // No hit: cold submit + start a fresh chain.
                let (sched_tx, sched_rx) = oneshot::channel();
                if let Err(e) = self.schedulers[idx].submit(
                    request,
                    sched_tx,
                    physical_page_ids.clone(),
                    last_page_len,
                ) {
                    tracing::error!("submit failed: {e}");
                    return;
                }
                let cur_page_idx =
                    active_page_idx.unwrap_or_else(|| physical_page_ids.len().saturating_sub(1));
                let mut all_pages = physical_page_ids;
                all_pages.extend(extra_pages);
                speculator::spawn_extend_chain(
                    sched_rx,
                    response,
                    scheduler_handle,
                    staged_batch_arc,
                    self.model_idx,
                    request_clone,
                    all_pages,
                    cur_page_idx,
                    last_page_len,
                    speculation_depth,
                    Arc::new(AtomicBool::new(allow_pass_speculation)),
                );
            }
            Message::GetStats { response } => {
                let _ = response.send(self.aggregate_stats());
            }
            Message::InvalidateSpeculationForCtx { ctx_id } => {
                speculator::invalidate_ctx(self.model_idx, ctx_id);
            }
        }
    }
}
