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
use scheduler::BatchScheduler;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering::Relaxed;

pub use scheduler::SchedulerStats;
pub use speculator::{
    BYPASS_HIT_COUNT, CHAIN_DROP_COUNT, CHAIN_SUBMIT_COUNT, StagedBatch, lookup_for_ctx, try_hit,
};

use speculator::StagedBatchMap;

const ADAPTIVE_DENSE_SPEC_MAX_BATCH_LATENCY_US: u64 = 10_000;
const ADAPTIVE_DENSE_SPEC_TP1_MAX_BATCH_LATENCY_US: u64 = 60_000;

pub(crate) fn should_use_pass_speculation(driver_idx: usize) -> bool {
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
    pub last_request_queue_us: u64,
    pub cumulative_request_queue_us: u64,
    pub avg_request_queue_us: u64,
    pub last_batch_queue_us: u64,
    pub cumulative_batch_queue_us: u64,
    pub avg_batch_queue_us: u64,
    pub last_permit_wait_us: u64,
    pub cumulative_permit_wait_us: u64,
    pub avg_permit_wait_us: u64,
    pub last_batch_build_us: u64,
    pub cumulative_batch_build_us: u64,
    pub avg_batch_build_us: u64,
    pub last_driver_forward_us: u64,
    pub cumulative_driver_forward_us: u64,
    pub avg_driver_forward_us: u64,
    pub last_response_fanout_us: u64,
    pub cumulative_response_fanout_us: u64,
    pub avg_response_fanout_us: u64,
    pub last_response_classify_us: u64,
    pub cumulative_response_classify_us: u64,
    pub avg_response_classify_us: u64,
    pub last_response_token_output_build_us: u64,
    pub cumulative_response_token_output_build_us: u64,
    pub avg_response_token_output_build_us: u64,
    pub last_response_direct_send_us: u64,
    pub cumulative_response_direct_send_us: u64,
    pub avg_response_direct_send_us: u64,
    pub last_response_chunk_send_us: u64,
    pub cumulative_response_chunk_send_us: u64,
    pub avg_response_chunk_send_us: u64,
    pub last_response_extract_us: u64,
    pub cumulative_response_extract_us: u64,
    pub avg_response_extract_us: u64,
    pub last_response_error_us: u64,
    pub cumulative_response_error_us: u64,
    pub avg_response_error_us: u64,
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
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Submit {
            request,
            driver_idx,
            physical_page_ids,
            extra_pages,
            last_page_len,
            response: tx,
        },
    )?;
    rx.await
        .map_err(|_| anyhow::anyhow!("inference submit: scheduler dropped response channel"))?
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
    /// Greedy scheduling is the single-stream latency path. In that
    /// mode a low-concurrency chain is useful even when the model's
    /// per-batch latency is high; adaptive throughput mode only keeps
    /// dense self-staging for models whose measured batch latency is
    /// cheap enough to avoid starving normal batching.
    allow_latency_chains: bool,
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
        let allow_latency_chains = batch_policy == "greedy";
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
            .map(|_| Arc::new(Mutex::new(HashMap::new())))
            .collect();
        speculator::register_model(model_idx, &staged_batch, speculation_depth);

        InferenceService {
            model_idx,
            num_drivers,
            schedulers,
            scheduler_stats,
            speculation_depth,
            allow_latency_chains,
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
        let mut last_request_queue = 0u64;
        let mut cumulative_request_queue = 0u64;
        let mut last_batch_queue = 0u64;
        let mut cumulative_batch_queue = 0u64;
        let mut last_permit_wait = 0u64;
        let mut cumulative_permit_wait = 0u64;
        let mut last_batch_build = 0u64;
        let mut cumulative_batch_build = 0u64;
        let mut last_driver_forward = 0u64;
        let mut cumulative_driver_forward = 0u64;
        let mut last_response_fanout = 0u64;
        let mut cumulative_response_fanout = 0u64;
        let mut last_response_classify = 0u64;
        let mut cumulative_response_classify = 0u64;
        let mut last_response_token_output_build = 0u64;
        let mut cumulative_response_token_output_build = 0u64;
        let mut last_response_direct_send = 0u64;
        let mut cumulative_response_direct_send = 0u64;
        let mut last_response_chunk_send = 0u64;
        let mut cumulative_response_chunk_send = 0u64;
        let mut last_response_extract = 0u64;
        let mut cumulative_response_extract = 0u64;
        let mut last_response_error = 0u64;
        let mut cumulative_response_error = 0u64;

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
            last_request_queue = last_request_queue.max(s.last_request_queue_us.load(Relaxed));
            cumulative_request_queue += s.cumulative_request_queue_us.load(Relaxed);
            last_batch_queue = last_batch_queue.max(s.last_batch_queue_us.load(Relaxed));
            cumulative_batch_queue += s.cumulative_batch_queue_us.load(Relaxed);
            last_permit_wait = last_permit_wait.max(s.last_permit_wait_us.load(Relaxed));
            cumulative_permit_wait += s.cumulative_permit_wait_us.load(Relaxed);
            last_batch_build = last_batch_build.max(s.last_batch_build_us.load(Relaxed));
            cumulative_batch_build += s.cumulative_batch_build_us.load(Relaxed);
            last_driver_forward = last_driver_forward.max(s.last_driver_forward_us.load(Relaxed));
            cumulative_driver_forward += s.cumulative_driver_forward_us.load(Relaxed);
            last_response_fanout =
                last_response_fanout.max(s.last_response_fanout_us.load(Relaxed));
            cumulative_response_fanout += s.cumulative_response_fanout_us.load(Relaxed);
            last_response_classify =
                last_response_classify.max(s.last_response_classify_us.load(Relaxed));
            cumulative_response_classify += s.cumulative_response_classify_us.load(Relaxed);
            last_response_token_output_build = last_response_token_output_build
                .max(s.last_response_token_output_build_us.load(Relaxed));
            cumulative_response_token_output_build +=
                s.cumulative_response_token_output_build_us.load(Relaxed);
            last_response_direct_send =
                last_response_direct_send.max(s.last_response_direct_send_us.load(Relaxed));
            cumulative_response_direct_send += s.cumulative_response_direct_send_us.load(Relaxed);
            last_response_chunk_send =
                last_response_chunk_send.max(s.last_response_chunk_send_us.load(Relaxed));
            cumulative_response_chunk_send += s.cumulative_response_chunk_send_us.load(Relaxed);
            last_response_extract =
                last_response_extract.max(s.last_response_extract_us.load(Relaxed));
            cumulative_response_extract += s.cumulative_response_extract_us.load(Relaxed);
            last_response_error = last_response_error.max(s.last_response_error_us.load(Relaxed));
            cumulative_response_error += s.cumulative_response_error_us.load(Relaxed);
        }

        let avg_latency = if total_batches > 0 {
            cumulative_latency / total_batches
        } else {
            0
        };
        let avg_request_queue = if total_requests > 0 {
            cumulative_request_queue / total_requests
        } else {
            0
        };
        let avg_batch_queue = if total_batches > 0 {
            cumulative_batch_queue / total_batches
        } else {
            0
        };
        let avg_permit_wait = if total_batches > 0 {
            cumulative_permit_wait / total_batches
        } else {
            0
        };
        let avg_batch_build = if total_batches > 0 {
            cumulative_batch_build / total_batches
        } else {
            0
        };
        let avg_driver_forward = if total_batches > 0 {
            cumulative_driver_forward / total_batches
        } else {
            0
        };
        let avg_response_fanout = if total_batches > 0 {
            cumulative_response_fanout / total_batches
        } else {
            0
        };
        let avg_response_classify = if total_batches > 0 {
            cumulative_response_classify / total_batches
        } else {
            0
        };
        let avg_response_token_output_build = if total_batches > 0 {
            cumulative_response_token_output_build / total_batches
        } else {
            0
        };
        let avg_response_direct_send = if total_batches > 0 {
            cumulative_response_direct_send / total_batches
        } else {
            0
        };
        let avg_response_chunk_send = if total_batches > 0 {
            cumulative_response_chunk_send / total_batches
        } else {
            0
        };
        let avg_response_extract = if total_batches > 0 {
            cumulative_response_extract / total_batches
        } else {
            0
        };
        let avg_response_error = if total_batches > 0 {
            cumulative_response_error / total_batches
        } else {
            0
        };

        InferenceStats {
            total_batches,
            total_tokens_processed: total_tokens,
            total_requests_processed: total_requests,
            max_forward_requests_observed: max_forward_requests,
            batch_size_hist: hist,
            last_batch_latency_us: last_latency,
            cumulative_batch_latency_us: cumulative_latency,
            avg_batch_latency_us: avg_latency,
            last_request_queue_us: last_request_queue,
            cumulative_request_queue_us: cumulative_request_queue,
            avg_request_queue_us: avg_request_queue,
            last_batch_queue_us: last_batch_queue,
            cumulative_batch_queue_us: cumulative_batch_queue,
            avg_batch_queue_us: avg_batch_queue,
            last_permit_wait_us: last_permit_wait,
            cumulative_permit_wait_us: cumulative_permit_wait,
            avg_permit_wait_us: avg_permit_wait,
            last_batch_build_us: last_batch_build,
            cumulative_batch_build_us: cumulative_batch_build,
            avg_batch_build_us: avg_batch_build,
            last_driver_forward_us: last_driver_forward,
            cumulative_driver_forward_us: cumulative_driver_forward,
            avg_driver_forward_us: avg_driver_forward,
            last_response_fanout_us: last_response_fanout,
            cumulative_response_fanout_us: cumulative_response_fanout,
            avg_response_fanout_us: avg_response_fanout,
            last_response_classify_us: last_response_classify,
            cumulative_response_classify_us: cumulative_response_classify,
            avg_response_classify_us: avg_response_classify,
            last_response_token_output_build_us: last_response_token_output_build,
            cumulative_response_token_output_build_us: cumulative_response_token_output_build,
            avg_response_token_output_build_us: avg_response_token_output_build,
            last_response_direct_send_us: last_response_direct_send,
            cumulative_response_direct_send_us: cumulative_response_direct_send,
            avg_response_direct_send_us: avg_response_direct_send,
            last_response_chunk_send_us: last_response_chunk_send,
            cumulative_response_chunk_send_us: cumulative_response_chunk_send,
            avg_response_chunk_send_us: avg_response_chunk_send,
            last_response_extract_us: last_response_extract,
            cumulative_response_extract_us: cumulative_response_extract,
            avg_response_extract_us: avg_response_extract,
            last_response_error_us: last_response_error,
            cumulative_response_error_us: cumulative_response_error,
            avg_response_error_us: avg_response_error,
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
                    let mut sb = self.staged_batch[idx].lock().ok();
                    sb.as_mut().and_then(|sb| {
                        let deque = sb.get_mut(&ctx_id)?;
                        if let Some(front) = deque.front() {
                            let req_token = request.token_ids.first().copied();
                            let req_pos = request.position_ids.first().copied();
                            if Some(front.anchor_token) == req_token
                                && Some(front.anchor_pos) == req_pos
                            {
                                return deque.pop_front();
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
                let pinned = crate::context::pinned_count(idx);
                let last_batch_latency_us = self.scheduler_stats[idx]
                    .last_batch_latency_us
                    .load(Relaxed);
                let low_concurrency_latency_chain = self.allow_latency_chains && pinned <= 1;
                let adaptive_dense_spec_cutoff_us = if self.num_drivers == 1 {
                    ADAPTIVE_DENSE_SPEC_TP1_MAX_BATCH_LATENCY_US
                } else {
                    ADAPTIVE_DENSE_SPEC_MAX_BATCH_LATENCY_US
                };
                let cheap_adaptive_chain = last_batch_latency_us > 0
                    && last_batch_latency_us <= adaptive_dense_spec_cutoff_us;
                let dense_chain_budget = low_concurrency_latency_chain || cheap_adaptive_chain;
                let speculation_depth =
                    if request_clone.rs_slot_ids.is_empty() && dense_chain_budget {
                        self.speculation_depth
                    } else {
                        0
                    };

                if let Some(entry) = staged_entry {
                    // HIT: forward the staged rx; the chain
                    // extender that pushed this entry is still
                    // alive and will keep extending. No re-spawn
                    // here (it would duplicate the chain).
                    let mut output_rx = entry.output_rx;
                    match output_rx.try_recv() {
                        Ok(output) => {
                            let _ = response.send(output);
                        }
                        Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                            tokio::spawn(async move {
                                if let Ok(output) = output_rx.await {
                                    let _ = response.send(output);
                                }
                            });
                        }
                        Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                            let _ = response
                                .send(Err(anyhow::anyhow!("staged response channel closed")));
                        }
                    }
                    return;
                }

                // No hit: cold submit + start a fresh chain.
                let (sched_tx, sched_rx) = oneshot::channel();
                let cur_page_idx = physical_page_ids.len().saturating_sub(1);
                let active_page_len = physical_page_ids.len();
                let mut all_pages = physical_page_ids;
                all_pages.extend(extra_pages);
                let all_pages: Arc<[PhysicalPageId]> = Arc::from(all_pages);
                if let Err(e) = self.schedulers[idx].submit_shared(
                    request,
                    sched_tx,
                    all_pages.clone(),
                    active_page_len,
                    last_page_len,
                ) {
                    tracing::error!("submit failed: {e}");
                    return;
                }
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
