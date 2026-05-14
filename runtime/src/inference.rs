//! # Inference Module
//!
//! Forward pass management for model execution.
//!
//! Each model gets a dedicated InferenceService that:
//! - Translates logical KV page IDs to physical page IDs
//! - Routes requests to per-device BatchSchedulers based on page affinity
//!
//! Batch scheduling, RPC execution, and response notification are handled
//! by individual BatchScheduler instances (one per device).

pub mod brle;
pub mod request;
pub mod scheduler;
pub mod speculator;
pub mod structured;
mod adaptive_policy;

pub use speculator::{try_hit, lookup_for_ctx, StagedBatch, BYPASS_HIT_COUNT, CHAIN_SUBMIT_COUNT, CHAIN_DROP_COUNT};

use tokio::sync::oneshot;

use crate::service::{ServiceArray, ServiceHandler};
use crate::context::pagestore::PhysicalPageId;
use anyhow::Result;
use request::{ForwardPassOutput, ForwardPassRequest};
use scheduler::BatchScheduler;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::Ordering::Relaxed;

// Re-export public types
pub use request::{ForwardPassOutput as Output, Sampler};
pub use scheduler::SchedulerStats;

/// Aggregated inference stats for a single model (across all devices).
#[derive(Debug, Default, serde::Serialize)]
pub struct InferenceStats {
    pub total_batches: u64,
    pub total_tokens_processed: u64,
    pub total_requests_processed: u64,
    pub max_batch_size_observed: u64,
    /// Histogram buckets (1, 2-3, 4-7, 8-15, 16-31, 32-63, 64-127, 128+).
    pub batch_size_hist: [u64; 8],
    pub last_batch_latency_us: u64,
    pub avg_batch_latency_us: u64,
}

// =============================================================================
// Public API
// =============================================================================

static SERVICES: std::sync::LazyLock<ServiceArray<Message>> = std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference service for a model.
///
/// `speculation_depth` is the per-context depth of pass-level
/// speculative execution (`scheduler.speculation_depth` in toml).
/// `0` disables pass-level speculation entirely.
pub async fn spawn(
    device_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
    batch_policy: String,
    speculation_depth: u32,
) -> usize {
    // Fetch device info before entering the sync closure.
    let mut device_batch_limits = Vec::with_capacity(device_indices.len());
    for &device_idx in device_indices {
        let info = crate::device::get_spec(device_idx).await
            .unwrap_or_else(|e| panic!("Failed to get device info for index {device_idx}: {e}"));
        device_batch_limits.push((info.max_batch_size, info.max_batch_tokens));
    }

    let model_idx = SERVICES.len();
    let num_devices = device_indices.len();
    SERVICES.spawn(move || InferenceService::new(
        model_idx,
        num_devices,
        device_batch_limits,
        page_size,
        request_timeout_secs,
        batch_policy,
        speculation_depth as usize,
    )).expect("Failed to spawn inference service")
}

/// Submits a pre-resolved forward pass to the appropriate device scheduler.
///
/// All context operations (ensure_resident, page resolution) must be done
/// by the caller BEFORE calling this. The inference actor just dispatches
/// to the batch scheduler — it never blocks on context operations.
pub async fn submit(
    model_idx: usize,
    request: ForwardPassRequest,
    device_idx: usize,
    physical_page_ids: Vec<PhysicalPageId>,
    extra_pages: Vec<PhysicalPageId>,
    last_page_len: u32,
) -> Result<ForwardPassOutput> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::Submit {
        request, device_idx, physical_page_ids, extra_pages, last_page_len, response: tx,
    })?;
    Ok(rx.await.map_err(|_| anyhow::anyhow!(
        "inference submit: scheduler dropped response channel"
    ))?)
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
pub fn invalidate_speculation_for_ctx(
    model_idx: usize,
    ctx_id: crate::context::ContextId,
) {
    let _ = SERVICES.send(
        model_idx,
        Message::InvalidateSpeculationForCtx { ctx_id },
    );
}

// =============================================================================
// Inference Service
// =============================================================================

/// The inference service handles forward pass operations.
///
/// Routes requests to the appropriate per-device `BatchScheduler`
/// based on physical page affinity from the context service.
struct InferenceService {
    model_idx: usize,
    num_devices: usize,
    schedulers: Vec<BatchScheduler>,
    scheduler_stats: Vec<Arc<SchedulerStats>>,
    /// Per-context depth of pass-level speculation. Each ctx
    /// can have up to this many pre-fired stages in its deque.
    /// Sourced from `scheduler.speculation_depth` in the toml.
    /// `0` disables speculation: no staged entries are ever
    /// pushed and every submit goes through the cold path.
    speculation_depth: usize,
    /// Per-device speculation state: a deque of pre-fired
    /// PendingRequests per ctx_id. Inferlet `execute()` calls
    /// hit-check the front of the deque; the chain extender pushes
    /// new entries as each fire completes (bounded by
    /// `speculation_depth`).
    staged_batch: Vec<Arc<Mutex<HashMap<crate::context::ContextId, std::collections::VecDeque<StagedEntry>>>>>,
}

use speculator::StagedEntry;

impl std::fmt::Debug for InferenceService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceService").finish()
    }
}

impl InferenceService {

    fn new(
        model_idx: usize,
        num_devices: usize,
        device_batch_limits: Vec<(usize, usize)>,
        page_size: u32,
        request_timeout_secs: u64,
        batch_policy: String,
        speculation_depth: usize,
    ) -> Self {
        let schedulers: Vec<BatchScheduler> = (0..num_devices).map(|device_idx| {
            let (max_batch_size, max_batch_tokens) = device_batch_limits[device_idx];
            BatchScheduler::new(
                model_idx,
                device_idx,
                page_size,
                max_batch_size,
                max_batch_tokens,
                request_timeout_secs,
                batch_policy.clone(),
            )
        }).collect();

        let scheduler_stats: Vec<_> = schedulers.iter().map(|s| s.stats().clone()).collect();

        let staged_batch: Vec<Arc<Mutex<HashMap<crate::context::ContextId, std::collections::VecDeque<StagedEntry>>>>> =
            (0..num_devices).map(|_| Arc::new(Mutex::new(HashMap::new()))).collect();
        speculator::register_model(model_idx, &staged_batch, speculation_depth);
        InferenceService {
            model_idx,
            num_devices,
            schedulers,
            scheduler_stats,
            speculation_depth,
            staged_batch,
        }
    }

    /// Aggregate stats from all per-device schedulers.
    fn aggregate_stats(&self) -> InferenceStats {
        let mut total_batches = 0u64;
        let mut total_tokens = 0u64;
        let mut total_requests = 0u64;
        let mut max_batch_size = 0u64;
        let mut hist = [0u64; 8];
        let mut last_latency = 0u64;
        let mut cumulative_latency = 0u64;

        for s in &self.scheduler_stats {
            total_batches += s.total_batches.load(Relaxed);
            total_tokens += s.total_tokens_processed.load(Relaxed);
            total_requests += s.total_requests_processed.load(Relaxed);
            max_batch_size = max_batch_size.max(s.max_batch_size_observed.load(Relaxed));
            for (dst, src) in hist.iter_mut().zip(s.batch_size_hist.iter()) {
                *dst += src.load(Relaxed);
            }
            last_latency = last_latency.max(s.last_batch_latency_us.load(Relaxed));
            cumulative_latency += s.cumulative_latency_us.load(Relaxed);
        }

        let avg_latency = if total_batches > 0 {
            cumulative_latency / total_batches
        } else {
            0
        };

        InferenceStats {
            total_batches,
            total_tokens_processed: total_tokens,
            total_requests_processed: total_requests,
            max_batch_size_observed: max_batch_size,
            batch_size_hist: hist,
            last_batch_latency_us: last_latency,
            avg_batch_latency_us: avg_latency,
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
        request: ForwardPassRequest,
        device_idx: usize,
        physical_page_ids: Vec<PhysicalPageId>,
        /// Pages the ctx has reserved beyond the active prefix.
        /// Passed to the chain extender so it can extend across the
        /// full reserved range without re-allocating.
        extra_pages: Vec<PhysicalPageId>,
        last_page_len: u32,
        response: oneshot::Sender<ForwardPassOutput>,
    },
    GetStats { response: oneshot::Sender<InferenceStats> },
    /// Drop the speculation chain for a ctx (called from
    /// `api::context::destroy`). Empties the per-device chain
    /// queue if present.
    InvalidateSpeculationForCtx {
        ctx_id: crate::context::ContextId,
    },
}


impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Submit { request, device_idx, physical_page_ids, extra_pages, last_page_len, response } => {
                let idx = device_idx.min(self.num_devices.saturating_sub(1));
                let ctx_id = request.context_id;

                // Staged self-staging path: check staged_batch for a
                // hit, submit cold otherwise, and chain-extend up to
                // `speculation_depth` pre-fired stages. Inferlet-side
                // hits typically bypass this path via
                // `inference::try_hit` (lock-free) before reaching
                // the actor. Submits reaching this actor are
                // therefore either cold or post-miss (the api
                // layer's try_hit returned None).
                let staged_entry = self.staged_batch[idx].lock().ok().and_then(|mut sb| {
                    if let Some(deque) = sb.get_mut(&ctx_id) {
                        if let Some(front) = deque.front() {
                            let req_token = request.tokens.first().copied();
                            let req_pos = request.positions.first().copied();
                            if Some(front.anchor_token) == req_token
                                && Some(front.anchor_pos) == req_pos
                            {
                                return deque.pop_front();
                            }
                        }
                        // Fingerprint mismatch — drop the entire
                        // chain. Deeper stages were built on a now-
                        // invalid assumption.
                        deque.clear();
                    }
                    None
                });

                let scheduler_handle = self.schedulers[idx].handle();
                let staged_batch_arc = self.staged_batch[idx].clone();
                let request_clone = request.clone();
                let speculation_depth = self.speculation_depth;

                if let Some(entry) = staged_entry {
                    // HIT: forward the staged rx; the chain
                    // extender that pushed this entry is still alive
                    // and will keep extending. No re-spawn here (it
                    // would duplicate the chain).
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
                let cur_page_idx = physical_page_ids.len().saturating_sub(1);
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

