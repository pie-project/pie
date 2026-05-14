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
pub mod structured;

use tokio::sync::oneshot;

use crate::context::pagestore::PhysicalPageId;
use crate::driver::DriverId;
use crate::service::{ServiceArray, ServiceHandler};
use anyhow::Result;
use scheduler::BatchScheduler;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

pub use scheduler::SchedulerStats;

/// Aggregated inference stats for a single model (across all drivers).
#[derive(Debug, Default, serde::Serialize)]
pub struct InferenceStats {
    pub total_batches: u64,
    pub total_tokens_processed: u64,
    pub last_batch_latency_us: u64,
    pub avg_batch_latency_us: u64,
}

// =============================================================================
// Public API
// =============================================================================

static SERVICES: std::sync::LazyLock<ServiceArray<Message>> =
    std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference service for a model.
pub async fn spawn(
    driver_indices: &[usize],
    page_size: u32,
    request_timeout_secs: u64,
    batch_policy: String,
) -> usize {
    // Fetch driver info before entering the sync closure.
    let driver_ids: Vec<DriverId> = driver_indices.to_vec();
    let mut driver_batch_limits = Vec::with_capacity(driver_indices.len());
    for &driver_idx in driver_indices {
        let info = crate::driver::get_spec(driver_idx)
            .await
            .unwrap_or_else(|e| panic!("Failed to get driver info for index {driver_idx}: {e}"));
        driver_batch_limits.push((info.max_batch_size, info.max_batch_tokens));
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
            )
        })
        .expect("Failed to spawn inference service")
}

/// Submits a pre-resolved forward pass to the appropriate driver scheduler.
///
/// All context operations (ensure_resident, page resolution) must be done
/// by the caller BEFORE calling this. The inference actor just dispatches
/// to the batch scheduler — it never blocks on context operations.
pub async fn submit(
    model_idx: usize,
    request: pie_bridge::ForwardRequest,
    driver_idx: usize,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
) -> Result<pie_bridge::ForwardResponse> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(
        model_idx,
        Message::Submit {
            request,
            driver_idx,
            physical_page_ids,
            last_page_len,
            response: tx,
        },
    )?;
    Ok(rx
        .await
        .map_err(|_| anyhow::anyhow!("inference submit: scheduler dropped response channel"))?)
}

/// Returns aggregated inference stats for a model (lock-free, non-blocking).
pub async fn get_stats(model_idx: usize) -> InferenceStats {
    let (tx, rx) = oneshot::channel();
    let _ = SERVICES.send(model_idx, Message::GetStats { response: tx });
    rx.await.unwrap_or_default()
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
        driver_batch_limits: Vec<(usize, usize)>,
        page_size: u32,
        request_timeout_secs: u64,
        batch_policy: String,
    ) -> Self {
        let num_drivers = driver_ids.len();
        let schedulers: Vec<BatchScheduler> = driver_ids
            .iter()
            .enumerate()
            .map(|(driver_idx, &driver_id)| {
                let (max_batch_size, max_batch_tokens) = driver_batch_limits[driver_idx];
                BatchScheduler::new(
                    driver_id,
                    driver_idx,
                    page_size,
                    max_batch_size,
                    max_batch_tokens,
                    request_timeout_secs,
                    batch_policy.clone(),
                )
            })
            .collect();

        let scheduler_stats: Vec<_> = schedulers.iter().map(|s| s.stats().clone()).collect();

        InferenceService {
            model_idx,
            num_drivers,
            schedulers,
            scheduler_stats,
        }
    }

    /// Aggregate stats from all per-driver schedulers.
    fn aggregate_stats(&self) -> InferenceStats {
        let mut total_batches = 0u64;
        let mut total_tokens = 0u64;
        let mut last_latency = 0u64;
        let mut cumulative_latency = 0u64;

        for s in &self.scheduler_stats {
            total_batches += s.total_batches.load(Relaxed);
            total_tokens += s.total_tokens_processed.load(Relaxed);
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
        request: pie_bridge::ForwardRequest,
        driver_idx: usize,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
        response: oneshot::Sender<pie_bridge::ForwardResponse>,
    },
    GetStats {
        response: oneshot::Sender<InferenceStats>,
    },
}

impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Submit {
                request,
                driver_idx,
                physical_page_ids,
                last_page_len,
                response,
            } => {
                let idx = driver_idx.min(self.num_drivers.saturating_sub(1));
                if let Err(e) =
                    self.schedulers[idx].submit(request, response, physical_page_ids, last_page_len)
                {
                    tracing::error!("Failed to submit to scheduler: {}", e);
                }
            }
            Message::GetStats { response } => {
                let _ = response.send(self.aggregate_stats());
            }
        }
    }
}
