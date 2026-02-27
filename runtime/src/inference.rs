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
pub mod structured;
mod adaptive_policy;



use tokio::sync::oneshot;

use crate::context;
use crate::service::{ServiceArray, ServiceHandler};
use crate::device::DeviceId;
use anyhow::Result;
use request::{ForwardPassOutput, ForwardPassRequest};
use scheduler::BatchScheduler;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

// Re-export public types
pub use request::{ForwardPassOutput as Output, Sampler};
pub use scheduler::SchedulerStats;

/// Aggregated inference stats for a single model (across all devices).
#[derive(Debug, Default, serde::Serialize)]
pub struct InferenceStats {
    pub total_batches: u64,
    pub total_tokens_processed: u64,
    pub last_batch_latency_us: u64,
    pub avg_batch_latency_us: u64,
    pub in_flight_batches: u64,
}

// =============================================================================
// Public API
// =============================================================================

static SERVICES: std::sync::LazyLock<ServiceArray<Message>> = std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference service for a model.
pub async fn spawn(
    device_indices: &[usize],
    max_in_flight_batches: usize,
    request_timeout_secs: u64,
    max_wait_ms: u64,
    min_batch_for_optimization: usize,
) -> usize {
    // Fetch device info before entering the sync closure.
    let device_ids: Vec<DeviceId> = device_indices.to_vec();
    let mut device_batch_limits = Vec::with_capacity(device_indices.len());
    for &device_idx in device_indices {
        let info = crate::device::get_spec(device_idx).await
            .unwrap_or_else(|e| panic!("Failed to get device info for index {device_idx}: {e}"));
        device_batch_limits.push((info.max_batch_size, info.max_batch_tokens));
    }

    let model_idx = SERVICES.len();
    SERVICES.spawn(move || InferenceService::new(
        model_idx,
        device_ids,
        device_batch_limits,
        max_in_flight_batches,
        request_timeout_secs,
        max_wait_ms,
        min_batch_for_optimization,
    )).expect("Failed to spawn inference service")
}

/// Executes a forward pass and returns the output.
pub async fn forward_pass(model_idx: usize, request: ForwardPassRequest) -> Result<ForwardPassOutput> {
    let (tx, rx) = oneshot::channel();
    SERVICES.send(model_idx, Message::ForwardPass { request, response: tx })?;
    Ok(rx.await?)
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
/// Routes requests to the appropriate per-device `BatchScheduler`
/// based on physical page affinity from the context service.
struct InferenceService {
    model_idx: usize,
    num_devices: usize,
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
        device_ids: Vec<DeviceId>,
        device_batch_limits: Vec<(usize, usize)>,
        max_in_flight_batches: usize,
        request_timeout_secs: u64,
        max_wait_ms: u64,
        min_batch_for_optimization: usize,
    ) -> Self {
        let num_devices = device_ids.len();
        let schedulers: Vec<BatchScheduler> = device_ids.iter().enumerate().map(|(device_idx, &device_id)| {
            let (max_batch_size, max_batch_tokens) = device_batch_limits[device_idx];
            BatchScheduler::new(
                device_id,
                device_idx,
                max_batch_size,
                max_batch_tokens,
                max_in_flight_batches,
                request_timeout_secs,
                max_wait_ms,
                min_batch_for_optimization,
            )
        }).collect();

        let scheduler_stats: Vec<_> = schedulers.iter().map(|s| s.stats().clone()).collect();

        InferenceService {
            model_idx,
            num_devices,
            schedulers,
            scheduler_stats,
        }
    }

    /// Aggregate stats from all per-device schedulers.
    fn aggregate_stats(&self) -> InferenceStats {
        let mut total_batches = 0u64;
        let mut total_tokens = 0u64;
        let mut last_latency = 0u64;
        let mut cumulative_latency = 0u64;
        let mut in_flight = 0u64;

        for s in &self.scheduler_stats {
            total_batches += s.total_batches.load(Relaxed);
            total_tokens += s.total_tokens_processed.load(Relaxed);
            last_latency = last_latency.max(s.last_batch_latency_us.load(Relaxed));
            cumulative_latency += s.cumulative_latency_us.load(Relaxed);
            in_flight += s.in_flight_batches.load(Relaxed);
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
            in_flight_batches: in_flight,
        }
    }

    /// Resolves physical pages from context and queues the forward pass.
    /// Transparently handles swap-in and replay via the unified replay path.
    async fn forward_pass(&self, request: ForwardPassRequest, response_tx: oneshot::Sender<ForwardPassOutput>) -> Result<()> {
        let ctx_id = request.context_id;

        // Step 1: Ensure all pages are GPU-resident (swap-in from CPU, replay discarded)
        let replay_plan = context::ensure_resident(self.model_idx, ctx_id).await?;

        // Execute replay plan if pages were discarded and need rebuilding
        if let Some(replay_chunks) = replay_plan {
            for chunk in replay_chunks {
                let device_idx = (chunk.device_id as usize)
                    .min(self.num_devices.saturating_sub(1));

                let replay_fwd = request::ForwardPassRequest {
                    context_id: ctx_id,
                    tokens: chunk.tokens,
                    positions: chunk.positions,
                    masks: chunk.masks,
                    speculative_tokens: vec![],
                    speculative_positions: vec![],
                    output_speculative_tokens: false,
                    logit_mask: None,
                    sampling_indices: vec![],
                    samplers: vec![],
                    adapter_id: chunk.adapter,
                    adapter_seed: None,
                    arrival_time: None,
                };

                let (replay_tx, replay_rx) = oneshot::channel();
                self.schedulers[device_idx].submit(
                    replay_fwd, replay_tx,
                    chunk.physical_page_ids, chunk.last_page_len,
                )?;
                let _ = replay_rx.await;

                context::commit_replay_chunk(
                    self.model_idx, ctx_id, chunk.num_pages,
                ).await?;
            }

            context::finish_restore(self.model_idx, ctx_id)?;
        }

        // Step 2: Resolve physical page IDs (pure lookup — all pages must be GPU-resident)
        let pages_by_device = context::get_physical_page_ids(self.model_idx, ctx_id).await?;

        // Context parallelism not yet supported — pages must reside on a single device
        if pages_by_device.len() > 1 {
            anyhow::bail!(
                "Context pages span {} devices; context parallelism is not yet supported",
                pages_by_device.len()
            );
        }

        // Extract the single device entry, or default to device 0
        let (device_id, physical_page_ids) = pages_by_device
            .into_iter()
            .next()
            .unwrap_or((0, vec![]));

        // Compute FlashInfer's last_page_len
        let num_pages = physical_page_ids.len() as u32;
        let num_input_tokens = request.tokens.len() as u32;
        let kv_len = context::kv_len(self.model_idx, ctx_id);
        let page_size = context::tokens_per_page(self.model_idx, ctx_id);
        let total_kv = kv_len + num_input_tokens;
        let last_page_len = context::kvcache::compute_last_page_len(total_kv, num_pages, page_size);

        // Route to the appropriate BatchScheduler
        let device_idx = device_id.min(self.num_devices.saturating_sub(1));
        self.schedulers[device_idx].submit(request, response_tx, physical_page_ids, last_page_len)
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

/// Messages handled by InferenceService.
#[derive(Debug)]
enum Message {
    ForwardPass { request: ForwardPassRequest, response: oneshot::Sender<ForwardPassOutput> },
    GetStats { response: oneshot::Sender<InferenceStats> },
}


impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::ForwardPass { request, response } => {
                if let Err(e) = self.forward_pass(request, response).await {
                    tracing::error!("Failed to queue forward pass: {}", e);
                }
            }
            Message::GetStats { response } => {
                let _ = response.send(self.aggregate_stats());
            }
        }
    }
}

