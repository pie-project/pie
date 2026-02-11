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

pub mod request;
pub mod scheduler;
mod adaptive_policy;



use tokio::sync::oneshot;

use crate::context;
use crate::service::{ServiceArray, ServiceHandler};
use crate::device::DeviceId;
use anyhow::Result;
use request::{ForwardPassOutput, ForwardPassRequest};
use scheduler::BatchScheduler;

// Re-export public types
pub use request::{ForwardPassOutput as Output, Sampler};

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
        let schedulers = device_ids.iter().enumerate().map(|(device_idx, &device_id)| {
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

        InferenceService {
            model_idx,
            num_devices,
            schedulers,
        }
    }

    /// Resolves physical pages from context and queues the forward pass.
    async fn forward_pass(&self, request: ForwardPassRequest, response_tx: oneshot::Sender<ForwardPassOutput>) -> Result<()> {
        // Resolve physical page IDs from context
        let (pages_by_device, last_page_len) = if let Some(ctx_id) = request.context_id {
            context::get_physical_page_ids(self.model_idx, ctx_id).await?
        } else {
            (Default::default(), 0)
        };

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
        }
    }
}

