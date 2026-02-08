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
pub mod kvcache;
pub mod request;
pub mod scheduler;
mod adaptive_policy;



use tokio::sync::oneshot;

use crate::service::{ServiceArray, ServiceHandler};
use crate::inference::kvcache::{DeviceId, PageId, PageStore, PhysicalPageId};
use anyhow::Result;
use request::{ForwardPassOutput, ForwardPassRequest};
use scheduler::BatchScheduler;

// Re-export public types
pub use request::{ForwardPassOutput as Output, Sampler};

// =============================================================================
// Public API
// =============================================================================

static SERVICE_ARRAY: std::sync::LazyLock<ServiceArray<Message>> = std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference service for a model.
pub fn spawn(
    page_store: PageStore,
    scheduler_config: &crate::bootstrap::SchedulerConfig,
) -> usize {
    let scheduler_config = scheduler_config.clone();
    SERVICE_ARRAY.spawn(move || InferenceService::new(page_store, &scheduler_config)).expect("Failed to spawn inference service")
}

/// Executes a forward pass and returns the output.
pub async fn forward_pass(model_idx: usize, request: ForwardPassRequest) -> Result<ForwardPassOutput> {
    let (tx, rx) = oneshot::channel();
    SERVICE_ARRAY.send(model_idx, Message::ForwardPass { request, response: tx })?;
    Ok(rx.await?)
}

// =============================================================================
// Inference Service
// =============================================================================

/// The inference service handles forward pass operations.
///
/// Translates logical page IDs to physical page IDs and routes
/// requests to the appropriate per-device `BatchScheduler`.
pub struct InferenceService {
    page_store: PageStore,
    schedulers: Vec<BatchScheduler>,
}

impl std::fmt::Debug for InferenceService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceService").finish()
    }
}

impl InferenceService {

    pub fn new(
        page_store: PageStore,
        scheduler_config: &crate::bootstrap::SchedulerConfig,
    ) -> Self {
        // TODO: schedulers should be created per device index
        let schedulers = Vec::new();

        InferenceService {
            page_store,
            schedulers,
        }
    }

    /// Translate logical page IDs to physical page IDs for a specific device.
    /// Returns None if the page has no mapping on the given device.
    fn translate_page_ids_for_device(
        &self,
        page_ids: &[PageId],
        device_id: DeviceId,
    ) -> Option<Vec<PhysicalPageId>> {
        let mut result = Vec::with_capacity(page_ids.len());

        for &page_id in page_ids {
            let mappings = self.page_store.get_physical_mappings(page_id);
            // Find the mapping for this specific device
            if let Some((_, phys_id)) = mappings.into_iter().find(|(n, _)| *n == device_id) {
                result.push(phys_id);
            } else {
                return None; // Page not available on this device
            }
        }

        Some(result)
    }

    /// Get the primary device for a set of pages (based on first page).
    fn get_primary_device(&self, page_ids: &[PageId]) -> Option<DeviceId> {
        if page_ids.is_empty() {
            return None;
        }

        let mappings = self.page_store.get_physical_mappings(page_ids[0]);
        mappings.into_iter().next().map(|(device_id, _)| device_id)
    }

    /// Queues a forward pass request for execution.
    fn forward_pass(&self, request: ForwardPassRequest, response_tx: oneshot::Sender<ForwardPassOutput>) -> Result<()> {
        // Determine target device based on page affinity
        let device_id = self.get_primary_device(&request.page_ids).unwrap_or(0);

        // Translate page IDs
        let physical_page_ids = self
            .translate_page_ids_for_device(&request.page_ids, device_id)
            .unwrap_or_default();

        // Route to the appropriate BatchScheduler
        self.schedulers[device_id as usize].submit(request, response_tx, physical_page_ids)
    }
}

// =============================================================================
// ServiceHandler Implementation
// =============================================================================

/// Messages handled by InferenceService.
#[derive(Debug)]
pub(crate) enum Message {
    ForwardPass { request: ForwardPassRequest, response: oneshot::Sender<ForwardPassOutput> },
}


impl ServiceHandler for InferenceService {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::ForwardPass { request, response } => {
                if let Err(e) = self.forward_pass(request, response) {
                    tracing::error!("Failed to queue forward pass: {}", e);
                }
            }
        }
    }
}
