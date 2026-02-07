//! # Inference Module
//!
//! Forward pass management for model execution.
//!
//! Each model gets a dedicated InferenceActor that:
//! - Routes requests to per-node queues based on page affinity
//! - Uses adaptive scheduling for batch decisions
//! - Sends batches to Python via RPC
//!
//! Actors are stored per-model via a ServiceArray, accessed by model index.

pub mod batching;
pub mod brle;
pub mod request;
pub mod rpc;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use tokio::sync::{broadcast, mpsc, oneshot};

use crate::service::{ServiceArray, ServiceHandler};
use crate::kvcache::{NodeId, PageId, PageStore, PhysicalPageId};
use anyhow::Result;
use batching::{MultiNodeScheduler, SchedulerConfig, SharedScheduler};
use request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassOutput, ForwardPassRequest,
};
use rpc::RpcClient;

// Re-export public types
pub use request::{ForwardPassOutput as Output, Sampler};

// =============================================================================
// Public API
// =============================================================================

static SERVICE_ARRAY: std::sync::LazyLock<ServiceArray<Message>> = std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference service for a model.
pub fn spawn() -> usize {
    SERVICE_ARRAY.spawn(|| InferenceService::new()).expect("Failed to spawn inference service")
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
pub struct InferenceService {
    pub page_store: Arc<RwLock<PageStore>>,
    pub scheduler_config: SchedulerConfig,
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub request_timeout: Duration,
    request_tx: mpsc::UnboundedSender<PendingRequest>,
}

impl std::fmt::Debug for InferenceService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceService").finish()
    }
}

impl InferenceService {

    pub fn new() -> Self {
        // Create shared page store
        let page_store = Arc::new(RwLock::new(PageStore::new(64)));

        // Default config values
        let scheduler_config = SchedulerConfig::default();
        let max_batch_size = 64;
        let max_batch_tokens = 4096;
        let request_timeout = Duration::from_secs(300);

        // Create channels
        let (request_tx, request_rx) = mpsc::unbounded_channel();

        // Create scheduler
        let scheduler: SharedScheduler = Arc::new(Mutex::new(MultiNodeScheduler::new(
            scheduler_config.clone(),
            max_batch_size,
            1, // Will be updated when nodes are registered
        )));

        // No RPC clients initially - they'll be registered later
        let rpc_clients: HashMap<NodeId, RpcClient> = HashMap::new();

        // Start worker loop
        let max_in_flight_batches = scheduler_config.max_in_flight_batches;
        tokio::spawn(Self::worker_loop(
            max_batch_size,
            max_batch_tokens,
            max_in_flight_batches,
            request_timeout,
            request_rx,
            rpc_clients,
            scheduler,
        ));

        InferenceService {
            page_store,
            scheduler_config,
            max_batch_size,
            max_batch_tokens,
            request_timeout,
            request_tx,
        }
    }

    /// Translate logical page IDs to physical page IDs for a specific node.
    /// Returns None if the page has no mapping on the given node.
    pub fn translate_page_ids_for_node(
        &self,
        page_ids: &[PageId],
        node_id: NodeId,
    ) -> Option<Vec<PhysicalPageId>> {
        let page_store = self.page_store.read().unwrap();
        let mut result = Vec::with_capacity(page_ids.len());

        for &page_id in page_ids {
            let mappings = page_store.get_physical_mappings(page_id);
            // Find the mapping for this specific node
            if let Some((_, phys_id)) = mappings.into_iter().find(|(n, _)| *n == node_id) {
                result.push(phys_id);
            } else {
                return None; // Page not available on this node
            }
        }

        Some(result)
    }

    /// Get the primary node for a set of pages (based on first page).
    pub fn get_primary_node(&self, page_ids: &[PageId]) -> Option<NodeId> {
        if page_ids.is_empty() {
            return None;
        }

        let page_store = self.page_store.read().unwrap();
        let mappings = page_store.get_physical_mappings(page_ids[0]);
        mappings.into_iter().next().map(|(node_id, _)| node_id)
    }

    /// Runs the inference worker loop.
    ///
    /// This is the main batching and dispatch loop that:
    /// 1. Receives requests from the service
    /// 2. Routes them to per-node batches based on page affinity
    /// 3. Uses adaptive scheduling to decide when to fire batches
    /// 4. Sends batches to Python via RPC
    async fn worker_loop(
        max_batch_size: usize,
        max_batch_tokens: usize,
        max_in_flight_batches: usize,
        request_timeout: Duration,
        mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
        rpc_clients: HashMap<NodeId, RpcClient>,
        scheduler: SharedScheduler,
    ) {
        const SCHEDULER_POLL_INTERVAL: Duration = Duration::from_millis(1);

        // Per-node state
        let mut batches: HashMap<NodeId, NodeBatch> = HashMap::new();
        let mut in_flight_counts: HashMap<NodeId, usize> = HashMap::new();

        // Initialize per-node state
        for &node_id in rpc_clients.keys() {
            batches.insert(node_id, NodeBatch::new());
            in_flight_counts.insert(node_id, 0);
        }

        // Channel for batch completions
        let (completion_tx, mut completion_rx) =
            mpsc::unbounded_channel::<(NodeId, usize, usize, Duration)>();

        loop {
            // Process completed batches (non-blocking)
            while let Ok((node_id, batch_size, tokens, latency)) = completion_rx.try_recv() {
                if let Some(count) = in_flight_counts.get_mut(&node_id) {
                    *count = count.saturating_sub(1);
                }
                if let Ok(mut sched) = scheduler.lock() {
                    sched.on_batch_complete(node_id, batch_size, tokens, latency);
                }
            }

            // Check if all batches are empty
            let all_empty = batches.values().all(|b| b.is_empty());

            if all_empty {
                // Wait for first request
                let Some(pending) = req_rx.recv().await else {
                    break;
                };

                // Add to appropriate node batch
                let node_id = pending.node_id;
                if let Some(batch) = batches.get_mut(&node_id) {
                    if let Ok(mut sched) = scheduler.lock() {
                        let arrival_time = pending.request.arrival_time.unwrap_or_else(Instant::now);
                        sched.on_request_arrival(node_id, arrival_time);
                    }
                    batch.add(pending.request, pending.response_tx, pending.physical_page_ids);
                }
            }

            // Accumulate more requests (non-blocking)
            loop {
                match req_rx.try_recv() {
                    Ok(pending) => {
                        let node_id = pending.node_id;
                        if let Some(batch) = batches.get_mut(&node_id) {
                            if let Ok(mut sched) = scheduler.lock() {
                                let arrival_time = pending.request.arrival_time.unwrap_or_else(Instant::now);
                                sched.on_request_arrival(node_id, arrival_time);
                            }
                            batch.add(pending.request, pending.response_tx, pending.physical_page_ids);

                            // Stop if this batch is at capacity
                            if batch.len() >= max_batch_size
                                || batch.total_tokens >= max_batch_tokens
                            {
                                break;
                            }
                        }
                    }
                    Err(_) => break,
                }
            }

            // Check all nodes for firing
            let mut fired_any = false;
            for (&node_id, batch) in batches.iter_mut() {
                if batch.is_empty() {
                    continue;
                }

                let in_flight = *in_flight_counts.get(&node_id).unwrap_or(&0);

                let should_fire = if let Ok(mut sched) = scheduler.lock() {
                    sched.should_fire(
                        node_id,
                        batch.len(),
                        batch.total_tokens,
                        max_batch_size,
                        max_batch_tokens,
                        in_flight,
                    )
                } else {
                    true
                };

                if should_fire && in_flight < max_in_flight_batches {
                    // Fire this batch
                    let requests_to_fire = batch.take();
                    if let Some(count) = in_flight_counts.get_mut(&node_id) {
                        *count += 1;
                    }
                    fired_any = true;

                    if let Ok(mut sched) = scheduler.lock() {
                        sched.on_batch_fired(node_id);
                    }

                    // Spawn batch execution
                    let rpc_client = rpc_clients.get(&node_id).cloned();
                    let completion_tx_clone = completion_tx.clone();
                    let batch_size = requests_to_fire.len();
                    let tokens_in_batch: usize =
                        requests_to_fire.iter().map(|(r, _, _)| r.tokens.len()).sum();

                    tokio::spawn(async move {
                        let start = Instant::now();
                        Self::execute_batch(rpc_client.as_ref(), requests_to_fire, node_id, request_timeout).await;
                        let latency = start.elapsed();
                        completion_tx_clone
                            .send((node_id, batch_size, tokens_in_batch, latency))
                            .ok();
                    });
                }
            }

            // Wait logic
            if !fired_any {
                let any_at_limit = in_flight_counts
                    .values()
                    .any(|&c| c >= max_in_flight_batches);

                if any_at_limit {
                    // Wait for completion
                    if let Some((node_id, batch_size, tokens, latency)) = completion_rx.recv().await {
                        if let Some(count) = in_flight_counts.get_mut(&node_id) {
                            *count = count.saturating_sub(1);
                        }
                        if let Ok(mut sched) = scheduler.lock() {
                            sched.on_batch_complete(node_id, batch_size, tokens, latency);
                        }
                    }
                } else {
                    // Brief wait for more requests
                    tokio::select! {
                        _ = tokio::time::sleep(SCHEDULER_POLL_INTERVAL) => {}
                        maybe_req = req_rx.recv() => {
                            if let Some(pending) = maybe_req {
                                let node_id = pending.node_id;
                                if let Some(batch) = batches.get_mut(&node_id) {
                                    if let Ok(mut sched) = scheduler.lock() {
                                        let arrival_time = pending.request.arrival_time.unwrap_or_else(Instant::now);
                                        sched.on_request_arrival(node_id, arrival_time);
                                    }
                                    batch.add(pending.request, pending.response_tx, pending.physical_page_ids);
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Shutdown: fire remaining batches
        for (&node_id, batch) in batches.iter_mut() {
            if !batch.is_empty() {
                let requests = batch.take();
                let rpc_client = rpc_clients.get(&node_id).cloned();
                Self::execute_batch(rpc_client.as_ref(), requests, node_id, request_timeout).await;
            }
        }
    }

    /// Execute a batch of forward pass requests via RPC.
    async fn execute_batch(
        rpc_client: Option<&RpcClient>,
        requests: Vec<(ForwardPassRequest, oneshot::Sender<ForwardPassOutput>, Vec<PhysicalPageId>)>,
        node_id: NodeId,
        timeout: Duration,
    ) {
        let Some(client) = rpc_client else {
            // No RPC client - send failure responses
            for (_, tx, _) in requests {
                tx.send(ForwardPassOutput::None).ok();
            }
            return;
        };

        // Build batched request
        let mut batch_req = BatchedForwardPassRequest::new(node_id);
        for (req, _, phys_ids) in &requests {
            batch_req.add_request(req, &phys_ids.iter().map(|&p| p as u32).collect::<Vec<_>>());
        }

        // Send RPC
        let result: Result<BatchedForwardPassResponse, _> =
            client.call_with_timeout("fire_batch", &batch_req, timeout).await;

        match result {
            Ok(batch_resp) => {
                let mut resp_iter = batch_resp.results.into_iter();
                for (_, tx, _) in requests {
                    if let Some(resp) = resp_iter.next() {
                        let output = if !resp.tokens.is_empty() {
                            ForwardPassOutput::Tokens(resp.tokens)
                        } else if !resp.dists.is_empty() {
                            ForwardPassOutput::Distributions(resp.dists)
                        } else {
                            ForwardPassOutput::None
                        };
                        tx.send(output).ok();
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for node {}: {:?}", node_id, e);
                for (_, tx, _) in requests {
                    tx.send(ForwardPassOutput::None).ok();
                }
            }
        }
    }

    /// Queues a forward pass request for execution.
    fn forward_pass(&self, request: ForwardPassRequest, response_tx: oneshot::Sender<ForwardPassOutput>) -> Result<()> {
        // Determine target node based on page affinity
        let node_id = self.get_primary_node(&request.page_ids).unwrap_or(0);

        // Translate page IDs
        let physical_page_ids = self
            .translate_page_ids_for_node(&request.page_ids, node_id)
            .unwrap_or_default();

        // Queue the request
        let pending = PendingRequest {
            request,
            response_tx,
            node_id,
            physical_page_ids,
        };

        self.request_tx.send(pending)?;
        Ok(())
    }
}

// =============================================================================
// Worker Helper Types
// =============================================================================
struct PendingRequest {
    request: ForwardPassRequest,
    response_tx: oneshot::Sender<ForwardPassOutput>,
    node_id: NodeId,
    physical_page_ids: Vec<PhysicalPageId>,
}

/// Per-node batch accumulator.
struct NodeBatch {
    requests: Vec<(ForwardPassRequest, oneshot::Sender<ForwardPassOutput>, Vec<PhysicalPageId>)>,
    total_tokens: usize,
}

impl NodeBatch {
    fn new() -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
        }
    }

    fn add(&mut self, req: ForwardPassRequest, tx: oneshot::Sender<ForwardPassOutput>, phys_ids: Vec<PhysicalPageId>) {
        self.total_tokens += req.tokens.len();
        self.requests.push((req, tx, phys_ids));
    }

    fn len(&self) -> usize {
        self.requests.len()
    }

    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn take(&mut self) -> Vec<(ForwardPassRequest, oneshot::Sender<ForwardPassOutput>, Vec<PhysicalPageId>)> {
        self.total_tokens = 0;
        std::mem::take(&mut self.requests)
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
