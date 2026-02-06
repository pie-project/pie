//! Inference Service - Forward pass management for model execution
//!
//! This module provides a model-specific actor for executing forward passes
//! with configurable samplers, input tokens, and attention masks.
//!
//! Architecture:
//! - Requests come in to the actor
//! - They get routed to per-node queues based on page affinity
//! - Each node has its own batching scheduler
//! - Batches are sent to Python via RPC

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

use batching::{MultiNodeScheduler, SchedulerConfig, SharedScheduler};
use request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassOutput, ForwardPassRequest,
};
use rpc::RpcClient;

// Re-export public types
pub use request::{ForwardPassOutput as Output, Sampler};

/// Global registry for inference actors.
static ACTOR: std::sync::LazyLock<ServiceArray<Message>> = std::sync::LazyLock::new(ServiceArray::new);

/// Spawns a new inference actor.
pub(crate) fn spawn() -> usize {
    ACTOR.spawn::<InferenceActor>()
}

/// Messages for the inference actor.
#[derive(Debug)]
pub enum Message {
    /// Executes a forward pass.
    ForwardPass {
        request: ForwardPassRequest,
        response: oneshot::Sender<ForwardPassOutput>,
    },
}

impl Message {
    /// Sends this message to the inference actor for the given model.
    pub fn send(self, model_idx: usize) -> anyhow::Result<()> {
        ACTOR.send(model_idx, self)
    }
}

// =============================================================================
// Inference Service
// =============================================================================

/// Configuration for the inference service.
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub scheduler_config: SchedulerConfig,
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub request_timeout: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            scheduler_config: SchedulerConfig::default(),
            max_batch_size: 64,
            max_batch_tokens: 4096,
            request_timeout: Duration::from_secs(300),
        }
    }
}

/// The inference service handles forward pass operations.
/// This is the core business logic, separate from the actor message handling.
#[derive(Debug)]
pub struct InferenceService {
    pub page_store: Arc<RwLock<PageStore>>,
}

impl InferenceService {
    pub fn new(page_store: Arc<RwLock<PageStore>>) -> Self {
        InferenceService { page_store }
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
}

// =============================================================================
// Inference Worker
// =============================================================================

/// Internal request with routing info.
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

/// Runs the inference worker loop.
///
/// This is the main batching and dispatch loop that:
/// 1. Receives requests from the actor
/// 2. Routes them to per-node batches based on page affinity
/// 3. Uses adaptive scheduling to decide when to fire batches
/// 4. Sends batches to Python via RPC
async fn inference_worker(
    service: Arc<InferenceService>,
    mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
    mut shutdown_rx: broadcast::Receiver<()>,
    rpc_clients: HashMap<NodeId, RpcClient>,
    config: InferenceConfig,
    scheduler: SharedScheduler,
) {
    const SCHEDULER_POLL_INTERVAL: Duration = Duration::from_millis(1);

    let num_nodes = rpc_clients.len();

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
            let pending = tokio::select! {
                _ = shutdown_rx.recv() => break,
                maybe_req = req_rx.recv() => {
                    match maybe_req {
                        Some(req) => req,
                        None => break,
                    }
                }
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
                        if batch.len() >= config.max_batch_size
                            || batch.total_tokens >= config.max_batch_tokens
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
                    config.max_batch_size,
                    config.max_batch_tokens,
                    in_flight,
                )
            } else {
                true
            };

            if should_fire && in_flight < config.scheduler_config.max_in_flight_batches {
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
                let timeout = config.request_timeout;

                tokio::spawn(async move {
                    let start = Instant::now();
                    execute_batch(rpc_client.as_ref(), requests_to_fire, node_id, timeout).await;
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
                .any(|&c| c >= config.scheduler_config.max_in_flight_batches);

            if any_at_limit {
                // Wait for completion
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
                    maybe = completion_rx.recv() => {
                        if let Some((node_id, batch_size, tokens, latency)) = maybe {
                            if let Some(count) = in_flight_counts.get_mut(&node_id) {
                                *count = count.saturating_sub(1);
                            }
                            if let Ok(mut sched) = scheduler.lock() {
                                sched.on_batch_complete(node_id, batch_size, tokens, latency);
                            }
                        }
                    }
                }
            } else {
                // Brief wait for more requests
                tokio::select! {
                    _ = shutdown_rx.recv() => break,
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
            execute_batch(rpc_client.as_ref(), requests, node_id, config.request_timeout).await;
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
            eprintln!("[Error] fire_batch failed for node {}: {:?}", node_id, e);
            for (_, tx, _) in requests {
                tx.send(ForwardPassOutput::None).ok();
            }
        }
    }
}

// =============================================================================
// Inference Actor
// =============================================================================

/// The inference actor manages forward pass execution for a model.
struct InferenceActor {
    service: Arc<InferenceService>,
    request_tx: mpsc::UnboundedSender<PendingRequest>,
    #[allow(dead_code)]
    shutdown_tx: broadcast::Sender<()>,
}

impl std::fmt::Debug for InferenceActor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceActor").finish()
    }
}

impl Default for InferenceActor {
    fn default() -> Self {
        // Create shared page store
        let page_store = Arc::new(RwLock::new(PageStore::new(64)));
        let service = Arc::new(InferenceService::new(page_store));

        // Create channels
        let (request_tx, request_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);

        // Create scheduler
        let config = InferenceConfig::default();
        let scheduler: SharedScheduler = Arc::new(Mutex::new(MultiNodeScheduler::new(
            config.scheduler_config.clone(),
            config.max_batch_size,
            1, // Will be updated when nodes are registered
        )));

        // No RPC clients initially - they'll be registered later
        let rpc_clients: HashMap<NodeId, RpcClient> = HashMap::new();

        // Start worker
        let service_clone = Arc::clone(&service);
        tokio::spawn(inference_worker(
            service_clone,
            request_rx,
            shutdown_rx,
            rpc_clients,
            config,
            scheduler,
        ));

        InferenceActor {
            service,
            request_tx,
            shutdown_tx,
        }
    }
}

impl ServiceHandler for InferenceActor {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::ForwardPass { request, response } => {
                // Determine target node based on page affinity
                let node_id = self
                    .service
                    .get_primary_node(&request.page_ids)
                    .unwrap_or(0);

                // Translate page IDs
                let physical_page_ids = self
                    .service
                    .translate_page_ids_for_node(&request.page_ids, node_id)
                    .unwrap_or_default();

                // Queue the request
                let pending = PendingRequest {
                    request,
                    response_tx: response,
                    node_id,
                    physical_page_ids,
                };

                if self.request_tx.send(pending).is_err() {
                    eprintln!("[Error] Failed to queue forward pass request");
                }
            }
        }
    }
}
