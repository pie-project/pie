//! Per-device batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them based on adaptive
//! scheduling decisions.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot, Semaphore};


use crate::device::DeviceId;
use crate::context::kvcache::PhysicalPageId;

use crate::device;

use super::adaptive_policy::AdaptiveThroughputPolicy;
use super::request::{
    BatchedForwardPassRequest, BatchedForwardPassResponse, ForwardPassOutput, ForwardPassRequest,
};

// =============================================================================
// Scheduling Policy Trait
// =============================================================================

/// Pluggable scheduling policy.
///
/// A policy receives event callbacks (`on_arrival`, `on_complete`,
/// `on_fired`) and returns a [`Decision`] when asked whether to fire
/// the current batch.
pub(super) trait SchedulingPolicy: Send {
    /// A request was added to the accumulator.
    fn on_arrival(&mut self, arrival_time: Instant);

    /// A batch finished executing.
    fn on_complete(&mut self, stats: &BatchStats);

    /// The current batch was fired.
    fn on_fired(&mut self);

    /// Decide whether to fire or wait.
    fn decide(
        &self,
        current_batch_size: usize,
        current_total_tokens: usize,
        in_flight_batches: usize,
    ) -> Decision;
}

// =============================================================================
// Scheduling Decision
// =============================================================================

/// The outcome of a scheduling policy decision.
pub(super) enum Decision {
    /// Fire the current batch immediately.
    Fire,
    /// Wait for more requests, up to the given duration.
    Wait(Duration),
}

// =============================================================================
// BatchStats
// =============================================================================

/// Statistics from a completed batch execution, fed back to the policy.
pub(super) struct BatchStats {
    pub batch_size: usize,
    pub total_tokens: usize,
    pub latency: Duration,
}

// =============================================================================
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

/// Cumulative stats exposed for monitoring. Updated atomically after each batch.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub total_batches: AtomicU64,
    pub total_tokens_processed: AtomicU64,
    pub last_batch_latency_us: AtomicU64,
    pub cumulative_latency_us: AtomicU64,
    pub in_flight_batches: AtomicU64,
}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
struct PendingRequest {
    request: ForwardPassRequest,
    response_tx: oneshot::Sender<ForwardPassOutput>,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
}

// =============================================================================
// BatchAccumulator
// =============================================================================

/// Accumulates pending requests into a batch.
///
/// Pure synchronous struct — no async, no channels. Can be tested
/// independently from the scheduling loop.
struct BatchAccumulator {
    requests: Vec<PendingRequest>,
    total_tokens: usize,
    max_batch_size: usize,
    max_batch_tokens: usize,
}

impl BatchAccumulator {
    fn new(max_batch_size: usize, max_batch_tokens: usize) -> Self {
        Self {
            requests: Vec::new(),
            total_tokens: 0,
            max_batch_size,
            max_batch_tokens,
        }
    }

    fn push(&mut self, req: PendingRequest) {
        self.total_tokens += req.request.tokens.len();
        self.requests.push(req);
    }

    fn is_full(&self) -> bool {
        self.requests.len() >= self.max_batch_size
            || self.total_tokens >= self.max_batch_tokens
    }

    fn is_empty(&self) -> bool {
        self.requests.is_empty()
    }

    fn len(&self) -> usize {
        self.requests.len()
    }

    fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    fn take(&mut self) -> Vec<PendingRequest> {
        self.total_tokens = 0;
        std::mem::take(&mut self.requests)
    }
}

// =============================================================================
// BatchScheduler
// =============================================================================

/// Per-device batch scheduler.
///
/// Owns an RPC client, a scheduling policy, and a tokio task that
/// runs the batch accumulation and firing loop.
pub(crate) struct BatchScheduler {
    tx: mpsc::UnboundedSender<PendingRequest>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    /// Spawn a new batch scheduler for a single device.
    ///
    /// The RPC connection is owned by the device service; the scheduler
    /// only stores the device index for routing calls.
    pub fn new(
        device_id: DeviceId,
        device_idx: usize,
        max_batch_size: usize,
        max_batch_tokens: usize,
        max_in_flight_batches: usize,
        request_timeout_secs: u64,
        max_wait_ms: u64,
        min_batch_for_optimization: usize,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let stats = Arc::new(SchedulerStats::default());
        tokio::spawn(Self::run(
            device_id, device_idx, rx,
            max_batch_size, max_batch_tokens,
            max_in_flight_batches, request_timeout_secs, max_wait_ms, min_batch_for_optimization,
            stats.clone(),
        ));

        Self { tx, stats }
    }

    /// Get a handle to the cumulative scheduler stats (lock-free).
    pub fn stats(&self) -> &Arc<SchedulerStats> {
        &self.stats
    }

    /// Submit a pre-translated forward pass request.
    pub fn submit(
        &self,
        request: ForwardPassRequest,
        response_tx: oneshot::Sender<ForwardPassOutput>,
        physical_page_ids: Vec<PhysicalPageId>,
        last_page_len: u32,
    ) -> Result<()> {
        self.tx.send(PendingRequest {
            request,
            response_tx,
            physical_page_ids,
            last_page_len,
        })?;
        Ok(())
    }

    // =========================================================================
    // Internal: Scheduling Loop
    // =========================================================================

    /// Main scheduling loop for a single device.
    async fn run(
        device_id: DeviceId,
        device_idx: usize,
        mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
        max_batch_size: usize,
        max_batch_tokens: usize,
        max_in_flight_batches: usize,
        request_timeout_secs: u64,
        max_wait_ms: u64,
        min_batch_for_optimization: usize,
        stats: Arc<SchedulerStats>,
    ) {
        let max_wait_time = Duration::from_millis(max_wait_ms);
        let request_timeout = Duration::from_secs(request_timeout_secs);

        // Per-device state
        let mut batch = BatchAccumulator::new(max_batch_size, max_batch_tokens);
        let mut policy: Box<dyn SchedulingPolicy> =
            Box::new(AdaptiveThroughputPolicy::new(
                max_batch_size,
                max_wait_time,
                min_batch_for_optimization,
            ));
        let in_flight = Arc::new(Semaphore::new(max_in_flight_batches));

        // Channel for batch completion stats (latency feedback only)
        let (stats_tx, mut stats_rx) = mpsc::unbounded_channel::<BatchStats>();

        loop {
            // Drain completed batch stats (non-blocking)
            while let Ok(stats) = stats_rx.try_recv() {
                policy.on_complete(&stats);
            }

            // Wait for first request if batch is empty
            if batch.is_empty() {
                let Some(pending) = req_rx.recv().await else {
                    break;
                };
                policy.on_arrival(pending.request.arrival_time.unwrap_or_else(Instant::now));
                batch.push(pending);
            }

            // Accumulate more requests (non-blocking)
            while let Ok(pending) = req_rx.try_recv() {
                policy.on_arrival(pending.request.arrival_time.unwrap_or_else(Instant::now));
                batch.push(pending);
                if batch.is_full() {
                    break;
                }
            }

            // Ask the policy what to do
            let in_flight_count =
                max_in_flight_batches - in_flight.available_permits();

            match policy.decide(batch.len(), batch.total_tokens(), in_flight_count) {
                Decision::Fire => {
                    // Acquire a permit (may wait if at in-flight limit)
                    let permit = in_flight
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("semaphore closed");

                    let batch_size = batch.len();
                    let total_tokens = batch.total_tokens();
                    let requests_to_fire = batch.take();
                    policy.on_fired();

                    // Spawn batch execution
                    let stats_tx_clone = stats_tx.clone();
                    let stats_clone = stats.clone();
                    let timeout = request_timeout;

                    stats_clone.in_flight_batches.fetch_add(1, Relaxed);
                    tokio::spawn(async move {
                        let start = Instant::now();
                        Self::execute_batch(
                            device_idx,
                            requests_to_fire,
                            device_id,
                            timeout,
                        )
                        .await;
                        let latency = start.elapsed();

                        // Update cumulative atomic counters
                        stats_clone.total_batches.fetch_add(1, Relaxed);
                        stats_clone.total_tokens_processed.fetch_add(total_tokens as u64, Relaxed);
                        stats_clone.last_batch_latency_us.store(latency.as_micros() as u64, Relaxed);
                        stats_clone.cumulative_latency_us.fetch_add(latency.as_micros() as u64, Relaxed);
                        stats_clone.in_flight_batches.fetch_sub(1, Relaxed);

                        stats_tx_clone
                            .send(BatchStats {
                                batch_size,
                                total_tokens,
                                latency,
                            })
                            .ok();
                        drop(permit); // release in-flight slot
                    });
                }
                Decision::Wait(wait_duration) => {
                    tokio::select! {
                        _ = tokio::time::sleep(wait_duration) => {}
                        maybe_req = req_rx.recv() => {
                            if let Some(pending) = maybe_req {
                                policy.on_arrival(
                                    pending.request.arrival_time.unwrap_or_else(Instant::now),
                                );
                                batch.push(pending);
                            } else {
                                break; // channel closed
                            }
                        }
                        stats = stats_rx.recv() => {
                            if let Some(s) = stats {
                                policy.on_complete(&s);
                            }
                        }
                    }
                }
            }
        }

        // Shutdown: fire remaining batch
        if !batch.is_empty() {
            let requests = batch.take();
            Self::execute_batch(
                device_idx,
                requests,
                device_id,
                request_timeout,
            )
            .await;
        }
    }

    /// Execute a batch of forward pass requests via the device service.
    async fn execute_batch(
        device_idx: usize,
        requests: Vec<PendingRequest>,
        device_id: DeviceId,
        timeout: Duration,
    ) {
        // Build batched request
        let mut batch_req = BatchedForwardPassRequest::new(device_id);
        for req in &requests {
            batch_req.add_request(
                &req.request,
                &req.physical_page_ids
                    .iter()
                    .map(|&p| p as u32)
                    .collect::<Vec<_>>(),
                req.last_page_len,
            );
        }

        // Send via device service (typed call handles serialization + timeout)
        let result = device::call_with_timeout::<_, BatchedForwardPassResponse>(
            device_idx, "fire_batch", &batch_req, timeout,
        ).await;

        match result {
            Ok(batch_resp) => {
                if batch_resp.results.len() != requests.len() {
                    tracing::warn!(
                        device = device_id,
                        expected = requests.len(),
                        got = batch_resp.results.len(),
                        "Batch response count mismatch — some requests may get no output",
                    );
                }

                let mut resp_iter = batch_resp.results.into_iter();
                for req in requests {
                    if let Some(resp) = resp_iter.next() {
                        let output = if !resp.tokens.is_empty() {
                            if !resp.spec_tokens.is_empty() {
                                ForwardPassOutput::TokensWithSpeculation(
                                    resp.tokens,
                                    resp.spec_tokens,
                                    resp.spec_positions,
                                )
                            } else {
                                ForwardPassOutput::Tokens(resp.tokens)
                            }
                        } else if !resp.dists.is_empty() {
                            ForwardPassOutput::Distributions(resp.dists)
                        } else {
                            ForwardPassOutput::None
                        };
                        req.response_tx.send(output).ok();
                    } else {
                        tracing::warn!(device = device_id, "Fewer results than requests — sending None");
                        req.response_tx.send(ForwardPassOutput::None).ok();
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for device {}: {:?}", device_id, e);
                for req in requests {
                    req.response_tx.send(ForwardPassOutput::None).ok();
                }
            }
        }
    }
}
