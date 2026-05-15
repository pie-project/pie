//! Per-driver batch scheduler.
//!
//! Each `BatchScheduler` owns its own RPC client, scheduling policy,
//! and tokio task. It accepts pre-translated forward pass requests,
//! accumulates them into batches, and fires them based on adaptive
//! scheduling decisions.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{Semaphore, mpsc, oneshot};

use crate::context::pagestore::PhysicalPageId;
use crate::driver::DriverId;

use crate::driver;

use super::adaptive_policy::{AdaptivePolicy, EagerPolicy, GreedyPolicy};
use super::request;

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
    fn on_arrival(&mut self);

    /// A batch finished executing. `latency` is the wall-clock time
    /// the forward pass took on the driver.
    fn on_complete(&mut self, latency: Duration);

    /// The current batch was fired. `fired_size` is the number of
    /// requests in the batch — policies use it to learn the steady-
    /// state cohort size and avoid firing partial batches in the next
    /// cycle.
    fn on_fired(&mut self, fired_size: usize);

    /// Decide whether to fire or wait, given the current batch size.
    /// `&mut self` so policies can ratchet internal state (e.g.,
    /// AdaptivePolicy's `cohort_high_water`) on every poll.
    fn decide(&mut self, current_batch_size: usize) -> Decision;
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
// SchedulerStats (lock-free snapshot for monitoring)
// =============================================================================

/// Cumulative stats exposed for monitoring. Updated atomically after each batch.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub total_batches: AtomicU64,
    pub total_tokens_processed: AtomicU64,
    /// Total request count across all batches (sum of batch sizes).
    /// Divide by `total_batches` for mean batch size in requests.
    pub total_requests_processed: AtomicU64,
    /// Largest batch size (in requests) ever fired by this scheduler.
    pub max_batch_size_observed: AtomicU64,
    /// Coarse histogram of batch sizes. Buckets:
    /// [0]=1, [1]=2-3, [2]=4-7, [3]=8-15, [4]=16-31,
    /// [5]=32-63, [6]=64-127, [7]=128+.
    pub batch_size_hist: [AtomicU64; 8],
    pub last_batch_latency_us: AtomicU64,
    pub cumulative_latency_us: AtomicU64,
}

// =============================================================================
// PendingRequest
// =============================================================================

/// A forward pass request bundled with its response channel and physical pages.
struct PendingRequest {
    request: pie_bridge::ForwardRequest,
    response_tx: oneshot::Sender<pie_bridge::ForwardResponse>,
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
        self.total_tokens += req.request.token_ids.len();
        self.requests.push(req);
    }

    fn is_full(&self) -> bool {
        self.requests.len() >= self.max_batch_size || self.total_tokens >= self.max_batch_tokens
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
// SchedulerHandle
// =============================================================================

/// Cloneable submit handle. Used by the speculator's chain extender
/// (spawned outside the scheduler's `run` loop) to resubmit
/// pre-staged forward passes.
#[derive(Clone)]
pub(crate) struct SchedulerHandle {
    tx: mpsc::UnboundedSender<PendingRequest>,
}

impl SchedulerHandle {
    pub fn submit(
        &self,
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<pie_bridge::ForwardResponse>,
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
}

// =============================================================================
// BatchScheduler
// =============================================================================

/// Per-driver batch scheduler.
///
/// Owns an RPC client, a scheduling policy, and a tokio task that
/// runs the batch accumulation and firing loop.
pub(crate) struct BatchScheduler {
    tx: mpsc::UnboundedSender<PendingRequest>,
    stats: Arc<SchedulerStats>,
}

impl BatchScheduler {
    /// Spawn a new batch scheduler for a single driver.
    ///
    /// The RPC connection is owned by the driver service; the scheduler
    /// only stores the driver index for routing calls.
    pub fn new(
        driver_id: DriverId,
        driver_idx: usize,
        page_size: u32,
        max_batch_size: usize,
        max_batch_tokens: usize,
        request_timeout_secs: u64,
        batch_policy: String,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let stats = Arc::new(SchedulerStats::default());
        tokio::spawn(Self::run(
            driver_id,
            driver_idx,
            rx,
            page_size,
            max_batch_size,
            max_batch_tokens,
            request_timeout_secs,
            batch_policy,
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
        request: pie_bridge::ForwardRequest,
        response_tx: oneshot::Sender<pie_bridge::ForwardResponse>,
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

    /// Cloneable handle for tasks that need to submit outside the
    /// scheduler's `run` loop (e.g., the speculator chain extender).
    pub(crate) fn handle(&self) -> SchedulerHandle {
        SchedulerHandle {
            tx: self.tx.clone(),
        }
    }

    // =========================================================================
    // Internal: Scheduling Loop
    // =========================================================================

    /// Main scheduling loop for a single driver.
    async fn run(
        driver_id: DriverId,
        driver_idx: usize,
        mut req_rx: mpsc::UnboundedReceiver<PendingRequest>,
        page_size: u32,
        max_batch_size: usize,
        max_batch_tokens: usize,
        request_timeout_secs: u64,
        batch_policy: String,
        stats: Arc<SchedulerStats>,
    ) {
        let request_timeout = Duration::from_secs(request_timeout_secs);

        // Per-driver state
        let mut batch = BatchAccumulator::new(max_batch_size, max_batch_tokens);
        // Policy selection from config — see `adaptive_policy.rs` for the
        // design rationale. The per-model `[model.scheduler].batch_policy`
        // setting picks one of "adaptive", "eager", "greedy".
        let mut policy: Box<dyn SchedulingPolicy> = match batch_policy.as_str() {
            "greedy" => Box::new(GreedyPolicy::new()),
            "eager" => Box::new(EagerPolicy::new(max_batch_size, driver_idx)),
            "adaptive" => Box::new(AdaptivePolicy::new(max_batch_size, driver_idx)),
            other => panic!(
                "Unknown scheduler.batch_policy {other:?}; expected one of \
                'adaptive' | 'eager' | 'greedy'"
            ),
        };
        // Only one in-flight batch at a time to prevent pipelined KV cache corruption.
        let in_flight = Arc::new(Semaphore::new(1));

        // Channel for batch completion latency feedback to the policy.
        let (latency_tx, mut latency_rx) = mpsc::unbounded_channel::<Duration>();

        // Per-fire wall timing instrumentation. Gated on PIE_TIMING.
        let timing_on = std::env::var_os("PIE_TIMING").is_some();
        let mut last_fire_time: Option<Instant> = None;
        let mut sched_fire_count: u64 = 0;

        loop {
            // Drain completed batch latencies (non-blocking)
            while let Ok(latency) = latency_rx.try_recv() {
                policy.on_complete(latency);
            }
            let t_loop_top = Instant::now();

            // Wait for first request if batch is empty
            if batch.is_empty() {
                let Some(pending) = req_rx.recv().await else {
                    break;
                };
                policy.on_arrival();
                batch.push(pending);
            }
            let t_first_arrival = Instant::now();

            // Accumulate more requests (non-blocking)
            while let Ok(pending) = req_rx.try_recv() {
                policy.on_arrival();
                batch.push(pending);
                if batch.is_full() {
                    break;
                }
            }
            let t_accumulated = Instant::now();

            // Ask the policy what to do
            match policy.decide(batch.len()) {
                Decision::Fire => {
                    // Acquire a permit (may wait if at in-flight limit)
                    // if in_flight.available_permits() == 0 {
                    //     eprintln!("[SCHED dev={driver_idx}] semaphore full, waiting for in-flight batch to complete");
                    // }
                    let t_before_permit = Instant::now();
                    let permit = in_flight
                        .clone()
                        .acquire_owned()
                        .await
                        .expect("semaphore closed");
                    let t_permit = Instant::now();
                    let fire_n = sched_fire_count;
                    sched_fire_count += 1;
                    let log_outer = timing_on && fire_n >= 50 && fire_n % 50 == 0;
                    if log_outer {
                        let fire_to_fire = last_fire_time
                            .map(|t| t_permit.duration_since(t).as_micros())
                            .unwrap_or(0);
                        eprintln!(
                            "[outer-fire {} B={}] fire_to_fire={}us \
                             loop_top_after_prev={}us \
                             recv_first_req={}us accumulate_more={}us \
                             permit_wait_after_decide={}us",
                            fire_n,
                            batch.len(),
                            fire_to_fire,
                            last_fire_time
                                .map(|t| t_loop_top.duration_since(t).as_micros())
                                .unwrap_or(0),
                            (t_first_arrival - t_loop_top).as_micros(),
                            (t_accumulated - t_first_arrival).as_micros(),
                            (t_permit - t_before_permit).as_micros(),
                        );
                    }
                    last_fire_time = Some(t_permit);

                    let total_tokens = batch.total_tokens();
                    let requests_to_fire = batch.take();
                    policy.on_fired(requests_to_fire.len());

                    // Collect batch context IDs for accurate rent charging.
                    // Per-request shape stores the single context_id in
                    // `context_ids[0]`.
                    let batch_ctx_ids: Vec<u64> = requests_to_fire
                        .iter()
                        .map(|r| r.request.context_ids[0])
                        .collect();
                    let batch_size = batch_ctx_ids.len() as u64;

                    // Spawn batch execution
                    let latency_tx_clone = latency_tx.clone();
                    let stats_clone = stats.clone();
                    let timeout = request_timeout;

                    tokio::spawn(async move {
                        let start = Instant::now();
                        Self::execute_batch(
                            driver_idx,
                            requests_to_fire,
                            driver_id,
                            page_size,
                            timeout,
                        )
                        .await;
                        let latency = start.elapsed();

                        // Advance market clock for this driver: prices, rent, dividends.
                        // Pass batch context IDs so tick only charges contexts
                        // that were in this batch (not stale pinned contexts).
                        crate::context::tick(driver_idx, latency.as_secs_f64(), batch_ctx_ids);

                        // Update cumulative atomic counters (consumed by external
                        // monitoring; ignored by the policy).
                        stats_clone.total_batches.fetch_add(1, Relaxed);
                        stats_clone
                            .total_tokens_processed
                            .fetch_add(total_tokens as u64, Relaxed);
                        stats_clone
                            .total_requests_processed
                            .fetch_add(batch_size, Relaxed);
                        stats_clone
                            .max_batch_size_observed
                            .fetch_max(batch_size, Relaxed);
                        let bucket = match batch_size {
                            0 | 1 => 0,
                            2..=3 => 1,
                            4..=7 => 2,
                            8..=15 => 3,
                            16..=31 => 4,
                            32..=63 => 5,
                            64..=127 => 6,
                            _ => 7,
                        };
                        stats_clone.batch_size_hist[bucket].fetch_add(1, Relaxed);
                        stats_clone
                            .last_batch_latency_us
                            .store(latency.as_micros() as u64, Relaxed);
                        stats_clone
                            .cumulative_latency_us
                            .fetch_add(latency.as_micros() as u64, Relaxed);

                        latency_tx_clone.send(latency).ok();
                        drop(permit); // release in-flight slot
                    });
                }
                Decision::Wait(wait_duration) => {
                    tokio::select! {
                        _ = tokio::time::sleep(wait_duration) => {}
                        maybe_req = req_rx.recv() => {
                            if let Some(pending) = maybe_req {
                                policy.on_arrival();
                                batch.push(pending);
                            } else {
                                break; // channel closed
                            }
                        }
                        latency = latency_rx.recv() => {
                            if let Some(l) = latency {
                                policy.on_complete(l);
                            }
                        }
                    }
                }
            }
        }

        // Shutdown: fire remaining batch
        if !batch.is_empty() {
            let requests = batch.take();
            Self::execute_batch(driver_idx, requests, driver_id, page_size, request_timeout).await;
        }
    }

    /// Execute a batch of forward pass requests via the driver service.
    async fn execute_batch(
        driver_idx: usize,
        requests: Vec<PendingRequest>,
        driver_id: DriverId,
        page_size: u32,
        timeout: Duration,
    ) {
        // Per-stage timing for the Rust side of the inter-fire path.
        // Driver-side timing already lives in `request_handler.cpp`; this
        // tells us what fraction of the per-fire wall sits in scheduler /
        // msgpack / shmem-roundtrip / response-distribution.
        let timing_on = std::env::var_os("PIE_TIMING").is_some();
        static FIRE_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let fire_n = FIRE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let log_this = timing_on && fire_n >= 50 && fire_n % 50 == 0;
        let t0 = std::time::Instant::now();
        let r = requests.len();

        // Build batched request — a single `pie_bridge::ForwardRequest`
        // populated by folding each per-request shape into the batch.
        let mut batch_req = request::new_batched_forward_request();
        for req in &requests {
            request::append_request(
                &mut batch_req,
                &req.request,
                &req.physical_page_ids,
                req.last_page_len,
                page_size,
            );
        }
        let t_build = std::time::Instant::now();

        // Send via driver service (typed call handles serialization + timeout)
        let result = driver::fire_batch(driver_idx, batch_req).await;
        let t_resp = std::time::Instant::now();

        match result {
            Ok(batch_resp) => {
                let n_results = batch_resp.num_requests as usize;
                if n_results != requests.len() {
                    tracing::warn!(
                        driver = driver_id,
                        expected = requests.len(),
                        got = n_results,
                        "Batch response count mismatch — some requests may get no output",
                    );
                }

                for (r, req) in requests.into_iter().enumerate() {
                    if r < n_results {
                        // Extract this request's slice from the batched
                        // response. The api layer (build_wit_output)
                        // walks samplers + the single-request response
                        // to construct the WIT Output.
                        let per_req = request::extract_per_request(&batch_resp, r);
                        req.response_tx.send(per_req).ok();
                    } else {
                        tracing::warn!(
                            driver = driver_id,
                            "Fewer results than requests — sending empty"
                        );
                        req.response_tx
                            .send(pie_bridge::ForwardResponse::default())
                            .ok();
                    }
                }
            }
            Err(e) => {
                tracing::error!("fire_batch failed for driver {}: {:?}", driver_id, e);
                for req in requests {
                    req.response_tx
                        .send(pie_bridge::ForwardResponse::default())
                        .ok();
                }
            }
        }
        let t_done = std::time::Instant::now();
        if log_this {
            eprintln!(
                "[sched-fire {} R={}] build={}us roundtrip={}us distribute={}us TOTAL={}us",
                fire_n,
                r,
                (t_build - t0).as_micros(),
                (t_resp - t_build).as_micros(),
                (t_done - t_resp).as_micros(),
                (t_done - t0).as_micros(),
            );
        }
    }
}
